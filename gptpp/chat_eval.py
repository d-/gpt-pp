"""What the chat stage did, measured: results/chat/chat-stage.json.

    python -m gptpp.chat_eval --base gptpp-base --run gptpp-chat [--probes NAME,NAME,...] [--skip-plain-text]
    python -m gptpp.chat_eval --base gptpp-base --base-only          # the base model's half, while the stage runs;
                                                                     # then the first command with --reuse-base
    python -m gptpp.chat_eval --add-run gptpp-chat-s0.2   # another full run of the stage, measured beside the first
    python -m gptpp.chat_eval --tally-template        # results/chat/chat-tally.json, to be filled in by a reader
    python -m gptpp.chat_eval --tally                 # put the filled-in tally into chat-stage.json

Four measurements, each written with a note that says how it was computed:

    held_out      the loss on held-out conversations, on the tokens that carry it (the assistant's text and its
                  end token, in all and apart, and by source), for the base model and for the tuned one
    plain_text    bits per byte on held-out FineWeb-EDU documents for both, by `python -m gptpp.evaluate`:
                  what the stage cost the model's first skill
    questions     twenty fixed user messages of eight kinds. The base model continues the bare question (the
                  text after <|bos|>, as a page that begins with it); the tuned model answers it in the chat
                  format. Each is decoded greedily and at temperature 0.7 with a fixed seed.
    tally         how many of the tuned model's answers are right, wrong, or off the subject. This one is not
                  computed: a reader judges every answer by the criteria written in the tally file, which
                  records the answer judged (its hash), so that a tally can never be attached to other answers.

The questions were fixed before any answer was seen and are not changed afterwards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from . import repo_dir
from .chat import (ASSISTANT_END, ASSISTANT_START, BOS, USER_END, USER_START, ChatData, answer, checkpoint_path,
                   conversations, evaluate, load_run, pack, plain_batches, render, reply, rows_of, val_batches)

# kind, user message. Eight kinds: a definition, a fact from common knowledge, a how-to, a short story, a list,
# simple arithmetic, a question about itself, a question it cannot know the answer to.
QUESTIONS: list[tuple[str, str]] = [
    ("definition", "What is photosynthesis?"),
    ("definition", "What is a prime number?"),
    ("definition", "What does the word \"democracy\" mean?"),
    ("fact", "What is the capital of France?"),
    ("fact", "How many days are there in a week?"),
    ("fact", "Who wrote Romeo and Juliet?"),
    ("how-to", "How do I boil an egg?"),
    ("how-to", "How can I fall asleep faster?"),
    ("story", "Write a short story about a cat who is afraid of mice."),
    ("story", "Tell me a very short story about a lighthouse keeper."),
    ("list", "List five fruits."),
    ("list", "Give me three tips for studying for an exam."),
    ("arithmetic", "What is 7 + 5?"),
    ("arithmetic", "What is 12 times 3?"),
    ("arithmetic", "I have 10 apples and I eat 3. How many are left?"),
    ("itself", "Who are you?"),
    ("itself", "Are you a human?"),
    ("itself", "What can you help me with?"),
    ("unknowable", "What will the weather be in Paris tomorrow?"),
    ("unknowable", "What did I eat for breakfast this morning?"),
]

VERDICTS = ("right", "wrong", "off")
CRITERIA = {
    "right": "answers what was asked, and nothing it states that bears on the question is false; for a story or a "
             "list, the thing asked for is there, in the number asked for; for a question about itself, it says it "
             "is a program, a model or an assistant; for a question it cannot know, it says so or asks",
    "wrong": "on the subject, but with a false statement that bears on the question, the wrong result, a made-up "
             "answer to what it cannot know, or the task not done as asked",
    "off": "off the subject: it answers another question, or none",
    "repetition": "an answer that repeats itself until it is cut off is judged on what it says; that it did not "
                  "stop is counted apart (stopped: length)",
}


def digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def question_ids() -> list[str]:
    seen: Counter = Counter()
    ids = []
    for kind, _ in QUESTIONS:
        seen[kind] += 1
        ids.append(f"{kind}-{seen[kind]}")
    return ids


# ----------------------------------------------------------------------------------------------------------------
# The measurements


def held_out_loss(model, data: ChatData, end_id: int, batch: int) -> dict:
    """The held-out loss of `model`, in all and by source of the conversations."""
    device = model.embed.weight.device
    seq_len = model.cfg.block_size
    result = evaluate(model, val_batches(data, seq_len, batch, device), end_id)
    docs = data.dir / data.meta.get("val_docs_file", "../val_conversations.jsonl")
    sources = [json.loads(line)["source"] for line in docs.read_text(encoding="utf-8").splitlines()]
    result["conversations"] = len(sources)
    result["by_source"] = {}
    for source in sorted(set(sources)):
        keep = [i for i, s in enumerate(sources) if s == source]
        result["by_source"][source] = {"conversations": len(keep),
                                       **evaluate(model, val_batches(data, seq_len, batch, device, keep=keep), end_id)}
    return result


ROLES = ("assistant_text", "assistant_end", "user_text", "markers")


@torch.no_grad()
def held_out_by_role(model, data: ChatData, special: dict[str, int], batch: int) -> dict:
    """What a token of the held-out conversations costs the model, by whose token it is: the assistant's text and
    its end token (the two that carry the loss), the user's text, and the three markers the model is only ever
    given (<|user_start|>, <|user_end|>, <|assistant_start|>). The mean of -log p(token | the tokens before it)
    in nats, over the same rows as the held-out loss; <|bos|> and padding are not scored."""
    device = model.embed.weight.device
    seq_len = model.cfg.block_size
    tokens, mask = data.val_tokens(), data.val_mask()
    starts, lengths = conversations(tokens, data.bos_id)
    ids, carries = rows_of(tokens, mask, starts, lengths, pack(lengths, seq_len + 1), seq_len + 1, data.bos_id)
    in_user_turn = (np.cumsum(ids == special[USER_START], axis=1) - np.cumsum(ids == special[USER_END], axis=1)) > 0
    given = (ids == special[USER_START]) | (ids == special[USER_END]) | (ids == special[ASSISTANT_START])
    role = np.zeros(ids.shape, dtype=np.int64)  # 0: not scored
    role[carries == 1] = 1
    role[ids == special[ASSISTANT_END]] = 2
    role[in_user_turn & ~given] = 3
    role[given] = 4
    total = torch.zeros(2, len(ROLES) + 1, dtype=torch.float64, device=device)  # sums and counts by role
    was_training = model.training
    model.eval()
    for i in range(0, len(ids), batch):
        block = torch.from_numpy(ids[i:i + batch].astype(np.int64)).to(device)
        whose = torch.from_numpy(role[i:i + batch, 1:]).to(device).reshape(-1)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(block[:, :-1])
        losses = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), block[:, 1:].reshape(-1),
                                 reduction="none").double()
        total[0].index_add_(0, whose, losses)
        total[1].index_add_(0, whose, torch.ones_like(losses))
    model.train(was_training)
    sums, counts = total.tolist()
    return {name: {"loss": sums[k + 1] / max(counts[k + 1], 1), "tokens": int(counts[k + 1])}
            for k, name in enumerate(ROLES)}


def plain_text(checkpoint: Path, out: Path, docs: str = "fineweb-edu", extra: str = "", reuse: bool = False) -> dict:
    """Bits per byte on the held-out documents, by the lab's evaluation run as its own command. With `reuse`, a
    result already in `out` for the same checkpoint is taken as it is."""
    target = checkpoint.parent if checkpoint.name == "model.pt" else checkpoint
    command = [sys.executable, "-m", "gptpp.evaluate", "--model", str(target), "--docs", docs, "--out", str(out),
               *extra.split()]
    name = f"{checkpoint.parent.name}/{checkpoint.name}"
    if not (reuse and out.exists() and json.loads(out.read_text())["model"].get("checkpoint") == name):
        if out.exists():
            out.unlink()
        subprocess.run(command, check=True, cwd=repo_dir())
    result = json.loads(out.read_text())["bits_per_byte"][docs]
    keep = ("bits_per_byte", "bits_per_byte_se", "loss", "bytes_per_token", "n_docs", "n_bytes", "n_tokens", "window",
            "precision")
    return {**{key: result[key] for key in keep}, "docs_sha256": result["docs"]["sha256"],
            "command": ("python -m gptpp.evaluate --model <run> --docs " + docs + " " + extra).strip(),
            "file": str(out.relative_to(repo_dir())) if out.is_relative_to(repo_dir()) else out.name}


@torch.no_grad()
def token_losses(model, ids: list[int]) -> list[float]:
    """-log p(token | the tokens before it) in nats for every token of `ids` after the first, the sequence
    standing alone in the context."""
    device = model.embed.weight.device
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = model(x[:, :-1])
    return F.cross_entropy(logits[0].float(), x[0, 1:], reduction="none").tolist()


def example(tokenizer, data: ChatData, index: int, exchanges: int) -> dict:
    """One held-out conversation as the model sees it, cut after its first `exchanges` exchanges: its tokens and
    which of them carry the loss (what each costs the two models is added by `measure`)."""
    docs = data.dir / data.meta.get("val_docs_file", "../val_conversations.jsonl")
    row = json.loads(docs.read_text(encoding="utf-8").splitlines()[index])
    messages = row["messages"][:2 * exchanges]
    ids, mask = render(tokenizer, messages)
    names = {i: name for name, i in tokenizer.special.items()}
    return {
        "note": f"the first {len(messages) // 2} of the {len(row['messages']) // 2} exchanges of held-out conversation "
                f"{index} ({row['source']}), rendered; mask is 1 on the tokens that carry the loss; loss_before and "
                "loss_after are -log p(token | the tokens before it) in nats under the base model and the tuned "
                "one, the conversation alone in the context (null for the first token, which is given)",
        "index": index, "source": row["source"], "exchanges": len(messages) // 2,
        "exchanges_in_full": len(row["messages"]) // 2, "messages": messages, "ids": ids,
        "tokens": [names.get(i) or tokenizer.decode([i]) for i in ids],
        "bytes": [list(tokenizer.token_bytes(i)) if i not in names else None for i in ids],
        "special": [int(i in names) for i in ids], "mask": mask,
    }


def continuation(model, tokenizer, text: str, **sampling) -> dict:
    """What the model writes after <|bos|> and the bare text; it may end the document itself with <|bos|>."""
    stop = {tokenizer.special[BOS]: BOS}
    written, stopped = reply(model, tokenizer.encode(text, bos=True), stop, **sampling)
    return {"text": tokenizer.decode(written), "tokens": len(written), "stopped": "bos" if stopped == BOS else stopped}


def chat_answer(model, tokenizer, text: str, **sampling) -> dict:
    written, info = answer(model, tokenizer, [{"role": "user", "content": text}], **sampling)
    return {"text": written, **info}


def measure(args: argparse.Namespace) -> dict:
    start = time.time()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    base_file = checkpoint_path(args.base, args.base_ckpt)
    data = ChatData(args.data)
    saved = torch.load(base_file, map_location="cpu", weights_only=True)
    base_info = {"checkpoint": f"{base_file.parent.name}/{base_file.name}", "step": saved.get("step"),
                 "tokens": saved.get("tokens")}
    del saved
    sampling = {"temperature": args.temperature, "seed": args.seed}

    # ---- the base model's half: it does not depend on the stage, so it can be computed while the stage runs
    # (--base-only) and taken from its file afterwards (--reuse-base) if nothing it depends on has changed
    from .generate import load_tokenizer
    tokenizer = load_tokenizer(data.meta["tokenizer"])
    end_id = tokenizer.special[ASSISTANT_END]
    shown = example(tokenizer, data, args.example, args.example_exchanges)
    stat = base_file.stat()
    key = {"checkpoint": str(base_file.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
           "held_out_tokens": int(data.meta["n_val_tokens"]), "example": shown["ids"], **sampling,
           "base_tokens": args.base_tokens, "questions": [text for _, text in QUESTIONS]}
    base_cache = out.with_name(out.stem + ".base.json")
    before = None
    if args.reuse_base and base_cache.exists():
        cached = json.loads(base_cache.read_text(encoding="utf-8"))
        before = cached["result"] if cached["key"] == key else None
        print("the base model's half: " + ("taken from " + base_cache.name if before else "its file is of another "
              "checkpoint, data or setting; computed again"), flush=True)
    if before is None:
        base, _ = load_run(str(base_file), tokenizer=data.meta["tokenizer"])
        before = {
            "held_out": held_out_loss(base, data, end_id, args.batch),
            "roles": held_out_by_role(base, data, tokenizer.special, args.batch),
            "example": token_losses(base, shown["ids"]),
            "questions": [{"greedy": continuation(base, tokenizer, text, max_new_tokens=args.base_tokens),
                           "sampled": continuation(base, tokenizer, text, max_new_tokens=args.base_tokens, **sampling)}
                          for _, text in QUESTIONS],
        }
        del base
        torch.cuda.empty_cache()
        base_cache.write_text(json.dumps({"key": key, "result": before}, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"the base model's half: held-out loss {before['held_out']['loss']:.4f}, {len(QUESTIONS)} continuations "
              f"({time.time() - start:.0f} s)", flush=True)
    def on_pretraining_windows(model) -> float:
        """The loss on the held-out windows the base run's trainer evaluated on (data.py's ValLoader: the first
        windows of the pretraining data's held-out tokens), so that a model can be placed on that run's curve."""
        device, seq_len = model.embed.weight.device, model.cfg.block_size
        return evaluate(model, plain_batches(args.plain_data, seq_len, args.batch, args.plain_windows, device))["loss"]

    windows = f"{args.plain_data}:{args.plain_windows}"
    if args.plain_data != "none" and windows not in before.get("pretraining_windows", {}):
        base, _ = load_run(str(base_file), tokenizer=data.meta["tokenizer"])
        before.setdefault("pretraining_windows", {})[windows] = on_pretraining_windows(base)
        del base
        torch.cuda.empty_cache()
        base_cache.write_text(json.dumps({"key": key, "result": before}, ensure_ascii=False) + "\n", encoding="utf-8")
    base_plain = out.parent / f"plain-text-{base_file.parent.name}.json"
    if args.base_only:
        if not args.skip_plain_text:
            plain_text(base_file, base_plain, extra=args.evaluate_args, reuse=args.reuse_base)
        print(f"wrote {base_cache}", flush=True)
        return before

    chat_file = checkpoint_path(args.run)
    chat_dir = chat_file.parent
    config = json.loads((chat_dir / "config.json").read_text())
    summary = json.loads((chat_dir / "summary.json").read_text())
    tuned, _ = load_run(str(chat_file), tokenizer=data.meta["tokenizer"])

    meta = data.meta
    result: dict = {
        "what": "the chat stage of gpt-pp, measured (gptpp/chat_eval.py); every number is computed by that module "
                "from the two checkpoints named here, except the tally, which a reader made",
        "base": base_info,
        "chat": {
            "run": chat_dir.name, "base": config["base"], "lr_scale": config["lr_scale"],
            "groups": config["derived"]["groups"], "steps": summary["steps"], "tokens": summary["tokens"],
            "loss_tokens": summary["loss_tokens"], "passes": summary["passes"], "seconds": summary["seconds"],
            "tokens_per_second": summary["tokens_per_second"], "peak_memory_gb": summary["peak_memory_gb"],
            "batch": [config["train_config"]["device_batch"], config["derived"]["seq_len"],
                      config["train_config"]["grad_accum"]],
            "schedule": {key: config["train_config"][key] for key in ("schedule", "warmup_tokens", "final_lr_frac")},
            "hardware": config["hardware"],
        },
        "data": {
            "source": meta["source"], "license": meta["license"], "format": meta["format"],
            "train": {key: meta["train"][key] for key in ("conversations", "exchanges", "tokens", "loss_tokens")},
            "train_by_source": {name: part["conversations"] for name, part in meta["train"]["by_source"].items()},
            "train_by_category": {name: part["conversations"] for name, part in meta["train"]["by_category"].items()},
            "held_out": {key: meta["val"][key] for key in ("conversations", "exchanges", "tokens", "loss_tokens")},
            "selection": {key: meta["selection"][key] for key in ("rule", "max_answer_tokens", "context",
                                                                   "dropped_categories", "dropped_qualities")},
            "repeats": meta["selection"]["repeats"],
        },
    }

    # ---- 1. held-out conversations
    result["held_out"] = {
        "note": "mean cross-entropy in nats per token over the tokens that carry the loss (the assistant's text "
                "and its <|assistant_end|>) of the held-out conversations, which no training conversation "
                "repeats; loss_text and loss_end are the same mean over the text tokens and over the end tokens "
                "alone. The conversations are packed whole into rows of the context length, as in training, and "
                "scored in bfloat16 autocast with the loss in float32. by_source packs each source's "
                "conversations on their own.",
        "before": before["held_out"],
        "after": held_out_loss(tuned, data, end_id, args.batch),
    }
    print(f"held-out loss {result['held_out']['before']['loss']:.4f} -> {result['held_out']['after']['loss']:.4f} "
          f"({time.time() - start:.0f} s)", flush=True)

    result["roles"] = {
        "note": "what a token of the held-out conversations costs, by whose token it is: the mean of -log p(token | "
                "the tokens before it) in nats over the same rows as held_out. assistant_text and assistant_end "
                "carry the loss in the stage; user_text (the user's messages) and markers (<|user_start|>, "
                "<|user_end|>, <|assistant_start|>) are only read. <|bos|> and padding are not scored.",
        "before": before["roles"],
        "after": held_out_by_role(tuned, data, tokenizer.special, args.batch),
    }
    result["example"] = {**shown, "loss_before": [None, *before["example"]],
                         "loss_after": [None, *token_losses(tuned, shown["ids"])]}
    if args.plain_data != "none":
        result["pretraining_windows"] = {
            "note": "mean cross-entropy in nats per token on the held-out windows of the pretraining data that the "
                    "base run's trainer evaluated on (the first `windows` windows of the context length, every "
                    "token a target, bfloat16 autocast): `before` is the base model there and reproduces the last "
                    "val_loss of its run's log; `after` places the tuned model on that run's curve",
            "data": args.plain_data, "windows": args.plain_windows,
            "before": before["pretraining_windows"][windows], "after": on_pretraining_windows(tuned),
        }

    # ---- 2. the twenty questions
    questions = []
    for qid, (kind, text), continued in zip(question_ids(), QUESTIONS, before["questions"]):
        questions.append({
            "id": qid, "kind": kind, "text": text, "base": continued,
            "chat": {"greedy": chat_answer(tuned, tokenizer, text, max_new_tokens=args.chat_tokens),
                     "sampled": chat_answer(tuned, tokenizer, text, max_new_tokens=args.chat_tokens, **sampling)},
        })
    stops = Counter(q["chat"][mode]["stopped"] for q in questions for mode in ("greedy", "sampled"))
    result["generation"] = {
        "note": "base: the base model continues <|bos|> and the bare text of the question, for at most "
                f"{args.base_tokens} tokens or until it writes <|bos|> itself (stopped: bos); chat: the tuned model "
                "is given <|bos|><|user_start|> question <|user_end|><|assistant_start|> and writes until its "
                f"<|assistant_end|> (stopped: end) or {args.chat_tokens} tokens (stopped: length). greedy takes "
                f"the most likely token; sampled draws at temperature {args.temperature} from the whole "
                f"distribution with seed {args.seed}. Decoding uses the KV cache, in bfloat16 autocast.",
        "temperature": args.temperature, "seed": args.seed, "base_max_new_tokens": args.base_tokens,
        "chat_max_new_tokens": args.chat_tokens, "chat_stopped": dict(stops),
        "chat_stopped_by_itself": stops["end"], "chat_answers": 2 * len(questions),
    }
    result["questions"] = questions
    print(f"{len(questions)} questions answered; the tuned model stopped by itself in {stops['end']} of "
          f"{2 * len(questions)} answers ({time.time() - start:.0f} s)", flush=True)
    del tuned
    torch.cuda.empty_cache()
    # what has been measured so far is written at once: the steps below run other programs, which can fail
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    # ---- 3. plain text
    if not args.skip_plain_text:
        plain_before = plain_text(base_file, base_plain, extra=args.evaluate_args, reuse=args.reuse_base)
        after = plain_text(chat_file, out.parent / f"plain-text-{chat_dir.name}.json", extra=args.evaluate_args,
                           reuse=args.reuse_base)
        assert plain_before["docs_sha256"] == after["docs_sha256"]
        result["plain_text"] = {
            "note": "bits per byte on the held-out FineWeb-EDU documents, the protocol of gptpp/evaluate.py: each "
                    "document cut into windows of <|bos|> and 1,023 tokens, every token scored once, the bits "
                    "divided by the documents' UTF-8 bytes; se is the standard error over documents. The trainer's "
                    "own monitor (the loss on 128 held-out windows of the pretraining data, logged at each "
                    "evaluation of the stage) is in the run's log as plain_loss.",
            "before": plain_before, "after": after,
            "cost_bits_per_byte": after["bits_per_byte"] - plain_before["bits_per_byte"],
            "monitor_before": summary["before"]["plain_loss"], "monitor_after": summary["plain_loss"],
        }
        print(f"plain text {plain_before['bits_per_byte']:.4f} -> {after['bits_per_byte']:.4f} bits per byte "
              f"({time.time() - start:.0f} s)", flush=True)

    # ---- 4. the probes that chose the rate
    probes = []
    for name in [n for n in (args.probes or "").split(",") if n]:
        folder = repo_dir() / "results" / "runs" / name
        probe_config = json.loads((folder / "config.json").read_text())
        probe = json.loads((folder / "summary.json").read_text())
        probes.append({"run": name, "lr_scale": probe_config["lr_scale"], "base": probe_config["base"],
                       "tokens": probe["tokens"], "steps": probe["steps"], "val_loss": probe["val_loss"],
                       "val_loss_text": probe["val_loss_text"], "val_loss_end": probe["val_loss_end"],
                       "plain_loss": probe["plain_loss"], "before": probe["before"],
                       "schedule": {key: probe_config["train_config"][key]
                                    for key in ("schedule", "warmup_tokens", "final_lr_frac")}})
    if probes:
        result["probes"] = {
            "note": "short runs of the stage that differ only in lr_scale, the factor on every peak rate of "
                    "pretraining, each from the checkpoint named in its `base` (a probe run before the base model "
                    "had finished names an earlier checkpoint than the stage); val_loss is the held-out "
                    "conversation loss at the end of each, plain_loss the loss on 128 held-out windows of the "
                    "pretraining data. One seed each.",
            "runs": probes,
        }

    result["seconds"] = round(time.time() - start, 1)
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}", flush=True)
    return result


def add_run(args: argparse.Namespace) -> None:
    """Measure another full run of the stage from the same base (another rate, another seed) as the first was
    measured, and put it beside it in chat-stage.json under `other_runs`. The twenty questions are not asked of
    it and it has no tally: the answers of the file stay those of the first run."""
    start = time.time()
    out = Path(args.out)
    result = json.loads(out.read_text(encoding="utf-8"))
    chat_file = checkpoint_path(args.add_run)
    chat_dir = chat_file.parent
    config = json.loads((chat_dir / "config.json").read_text())
    summary = json.loads((chat_dir / "summary.json").read_text())
    if config["base"] != result["base"]:
        raise SystemExit(f"{chat_dir.name} starts from {config['base']}, the file's run from {result['base']}")
    data = ChatData(args.data)
    tuned, tokenizer = load_run(str(chat_file), tokenizer=data.meta["tokenizer"])
    device, seq_len = tuned.embed.weight.device, tuned.cfg.block_size
    entry = {
        "run": chat_dir.name, "lr_scale": config["lr_scale"], "groups": config["derived"]["groups"],
        "steps": summary["steps"], "tokens": summary["tokens"], "loss_tokens": summary["loss_tokens"],
        "passes": summary["passes"], "seconds": summary["seconds"], "tokens_per_second": summary["tokens_per_second"],
        "peak_memory_gb": summary["peak_memory_gb"],
        "batch": [config["train_config"]["device_batch"], config["derived"]["seq_len"],
                  config["train_config"]["grad_accum"]],
        "schedule": {key: config["train_config"][key] for key in ("schedule", "warmup_tokens", "final_lr_frac")},
        "held_out": held_out_loss(tuned, data, tokenizer.special[ASSISTANT_END], args.batch),
        "roles": held_out_by_role(tuned, data, tokenizer.special, args.batch),
    }
    if args.plain_data != "none":
        windows = plain_batches(args.plain_data, seq_len, args.batch, args.plain_windows, device)
        entry["pretraining_windows"] = {"data": args.plain_data, "windows": args.plain_windows,
                                        "after": evaluate(tuned, windows)["loss"]}
    del tuned
    torch.cuda.empty_cache()
    if not args.skip_plain_text:
        entry["plain_text"] = plain_text(chat_file, out.parent / f"plain-text-{chat_dir.name}.json",
                                         extra=args.evaluate_args, reuse=args.reuse_base)
        if entry["plain_text"]["docs_sha256"] != result["plain_text"]["before"]["docs_sha256"]:
            raise SystemExit("the plain-text documents are not those of the file's run")
    entry["monitor"] = {"note": "what the run's own log says at its last evaluation",
                        **{key: summary[key] for key in ("val_loss", "val_loss_text", "val_loss_end", "plain_loss")}}
    others = result.setdefault("other_runs", {"runs": []})
    others["note"] = (
        "other full runs of the stage from the same base checkpoint, each measured as the run of this file: "
        "held_out (the loss on the tokens that carry it over the same held-out conversations, with loss_text and "
        "loss_end, and by source), roles (the same rows, by whose token it is), pretraining_windows (the base "
        "run's own held-out windows, nats a token) and plain_text (bits per byte on the same held-out documents by "
        "gptpp.evaluate) are computed as the sections of those names above say; `before` is the same for every "
        "run and stands in those sections. No questions were asked of these runs and they have no tally.")
    others["runs"] = [run for run in others["runs"] if run["run"] != entry["run"]] + [entry]
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    plain = entry.get("plain_text")
    print(f"{entry['run']} (x{entry['lr_scale']}): held-out loss {entry['held_out']['loss']:.4f}, end token "
          f"{entry['held_out']['loss_end']:.4f}"
          + (f", plain text {plain['bits_per_byte']:.4f} bits per byte" if plain else "")
          + f" ({time.time() - start:.0f} s); added to {out}", flush=True)


# ----------------------------------------------------------------------------------------------------------------
# The tally, which a reader makes


def tally_template(args: argparse.Namespace) -> None:
    """Write the tally file with every answer's hash and an empty verdict, keeping verdicts already given to
    the same answers."""
    result = json.loads(Path(args.out).read_text(encoding="utf-8"))
    path = Path(args.tally_file)
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    verdicts = {}
    for q in result["questions"]:
        verdicts[q["id"]] = {"question": q["text"]}
        for mode in ("greedy", "sampled"):
            sha = digest(q["chat"][mode]["text"])
            kept = old.get("verdicts", {}).get(q["id"], {}).get(mode, {})
            same = kept.get("sha") == sha
            verdicts[q["id"]][mode] = {"sha": sha, "verdict": kept.get("verdict") if same else None,
                                       "note": kept.get("note", "") if same else ""}
    template = {"judge": old.get("judge", ""), "criteria": CRITERIA, "verdicts": verdicts}
    path.write_text(json.dumps(template, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {path}: fill in every verdict with one of {VERDICTS}")


def add_tally(args: argparse.Namespace) -> None:
    """Check the tally against the answers it judged and put it into chat-stage.json."""
    out = Path(args.out)
    result = json.loads(out.read_text(encoding="utf-8"))
    tally = json.loads(Path(args.tally_file).read_text(encoding="utf-8"))
    counts = {mode: Counter() for mode in ("greedy", "sampled")}
    by_kind: dict[str, Counter] = {}
    for q in result["questions"]:
        for mode in ("greedy", "sampled"):
            entry = tally["verdicts"][q["id"]][mode]
            if entry["sha"] != digest(q["chat"][mode]["text"]):
                raise SystemExit(f"{q['id']} ({mode}): the tally judged another answer; make the template again")
            if entry["verdict"] not in VERDICTS:
                raise SystemExit(f"{q['id']} ({mode}): the verdict must be one of {VERDICTS}, not {entry['verdict']!r}")
            q["chat"][mode]["verdict"], q["chat"][mode]["verdict_note"] = entry["verdict"], entry.get("note", "")
            counts[mode][entry["verdict"]] += 1
            if mode == "greedy":
                by_kind.setdefault(q["kind"], Counter())[entry["verdict"]] += 1
    result["tally"] = {
        "note": "not computed: every answer of the tuned model was read and judged once by " + tally["judge"]
                + ", by the criteria below; nothing was regenerated or dropped, and the answers judged wrong or off "
                "the subject stand in `questions` with their verdicts. by_kind counts the greedy answers.",
        "judge": tally["judge"], "criteria": tally["criteria"],
        "greedy": {v: counts["greedy"][v] for v in VERDICTS}, "sampled": {v: counts["sampled"][v] for v in VERDICTS},
        "by_kind": {kind: {v: c[v] for v in VERDICTS} for kind, c in by_kind.items()},
        "answers": len(result["questions"]),
    }
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"tally: greedy {dict(counts['greedy'])}, sampled {dict(counts['sampled'])}; wrote {out}")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="gptpp-base", help="the pretrained run, or a checkpoint file")
    parser.add_argument("--base-ckpt", default="model.pt")
    parser.add_argument("--run", default="gptpp-chat", help="the run of the chat stage")
    parser.add_argument("--data", default="chat/bpe32k")
    parser.add_argument("--out", default=str(repo_dir() / "results" / "chat" / "chat-stage.json"))
    parser.add_argument("--probes", default="", help="comma-separated probe runs (under results/runs) to record")
    parser.add_argument("--batch", type=int, default=16, help="rows per forward pass of the held-out loss")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-tokens", type=int, default=128, help="tokens of the base model's continuation")
    parser.add_argument("--chat-tokens", type=int, default=320,
                        help="most tokens of an answer (the longest in the data)")
    parser.add_argument("--example", type=int, default=267, help="the held-out conversation shown token by token")
    parser.add_argument("--example-exchanges", type=int, default=2, help="its first exchanges kept")
    parser.add_argument("--skip-plain-text", action="store_true", help="do not run gptpp.evaluate")
    parser.add_argument("--plain-data", default="fineweb-edu/bpe32k",
                        help="the pretraining data, for the loss on its trainer's held-out windows; 'none' to skip")
    parser.add_argument("--plain-windows", type=int, default=1024, help="as many as the base run's trainer evaluated")
    parser.add_argument("--base-only", action="store_true",
                        help="compute the base model's half only and keep it in <out stem>.base.json")
    parser.add_argument("--reuse-base", action="store_true",
                        help="take the base model's half from that file if it is of the same checkpoint and settings, "
                             "and the plain-text results already in their files for the same checkpoints")
    parser.add_argument("--evaluate-args", default="", help="further arguments of gptpp.evaluate (a trial: "
                                                            "'--max-docs 20 --batch-tokens 4096')")
    parser.add_argument("--tally-file", default=str(repo_dir() / "results" / "chat" / "chat-tally.json"))
    parser.add_argument("--add-run", default=None, metavar="RUN",
                        help="measure this other run of the stage and put it beside the first in the file")
    parser.add_argument("--tally-template", action="store_true")
    parser.add_argument("--tally", action="store_true")
    args = parser.parse_args(argv)
    if args.add_run:
        add_run(args)
    elif args.tally_template:
        tally_template(args)
    elif args.tally:
        add_tally(args)
    else:
        measure(args)


if __name__ == "__main__":
    main()
