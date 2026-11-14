"""Prepare the conversations of the chat stage: download, select, render with the special tokens, write shards.

    HF_TOKEN=$(cat <token file>) python scripts/prepare_chat.py [--tokenizer bpe32k] [--max-answer-tokens 320]

The corpus is SmolTalk (Allal et al., "SmolLM2: when smol goes big", 2025, arXiv:2502.02737), the
instruction-tuning set of the SmolLM2 models, read as parquet from the Hugging Face Hub at a pinned revision.
It is a mix of subsets. Three of them are plain conversations between a user and an assistant, and those are
the ones used here:

    everyday-conversations   greetings and small talk, three or four exchanges each (Apache 2.0, from
                             HuggingFaceTB/everyday-conversations-llama3.1-2k)
    smol-constraints         one instruction with constraints on the form of the answer ("at least three
                             sentences", "include the word ...") and its answer (Apache 2.0)
    smol-magpie-ultra        the core of SmolTalk: three exchanges on advice, brainstorming, creative writing,
                             data analysis, editing, information, mathematics, planning, reasoning, role play
                             and coding, written by Llama 3.1 405B with the Magpie pipeline (Apache 2.0)

Left out: tool calls (apigen-80k), long documents (longalign), code (self-oss-instruct), competition mathematics
(numina-cot-100k, metamathqa-50k), the subsets driven by a system prompt (smol-rewrite, smol-summarize,
systemchats-30k, explore-instruct-rewriting) and openhermes-100k, whose source states no license.

The selection, made for a model of 135M parameters with a context of 1,024 tokens whose pretraining text held
no code and no braces. A conversation is kept up to its last exchange (a user's message and the assistant's
answer) such that, from the first exchange on,

    1. neither message contains a code fence, a brace, a backslash or a markup tag,
    2. the answer has at most `--max-answer-tokens` tokens (concise answers; SmolLM2's authors select "more
       concise conversations" for their models of this size), and
    3. the conversation so far, rendered, fits in the context;

and it is dropped if its first exchange fails. smol-magpie-ultra loses its coding category and the
conversations its authors rated of poor quality. No conversation is cut inside a message.

Rendering (gptpp/chat.py): <|bos|>, then <|user_start|> text <|user_end|> and <|assistant_start|> text
<|assistant_end|> in turn. The mask says which tokens carry the loss: the assistant's text and its
<|assistant_end|>.

What is written, under $GPTPP_WORK/data/chat/ (Contract 1 of the lab, plus the masks):

    <tokenizer>/train_0000.bin ...       uint16 little-endian token ids, conversations one after another
    <tokenizer>/train_0000.mask.bin ...  uint8, one per token: 1 where the token carries the loss
    <tokenizer>/val.bin, val.mask.bin    the held-out conversations
    <tokenizer>/meta.json                counts by source, the rules above with their numbers, source, licenses
    val_conversations.jsonl              the held-out conversations as text, in the order of val.bin

Repeats. The corpus repeats some prompts thousands of times (one riddle about a snail in a well opens 4,511
of the training conversations that pass the rules). Of those that open with the same user message of at least
`--long-opening` characters, `--max-repeats` are kept. The held-out conversations are a seeded sample of the
corpus's own test split, selected by the same rules, among those whose opening (if that long) opens no
training conversation: the held-out questions are new. Greetings are short and shared by nature ("Hi"); a
training conversation identical in full to a held-out one is dropped.
"""

from __future__ import annotations

import argparse
import glob
import json
import multiprocessing
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

# This directory holds queue.py (the job queue), which would shadow the standard library's `queue` that
# multiprocessing needs: take the directory off the import path and put the repository's root there.
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gptpp import work_dir  # noqa: E402
from gptpp.chat import assemble  # noqa: E402
from gptpp.tokenizer import Tokenizer  # noqa: E402

REPO = "HuggingFaceTB/smoltalk"
REVISION = "5feaf2fd3ffca7c237fc38d1861bc30365d48ffa"
CONFIGS = ("everyday-conversations", "smol-constraints", "smol-magpie-ultra")
LICENSES = {
    "everyday-conversations": "Apache 2.0 (HuggingFaceTB/everyday-conversations-llama3.1-2k)",
    "smol-constraints": "Apache 2.0",
    "smol-magpie-ultra": "Apache 2.0",
}
DROPPED_CATEGORIES = ("coding",)  # of smol-magpie-ultra: the pretraining text held no code
DROPPED_QUALITIES = ("poor", "very poor")  # of smol-magpie-ultra: its authors' own rating
NOT_PLAIN = re.compile(r"```|[{}\\]|</?[a-zA-Z]+>")  # a code fence, a brace, a backslash, a markup tag
VAL_CONVERSATIONS = {"everyday-conversations": 119, "smol-constraints": 660, "smol-magpie-ultra": 2221}
WORK = work_dir()  # $GPTPP_WORK


def download(raw: Path) -> None:
    """Fetch the parquet files of the three configs (and the dataset card) at the pinned revision."""
    from huggingface_hub import HfApi, hf_hub_download

    token = os.environ.get("HF_TOKEN")
    names = [s.rfilename for s in HfApi(token=token).dataset_info(REPO, revision=REVISION).siblings]
    wanted = [n for n in names if n.endswith(".parquet") and n.split("/")[1] in CONFIGS]
    for name in ["README.md", *wanted]:
        if not (raw / name).exists():
            hf_hub_download(REPO, name, repo_type="dataset", revision=REVISION, local_dir=raw, token=token)


def read_config(raw: Path, config: str, split: str) -> list[dict]:
    """The conversations of one config and split: {"source", "category", "messages"}, in the files' order."""
    import pyarrow.parquet as pq

    rows: list[dict] = []
    for path in sorted(glob.glob(str(raw / "data" / config / f"{split}-*.parquet"))):
        table = pq.read_table(path)
        messages = table["messages"].to_pylist()
        if config == "smol-magpie-ultra":
            categories, qualities = table["category"].to_pylist(), table["quality"].to_pylist()
        else:
            categories, qualities = [config] * len(messages), [""] * len(messages)
        for m, category, quality in zip(messages, categories, qualities):
            if category in DROPPED_CATEGORIES or quality in DROPPED_QUALITIES:
                continue
            rows.append({"source": config, "category": category, "messages": m})
    return rows


# -- selecting and rendering in worker processes ---------------------------------------------------

_tok: Tokenizer | None = None
_limits: tuple[int, int] = (0, 0)


def _init_worker(tokenizer: str, max_answer_tokens: int, context: int) -> None:
    global _tok, _limits
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # one core for each worker process
    _tok = Tokenizer.load(tokenizer)
    _limits = (max_answer_tokens, context)


def _select(batch: list[dict]) -> list[tuple[int, np.ndarray, np.ndarray, int, int]]:
    """For each conversation of the batch that keeps at least one exchange: its index in the batch, its token
    ids, its mask, the exchanges kept and the UTF-8 bytes of the messages kept."""
    max_answer, context = _limits
    texts: list[list[str]] = []
    for row in batch:
        messages = row["messages"]
        ok = len(messages) >= 2 and all(m["role"] == ("user", "assistant")[i % 2] for i, m in enumerate(messages))
        texts.append([m["content"].strip() for m in messages] if ok else [])
    flat = [text for conversation in texts for text in conversation]
    encoded = iter(_tok.encode_batch(flat, threads=1))
    out = []
    for index, conversation in enumerate(texts):
        pieces = [next(encoded) for _ in conversation]
        kept, length = 0, 1  # exchanges kept; tokens of the conversation so far, <|bos|> included
        for k in range(len(conversation) // 2):
            question, answer = conversation[2 * k], conversation[2 * k + 1]
            if not question or not answer or NOT_PLAIN.search(question) or NOT_PLAIN.search(answer):
                break
            n_question, n_answer = len(pieces[2 * k]), len(pieces[2 * k + 1])
            if n_answer > max_answer or length + n_question + n_answer + 4 > context:
                break
            kept, length = k + 1, length + n_question + n_answer + 4
        if kept == 0:
            continue
        ids, mask = assemble(["user", "assistant"] * kept, pieces[:2 * kept], _tok.special)
        n_bytes = sum(len(text.encode("utf-8")) for text in conversation[:2 * kept])
        out.append((index, np.array(ids, dtype="<u2"), np.array(mask, dtype=np.uint8), kept, n_bytes))
    return out


def select(rows: list[dict], pool: multiprocessing.pool.Pool, batch: int = 1000) -> list[dict]:
    """The conversations that pass the rules, in the order given, each with its kept messages, ids and mask."""
    batches = [rows[i:i + batch] for i in range(0, len(rows), batch)]
    kept: list[dict] = []
    for part, results in zip(batches, pool.imap(_select, batches)):
        for index, ids, mask, exchanges, n_bytes in results:
            row = part[index]
            messages = [{"role": m["role"], "content": m["content"].strip()} for m in row["messages"][:2 * exchanges]]
            kept.append({"source": row["source"], "category": row["category"], "messages": messages, "ids": ids,
                         "mask": mask, "exchanges": exchanges, "bytes": n_bytes})
    return kept


def counts(conversations: list[dict]) -> dict:
    """Conversations, exchanges, tokens and loss-carrying tokens, in all and by source and category."""
    def one(group: list[dict]) -> dict:
        return {"conversations": len(group), "exchanges": sum(c["exchanges"] for c in group),
                "tokens": int(sum(len(c["ids"]) for c in group)),
                "loss_tokens": int(sum(int(c["mask"].sum()) for c in group))}

    by_source = {s: one([c for c in conversations if c["source"] == s]) for s in CONFIGS}
    categories = sorted({c["category"] for c in conversations if c["source"] == "smol-magpie-ultra"})
    by_category = {k: one([c for c in conversations if c["category"] == k]) for k in categories}
    return {**one(conversations), "by_source": by_source, "by_category": by_category}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokenizer", default="bpe32k")
    parser.add_argument("--max-answer-tokens", type=int, default=320, help="longest answer kept, in tokens")
    parser.add_argument("--context", type=int, default=1024,
                        help="a rendered conversation has at most this many tokens")
    parser.add_argument("--max-repeats", type=int, default=20,
                        help="training conversations kept that open with the same long user message")
    parser.add_argument("--long-opening", type=int, default=40, help="characters from which an opening counts as long")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard-tokens", type=float, default=100e6, help="a shard is closed once it holds this many")
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 24))
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()

    start = time.time()
    raw = WORK / "data" / "raw" / "smoltalk"
    out = WORK / "data" / "chat" / args.tokenizer
    download(raw)
    if args.download_only:
        return
    out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.load(args.tokenizer)
    rng = np.random.default_rng(args.seed)

    def opening(conversation: dict) -> str:
        return conversation["messages"][0]["content"]

    def is_long(text: str) -> bool:
        return len(text) >= args.long_opening

    read: dict[str, dict[str, int]] = {}
    train: list[dict] = []
    tests: dict[str, list[dict]] = {}
    initializer = (args.tokenizer, args.max_answer_tokens, args.context)
    with multiprocessing.get_context("fork").Pool(args.workers, _init_worker, initializer) as pool:
        for config in CONFIGS:
            rows_train, rows_test = read_config(raw, config, "train"), read_config(raw, config, "test")
            kept_train, tests[config] = select(rows_train, pool), select(rows_test, pool)
            train += kept_train
            read[config] = {"train_read": len(rows_train), "train_kept": len(kept_train),
                            "test_read": len(rows_test), "test_kept": len(tests[config])}
            print(f"{config}: {len(kept_train):,} of {len(rows_train):,} training and {len(tests[config]):,} of "
                  f"{len(rows_test):,} test conversations pass the rules ({time.time() - start:.0f} s)", flush=True)

    # Held out: test conversations whose question is new, that is, whose long opening opens no training conversation.
    repeats = Counter(opening(c) for c in train if is_long(opening(c)))
    val: list[dict] = []
    for config in CONFIGS:
        new = [c for c in tests[config] if opening(c) not in repeats]
        sample = np.sort(rng.permutation(len(new))[:VAL_CONVERSATIONS[config]])
        val += [new[i] for i in sample]
        read[config].update(test_with_new_opening=len(new), held_out=len(sample))
    val = [val[i] for i in rng.permutation(len(val))]

    # Training: shuffled; at most max_repeats conversations with the same long opening; none identical to a
    # held-out one.
    def whole(conversation: dict) -> tuple:
        return tuple(m["content"] for m in conversation["messages"])

    held_out_whole = {whole(c) for c in val}
    shuffled = [train[i] for i in rng.permutation(len(train))]
    taken: Counter = Counter()
    train, n_repeats_dropped, n_identical_dropped = [], 0, 0
    for c in shuffled:
        if whole(c) in held_out_whole:
            n_identical_dropped += 1
            continue
        if is_long(opening(c)):
            taken[opening(c)] += 1
            if taken[opening(c)] > args.max_repeats:
                n_repeats_dropped += 1
                continue
        train.append(c)
    most_repeated = [{"opening": text[:80], "training_conversations": n} for text, n in repeats.most_common(5)]
    print(f"repeats: {n_repeats_dropped:,} training conversations dropped beyond {args.max_repeats} of the same "
          f"opening (the most repeated opens {repeats.most_common(1)[0][1]:,}); {n_identical_dropped} identical to a "
          f"held-out conversation dropped", flush=True)

    # Training shards: whole conversations until a shard has `shard_tokens` tokens; a mask file beside each.
    shards: list[str] = []
    shard_tokens: list[int] = []
    pending: list[dict] = []

    def close_shard() -> None:
        name = f"train_{len(shards):04d}.bin"
        tokens = np.concatenate([c["ids"] for c in pending])
        tokens.tofile(out / name)
        np.concatenate([c["mask"] for c in pending]).tofile(out / name.replace(".bin", ".mask.bin"))
        shards.append(name)
        shard_tokens.append(len(tokens))
        pending.clear()
        print(f"  {name}: {len(tokens):,} tokens ({time.time() - start:.0f} s)", flush=True)

    size = 0
    for conversation in train:
        pending.append(conversation)
        size += len(conversation["ids"])
        if size >= args.shard_tokens:
            close_shard()
            size = 0
    if pending:
        close_shard()
    for stale in sorted(out.glob("train_*.bin")):  # shards of an earlier, longer run
        if stale.name.replace(".mask.bin", ".bin") not in shards:
            stale.unlink()

    val_tokens = np.concatenate([c["ids"] for c in val])
    val_tokens.tofile(out / "val.bin")
    np.concatenate([c["mask"] for c in val]).tofile(out / "val.mask.bin")
    with open(out.parent / "val_conversations.jsonl", "w", encoding="utf-8") as f:
        for c in val:
            f.write(json.dumps({"source": c["source"], "category": c["category"], "messages": c["messages"]},
                               ensure_ascii=False) + "\n")

    train_counts, val_counts = counts(train), counts(val)
    val_bytes = sum(c["bytes"] for c in val)
    lengths = np.array([len(c["ids"]) for c in train])
    answers = Counter()
    for c in train:
        answers[c["exchanges"]] += 1
    meta = {
        "dataset": "chat",
        "tokenizer": args.tokenizer,
        "vocab_size": tok.vocab_size,
        "bos_id": tok.bos_id,
        "special_tokens": tok.special,
        "dtype": "uint16 little-endian",
        "mask": "beside every token file, <name>.mask.bin: uint8, one per token, 1 where the token carries the loss "
                "(the assistant's text and its <|assistant_end|>)",
        "format": "<|bos|> then, in turn, <|user_start|> text <|user_end|> and <|assistant_start|> text "
                  "<|assistant_end|>",
        "n_train_tokens": train_counts["tokens"],
        "n_val_tokens": val_counts["tokens"],
        "train_shards": shards,
        "train_shard_tokens": shard_tokens,
        "n_train_docs": len(train),
        "train": train_counts,
        "val": val_counts,
        "train_conversation_tokens": {"mean": float(lengths.mean()), "median": float(np.median(lengths)),
                                      "p90": float(np.percentile(lengths, 90)), "max": int(lengths.max())},
        "train_conversations_by_exchanges": {str(k): v for k, v in sorted(answers.items())},
        "val_file": "val.bin",
        "val_docs": len(val),
        "val_docs_file": "../val_conversations.jsonl",
        "val_bytes": val_bytes,
        "bytes_per_token_val": val_bytes / len(val_tokens),
        "note": "token counts include <|bos|> and the four markers; bytes are the UTF-8 bytes of the messages",
        "selection": {
            "configs": list(CONFIGS),
            "dropped_categories": list(DROPPED_CATEGORIES),
            "dropped_qualities": list(DROPPED_QUALITIES),
            "not_plain": NOT_PLAIN.pattern,
            "max_answer_tokens": args.max_answer_tokens,
            "context": args.context,
            "rule": "a conversation is kept up to its last exchange such that, from the first on, no message "
                    "matches not_plain, every answer has at most max_answer_tokens tokens and the rendered "
                    "conversation has at most context tokens; dropped if the first exchange fails",
            "read_and_kept": read,
            "repeats": {
                "rule": "of the training conversations that open with the same user message of at least "
                        "long_opening characters, max_repeats are kept; a held-out conversation's opening, if that "
                        "long, opens no training conversation; a training conversation identical in full to a "
                        "held-out one is dropped",
                "max_repeats": args.max_repeats,
                "long_opening": args.long_opening,
                "n_train_dropped_as_repeats": n_repeats_dropped,
                "n_train_dropped_identical_to_held_out": n_identical_dropped,
                "most_repeated_before": most_repeated,
            },
        },
        "seed": args.seed,
        "source": f"https://huggingface.co/datasets/{REPO} at revision {REVISION}, configs {', '.join(CONFIGS)}",
        "license": "Apache 2.0 (the three configs used; per config: " + "; ".join(
            f"{name}: {text}" for name, text in LICENSES.items()) + ")",
        "seconds": round(time.time() - start, 1),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"chat/{args.tokenizer}: {len(train):,} conversations, {train_counts['tokens']:,} tokens "
          f"({train_counts['loss_tokens']:,} carry the loss) in {len(shards)} shard(s); held out {len(val):,} "
          f"conversations, {val_counts['tokens']:,} tokens; {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
