"""The main run, replayed: the file the site's figure `gpt-training-run-replayed` (fig-07-01) draws,
`assets/data/gpt/training-run-replayed.json`.

    python -m gptpp.export.training_run_replayed [--run gptpp-base] [--out <dir>] [--name training-run-replayed]
        [--score] [--score-budget 420] [--score-batch-tokens 4096] [--device cuda] [--threads 8]

It can be run while the run is still training and again when it has ended: it reads the run's `config.json`,
`log.jsonl` and (once it exists) `summary.json`, and every checkpoint that has been written. What costs time is
kept per checkpoint in a side file (`$GPTPP_WORK/data/f7/training-run-replayed.<run>.json`), so a second call
only does what is new:

    writing     from three fixed openings the model continues by 60 tokens at temperature 0.8, seed 0, the same
                at every checkpoint (`gptpp.generate.generate`, on the CPU in float32, so that the text does not
                depend on which card was free);
    monitor     with --score: the trainer's own held-out loss computed again on the saved checkpoint (the first
                1,048,576 tokens of val.bin in windows of 1,024, bfloat16 autocast: `gptpp.train.evaluate` on
                `gptpp.data.ValLoader`), which gives the held-out curve a point at every checkpoint, also before
                the trainer's first evaluation;
    documents   with --score: bits per byte under the document-level protocol of `gptpp.evaluate` on the first
                1,000 held-out FineWeb-EDU documents, the protocol and the documents of the released models'
                scores, so that the line of GPT-2 small is compared like for like.

Scoring needs a card with about 3 GB free and stops after --score-budget seconds; call again for the rest.

The file, field by field:

    generated_by   "gptpp.export.training_run_replayed"
    run            the run's name
    complete       true when the run has written its summary.json
    model          {n_layer, d_model, n_head, block_size, vocab_size, params: {embedding, ..., total}}
    data           {dataset, tokenizer, bytes_per_token_val, bytes_per_token_train, val_tokens, val_windows}
    plan           {steps, tokens, tokens_per_step, n_gpu, device_batch, seq_len, eval_every_steps, checkpoint_steps}
    schedule       {optimizer, schedule, lr, lr_embedding, lr_unembedding, warmup_tokens, cooldown_frac, final_lr_frac}
    progress       {step, tokens, seconds}: the last training record of the log
    summary        null while the run trains; then the trainer's summary.json without the parameter table, plus
                   hours (seconds / 3600) and train_hours
    train          {step, loss, lr_mult, tok_s, time}: every training record (one every 10 steps): the mean loss in
                   nats per token since the record before, the schedule's multiplier, tokens a second, seconds
                   since the loop began
    val            {step, loss, bpb, time}: the trainer's evaluations (nats per token; bits per byte =
                   loss / ln 2 / bytes_per_token_val)
    levels         what gives the curve meaning, each {key, label, bits_per_byte, what}: `uniform` (15 bits a token
                   over the bytes a token holds), `unigram`, `bigram`, `trigram`, `4gram` (interpolated
                   Kneser-Ney models scored on the held-out stream, from the corpus statistics, in the trainer's
                   convention), and `gpt2`, the released GPT-2 small: its bits_per_byte is its document-level
                   score on the first 1,000 held-out documents (protocol "documents", the protocol of the
                   benchmark tables; also under `docs` {bits_per_byte, se, loss, n_docs, sha256}); `monitor`
                   (null until --score has run) is the same model read the trainer's way: scored once on the text
                   of the trainer's held-out windows, in windows of its own tokens, its bits divided by the bytes
                   the trainer divides ours by {bits_per_byte, loss, tokens, n_docs, bits_per_byte_of_its_text}
                   (see `released_on_monitor_text`)
    openings       [{key, text, tokens}]: the three openings (tokens: the opening's length without the start token)
    sampling       {max_new_tokens, temperature, seed, device, dtype}
    checkpoints    one entry a saved checkpoint, in order: {step, tokens, time (seconds into the run, read off the
                   training records), file, texts: [the three continuations, without the opening],
                   monitor: {loss, bpb} | null, docs: {bits_per_byte, se, loss, n_docs, n_tokens, sha256} | null}
    checkpoints_planned   how many the run will have written when it ends
    held_out       {step, tokens, loss, bpb, source}: the held-out curve the figure draws, the trainer's
                   evaluations merged with `monitor` of the checkpoints (same protocol), by step; source "log"
                   or "checkpoint" (where both exist the log's value is kept)
    numbers        what the post quotes (null where not there yet): tokens, steps, hours, seconds,
                   tokens_per_second, final_val_loss, final_val_bpb, best_val_loss, best_val_bpb, best_step,
                   first_val_bpb, level_<key> for each level, final_docs_bpb, final_docs_se, final_docs_loss,
                   monitor_check (largest |monitor - log| in nats over the steps both have),
                   protocol_gap (largest |docs - monitor| in bits per byte over the checkpoints from step 100 on
                   that have both),
                   crossed: {<level key>: {tokens, step, hours, between: [tokens, tokens], sure, protocol} | null}:
                   where the held-out curve first falls below the level, between two of its points, linear in
                   log10 tokens (`sure`: the two points are within a factor 2.5, so the interpolation means
                   something); the n-gram levels against `held_out`; GPT-2 small (protocol "documents") against
                   `docs` of the checkpoints, like for like.
                   level_gpt2_monitor, crossed_gpt2_monitor: GPT-2 small read the trainer's way on the trainer's
                   held-out text, and where `held_out` passes that level (like for like under the other protocol).
                   post: the readings the post of part 7 quotes, each one of the trainer's own evaluations
                   {step, tokens, val_loss, val_bpb, hours}: first_reading (the first after training began),
                   last_reading, reading_at_1100M (the one nearest 1.1 billion tokens), first_reading_under_gpt2
                   (the first below GPT-2 small's document-level score: two protocols, as the post says; with
                   level and protocol_of_level), decay_start_tokens (where the learning rate begins to fall),
                   reading_at_decay_start (the nearest reading), and, once the run has ended,
                   fall_after_decay_share (the share of the fall since the first reading that came after the
                   decay began) and last_500M {from, to, fall_bpb, fall_loss, bpb_per_100M_tokens}

Losses keep six significant digits.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

from .common import CKPT, DATA, WORK, write_atomic

REPO = Path(__file__).resolve().parents[2]
OPENINGS = [
    {"key": "science", "text": "Photosynthesis is the process by which"},
    {"key": "history", "text": "The Roman Empire fell because"},
    {"key": "geometry", "text": "To find the area of a circle,"},
]
SAMPLING = {"max_new_tokens": 60, "temperature": 0.8, "seed": 0, "device": "cpu", "dtype": "float32"}
NGRAM_KEYS = {1: ("unigram", "unigram"), 2: ("bigram", "bigram"), 3: ("trigram", "trigram"), 4: ("4gram", "4-gram")}
LN2 = math.log(2.0)


def sig(x: float | None, digits: int = 6) -> float | None:
    return None if x is None else float(f"{x:.{digits}g}")


def read_log(path: Path) -> dict:
    train, val = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:          # the last line of a log that is being written
                continue
            if r.get("kind") == "train":
                train.append(r)
            elif r.get("kind") == "val":
                val.append(r)
    return {"train": train, "val": val}


def crossing(tokens: list[float], values: list[float], level: float) -> dict | None:
    """Where a falling curve first lies below `level`: between the last point at or above it and the first below,
    linear in log10 tokens (linear in tokens when the earlier point stands at 0). None if it never does. `sure`
    says that the two points are close enough for the interpolation to mean something: the earlier one is not at 0
    and the later one is at most 2.5 times as far."""
    for i, y in enumerate(values):
        if y < level:
            if i == 0:
                return {"tokens": float(tokens[0]), "between": [float(tokens[0]), float(tokens[0])], "sure": True}
            x0, y0, x1 = tokens[i - 1], values[i - 1], tokens[i]
            f = (y0 - level) / (y0 - y)
            at = x0 + f * (x1 - x0) if x0 <= 0 else 10 ** (math.log10(x0) + f * (math.log10(x1) - math.log10(x0)))
            return {"tokens": float(at), "between": [float(x0), float(x1)], "sure": bool(x0 > 0 and x1 <= 2.5 * x0)}
    return None


def interp(x: float, xs: list[float], ys: list[float]) -> float:
    if x <= xs[0]:
        return ys[0]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            return ys[i - 1] + (ys[i] - ys[i - 1]) * (x - xs[i - 1]) / (xs[i] - xs[i - 1])
    return ys[-1]


def write_samples(ckpt: Path, tokenizer, threads: int) -> dict:
    """The three continuations of one checkpoint: on the CPU, float32, the same seed for each."""
    import torch
    from gptpp.generate import generate
    from gptpp.model import GPT

    torch.set_num_threads(threads)
    model, _ = GPT.from_checkpoint(str(ckpt), device="cpu")
    texts = []
    for o in OPENINGS:
        ids = tokenizer.encode(o["text"], bos=True)
        out = generate(model, torch.tensor([ids]), SAMPLING["max_new_tokens"], temperature=SAMPLING["temperature"],
                       seed=SAMPLING["seed"])
        texts.append(tokenizer.decode(out[0].tolist()[len(ids):]))
    return {"texts": texts, "sampling": SAMPLING, "openings": [o["text"] for o in OPENINGS]}


def score(ckpt: Path, device: str, batch_tokens: int, config: dict, docs: tuple, want: set[str]) -> dict:
    """The two held-out scores of one checkpoint."""
    import torch
    from gptpp.evaluate import bits_per_byte, load_model

    out = {}
    m = load_model(str(ckpt), device=device, precision="tf32")
    if "docs" in want:
        texts, source = docs
        r = bits_per_byte(m, texts, batch_tokens=batch_tokens)
        out["docs"] = {"bits_per_byte": r["bits_per_byte"], "se": r["bits_per_byte_se"], "loss": r["loss"],
                       "n_docs": r["n_docs"], "n_tokens": r["n_tokens"], "n_bytes": r["n_bytes"],
                       "sha256": source["sha256"], "precision": "tf32", "seconds": r["seconds"]}
    if "monitor" in want:
        from gptpp.data import TokenData, ValLoader
        from gptpp.train import evaluate

        tc, d = config["train_config"], config["derived"]
        data = TokenData(tc["data"])
        rows = max(1, batch_tokens // d["seq_len"])
        loader = ValLoader(data, d["seq_len"], rows, 0, 1, tc["val_tokens"], m.device)
        t0 = time.time()
        loss = evaluate(m.model, loader, torch.autocast("cuda", dtype=torch.bfloat16, enabled=m.device.type == "cuda"), 1)
        out["monitor"] = {"loss": loss, "bpb": loss / LN2 / config["data_meta"]["bytes_per_token_val"],
                          "windows": loader.n_windows, "seconds": round(time.time() - t0, 2)}
    del m
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return out


def released_on_monitor_text(spec: str, config: dict, device: str, batch_tokens: int) -> dict:
    """A released model on the text of the trainer's held-out windows, scored the way the trainer scores ours.

    The trainer's monitor is the mean loss over the targets of the first `val_windows` windows of val.bin, and
    its bits per byte divide the total bits by (targets x bytes_per_token_val). Here the same stretch of text
    (the documents those windows cover, the last one cut where the windows end) is encoded by the released
    model's own tokenizer, each document after its beginning-of-text token, as one stream; the stream is cut into
    windows of the model's context, each beginning with the last token of the one before, so that every token
    after the first is predicted once, from the tokens of its own window, document boundaries included. The
    total bits are divided by the same number of bytes as the trainer's: the two numbers then say how many bits
    each model needs for the same text, counted the same way."""
    import numpy as np
    from gptpp.data import TokenData
    from gptpp.evaluate import load_model, score_sequences
    from gptpp.generate import load_tokenizer

    tc, d, meta = config["train_config"], config["derived"], config["data_meta"]
    n_targets = d["val_windows"] * d["seq_len"]
    ours = np.asarray(TokenData(tc["data"]).val_tokens()[: n_targets + 1]).tolist()   # every input and the last target
    tokenizer, bos = load_tokenizer(meta["tokenizer"]), int(meta["bos_id"])
    if ours[0] != bos:
        raise SystemExit("val.bin does not begin with a document")
    docs: list[list[int]] = []
    for t in ours:
        if t == bos:
            docs.append([])
        else:
            docs[-1].append(t)
    texts = [tokenizer.decode(doc) for doc in docs]
    m = load_model(spec, device=device, precision="tf32")
    stream: list[int] = []
    for ids in m.encode(texts):
        stream.append(m.bos_id)
        stream.extend(ids)
    step = m.window - 1
    windows = [stream[i:i + m.window] for i in range(0, len(stream) - 1, step)]
    t0 = time.time()
    scores = score_sequences(m, windows, starts=1, batch_tokens=batch_tokens)
    n_scored = int(scores.n_scored.sum())
    assert n_scored == len(stream) - 1, "every token after the first is scored exactly once"
    bits = float(scores.nll.sum()) / LN2
    nominal = n_targets * meta["bytes_per_token_val"]
    actual = sum(len(t.encode("utf-8")) for t in texts)
    return {"model": m.info["name"], "revision": m.info.get("revision"), "bits": bits, "tokens": n_scored,
            "loss": bits * LN2 / n_scored, "bits_per_byte": bits / nominal, "bits_per_byte_of_its_text": bits / actual,
            "bytes_nominal": nominal, "bytes_text": actual, "n_docs": len(texts), "windows": len(windows),
            "window": m.window, "our_targets": n_targets, "precision": "tf32", "seconds": round(time.time() - t0, 2)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="gptpp-base")
    ap.add_argument("--ckpt-root", default=str(CKPT))
    ap.add_argument("--out", default=str(DATA))
    ap.add_argument("--name", default="training-run-replayed")
    ap.add_argument("--side", default=None, help="the side file (default $GPTPP_WORK/data/f7/<name>.<run>.json)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--score", action="store_true", help="also score the checkpoints that have no scores yet")
    ap.add_argument("--score-budget", type=float, default=420.0, help="seconds; stop scoring after this")
    ap.add_argument("--score-batch-tokens", type=int, default=4096)
    ap.add_argument("--score-only", default="monitor,docs", help="which scores: monitor, docs")
    ap.add_argument("--score-order", default="spread", choices=["spread", "up", "down"],
                    help="which checkpoints first when the budget may not reach all")
    ap.add_argument("--score-min-step", type=int, default=0, help="score only checkpoints from this step on")
    ap.add_argument("--score-max-step", type=int, default=None, help="score only checkpoints up to this step")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--min-free-gb", type=float, default=3.0)
    args = ap.parse_args()

    run_dir = Path(args.ckpt_root) / args.run
    config = json.loads((run_dir / "config.json").read_text())
    summary_file = run_dir / "summary.json"
    summary = json.loads(summary_file.read_text()) if summary_file.exists() else None
    log = read_log(run_dir / "log.jsonl")
    d, mc, tc, meta = config["derived"], config["model_config"], config["train_config"], config["data_meta"]
    bpt_val, bpt_train = meta["bytes_per_token_val"], meta["bytes_per_token_train"]

    side_file = Path(args.side) if args.side else WORK / "data" / "f7" / f"{args.name}.{args.run}.json"
    side_file.parent.mkdir(parents=True, exist_ok=True)
    side = json.loads(side_file.read_text()) if side_file.exists() else {}
    for k in ("samples", "monitor", "docs"):
        side.setdefault(k, {})
    gpt2_file = REPO / "results" / "eval" / "gpt2.json"
    gpt2_eval = json.loads(gpt2_file.read_text()) if gpt2_file.exists() else None
    gpt2_spec = f"hf:{gpt2_eval['model']['hub_id']}@{gpt2_eval['model']['revision']}" if gpt2_eval else None

    def save_side() -> None:
        """Merged with what another call may have written meanwhile, so that two calls at once lose nothing."""
        if side_file.exists():
            other = json.loads(side_file.read_text())
            for k in ("samples", "monitor", "docs"):
                for step, r in other.get(k, {}).items():
                    side[k].setdefault(step, r)
            if "gpt2_monitor" in other:
                side.setdefault("gpt2_monitor", other["gpt2_monitor"])
        write_atomic(side_file, json.dumps(side, ensure_ascii=False).encode("utf-8"))

    present = [s for s in d["checkpoint_steps"] if (run_dir / f"ckpt_{s}.pt").exists()]

    # ---- writing: every checkpoint that has none, or whose openings or sampling differ from today's
    todo = [s for s in present if not (side["samples"].get(str(s)) and side["samples"][str(s)].get("sampling") == SAMPLING
                                         and side["samples"][str(s)].get("openings") == [o["text"] for o in OPENINGS])]
    if todo:
        from gptpp.generate import load_tokenizer

        tokenizer = load_tokenizer(meta["tokenizer"])
        if tokenizer is None:
            raise SystemExit(f"the tokenizer {meta['tokenizer']!r} could not be loaded")
        t0 = time.time()
        for s in todo:
            side["samples"][str(s)] = write_samples(run_dir / f"ckpt_{s}.pt", tokenizer, args.threads)
            save_side()
        print(f"wrote from {len(OPENINGS)} openings at {len(todo)} checkpoints in {time.time() - t0:.0f} s")
    opening_tokens = side.get("opening_tokens")
    if opening_tokens is None or len(opening_tokens) != len(OPENINGS):
        from gptpp.generate import load_tokenizer

        tokenizer = load_tokenizer(meta["tokenizer"])
        opening_tokens = [len(tokenizer.encode(o["text"], bos=False)) for o in OPENINGS]
        side["opening_tokens"] = opening_tokens
        save_side()

    # ---- scores
    if args.score:
        import torch

        want_all = {w.strip() for w in args.score_only.split(",") if w.strip()}
        todo = [s for s in present if any(str(s) not in side[w] for w in want_all) and s >= args.score_min_step
                and (args.score_max_step is None or s <= args.score_max_step)]
        if args.score_order == "down":
            todo = todo[::-1]
        elif args.score_order == "spread":       # the two ends first, then by halves: any prefix covers the run
            order, queue = [], [todo]
            while queue:
                part = queue.pop(0)
                if not part:
                    continue
                mid = len(part) // 2
                order.append(part[mid])
                queue += [part[:mid], part[mid + 1:]]
            ends = [s for s in (todo[:1] + todo[-1:])]
            todo = list(dict.fromkeys(ends + order))
        ok = args.device.startswith("cuda") and torch.cuda.is_available()
        if ok:
            free = torch.cuda.mem_get_info()[0] / 2**30
            if free < args.min_free_gb:
                print(f"not scoring: the card has {free:.1f} GB free, {args.min_free_gb} GB are needed")
                ok = False
        elif args.device.startswith("cuda"):
            print("not scoring: no card")
        else:
            ok = True
        if ok and (todo or ("gpt2_monitor" not in side and gpt2_spec and "monitor" in want_all)):
            from gptpp.evaluate import load_docs

            docs = load_docs("fineweb-edu") if "docs" in want_all and todo else None
            t0, n = time.time(), 0
            for s in todo:
                if time.time() - t0 > args.score_budget:
                    break
                want = {w for w in want_all if str(s) not in side[w]}
                res = score(run_dir / f"ckpt_{s}.pt", args.device, args.score_batch_tokens, config, docs, want)
                for w, r in res.items():
                    side[w][str(s)] = r
                save_side()
                n += 1
                print(f"scored step {s}: " + ", ".join(f"{w} {r.get('bpb', r.get('bits_per_byte')):.4f}" for w, r in res.items())
                      + f"  ({time.time() - t0:.0f} s)", flush=True)
            if "monitor" in want_all and "gpt2_monitor" not in side and gpt2_spec and time.time() - t0 < args.score_budget:
                side["gpt2_monitor"] = released_on_monitor_text(gpt2_spec, config, args.device, args.score_batch_tokens)
                save_side()
                print(f"scored GPT-2 small on the monitor's text: {side['gpt2_monitor']['bits_per_byte']:.4f} bits per byte")
            left = [s for s in present if any(str(s) not in side[w] for w in want_all) and s >= args.score_min_step
                    and (args.score_max_step is None or s <= args.score_max_step)]
            print(f"scored {n} checkpoints in {time.time() - t0:.0f} s; {len(left)} of {len(present)} still to score")
        elif ok:
            print("scored 0 checkpoints: 0 still to score")

    # ---- the curves of the log
    tr, va = log["train"], log["val"]
    train = {"step": [r["step"] for r in tr], "loss": [sig(r["loss"]) for r in tr],
             "lr_mult": [sig(r["lr_mult"], 5) for r in tr], "tok_s": [int(r["tok_s"]) for r in tr],
             "time": [round(r["time"], 1) for r in tr]}
    val = {"step": [r["step"] for r in va], "loss": [sig(r["val_loss"]) for r in va],
           "bpb": [sig(r["val_bpb"]) for r in va], "time": [round(r["time"], 1) for r in va]}
    tps = d["tokens_per_step"]
    t_steps, t_times = [0] + train["step"], [0.0] + train["time"]

    # ---- the levels
    levels = [{"key": "uniform", "label": "uniform", "bits_per_byte": sig(math.log2(mc["vocab_size"]) / bpt_val),
               "what": f"every one of the {mc['vocab_size']:,} tokens equally likely"}]
    stats = json.loads((REPO / "results" / "data" / tc["data"] / "stats.json").read_text())
    for o in stats["ngrams"]["orders"]:
        key, label = NGRAM_KEYS[o["n"]]
        levels.append({"key": key, "label": label, "bits_per_byte": sig(o["held_out_bits_per_byte"]),
                       "what": ("token frequencies alone" if o["n"] == 1 else f"an n-gram model that sees the {o['n'] - 1} token"
                                + ("s" if o["n"] > 2 else "") + " before") + "; " + stats["ngrams"]["smoothing"].split(",")[0]
                       + f", counted on {stats['ngrams']['sample_tokens']:,} training tokens"})
    if gpt2_eval:
        g, e, gm = gpt2_eval, gpt2_eval["bits_per_byte"]["fineweb-edu"], side.get("gpt2_monitor")
        gpt2 = {"key": "gpt2", "label": g["model"].get("label", "GPT-2 small"), "n_params": g["model"]["n_params"],
                "docs": {"bits_per_byte": sig(e["bits_per_byte"]), "se": sig(e["bits_per_byte_se"], 3), "loss": sig(e["loss"]),
                         "n_docs": e["n_docs"], "sha256": e["docs"]["sha256"]}}
        gpt2.update(bits_per_byte=sig(e["bits_per_byte"]), protocol="documents", n_docs=e["n_docs"],
                    se=gpt2["docs"]["se"], sha256=gpt2["docs"]["sha256"],
                    what="the released GPT-2 small, scored document by document on the first held-out documents")
        gpt2["monitor"] = None if not gm else {
            "bits_per_byte": sig(gm["bits_per_byte"]), "loss": sig(gm["loss"]), "tokens": gm["tokens"], "n_docs": gm["n_docs"],
            "bits_per_byte_of_its_text": sig(gm["bits_per_byte_of_its_text"]),
            "what": "the released GPT-2 small on the text of the trainer's held-out windows, read the way the trainer "
                    "reads: its bits over the same bytes"}
        levels.append(gpt2)

    # ---- the checkpoints
    checkpoints = []
    for s in present:
        smp, mon, doc = side["samples"].get(str(s)), side["monitor"].get(str(s)), side["docs"].get(str(s))
        checkpoints.append({
            "step": s, "tokens": s * tps, "time": round(interp(s, t_steps, t_times), 1), "file": f"ckpt_{s}.pt",
            "texts": smp["texts"] if smp else None,
            "monitor": {"loss": sig(mon["loss"]), "bpb": sig(mon["bpb"])} if mon else None,
            "docs": {"bits_per_byte": sig(doc["bits_per_byte"]), "se": sig(doc["se"], 3), "loss": sig(doc["loss"]),
                     "n_docs": doc["n_docs"], "n_tokens": doc["n_tokens"], "sha256": doc["sha256"]} if doc else None})

    # ---- the held-out curve: the log's evaluations and the checkpoints' monitor, one protocol
    by_step = {c["step"]: (c["monitor"]["loss"], c["monitor"]["bpb"], "checkpoint") for c in checkpoints if c["monitor"]}
    checks = [abs(by_step[s][0] - l) for s, l in zip(val["step"], val["loss"]) if s in by_step]
    for s, l, b in zip(val["step"], val["loss"], val["bpb"]):
        by_step[s] = (l, b, "log")
    steps = sorted(by_step)
    held = {"step": steps, "tokens": [s * tps for s in steps], "loss": [by_step[s][0] for s in steps],
            "bpb": [by_step[s][1] for s in steps], "source": [by_step[s][2] for s in steps]}

    # ---- the numbers the post quotes
    last = tr[-1] if tr else {"step": 0, "tokens": 0, "time": 0.0}
    final_ckpt = next((c for c in checkpoints if c["step"] == d["steps"]), None)
    gaps = [abs(c["docs"]["bits_per_byte"] - c["monitor"]["bpb"]) for c in checkpoints
            if c["docs"] and c["monitor"] and c["step"] >= 100]
    numbers = {
        "tokens": summary["tokens"] if summary else None, "steps": summary["steps"] if summary else None,
        "seconds": summary["seconds"] if summary else None, "hours": sig(summary["seconds"] / 3600, 4) if summary else None,
        "tokens_per_second": summary["tokens_per_second"] if summary else None,
        "final_val_loss": sig(summary["val_loss"]) if summary else None, "final_val_bpb": sig(summary["val_bpb"]) if summary else None,
        "best_val_loss": sig(summary["best_val_loss"]) if summary else None,
        "best_val_bpb": sig(summary["best_val_bpb"]) if summary else None, "best_step": summary["best_step"] if summary else None,
        "first_val_bpb": val["bpb"][0] if val["bpb"] else None,
        "final_docs_bpb": final_ckpt["docs"]["bits_per_byte"] if final_ckpt and final_ckpt["docs"] else None,
        "final_docs_se": final_ckpt["docs"]["se"] if final_ckpt and final_ckpt["docs"] else None,
        "final_docs_loss": final_ckpt["docs"]["loss"] if final_ckpt and final_ckpt["docs"] else None,
        "monitor_check": sig(max(checks), 3) if checks else None, "protocol_gap": sig(max(gaps), 3) if gaps else None,
    }
    for lv in levels:
        numbers["level_" + lv["key"]] = lv["bits_per_byte"]
    crossed = {}
    for lv in levels:
        if lv["key"] == "uniform":
            continue
        if lv.get("protocol") == "documents":
            pts = [(c["tokens"], c["docs"]["bits_per_byte"]) for c in checkpoints if c["docs"]]
            protocol = "documents"
        else:
            pts = list(zip(held["tokens"], held["bpb"]))
            protocol = "monitor"
        at = crossing([p[0] for p in pts], [p[1] for p in pts], lv["bits_per_byte"]) if pts else None
        crossed[lv["key"]] = None if at is None else {
            "tokens": round(at["tokens"]), "step": round(at["tokens"] / tps),
            "hours": sig(interp(at["tokens"] / tps, t_steps, t_times) / 3600, 4), "between": [round(x) for x in at["between"]],
            "sure": at["sure"], "protocol": protocol}
    numbers["crossed"] = crossed

    def where(at: dict | None, protocol: str) -> dict | None:
        return None if at is None else {
            "tokens": round(at["tokens"]), "step": round(at["tokens"] / tps),
            "hours": sig(interp(at["tokens"] / tps, t_steps, t_times) / 3600, 4), "between": [round(x) for x in at["between"]],
            "sure": at["sure"], "protocol": protocol}

    def reading(i: int) -> dict:
        return {"step": val["step"][i], "tokens": val["step"][i] * tps, "val_loss": val["loss"][i], "val_bpb": val["bpb"][i],
                "hours": sig(val["time"][i] / 3600, 4)}

    gl = next((lv for lv in levels if lv["key"] == "gpt2"), None)
    gm = gl["monitor"] if gl else None
    numbers["level_gpt2_monitor"] = gm["bits_per_byte"] if gm else None
    numbers["crossed_gpt2_monitor"] = where(crossing(held["tokens"], held["bpb"], gm["bits_per_byte"]), "monitor") if gm else None
    # the readings the post quotes, each from the trainer's own evaluations
    post = {"first_reading": reading(1) if len(val["step"]) > 1 else None, "last_reading": reading(len(val["step"]) - 1) if val["step"] else None}
    near = lambda tokens: min(range(len(val["step"])), key=lambda i: abs(val["step"][i] * tps - tokens)) if val["step"] else None
    post["reading_at_1100M"] = reading(near(1.1e9)) if val["step"] and val["step"][-1] * tps >= 1.075e9 else None
    under = next((i for i, b in enumerate(val["bpb"]) if gl and b < gl["bits_per_byte"]), None)
    post["first_reading_under_gpt2"] = None if under is None else {**reading(under), "level": gl["bits_per_byte"], "protocol_of_level": "documents"}
    decay = (1.0 - tc["cooldown_frac"]) * d["tokens"] if tc["schedule"] == "trapezoid" else None
    post["decay_start_tokens"] = round(decay) if decay else None
    at_decay = near(decay) if decay and val["step"] and val["step"][-1] * tps >= decay else None
    post["reading_at_decay_start"] = reading(at_decay) if at_decay is not None else None
    post["fall_after_decay_share"] = post["last_500M"] = None
    if summary and at_decay is not None and len(val["bpb"]) > 2:
        first, at, end = val["bpb"][1], val["bpb"][at_decay], val["bpb"][-1]
        post["fall_after_decay_share"] = sig((at - end) / (first - end), 4)       # of the fall since the first reading
        back = near(d["tokens"] - 5e8)
        span = (val["step"][-1] - val["step"][back]) * tps
        post["last_500M"] = {"from": reading(back), "to": reading(len(val["step"]) - 1), "fall_bpb": sig(val["bpb"][back] - end, 4),
                             "fall_loss": sig(val["loss"][back] - val["loss"][-1], 4),
                             "bpb_per_100M_tokens": sig((end - val["bpb"][back]) / span * 1e8, 4)}
    numbers["post"] = post

    out = {
        "generated_by": "gptpp.export.training_run_replayed", "run": args.run, "complete": summary is not None,
        "model": {"n_layer": mc["n_layer"], "d_model": mc["d_model"], "n_head": mc["n_head"], "block_size": mc["block_size"],
                  "vocab_size": mc["vocab_size"], "params": config["params"]},
        "data": {"dataset": meta["dataset"], "tokenizer": meta["tokenizer"], "bytes_per_token_val": bpt_val,
                 "bytes_per_token_train": bpt_train, "val_tokens": int(tc["val_tokens"]), "val_windows": d["val_windows"]},
        "plan": {"steps": d["steps"], "tokens": d["tokens"], "tokens_per_step": tps, "n_gpu": d["n_gpu"],
                 "device_batch": tc["device_batch"], "seq_len": d["seq_len"], "eval_every_steps": d["eval_every_steps"],
                 "checkpoint_steps": d["checkpoint_steps"]},
        "schedule": {"optimizer": tc["optimizer"], "schedule": tc["schedule"], "lr": tc["lr"], "lr_embedding": tc["lr_embedding"],
                     "lr_unembedding": tc["lr_unembedding"], "warmup_tokens": tc["warmup_tokens"],
                     "cooldown_frac": tc["cooldown_frac"], "final_lr_frac": tc["final_lr_frac"]},
        "progress": {"step": last["step"], "tokens": last["tokens"], "seconds": round(last["time"], 1)},
        "summary": None if summary is None else {**{k: v for k, v in summary.items() if k != "params"},
                                                 "hours": sig(summary["seconds"] / 3600, 4),
                                                 "train_hours": sig(summary["train_seconds"] / 3600, 4)},
        "train": train, "val": val, "levels": levels,
        "openings": [{**o, "tokens": n} for o, n in zip(OPENINGS, opening_tokens)],
        "sampling": SAMPLING, "checkpoints": checkpoints, "checkpoints_planned": len(d["checkpoint_steps"]),
        "held_out": held, "numbers": numbers,
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(out, separators=(",", ":"), ensure_ascii=False)
    write_atomic(out_dir / f"{args.name}.json", text.encode("utf-8"))
    scored = sum(1 for c in checkpoints if c["monitor"]), sum(1 for c in checkpoints if c["docs"])
    print(f"{args.name}.json: {len(text.encode('utf-8')) / 1e3:.1f} kB; {'complete' if summary else 'in progress'}: step "
          f"{last['step']} of {d['steps']}; {len(tr)} training records, {len(va)} evaluations, {len(checkpoints)} of "
          f"{len(d['checkpoint_steps'])} checkpoints (monitor at {scored[0]}, documents at {scored[1]})")


if __name__ == "__main__":
    main()
