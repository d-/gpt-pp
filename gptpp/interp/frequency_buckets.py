"""Loss by token frequency: which tokens each optimizer learns.

Words follow Zipf's law: a handful of tokens make up most of a text and thousands of others share the rest. The
rows of the embedding and of the unembedding inherit that imbalance. A row's gradient is a sum over the places
where its token occurs, so the gradient of a frequent token is orders of magnitude larger than that of a rare
one. Gradient descent takes one step size for all coordinates; the largest it can use is set by the frequent
tokens, and at that size the rare tokens hardly move. Adam divides each coordinate by a running size of its own
gradient, and sign descent keeps only the sign: both move every row at about the same speed. This is the
account of Kunstner et al. (2024, "Heavy-tailed class imbalance and why Adam outperforms gradient descent on
language models", arXiv:2402.19449) of why SGD fails on language, and this file measures it on pico:

    train    pico (preset gptpp-pico, TinyStories) under SGD with momentum (plain, and with the gradient
             clipped), under Adam and under sign descent with momentum, each at several learning rates, each
             with checkpoints at log-spaced and at evenly spaced steps (the lab's trainer does the training)
    score    every checkpoint on the trainer's fixed held-out tokens, the loss averaged separately over ten
             groups of target tokens: the vocabulary is sorted by its frequency in the training set and cut into
             ten buckets that each hold about a tenth of the training tokens, so bucket 1 is the two or three most
             frequent tokens and bucket 10 the thousands of rarest ones

    CUDA_VISIBLE_DEVICES=3 python -m gptpp.interp.frequency_buckets train [--only NAME ...] [--tokens 100e6]
    CUDA_VISIBLE_DEVICES=3 python -m gptpp.interp.frequency_buckets score [--only NAME ...] [--final-only]
    python -m gptpp.interp.frequency_buckets list

The optimizers, all with one learning rate for every parameter, warmup over 4M tokens, cosine decay to a tenth,
no weight decay (the settings of the ladder's optimizer sweep, at pico's size), 65,536 tokens a step:

    sgdraw  torch.optim.SGD, Nesterov momentum 0.9, on the gradient as it is: plain SGD with momentum
    sgd     the same with the gradient norm clipped to 1, as the ladder's sweep runs it. Where the clip binds
            the step has a fixed length whatever the gradient, which is normalized gradient descent and no
            longer plain SGD; on this model the norm stays under 1 at the rates that train (the two variants
            end within 0.002 nats of each other at rates 0.1 and 0.3)
    adam    torch.optim.Adam, betas (0.9, 0.95), gradient norm clipped to 1
    sign    sign descent with momentum: the lab's Lion with beta1 = beta2 = 0.9, which is
            W <- W - lr * sign(M), M an exponential moving average of the gradient. Every coordinate moves by
            exactly lr a step.

The rates of each are half a decade apart, and each grid was extended until the rate that ends lowest had a
neighbor on both sides. A run is named freq-<optimizer>-lr<rate> and lives where every run lives
($GPTPP_WORK/ckpt/<run>/, small copies in results/runs/<run>/). `score` adds frequency_buckets.json to both:

    {"run", "optimizer", "lr", "buckets": {"tokens": [...], "mass": [...], "held_out_tokens": [...],
     "unigram_loss": [...], "examples": [[...], ...]}, "checkpoints": [{"step", "tokens", "loss", "bucket_loss": [10]}]}

The loss is in nats per token; `unigram_loss` is what predicting every token by its training frequency would
cost on the bucket's held-out tokens, and log(vocabulary) is what a uniform guess costs on any of them.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from .. import repo_dir, work_dir

N_BUCKETS = 10
PRESET, DATA = "gptpp-pico", "tinystories/bpe4k"
COMMON = ["schedule=cosine", "warmup_tokens=4e6", "final_lr_frac=0.1", "weight_decay=0.0"]
# A step is pico's 256 sequences of 256 tokens, taken as four passes of 64: the same gradient in a quarter of the
# memory, because the card is shared with other jobs.
BATCH = ["device_batch=64", "grad_accum=4"]
OPTIMIZERS = {
    "sgd": ["optimizer=sgd", "momentum=0.9", "grad_clip=1.0"],
    "adam": ["optimizer=adam", "beta1=0.9", "grad_clip=1.0"],
    "sign": ["optimizer=lion", "beta1=0.9", "beta2=0.9"],
    "sgdraw": ["optimizer=sgd", "momentum=0.9"],  # the same SGD on the gradient as it is
}
# The rates of each optimizer: half a decade apart, extended until the best rate had a neighbor on both sides.
RATES = {
    "sgd": [0.03, 0.1, 0.3, 1.0, 3.0],
    "adam": [1e-3, 3e-3, 1e-2, 3e-2],
    "sign": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    "sgdraw": [0.1, 0.3, 1.0, 3.0],
}


def run_name(optimizer: str, rate: float, prefix: str = "freq") -> str:
    return f"{prefix}-{optimizer}-lr{rate:g}"


def jobs(prefix: str = "freq") -> dict[str, dict]:
    return {run_name(o, r, prefix): {"optimizer": o, "lr": r} for o in OPTIMIZERS for r in RATES[o]}


def overrides(optimizer: str, rate: float) -> list[str]:
    r = f"{rate:g}"
    return [f"lr={r}", f"lr_embedding={r}", f"lr_unembedding={r}", *OPTIMIZERS[optimizer], *COMMON, *BATCH]


def checkpoint_flags(tokens: float, tokens_per_step: int = 65536, n_log: int = 20, n_even: int = 10) -> list[str]:
    """Checkpoints at n_log log-spaced steps (the early part of training) and at n_even evenly spaced ones."""
    steps = max(1, round(tokens / tokens_per_step))
    even = sorted({round(steps * k / n_even) for k in range(1, n_even)})
    return ["--save-log", str(n_log), "--save-steps", ",".join(str(s) for s in even)]


def train(names: list[str], tokens: float, prefix: str, overwrite: bool, probe: bool = False) -> None:
    """Train the named runs one after another with the lab's trainer, on the card the environment names. A probe
    is a short run that only looks for the range of rates worth running in full: it keeps no checkpoints and
    leaves no copy in results/."""
    table = jobs(prefix)
    for name in names:
        job = table[name]
        if (work_dir() / "ckpt" / name / "summary.json").exists() and not overwrite:
            print(f"{name}: done already", flush=True)
            continue
        cmd = [sys.executable, "-m", "gptpp.train", "--preset", PRESET, "--data", DATA, "--run", name,
               "--tokens", f"{tokens:g}", "--eval-tokens", "10e6",
               *(["--results", "none"] if probe else checkpoint_flags(tokens)),
               "--set", *overrides(job["optimizer"], job["lr"])]
        if overwrite:
            cmd.append("--overwrite")
        print(" ".join(cmd), flush=True)
        done = subprocess.run(cmd, cwd=repo_dir(), capture_output=True, text=True)
        tail = (done.stdout or "").strip().splitlines()[-1:] or [""]
        if done.returncode != 0:  # a run whose loss is not finite stops with an error record in its log: a result
            error = (done.stderr or "").strip().splitlines()[-1:] or [""]
            print(f"{name}: stopped ({error[0][:200]})", flush=True)
        else:
            print(f"{name}: {tail[0]}", flush=True)


# ----------------------------------------------------------------------------------------------------------------
# The buckets


def token_counts(data) -> np.ndarray:
    """How often each token id occurs in the training set (every shard, read once)."""
    counts = np.zeros(data.vocab_size, dtype=np.int64)
    for i in range(len(data.train_files)):
        counts += np.bincount(data.shard(i), minlength=data.vocab_size)
    return counts


def frequency_buckets(counts: np.ndarray, n: int = N_BUCKETS) -> np.ndarray:
    """The bucket of every token id, 0 for the most frequent tokens to n - 1 for the rarest: the ids are sorted
    by falling count and the running share of the training set is cut into n equal parts. A token belongs to
    the part that holds the middle of its own share, so the buckets hold about, not exactly, a tenth each (the
    most frequent token alone is 6.5% of TinyStories). Ties in count keep the order of the ids."""
    order = np.argsort(-counts, kind="stable")
    share = counts[order] / counts.sum()
    middle = np.cumsum(share) - share / 2
    bucket = np.empty(len(counts), dtype=np.int64)
    bucket[order] = np.minimum((middle * n).astype(np.int64), n - 1)
    return bucket


def bucket_table(counts: np.ndarray, bucket: np.ndarray, tokenizer=None, n_examples: int = 6) -> dict:
    """What each bucket holds: how many distinct tokens, their share of the training set, the share of its most
    and least frequent token, and a few of its tokens as text (the most frequent ones of the bucket)."""
    n, total = int(bucket.max()) + 1, counts.sum()
    order = np.argsort(-counts, kind="stable")
    table = {"tokens": [], "mass": [], "top_share": [], "least_share": [], "examples": [], "example_ids": []}
    for b in range(n):
        ids = order[bucket[order] == b]
        table["tokens"].append(int(len(ids)))
        table["mass"].append(float(counts[ids].sum() / total))
        table["top_share"].append(float(counts[ids[0]] / total))
        table["least_share"].append(float(counts[ids[-1]] / total))
        shown = [int(i) for i in ids[:n_examples]]
        table["example_ids"].append(shown)
        table["examples"].append([tokenizer.decode([i]) for i in shown] if tokenizer is not None else [])
    return table


# ----------------------------------------------------------------------------------------------------------------
# Scoring


def score_checkpoint(model, batches, bucket, autocast) -> tuple[np.ndarray, np.ndarray]:
    """Sum of the held-out token losses and number of tokens, by the bucket of the target token."""
    import torch
    import torch.nn.functional as F

    n = int(bucket.max().item()) + 1
    sums = torch.zeros(n, dtype=torch.float64, device=bucket.device)
    counts = torch.zeros(n, dtype=torch.float64, device=bucket.device)
    with torch.no_grad():
        for x, y, _ in batches:
            with autocast:
                logits = model(x)
            flat = y.reshape(-1)
            keep = flat >= 0
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), flat.clamp(min=0), reduction="none")
            which = bucket[flat[keep]]
            sums.index_add_(0, which, loss[keep].double())
            counts.index_add_(0, which, torch.ones_like(which, dtype=torch.float64))
    return sums.cpu().numpy(), counts.cpu().numpy()


def score(names: list[str], prefix: str, final_only: bool = False) -> None:
    """Score every checkpoint of the named runs (with final_only: the end of each run alone, which is all that
    is needed of a rate far from the best one) and write frequency_buckets.json beside each run."""
    import torch

    from ..data import TokenData, ValLoader
    from ..generate import load_tokenizer
    from ..model import GPT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")
    data = TokenData(DATA)
    counts = token_counts(data)
    bucket_np = frequency_buckets(counts)
    tokenizer = load_tokenizer(data.meta["tokenizer"]) if data.meta.get("tokenizer") else None
    table = bucket_table(counts, bucket_np, tokenizer)
    bucket = torch.from_numpy(bucket_np).to(device)
    loader = None
    job_table = jobs(prefix)
    for name in names:
        run_dir = work_dir() / "ckpt" / name
        config_file = run_dir / "config.json"
        if not config_file.exists():
            print(f"{name}: no run", flush=True)
            continue
        config = json.loads(config_file.read_text())
        seq_len, tokens_per_step = config["derived"]["seq_len"], config["derived"]["tokens_per_step"]
        if loader is None:  # the trainer's held-out windows: the first val_tokens of val.bin
            loader = ValLoader(data, seq_len, 64, 0, 1, config["train_config"]["val_tokens"], device)
            held_out = torch.cat([y.reshape(-1) for _, y, _ in loader])
            held_out = held_out[held_out >= 0].cpu().numpy()
            share = np.maximum(counts, 1) / counts.sum()
            table["held_out_tokens"] = [int((bucket_np[held_out] == b).sum()) for b in range(N_BUCKETS)]
            table["unigram_loss"] = [float(-np.log(share[held_out[bucket_np[held_out] == b]]).mean())
                                     for b in range(N_BUCKETS)]
        files = {int(re.fullmatch(r"ckpt_(\d+)\.pt", p.name).group(1)): p for p in run_dir.glob("ckpt_*.pt")}
        if (run_dir / "model.pt").exists():
            files[config["derived"]["steps"]] = run_dir / "model.pt"
        if final_only:
            files = {step: path for step, path in files.items() if step == max(files)}
        rows = []
        for step in sorted(files):
            model, _ = GPT.from_checkpoint(files[step], device)
            sums, n = score_checkpoint(model, loader, bucket, autocast)
            rows.append({"step": step, "tokens": step * tokens_per_step, "loss": float(sums.sum() / n.sum()),
                         "bucket_loss": [float(v) for v in sums / n]})
        summary = run_dir / "summary.json"
        out = {
            "run": name, "optimizer": job_table.get(name, {}).get("optimizer"), "lr": job_table.get(name, {}).get("lr"),
            "preset": config["preset"], "data": config["train_config"]["data"], "overrides": config["overrides"],
            "status": "completed" if summary.exists() else "stopped",
            "val_loss": json.loads(summary.read_text())["val_loss"] if summary.exists() else None,
            "tokens": config["derived"]["tokens"], "steps": config["derived"]["steps"],
            "vocab_size": data.vocab_size, "uniform_loss": math.log(data.vocab_size),
            "held_out_tokens": int(sum(table["held_out_tokens"])), "buckets": table, "checkpoints": rows,
        }
        text = json.dumps(out, indent=1) + "\n"
        (run_dir / "frequency_buckets.json").write_text(text)
        results = repo_dir() / "results" / "runs" / name
        if results.is_dir() or summary.exists():
            results.mkdir(parents=True, exist_ok=True)
            (results / "frequency_buckets.json").write_text(text)
        last = rows[-1] if rows else None
        if last:
            print(f"{name}: {len(rows)} checkpoints; at step {last['step']} loss {last['loss']:.4f}, most frequent "
                  f"bucket {last['bucket_loss'][0]:.3f}, rarest {last['bucket_loss'][-1]:.3f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["train", "score", "list"])
    parser.add_argument("--only", nargs="*", default=None, help="run names (default: every run of the table)")
    parser.add_argument("--tokens", type=float, default=100e6, help="training budget of each run")
    parser.add_argument("--prefix", default="freq", help="first part of the run names")
    parser.add_argument("--rates", default=None,
                        help="other rates, e.g. 'sgd=0.01,0.03;adam=3e-4' (replaces the table for those optimizers)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--probe", action="store_true", help="train: short runs without checkpoints or results")
    parser.add_argument("--final-only", action="store_true", help="score: only the last checkpoint of each run")
    args = parser.parse_args()
    if args.rates:
        for part in args.rates.split(";"):
            optimizer, _, values = part.partition("=")
            RATES[optimizer.strip()] = [float(v) for v in values.split(",") if v.strip()]
    names = args.only if args.only else list(jobs(args.prefix))
    if args.action == "list":
        for name in names:
            job = jobs(args.prefix)[name]
            print(name, " ".join(overrides(job["optimizer"], job["lr"])))
    elif args.action == "train":
        train(names, args.tokens, args.prefix, args.overwrite, args.probe)
    else:
        score(names, args.prefix, args.final_only)


if __name__ == "__main__":
    main()
