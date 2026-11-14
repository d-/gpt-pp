"""Loss by token frequency under each optimizer: `rare-tokens-under-sgd.json` for fig-06-05.

    python -m gptpp.export.rare_tokens [--out <dir>]

Reads what `gptpp.interp.frequency_buckets` measured: pico trained on 100M tokens of TinyStories by SGD with
momentum (plain, and with the gradient norm clipped to 1), by sign descent with momentum and by Adam, each at
several learning rates, and at every saved checkpoint the held-out loss of ten groups of target tokens: the
vocabulary sorted by its frequency in the training set and cut into ten buckets that hold about a tenth of the
training tokens each.

Every run found is exported (results/runs/freq-*/frequency_buckets.json, or the same file in the workspace): its
rate, its status, and for each checkpoint the step, the loss over all held-out tokens and the loss of each
bucket. For each optimizer the best run is the one with the lowest held-out loss at the end, and `interior`
says whether that rate has a neighbor on both sides in the grid of rates tried (if not, the grid should be
extended before the optimizer is judged). A run that stopped on a loss that is not finite is kept as a result,
with the checkpoints it reached. `missing` lists the runs of the helper's table that have not been scored.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .. import repo_dir
from ..interp.frequency_buckets import jobs
from .common import CKPT, DATA, write_json

NAME = "rare-tokens-under-sgd"
OPTIMIZERS = [
    {"key": "sgdraw", "label": "SGD with momentum", "short": "SGD",
     "how": "Nesterov momentum 0.9, one rate for every parameter, the gradient as it is"},
    {"key": "sgd", "label": "SGD with momentum, clipped", "short": "clipped SGD",
     "how": "the same, with the norm of the gradient clipped to 1: where the clip binds, the step has a fixed length"},
    {"key": "sign", "label": "sign descent with momentum", "short": "sign descent",
     "how": "every coordinate moves by the rate, in the direction of the sign of its averaged gradient (momentum 0.9)"},
    {"key": "adam", "label": "Adam", "short": "Adam",
     "how": "betas (0.9, 0.95), one rate for every parameter, the norm of the gradient clipped to 1"},
]


def clean(token: str) -> str | None:
    """A token as the figure prints it: without its leading space, a line break as a mark; None for a token that is
    not text (the document-start mark)."""
    if re.fullmatch(r"<\|.*\|>", token):
        return None
    text = token.strip(" ").replace("\n", "↵")
    return text or None


def find_runs() -> dict[str, dict]:
    runs = {}
    for root in (repo_dir() / "results" / "runs", CKPT):  # the workspace's copy is the fresher one
        for file in sorted(root.glob("freq-*/frequency_buckets.json")):
            runs[file.parent.name] = json.loads(file.read_text())
    return runs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(DATA))
    args = ap.parse_args()
    runs = find_runs()
    if not runs:
        raise SystemExit("no scored run: run gptpp.interp.frequency_buckets train and score first")
    first = next(iter(runs.values()))
    buckets = first["buckets"]
    config = json.loads((CKPT / first["run"] / "config.json").read_text()) if (CKPT / first["run"] / "config.json").exists() \
        else json.loads((repo_dir() / "results" / "runs" / first["run"] / "config.json").read_text())
    optimizers = []
    for spec in OPTIMIZERS:
        mine = sorted((r for r in runs.values() if r["optimizer"] == spec["key"]), key=lambda r: r["lr"])
        entries = []
        for r in mine:
            rows = r["checkpoints"]
            entries.append({
                "run": r["run"], "lr": r["lr"], "status": r["status"], "val_loss": r["val_loss"],
                "steps": [c["step"] for c in rows], "loss": [c["loss"] for c in rows],
                "bucket_loss": [c["bucket_loss"] for c in rows],
            })
        done = [k for k, e in enumerate(entries) if e["status"] == "completed"]
        best = min(done, key=lambda k: entries[k]["val_loss"]) if done else None
        optimizers.append({**spec, "runs": entries, "best": best,
                           "interior": best is not None and 0 < best < len(entries) - 1})
    examples = []
    for tokens in buckets["examples"]:
        shown = [t for t in (clean(token) for token in tokens) if t]
        examples.append(shown[:4])
    out = {
        "model": {"preset": first["preset"], "layers": config["model_config"]["n_layer"], "width": config["model_config"]["d_model"],
                  "params": config["params"]["total"]},
        "data": first["data"], "tokens": first["tokens"], "steps": first["steps"],
        "tokens_per_step": config["derived"]["tokens_per_step"], "vocab_size": first["vocab_size"],
        "uniform_loss": first["uniform_loss"], "held_out_tokens": first["held_out_tokens"],
        "schedule": {k: config["train_config"][k] for k in ("schedule", "warmup_tokens", "final_lr_frac")},
        "buckets": {"n": len(buckets["tokens"]), "tokens": buckets["tokens"], "mass": buckets["mass"],
                    "top_share": buckets["top_share"], "least_share": buckets["least_share"],
                    "held_out_tokens": buckets["held_out_tokens"], "unigram_loss": buckets["unigram_loss"], "examples": examples},
        "optimizers": optimizers, "missing": sorted(set(jobs()) - set(runs)),
        "note": "bucket 0 holds the most frequent tokens and the last bucket the rarest; bucket_loss[c][b] is the mean "
                "held-out loss in nats of the tokens of bucket b at checkpoint c, loss[c] the mean over all held-out tokens; "
                "a step is tokens_per_step tokens",
    }
    size = write_json(Path(args.out) / f"{NAME}.json", out)
    print(f"{NAME}.json: {size / 1e3:.1f} kB")
    for o in optimizers:
        cells = [f"{e['lr']:g}: {e['val_loss']:.4f}" if e["val_loss"] is not None else f"{e['lr']:g}: {e['status']}" for e in o["runs"]]
        best = o["runs"][o["best"]] if o["best"] is not None else None
        print(f"  {o['label']:<28} {' | '.join(cells) or 'no run'}")
        if best:
            last = best["bucket_loss"][-1]
            print(f"  {'':<28} best {best['lr']:g}{'' if o['interior'] else ' (at an end of its grid)'}: most frequent bucket "
                  f"{last[0]:.3f}, rarest {last[-1]:.3f}")
    if out["missing"]:
        print("  not scored yet:", ", ".join(out["missing"]))


if __name__ == "__main__":
    main()
