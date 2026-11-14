"""The ladder model under five optimizers, three learning rates each: `optimizers-on-the-ladder.json` for fig-06-04.

    python -m gptpp.export.optimizer_sweep [--out <dir>] [--runs name=run ...]

The runs are those of experiments/small.jsonl: the same model (preset gptpp-ladder: 8 layers, 512 wide), the same
50M tokens of FineWeb-EDU in the same order, the same seed, and

    sgd     opt-sgd-lr0.03 | 0.1 | 0.3          SGD, Nesterov momentum 0.9, gradient norm clipped to 1
    adam    opt-adam-lr1.2e-3 | 2.4e-3 | 4.8e-3  Adam, betas (0.9, 0.95), clipped to 1
    adamw   opt-adamw-lr...                     AdamW, the same with decoupled weight decay 0.1 on the hidden matrices
    lion    opt-lion-lr1e-4 | 3e-4 | 1e-3        Lion, betas (0.9, 0.99), weight decay 0.1
    muon    opt-muon-lr0.01 | swap-base (0.02) | opt-muon-lr0.04
                                                the 2026 recipe: Muon on the hidden matrices, AdamW with its own
                                                rates on the embedding and the unembedding, and the recipe's own
                                                schedule (2M tokens of warmup, flat, linear decay over the last 65%)

The first four take one rate for every parameter, 4M tokens of warmup and a cosine decay to a tenth. The exporter
uses whatever has finished and says what is missing: run it again when the queue is done. A run that stopped on
a loss that is not finite is kept, with its curve up to there, as a result. If swap-base has not finished, a
run of exactly the same configuration made earlier (core-probe-final) stands in for it and is marked.

What is written, for every run: its rate, its status, the loss on the fixed held-out set at each evaluation
(`val`), and the training loss at each log record (`train`: the mean over ten steps of the loss on tokens not
yet trained on; `steps` are the steps at which the records were written).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import preset, train_preset
from .common import DATA, write_json
from .run_curves import read_run

NAME = "optimizers-on-the-ladder"
SWEEP = [
    {"key": "sgd", "label": "SGD with momentum", "short": "SGD",
     "runs": [["opt-sgd-lr0.03"], ["opt-sgd-lr0.1"], ["opt-sgd-lr0.3"]]},
    {"key": "lion", "label": "Lion", "short": "Lion",
     "runs": [["opt-lion-lr1e-4"], ["opt-lion-lr3e-4"], ["opt-lion-lr1e-3"]]},
    {"key": "adam", "label": "Adam", "short": "Adam",
     "runs": [["opt-adam-lr1.2e-3"], ["opt-adam-lr2.4e-3"], ["opt-adam-lr4.8e-3"]]},
    {"key": "adamw", "label": "AdamW", "short": "AdamW",
     "runs": [["opt-adamw-lr1.2e-3"], ["opt-adamw-lr2.4e-3"], ["opt-adamw-lr4.8e-3"]]},
    {"key": "muon", "label": "Muon with AdamW", "short": "Muon+AdamW", "recipe": True,
     "runs": [["opt-muon-lr0.01"], ["swap-base", "core-probe-final"], ["opt-muon-lr0.04"]]},
]
SETTINGS = ("optimizer", "lr", "lr_embedding", "lr_unembedding", "weight_decay", "beta1", "beta2", "momentum",
            "nesterov", "grad_clip", "schedule", "warmup_tokens", "cooldown_frac", "final_lr_frac")


def is_the_recipe(run: dict, tokens: float = 50e6, name: str = "gptpp-ladder") -> bool:
    """Was this run made with exactly the preset's model and training recipe (the 2026 recipe at the ladder size by
    default) on this budget?"""
    want_model = preset(name).to_dict()
    want_train = train_preset(name).replace(tokens=tokens).to_dict()
    return run["config"]["model_config"] == want_model and run["config"]["train_config"] == want_train


def entry(run: dict, stand_in: bool) -> dict:
    cfg = run["config"]["train_config"]
    out = {
        "run": run["run"], "status": run["status"], "lr": cfg["lr"], "val_loss": run["val_loss"],
        "steps": run["steps"], "tokens": run["tokens"],
        "settings": {k: cfg[k] for k in SETTINGS},
        "val": {"steps": run["val"]["step"], "loss": run["val"]["loss"]},
        "train": {"steps": run["train"]["step"], "loss": run["train"]["loss"]},
    }
    if run["status"] == "stopped":
        out["stopped_at_tokens"], out["error"] = run["stopped_at"], run["error"]
    if stand_in:
        out["stand_in"] = True
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(DATA))
    ap.add_argument("--runs", nargs="*", default=[], metavar="NAME=RUN",
                    help="read another run under a sweep's name (for trying the figure before the sweep has finished)")
    args = ap.parse_args()
    alias = dict(item.split("=", 1) for item in args.runs)
    optimizers, missing, first = [], [], None
    for group in SWEEP:
        runs = []
        for candidates in group["runs"]:
            found = None
            for k, name in enumerate(candidates):
                run = read_run(alias.get(name, name))
                if run is None or run["status"] == "running":
                    continue
                if k > 0 and not is_the_recipe(run):  # a stand-in must be the same configuration
                    continue
                run["run"] = name if name in alias else run["run"]
                found = entry(run, stand_in=k > 0 or name in alias)
                if found.get("stand_in"):
                    found["stands_in_for"] = candidates[0]
                break
            if found is None:
                missing.append(candidates[0])
            else:
                runs.append(found)
                first = first or run
        done = [k for k, r in enumerate(runs) if r["status"] == "completed"]
        best = min(done, key=lambda k: runs[k]["val_loss"]) if done else None
        optimizers.append({"key": group["key"], "label": group["label"], "short": group["short"],
                           "recipe": bool(group.get("recipe")), "runs": runs, "best": best})
    if first is None:
        raise SystemExit("no run of the sweep has finished")
    config = first["config"]
    out = {
        "model": {"preset": "gptpp-ladder", "layers": config["model_config"]["n_layer"], "width": config["model_config"]["d_model"],
                  "params": config["params"]["total"]},
        "data": config["train_config"]["data"], "tokens": first["tokens"], "steps": first["steps"],
        "tokens_per_step": first["tokens_per_step"], "log_every": config["train_config"]["log_every"],
        "eval_tokens": config["train_config"]["eval_tokens"], "val_tokens": config["train_config"]["val_tokens"],
        "seed": config["train_config"]["seed"], "optimizers": optimizers, "missing": missing,
        "note": "train.loss[k] is the mean training loss of the steps (train.steps[k-1], train.steps[k]]; "
                "val.loss[k] the loss on the fixed held-out tokens after val.steps[k] steps; a step is tokens_per_step tokens",
    }
    size = write_json(Path(args.out) / f"{NAME}.json", out)
    print(f"{NAME}.json: {size / 1e3:.1f} kB")
    for o in optimizers:
        cells = [f"{r['lr']:g}: {r['val_loss']:.4f}" if r["val_loss"] is not None else f"{r['lr']:g}: {r['status']}"
                 for r in o["runs"]]
        print(f"  {o['label']:<18} {' | '.join(cells) or 'no run yet'}" + (f"   best {o['runs'][o['best']]['lr']:g}" if o["best"] is not None else ""))
    if missing:
        print("  not finished:", ", ".join(missing))


if __name__ == "__main__":
    main()
