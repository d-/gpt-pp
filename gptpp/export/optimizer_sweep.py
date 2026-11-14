"""The ladder's model under five rules for its hidden matrices: `optimizers-on-the-ladder.json` for fig-06-04.

    python -m gptpp.export.optimizer_sweep [--out <dir>] [--runs name=run ...]

The comparison (experiments/*.jsonl, runs opt-<rule>-lr<rate>): the 2026 recipe on the ladder's model (preset
gptpp-ladder: 8 layers, 512 wide), the same 50M tokens of FineWeb-EDU in the same order, the same seed, with ONE
line changed: the rule that updates the hidden matrices.

    muon    opt-muon-lr0.01 | swap-base (0.02, the recipe) | opt-muon-lr0.04       Muon, momentum 0.95
    adamw   opt-adamw-lr3e-4 | 6e-4 | 1.2e-3 | 2.4e-3 | 4.8e-3                      AdamW, betas (0.9, 0.95), decay 0.1
    adam    opt-adam-lr3e-4 | 6e-4 | 1.2e-3 | 2.4e-3 | 4.8e-3                       the same with its weight decay set to zero
    lion    opt-lion-lr3e-5 | 1e-4 | 3e-4 | 1e-3                                    Lion, betas (0.9, 0.99), decay 0.1
    sgd     opt-sgd-lr0.03 | 0.1 | 0.3 | 1                                          SGD, Nesterov momentum 0.9

Everything else is the recipe's in every one of these runs: the embedding and the unembedding are trained by AdamW
with betas (0.8, 0.95) at their own rates (0.3674 and 0.0098), the schedule is the recipe's trapezoid (2M tokens
of warmup, flat, a line down over the last 65%), nothing is clipped. The exporter checks this from the
configurations (`differs`: the fields in which a run differs from swap-base).

That schedule was chosen with Muon. A review of the runs found that it handicaps AdamW (30 steps of warmup and no
clipping: the cliff in its losses above 1.2e-3). So every rule but Adam (which is AdamW with its decay set to zero)
is also run under a schedule of its own, the warmup and the clipping of AdamW's recipe at this size, with nothing
else changed, and AdamW with that recipe's cosine as well:

    own         opt-sgd-own-lr0.3 | 1 | 3,  opt-lion-own-lr1e-4 | 3e-4 | 1e-3,
                opt-adamw-own-lr6e-4 | 1.2e-3 | 2.4e-3 | 4.8e-3                8M tokens of warmup, clipping at norm one
    own-cosine  opt-adamw-cos-lr1.2e-3 | 2.4e-3                               the same, and a cosine to a tenth

Every run carries its `kind` (recipe, own, own-cosine). A rule's `best` is its run with the lowest final held-out
loss under any schedule, and `at_end` is asked of the rates of that run's schedule. The two best rows have a second
seed (`second_seed`: opt-adamw-own-lr2.4e-3-s1, swap-base-s1), checked to differ from the first in the seed alone.

An earlier job list gave AdamW and Adam ONE rate for every parameter group (the way of 2019, with a cosine and
clipping). Those runs ended near 5.0 nats at every rate; the likely reason, which no run isolates, is that the
embedding, normalized after the lookup, wants a rate hundreds of times that of a hidden matrix. They are kept as
optflat-adamw-lr*, optflat-adam-lr* and exported under `flat`: a caution, not a comparison of optimizers.

The exporter uses whatever has finished and lists what is missing: run it again when the queue is done. A run
that stopped on a loss that is not finite is kept, with its curve up to there, as a result. If swap-base has
not finished, a run of exactly the same configuration made earlier (core-probe-final) stands in and is marked.

What is written, for every run: its rate, its status, the loss on the fixed held-out set at each evaluation
(`val`), and the training loss at each log record (`train`: the mean over ten steps of the loss on tokens not
yet trained on; `steps` are the steps at which the records were written). For every rule: `best`, the index of
the run with the lowest final held-out loss, and `at_end`, whether that rate is the lowest or the highest of
the rule's rates that have been run (then the rule may be undertuned on that side).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..config import preset, train_preset
from .common import DATA, write_json
from .run_curves import read_run, same_recipe

NAME = "optimizers-on-the-ladder"
RECIPE_RUN = ["swap-base", "core-probe-final"]
OWN = ("warmup_tokens", "grad_clip")            # what a rule's own schedule changes besides the rule
KINDS = {"recipe": (), "own": OWN, "own-cosine": OWN + ("schedule", "final_lr_frac")}
SWEEP = [
    {"key": "sgd", "label": "SGD with momentum", "short": "SGD", "rule": "SGD with Nesterov momentum 0.9",
     "runs": {"recipe": [["opt-sgd-lr0.03"], ["opt-sgd-lr0.1"], ["opt-sgd-lr0.3"], ["opt-sgd-lr1"]],
              "own": [["opt-sgd-own-lr0.3"], ["opt-sgd-own-lr1"], ["opt-sgd-own-lr3"]]}},
    {"key": "lion", "label": "Lion", "short": "Lion", "rule": "Lion, betas (0.9, 0.99), weight decay 0.1",
     "runs": {"recipe": [["opt-lion-lr3e-5"], ["opt-lion-lr1e-4"], ["opt-lion-lr3e-4"], ["opt-lion-lr1e-3"]],
              "own": [["opt-lion-own-lr1e-4"], ["opt-lion-own-lr3e-4"], ["opt-lion-own-lr1e-3"]]}},
    {"key": "adam", "label": "Adam", "short": "Adam", "rule": "AdamW with its weight decay set to zero, betas (0.9, 0.95)",
     "runs": {"recipe": [["opt-adam-lr3e-4"], ["opt-adam-lr6e-4"], ["opt-adam-lr1.2e-3"], ["opt-adam-lr2.4e-3"], ["opt-adam-lr4.8e-3"]]}},
    {"key": "adamw", "label": "AdamW", "short": "AdamW", "rule": "AdamW, betas (0.9, 0.95), weight decay 0.1",
     "runs": {"recipe": [["opt-adamw-lr3e-4"], ["opt-adamw-lr6e-4"], ["opt-adamw-lr1.2e-3"], ["opt-adamw-lr2.4e-3"], ["opt-adamw-lr4.8e-3"]],
              "own": [["opt-adamw-own-lr6e-4"], ["opt-adamw-own-lr1.2e-3"], ["opt-adamw-own-lr2.4e-3"], ["opt-adamw-own-lr4.8e-3"]],
              "own-cosine": [["opt-adamw-cos-lr1.2e-3"], ["opt-adamw-cos-lr2.4e-3"]]}},
    {"key": "muon", "label": "Muon", "short": "Muon", "rule": "Muon, momentum 0.95, five Newton-Schulz steps", "recipe": True,
     "runs": {"recipe": [["opt-muon-lr0.01"], RECIPE_RUN, ["opt-muon-lr0.04"]]}},
]
SECOND_SEED = [("adamw", "opt-adamw-own-lr2.4e-3", "opt-adamw-own-lr2.4e-3-s1"), ("muon", "swap-base", "swap-base-s1")]
FLAT = ["optflat-adamw-lr1.2e-3", "optflat-adamw-lr2.4e-3", "optflat-adamw-lr4.8e-3",
        "optflat-adam-lr1.2e-3", "optflat-adam-lr2.4e-3", "optflat-adam-lr4.8e-3"]
SETTINGS = ("optimizer", "lr", "lr_embedding", "lr_unembedding", "weight_decay", "beta1", "beta2", "momentum",
            "nesterov", "grad_clip", "schedule", "warmup_tokens", "cooldown_frac", "final_lr_frac")
KEPT = ("lr_embedding", "lr_unembedding", "schedule", "warmup_tokens", "cooldown_frac", "final_lr_frac", "grad_clip")


def is_the_recipe(run: dict, tokens: float = 50e6, name: str = "gptpp-ladder") -> bool:
    """Was this run made with exactly the preset's model and training recipe (the 2026 recipe at the ladder size by
    default) on this budget?"""
    want_model = preset(name).to_dict()
    want_train = train_preset(name).replace(tokens=tokens).to_dict()
    return run["config"]["model_config"] == want_model and run["config"]["train_config"] == want_train


def entry(run: dict, stand_in: bool, recipe: dict | None) -> dict:
    cfg = run["config"]["train_config"]
    out = {
        "run": run["run"], "status": run["status"], "lr": cfg["lr"], "val_loss": run["val_loss"],
        "steps": run["steps"], "tokens": run["tokens"],
        "settings": {k: cfg[k] for k in SETTINGS},
        "differs": same_recipe(run, recipe) if recipe is not None else None,
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
    recipe = next((r for r in (read_run(n) for n in RECIPE_RUN) if r is not None and r["status"] == "completed" and is_the_recipe(r)), None)
    optimizers, missing, problems = [], [], []
    for group in SWEEP:
        runs = []
        for kind, lists in group["runs"].items():
            for candidates in lists:
                found = None
                for k, name in enumerate(candidates):
                    run = read_run(alias.get(name, name))
                    if run is None or run["status"] == "running":
                        continue
                    if k > 0 and not is_the_recipe(run):  # a stand-in must be the same configuration
                        continue
                    run["run"] = name if name in alias else run["run"]
                    found = entry(run, k > 0 or name in alias, recipe)
                    if found.get("stand_in"):
                        found["stands_in_for"] = candidates[0]
                    break
                if found is None:
                    missing.append(candidates[0])
                    continue
                found["kind"] = kind
                runs.append(found)
                # nothing the recipe keeps may differ from the recipe's run, but what a schedule of the rule's own changes
                differs = found["differs"] or []
                moved = [f for f in differs if f in KEPT and f not in KINDS[kind]]
                if moved and not found.get("stand_in"):
                    problems.append(f"{found['run']} differs from the recipe in {moved}, which a run of kind {kind} should keep")
                lacking = [f for f in KINDS[kind] if f not in differs]
                if lacking and recipe is not None:
                    problems.append(f"{found['run']} should differ from the recipe in {lacking} and does not")
        order = list(KINDS)
        runs.sort(key=lambda r: (order.index(r["kind"]), r["lr"]))
        done = [k for k, r in enumerate(runs) if r["status"] == "completed"]
        best = min(done, key=lambda k: runs[k]["val_loss"]) if done else None
        same = [k for k, r in enumerate(runs) if best is not None and r["kind"] == runs[best]["kind"]]
        schedules = {}
        for r in runs:  # what each schedule of the rule's own changes, as its runs have it
            if r["kind"] != "recipe" and r["kind"] not in schedules:
                schedules[r["kind"]] = {f: r["settings"][f] for f in KINDS[r["kind"]]}
        optimizers.append({"key": group["key"], "label": group["label"], "short": group["short"], "rule": group["rule"],
                           "recipe": bool(group.get("recipe")), "runs": runs, "best": best,
                           "at_end": best is not None and len(same) > 1 and best in (same[0], same[-1]),
                           "schedules": schedules, "planned": sum(len(v) for v in group["runs"].values())})
    second = []
    for key, first, name in SECOND_SEED:  # the two best rows in a second seed
        a, b = read_run(first), read_run(name)
        if a is None or b is None or b["status"] != "completed":
            missing.append(name)
            continue
        differs = same_recipe(b, a)
        if list(differs) != ["seed"]:
            problems.append(f"{name} should differ from {first} in its seed alone and differs in {differs}")
        second.append({"key": key, "run": b["run"], "of": a["run"], "lr": b["config"]["train_config"]["lr"],
                       "val_loss": b["val_loss"], "first": a["val_loss"]})
    if recipe is None and not any(o["runs"] for o in optimizers):
        raise SystemExit("no run of the sweep has finished")
    first = recipe or read_run(next(o["runs"][0]["run"] for o in optimizers if o["runs"]))
    config = first["config"]
    flat = []
    for name in FLAT:  # one rate for every parameter group: kept as a caution
        run = read_run(name)
        if run is not None and run["status"] == "completed":
            cfg = run["config"]["train_config"]
            flat.append({"run": name, "optimizer": cfg["optimizer"], "lr": cfg["lr"], "val_loss": run["val_loss"],
                         "one_rate": cfg["lr"] == cfg["lr_embedding"] == cfg["lr_unembedding"]})
    train_cfg = config["train_config"]
    out = {
        "model": {"preset": "gptpp-ladder", "layers": config["model_config"]["n_layer"], "width": config["model_config"]["d_model"],
                  "params": config["params"]["total"]},
        "data": train_cfg["data"], "tokens": first["tokens"], "steps": first["steps"],
        "tokens_per_step": first["tokens_per_step"], "log_every": train_cfg["log_every"],
        "eval_tokens": train_cfg["eval_tokens"], "val_tokens": train_cfg["val_tokens"], "seed": train_cfg["seed"],
        "design": {"changed": "the rule that updates the hidden matrices",
                   "kept": {"lr_embedding": train_cfg["lr_embedding"], "lr_unembedding": train_cfg["lr_unembedding"],
                            "tables": "AdamW, betas (0.8, 0.95), no weight decay", "schedule": train_cfg["schedule"],
                            "warmup_tokens": train_cfg["warmup_tokens"], "cooldown_frac": train_cfg["cooldown_frac"],
                            "final_lr_frac": train_cfg["final_lr_frac"], "grad_clip": train_cfg["grad_clip"]}},
        "optimizers": optimizers, "second_seed": second, "flat": flat, "missing": missing, "problems": problems,
        "note": "train.loss[k] is the mean training loss of the steps (train.steps[k-1], train.steps[k]]; "
                "val.loss[k] the loss on the fixed held-out tokens after val.steps[k] steps; a step is tokens_per_step tokens; "
                "flat: runs with one rate for every parameter group, not part of the comparison",
    }
    size = write_json(Path(args.out) / f"{NAME}.json", out)
    print(f"{NAME}.json: {size / 1e3:.1f} kB")
    for o in optimizers:
        for kind in KINDS:
            cells = [f"{r['lr']:g}: {r['val_loss']:.4f}" if r["val_loss"] is not None else f"{r['lr']:g}: {r['status']}"
                     for r in o["runs"] if r["kind"] == kind]
            if cells:
                b = o["runs"][o["best"]] if o["best"] is not None else None
                tail = f"   BEST {b['lr']:g}" + (" (AT AN END OF ITS GRID)" if o["at_end"] else "") if b and b["kind"] == kind else ""
                print(f"  {o['label']:<18} {kind:<11} {' | '.join(cells)}{tail}")
    for s in second:
        print(f"  second seed of {s['of']}: {s['val_loss']:.4f} (first {s['first']:.4f})")
    print("  one rate for every group:", ", ".join(f"{r['run']} {r['val_loss']:.4f}" for r in flat) or "none")
    if missing:
        print("  not finished:", ", ".join(missing))
    for p in problems:
        print("  PROBLEM:", p)


if __name__ == "__main__":
    main()
