"""Learning-rate schedules and the runs that differ only in schedule or in warmup: `learning-rate-schedules.json`
for fig-06-03.

    python -m gptpp.export.schedule_runs [--out <dir>] [--runs name=run ...]

Two families of runs of the ladder model (8 layers, 512 wide) on the same 50M tokens in the same order:

    the 2026 recipe (preset gptpp-ladder), changing only the schedule
        swap-base             trapezoid: 2M tokens of warmup, flat, linear decay over the last 65% to a twentieth
        swap-sched-cosine     warmup, then half a cosine down to a twentieth
        swap-sched-linear     warmup, then a straight line down to a twentieth
        swap-sched-constant   warmup, then flat to the end
        swap-no-warmup        the trapezoid without its warmup
    the 2019 recipe (preset gpt2-2019-ladder: AdamW, 8M tokens of warmup, cosine to a tenth)
        warm-2019-base        the recipe as it is, on 50M tokens
        warm-2019-none        the same without warmup
        warm-2019-post        the recipe with normalization after each sublayer, as in the transformer of 2017
        warm-2019-post-none   the same without warmup
        ladder-0-gpt2-2019-s0 the recipe on 100M tokens, of which the first 50M are kept: the same warmup, a
                              cosine twice as long, so at 50M tokens its rate is still at 61% of the peak
    the 2019 recipe at steps of 8,192 tokens and rate 6e-4 (the same 8M tokens of warmup are then 977 steps, not 122)
        warm-2019-small-pre        normalization before each sublayer (the recipe's), with warmup
        warm-2019-small-pre-none   the same without warmup
        warm-2019-small-post       normalization after each sublayer, with warmup
        warm-2019-small-post-none  the same without warmup

A review of the runs found that warm-2019-post and warm-2019-post-none (7.690 both) are a collapse within 40 steps
to predicting token frequencies, with warmup and without: they say nothing about warmup. The four runs at small
steps are the comparison of norm placement with and without warmup; the four at the ladder's step are exported
for one line of text. The training records of a run with smaller steps are averaged in groups, so that one
record spans as many tokens as a record of the other runs.

Every run is compared with the run of its own recipe that keeps the recipe's schedule (`control`); the exporter
checks from the two configurations that a run and its control differ in nothing but the fields named in
`differs`. The schedule of each run is drawn from the trainer's own function, `gptpp.optim.lr_multiplier`, with
the run's configuration, at `samples` evenly spaced token counts; the rate the trainer logged at every record
is compared with that function (`schedule_matches_log`).

A run that has not finished is exported as `planned`: its schedule is drawn all the same, from the configuration
its command in experiments/*.jsonl will produce, and its losses are empty.

The exporter uses whatever has finished and lists what is missing: run it again when the queue is done. Two
runs have a stand-in of exactly the same configuration, made earlier, which is used and marked until they finish:
core-probe-final for swap-base, core-probe-gpt2-lr24 for warm-2019-base.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import json
import shlex

from .. import repo_dir
from ..config import TrainConfig
from ..optim import lr_multiplier
from .common import DATA, write_json
from .optimizer_sweep import is_the_recipe
from .run_curves import read_run, same_recipe

NAME = "learning-rate-schedules"
SAMPLES = 200  # the schedule is sampled at SAMPLES + 1 token counts from 0 to the budget
RUNS = [
    {"key": "trapezoid", "label": "trapezoid", "recipe": "2026", "runs": ["swap-base", "core-probe-final"], "control": None,
     "differs": []},
    {"key": "cosine", "label": "cosine", "recipe": "2026", "runs": ["swap-sched-cosine"], "control": "trapezoid",
     "differs": ["schedule"]},
    {"key": "linear", "label": "linear", "recipe": "2026", "runs": ["swap-sched-linear"], "control": "trapezoid",
     "differs": ["schedule"]},
    {"key": "constant", "label": "constant", "recipe": "2026", "runs": ["swap-sched-constant"], "control": "trapezoid",
     "differs": ["schedule"]},
    {"key": "no-warmup", "label": "trapezoid without warmup", "recipe": "2026", "runs": ["swap-no-warmup"],
     "control": "trapezoid", "differs": ["warmup_tokens"]},
    {"key": "2019", "label": "cosine with 8M tokens of warmup", "recipe": "2019", "runs": ["warm-2019-base", "core-probe-gpt2-lr24"],
     "control": None, "differs": [], "preset": "gpt2-2019-ladder"},
    {"key": "2019-no-warmup", "label": "cosine without warmup", "recipe": "2019", "runs": ["warm-2019-none"],
     "control": "2019", "differs": ["warmup_tokens"]},
    {"key": "2019-post", "label": "norm after each sublayer, cosine with 8M tokens of warmup", "recipe": "2019-post",
     "runs": ["warm-2019-post"], "control": None, "differs": []},
    {"key": "2019-post-no-warmup", "label": "norm after each sublayer, cosine without warmup", "recipe": "2019-post",
     "runs": ["warm-2019-post-none"], "control": "2019-post", "differs": ["warmup_tokens"]},
    {"key": "2019-post-w32", "label": "norm after each sublayer, cosine with 32M tokens of warmup", "recipe": "2019-post",
     "runs": ["warm-2019-post-w32M"], "control": "2019-post", "differs": ["warmup_tokens"]},
    {"key": "2019-post-w32-slow", "label": "norm after each sublayer, cosine with 32M tokens of warmup, rate 6e-4", "recipe": "2019-post",
     "runs": ["warm-2019-post-w32M-lr6e-4"], "control": "2019-post", "differs": ["lr", "warmup_tokens"]},
    {"key": "small-pre", "label": "pre-norm, cosine with 8M tokens of warmup, steps of 8,192 tokens", "recipe": "2019-small",
     "runs": ["warm-2019-small-pre"], "control": None, "differs": []},
    {"key": "small-pre-no-warmup", "label": "pre-norm, cosine without warmup, steps of 8,192 tokens", "recipe": "2019-small",
     "runs": ["warm-2019-small-pre-none"], "control": "small-pre", "differs": ["warmup_tokens"]},
    {"key": "small-post", "label": "norm after each sublayer, cosine with 8M tokens of warmup, steps of 8,192 tokens",
     "recipe": "2019-small-post", "runs": ["warm-2019-small-post"], "control": None, "differs": []},
    {"key": "small-post-no-warmup", "label": "norm after each sublayer, cosine without warmup, steps of 8,192 tokens",
     "recipe": "2019-small-post", "runs": ["warm-2019-small-post-none"], "control": "small-post", "differs": ["warmup_tokens"]},
    {"key": "2019-long", "label": "the first 50M tokens of the 100M-token run", "recipe": "2019",
     "runs": ["ladder-0-gpt2-2019-s0"], "control": "2019", "differs": ["tokens"], "cut": True},
]
SCHEDULE_FIELDS = ("schedule", "warmup_tokens", "cooldown_frac", "final_lr_frac")


def planned(name: str):
    """The configuration a queued run will have, from its command in the job lists: (TrainConfig, total tokens,
    tokens per step), or None if no job has that name."""
    from ..train import build_configs, parse_args

    for jobs in sorted((repo_dir() / "experiments").glob("*.jsonl")):
        for line in jobs.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            job = json.loads(line)
            if job.get("name") != name or "gptpp.train" not in job.get("cmd", ""):
                continue
            words = shlex.split(job["cmd"])
            model_cfg, train_cfg = build_configs(parse_args(words[words.index("gptpp.train") + 1:]))
            tps = train_cfg.device_batch * (train_cfg.seq_len or model_cfg.block_size) * train_cfg.grad_accum
            return train_cfg, max(1, round(train_cfg.tokens / tps)) * tps, tps
    return None


def thin(steps: list, loss: list, factor: int) -> tuple[list, list]:
    """Records averaged in groups of `factor`: the last step of each group, the mean of its losses."""
    out_steps, out_loss = [], []
    for i in range(0, len(steps), factor):
        chunk = loss[i:i + factor]
        out_steps.append(steps[min(i + factor, len(steps)) - 1])
        out_loss.append(sum(chunk) / len(chunk))
    return out_steps, out_loss


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(DATA))
    ap.add_argument("--tokens", type=float, default=50e6, help="the budget the runs are compared on")
    ap.add_argument("--runs", nargs="*", default=[], metavar="NAME=RUN",
                    help="read another run under a name of the table (for trying the figure before the runs finish)")
    args = ap.parse_args()
    alias = dict(item.split("=", 1) for item in args.runs)
    found: dict[str, dict] = {}
    out_runs, missing, problems = [], [], []
    budget, ref_tps = None, None
    for spec in RUNS:
        run, stand_in = None, False
        for k, name in enumerate(spec["runs"]):
            candidate = read_run(alias.get(name, name), max_tokens=None)
            if candidate is None or candidate["status"] == "running":
                continue
            if k > 0 and not is_the_recipe(candidate, args.tokens, spec.get("preset", "gptpp-ladder")):
                continue
            run, stand_in = candidate, k > 0 or name in alias
            break
        if run is None:
            missing.append(spec["runs"][0])
            plan = planned(spec["runs"][0])
            if plan is not None and not spec.get("cut"):  # not run yet: its schedule can be drawn all the same
                cfg, total, tps = plan
                out_runs.append({
                    "key": spec["key"], "label": spec["label"], "recipe": spec["recipe"], "control": spec["control"],
                    "run": spec["runs"][0], "status": "planned", "differs": None,
                    "schedule": {k: getattr(cfg, k) for k in SCHEDULE_FIELDS}, "peak_lr": cfg.lr, "optimizer": cfg.optimizer,
                    "run_tokens": total, "shown_tokens": total, "tokens_per_step": tps, "val_loss": None,
                    "mult": [lr_multiplier(total * k / SAMPLES, total, cfg) for k in range(SAMPLES + 1)],
                    "schedule_matches_log": None, "val": {"steps": [], "loss": []}, "train": {"steps": [], "loss": []},
                })
            continue
        cfg = TrainConfig.from_dict(run["config"]["train_config"])
        total, tps = run["tokens"], run["tokens_per_step"]
        if spec.get("cut"):  # a longer run: keep the records of the first `budget` tokens
            limit = budget or args.tokens
            run = read_run(run["run"], max_tokens=limit * 1.005)  # the evaluation nearest the budget is kept
            run["status"] = "completed"
            shown = limit
        else:
            shown = total
            budget = budget or total
        # the trainer's schedule, sampled; and checked against the rate the trainer logged at every record
        mult = [lr_multiplier(shown * k / SAMPLES, total, cfg) for k in range(SAMPLES + 1)]
        logged_ok = all(abs(lr_multiplier((step - 0.5) * tps, total, cfg) - m) < 1e-9
                        for step, m in zip(run["train"]["step"], run["train"]["lr_mult"]))
        if not logged_ok:
            problems.append(f"{run['run']}: the logged rate is not the schedule function's")
        steps_out, loss_out = run["train"]["step"], run["train"]["loss"]
        if ref_tps and tps < ref_tps:  # smaller steps: one record per as many tokens as a record of the other runs spans
            steps_out, loss_out = thin(steps_out, loss_out, round(ref_tps / tps))
        ref_tps = ref_tps or tps
        differs = None
        if spec["control"] and spec["control"] in found:
            differs = same_recipe(run, found[spec["control"]])
            if sorted(differs) != sorted(spec["differs"]) and not stand_in and not found[spec["control"]].get("stand_in"):
                problems.append(f"{run['run']} differs from its control in {differs}, expected {spec['differs']}")
        run["stand_in"] = stand_in
        found[spec["key"]] = run
        train_cfg = run["config"]["train_config"]
        entry = {
            "key": spec["key"], "label": spec["label"], "recipe": spec["recipe"], "control": spec["control"],
            "run": spec["runs"][0] if stand_in and spec["runs"][0] in alias else run["run"], "status": run["status"],
            "differs": differs, "schedule": {k: train_cfg[k] for k in SCHEDULE_FIELDS},
            "peak_lr": train_cfg["lr"], "optimizer": train_cfg["optimizer"], "run_tokens": total, "shown_tokens": shown,
            "tokens_per_step": tps, "norm_placement": run["config"]["model_config"].get("norm_placement"),
            "val_loss": run["val"]["loss"][-1] if spec.get("cut") and run["val"]["loss"] else run["val_loss"],
            "mult": mult, "schedule_matches_log": logged_ok,
            "val": {"steps": run["val"]["step"], "loss": run["val"]["loss"]},
            "train": {"steps": steps_out, "loss": loss_out},
        }
        if run["status"] == "stopped":
            entry["stopped_at_tokens"], entry["error"] = run["stopped_at"], run["error"]
        if stand_in:
            entry["stand_in"], entry["stands_in_for"] = True, spec["runs"][0]
        if spec.get("cut"):
            entry["cut"] = True
        out_runs.append(entry)
    if not found:
        raise SystemExit("no run has finished")
    any_run = next(iter(found.values()))
    out = {
        "model": {"layers": any_run["config"]["model_config"]["n_layer"], "width": any_run["config"]["model_config"]["d_model"]},
        "data": any_run["config"]["train_config"]["data"], "tokens": budget or args.tokens,
        "tokens_per_step": any_run["tokens_per_step"], "log_every": any_run["config"]["train_config"]["log_every"],
        "samples": SAMPLES, "runs": out_runs, "missing": missing, "problems": problems,
        "note": "mult[k] is the trainer's schedule at tokens = shown_tokens * k / samples, as a fraction of the peak rate; "
                "train.loss[k] is the mean training loss of the steps (train.steps[k-1], train.steps[k]]; val.loss[k] the "
                "loss on the fixed held-out tokens after val.steps[k] steps",
    }
    size = write_json(Path(args.out) / f"{NAME}.json", out)
    print(f"{NAME}.json: {size / 1e3:.1f} kB")
    for r in out_runs:
        print(f"  {r['key']:<21} {r['run']:<26} {r['status']:<10} final {r['val_loss'] if r['val_loss'] is None else round(r['val_loss'], 4)}"
              f"  differs from its control in {r['differs']}" + ("  (stand-in)" if r.get("stand_in") else ""))
    if missing:
        print("  not finished:", ", ".join(missing))
    for p in problems:
        print("  PROBLEM:", p)


if __name__ == "__main__":
    main()
