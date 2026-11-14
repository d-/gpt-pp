"""The recipe race: the 2019 recipe against the 2026 recipe at the same shape, batch and token budget, as the file
the site's figure `gpt-recipe-race` (fig-07-03) draws, `assets/data/gpt/recipe-race.json`.

    python -m gptpp.export.recipe_race [--out <dir>] [--name recipe-race]
        [--runs2019 race-2019-lr6e-4-100M,race-2019-lr1.2e-3-100M,race-2019-lr2.4e-3-100M] [--run2026 race-2026-100M]

It reads `config.json`, `log.jsonl` and `summary.json` of each run under `$GPTPP_WORK/ckpt/<run>/` (or the small
copy under `results/runs/<run>/`), takes the runs that have finished and names the ones that have not, so it can be
run at any time and again when more runs land. With the default names, any other finished run called
`race-2019-*` under `$GPTPP_WORK/ckpt/` (the 2019 recipe at another warmup or rate) joins the race. Nothing is trained or scored here. Given other run names than the
default four, the file says `stand_in: true`: the figure then says that it is not showing the race.

How "the tokens each recipe needs to reach a loss" is read. A run evaluates its held-out loss only a few times (the
race: every 20M tokens), so the curve between evaluations is the training loss: in a single pass every batch is text
the model has not seen, and the loss the trainer logs (the mean over the ten steps before a record, each computed
before its update) measures the same thing as the held-out evaluation, with more noise. It is smoothed over the
records whose token counts lie within 15% of a record's own (`smooth`). A level is a held-out evaluation of the best
2019 run (the one with the lowest final held-out loss); the 2026 recipe "reaches" it where its smoothed training
loss first falls below it, between two records, linear in log tokens (`crossing`). `train_gap` says how far a run's
smoothed training loss lies above its own held-out evaluations on average: where it is positive the 2026 crossing is
read a little late, and the ratio is understated.

The file, field by field:

    generated_by   "gptpp.export.recipe_race"
    stand_in       false for the race; true when other runs were named
    wanted         the run names asked for; missing: those that have not finished (not there, running, or failed)
    shape          {n_layer, d_model, n_head, block_size, vocab_size}: what every run shares (checked)
    budget         {tokens, steps, tokens_per_step, device_batch, seq_len, n_gpu, eval_every_tokens}: the same (checked)
    data           {dataset, tokenizer, bytes_per_token_val, bytes_per_token_train}
    runs           the finished runs, the 2019 ones by learning rate, then the 2026 one; each
      name, recipe ("2019" | "2026"), preset, optimizer, lr (the matrices' peak rate: AdamW's or Muon's), seed
      schedule       {schedule, warmup_tokens, cooldown_frac, final_lr_frac}
      params         {embedding, position, unembedding, attention, mlp, norm, total, trunk (attention + mlp)}
      tokens, steps, seconds, train_seconds, tokens_per_second, peak_memory_gb      from the trainer's summary
      val_loss, val_bpb, best_val_loss, best_val_bpb, train_loss                    the same
      val            {step, tokens, loss, bpb}: the held-out evaluations, the one before training included
      train          {step, loss, lr_mult}: every training record (nats per token; the schedule's multiplier)
      train_gap      mean over the evaluations after the first of (smoothed training loss - held-out loss), in bits
                     per byte: how far the curve the figure draws lies from the run's own held-out points
    best_2019      the name of the 2019 run with the lowest final held-out loss (null if none has finished)
    run_2026       the name of the 2026 run (null if it has not finished)
    levels         when both exist: for the evaluations of the best 2019 run nearest 20%, 60% and 100% of the budget
                   {bpb, loss (the 2019 run's held-out loss there), tokens_2019, step_2019, seconds_2019,
                    tokens_2026 (where the 2026 run's smoothed training loss first falls below it; null if never),
                    between: [tokens, tokens] (the two records around it), seconds_2026, ratio (tokens_2019 /
                    tokens_2026), time_ratio (seconds_2019 / seconds_2026)}; seconds are tokens divided by the run's
                   own steady tokens per second
    numbers        what the post quotes: tokens, final_2026_bpb, final_2026_loss, final_2019_bpb, final_2019_loss,
                   best_2019_lr, gap_bpb (2019 minus 2026 at the end), ratio_at_end, time_ratio_at_end,
                   tokens_2026_at_end (the tokens at which the 2026 recipe had reached the 2019 recipe's final
                   loss), tokens_per_second_2019, tokens_per_second_2026, params_2019, params_2026, trunk_params

Losses keep six significant digits.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .common import CKPT, DATA, write_atomic

REPO = Path(__file__).resolve().parents[2]
RUNS_2019 = ["race-2019-lr6e-4-100M", "race-2019-lr1.2e-3-100M", "race-2019-lr2.4e-3-100M"]
RUN_2026 = "race-2026-100M"
SMOOTH = 0.15            # a record is averaged with those whose tokens lie within this share of its own
FRACTIONS = (0.2, 0.6, 1.0)
LN2 = math.log(2.0)


def sig(x: float | None, digits: int = 6) -> float | None:
    return None if x is None else float(f"{x:.{digits}g}")


def smooth(tokens: list[float], values: list[float], rel: float = SMOOTH) -> list[float]:
    """Each value averaged with those whose token counts lie within a factor 1 + rel of its own."""
    out, n = [], len(tokens)
    for i in range(n):
        lo, hi, total, count = tokens[i] / (1 + rel), tokens[i] * (1 + rel), 0.0, 0
        for j in range(n):
            if lo <= tokens[j] <= hi:
                total += values[j]
                count += 1
        out.append(total / count)
    return out


def crossing(tokens: list[float], values: list[float], level: float) -> dict | None:
    """Where a curve first lies below `level`: between the last record at or above it and the first below, linear
    in log tokens. None if it never does."""
    for i, y in enumerate(values):
        if y < level:
            if i == 0:
                return {"tokens": float(tokens[0]), "between": [float(tokens[0]), float(tokens[0])]}
            x0, y0, x1 = tokens[i - 1], values[i - 1], tokens[i]
            f = (y0 - level) / (y0 - y)
            return {"tokens": float(math.exp(math.log(x0) + f * (math.log(x1) - math.log(x0)))), "between": [float(x0), float(x1)]}
    return None


def interp_log(x: float, xs: list[float], ys: list[float]) -> float:
    """ys at x, linear in log x between the records; the end values outside."""
    if x <= xs[0]:
        return ys[0]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            f = (math.log(x) - math.log(xs[i - 1])) / (math.log(xs[i]) - math.log(xs[i - 1]))
            return ys[i - 1] + f * (ys[i] - ys[i - 1])
    return ys[-1]


def read_run(name: str, root: Path) -> dict | None:
    """A finished run, or None."""
    for base in (root / name, REPO / "results" / "runs" / name):
        if not (base / "summary.json").exists():
            continue
        summary = json.loads((base / "summary.json").read_text())
        if summary.get("status") != "completed":
            continue
        config = json.loads((base / "config.json").read_text())
        train, val = [], []
        for line in (base / "log.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["kind"] == "train":
                train.append(r)
            elif r["kind"] == "val":
                val.append(r)
        mc, tc, d, meta, p = config["model_config"], config["train_config"], config["derived"], config["data_meta"], config["params"]
        tps = d["tokens_per_step"]
        bpt_train = meta["bytes_per_token_train"]
        run = {
            "name": name, "recipe": "2019" if config["preset"].startswith("gpt2-2019") else "2026", "preset": config["preset"],
            "optimizer": tc["optimizer"], "lr": tc["lr"], "seed": tc["seed"],
            "schedule": {k: tc[k] for k in ("schedule", "warmup_tokens", "cooldown_frac", "final_lr_frac")},
            "params": {**p, "trunk": p["attention"] + p["mlp"]},
            "tokens": summary["tokens"], "steps": summary["steps"], "seconds": summary["seconds"],
            "train_seconds": summary["train_seconds"], "tokens_per_second": summary["tokens_per_second"],
            "peak_memory_gb": summary.get("peak_memory_gb"),
            "val_loss": sig(summary["val_loss"]), "val_bpb": sig(summary["val_bpb"]),
            "best_val_loss": sig(summary["best_val_loss"]), "best_val_bpb": sig(summary["best_val_bpb"]),
            "train_loss": sig(summary["train_loss"]),
            "val": {"step": [r["step"] for r in val], "tokens": [r["tokens"] for r in val],
                    "loss": [sig(r["val_loss"]) for r in val], "bpb": [sig(r["val_bpb"]) for r in val]},
            "train": {"step": [r["step"] for r in train], "loss": [sig(r["loss"]) for r in train],
                      "lr_mult": [sig(r["lr_mult"], 5) for r in train]},
        }
        tokens = [s * tps for s in run["train"]["step"]]
        curve = smooth(tokens, [l / LN2 / bpt_train for l in run["train"]["loss"]])
        gaps = [interp_log(t, tokens, curve) - b for t, b in zip(run["val"]["tokens"][1:], run["val"]["bpb"][1:])]
        run["train_gap"] = sig(sum(gaps) / len(gaps), 3) if gaps else None
        run["_shape"] = {k: mc[k] for k in ("n_layer", "d_model", "n_head", "block_size", "vocab_size")}
        run["_budget"] = {"tokens": d["tokens"], "steps": d["steps"], "tokens_per_step": tps, "device_batch": tc["device_batch"],
                          "seq_len": d["seq_len"], "n_gpu": d["n_gpu"], "eval_every_tokens": d["eval_every_steps"] * tps}
        run["_data"] = {"dataset": meta["dataset"], "tokenizer": meta["tokenizer"], "bytes_per_token_val": meta["bytes_per_token_val"],
                        "bytes_per_token_train": bpt_train}
        return run
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs2019", default=",".join(RUNS_2019), help="the 2019 recipe's runs, one a learning rate")
    ap.add_argument("--run2026", default=RUN_2026)
    ap.add_argument("--ckpt-root", default=str(CKPT))
    ap.add_argument("--out", default=str(DATA))
    ap.add_argument("--name", default="recipe-race")
    args = ap.parse_args()

    names19 = [n.strip() for n in args.runs2019.split(",") if n.strip()]
    stand_in = names19 != RUNS_2019 or args.run2026 != RUN_2026
    if not stand_in:  # further 2019 runs of the race (another warmup, another rate) join it by their name
        for d in sorted(Path(args.ckpt_root).glob("race-2019-*")):
            if d.name not in names19 and (d / "summary.json").exists():
                names19.append(d.name)
    wanted = names19 + [args.run2026]
    found = {n: read_run(n, Path(args.ckpt_root)) for n in wanted}
    runs = [found[n] for n in wanted if found[n]]
    missing = [n for n in wanted if not found[n]]
    for r in runs:
        if r["name"] in names19 and r["recipe"] != "2019" or r["name"] == args.run2026 and r["recipe"] != "2026":
            raise SystemExit(f"{r['name']} is a run of the {r['recipe']} recipe ({r['preset']})")
    shape = budget = data = None
    for r in runs:
        s, b, dt = r.pop("_shape"), r.pop("_budget"), r.pop("_data")
        if shape is None:
            shape, budget, data = s, b, dt
        elif s != shape or b != budget or dt != data:
            raise SystemExit(f"{r['name']} differs from {runs[0]['name']} in shape, budget or data: {s} {b}")
    runs19 = sorted((r for r in runs if r["recipe"] == "2019"), key=lambda r: (r["lr"], -r["schedule"]["warmup_tokens"]))
    run26 = next((r for r in runs if r["recipe"] == "2026"), None)
    runs = runs19 + ([run26] if run26 else [])
    best = min(runs19, key=lambda r: r["val_loss"]) if runs19 else None

    levels, numbers = [], {}
    if best and run26:
        tps = budget["tokens_per_step"]
        tok26 = [s * tps for s in run26["train"]["step"]]
        curve26 = smooth(tok26, [l / LN2 / data["bytes_per_token_train"] for l in run26["train"]["loss"]])
        picked = []
        for f in FRACTIONS:
            i = min(range(1, len(best["val"]["tokens"])), key=lambda j: abs(best["val"]["tokens"][j] - f * budget["tokens"]))
            if i not in picked:
                picked.append(i)
        for i in picked:
            t19, bpb = best["val"]["tokens"][i], best["val"]["bpb"][i]
            at = crossing(tok26, curve26, bpb)
            s19 = t19 / best["tokens_per_second"]
            s26 = at["tokens"] / run26["tokens_per_second"] if at else None
            levels.append({"bpb": bpb, "loss": best["val"]["loss"][i], "tokens_2019": t19, "step_2019": best["val"]["step"][i],
                           "seconds_2019": sig(s19, 5), "tokens_2026": round(at["tokens"]) if at else None,
                           "between": [round(x) for x in at["between"]] if at else None, "seconds_2026": sig(s26, 5),
                           "ratio": sig(t19 / at["tokens"], 5) if at else None, "time_ratio": sig(s19 / s26, 5) if at else None})
        end = levels[-1]
        numbers = {
            "tokens": budget["tokens"], "final_2026_bpb": run26["val_bpb"], "final_2026_loss": run26["val_loss"],
            "final_2019_bpb": best["val_bpb"], "final_2019_loss": best["val_loss"], "best_2019_lr": best["lr"],
            "gap_bpb": sig(best["val_bpb"] - run26["val_bpb"], 4), "ratio_at_end": end["ratio"], "time_ratio_at_end": end["time_ratio"],
            "tokens_2026_at_end": end["tokens_2026"], "tokens_per_second_2019": best["tokens_per_second"],
            "tokens_per_second_2026": run26["tokens_per_second"], "params_2019": best["params"]["total"],
            "params_2026": run26["params"]["total"], "trunk_params": run26["params"]["trunk"],
        }

    out = {"generated_by": "gptpp.export.recipe_race", "stand_in": stand_in, "wanted": wanted, "missing": missing,
           "shape": shape, "budget": budget, "data": data, "runs": runs, "best_2019": best["name"] if best else None,
           "run_2026": run26["name"] if run26 else None, "smooth": SMOOTH, "levels": levels, "numbers": numbers}
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(out, separators=(",", ":"), ensure_ascii=False)
    write_atomic(out_dir / f"{args.name}.json", text.encode("utf-8"))
    print(f"{args.name}.json: {len(text.encode('utf-8')) / 1e3:.1f} kB; {'STAND-IN, ' if stand_in else ''}{len(runs)} of {len(wanted)} runs"
          + (f"; missing: {', '.join(missing)}" if missing else "")
          + (f"; best 2019 rate {best['lr']:g}: {best['val_bpb']:.4f} bits per byte" if best else "")
          + (f"; 2026: {run26['val_bpb']:.4f}" if run26 else "")
          + (f"; at the end the 2026 recipe needed {levels[-1]['ratio']} times fewer tokens" if levels and levels[-1]["ratio"] else ""))


if __name__ == "__main__":
    main()
