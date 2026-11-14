"""Collect the results of the small runs (the ladder from the 2019 recipe to the 2026 recipe, and the one-at-a-time
swaps around the 2026 recipe) into one file the site's figures read: `assets/data/gpt/ladder.json`.

    python -m gptpp.export.ladder_results [--runs results/runs] [--out <dir>] [--prefixes ladder- swap-] [--name ladder]

It reads the small copy of every finished run (`results/runs/<run>/config.json`, `summary.json`, `log.jsonl`; a run
counts as finished when its summary says "completed"), so it can be run at any time, takes what has finished, and is
run again when more runs land. Nothing is measured here: every number is copied from the trainer's own files.

The file (`ladder.json`), field by field:

    generated_by     "gptpp.export.ladder_results"
    model            the shape every run shares: {n_layer, d_model, n_head, block_size, vocab_size}
    data             {dataset, tokenizer, bytes_per_token (of the held-out text), val_tokens (the fixed held-out set)}
    ladder           the ten rungs, in order (a rung is listed even when no run of it has finished):
      tokens           the training budget of a rung, in tokens (from the first finished run; null if none)
      seeds            the seeds for which at least one rung has finished, e.g. [0] or [0, 1]
      complete         true when every rung has a finished run of seed 0
      spread           how far two seeds of the same rung lie apart, over the rungs that have two:
                       {rungs (how many), mean_abs_diff, max_abs_diff, min_abs_diff, rms_abs_diff} of the final
                       validation loss; nulls if none. The root mean square estimates the standard deviation of the
                       difference between two runs that differ only by seed: the yardstick is twice it
      split            the step of rung 7 taken apart in one seed, or null: {rung, seed, runs (the runs between the two
                       rungs: the rung-6 model under the 2026 training recipe with AdamW in place of Muon), best,
                       before (rung 6), with_adamw (the best of them), after (rung 7), rest_of_recipe (with_adamw less
                       before), rule (after less with_adamw: Muon in place of AdamW)}
      steps[k]         {k, name (the key: "gpt2-2019", "no-biases", ...), label (what to print: rung 7, key "muon",
                        is "the training recipe of 2026 (Muon, its rates and its schedule)"), short (for a tread:
                        "2026 training"), runs: {"<seed>": "<run name>"},
                        val_loss: [per seed, in the order of `seeds`; null where that seed has not finished],
                        val_bpb: [the same in bits per byte], mean_val_loss, mean_val_bpb (over finished seeds; null if
                        none), changed: {field: [before, after]} against the rung before it (the model's and the
                        trainer's fields that differ; {} for rung 0; null while either rung has no finished run),
                        params (the total parameter count; null if not finished)}
      longer_warmup    the 2019 side run again with a warmup of 32M tokens in place of 8M (ladder-<k>-<name>-w32M), or
                       null: {warmup_tokens, seed, runs[k], val_loss[k], gain[k] (less the published rung of that seed)};
                       null entries for the rungs without such a run. The stair keeps the published runs
      (split also has by_seed: {"<seed>": {seed, runs, best, before, with_adamw, after, rest_of_recipe, rule}} and,
       where rung 6 has a longer-warmup run, against_longer_warmup: {seed, before, step, rest_of_recipe})
    swaps            the runs that change one thing in the 2026 recipe, each at the same shorter budget:
      base             "swap-base", or null if it has not finished; bases: {"<seed>": the recipe's run of that seed}
      pairs            {"<what>": {runs: {"<seed>": name}, val_loss: [by seed], delta: [by seed, each against the
                       recipe of the same seed], mean_delta}}: a swap is `swap-<what>` (seed 0) and `swap-<what>-s1`
      spread           how far two seeds of one configuration lie apart under the 2026 recipe (the recipe itself and
                       every swap run twice): {pairs, min_abs_diff, max_abs_diff, mean_abs_diff, rms_abs_diff}; twice
                       the root mean square is the yardstick for a put-back (the ladder's is for AdamW-trained rungs)
      tokens           their budget in tokens
      families         {"act": [run names], "pos": [...], "norm": [...], "attn": [...], "sched": [...], "other": [...]}
    runs             every finished run by name:
      family           "ladder" | "swap" (| whatever other prefix was asked for)
      step, name       for a ladder run: its rung and the rung's name; for a swap: null and what follows "swap-"
      group            for a swap: "act", "pos", "norm", "attn", "sched" or "other"; null for a ladder run
      seed             the seed of the run
      preset, train_preset, overrides     as given to the trainer
      reference        the run it is compared with: the rung before it (same seed if finished, else any seed) or the
                       swaps' base; null for rung 0 and for the base
      changed          {field: [in the reference, in this run]}: the fields of the model's and the trainer's
                       configuration that differ from the reference ({} for rung 0 and the base; null if the
                       reference has not finished). The budget, the seed and the run's name are not counted.
      tokens, steps    tokens seen and optimizer steps
      seconds, train_seconds, tokens_per_second      wall-clock of the whole run, of training alone, and throughput
      params           parameter counts: {embedding, position, unembedding, attention, mlp, norm, total}
      val_loss, val_bpb            at the end of training: nats per token and bits per byte on the held-out set
      best_val_loss, best_val_bpb  the lowest of the evaluations
      train_loss       the last logged training loss
      curve            the validation curve: {tokens: [...], val_loss: [...], val_bpb: [...]}, one entry an evaluation,
                       the evaluation before training (tokens 0) included

Losses keep six significant digits (two seeds of one configuration differ in the fourth decimal).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .common import DATA, write_atomic

REPO = Path(__file__).resolve().parents[2]

# The ten rungs of the ladder, in order, as the job list names them (experiments/small.jsonl).
RUNGS = ["gpt2-2019", "no-biases", "rotary", "rmsnorm", "relu2", "qk-norm", "zero-init", "muon", "untied", "softcap"]
GROUPS = ("act", "pos", "norm", "attn", "sched")
# What a rung is called in print, and a short form for a tread of the stair or a list on a phone. From rung 6 to
# rung 7 the whole training recipe changes (Muon on the hidden matrices, and with it the tables' rates, beta1, the
# decay, the clipping, the warmup and the schedule): that rung is never "Muon" alone.
LABELS = {
    "gpt2-2019": ("the 2019 recipe", "2019"), "no-biases": ("no biases", "no biases"), "rotary": ("rotary positions", "rotary"),
    "rmsnorm": ("RMS norm without parameters", "RMSNorm"), "relu2": ("squared ReLU", "ReLU²"),
    "qk-norm": ("cosine scores (QK normalization)", "QK norm"), "zero-init": ("zero-initialized writers", "zero init"),
    "muon": ("the training recipe of 2026 (Muon, its rates and its schedule)", "2026 training"),
    "untied": ("untied tables, normalized embedding", "untied"), "softcap": ("the logit soft cap", "soft-cap"),
}
# Fields that say which run it is or how long it trained, not what the recipe is.
NOT_RECIPE = {"seed", "tokens", "eval_tokens", "val_tokens", "log_every", "sample_tokens", "compile"}


def round_sig(o, digits: int = 6):
    """Floats to `digits` significant digits (a whole number stays whole), through lists and dictionaries."""
    if isinstance(o, float):
        return int(o) if o.is_integer() and abs(o) < 1e15 else float(f"{o:.{digits}g}")
    if isinstance(o, dict):
        return {k: round_sig(v, digits) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [round_sig(v, digits) for v in o]
    return o


def read_run(folder: Path) -> dict | None:
    """A finished run's record, or None if the run has not finished."""
    summary_file, config_file, log_file = folder / "summary.json", folder / "config.json", folder / "log.jsonl"
    if not (summary_file.exists() and config_file.exists() and log_file.exists()):
        return None
    summary, config = json.loads(summary_file.read_text()), json.loads(config_file.read_text())
    if summary.get("status") != "completed":
        return None
    curve = {"tokens": [], "val_loss": [], "val_bpb": []}
    for line in log_file.read_text().splitlines():
        if '"val"' not in line:
            continue
        rec = json.loads(line)
        if rec.get("kind") == "val":
            curve["tokens"].append(int(rec["tokens"]))
            curve["val_loss"].append(rec["val_loss"])
            curve["val_bpb"].append(rec["val_bpb"])
    return {
        "preset": config.get("preset"), "train_preset": config.get("train_preset"), "overrides": config.get("overrides", []),
        "seed": config["train_config"].get("seed", 0),
        "tokens": summary["tokens"], "steps": summary["steps"], "seconds": summary["seconds"],
        "train_seconds": summary.get("train_seconds"), "tokens_per_second": summary.get("tokens_per_second"),
        "params": summary["params"], "val_loss": summary["val_loss"], "val_bpb": summary["val_bpb"],
        "best_val_loss": summary["best_val_loss"], "best_val_bpb": summary["best_val_bpb"],
        "train_loss": summary.get("train_loss"), "curve": curve,
        "_model": config["model_config"], "_train": config["train_config"], "_data": config.get("data_meta", {}),
    }


def changed(reference: dict, run: dict) -> dict:
    """The recipe's fields that differ between two runs: {field: [in the reference, in the run]}."""
    out = {}
    for part in ("_model", "_train"):
        a, b = reference[part], run[part]
        for field in sorted(set(a) | set(b)):
            if field in NOT_RECIPE:
                continue
            if a.get(field) != b.get(field):
                out[field] = [a.get(field), b.get(field)]
    return out


def collect(runs_dir: Path, prefixes: list[str]) -> dict:
    runs: dict[str, dict] = {}
    for folder in sorted(p for p in runs_dir.iterdir() if p.is_dir() and any(p.name.startswith(x) for x in prefixes)):
        rec = read_run(folder)
        if rec is not None:
            runs[folder.name] = rec

    # ---- the ladder: ladder-<k>-<name>-s<seed>
    rung_runs: list[dict[int, str]] = [{} for _ in RUNGS]
    for name, rec in runs.items():
        m = re.fullmatch(r"ladder-(\d+)-(.+)-s(\d+)", name)
        if m and int(m.group(1)) < len(RUNGS) and m.group(2) == RUNGS[int(m.group(1))]:
            rec.update(family="ladder", step=int(m.group(1)), name=m.group(2), group=None, seed=int(m.group(3)))
            rung_runs[rec["step"]][rec["seed"]] = name
    seeds = sorted({s for r in rung_runs for s in r})

    def any_run(k: int, seed: int | None = None) -> str | None:
        r = rung_runs[k]
        if seed is not None and seed in r:
            return r[seed]
        return r[min(r)] if r else None

    steps = []
    for k, rung in enumerate(RUNGS):
        here = any_run(k)
        before = any_run(k - 1) if k else None
        losses = [runs[rung_runs[k][s]]["val_loss"] if s in rung_runs[k] else None for s in seeds]
        bpbs = [runs[rung_runs[k][s]]["val_bpb"] if s in rung_runs[k] else None for s in seeds]
        have = [v for v in losses if v is not None]
        steps.append({
            "k": k, "name": rung, "label": LABELS[rung][0], "short": LABELS[rung][1],
            "runs": {str(s): n for s, n in sorted(rung_runs[k].items())},
            "val_loss": losses, "val_bpb": bpbs,
            "mean_val_loss": sum(have) / len(have) if have else None,
            "mean_val_bpb": sum(v for v in bpbs if v is not None) / len(have) if have else None,
            "changed": {} if k == 0 else (changed(runs[before], runs[here]) if here and before else None),
            "params": runs[here]["params"]["total"] if here else None,
        })
        for seed, name in rung_runs[k].items():
            ref = any_run(k - 1, seed) if k else None
            runs[name]["reference"] = ref
            runs[name]["changed"] = {} if k == 0 else (changed(runs[ref], runs[name]) if ref else None)
    diffs = [abs(s["val_loss"][0] - s["val_loss"][1]) for s in steps
             if len(seeds) >= 2 and s["val_loss"][0] is not None and s["val_loss"][1] is not None]
    first = any_run(0) or next((any_run(k) for k in range(len(RUNGS)) if any_run(k)), None)
    ladder = {
        "tokens": runs[first]["tokens"] if first else None, "seeds": seeds,
        "complete": all(0 in r for r in rung_runs),
        "spread": {"rungs": len(diffs), "mean_abs_diff": sum(diffs) / len(diffs) if diffs else None,
                   "max_abs_diff": max(diffs) if diffs else None, "min_abs_diff": min(diffs) if diffs else None,
                   "rms_abs_diff": (sum(d * d for d in diffs) / len(diffs)) ** 0.5 if diffs else None},
        "steps": steps,
    }

    # ---- the swaps: swap-base[-s<seed>] and swap-<what>[-s<seed>]; a run is measured against the recipe of its own seed
    def seed_of(name: str) -> tuple[str, int]:
        m = re.fullmatch(r"(.+)-s(\d+)", name)
        return (m.group(1), int(m.group(2))) if m else (name, 0)

    bases = {seed_of(n)[1]: n for n in runs if seed_of(n)[0] == "swap-base"}
    base = bases.get(0)
    families: dict[str, list[str]] = {g: [] for g in GROUPS + ("other",)}
    pairs: dict[str, dict] = {}
    for name, rec in runs.items():
        if rec.get("family") == "ladder":
            continue
        m = re.fullmatch(r"ladder-(\d+)-(.+)-w32M", name)
        if m:   # a rung of the 2019 side run again with a longer warmup: not on the stair; taken up below
            rec.update(family="ladder-w32M", step=int(m.group(1)), name=m.group(2), group=None, reference=None, changed=None)
            continue
        family = next((x.rstrip("-") for x in prefixes if name.startswith(x)), "other")
        stem, seed = seed_of(name)
        what = stem[len(family) + 1:]
        rec.update(family=family, step=None, name=what, group=None)
        if family != "swap":
            rec.update(reference=None, changed=None)
            continue
        if stem == "swap-base":
            rec.update(reference=None, changed={})
            continue
        group = what.split("-")[0] if what.split("-")[0] in GROUPS else "other"
        ref = bases.get(seed)
        rec.update(group=group, reference=ref, changed=changed(runs[ref], rec) if ref else None)
        if seed == 0:
            families[group].append(name)
        pairs.setdefault(what, {"runs": {}})["runs"][str(seed)] = name
    swap_seeds = sorted(bases)
    for p in pairs.values():
        p["val_loss"] = [runs[p["runs"][str(s)]]["val_loss"] if str(s) in p["runs"] else None for s in swap_seeds]
        p["delta"] = [runs[p["runs"][str(s)]]["val_loss"] - runs[bases[s]]["val_loss"] if str(s) in p["runs"] else None for s in swap_seeds]
        have = [d for d in p["delta"] if d is not None]
        p["mean_delta"] = sum(have) / len(have) if have else None
    # how far two seeds of one configuration lie apart under the 2026 recipe: the recipe itself and every swap run twice
    both = ([[runs[bases[s]]["val_loss"] for s in swap_seeds[:2]]] if len(swap_seeds) > 1 else []) + [p["val_loss"][:2] for p in pairs.values()]
    twice = [abs(v[0] - v[1]) for v in both if len(v) > 1 and None not in v]
    swaps = {"base": base, "bases": {str(s): n for s, n in sorted(bases.items())}, "seeds": swap_seeds,
             "tokens": runs[base]["tokens"] if base else None, "families": families, "pairs": pairs,
             "spread": {"pairs": len(twice), "min_abs_diff": min(twice) if twice else None, "max_abs_diff": max(twice) if twice else None,
                        "mean_abs_diff": sum(twice) / len(twice) if twice else None,
                        "rms_abs_diff": (sum(d * d for d in twice) / len(twice)) ** 0.5 if twice else None}}

    # ---- the 2019 side again with a longer warmup (ladder-<k>-<name>-w32M): one seed, kept off the stair
    longer = {r["step"]: n for n, r in runs.items() if r.get("family") == "ladder-w32M"}
    ladder["longer_warmup"] = None
    if longer:
        any32 = runs[next(iter(longer.values()))]
        seed32 = any32["seed"]
        for k, n in longer.items():
            ref = rung_runs[k].get(seed32)
            runs[n].update(reference=ref, changed=changed(runs[ref], runs[n]) if ref else None)
        ladder["longer_warmup"] = {
            "warmup_tokens": any32["_train"].get("warmup_tokens"), "seed": seed32, "runs": [longer.get(k) for k in range(len(RUNGS))],
            "val_loss": [runs[longer[k]]["val_loss"] if k in longer else None for k in range(len(RUNGS))],
            "gain": [runs[longer[k]]["val_loss"] - runs[rung_runs[k][seed32]]["val_loss"] if k in longer and seed32 in rung_runs[k] else None
                     for k in range(len(RUNGS))]}

    # ---- the step of rung 7, split: the rung-6 model under the 2026 training recipe with AdamW in place of Muon on
    # the hidden matrices (split-rung7-<what>[-s<seed>]), against the same seed of rungs 6 and 7; the best rate of a seed
    split = None
    between = sorted(n for n, r in runs.items() if r.get("family") == "split" and n.startswith("split-rung7-"))
    by_seed: dict[str, dict] = {}
    for seed in sorted({runs[n]["seed"] for n in between}):
        mine = [n for n in between if runs[n]["seed"] == seed]
        best = min(mine, key=lambda n: runs[n]["val_loss"])
        lo, hi = rung_runs[6].get(seed), rung_runs[7].get(seed)
        for n in mine:
            runs[n].update(reference=hi, changed=changed(runs[hi], runs[n]) if hi else None)
        if lo and hi:
            by_seed[str(seed)] = {"seed": seed, "runs": mine, "best": best, "before": runs[lo]["val_loss"],
                                  "with_adamw": runs[best]["val_loss"], "after": runs[hi]["val_loss"],
                                  "rest_of_recipe": runs[best]["val_loss"] - runs[lo]["val_loss"],
                                  "rule": runs[hi]["val_loss"] - runs[best]["val_loss"]}
    if by_seed:
        split = {"rung": 7, **by_seed[min(by_seed, key=int)], "runs": between, "by_seed": by_seed}
        lw = ladder["longer_warmup"]
        if lw and lw["val_loss"][6] is not None and str(lw["seed"]) in by_seed:   # the same step against rung 6 with the longer warmup
            mine = by_seed[str(lw["seed"])]
            split["against_longer_warmup"] = {"seed": lw["seed"], "before": lw["val_loss"][6], "step": mine["after"] - lw["val_loss"][6],
                                              "rest_of_recipe": mine["with_adamw"] - lw["val_loss"][6]}
    ladder["split"] = split

    some = next(iter(runs.values()), None)
    model = {k: some["_model"][k] for k in ("n_layer", "d_model", "n_head", "block_size", "vocab_size")} if some else None
    data = None
    if some:
        meta = some["_data"]
        data = {"dataset": meta.get("dataset"), "tokenizer": meta.get("tokenizer"),
                "bytes_per_token": meta.get("bytes_per_token_val"), "val_tokens": some["_train"].get("val_tokens")}
    for rec in runs.values():
        for key in ("_model", "_train", "_data"):
            del rec[key]
    return {"generated_by": "gptpp.export.ladder_results", "model": model, "data": data,
            "ladder": ladder, "swaps": swaps, "runs": runs}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", default=str(REPO / "results" / "runs"), help="the folder of finished runs' small copies")
    ap.add_argument("--out", default=str(DATA))
    ap.add_argument("--prefixes", nargs="+", default=["ladder-", "swap-", "split-"], help="which runs to collect, by name prefix")
    ap.add_argument("--name", default="ladder", help="the file's name, without .json")
    args = ap.parse_args()
    result = collect(Path(args.runs), args.prefixes)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    text = json.dumps(round_sig(result), separators=(",", ":"), ensure_ascii=False)
    write_atomic(out / f"{args.name}.json", text.encode("utf-8"))
    lad = result["ladder"]
    print(f"{out / (args.name + '.json')}: {len(text.encode('utf-8'))} bytes, {len(result['runs'])} finished runs")
    for s in lad["steps"]:
        loss = " ".join("   n/a " if v is None else f"{v:.4f}" for v in s["val_loss"]) or "   n/a "
        what = "" if not s["changed"] else ", ".join(f"{k}: {a} -> {b}" for k, (a, b) in list(s["changed"].items())[:4])
        print(f"  rung {s['k']} {s['name']:<10} {loss}   {what}")
    print(f"  seeds {lad['seeds']}, complete {lad['complete']}, spread {lad['spread']}")
    for group, names in result["swaps"]["families"].items():
        for n in names:
            r = result["runs"][n]
            print(f"  swap {group:<6} {n:<26} {r['val_loss']:.4f}   {r['changed']}")


if __name__ == "__main__":
    main()
