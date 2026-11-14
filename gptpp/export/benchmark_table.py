"""Collect every model's scores under the evaluation protocol of `docs/EVAL.md` into the one file the figure
`gpt-benchmark-table` (fig-07-02) reads: `assets/data/gpt/benchmark-table.json`.

    python -m gptpp.export.benchmark_table [--eval results/eval] [--runs results/runs] [--out <dir>]
                                           [--dev-ours <result file>]

Nothing is measured here: every number is copied from the result files that `gptpp.evaluate`, `gptpp.benchmarks`,
`scripts/eval_released_models.py` and `scripts/eval_openrouter.py` wrote, from the trainer's own summaries, or (the
rows reported by others) from the research dossiers, where each value is checked to stand as printed. It takes what
exists and can be run again when more lands:

    our models      `results/eval/gptpp-base.json` is gpt-pp (the main run). Of `results/eval/race-2019-*.json` the
                    run with the lowest final held-out loss in `results/runs/<run>/summary.json` is "the 2019
                    recipe"; `results/eval/race-2026-*.json` is the 2026 recipe on the same budget. What is not
                    there yet is listed under `missing`, and the figure says so.
    --dev-ours F    development only: a checkpoint of the main run scored into a scratch file F (never in
                    results/eval); it stands in gpt-pp's place, marked `final: false` with its tokens.

The file, field by field:

    generated_by     "gptpp.export.benchmark_table"
    sets[k]          the held-out sets, in the figure's order: {key, what (as summary.json words it), docs, bytes}
    tasks[k]         the benchmarks: {key, what, n (items), chance, rule ("per byte" | "total" | "first choice": which
                     option is credited, EVAL.md section 3)}
    models[k]        every model scored locally, ours first:
      key              the result file's name
      label            its name in the figure ("gpt-pp", "GPT-2 small", ...)
      group            "ours" | "released"
      role             for ours: "main" (gpt-pp) | "recipe-2019" | "recipe-2026"; null for released models
      final            false only for a development stand-in (--dev-ours)
      year, layers, width, window
      params           every parameter, a tied matrix counted once
      params_trunk     without the token embedding (and without the unembedding where it is a separate matrix)
      tokens           training tokens: for ours from the run's own files; for a released model the count its model
                       card states for that model, else null
      tokens_source    the words the count was read from (or why there is none)
      bpb              {set: [bits per byte, standard error over documents]}
      acc              {task: [accuracy, standard error over items]}
      centered         mean over the nine tasks of (accuracy - chance) / (1 - chance); null if a task is missing
      seen             {set: "train" (the set's distribution is its training data) | "possible" (it may have read
                       such text) | "no"}, with `seen_words` the sentence of the result file it was read from
      subset           {task: [accuracy, standard error, n]} on the items the hosted models were asked
      monitor          for ours: the trainer's own held-out numbers {val_loss, val_bpb, best_val_loss, best_val_bpb,
                       tokens, seconds, n_gpu} from summary.json (null while the run has no summary)
    hosted[k]        the hosted models: {key, label, tier, acc: {task: [accuracy, standard error, n]}, cost (USD),
                     complete (it answered the whole subset)}
    subset           {task: size} of the items the hosted models were asked: the first of a fixed order
    reported[k]      rows reported by others, not measured here, each by its own protocol (the rows of the post's
                     table of what nanoGPT and nanochat report): {who, shape, text, result}
    paired[k]        two models on the same documents and the same items: {a, b (keys of `models`), bpb: {set:
                     [a minus b in bits per byte, its paired standard error, the share of documents on which a is
                     better]}, acc: {task: [a minus b in accuracy, its paired standard error]}}: each of ours against
                     GPT-2 small and SmolLM2 135M, and the 2026 recipe against the 2019 recipe at the same budget
    missing          which of "gptpp-base", "recipe-2019", "recipe-2026" have no result file yet

Values keep five significant digits.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from .. import repo_dir
from ..benchmarks import fixed_subset
from ..evaluate import ratio_standard_error
from .common import DATA, WORK, write_json

NAME = "benchmark-table"
SETS = ["wikitext103", "tinystories", "fineweb-edu", "fresh"]          # the order of the post's table
TASKS = ["hellaswag", "blimp", "arc_easy", "arc_challenge", "lambada", "piqa", "winogrande", "sciq", "commonsense_qa"]
RULES = {"acc_norm": "per byte", "acc": "total", "greedy": "first choice"}
OURS = {"main": "gpt-pp", "recipe-2019": "same shape, 2019 recipe", "recipe-2026": "same shape, 2026 recipe"}
RESEARCH = WORK / "research"
# What a model of ours, trained on FineWeb-EDU alone, can have read of each held-out set.
OUR_EXPOSURE = {
    "fineweb-edu": ("train", "yes (trained on FineWeb-EDU; these documents, and any with their text or URL, were held out of its training set)"),
    "wikitext103": ("possible", "possible (FineWeb-EDU is filtered from a web crawl, which holds Wikipedia pages)"),
    "tinystories": ("possible", "possible (a web crawl up to 2024; not checked)"),
    "fresh": ("no", "no (published in October and November 2026, after the crawl)"),
}

# Rows reported by others. Each value must stand, as printed here, in the dossier that read the source: `evidence`
# is (dossier file, a line or phrase of it). Nothing here was measured by this repository.
REPORTED = [
    {"who": "nanoGPT, 2023: GPT-2 small rebuilt", "shape": "12 × 768, 124M", "text": "OpenWebText, about 300B tokens",
     "result": "2.85 nats per token on its validation text, where released GPT-2 small scores 3.12",
     "evidence": [("nanogpt-vs-gpt2.md", "| GPT-2 124M reproduction, OpenWebText validation loss | \"~2.85\" |"),
                  ("nanogpt-vs-gpt2.md", "| OpenAI `gpt2` (124M) on OWT, train / val | 3.11 / 3.12 |"),
                  ("nanogpt-vs-gpt2.md", "600,000 iterations = 294.9B")]},
    {"who": "its successor of 2024", "shape": "12 × 768, 124M", "text": "FineWeb-EDU, 10B tokens",
     "result": "HellaSwag above released GPT-2 small (no figure given); 33.24% after 40B tokens",
     "evidence": [("nanogpt-vs-gpt2.md", "19,073 steps × 524,288 tokens = 10.0B tokens"),
                  ("nanogpt-vs-gpt2.md", "| HellaSwag after 40B tokens (4 passes, step about 76,290) | 33.24% |")]},
    {"who": "nanochat, first release, October 2025", "shape": "20 × 1,280, 561M", "text": "FineWeb-EDU, 11.2B tokens",
     "result": "CORE 0.2219; about 0.81 bits per byte",
     "evidence": [("nanochat.md", "| parameters | 560,988,160 |"), ("nanochat.md", "| training tokens | 11.2B |"),
                  ("nanochat.md", "| validation bpb | about 0.81 |"), ("nanochat.md", "| CORE (base) | 0.2219 |")]},
    {"who": "nanochat, the 12-layer model of January 2026", "shape": "12 × 768, 135M", "text": "FineWeb-EDU, 1.08B tokens",
     "result": "CORE 0.1059; 0.9825 bits per byte",
     "evidence": [("nanochat.md", "| 12 | 768 | 135 | 1.08 | 0.9825 | 0.1059 | 7.8 |")]},
    {"who": "nanochat, record of 14 March 2026", "shape": "24 × 1,536, 1,384M", "text": "a searched mixture, about 5.8B tokens",
     "result": "CORE 0.2626, where released GPT-2 XL scores 0.2565; 0.718 bits per byte",
     "evidence": [("nanochat.md", "| 6 | 14 Mar 2026 | agent round 2 | 8 | 5,568 (our arithmetic) | 1.65 h (99 min) | 0.71800 | 0.262634 (mean of 5) | $40 |"),
                  ("nanochat.md", "| d24 | 24 | 1536 | 12 | 50,331,648 | 603,979,776 (12) | 50,331,648 | 679,478,976 | 1,384,122,122 | 4.78e9 |"),
                  ("nanochat.md", "5.84B tokens in 5,940 s"), ("nanochat.md", "0.256525")]},
]


def exposure(words: str) -> str:
    """What a result file says a model can have read of a held-out set, as one of three words."""
    w = words.strip().lower()
    if w.startswith("yes") or "is its only data" in w:
        return "train"
    if w.startswith("no") and not w.startswith("not stated"):
        return "no"
    return "possible"


def tokens_on_card(words: str) -> tuple[int | None, str]:
    """The training tokens a model card states for the model itself, with the words they were read from."""
    if re.search(r"not (stated|disclosed)", words):
        return None, "not stated for this model"
    m = re.search(r"(?:about )?([\d.]+) (billion|trillion) tokens", words)
    if not m:
        return None, "not stated"
    return int(round(float(m.group(1)) * {"billion": 1e9, "trillion": 1e12}[m.group(2)])), m.group(0)


def centered(acc: dict, chance: dict) -> float | None:
    if any(t not in acc for t in TASKS):
        return None
    return sum((acc[t][0] - chance[t]) / (1 - chance[t]) for t in TASKS) / len(TASKS)


def on_subset(record: dict, sizes: dict) -> dict:
    """A local model's accuracy on the items the hosted models were asked, from its per-item marks."""
    out = {}
    for task, size in sizes.items():
        done = record.get("benchmarks", {}).get(task)
        if not done or done.get("limit") is not None or "correct" not in done:
            continue
        marks = done["correct"]
        picked = [marks[i] == "1" for i in fixed_subset(task, len(marks), size)]
        acc = sum(picked) / len(picked)
        out[task] = [acc, math.sqrt(acc * (1 - acc) / len(picked)), len(picked)]
    return out


def paired(a: dict, b: dict) -> tuple[dict, dict]:
    """Model a minus model b on the same documents and items, as `scripts/eval_released_models.py --compare` does."""
    import numpy as np

    bpb, acc = {}, {}
    for name in SETS:
        ra, rb = a.get("bits_per_byte", {}).get(name), b.get("bits_per_byte", {}).get(name)
        if ra is None or rb is None or ra["docs"]["sha256"] != rb["docs"]["sha256"]:
            continue
        nbytes = np.array(ra["doc_bytes"], dtype=np.float64)
        diff = np.array(ra["doc_bits"]) - np.array(rb["doc_bits"])
        bpb[name] = [float(diff.sum() / nbytes.sum()), float(ratio_standard_error(diff, nbytes)), float(np.mean(diff < 0))]
    for name in TASKS:
        ra, rb = a.get("benchmarks", {}).get(name), b.get("benchmarks", {}).get(name)
        if ra is None or rb is None or len(ra["correct"]) != len(rb["correct"]) or ra.get("limit") is not None or rb.get("limit") is not None:
            continue
        diff = np.array([int(x) - int(y) for x, y in zip(ra["correct"], rb["correct"])], dtype=np.float64)
        acc[name] = [float(diff.mean()), float(diff.std(ddof=1) / np.sqrt(len(diff)))]
    return bpb, acc


def scores(record: dict) -> tuple[dict, dict]:
    bpb = {s: [r["bits_per_byte"], r["bits_per_byte_se"]] for s, r in record.get("bits_per_byte", {}).items() if s in SETS}
    acc = {t: [r["accuracy"], r.get("stderr", r.get(r.get("headline", "") + "_se"))]
           for t, r in record.get("benchmarks", {}).items() if t in TASKS and r.get("limit") is None}
    return bpb, acc


def run_facts(runs: Path, run: str) -> dict:
    """What the trainer's own files say of a run of ours: tokens, and its monitor when the run has ended."""
    out = {"tokens": None, "monitor": None}
    summary, config = runs / run / "summary.json", runs / run / "config.json"
    if not config.exists():
        config = WORK / "ckpt" / run / "config.json"
    if summary.exists():
        s = json.loads(summary.read_text())
        out["tokens"] = int(s["tokens"])
        out["monitor"] = {k: s.get(k) for k in ("val_loss", "val_bpb", "best_val_loss", "best_val_bpb", "tokens", "seconds", "n_gpu")}
    elif config.exists():
        out["tokens"] = int(json.loads(config.read_text())["derived"]["tokens"])
    return out


def our_model(record: dict, key: str, role: str, runs: Path, sizes: dict, chance: dict, final: bool = True) -> dict:
    info = record["model"]
    run = info["name"].split("/")[0]
    facts = run_facts(runs, run)
    parts = info["n_params_by_part"]
    tokens, source = facts["tokens"], "the run's own budget"
    if not final:  # a checkpoint: the tokens it had seen, from the run's configuration
        cfg = json.loads((WORK / "ckpt" / run / "config.json").read_text())
        step = int(re.search(r"ckpt_(\d+)", info["checkpoint"]).group(1))
        tokens, source = step * int(cfg["derived"]["tokens_per_step"]), f"checkpoint at step {step} of {cfg['derived']['steps']}"
    bpb, acc = scores(record)
    label = OURS[role] if final else "gpt-pp, unfinished"
    return {"key": key, "label": label, "group": "ours", "role": role, "final": final, "year": 2026,
            "layers": info["n_layer"], "width": info["d_model"], "window": min(info["context"], 1024),
            "params": parts["total"], "params_trunk": parts["total"] - parts["embedding"] - parts["unembedding"],
            "tokens": tokens, "tokens_source": source, "bpb": bpb, "acc": acc, "centered": centered(acc, chance),
            "seen": {s: OUR_EXPOSURE[s][0] for s in SETS}, "seen_words": {s: OUR_EXPOSURE[s][1] for s in SETS},
            "subset": on_subset(record, sizes), "monitor": facts["monitor"] if final else None}


def released_model(record: dict, key: str, sizes: dict, chance: dict) -> dict:
    info = record["model"]
    tokens, source = tokens_on_card(info.get("training_data", ""))
    bpb, acc = scores(record)
    return {"key": key, "label": info.get("label", info["name"]), "group": "released", "role": None, "final": True,
            "year": info.get("year"), "layers": info["n_layer"], "width": info["d_model"], "window": info.get("window", min(info["context"], 1024)),
            "params": info["n_params"], "params_trunk": info["n_params_non_embedding"],
            "tokens": tokens, "tokens_source": source, "bpb": bpb, "acc": acc, "centered": centered(acc, chance),
            "seen": {s: exposure(info["seen"][s]) for s in SETS if s in info.get("seen", {})},
            "seen_words": {s: info["seen"][s] for s in SETS if s in info.get("seen", {})},
            "subset": on_subset(record, sizes), "monitor": None}


def check_reported() -> list[str]:
    """Every reported value stands in its dossier as printed here; returns what could not be checked."""
    problems = []
    for file, phrase in [e for r in REPORTED for e in r["evidence"]]:
        path = RESEARCH / file
        if not path.exists():
            problems.append(f"{file} is not here: {phrase!r} was not checked")
        elif phrase not in path.read_text(encoding="utf-8"):
            raise SystemExit(f"a reported value does not stand in {file} as printed: {phrase!r}")
    return problems


def export(eval_dir: Path, runs: Path, dev_ours: Path | None = None) -> dict:
    summary = json.loads((eval_dir / "summary.json").read_text())
    sets = [{"key": s, "what": summary["held_out"][s]["what"], "docs": summary["held_out"][s]["n_docs"],
             "bytes": summary["held_out"][s]["n_bytes"]} for s in SETS]
    chance = {t: summary["tasks"][t]["chance"] for t in TASKS}
    tasks = [{"key": t, "what": summary["tasks"][t]["what"], "n": summary["tasks"][t]["n"], "chance": chance[t],
              "rule": RULES[summary["tasks"][t]["headline"]]} for t in TASKS]

    # the hosted models, and the size of the subset they were asked
    hosted, sizes = [], {}
    for path in sorted((eval_dir / "hosted").glob("*.json")):
        if path.name.startswith("run-"):
            continue
        r = json.loads(path.read_text())
        acc = {t: [v["accuracy"], v["stderr"], v["n"]] for t, v in r["results"].items()}
        hosted.append({"key": path.stem, "label": r["model"]["name"].split(": ")[-1], "id": r["model"]["id"], "tier": r["model"]["tier"],
                       "acc": acc, "cost": r["cost_usd"], "indices": {t: v["indices"] for t, v in r["results"].items()}})
        for t, v in acc.items():
            sizes[t] = max(sizes.get(t, 0), v[2])
    for h in hosted:  # every model was asked items of the fixed subset, and a complete one all of them
        h["complete"] = all(h["acc"].get(t, [0, 0, 0])[2] == sizes[t] for t in sizes)
        for t, idx in h.pop("indices").items():
            subset = set(fixed_subset(t, summary["tasks"][t]["n"], sizes[t]))
            if not set(idx) <= subset or (h["complete"] and set(idx) != subset) or len(set(idx)) != len(idx):
                raise SystemExit(f"{h['key']}: its {t} items are not the fixed subset of {sizes[t]}")
    listed = re.findall(r'\{"id": "([^"]+)"', (repo_dir() / "scripts" / "eval_openrouter.py").read_text())   # the order they were chosen in
    hosted.sort(key=lambda h: (listed.index(h["id"]) if h["id"] in listed else len(listed), h["label"]))

    released, ours_files = [], {}
    for path in sorted(eval_dir.glob("*.json")):
        record = json.loads(path.read_text())
        if "model" not in record or not ("bits_per_byte" in record or "benchmarks" in record):
            continue
        if record["model"].get("kind") == "gptpp":
            ours_files[path.stem] = record
        else:
            released.append((path.stem, record))
    rank = {m["file"]: i for i, m in enumerate(summary["models"])}
    released.sort(key=lambda kr: rank.get(kr[0], len(rank)))

    models, missing, records = [], [], dict(released)
    if dev_ours is not None:
        records[dev_ours.stem] = json.loads(dev_ours.read_text())
        models.append(our_model(records[dev_ours.stem], dev_ours.stem, "main", runs, sizes, chance, final=False))
    elif "gptpp-base" in ours_files:
        records["gptpp-base"] = ours_files["gptpp-base"]
        models.append(our_model(ours_files["gptpp-base"], "gptpp-base", "main", runs, sizes, chance))
    else:
        missing.append("gptpp-base")
    race19 = {k: r for k, r in ours_files.items() if k.startswith("race-2019-")}
    if race19:  # the best of the 2019 recipe's learning rates, by the trainer's final held-out loss
        def final_loss(key: str) -> float:
            s = runs / key / "summary.json"
            return json.loads(s.read_text())["val_loss"] if s.exists() else float("inf")
        best = min(race19, key=final_loss)
        records[best] = race19[best]
        models.append(our_model(race19[best], best, "recipe-2019", runs, sizes, chance))
    else:
        missing.append("recipe-2019")
    race26 = sorted(k for k in ours_files if k.startswith("race-2026-"))
    if race26:
        records[race26[0]] = ours_files[race26[0]]
        models.append(our_model(ours_files[race26[0]], race26[0], "recipe-2026", runs, sizes, chance))
    else:
        missing.append("recipe-2026")
    for m in models:
        if m["role"] in ("recipe-2019", "recipe-2026") and m["tokens"]:
            m["label"] += f", {m['tokens'] / 1e6:.0f}M tokens"
    pairs = [(m["key"], other) for m in models for other in ("gpt2", "smollm2-135m") if other in records]
    by_role = {m["role"]: m["key"] for m in models}
    if "recipe-2026" in by_role and "recipe-2019" in by_role:
        pairs.append((by_role["recipe-2026"], by_role["recipe-2019"]))
    compared = []
    for a, b in pairs:
        bpb, acc = paired(records[a], records[b])
        compared.append({"a": a, "b": b, "bpb": bpb, "acc": acc})
    models += [released_model(record, key, sizes, chance) for key, record in released]

    unchecked = check_reported()
    return {"generated_by": "gptpp.export.benchmark_table", "sets": sets, "tasks": tasks, "models": models, "hosted": hosted,
            "subset": sizes, "reported": [{k: v for k, v in r.items() if k != "evidence"} for r in REPORTED],
            "reported_unchecked": unchecked, "paired": compared, "missing": missing}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eval", default=str(repo_dir() / "results" / "eval"), help="the folder of the result files")
    parser.add_argument("--runs", default=str(repo_dir() / "results" / "runs"), help="the small copies of our runs")
    parser.add_argument("--out", default=str(DATA), help="the site's data folder")
    parser.add_argument("--dev-ours", default=None, help="development only: a scratch result file of a checkpoint of the main run")
    args = parser.parse_args(argv)
    data = export(Path(args.eval), Path(args.runs), Path(args.dev_ours) if args.dev_ours else None)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    size = write_json(out / f"{NAME}.json", data)
    for m in data["models"]:
        print(f"{m['label']:<34} {m['group']:<9} {m['params'] / 1e6:8.1f}M  tokens {m['tokens'] if m['tokens'] else '-':>14}  "
              f"fineweb-edu {m['bpb'].get('fineweb-edu', [float('nan')])[0]:.4f}  centered {m['centered'] if m['centered'] is not None else float('nan'):.4f}")
    print(f"hosted: {', '.join(h['label'] + ('' if h['complete'] else ' (incomplete)') for h in data['hosted'])}; subset {data['subset']}")
    print(f"missing: {data['missing'] or 'nothing'}; {NAME}.json {size} bytes" + (f"; not checked: {data['reported_unchecked']}" if data["reported_unchecked"] else ""))


if __name__ == "__main__":
    main()
