"""The one picture of the result, for the site's figure `gpt-headline-benchmark` (root page): bits per byte on
held-out text against model size, for gpt-pp, for the 2019 recipe at the same shape, and for the released models
scored under the same protocol.

    python -m gptpp.export.headline_benchmark [--eval results/eval] [--runs results/runs] [--out <dir>]

Nothing is measured here. The released models and, once they exist, our own are read from the evaluation records
(`results/eval/<model>.json`, written by `gptpp.evaluate`: bits per byte on each held-out set with its standard
error over documents). Our models are found by the run their checkpoint came from: `gptpp-base` (gpt-pp), `race-2019-*` (the 2019
recipe at gpt-pp's shape, 100M tokens: the rate with the lowest loss is taken) and `race-2026-*` (the 2026 recipe on
the same 100M tokens).

Until a run of ours has an evaluation record, its point is PROVISIONAL: the held-out loss the trainer itself
measured at its last evaluation (the fixed held-out batches of FineWeb-EDU; `results/runs/<run>/` if the run has
finished, else `$GPTPP_WORK/ckpt/<run>/`), given for the set "fineweb-edu" only and marked `"provisional": true`
with the tokens seen so far. The trainer's number is not the protocol's (other windows, other documents of the same
held-out split), so it is replaced as soon as the record exists: run this script again after
`python -m gptpp.evaluate --model $GPTPP_WORK/ckpt/gptpp-base --docs fineweb-edu,wikitext103,tinystories,fresh
--out results/eval/gptpp-base.json`.

The file (`headline-benchmark.json`), field by field:

    generated_by   "gptpp.export.headline_benchmark"
    sets           the held-out sets in the order shown: [{key, label, what, n_docs, n_bytes}]
    models         one entry a model, the released ones by size, then ours:
      key            the record's name ("gpt2", "gptpp-base", ...)
      label          "GPT-2 small", "gpt-pp", "the 2019 recipe", ...
      role           "released" | "gptpp" | "recipe2019" | "recipe2026"
      family         the released family a line joins ("GPT-2", "SmolLM2"), or null
      year, params, layers, width, window
      bpb            {set: [bits per byte, standard error or null]} for the sets the model is scored on
      provisional    true when `bpb` is the trainer's own number (see above); false otherwise
      seen           {set: whether the model may have seen that text in training, in words} (from the records)
      trained        ours only: {tokens, budget, cards, card_hours, status ("completed" | "running" | "stopped")}
    notes          what is provisional or missing, in words (also printed)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .. import repo_dir
from .common import CKPT, DATA, write_json

NAME = "headline-benchmark"
SETS = [("fineweb-edu", "FineWeb-EDU"), ("wikitext103", "WikiText-103"), ("tinystories", "TinyStories"), ("fresh", "prose of late 2026")]
OURS = [("gptpp-base", "gptpp", "gpt-pp"), ("race-2019", "recipe2019", "the 2019 recipe, 100M tokens"),
        ("race-2026", "recipe2026", "the 2026 recipe, 100M tokens")]
OWN_SEEN = {"fineweb-edu": "no: other documents of the corpus it was trained on", "wikitext103": "no", "tinystories": "no", "fresh": "no"}


def sig(v, digits: int = 5):
    return None if v is None else float(f"{v:.{digits}g}")


def records(folder: Path) -> list[dict]:
    out = []
    for path in sorted(folder.glob("*.json")):
        r = json.loads(path.read_text())
        if "model" in r and "bits_per_byte" in r:
            r["file"] = path.stem
            out.append(r)
    return out


def run_state(name: str, runs: Path) -> dict | None:
    """What the trainer itself knows of a run: its last evaluation, tokens, cards, time; None if it never started."""
    for folder in (runs / name, CKPT / name):
        if not (folder / "log.jsonl").exists():
            continue
        config = json.loads((folder / "config.json").read_text())
        summary = json.loads((folder / "summary.json").read_text()) if (folder / "summary.json").exists() else None
        if summary and summary.get("status") != "completed":
            summary = None
        rows = []
        for line in (folder / "log.jsonl").read_text().splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    break
        vals = [r for r in rows if r.get("kind") == "val" and r.get("val_bpb") is not None and r.get("tokens", 0) > 0]
        if not vals:
            return None
        d, m = config["derived"], config["model_config"]
        cards = (summary or {}).get("n_gpu", d.get("n_gpu", 1))
        seconds = summary["seconds"] if summary else max((r.get("time", 0.0) for r in rows), default=0.0)
        params = summary["params"]["total"] if summary else None
        if params is None:
            from ..config import ModelConfig
            from ..model import GPT

            params = GPT(ModelConfig.from_dict(m)).num_params()["total"]
        error = any(r.get("kind") == "error" for r in rows)
        return {"bpb": vals[-1]["val_bpb"], "tokens": summary["tokens"] if summary else vals[-1]["tokens"], "budget": d["tokens"], "cards": cards,
                "card_hours": sig(cards * seconds / 3600, 4), "status": "completed" if summary else "stopped" if error else "running",
                "params": params, "layers": m["n_layer"], "width": m["d_model"], "window": m["block_size"]}
    return None


def build(eval_dir: Path, runs: Path) -> dict:
    recs, notes = records(eval_dir), []
    present = [s for s, _ in SETS if any(s in r["bits_per_byte"] for r in recs)]
    first = lambda s: next(r["bits_per_byte"][s] for r in recs if s in r["bits_per_byte"])
    sets = [{"key": s, "label": lab, "what": first(s)["docs"]["what"], "n_docs": first(s)["n_docs"], "n_bytes": first(s)["n_bytes"]}
            for s, lab in SETS if s in present]
    released = [r for r in recs if r["model"].get("kind") != "gptpp"]
    counts = {}
    for r in released:
        fam = r["model"].get("label", r["file"]).split(" ")[0]
        counts[fam] = counts.get(fam, 0) + 1
    models = []
    for r in sorted(released, key=lambda r: r["model"]["n_params"]):
        m = r["model"]
        fam = m.get("label", r["file"]).split(" ")[0]
        models.append({"key": r["file"], "label": m.get("label", r["file"]), "role": "released", "family": fam if counts[fam] > 1 else None,
                       "year": m.get("year"), "params": m["n_params"], "layers": m.get("n_layer"), "width": m.get("d_model"), "window": m["window"],
                       "bpb": {s: [sig(b["bits_per_byte"]), sig(b["bits_per_byte_se"], 3)] for s, b in r["bits_per_byte"].items() if s in present},
                       "provisional": False, "seen": {s: v for s, v in (m.get("seen") or {}).items() if s in present}, "trained": None})
    # a record of ours belongs to the run its checkpoint came from ("<run>/model.pt"), whatever the file is called
    ours = {(r["model"].get("checkpoint") or r["file"]).split("/")[0]: r for r in recs if r["model"].get("kind") == "gptpp"}
    names = sorted(set(ours) | {p.name for p in runs.iterdir() if p.is_dir()} | ({p.name for p in CKPT.iterdir() if p.is_dir()} if CKPT.exists() else set()))
    for prefix, role, label in OURS:
        best = None
        for name in [n for n in names if n == prefix or (role != "gptpp" and n.startswith(prefix + "-"))]:   # gpt-pp is one run
            state = run_state(name, runs)
            rec = ours.get(name)
            if state is None and rec is None:
                continue
            if rec is not None:
                bpb = {s: [sig(b["bits_per_byte"]), sig(b["bits_per_byte_se"], 3)] for s, b in rec["bits_per_byte"].items() if s in present}
                provisional = False
            else:
                bpb, provisional = {"fineweb-edu": [sig(state["bpb"]), None]}, True
            score = bpb.get("fineweb-edu", [float("inf")])[0]
            entry = {"key": name, "label": label, "role": role, "family": None, "year": 2026,
                     "params": (rec["model"]["n_params"] if rec else state["params"]), "layers": (state or {}).get("layers") or rec["model"].get("n_layer"),
                     "width": (state or {}).get("width") or rec["model"].get("d_model"), "window": (rec["model"].get("window") if rec and rec["model"].get("window") else state["window"] if state else None),
                     "bpb": bpb, "provisional": provisional, "seen": dict(OWN_SEEN),
                     "trained": None if state is None else {k: state[k] for k in ("tokens", "budget", "cards", "card_hours", "status")}}
            if best is None or score < best[0]:
                best = (score, entry)
        if best is None:
            notes.append(f"{label}: no run of it has an evaluation yet; it is not in the file")
            continue
        e = best[1]
        if e["provisional"]:
            notes.append(f"{label} ({e['key']}): PROVISIONAL, the trainer's own held-out loss after {e['trained']['tokens'] / 1e6:.0f}M tokens "
                         f"({e['trained']['status']}); no evaluation record yet")
        elif e["trained"] and e["trained"]["status"] != "completed":
            notes.append(f"{label} ({e['key']}): scored from a checkpoint; the run is {e['trained']['status']}")
        models.append(e)
    return {"generated_by": "gptpp.export.headline_benchmark", "sets": sets, "models": models, "notes": notes}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--eval", default=str(repo_dir() / "results" / "eval"))
    ap.add_argument("--runs", default=str(repo_dir() / "results" / "runs"))
    ap.add_argument("--out", default=str(DATA))
    args = ap.parse_args()
    data = build(Path(args.eval), Path(args.runs))
    size = write_json(Path(args.out) / f"{NAME}.json", data)
    print(f"{NAME}.json: {len(data['models'])} models on {len(data['sets'])} sets, {size:,} bytes")
    for m in data["models"]:
        print(f"  {m['role']:10} {m['label'][:30]:30} {m['params'] / 1e6:7.0f}M  " +
              "  ".join(f"{s['key'][:6]} {m['bpb'][s['key']][0]:.3f}" if s["key"] in m["bpb"] else f"{s['key'][:6]}   -  " for s in data["sets"]) +
              ("  PROVISIONAL" if m["provisional"] else "") + (f"  {m['trained']['card_hours']} card-hours, {m['trained']['status']}" if m["trained"] else ""))
    for note in data["notes"]:
        print("  note: " + note)


if __name__ == "__main__":
    main()
