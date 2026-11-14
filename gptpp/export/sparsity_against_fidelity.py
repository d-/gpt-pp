"""Export the scores of a family of sparse dictionaries: what fig-10-04 (`gpt-sparsity-against-fidelity`) draws.

    python -m gptpp.export.sparsity_against_fidelity --run <run dir> --layer 2 --used topk-n4096-k16 [--out <dir>]

Reads the records that `gptpp.interp.sae` wrote beside the dictionaries of that run and layer (one `<name>.json`
each: how it was trained, what it scored on held-out text) and writes them, by model, into
`sparsity-against-fidelity.json`. The file holds one entry per model and running the exporter for one model leaves
the other's entry as it is, so that pico and gpt-pp can be exported at different times:

    { "models": { "pico": { label, layers, width, after (blocks before the layer read), dataset, checkpoint,
                            train_tokens, eval_tokens, splice_targets, loss_clean, loss_zero, loss_mean, used,
                            dicts: [{ name, kind, n, k, l1, l0, fvu, fvu_rescaled, gain, loss_spliced,
                                      loss_recovered, loss_recovered_mean, dead, dense }] },
                  "gpt-pp": { ... } },
      "order": ["pico", "gpt-pp"] }
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gptpp.export.common import DATA, write_json
from gptpp.interp import sae as S

NAME = "sparsity-against-fidelity"
DATASETS = {"tinystories": "TinyStories", "fineweb-edu": "FineWeb-EDU"}
KEEP = ("name", "kind", "n", "k", "l1", "l0", "fvu", "fvu_rescaled", "gain", "loss_spliced", "loss_recovered",
        "loss_recovered_mean", "dead", "dense")


def model_label(run_dir: Path) -> str:
    """The name the series gives the model of a run: `pico` for the pico preset, `gpt-pp` for the base size."""
    preset = json.loads((run_dir / "config.json").read_text()).get("preset", "")
    return "pico" if preset.endswith("pico") else "gpt-pp" if preset.endswith("base") else run_dir.name


def family(run_dir: Path, layer: int, used: str, folder: Path | None = None) -> dict:
    """One model's entry: the facts of the run and of the evaluation, and one row per dictionary."""
    config = json.loads((run_dir / "config.json").read_text())
    folder = folder or S.dict_dir(run_dir.name, layer)
    records = [json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))]
    if not records:
        raise FileNotFoundError(f"no dictionary records in {folder}")
    if used not in {r["eval"]["name"] for r in records}:
        raise KeyError(f"{used} is not a dictionary of {folder}")
    first = records[0]["eval"]
    order = {"topk": 0, "relu": 1}
    rows = sorted((r["eval"] for r in records), key=lambda e: (order[e["kind"]], e["n"], e["k"], -e["l1"]))
    return {
        "label": model_label(run_dir), "layers": config["model_config"]["n_layer"], "width": config["model_config"]["d_model"],
        "after": layer + 1, "dataset": DATASETS.get(records[0]["dataset"], records[0]["dataset"]),
        "checkpoint": records[0]["checkpoint"], "train_tokens": records[0]["train_tokens"],
        "eval_tokens": first["eval_tokens"], "splice_targets": first["splice_targets"],
        "loss_clean": first["loss_clean"], "loss_zero": first["loss_zero"], "loss_mean": first["loss_mean"],
        "used": used, "dicts": [{k: e[k] for k in KEEP} for e in rows],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="a run directory, or a run's name under $GPTPP_WORK/ckpt")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--used", required=True, help="the dictionary the other figures use, e.g. topk-n4096-k16")
    ap.add_argument("--dicts", default=None, help="the folder of the dictionaries (default: $GPTPP_WORK/sae/<run>-L<layer>)")
    ap.add_argument("--out", default=str(DATA))
    args = ap.parse_args()
    run_dir = Path(args.run) if Path(args.run).is_dir() else S.work_dir() / "ckpt" / args.run
    entry = family(run_dir, args.layer, args.used, Path(args.dicts) if args.dicts else None)
    path = Path(args.out) / f"{NAME}.json"
    doc = json.loads(path.read_text()) if path.exists() else {"models": {}}
    doc["models"][entry["label"]] = entry
    doc["order"] = [m for m in ("pico", "gpt-pp") if m in doc["models"]] + sorted(set(doc["models"]) - {"pico", "gpt-pp"})
    path.parent.mkdir(parents=True, exist_ok=True)
    size = write_json(path, doc)
    print(f"{path.name}: {entry['label']}, {len(entry['dicts'])} dictionaries, used {args.used}; {size / 1e3:.1f} kB")


if __name__ == "__main__":
    main()
