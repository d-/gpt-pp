"""A run's curves, read from its log: what the exporters of the optimization figures share.

    run = read_run("swap-base")        # None if the run has not started
    run["status"]                      # "completed" | "stopped" (the log ends with an error record) | "running"
    run["train"]                       # {"tokens": [...], "loss": [...], "lr_mult": [...], "step": [...]}
    run["val"]                         # {"tokens": [...], "loss": [...]}

The trainer writes a training record every `log_every` steps: the mean loss of the steps since the previous
record. Every batch is text the model has not seen (one pass over the data), so this is a held-out loss too,
measured on the tokens about to be trained on; it is attributed here to the middle of the steps it averages
over. The loss on the fixed held-out set is measured far less often (every `eval_tokens`).

A run is looked for in the workspace ($GPTPP_WORK/ckpt/<run>/, the full log) and then in the repository
(results/runs/<run>/, the copy the trainer leaves when a run completes).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .. import repo_dir
from .common import CKPT


def run_dir(name: str) -> Optional[Path]:
    for folder in (CKPT / name, repo_dir() / "results" / "runs" / name):
        if (folder / "log.jsonl").exists() and (folder / "config.json").exists():
            return folder
    return None


def read_run(name: str, max_tokens: Optional[float] = None) -> Optional[dict]:
    """The run's configuration, status and curves; with max_tokens, only the records up to that many tokens."""
    folder = run_dir(name)
    if folder is None:
        return None
    config = json.loads((folder / "config.json").read_text())
    records = []
    for line in (folder / "log.jsonl").read_text().splitlines():
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:  # a line being written while we read
                break
    summary_file = folder / "summary.json"
    summary = json.loads(summary_file.read_text()) if summary_file.exists() else None
    error = next((r for r in reversed(records) if r["kind"] == "error"), None)
    status = "completed" if summary else "stopped" if error else "running"
    tps = config["derived"]["tokens_per_step"]
    train = {"step": [], "tokens": [], "loss": [], "lr_mult": []}
    val = {"step": [], "tokens": [], "loss": []}
    previous = 0
    for r in records:
        if max_tokens is not None and r.get("tokens", 0) > max_tokens:
            continue
        if r["kind"] == "train":
            if r["loss"] is not None:
                train["step"].append(r["step"])
                train["tokens"].append((previous + r["step"]) / 2 * tps)  # the middle of the steps averaged over
                train["loss"].append(r["loss"])
                train["lr_mult"].append(r["lr_mult"])
            previous = r["step"]
        elif r["kind"] == "val" and r["val_loss"] is not None:
            val["step"].append(r["step"])
            val["tokens"].append(r["tokens"])
            val["loss"].append(r["val_loss"])
    return {
        "run": name, "status": status, "folder": str(folder), "config": config, "summary": summary,
        "error": error["error"] if error else None, "stopped_at": error["tokens"] if error else None,
        "val_loss": summary["val_loss"] if summary else None,
        "tokens": config["derived"]["tokens"], "steps": config["derived"]["steps"], "tokens_per_step": tps,
        "train": train, "val": val,
    }


def same_recipe(a: dict, b: dict, ignore: tuple[str, ...] = ()) -> list[str]:
    """The fields of the model and training configurations in which two runs differ, apart from `ignore`."""
    out = []
    for part in ("model_config", "train_config"):
        ca, cb = a["config"][part], b["config"][part]
        out += [key for key in sorted(set(ca) | set(cb)) if ca.get(key) != cb.get(key) and key not in ignore]
    return out
