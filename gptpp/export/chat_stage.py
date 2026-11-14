"""The data of fig-07-08 `gpt-chat-stage`: the chat stage in one file, assets/data/gpt/chat-stage.json.

    python -m gptpp.export.chat_stage [--measurements results/chat/chat-stage.json] [--run gptpp-chat] [--out DIR]

Nothing is computed here but a thinning of the curve: the file is the measurements of `gptpp.chat_eval`
(the held-out losses, the plain-text cost, the example conversation token by token, the twenty questions with
both models' texts, the tally and its verdicts) and the log of the stage's run (training loss, and at each
evaluation the held-out conversation loss, its two parts, and the loss on plain text).

Re-run it after `python -m gptpp.chat_eval` (and again after `--tally`); it needs no card. The figure's module and
its test read the file as it is, so they hold for any run of the stage.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .. import repo_dir
from .common import CKPT, DATA, write_json

NAME = "chat-stage"


def read_log(run: str) -> tuple[dict, dict, list[dict], Path]:
    """The run's config, summary and log records: `run` is a run directory, or a name looked for in the
    workspace and then among the repository's copies."""
    for folder in (Path(run), CKPT / run, repo_dir() / "results" / "runs" / run):
        if (folder / "log.jsonl").exists() and (folder / "summary.json").exists():
            records = [json.loads(line) for line in (folder / "log.jsonl").read_text().splitlines() if line.strip()]
            return (json.loads((folder / "config.json").read_text()), json.loads((folder / "summary.json").read_text()),
                    records, folder)
    raise SystemExit(f"no completed run named {run!r} in {CKPT} or in results/runs")


def thin(records: list[dict], most: int) -> list[dict]:
    """At most `most` records, evenly spaced, the last one kept."""
    stride = max(1, -(-len(records) // most))
    return [r for i, r in enumerate(records) if (i + 1) % stride == 0 or i == len(records) - 1]


def export(measurements: Path, run: str, out: Path, max_train_points: int = 240) -> dict:
    m = json.loads(measurements.read_text(encoding="utf-8"))
    config, summary, records, folder = read_log(run)
    if m["chat"]["run"] != folder.name or m["chat"]["steps"] != summary["steps"]:
        raise SystemExit(f"{measurements} measures run {m['chat']['run']} of {m['chat']['steps']} steps, not this "
                         f"{run} of {summary['steps']} steps: run gptpp.chat_eval again")
    tps = config["derived"]["tokens_per_step"]
    train_records = [r for r in records if r["kind"] == "train" and r["loss"] is not None]
    previous = {r["step"]: (train_records[i - 1]["step"] if i else 0) for i, r in enumerate(train_records)}
    train = thin(train_records, max_train_points)
    val = [r for r in records if r["kind"] == "val"]
    samples = [r for r in records if r["kind"] == "sample"]

    def rate_row(name: str, scale: float, chosen: bool, measured: dict, seconds: float, tokens_per_second: int) -> dict:
        """One full run of the stage as a row of the table of rates. `measured` holds its held_out and roles and,
        if they were measured, its plain_text and its loss on the base run's own windows."""
        held, roles, plain = measured["held_out"], measured["roles"], measured.get("plain_text")
        return {"run": name, "lr_scale": scale, "chosen": chosen, "loss": held["loss"],
                "loss_text": held["loss_text"], "loss_end": held["loss_end"],
                "user_text": roles["user_text"]["loss"], "markers": roles["markers"]["loss"],
                "plain_bpb": plain["bits_per_byte"] if plain else None, "plain_nats": plain["loss"] if plain else None,
                "pretraining_windows": measured.get("pretraining_windows", {}).get("after"),
                "seconds": seconds, "tokens_per_second": tokens_per_second}

    shown = {"held_out": m["held_out"]["after"], "roles": m["roles"]["after"],
             "pretraining_windows": m.get("pretraining_windows", {})}
    if "plain_text" in m:
        shown["plain_text"] = m["plain_text"]["after"]
    rates = {
        # the run of this file first (the one the rule chose and the figure shows), then the other full runs
        "runs": [rate_row(folder.name, config["lr_scale"], True, shown, summary["seconds"],
                          summary["tokens_per_second"])]
                + [rate_row(o["run"], o["lr_scale"], False, o, o["seconds"], o["tokens_per_second"])
                   for o in m.get("other_runs", {}).get("runs", [])],
        "probes": [{"lr_scale": p["lr_scale"], "tokens": p["tokens"], "base_step": p["base"]["step"],
                    "loss": p["val_loss"], "loss_end": p["val_loss_end"], "plain_nats": p["plain_loss"],
                    "loss_before": p["before"]["val_loss"], "plain_nats_before": p["before"]["plain_loss"]}
                   for p in m.get("probes", {}).get("runs", [])],
    }

    def side(q: dict, mode: str) -> dict:
        a = q["chat"][mode]
        return {"text": a["text"], "tokens": a["tokens"], "stopped": a["stopped"], "verdict": a.get("verdict"),
                "note": a.get("verdict_note", "")}

    data = {
        "_generated": {"by": "gptpp.export.chat_stage", "on": time.strftime("%Y-%m-%d"),
                       "from": [measurements.name, f"the log of {run}"]},
        "model": {"n_layer": config["model_config"]["n_layer"], "d_model": config["model_config"]["d_model"],
                  "context": config["model_config"]["block_size"], "vocabulary": config["model_config"]["vocab_size"],
                  "parameters": summary["params"]["total"], "pretraining_tokens": m["base"]["tokens"],
                  "pretraining_steps": m["base"]["step"], "base_checkpoint": m["base"]["checkpoint"]},
        "stage": {
            "run": folder.name, "steps": summary["steps"], "tokens": summary["tokens"],
            "loss_tokens": summary["loss_tokens"], "passes": summary["passes"], "seconds": summary["seconds"],
            "tokens_per_second": summary["tokens_per_second"],
            "peak_memory_gb": summary["peak_memory_gb"], "tokens_per_step": tps, "lr_scale": config["lr_scale"],
            "rates": {name: group["lr"] for name, group in config["derived"]["groups"].items()},
            "warmup_tokens": config["train_config"]["warmup_tokens"],
            "final_lr_frac": config["train_config"]["final_lr_frac"],
            "gpu": config["hardware"]["gpu"],
        },
        "data": {
            "source": "SmolTalk", "hub": m["data"]["source"], "license": m["data"]["license"],
            "conversations": m["data"]["train"]["conversations"], "exchanges": m["data"]["train"]["exchanges"],
            "tokens": m["data"]["train"]["tokens"], "loss_tokens": m["data"]["train"]["loss_tokens"],
            "by_source": m["data"]["train_by_source"],
            "held_out": m["data"]["held_out"], "max_answer_tokens": m["data"]["selection"]["max_answer_tokens"],
        },
        # the curve. A training record is the mean loss of the steps since the previous record, on batches the model
        # had not yet trained on; it stands at the middle of those steps.
        "curve": {
            "train": {"tokens": [(previous[r["step"]] + r["step"]) / 2 * tps for r in train],
                      "loss": [r["loss"] for r in train]},
            "val": {"tokens": [r["tokens"] for r in val], "loss": [r["val_loss"] for r in val],
                    "text": [r["val_loss_text"] for r in val], "end": [r["val_loss_end"] for r in val],
                    "plain": [r["plain_loss"] for r in val]},
            "lr_mult": {"tokens": [r["tokens"] for r in train], "value": [r["lr_mult"] for r in train]},
            "sample": {"prompt": samples[0]["prompt"] if samples else None,
                       "max_tokens": config["derived"].get("sample_tokens", 48),
                       "tokens": [r["tokens"] for r in samples],
                       "text": [r["text"] for r in samples], "stopped": [r["stopped"] for r in samples]},
        },
        "held_out": {side_: {key: m["held_out"][side_][key] for key in ("loss", "loss_text", "loss_end", "tokens",
                                                                         "end_tokens", "conversations")}
                     for side_ in ("before", "after")},
        "held_out_by_source": {side_: {name: part["loss"] for name, part in m["held_out"][side_]["by_source"].items()}
                               for side_ in ("before", "after")},
        "roles": {side_: m["roles"][side_] for side_ in ("before", "after")},
        "rates": rates,
        "plain_text": ({side_: {key: m["plain_text"][side_][key]
                                for key in ("bits_per_byte", "bits_per_byte_se", "loss", "n_docs", "n_tokens")}
                        for side_ in ("before", "after")} if "plain_text" in m else None),
        "example": {key: m["example"][key] for key in ("source", "exchanges", "exchanges_in_full", "tokens", "special",
                                                        "mask", "loss_before", "loss_after")},
        "generation": {key: m["generation"][key] for key in ("temperature", "seed", "base_max_new_tokens",
                                                              "chat_max_new_tokens", "chat_stopped_by_itself",
                                                              "chat_answers")},
        "questions": [{"id": q["id"], "kind": q["kind"], "text": q["text"],
                       "base": {mode: {key: q["base"][mode][key] for key in ("text", "tokens", "stopped")}
                                for mode in ("greedy", "sampled")},
                       "chat": {mode: side(q, mode) for mode in ("greedy", "sampled")}} for q in m["questions"]],
        "tally": ({key: m["tally"][key] for key in ("greedy", "sampled", "by_kind", "answers", "criteria")}
                  if "tally" in m else None),
    }
    out.mkdir(parents=True, exist_ok=True)
    size = write_json(out / f"{NAME}.json", data)
    print(f"{NAME}.json: {size:,} bytes; {len(train)} training points, {len(val)} evaluations, "
          f"{len(data['questions'])} questions, tally {'in' if data['tally'] else 'NOT YET in'} the file; "
          f"plain text {'in' if data['plain_text'] else 'NOT in'} the file ({out})")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--measurements", default=str(repo_dir() / "results" / "chat" / "chat-stage.json"))
    parser.add_argument("--run", default="gptpp-chat")
    parser.add_argument("--out", default=str(DATA))
    args = parser.parse_args()
    export(Path(args.measurements), args.run, Path(args.out))


if __name__ == "__main__":
    main()
