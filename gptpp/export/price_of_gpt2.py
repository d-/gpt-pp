"""Export for fig-07-07 `gpt-price-of-gpt2`: what it has taken to train a model of GPT-2's quality, 2019 to 2026,
as two record histories that measure different things, a few reference points, and our own run.

    python -m gptpp.export.price_of_gpt2 [--research <dir of the dossiers>] [--run gptpp-base] [--out <dir>]

A second on a processor. Every row carries its source. Nothing here is measured by the lab except the last entry.

    speedrun     the 124M-parameter race to validation loss 3.28 on FineWeb on eight H100 cards: every record of the
                 table "Track 1" of the dossier speedrun-and-muon.md (its Section 2b, copied from the README of
                 modded-nanogpt), with what changed; rows marked there as re-timings of a standing record are
                 kept apart (`retimings`): the line passes through them, they are not steps of the race
    medium       the same race at 350M parameters, to loss 2.92: the table "Track 2" (Section 2c); a row without a
                 date is kept with date null
    benchmark    the time on eight H100 cards to pass the benchmark score (CORE) of the released 1.5B-parameter
                 GPT-2: the leaderboard table of the dossier nanochat.md (Section 6.4)
    references   GPT-2's own training as reported (second-hand), nanoGPT's reproduction, the two llm.c
                 reproductions: literal rows, each checked against a phrase that must stand in a dossier
    ours         the main run of gpt-pp: tokens, cards and seconds from results/runs/<run>/summary.json when the run
                 has ended; before that, from its config and the last line of its log, marked provisional

Writes `price-of-gpt2.json`.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path

from .. import repo_dir, work_dir
from .common import DATA, write_atomic

NAME = "price-of-gpt2"
MONTHS = {m: i + 1 for i, m in enumerate("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split())}


def section(text: str, start: str, stop: str) -> str:
    a = text.index(start)
    return text[a:text.index(stop, a + len(start))]


def cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def table_rows(block: str) -> list[list[str]]:
    """The rows of the first Markdown table of a block whose first cell is a number."""
    out = []
    for line in block.splitlines():
        if line.startswith("|") and re.match(r"\|\s*\d+\s*\|", line):
            out.append(cells(line))
    return out


def minutes(text: str) -> float:
    """'5.8 h' -> 348, '29.3 min' -> 29.3, '24.9' -> 24.9 (the tables give minutes unless they say hours)."""
    m = re.match(r"([\d.]+)\s*(h|min)?", text)
    return float(m.group(1)) * (60.0 if m.group(2) == "h" else 1.0)


def speedrun(text: str) -> tuple[list[dict], list[dict], list[dict]]:
    small, retimed = [], []
    for row in table_rows(section(text, "### 2b.", "### 2c.")):
        n, date, time, what = int(row[0]), row[1], row[2], row[3]
        if "not a record" in what:                             # the same record timed again under new rules: on the line, not a step of the race
            retimed.append({"n": n, "date": date, "minutes": minutes(time), "what": what.strip("()")})
            continue
        small.append({"n": n, "date": date, "minutes": minutes(time), "what": re.sub(r"\s*\(sys\)", "", what), "systems": "(sys)" in what})
    medium = []
    for row in table_rows(section(text, "### 2c.", "### 2d.")):
        medium.append({"n": int(row[0]), "date": None if row[1] in ("-", "") else row[1], "minutes": minutes(row[2]), "what": row[3]})
    numbers = [r["n"] for r in small]
    if numbers != list(range(1, len(numbers) + 1)):
        raise SystemExit(f"the record numbers of the speedrun are not 1 .. {len(numbers)}: {numbers}")
    return small, medium, retimed


def benchmark(text: str) -> list[dict]:
    block = section(text, "**The leaderboard**", "Costs for runs 2 to 6")
    out = []
    for row in table_rows(block):
        n, date, what, time, bpb, core, cost = int(row[0]), row[1], row[2], row[5], row[6], row[7], row[8]
        hours = float(re.search(r"([\d.]+) h", time).group(1))
        m = re.match(r"(\d+) (\w{3}) (\d{4})", date)
        iso = f"{m.group(3)}-{MONTHS[m.group(2)]:02d}-{int(m.group(1)):02d}" if m else None
        out.append({"n": n, "date": iso, "year": int(date) if date.isdigit() else None, "minutes": hours * 60.0, "what": what,
                    "core": float(re.match(r"[\d.]+", core).group(0)), "cost": cost, "quoted": "as quoted" in time})
    return out


def references(dossiers: dict[str, str]) -> list[dict]:
    """Literal rows; each names the dossier and a phrase of it that carries the number, and the phrase must be there."""
    rows = [
        {"key": "gpt2-xl", "label": "GPT-2, 1.5B parameters, as reported", "year": 2019, "date": None, "minutes": 168 * 60.0, "machine": "32 TPU v3 chips",
         "target": "the released model itself", "note": "about a week; second-hand: the model card cites a comment by one of its authors. The cost often quoted, 43,000 dollars, is a later estimate at 8 dollars a chip-hour",
         "second_hand": True, "dossier": "nanogpt-vs-gpt2.md", "phrase": "32 TPUv3 chips, 168 hours", "source": "Hugging Face model card of gpt2-xl, citing an author's comment; nanochat discussion 481"},
        {"key": "nanogpt", "label": "nanoGPT reproduces the 124M model", "year": 2023, "date": "2023-02-03", "minutes": 4 * 24 * 60.0, "machine": "eight A100 cards",
         "target": "validation loss about 2.85 on OpenWebText, 300B tokens", "note": "about four days", "second_hand": False, "dossier": "nanogpt-vs-gpt2.md",
         "phrase": "a single 8XA100 40GB node in about 4 days", "source": "nanoGPT README"},
        {"key": "llmc-124m", "label": "llm.c reproduces the 124M model", "year": 2024, "date": "2024-05-28", "minutes": 90.0, "machine": "eight A100 cards",
         "target": "validation loss about 3.29 on FineWeb, 10B tokens", "note": "about 90 minutes, about 20 dollars", "second_hand": False, "dossier": "nanogpt-vs-gpt2.md",
         "phrase": "Reproducing GPT-2 (124M) in llm.c in 90 minutes for $20", "source": "llm.c discussion 481"},
        {"key": "llmc-1558m", "label": "llm.c reproduces the 1.6B model", "year": 2024, "date": "2024-07-11", "minutes": 24 * 60.0, "machine": "eight H100 cards",
         "target": "the 1.6B model reproduced", "note": "24 hours, 672 dollars", "second_hand": False, "dossier": "nanogpt-vs-gpt2.md",
         "phrase": "one 8XH100 node, 24 hours, $672", "source": "llm.c discussion 677"},
    ]
    for r in rows:
        if r["phrase"] not in dossiers[r["dossier"]]:
            raise SystemExit(f"the dossier {r['dossier']} does not hold the phrase {r['phrase']!r} that the row {r['key']} rests on")
    return rows


def ours(run: str) -> dict:
    """The main run: from its summary when it has ended, else from its config and the last line of its log."""
    summary = repo_dir() / "results" / "runs" / run / "summary.json"
    ckpt = work_dir() / "ckpt" / run
    if not summary.exists() and (ckpt / "summary.json").exists():
        summary = ckpt / "summary.json"
    config_file = (summary.parent / "config.json") if (summary.parent / "config.json").exists() else ckpt / "config.json"
    config = json.loads(config_file.read_text())
    base = {"run": run, "label": "gpt-pp", "layers": config["model_config"]["n_layer"], "width": config["model_config"]["d_model"],
            "parameters": config["params"]["total"], "data": config["train_config"]["data"], "gpu": config["hardware"]["gpu"]}
    if summary.exists():
        s = json.loads(summary.read_text())
        day = dt.datetime.fromtimestamp(summary.stat().st_mtime, dt.timezone.utc).date().isoformat()
        return {**base, "provisional": s.get("status") != "completed", "tokens": s["tokens"], "cards": s["n_gpu"], "seconds": s["seconds"], "train_seconds": s.get("train_seconds"),
                "minutes": s["seconds"] / 60.0, "val_bpb": s.get("val_bpb"), "val_loss": s.get("val_loss"), "date": day, "source": f"results/runs/{run}/summary.json"}
    last = None
    with open(ckpt / "log.jsonl") as f:
        for line in f:
            if '"kind": "train"' in line:
                last = line
    t = json.loads(last)
    total = config["derived"]["tokens"]
    seconds = t["time"] * total / t["tokens"]                  # the log's clock, stretched to the planned tokens
    day = dt.datetime.fromtimestamp((ckpt / "log.jsonl").stat().st_mtime, dt.timezone.utc).date().isoformat()
    return {**base, "provisional": True, "tokens": total, "cards": config["derived"]["n_gpu"], "seconds": seconds, "train_seconds": None, "minutes": seconds / 60.0,
            "val_bpb": None, "val_loss": None, "date": day, "tokens_done": t["tokens"], "seconds_done": t["time"],
            "source": f"config.json and log.jsonl of the run in progress ({t['tokens'] / total:.0%} of its tokens): seconds projected"}


def export(research: Path, run: str, out: Path) -> dict:
    dossiers = {name: (research / name).read_text() for name in ("speedrun-and-muon.md", "nanochat.md", "nanogpt-vs-gpt2.md")}
    small, medium, retimed = speedrun(dossiers["speedrun-and-muon.md"])
    bench = benchmark(dossiers["nanochat.md"])
    data = {
        "name": NAME,
        "tracks": {
            "speedrun": {"label": "the 124M race", "model": "124M parameters", "target": "validation loss 3.28 on FineWeb", "machine": "eight H100 cards",
                         "source": "README record table of modded-nanogpt, as copied in the dossier speedrun-and-muon.md, Section 2b", "records": small, "retimings": retimed},
            "medium": {"label": "the 350M race", "model": "350M parameters", "target": "validation loss 2.92 on FineWeb", "machine": "eight H100 cards",
                       "source": "the same README, dossier Section 2c", "records": medium},
            "benchmark": {"label": "time to GPT-2", "model": "models of 0.5B to 1.9B parameters", "target": "the benchmark score (CORE) of the released 1.5B GPT-2, 0.2565", "machine": "eight H100 cards",
                          "source": "leaderboard of nanochat (README and LEADERBOARD), as copied in the dossier nanochat.md, Section 6.4", "records": [r for r in bench if r["date"]]},
        },
        "references": references(dossiers),
        "ours": ours(run),
    }
    quoted = [r for r in bench if not r["date"]]
    if quoted:                                                  # the leaderboard's own row for 2019 agrees with the reference row
        assert abs(quoted[0]["minutes"] - data["references"][0]["minutes"]) < 1e-6
    out.mkdir(parents=True, exist_ok=True)
    write_atomic(out / f"{NAME}.json", json.dumps(data, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    o = data["ours"]
    return {"speedrun_records": len(small), "first": [small[0]["date"], small[0]["minutes"]], "last": [small[-1]["date"], small[-1]["minutes"]], "medium_records": len(medium),
            "benchmark_records": len(data["tracks"]["benchmark"]["records"]), "references": [r["key"] for r in data["references"]],
            "ours": {k: o[k] for k in ("provisional", "tokens", "cards", "seconds", "date", "source")}, "bytes": (out / f"{NAME}.json").stat().st_size}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--research", default=str(work_dir() / "research"))
    ap.add_argument("--run", default="gptpp-base")
    ap.add_argument("--out", default=str(DATA))
    args = ap.parse_args()
    print(json.dumps(export(Path(args.research), args.run, Path(args.out)), indent=1))


if __name__ == "__main__":
    main()
