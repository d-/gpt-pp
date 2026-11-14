"""Every run of the lab as one file, for the site's figure `gpt-every-run` (part 11, the model card): the summary of
each run and its held-out loss at every evaluation.

    python -m gptpp.export.every_run [--runs results/runs] [--jobs experiments] [--out <dir>] [--show]

Nothing is measured here: every number is copied from the trainer's own files. A finished run is read from the
small copy the trainer leaves in the repository (`results/runs/<run>/config.json`, `summary.json`, `log.jsonl`). A
run that is named in the job lists (`experiments/*.jsonl`) and has started but not finished is read from the
workspace (`$GPTPP_WORK/ckpt/<run>/`) and marked "running" (or "stopped" if its log ends with an error), with its
curve so far: the script can be run at any time and is run again when the queue has finished.

Families (by the run's name): main (`gptpp-base`), ladder (`ladder-*`), swaps (`swap-*`, `warm-*`), optimizers
(`opt-*`, `optflat-*`, `freq-*`), grid (`grid-*`), race (`race-*`), small (`pico*`, `circ-*`), chat (`gptpp-chat*`: the
chat stage has its own trainer, whose log has the loss on conversations in nats a token and no bits per byte, so
`bpb` is null there); `split-*` (the ladder's step 7 split in two) is listed with the ladder, `gptpp-base-*` with the
main run. A run of another name is listed under "small" and named on the terminal.

The file (`every-run.json`), field by field:

    generated_by   "gptpp.export.every_run"
    families       [{key, label, what}] in the order drawn
    runs           one entry a run, by family, then by name in natural order:
      run            the run's name in the repository
      label          what the run is, in words (made from the name)
      family         a family key
      status         "completed" | "running" | "stopped"
      data           the corpus, e.g. "FineWeb-EDU"; tokenizer: e.g. "bpe32k"
      layers, width, heads, vocabulary, context      the model's shape
      params         total parameters
      recipe         "2026" | "2019": the preset's recipe; changed: the overrides given to the trainer, as written
                     ("act=gelu"); optimizer: e.g. "muon+adamw"; lr: the rate of the hidden matrices
      budget         tokens the run was to see; tokens: tokens it has seen; steps: optimizer steps taken
      cards          GPUs; seconds: wall-clock so far or in all; card_hours: cards x seconds / 3600
      tokens_per_second   throughput of training (null while running)
      loss, bpb      held-out loss at the last evaluation: nats per token, bits per byte (null for the chat stage)
      curve          {tokens: [...], loss: [...], bpb: [...]}: every evaluation on the fixed held-out set, the one
                     before training (tokens 0) included
    totals         {runs, completed, running, stopped, tokens, card_hours}

Losses keep five significant digits.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .. import repo_dir
from .common import CKPT, DATA, write_json

NAME = "every-run"
FAMILIES = [
    ("main", "the main run", "gpt-pp: the 2026 recipe at 12 layers, 768 wide"),
    ("race", "the race", "the 2019 recipe against the 2026 recipe at gpt-pp's shape, 100M tokens each"),
    ("ladder", "the ladder", "one more change at each step from the 2019 recipe to the 2026 recipe, 8 layers, 512 wide"),
    ("swaps", "the swaps", "the 2026 recipe with one component swapped, 8 layers, 512 wide, 50M tokens"),
    ("optimizers", "the optimizers", "one model under SGD, Adam, AdamW, Lion and Muon at several rates"),
    ("grid", "the grid", "six sizes by four token budgets with the recipe's width rule: the grid the scaling laws are fitted on"),
    ("grid0", "the earlier grid", "the first version of the grid: the widths other than 512 trained with the table rates of the 512-wide model"),
    ("small", "the small models", "pico and the attention-only models, on TinyStories"),
    ("chat", "the chat stage", "gpt-pp trained further on conversations; scored on conversations, in nats a token only"),
]
PREFIX = [("gptpp-base", "main"), ("gptpp-chat", "chat"), ("ladder-", "ladder"), ("split-", "ladder"), ("swap-", "swaps"),
          ("warm-", "swaps"), ("opt-", "optimizers"), ("optflat-", "optimizers"), ("freq-", "optimizers"), ("gridr-", "grid"), ("grid-", "grid"),
          ("race-", "race"), ("pico", "small"), ("circ-", "small")]
RUNGS = {"gpt2-2019": "the 2019 recipe", "no-biases": "no biases", "rotary": "rotary positions", "rmsnorm": "RMS norm",
         "relu2": "squared ReLU", "qk-norm": "cosine scores", "zero-init": "zero writers", "muon": "the 2026 training recipe",
         "untied": "untied", "softcap": "soft cap"}
WORDS = {"act": "activation", "pos": "position", "norm": "normalization", "attn": "attention", "sched": "schedule"}
VALUES = {"relu": "ReLU", "gelu": "GELU", "silu": "SiLU", "swiglu": "SwiGLU", "geglu": "GeGLU", "learned": "learned table",
          "sinusoidal": "sinusoids added before the embedding norm (a fading signal)", "sinusoidal-after-norm": "sinusoids",
          "alibi": "ALiBi", "none": "none", "layernorm": "layer norm", "rmsnorm-gain": "RMS norm with a gain",
          "dyt": "dynamic tanh", "post": "layer norm after the addition", "no-qk-norm": "dot-product scores",
          "qk-learned": "a learned scale a head", "gqa4": "4 key-value heads", "gqa2": "2 key-value heads", "mqa": "1 key-value head",
          "cosine": "cosine", "linear": "linear", "constant": "constant"}
OTHER = {"swap-base": "the 2026 recipe (the swaps' reference)", "swap-no-warmup": "no warmup", "swap-no-softcap": "no soft cap",
         "swap-tied": "tied embedding", "swap-init-gpt2": "the 2019 initialization", "swap-no-embed-norm": "no norm after the embedding",
         "swap-bias": "with biases", "warm-2019-none": "the 2019 recipe, no warmup", "warm-2019-post": "the 2019 recipe, post-norm",
         "warm-2019-post-none": "the 2019 recipe, post-norm, no warmup", "pico": "pico", "pico-2019": "pico, the 2019 recipe",
         "gptpp-base": "gpt-pp", "gptpp-base-1B": "gpt-pp’s recipe on a budget of 1.08B tokens", "gptpp-chat": "gpt-pp, the chat stage",
         "warm-2019-base": "the 2019 recipe, 50M tokens (the reference of the warmup runs)",
         "warm-2019-post-w32M": "the 2019 recipe, post-norm, warmup of 32M tokens",
         "warm-2019-post-w32M-lr6e-4": "the 2019 recipe, post-norm, warmup of 32M tokens, rate 6e-4"}
OPTIMIZERS = {"adamw": "AdamW", "adam": "Adam", "lion": "Lion", "sgd": "SGD with momentum", "sgdraw": "SGD with momentum, no clipping",
              "sign": "sign descent with momentum", "muon": "Muon with AdamW"}
CORPORA = {"fineweb-edu": "FineWeb-EDU", "tinystories": "TinyStories", "tinystories-repeats": "TinyStories with repeated sequences"}


def family_of(name: str) -> str | None:
    """The family of a run, from its name. The grid that is fitted is the runs made again with the recipe's width rule
    (gridr-*) and the four 512-wide runs of the first grid, where the rule changes nothing; the first grid's other
    runs are the earlier version."""
    if name.startswith("grid-") and not name.startswith("grid-w512-"):
        return "grid0"
    return next((fam for prefix, fam in PREFIX if name.startswith(prefix)), None)


def label_of(name: str) -> str:
    """What a run is, in words, from its name."""
    if name in OTHER:
        return OTHER[name]
    m = re.match(r"ladder-(\d+)-(.+)-s(\d+)$", name)
    if m:
        k, rung, seed = int(m.group(1)), m.group(2), int(m.group(3))
        return (f"step {k}: {RUNGS.get(rung, rung)}" if k else RUNGS.get(rung, rung)) + (f", seed {seed}" if seed else "")
    m = re.match(r"ladder-(\d+)-(.+)-w(\d+)M$", name)           # a rung of the 2019 side again with a longer warmup
    if m:
        k, rung = int(m.group(1)), m.group(2)
        return (f"step {k}: {RUNGS.get(rung, rung)}" if k else RUNGS.get(rung, rung)) + f", warmup of {m.group(3)}M tokens"
    m = re.match(r"(.+)-s([1-9]\d*)$", name)                    # a second seed of any other run
    if m and not name.startswith("gptpp-chat"):
        return f"{label_of(m.group(1))}, seed {m.group(2)}"
    m = re.match(r"(.+)-lrg(.+)$", name)
    if m:
        return f"{label_of(m.group(1))}, gains and biases trained at a rate of {m.group(2)}"
    m = re.match(r"swap-norm-dyt-a(.+)$", name)
    if m:
        return f"normalization: dynamic tanh, slope {m.group(1)}"
    m = re.match(r"swap-(act|pos|norm|attn|sched)-(.+)$", name)
    if m:
        return f"{WORDS[m.group(1)]}: {VALUES.get(m.group(2), m.group(2))}"
    m = re.match(r"gptpp-chat-(probe-)?s(.+)$", name)
    if m:
        return "the chat stage, " + ("a short probe, " if m.group(1) else "") + f"rates scaled by {m.group(2)}"
    m = re.match(r"split-rung7-([a-z]+)-lr(.+)$", name)
    if m:
        return f"step 7 split: the 2026 training recipe with {OPTIMIZERS.get(m.group(1), m.group(1))} at {m.group(2)} on the hidden matrices"
    m = re.match(r"opt-([a-z]+)-own-lr(.+)$", name)
    if m:
        return f"{OPTIMIZERS.get(m.group(1), m.group(1))} at {m.group(2)}, its own warmup and clipping"
    m = re.match(r"opt-([a-z]+)-cos-lr(.+)$", name)
    if m:
        return f"{OPTIMIZERS.get(m.group(1), m.group(1))} at {m.group(2)}, its own warmup and clipping, cosine decay"
    m = re.match(r"optflat-([a-z]+)-lr(.+)$", name)
    if m:
        return f"{OPTIMIZERS.get(m.group(1), m.group(1))} at {m.group(2)} on every parameter, one rate"
    m = re.match(r"warm-2019-small-(pre|post)(-none)?$", name)
    if m:
        return f"the 2019 recipe at steps of 8,192 tokens: {m.group(1)}-norm, " + ("no warmup" if m.group(2) else "warmup")
    m = re.match(r"race-2019-lr(.+?)(-w8M)?-(\d+)M$", name)
    if m:
        return f"the 2019 recipe at {m.group(1)}, warmup of " + ("8M" if m.group(2) else "32M") + " tokens"
    m = re.match(r"(opt|freq)-([a-z]+)-lr(.+)$", name)
    if m:
        return f"{OPTIMIZERS.get(m.group(2), m.group(2))} at {m.group(3)}" + (", pico on TinyStories" if m.group(1) == "freq" else "")
    m = re.match(r"(gridr?)-w(\d+)-l(\d+)-(\d+)M$", name)
    if m:
        return f"{m.group(2)} wide, {m.group(3)} layers, {m.group(4)}M tokens" + (", before the width rule" if family_of(name) == "grid0" else "")
    if re.match(r"race-2026-(\d+)M$", name):
        return "the 2026 recipe"
    m = re.match(r"race-2026-lr(.+)-(\d+)M$", name)
    if m:
        return f"the 2026 recipe, Muon at {m.group(1)}"
    m = re.match(r"race-2026-ropebase(.+)-(\d+)M$", name)
    if m:
        return f"the 2026 recipe, rotary base {int(float(m.group(1))):,}"
    m = re.match(r"circ-attn(\d)(-repeats)?$", name)
    if m:
        return ("one-layer" if m.group(1) == "1" else "two-layer") + " attention-only model" + (", repeated sequences" if m.group(2) else "")
    return name.replace("-", " ")


def natural(name: str) -> list:
    return [float(x) if re.match(r"^\d+(\.\d+)?$", x) else x for x in re.split(r"(\d+(?:\.\d+)?)", name)]


def sig(v, digits: int = 5):
    return None if v is None else float(f"{v:.{digits}g}")


def read(folder: Path, name: str) -> dict | None:
    if not (folder / "log.jsonl").exists() or not (folder / "config.json").exists():
        return None
    config = json.loads((folder / "config.json").read_text())
    records = []
    for line in (folder / "log.jsonl").read_text().splitlines():
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:                     # a line being written while we read
                break
    summary = json.loads((folder / "summary.json").read_text()) if (folder / "summary.json").exists() else None
    if summary and summary.get("status") != "completed":
        summary = None
    error = next((r for r in reversed(records) if r.get("kind") == "error"), None)
    vals = [r for r in records if r.get("kind") == "val" and r.get("val_loss") is not None]
    if not vals:
        return None
    m, t, d = config["model_config"], config.get("train_config", {}), config.get("derived", {})
    chat = config.get("stage") == "chat"
    last = max((r for r in records if "tokens" in r), key=lambda r: r["tokens"])
    seconds = summary["seconds"] if summary else max((r.get("time", 0.0) for r in records), default=0.0)
    cards = (summary or {}).get("n_gpu", d.get("n_gpu", 1))
    corpus, _, tokenizer = str(t.get("data", "")).partition("/")
    params = (summary.get("params") or {}).get("total") if summary else None
    return {
        "run": name, "label": label_of(name), "family": family_of(name) or "small",
        "status": "completed" if summary else "stopped" if error else "running",
        "data": "conversations" if chat else CORPORA.get(corpus, corpus), "tokenizer": tokenizer,
        "layers": m["n_layer"], "width": m["d_model"], "heads": m["n_head"], "vocabulary": m["vocab_size"], "context": m["block_size"],
        "params": params, "recipe": "2019" if str(config.get("preset", "")).startswith("gpt2-2019") else "2026",
        "changed": ([f"rates scaled by {config['lr_scale']}"] if chat and "lr_scale" in config else []) + list(config.get("overrides", [])) +
                   ([f"trained as {config['train_preset']}"] if config.get("train_preset") and config["train_preset"] != config.get("preset") else []),
        "optimizer": t.get("optimizer"), "lr": t.get("lr"),
        "budget": d.get("tokens", (summary or last)["tokens"]), "tokens": summary["tokens"] if summary else last["tokens"], "steps": summary["steps"] if summary else last.get("step"),
        "cards": cards, "seconds": round(seconds, 1), "card_hours": sig(cards * seconds / 3600, 4),
        "tokens_per_second": summary["tokens_per_second"] if summary else None,
        "loss": sig(vals[-1]["val_loss"], 6), "bpb": sig(vals[-1].get("val_bpb"), 6),
        "curve": {"tokens": [r["tokens"] for r in vals], "loss": [sig(r["val_loss"]) for r in vals], "bpb": [sig(r.get("val_bpb")) for r in vals]},
    }


def count_params(run: dict, folder: Path) -> int | None:
    """The parameter count of a run that has no summary yet: that of a freshly built model of its configuration."""
    try:
        from ..config import ModelConfig
        from ..model import GPT

        cfg = ModelConfig.from_dict(json.loads((folder / "config.json").read_text())["model_config"])
        return GPT(cfg).num_params()["total"]
    except Exception as e:                                   # the figure can do without it
        print(f"  {run['run']}: parameters not counted ({e})")
        return None


def build(runs_dir: Path, jobs_dir: Path) -> tuple[dict, list[str]]:
    names, notes = {}, []
    for folder in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        names[folder.name] = folder
    for jobs in sorted(jobs_dir.glob("*.jsonl")):
        for line in jobs.read_text().splitlines():
            if line.strip().startswith("{"):                 # a job list may carry comment lines
                name = json.loads(line)["name"]
                if name not in names and (CKPT / name / "log.jsonl").exists():
                    names[name] = CKPT / name
    runs = []
    for name, folder in names.items():
        r = read(folder, name)
        if r is None:
            notes.append(f"{name}: no evaluation yet, left out")
            continue
        if family_of(name) is None:
            notes.append(f"{name}: a name of no family, listed under the small models")
        if r["params"] is None:
            r["params"] = count_params(r, folder)
        runs.append(r)
    order = [k for k, _, _ in FAMILIES]
    runs.sort(key=lambda r: (order.index(r["family"]), natural(re.sub(r"^gridr-", "grid-", r["run"]))))   # the grid by width, then tokens
    totals = {"runs": len(runs), "completed": sum(r["status"] == "completed" for r in runs),
              "running": sum(r["status"] == "running" for r in runs), "stopped": sum(r["status"] == "stopped" for r in runs),
              "tokens": sum(r["tokens"] for r in runs), "card_hours": sig(sum(r["card_hours"] for r in runs), 4)}
    # runs that are counted and not listed: the short probes of the defaults and of throughput (core-*), which are in
    # the workspace only; with them the totals are those of the ledger of all finished runs
    left, strays = [], []
    for p in (sorted(CKPT.iterdir()) if CKPT.exists() else []):
        if not p.is_dir() or p.name in names or not (p / "summary.json").exists():
            continue
        s = json.loads((p / "summary.json").read_text())
        if s.get("status") == "completed":
            (left if p.name.startswith("core-") else strays).append(s)
            if not p.name.startswith("core-"):
                notes.append(f"{p.name}: finished in the workspace, neither listed nor counted")
    totals["left_out"] = {"runs": len(left), "tokens": sum(s.get("tokens", 0) for s in left),
                          "card_hours": sig(sum(s.get("seconds", 0) * s.get("n_gpu", 1) for s in left) / 3600, 4),
                          "what": "short probes of the defaults and of throughput, made while the recipe was being settled"} if left else None
    return {"generated_by": "gptpp.export.every_run",
            "families": [{"key": k, "label": lab, "what": what} for k, lab, what in FAMILIES], "runs": runs, "totals": totals}, notes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", default=str(repo_dir() / "results" / "runs"))
    ap.add_argument("--jobs", default=str(repo_dir() / "experiments"))
    ap.add_argument("--out", default=str(DATA))
    ap.add_argument("--show", action="store_true", help="print every run")
    args = ap.parse_args()
    data, notes = build(Path(args.runs), Path(args.jobs))
    size = write_json(Path(args.out) / f"{NAME}.json", data)
    t = data["totals"]
    print(f"{NAME}.json: {t['runs']} runs ({t['completed']} completed, {t['running']} running, {t['stopped']} stopped), "
          f"{t['tokens'] / 1e9:.2f}B tokens, {t['card_hours']} card-hours, {size:,} bytes")
    for k, lab, _ in FAMILIES:
        fam = [r for r in data["runs"] if r["family"] == k]
        sc = [r["bpb"] for r in fam if r["bpb"] is not None]
        print(f"  {lab:18} {len(fam):3} runs" + ("" if not sc else f"   bits per byte {min(sc):.4f} to {max(sc):.4f}"))
    for note in notes:
        print("  " + note)
    if args.show:
        for r in data["runs"]:
            print(f"  {r['family']:10} {r['status']:9} {r['run']:28} {r['label'][:44]:44} {r['layers']:2}x{r['width']:<4} {r['tokens'] / 1e6:7.0f}M "
                  f"{r['card_hours']:7.3f} h  {r['loss']:.4f}  {r['bpb'] or 0:.4f}  {len(r['curve']['tokens'])} evals  {','.join(r['changed'])[:40]}")


if __name__ == "__main__":
    main()
