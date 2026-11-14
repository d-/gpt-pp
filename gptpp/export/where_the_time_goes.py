"""Tokens per second of the 12-layer model as each piece of engineering is switched on, measured on this machine with
the trainer's own code, for the figure `gpt-where-the-time-goes` (fig-07-04).

    CUDA_VISIBLE_DEVICES=<k> python -m gptpp.export.where_the_time_goes --measure [--seconds 25] [--only compiled,split48]
    python -m gptpp.export.where_the_time_goes [--out <dir>]

Two halves. `--measure` runs short pieces of training on one card and writes the raw timings to
`results/interp/throughput-waterfall.json`; about four minutes in all, each configuration under a minute. Without it
the script only reads: that file, the summaries of the runs on one, two and three cards, and the table of
`docs/DESIGN.md`, Section 5.2, and writes `assets/data/gpt/where-the-time-goes.json`. Run the second half again
whenever the first has been repeated or the main run has ended.

What is measured (`--measure`). The model of the main run (preset `gptpp-base`: 12 layers, 768 wide, context 1,024),
its optimizers (`build_optimizers`: Muon and AdamW at the recipe's rates), its loader (`TrainLoader` on FineWeb-EDU)
and one optimizer step for every 48 sequences of 1,024 tokens, as in the main run on one card. Five configurations,
each adding one piece to the one before:

    float32     float32 arithmetic without TensorFloat-32, attention written out (`forward_explicit`), not compiled
    tf32        + TensorFloat-32 matrix products
    bf16        + bfloat16 autocast
    fused       + the fused attention kernel (`forward`, F.scaled_dot_product_attention)
    compiled    + torch.compile: the recipe of the main run on one card

The 48 sequences of a step are read in as few passes as fit the card's memory with that configuration (the
gradient is accumulated over the passes, as the trainer's `grad_accum` does): attention written out keeps a table of
scores of 1,024 x 1,024 numbers for every head, layer and sequence, and does not fit 48 sequences in 32 GB. Each
configuration takes its first step unmeasured (the compiled one pays its compilation there), then at least two
more steps and at least 12 seconds of them, also unmeasured (the cards are capped at 165 W and slow their clock under
load, so a cold card reads about 5% fast), then steps for at least `--seconds` seconds, timed with the card
synchronized.

`throughput-waterfall.json` (raw), field by field:

    machine          {device, torch, cuda, card, when (of the last call), shared (true if other processes held more than
                     500 MiB of the card during any timing kept in the file), other_mib (at the start of the last call)}
    step             {sequences: 48, seq_len: 1024, tokens: 49152}
    configs[k]       {key, what, passes (micro-batches a step), micro_batch, steps (timed), seconds, tokens_per_second
                      (the timed steps' tokens over their seconds), tokens_per_second_median (from the median step),
                      tokens_per_second_halves (the first and the second half of the timed steps: a drift shows
                      here), step_seconds (each timed step), warmup_seconds and warmup_steps (unmeasured), peak_memory_gb,
                      tried: [{micro_batch, fits}], other_mib (the most other processes held on the card, MiB, read
                      before and after the timing)}
    split            one step of the compiled recipe taken apart, each part timed alone with the card synchronized,
                     at 48 and at 32 sequences of 1,024 tokens: {sequences, forward_backward_ms, head_ms (the
                     unembedding, the soft-cap and the loss, forward and backward, on the final hidden states,
                     compiled as the model is; head_passes: 2 if it did not fit the card alone and was timed on half
                     the sequences),
                     optimizer_ms (Muon and AdamW), optimizer_adamw_ms (if every parameter used AdamW), load_ms}

`where-the-time-goes.json` (the figure's file):

    generated_by     "gptpp.export.where_the_time_goes"
    measured         true when the raw file is there; false: the single-card bars are missing and the figure says so
    step, machine    as above
    bars[k]          the waterfall, in order: {key, label (what was switched on), tokens_per_second, cards, source
                     ("measured here: <n> steps in <s> s" | "run <name>: its summary"), passes, micro_batch,
                     peak_memory_gb, factor (over the bar before), of_linear (for more than one card: the rate as a
                     share of that many times the one-card run), main (true for the configuration of the main run)}
                     The cards' bars are the trainer's own summaries: `core-tput-base-b48` (one card, the same
                     configuration as `compiled`, a whole run of it), `core-ddp-base` (two), `gptpp-base` (three; from
                     the median of its log while it has no summary, marked `final: false`)
    split            the parts of one step as docs/DESIGN.md, Section 5.2 records them (12 layers, 32 x 1,024
                     tokens), parsed from that table: {sequences, forward_backward_ms, head_ms, optimizer_ms,
                     optimizer_adamw_ms, load_ms, source}; and `split_measured`, this script's own timing of the
                     same parts, for comparison
    numbers          {factor_single (compiled over float32), largest_step (key and factor), three_cards_of_linear, ...}
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import subprocess
import time
from pathlib import Path

from .. import repo_dir
from .common import CKPT, DATA, write_json

NAME = "where-the-time-goes"
RAW = repo_dir() / "results" / "interp" / "throughput-waterfall.json"
SEQUENCES, SEQ_LEN = 48, 1024
WARM_SECONDS = 12.0  # unmeasured steps before a timing: the 165 W cap lowers a card's clock under load, and a cold card reads about 5% fast
CONFIGS = [
    {"key": "float32", "label": "nothing: float32, attention written out, not compiled", "tf32": False, "bf16": False, "fused": False, "compile": False},
    {"key": "tf32", "label": "TensorFloat-32 products", "tf32": True, "bf16": False, "fused": False, "compile": False},
    {"key": "bf16", "label": "bfloat16 products", "tf32": True, "bf16": True, "fused": False, "compile": False},
    {"key": "fused", "label": "the fused attention kernel", "tf32": True, "bf16": True, "fused": True, "compile": False},
    {"key": "compiled", "label": "compilation", "tf32": True, "bf16": True, "fused": True, "compile": True},
]
CARD_RUNS = [("core-tput-base-b48", 1), ("core-ddp-base", 2), ("gptpp-base", 3)]


# ----------------------------------------------------------------------------------------------------------------
# measuring


def measure(seconds: float, only: list[str] | None, raw_path: Path, *, device: str = "cuda", sequences: int = SEQUENCES,
            seq_len: int = SEQ_LEN, shape: dict | None = None, compile: bool = True, split_at: tuple[int, ...] = (48, 32),
            split_seconds: float = 3.0, warm_seconds: float = WARM_SECONDS) -> None:
    """Time the configurations and the parts of a step, and merge them into the raw file. The keyword arguments are for
    the test, which runs a model of two layers on the CPU without compiling: the timings themselves always use the
    defaults."""
    import os

    import torch
    import torch.nn.functional as F

    from ..config import preset, train_preset
    from ..data import TokenData, TrainLoader
    from ..model import GPT
    from ..optim import build_optimizers, set_lr

    device = torch.device(device, 0) if device == "cuda" else torch.device(device)
    cuda = device.type == "cuda"
    model_cfg, train_cfg = preset("gptpp-base"), train_preset("gptpp-base")
    data = TokenData(train_cfg.data)
    model_cfg = model_cfg.replace(vocab_size=data.vocab_size, **(shape or {}))
    sync = (lambda: torch.cuda.synchronize(device)) if cuda else (lambda: None)
    clear = torch.cuda.empty_cache if cuda else (lambda: None)
    out_of_memory = torch.OutOfMemoryError if cuda else MemoryError

    def others() -> int:
        """MiB that other processes hold on the card this one runs on."""
        return other_mib(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0], os.getpid()) if cuda else 0

    raw = json.loads(raw_path.read_text()) if raw_path.exists() else {}
    raw["machine"] = {"device": torch.cuda.get_device_name(device) if cuda else "cpu", "torch": torch.__version__, "cuda": torch.version.cuda,
                      "card": os.environ.get("CUDA_VISIBLE_DEVICES", "0") if cuda else None, "shared": False, "other_mib": others(),
                      "when": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}
    raw["step"] = {"sequences": sequences, "seq_len": seq_len, "tokens": sequences * seq_len}
    done = {c["key"]: c for c in raw.get("configs", [])}
    splits = {s["sequences"]: s for s in raw.get("split", [])}

    def save() -> None:
        raw["configs"] = [done[c["key"]] for c in CONFIGS if c["key"] in done]
        if splits:
            raw["split"] = [splits[k] for k in sorted(splits, reverse=True)]
        raw["machine"]["shared"] = any(c.get("other_mib", 0) > 500 for c in raw["configs"] + raw.get("split", []))
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps(raw, indent=1) + "\n")

    def setup(cfg: dict, micro: int):
        torch.backends.cuda.matmul.allow_tf32 = cfg["tf32"]
        torch.backends.cudnn.allow_tf32 = cfg["tf32"]
        torch.manual_seed(0)
        model = GPT(model_cfg).to(device)
        optimizers = build_optimizers(model, train_cfg, sequences * seq_len)
        set_lr(optimizers, 1.0)
        net = torch.compile(model) if cfg["compile"] and compile else model
        loader = TrainLoader(data, seq_len, micro, 0, 1, 0, device)
        autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg["bf16"] and cuda)
        return model, net, optimizers, loader, autocast

    def one_step(cfg: dict, model, net, optimizers, loader, autocast, passes: int) -> None:
        for _ in range(passes):
            x, y = loader.next_batch()
            with autocast:
                if cfg["fused"]:
                    loss = net(x, y)
                else:  # attention written out: the explicit path gives the logits, the loss is taken as the fast path takes it
                    logits = model.forward_explicit(x)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1), ignore_index=-1)
            (loss / passes).backward()
        for optimizer in optimizers:
            optimizer.step()
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)

    for cfg in CONFIGS:
        if only and cfg["key"] not in only:
            continue
        tried, result, other = [], None, others()
        for micro in (m for m in (48, 24, 16, 12, 8, 6, 4, 3, 2, 1) if m <= sequences and sequences % m == 0):
            passes = sequences // micro
            state = None
            try:
                clear()
                if cuda:
                    torch.cuda.reset_peak_memory_stats(device)
                state = setup(cfg, micro)
                sync(); t = time.perf_counter()
                one_step(cfg, *state, passes)                # unmeasured: the first step pays for compilation,
                sync(); loaded, warmed = time.perf_counter(), 1
                while warmed < 3 or time.perf_counter() - loaded < warm_seconds:   # the next ones let the card's clock settle under load
                    one_step(cfg, *state, passes)
                    sync(); warmed += 1
                warm = time.perf_counter() - t
                steps, start = [], time.perf_counter()
                while time.perf_counter() - start < seconds or len(steps) < 5:
                    t = time.perf_counter()
                    one_step(cfg, *state, passes)
                    sync()
                    steps.append(time.perf_counter() - t)
                total = sum(steps)
                result = {"key": cfg["key"], "what": cfg["label"], "passes": passes, "micro_batch": micro, "steps": len(steps), "seconds": total,
                          "tokens_per_second": len(steps) * sequences * seq_len / total,
                          "tokens_per_second_median": sequences * seq_len / statistics.median(steps),
                          "tokens_per_second_halves": [len(h) * sequences * seq_len / sum(h) for h in (steps[:len(steps) // 2], steps[len(steps) // 2:])],
                          "step_seconds": steps, "warmup_seconds": warm, "warmup_steps": warmed,
                          "peak_memory_gb": torch.cuda.max_memory_allocated(device) / 2**30 if cuda else None}
                tried.append({"micro_batch": micro, "fits": True})
            except out_of_memory:
                tried.append({"micro_batch": micro, "fits": False})
            except RuntimeError as error:                    # a full card also fails inside cuBLAS or cuDNN, under another name
                if not any(word in str(error) for word in ("ALLOC_FAILED", "out of memory")):
                    raise
                tried.append({"micro_batch": micro, "fits": False})
            finally:
                del state
                clear()
            if result:
                break
        if result is None:
            raise SystemExit(f"{cfg['key']}: no micro-batch fits the card ({tried})")
        result["tried"], result["other_mib"] = tried, max(other, others())  # what other processes held on the card around the timing
        done[cfg["key"]] = result
        print(f"{cfg['key']:<9} {result['tokens_per_second']:>9,.0f} tokens a second  ({result['passes']} x {result['micro_batch']} sequences a step, "
              f"{result['steps']} steps in {result['seconds']:.1f} s" + (f", peak {result['peak_memory_gb']:.1f} GB" if cuda else "")
              + (f"; OTHER PROCESSES HELD {result['other_mib']} MiB OF THE CARD" if result["other_mib"] > 500 else "") + ")", flush=True)
        save()

    for n in split_at:                                   # `split` takes both, `split48` or `split32` one of them
        if only and "split" not in only and f"split{n}" not in only:
            continue
        other = others()
        s = splits[n] = split(n, seq_len, setup, CONFIGS[-1], sync, split_seconds)
        s["other_mib"] = max(other, others())
        print(f"one step at {s['sequences']} x {seq_len}: forward and backward {s['forward_backward_ms']:.0f} ms, of which the head {s['head_ms']:.0f}; "
              f"optimizer {s['optimizer_ms']:.0f} (AdamW alone {s['optimizer_adamw_ms']:.0f}); loading {s['load_ms']:.2f}", flush=True)
        clear()
        save()
    save()
    print(f"wrote {raw_path}")


def other_mib(card: str, pid: int) -> int:
    """MiB held on card `card` (its index in nvidia-smi) by processes other than `pid`."""
    def query(what: str) -> list[list[str]]:
        out = subprocess.run(["nvidia-smi", what, "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
        return [[x.strip() for x in line.split(",")] for line in out.strip().splitlines() if line.strip()]

    uuid = {row[0]: row[1] for row in query("--query-gpu=index,uuid")}.get(card)
    return sum(int(mib) for p, mib, gpu in query("--query-compute-apps=pid,used_memory,gpu_uuid") if gpu == uuid and int(p) != pid)


def split(sequences: int, seq_len: int, setup, cfg: dict, sync, seconds: float = 3.0) -> dict:
    """One step of the compiled recipe taken apart: every part repeated for about three seconds, timed alone."""
    import torch
    import torch.nn.functional as F

    from ..config import train_preset
    from ..optim import build_optimizers, set_lr

    model, net, optimizers, loader, autocast = setup(cfg, sequences)

    def clock(fn, seconds: float = seconds, warm: int = 3) -> float:
        for _ in range(warm):
            fn()
        sync(); n, start = 0, time.perf_counter()
        while time.perf_counter() - start < seconds or n < 5:
            fn(); n += 1
        sync()
        return 1000 * (time.perf_counter() - start) / n

    x, y = loader.next_batch()

    def forward_backward() -> None:
        with autocast:
            loss = net(x, y)
        loss.backward()

    forward_backward_ms = clock(forward_backward)
    with torch.no_grad(), autocast:  # the final hidden states, as the head receives them
        h = model.embed_tokens(x)
        rot = model.rotation(0, x.size(1))
        for block in model.blocks:
            h = block(h, rot, None)
        h = model.final_norm(h)
    h = h.detach().requires_grad_(True)

    def head_loss(hidden, targets):
        logits = model.unembed_logits(hidden)
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)

    # The head is compiled when the model is, as it is inside the step: left to the interpreter it writes and reads the
    # float32 logits (6 GB at 48 sequences) once for every operation, takes three times as long and is no part of the
    # compiled pass.
    head_net = torch.compile(head_loss) if net is not model else head_loss

    def head_of(hidden, targets):
        def run() -> None:
            with autocast:
                loss = head_net(hidden, targets)
            loss.backward()
        return run

    # If the head alone does not fit the card it is timed on half the sequences and counted twice.
    head_passes, full = 1, True
    try:
        head_ms = clock(head_of(h, y))
    except RuntimeError as error:
        if not isinstance(error, torch.OutOfMemoryError) and not any(word in str(error) for word in ("ALLOC_FAILED", "out of memory")):
            raise
        full = False
    if not full:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        half = sequences // 2
        head_passes = 2
        head_ms = 2 * clock(head_of(h[:half].detach().requires_grad_(True), y[:half]))
    forward_backward()                                   # gradients for the optimizers to use
    optimizer_ms = clock(lambda: [o.step() for o in optimizers])
    adamw = build_optimizers(model, train_preset("gptpp-base").replace(optimizer="adamw"), sequences * seq_len)
    set_lr(adamw, 1.0)
    optimizer_adamw_ms = clock(lambda: [o.step() for o in adamw])
    load_ms = clock(lambda: loader.next_batch(), seconds=min(1.0, seconds))
    return {"sequences": sequences, "seq_len": seq_len, "forward_backward_ms": forward_backward_ms, "head_ms": head_ms, "head_passes": head_passes, "optimizer_ms": optimizer_ms,
            "optimizer_adamw_ms": optimizer_adamw_ms, "load_ms": load_ms}


# ----------------------------------------------------------------------------------------------------------------
# reading


def design_split() -> dict | None:
    """The column of the 12-layer model in the table of docs/DESIGN.md, Section 5.2, in milliseconds."""
    text = (repo_dir() / "docs" / "DESIGN.md").read_text(encoding="utf-8")
    section = text[text.index("### 5.2 Where a step's time goes"):]
    rows = {"forward_backward_ms": r"\| forward and backward \|", "head_ms": r"\| of which unembedding, soft-cap and loss \|",
            "optimizer_ms": r"\| optimizer step, Muon and AdamW \|", "optimizer_adamw_ms": r"\| optimizer step if everything used AdamW \|",
            "load_ms": r"\| loading the batch \|"}
    head = re.search(r"\| `gptpp-base`, (\d+) × ([\d,]+) \|", section)
    out = {"sequences": int(head.group(1)), "seq_len": int(head.group(2).replace(",", "")), "source": "docs/DESIGN.md, Section 5.2"}
    for key, row in rows.items():
        m = re.search(row + r" [\d.]+ ms \| ([\d.]+) ms \|", section)
        if not m:
            return None
        out[key] = float(m.group(1))
    return out


def run_rate(run: str) -> dict | None:
    """Tokens per second of a run of the trainer: its summary, or the median of its log while it is still running."""
    for root in (repo_dir() / "results" / "runs", CKPT):
        summary = root / run / "summary.json"
        if summary.exists():
            s = json.loads(summary.read_text())
            return {"run": run, "tokens_per_second": s["tokens_per_second"], "final": True, "seconds": s["seconds"], "tokens": s["tokens"],
                    "peak_memory_gb": s.get("peak_memory_gb")}
    log = CKPT / run / "log.jsonl"
    if log.exists():
        rates = [json.loads(line)["tok_s"] for line in log.read_text().splitlines() if '"kind": "train"' in line][1:]
        if rates:
            return {"run": run, "tokens_per_second": statistics.median(rates), "final": False, "seconds": None, "tokens": None, "peak_memory_gb": None}
    return None


def export(raw_path: Path) -> dict:
    raw = json.loads(raw_path.read_text()) if raw_path.exists() else None
    bars, numbers = [], {}
    if raw:
        for cfg in CONFIGS:
            c = next((q for q in raw.get("configs", []) if q["key"] == cfg["key"]), None)
            if c is None:
                continue
            bars.append({"key": c["key"], "label": cfg["label"], "tokens_per_second": c["tokens_per_second"], "cards": 1,
                         "source": f"measured here: {c['steps']} steps in {c['seconds']:.0f} s", "passes": c["passes"], "micro_batch": c["micro_batch"],
                         "peak_memory_gb": c["peak_memory_gb"], "main": False, "final": True})
    one = None
    for run, cards in CARD_RUNS:
        r = run_rate(run)
        if r is None:
            continue
        if cards == 1:
            one = r["tokens_per_second"]
        bars.append({"key": f"cards-{cards}", "label": {1: "a whole run on one card", 2: "two cards", 3: "three cards: the main run"}[cards],
                     "tokens_per_second": r["tokens_per_second"], "cards": cards, "source": f"run {run}: " + ("its summary" if r["final"] else "the median of its log so far"),
                     "passes": 1, "micro_batch": SEQUENCES, "peak_memory_gb": r["peak_memory_gb"], "main": run == "gptpp-base", "final": r["final"],
                     "of_linear": r["tokens_per_second"] / (cards * one) if one and cards > 1 else None})
    for prev, bar in zip([None] + bars[:-1], bars):
        bar["factor"] = bar["tokens_per_second"] / prev["tokens_per_second"] if prev else None
    by = {b["key"]: b for b in bars}
    if "float32" in by and "compiled" in by:
        numbers["factor_single"] = by["compiled"]["tokens_per_second"] / by["float32"]["tokens_per_second"]
        steps = [b for b in bars if b["cards"] == 1 and b["factor"] and b["key"] != "cards-1"]
        top = max(steps, key=lambda b: b["factor"])
        numbers["largest_step"], numbers["largest_step_factor"] = top["key"], top["factor"]
        if "cards-1" in by:
            numbers["compiled_over_run"] = by["compiled"]["tokens_per_second"] / by["cards-1"]["tokens_per_second"]
    if "cards-3" in by:
        numbers["three_cards"] = by["cards-3"]["tokens_per_second"]
        numbers["three_cards_of_linear"] = by["cards-3"]["of_linear"]
    if "cards-2" in by:
        numbers["two_cards_of_linear"] = by["cards-2"]["of_linear"]
    split_design = design_split()
    measured = [s for s in (raw or {}).get("split", []) if split_design and s["sequences"] == split_design["sequences"]]
    return {"generated_by": "gptpp.export.where_the_time_goes", "measured": bool(raw and raw.get("configs")), "complete": bool(raw) and len(bars) == len(CONFIGS) + len(CARD_RUNS) and all(b["final"] for b in bars),
            "step": {"sequences": SEQUENCES, "seq_len": SEQ_LEN, "tokens": SEQUENCES * SEQ_LEN}, "machine": (raw or {}).get("machine"), "bars": bars,
            "split": split_design, "split_measured": measured[0] if measured else None, "split_measured_all": (raw or {}).get("split"), "numbers": numbers}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--measure", action="store_true", help="run the timings on the visible card and write the raw file")
    parser.add_argument("--seconds", type=float, default=25.0, help="how long each configuration is timed")
    parser.add_argument("--only", default=None, help="comma-separated configurations (and `split`, or `split48`, `split32`) to measure again; the others are kept")
    parser.add_argument("--raw", default=str(RAW))
    parser.add_argument("--out", default=str(DATA))
    args = parser.parse_args(argv)
    if args.measure:
        measure(args.seconds, args.only.split(",") if args.only else None, Path(args.raw))
    data = export(Path(args.raw))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    size = write_json(out / f"{NAME}.json", data)
    for b in data["bars"]:
        print(f"{b['label']:<56} {b['tokens_per_second']:>9,.0f}" + (f"  x{b['factor']:.2f}" if b["factor"] else "") + (f"  {100 * b['of_linear']:.0f}% of linear" if b.get("of_linear") else "")
              + ("" if b["final"] else "  (not final)"))
    print(f"{NAME}.json {size} bytes; measured: {data['measured']}; complete: {data['complete']}")


if __name__ == "__main__":
    main()
