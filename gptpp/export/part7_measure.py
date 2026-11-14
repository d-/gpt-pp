"""Measure, on the trained main model, the few things the post of part 7 quotes that no other file holds:
`results/interp/part-7-measured.json`.

    CUDA_VISIBLE_DEVICES=<k> python -m gptpp.export.part7_measure [--run gptpp-base] [--ckpt model.pt]
                                                                  [--out results/interp/part-7-measured.json]
                                                                  [--only val,writing,samples] [--batch 16]

One short job on one card (about a minute; it also runs on a card that other jobs share, and says so). It is the
measuring half of `gptpp.export.part7_numbers`, which only reads. What it measures, field by field:

    model            {run, checkpoint, step, tokens, params}: what was loaded
    machine          {device, torch, cuda, shared (other processes held memory on the card), other_mib, when}
    val              the trainer's monitor, repeated at two window lengths, in the trainer's arithmetic (bfloat16
                     autocast): the first 1,048,576 tokens of the held-out stream cut into windows of `seq_len`,
                     the mean loss of every position of every window
      w512             {seq_len: 512, windows, tokens, val_loss (nats per token), val_bpb}: the protocol of the
                       scaling grid, whose models read windows of 512
      w1024            the same at 1,024: the main run's own monitor (it must reproduce summary.json's val_loss)
    writing          tokens per second while the model writes `new_tokens` tokens after a short prompt, batch 1,
                     temperature 0.8, bfloat16 autocast, each way timed `repeats` times after one warm-up
      cache            {tokens_per_second (the best repeat), all: [each repeat], seconds}: `gptpp.generate.generate`,
                       which runs the prompt once and then one token a step on the cached keys and values
      no_cache         the same text written by running the whole context through the fast path at every step
      same_tokens      true when the two ways wrote the same tokens; they draw with the same seed, and part when the
                       rounding of the two paths moves a draw across a boundary: tokens_in_common says how far they agree
      prompt, new_tokens, repeats
      other_mib        the most that other processes held on the card around the timing (MiB): above 500, time it again
    samples          the four continuations of the post's Section 7.7, each {prompt, settings, tokens (ids), text
                     (what was written after the prompt)}; and
      brace            {prompt, token "{", id, probability (float32), probability_bf16}: what the model gives a
                       token that its training text never contained

Seeds are fixed (0), so the samples are the same on every run of the same checkpoint and arithmetic.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import torch

from .. import repo_dir, work_dir
from ..data import TokenData, ValLoader
from ..generate import generate, load_tokenizer, sample
from ..model import GPT

PROMPT_WRITING = "The history of the bicycle begins"
SAMPLES = [
    {"key": "gold", "prompt": "The chemical symbol for gold is", "new_tokens": 12, "temperature": 0.0},
    {"key": "code", "prompt": "def f(x):\n    return ", "new_tokens": 16, "temperature": 0.8},
    {"key": "poem", "prompt": "Write a poem about the sea.", "new_tokens": 40, "temperature": 0.8},
    {"key": "sum", "prompt": "Question: What is 17 + 26?\nAnswer:", "new_tokens": 8, "temperature": 0.0},
]


def other_memory_mib() -> int | None:
    """MiB that other processes hold on the visible card, or None if nvidia-smi cannot be asked."""
    try:
        import os

        from .where_the_time_goes import other_mib

        return other_mib(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0], os.getpid()) if torch.cuda.is_available() else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


@torch.no_grad()
def monitor(model: GPT, data: TokenData, seq_len: int, batch: int, device: torch.device) -> dict:
    """The trainer's evaluation (train.py, `evaluate`) at a window length: mean loss over the fixed windows."""
    loader = ValLoader(data, seq_len, batch, 0, 1, 1048576.0, device)
    total = torch.zeros(2, dtype=torch.float64, device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for x, y, n_targets in loader:
            total[0] += model(x, y).double() * n_targets
            total[1] += n_targets
    loss = (total[0] / total[1]).item()
    bytes_per_token = data.meta.get("bytes_per_token_val")
    return {"seq_len": seq_len, "windows": loader.n_windows, "tokens": int(total[1].item()), "val_loss": loss,
            "val_bpb": loss / math.log(2) / bytes_per_token if bytes_per_token else None}


@torch.no_grad()
def write_without_cache(model: GPT, idx: torch.Tensor, max_new_tokens: int, temperature: float, seed: int) -> torch.Tensor:
    """What `generate` writes, computed the slow way: the whole context through the fast path at every step. The
    context is cut exactly as the cached generator cuts it, so the tokens are the same."""
    generator = torch.Generator(device=idx.device).manual_seed(seed)
    window = model.cfg.block_size
    start = max(0, idx.size(1) - window)          # where the context that the cache would hold begins
    for _ in range(max_new_tokens):
        if idx.size(1) - start > window:          # the cache was full: it is rebuilt from the most recent half
            start = idx.size(1) - max(1, window // 2)
        logits = model(idx[:, start:])[:, -1]
        idx = torch.cat((idx, sample(logits, temperature, generator=generator)), dim=1)
    return idx


def timed(fn, repeats: int, sync) -> tuple[list[float], torch.Tensor]:
    out, seconds = fn(), []                        # one warm-up, not timed
    for _ in range(repeats):
        sync()
        t = time.perf_counter()
        out = fn()
        sync()
        seconds.append(time.perf_counter() - t)
    return seconds, out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default="gptpp-base", help="run name under $GPTPP_WORK/ckpt, or a run directory")
    parser.add_argument("--ckpt", default="model.pt", help="checkpoint file inside the run")
    parser.add_argument("--out", default=str(repo_dir() / "results" / "interp" / "part-7-measured.json"))
    parser.add_argument("--only", default="val,writing,samples", help="which parts to measure; the others are kept from the file")
    parser.add_argument("--batch", type=int, default=16, help="windows a batch in the monitor")
    parser.add_argument("--new-tokens", type=int, default=1024, help="tokens written when the writing speed is timed")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    parts = set(args.only.split(","))
    device = torch.device(args.device)
    run_dir = Path(args.run) if Path(args.run).is_dir() else work_dir() / "ckpt" / args.run
    config = json.loads((run_dir / "config.json").read_text())
    model, cfg = GPT.from_checkpoint(run_dir / args.ckpt, device=device)
    checkpoint = torch.load(run_dir / args.ckpt, map_location="cpu", weights_only=True)
    tokenizer = load_tokenizer(config["data_meta"]["tokenizer"])
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    out_path = Path(args.out)
    result = json.loads(out_path.read_text()) if out_path.exists() else {}
    other = other_memory_mib()
    result["model"] = {"run": run_dir.name, "checkpoint": args.ckpt, "step": int(checkpoint["step"]), "tokens": int(checkpoint["tokens"]),
                       "params": model.num_params()["total"]}
    result["machine"] = {"device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu", "torch": torch.__version__,
                         "cuda": torch.version.cuda, "shared": bool(other and other > 500), "other_mib": other,
                         "when": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}

    if "val" in parts:
        data = TokenData(config["train_config"]["data"])
        result["val"] = {"w512": monitor(model, data, 512, 2 * args.batch, device), "w1024": monitor(model, data, 1024, args.batch, device),
                         "note": "the trainer's monitor: the first 1,048,576 held-out tokens in windows of seq_len, bfloat16 autocast"}
        print(f"monitor: {result['val']['w512']['val_loss']:.4f} nats in windows of 512, {result['val']['w1024']['val_loss']:.4f} in windows of 1,024")

    if "writing" in parts:
        prompt = torch.tensor([tokenizer.encode(PROMPT_WRITING, bos=True)], dtype=torch.long, device=device)
        with autocast:
            fast, a = timed(lambda: generate(model, prompt, args.new_tokens, temperature=0.8, seed=0), args.repeats, sync)
            slow, b = timed(lambda: write_without_cache(model, prompt, args.new_tokens, 0.8, 0), args.repeats, sync)
        same = bool(torch.equal(a, b))
        agree = int((a == b).long().cumprod(dim=1).sum().item()) - prompt.size(1)   # tokens written before the two texts part
        result["writing"] = {
            "prompt": PROMPT_WRITING, "new_tokens": args.new_tokens, "repeats": args.repeats, "temperature": 0.8,
            "cache": {"tokens_per_second": args.new_tokens / min(fast), "all": [args.new_tokens / s for s in fast], "seconds": min(fast)},
            "no_cache": {"tokens_per_second": args.new_tokens / min(slow), "all": [args.new_tokens / s for s in slow], "seconds": min(slow)},
            "same_tokens": same, "tokens_in_common": agree, "other_mib": max(other or 0, other_memory_mib() or 0),
            "note": "batch 1, bfloat16 autocast; the best of the repeats; without the cache the whole context goes through the fused path at every step"}
        print(f"writing {args.new_tokens} tokens: {result['writing']['cache']['tokens_per_second']:.1f} a second with the cache, "
              f"{result['writing']['no_cache']['tokens_per_second']:.1f} without; the same tokens: {same} ({agree} in common)")

    if "samples" in parts:
        samples = {}
        for s in SAMPLES:
            ids = tokenizer.encode(s["prompt"], bos=True)
            idx = torch.tensor([ids], dtype=torch.long, device=device)
            with autocast:
                written = generate(model, idx, s["new_tokens"], temperature=s["temperature"], seed=0)[0, len(ids):].tolist()
            samples[s["key"]] = {"prompt": s["prompt"], "settings": {"new_tokens": s["new_tokens"], "temperature": s["temperature"], "seed": 0,
                                                                     "decoding": "greedy" if s["temperature"] == 0 else "sampling, nothing cut"},
                                 "tokens": written, "text": tokenizer.decode(written)}
            print(f"{s['prompt']!r} -> {samples[s['key']]['text']!r}")
        prompt = next(s["prompt"] for s in SAMPLES if s["key"] == "code")
        idx = torch.tensor([tokenizer.encode(prompt, bos=True)], dtype=torch.long, device=device)
        brace = tokenizer.encode("{", bos=False)
        with torch.no_grad():
            p32 = torch.softmax(model(idx)[0, -1].float(), dim=-1)
            with autocast:
                p16 = torch.softmax(model(idx)[0, -1].float(), dim=-1)
        samples["brace"] = {"prompt": prompt, "token": "{", "id": brace[0] if len(brace) == 1 else brace,
                            "probability": float(p32[brace[0]]), "probability_bf16": float(p16[brace[0]]),
                            "rank": int((p32 > p32[brace[0]]).sum().item()) + 1, "uniform": 1.0 / p32.numel(),
                            "likeliest": tokenizer.decode([int(p32.argmax())]), "likeliest_probability": float(p32.max())}
        result["samples"] = samples
        print(f"p('{{') = {samples['brace']['probability']:.3e} (rank {samples['brace']['rank']} of {p32.numel()})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
