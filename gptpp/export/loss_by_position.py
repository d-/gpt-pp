"""Export for fig-08-03 `gpt-loss-against-position`: the loss at each position of the context, through training.

    python -m gptpp.export.loss_by_position --run $GPTPP_WORK/ckpt/pico
    python -m gptpp.export.loss_by_position --run $GPTPP_WORK/ckpt/gptpp-base --windows 384 --batch 4 --budget 480

For every checkpoint of the run (`ckpt_<step>.pt`, and `model.pt` once the run has ended), two curves of loss
against position:

    text       windows of held-out text, each inside one document and begun at a random place in it: the loss of
               the token at position t, predicted from the t tokens of the window before it, averaged over the
               windows. Every position holds the same kind of text; only the amount of context differs, so the
               curve falls with t exactly as far as the model uses its context.
    repeated   `<|bos|>`, then h tokens drawn at random, then the same h tokens again (h = 50, as in Olsson et
               al., 2022). Nothing predicts the first copy. The second is predictable only by looking back for the
               last occurrence of the current token and copying what followed it: in-context learning in its
               barest form, the behavior of an induction head. The random tokens are uniform over the tokens of
               the vocabulary that occur in the training set, special tokens excluded.

The loss is in bits per token and is computed as the trainer evaluates (bfloat16 autocast on a card).

A checkpoint's two curves are kept in `$GPTPP_WORK/ckpt/loss-by-position/`, so running the command again measures
only the checkpoints that are new (a run still in training) and then writes the files; `--budget 480` begins no
new checkpoint after that many seconds, and the same command continues.

Writes `loss-against-position-<run>.json` (steps, tokens seen, the means of each copy, the two positions of the
in-context learning score) and `loss-against-position-<run>.bin` with two float16 arrays, (checkpoints, context)
and (checkpoints, 2h), and their entries for the manifest (`python -m gptpp.export.manifest` merges them).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from ..config import ModelConfig
from ..data import TokenData
from ..model import GPT
from .common import CKPT, DATA, Blob, write_json

NAME = "loss-against-position"


def checkpoints(run: Path) -> list[tuple[int, Path]]:
    """The run's `ckpt_<step>.pt` files by step, then `model.pt` if the run has ended at a later step."""
    found = sorted((int(m.group(1)), f) for f in run.glob("ckpt_*.pt") if (m := re.fullmatch(r"ckpt_(\d+)\.pt", f.name)))
    final, summary = run / "model.pt", run / "summary.json"
    if final.exists() and summary.exists():
        step = int(json.loads(summary.read_text())["steps"])
        if not found or step > found[-1][0]:
            found.append((step, final))
    return found


def document_windows(data: TokenData, context: int, n: int, seed: int, margin: int = 32) -> Tensor:
    """n windows of context + 1 held-out tokens, each inside one document and begun at a random place after its
    `<|bos|>`: (n, context + 1). The first n documents long enough (context + margin tokens of text) give one
    window each."""
    tokens = np.asarray(data.val_tokens()).astype(np.int64)
    starts = np.flatnonzero(tokens == data.meta["bos_id"])
    lengths = np.diff(np.append(starts, len(tokens)))                       # `<|bos|>` included
    long = np.flatnonzero(lengths >= context + 1 + margin)[:n]
    rng = np.random.default_rng(seed)
    offsets = np.array([rng.integers(1, lengths[d] - context) for d in long])   # the window ends inside the document
    return torch.from_numpy(tokens[(starts[long] + offsets)[:, None] + np.arange(context + 1)[None, :]])


def drawable_tokens(data: TokenData) -> Tensor:
    """The ids a random sequence is drawn from: every token that occurs in the training set, except the special ones."""
    stats = json.loads((data.dir / "stats.json").read_text())["rank_frequency"]
    special = set(range(data.vocab_size - 5, data.vocab_size))
    return torch.tensor(sorted(i for i, count in zip(stats["ids"], stats["counts"]) if count > 0 and i not in special))


def repeated_sequences(pool: Tensor, bos: int, half: int, n: int, seed: int) -> Tensor:
    """n sequences `<|bos|>` r_1 .. r_h r_1 .. r_h with the r drawn uniformly from the pool: (n, 1 + 2 half)."""
    generator = torch.Generator().manual_seed(seed)
    first = pool[torch.randint(len(pool), (n, half), generator=generator)]
    return torch.cat((torch.full((n, 1), bos), first, first), dim=1)


@torch.no_grad()
def loss_by_position(model: GPT, sequences: Tensor, batch: int) -> Tensor:
    """The mean loss in nats of each predicted position of sequences (N, L): (L - 1,) float64; entry t - 1 is the
    token at position t, predicted from the t tokens before it."""
    total = torch.zeros(sequences.size(1) - 1, dtype=torch.float64, device=sequences.device)
    cuda = sequences.device.type == "cuda"
    for start in range(0, sequences.size(0), batch):
        chunk = sequences[start:start + batch]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
            logits = model(chunk[:, :-1])
        loss = F.cross_entropy(logits.float().transpose(1, 2), chunk[:, 1:], reduction="none")
        total += loss.double().sum(dim=0)
    return total / sequences.size(0)


def load(path: Path, device: str) -> tuple[GPT, int, int]:
    """A checkpoint of the trainer, read once: the model in evaluation mode, its step, the tokens it has seen."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = GPT(ModelConfig.from_dict(checkpoint["model_config"]))
    model.load_state_dict({re.sub(r"^(module\.|_orig_mod\.)+", "", k): v for k, v in checkpoint["model"].items()})
    return model.to(device).eval(), int(checkpoint["step"]), int(checkpoint["tokens"])


def export(run: Path, out: Path, device: str, windows: int, sequences: int, half: int, batch: int, seed: int,
           name: str | None, budget: float | None) -> bool:
    """Measure the checkpoints that are not in the cache yet, then write the figure's files with every checkpoint
    measured so far. False if the budget ran out before the last one (the files then lack the later checkpoints
    and say so; the same command continues)."""
    started = time.time()
    config = json.loads((run / "config.json").read_text())
    data = TokenData(config["train_config"]["data"])
    cfg = config["model_config"]
    context = cfg["block_size"]
    text = document_windows(data, context, windows, seed).to(device)
    pool = drawable_tokens(data)
    repeated = repeated_sequences(pool, data.meta["bos_id"], half, sequences, seed).to(device)
    bits = 1 / math.log(2)
    cache = CKPT / "loss-by-position" / f"{run.name}-w{text.size(0)}-r{sequences}x{half}-seed{seed}"
    cache.mkdir(parents=True, exist_ok=True)
    steps, tokens, on_text, on_repeat, missing = [], [], [], [], []
    for step, path in checkpoints(run):
        kept = cache / f"{step}.npz"
        if not kept.exists():
            if missing or (budget is not None and time.time() - started > budget):
                if not missing:
                    print(f"stopped before step {step}: the budget of {budget:.0f} s is spent; run the same command again to continue")
                missing.append(step)
                continue
            t0 = time.time()
            try:
                model, _, seen = load(path, device)
            except Exception as error:                  # a checkpoint still being written by a run in progress
                print(f"skipped {path.name}: {type(error).__name__}: {error}")
                continue
            t1 = time.time()
            a = (loss_by_position(model, text, batch) * bits).cpu().numpy()
            b = (loss_by_position(model, repeated, 4 * batch) * bits).cpu().numpy()
            del model
            np.savez(kept.with_suffix(".tmp.npz"), text=a, repeated=b, tokens=seen)
            kept.with_suffix(".tmp.npz").replace(kept)
            print(f"{run.name} step {step:7d}  text {a.mean():6.3f} bits (position 1: {a[0]:.2f}, last: {a[-8:].mean():.2f})  "
                  f"repeated: first copy {b[:half].mean():6.3f}, second copy {b[half:].mean():6.3f}  "
                  f"[{t1 - t0:.0f} s to load, {time.time() - t1:.0f} s to measure]", flush=True)
        kept_arrays = np.load(kept)
        steps.append(step), tokens.append(int(kept_arrays["tokens"]))
        on_text.append(kept_arrays["text"]), on_repeat.append(kept_arrays["repeated"])
    short = name or run.name
    file = f"{NAME}-{short}"
    blob = Blob(file, out)
    on_text, on_repeat = np.stack(on_text), np.stack(on_repeat)
    blob.put(f"{file}.text", on_text.astype(np.float16),
             f"loss in bits per token of position t = 1 .. {context} of windows of held-out text, one row a checkpoint")
    blob.put(f"{file}.repeated", on_repeat.astype(np.float16),
             f"loss in bits per token of position t = 1 .. {2 * half} of bos + {half} random tokens + the same again, one row a checkpoint")
    size = blob.save()
    early, late = max(1, round(50 * context / 1024)), round(500 * context / 1024)   # Olsson et al.: tokens 50 and 500 of 1,024
    meta = {
        "run": short, "preset": config.get("preset"), "data": config["train_config"]["data"],
        "layers": cfg["n_layer"], "width": cfg["d_model"], "context": context, "vocabulary": cfg["vocab_size"],
        "half": half, "windows": int(text.size(0)), "sequences": sequences, "pool": int(len(pool)), "seed": seed,
        "unit": "bits per token", "uniform": math.log2(len(pool)),
        "complete": not missing, "steps_not_measured_yet": missing,
        "steps": steps, "tokens": tokens, "total_steps": int(json.loads((run / "summary.json").read_text())["steps"]) if (run / "summary.json").exists() else None,
        "text_mean": on_text.mean(axis=1), "repeat_first": on_repeat[:, :half].mean(axis=1),
        # the second copy without its first token, which nothing announces
        "repeat_second": on_repeat[:, half + 1:].mean(axis=1),
        "score_positions": [early, late],
        "text_early": on_text[:, max(0, early - 1 - early // 4):early + early // 4].mean(axis=1),
        "text_late": on_text[:, late - 1 - late // 4:late + late // 4].mean(axis=1),
        "keys": {"text": f"{file}.text", "repeated": f"{file}.repeated"},
    }
    json_size = write_json(out / f"{file}.json", meta)
    print(f"wrote {file}.bin ({size:,} bytes) and {file}.json ({json_size:,} bytes): {len(steps)} checkpoints, "
          f"{text.size(0)} windows of text, {sequences} repeated sequences of 2 x {half}"
          + (f"; NOT YET MEASURED: steps {missing}" if missing else ""))
    return not missing


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="a run directory with ckpt_<step>.pt files")
    ap.add_argument("--out", default=str(DATA))
    ap.add_argument("--name", default=None, help="the name in the file names (default: the run directory's)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--windows", type=int, default=1024, help="windows of held-out text, one a document")
    ap.add_argument("--sequences", type=int, default=256, help="repeated random sequences")
    ap.add_argument("--half", type=int, default=50, help="random tokens in a sequence before it repeats")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=float, default=None, help="seconds after which no new checkpoint is begun")
    args = ap.parse_args()
    torch.set_num_threads(4)
    done = export(Path(args.run), Path(args.out), args.device, args.windows, args.sequences, args.half, args.batch,
                  args.seed, args.name, args.budget)
    raise SystemExit(0 if done else 3)


if __name__ == "__main__":
    main()
