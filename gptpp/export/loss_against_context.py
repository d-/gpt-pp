"""Export for fig-01-04 `gpt-loss-against-context`: what context buys, measured.

    python -m gptpp.export.loss_against_context --ckpt $GPTPP_WORK/ckpt/gptpp-base             # the run's model.pt
    python -m gptpp.export.loss_against_context --ckpt $GPTPP_WORK/ckpt/gptpp-base/ckpt_13217.pt
    python -m gptpp.export.loss_against_context --ckpt $GPTPP_WORK/ckpt/pico --device cpu
    python -m gptpp.export.loss_against_context --ckpt $GPTPP_WORK/ckpt/gptpp-base --minutes 8   # in pieces: run it again until it says it is done

The same held-out tokens are scored again and again by one model, each time with only the last k tokens before
them visible, for k = 1, 2, 4, ... up to the model's context, and for k = 3 (the table of order 4 sees three tokens,
and the network is set against it with as many). A scored token (a target) is drawn at random among
the held-out tokens that have at least `context` tokens of their own document before them, so that every window
lies inside one document and every k is measured on the same targets. The window of k tokens is fed alone (the
model sees nothing else, and no `<|bos|>` is put in front: the trainer's windows also begin anywhere), and the loss
of the target is read at the last position.

Beside the network, tables: interpolated Kneser-Ney models of orders 1 to 4 counted on the first 200 million
training tokens (`counted_models.KneserNey`, the model of the corpus statistics), scored on the same targets. The
table of order n sees n - 1 tokens, so the tables of orders 2, 3 and 4 can be set against the network with 1, 2
and 3 tokens in view: `atEqualContext` holds the three differences, paired on the targets, with their standard
errors. Their scores on the targets are kept in
`$GPTPP_WORK/ckpt/loss-against-context/`, so that a second checkpoint of the same dataset does not count again.

What the network has been measured on is kept in the same folder as it goes (a file a checkpoint), so the command
can be stopped and run again, and it goes on where it stopped. `--minutes` stops it by itself: on a card shared
with other jobs the pass over 1,024 tokens of context can take longer than one job is allowed, and the measurement
is then made in pieces, each a job of its own. The exit code is 3 while it is not finished.

Bits per byte are bits per token over the bytes per token of the held-out set (`bytes_per_token_val`), the
trainer's convention, in which the validation loss and the n-gram models of the corpus statistics are stated: the
figure's numbers can then be set beside theirs. (The targets themselves are a little shorter than the average
token; `bytesPerTarget` in the file says by how much.)

Each model is one entry of `loss-against-context.json` (`--name`, by default the run's directory name without
`-base`), so the command is run once a model and the file keeps the others. For the model named `gptpp` the
three numbers that the post quotes are also written to `results/interp/part-1.json`, keyed by the post's words.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .. import repo_dir, work_dir
from ..model import GPT
from ..tokenizer import Tokenizer
from .common import DATA, write_atomic
from .counted_models import KneserNey, training_tokens

NAME = "loss-against-context"
LABELS = {"gptpp": "gpt-pp", "pico": "pico"}
CORPORA = {"fineweb-edu": "FineWeb-EDU", "tinystories": "TinyStories"}
QUOTED = {2: "bits per byte of gpt-pp on held-out FineWeb-EDU with the context cut to the last 2 tokens",
          3: "bits per byte of gpt-pp on held-out FineWeb-EDU with the context cut to the last 3 tokens",
          128: "bits per byte of gpt-pp on held-out FineWeb-EDU with the context cut to the last 128 tokens",
          1024: "bits per byte of gpt-pp on the same tokens with its full context of 1,024 tokens"}


def choose_targets(stream: np.ndarray, bos: int, context: int, n: int, seed: int) -> tuple[np.ndarray, float]:
    """n positions of the stream, sorted, each with at least `context` tokens of its own document before it; and
    the share of all held-out tokens that qualify."""
    starts = np.flatnonzero(stream == bos)
    owner = np.searchsorted(starts, np.arange(len(stream)), side="right") - 1
    depth = np.arange(len(stream)) - starts[np.maximum(owner, 0)]
    eligible = np.flatnonzero((owner >= 0) & (depth >= context))
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(eligible, size=min(n, len(eligible)), replace=False)), len(eligible) / len(stream)


@torch.no_grad()
def last_token_loss(model: GPT, windows: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """The loss in nats of each target given its window (B, k): the blocks run on the window, and only the last
    position is read out (the logits of the other positions are never formed)."""
    T = windows.size(1)
    model.extend_positions(T)
    x = model.embed_tokens(windows)
    rot = model.rotation(0, T)
    bias = model.alibi[:, :T, :T] if model.cfg.pos == "alibi" else None
    for block in model.blocks:
        x = block(x, rot, bias)
    logits = model.unembed_logits(model.final_norm(x[:, -1:]))[:, 0]
    return F.cross_entropy(logits.float(), targets, reduction="none")


class NotFinished(Exception):
    """The time given to `measure` ran out. What was measured is kept, and the same call goes on from it."""


def measure(model: GPT, stream: np.ndarray, targets: np.ndarray, ks: list[int], device: str, batch_tokens: int,
            keep: Path | None = None, stamp: str = "", seconds: float | None = None) -> np.ndarray:
    """Bits of every target at every k: (len(ks), len(targets)) float64.

    With `keep`, what is measured is kept in that file, each k as it ends and when the time is up, and a later call
    with the same targets and the same `stamp` (which names the weights) goes on from there, context by context: a
    context that was not asked for before is measured, the others are not measured again. With `seconds`, the call
    raises `NotFinished` once that much time has passed."""
    out = np.zeros((len(ks), len(targets)))
    done = np.zeros(len(ks), dtype=np.int64)                  # how many of the targets are measured, a k
    if keep is not None and keep.exists():
        kept = np.load(keep, allow_pickle=False)
        if str(kept["stamp"]) == stamp and np.array_equal(kept["targets"], targets):
            was = {int(k): j for j, k in enumerate(kept["ks"])}
            for row, k in enumerate(ks):
                if k in was:
                    out[row], done[row] = kept["bits"][was[k]], kept["done"][was[k]]

    def save() -> None:
        if keep is None:
            return
        keep.parent.mkdir(parents=True, exist_ok=True)
        scratch = keep.with_name(keep.stem + ".tmp.npz")
        np.savez(scratch, targets=targets, ks=np.asarray(ks), bits=out, done=done, stamp=np.asarray(stamp))
        scratch.rename(keep)

    cuda = device.startswith("cuda")
    began = time.time()
    for row, k in enumerate(ks):
        if done[row] == len(targets):
            print(f"  k = {k:5d}: {out[row].mean():.4f} bits per token  (kept)", flush=True)
            continue
        step = max(1, min(512, batch_tokens // k))            # at most 512 windows at once: their logits are 512 x vocabulary
        t0 = time.time()
        for a in range(int(done[row]), len(targets), step):
            if seconds is not None and time.time() - began > seconds:
                save()
                raise NotFinished(f"stopped at k = {k}, {a} of {len(targets)} targets")
            pos = targets[a:a + step]
            windows = torch.from_numpy(stream[pos[:, None] - k + np.arange(k)[None, :]].astype(np.int64)).to(device)
            goal = torch.from_numpy(stream[pos].astype(np.int64)).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
                loss = last_token_loss(model, windows, goal)
            out[row, a:a + step] = loss.double().cpu().numpy() / math.log(2.0)
            done[row] = min(len(targets), a + step)
        save()
        print(f"  k = {k:5d}: {out[row].mean():.4f} bits per token  ({time.time() - t0:.1f} s)", flush=True)
    return out


def table_bits(dataset: str, stream: np.ndarray, targets: np.ndarray, vocab: int, context: int, seed: int, sample: int) -> np.ndarray:
    """Bits of every target under the tables of orders 1 to 4: (4, len(targets)); kept on disk a dataset."""
    keep = work_dir() / "ckpt" / NAME / f"{dataset.replace('/', '-')}-K{context}-N{len(targets)}-seed{seed}-S{sample}.npz"
    if keep.exists():
        kept = np.load(keep)
        if np.array_equal(kept["targets"], targets):
            return kept["bits"]
    keep.parent.mkdir(parents=True, exist_ok=True)
    tables = KneserNey(training_tokens(dataset, sample), vocab, 4)
    bits = np.zeros((4, len(targets)))
    for n in range(1, 5):          # the n-gram of a target: its n - 1 predecessors and itself, cut out of the stream
        bits[n - 1] = -np.log2(tables.of_grams(stream[targets[:, None] - (n - 1) + np.arange(n)[None, :]]))
    scratch = keep.with_name(keep.stem + ".tmp.npz")
    np.savez(scratch, targets=targets, bits=bits)
    scratch.rename(keep)
    return bits


def export(ckpt: Path, name: str | None, out: Path, device: str, n_targets: int, batch_tokens: int, seed: int, sample: int, minutes: float | None = None) -> dict:
    began = time.time()
    run = ckpt if ckpt.is_dir() else ckpt.parent
    config = json.loads((run / "config.json").read_text())
    dataset = config["train_config"]["data"]
    data_dir = work_dir() / "data" / dataset
    meta = json.loads((data_dir / "meta.json").read_text())
    stream = np.fromfile(data_dir / meta.get("val_file", "val.bin"), dtype="<u2")
    model, cfg = GPT.from_checkpoint(ckpt, device)
    file = (ckpt / "model.pt") if ckpt.is_dir() else ckpt
    state = torch.load(file, map_location="cpu", weights_only=True)
    step, seen = int(state.get("step", 0)), int(state.get("tokens", 0))
    del state
    context = int(cfg.block_size)
    ks = sorted({2 ** i for i in range(int(math.log2(context)) + 1)} | {3})   # the doublings, and 3: as many tokens as the table of order 4 sees
    targets, share = choose_targets(stream, int(meta["bos_id"]), context, n_targets, seed)
    tok = Tokenizer.load(meta["tokenizer"])
    per_target = float(np.mean([len(tok.token_bytes(int(t))) for t in stream[targets]]))
    per_token = float(meta["bytes_per_token_val"])
    print(f"{run.name}: {file.name}, step {step}, {len(targets)} targets of {dataset} ({share:.1%} of the held-out tokens qualify), {per_target:.3f} bytes a target", flush=True)

    keep = work_dir() / "ckpt" / NAME / f"{run.name}-{file.stem}-N{len(targets)}-seed{seed}.measured.npz"
    stamp = f"{file.name} of step {step}, {file.stat().st_size} bytes, written {file.stat().st_mtime_ns}"
    bits = measure(model, stream, targets, ks, device, batch_tokens, keep, stamp, None if minutes is None else minutes * 60 - (time.time() - began))
    tables = table_bits(dataset, stream, targets, int(meta["vocab_size"]), context, seed, sample)
    n = len(targets)
    def per_byte(b: np.ndarray) -> float:
        return float(b.mean() / per_token)
    def se(b: np.ndarray) -> float:                          # of a mean over the targets
        return float(b.std(ddof=1) / math.sqrt(n) / per_token)
    # what a point buys over the one it is set against: half as many tokens (a doubling), or for 3 the 2 before it
    gain_from = [k // 2 if k % 2 == 0 else ks[ks.index(k) - 1] for k in ks[1:]]
    gain = [per_byte(bits[ks.index(f)]) - per_byte(bits[i + 1]) for i, f in enumerate(gain_from)]
    # per byte, six decimals are written: the figure and the prose show three, and a number rounded twice can be off by one in the last place
    gain_se = [se(bits[ks.index(f)] - bits[i + 1]) for i, f in enumerate(gain_from)]
    # at equal context: the network with 1, 2 and 3 tokens in view against the table that sees as many (orders 2, 3
    # and 4, rows 1, 2 and 3), paired on the targets; above zero, the table is ahead
    equal = {"sees": [1, 2, 3], "networkMinusTable": [round(per_byte(bits[ks.index(k)] - tables[k]), 6) for k in (1, 2, 3)],
             "se": [round(se(bits[ks.index(k)] - tables[k]), 6) for k in (1, 2, 3)]}
    stats = json.loads((data_dir / "stats.json").read_text())
    whole = [float(o["held_out_bits_per_byte"]) for o in stats["ngrams"]["orders"]]
    key = name or run.name.replace("-base", "").replace("-", "")
    entry = {"label": LABELS.get(key, key), "run": run.name, "checkpoint": file.name, "final": file.name == "model.pt", "step": step, "tokensSeen": seen,
             "corpus": CORPORA.get(dataset.split("/")[0], dataset), "context": context, "targets": n, "eligibleShare": round(share, 4), "bytesPerToken": round(per_token, 4), "bytesPerTarget": round(per_target, 4),
             "seed": seed, "k": ks, "bitsPerToken": [round(float(b.mean()), 4) for b in bits], "bitsPerByte": [round(per_byte(b), 6) for b in bits],
             "se": [round(se(b), 6) for b in bits], "gain": [round(g, 6) for g in gain], "gainSe": [round(g, 6) for g in gain_se], "gainFrom": gain_from,
             "tables": {"orders": [1, 2, 3, 4], "sees": [0, 1, 2, 3], "countedOn": sample, "bitsPerByte": [round(per_byte(b), 6) for b in tables],
                        "bitsPerToken": [round(float(b.mean()), 4) for b in tables], "se": [round(se(b), 6) for b in tables], "wholeStream": [round(v, 4) for v in whole]},
             "atEqualContext": equal}
    path = out / f"{NAME}.json"
    data = json.loads(path.read_text()) if path.exists() else {"name": NAME, "models": {}}
    data["models"][key] = entry
    data["protocol"] = ("targets: held-out tokens with a full context of their own document before them, drawn at random; the model sees the last k tokens "
                        "and nothing else; tables: interpolated Kneser-Ney, orders 1 to 4, on the same targets; bits per byte: bits per token over the held-out set's bytes per token")
    write_atomic(path, json.dumps(data, separators=(",", ":")).encode("utf-8"))
    if key == "gptpp":
        quoted = repo_dir() / "results" / "interp" / "part-1.json"
        quoted.parent.mkdir(parents=True, exist_ok=True)
        numbers = json.loads(quoted.read_text()) if quoted.exists() else {}
        for k, words in QUOTED.items():
            numbers[words] = round(entry["bitsPerByte"][ks.index(k)], 4)
        numbers["bits per byte of the 4-gram table on the same tokens (it sees the same 3 tokens)"] = round(entry["tables"]["bitsPerByte"][3], 4)
        numbers["gpt-pp with the last 3 tokens minus the 4-gram table on the same tokens, bits per byte (paired; below zero, the network is ahead)"] = round(equal["networkMinusTable"][2], 4)
        numbers["standard error of gpt-pp with the last 3 tokens minus the 4-gram table, bits per byte"] = round(equal["se"][2], 4)
        numbers["_loss_against_context"] = {"checkpoint": f"{run.name}/{file.name}", "step": step, "final": entry["final"], "targets": n,
                                             "tokens in view": ks, "bits per byte": entry["bitsPerByte"], "bits per token": entry["bitsPerToken"],
                                             "what each point buys over the one it is set against, bits per byte": entry["gain"], "its standard error": entry["gainSe"],
                                             "the tokens in view of the point it is set against (half as many; for 3, 2)": gain_from,
                                             "tables on the same targets, orders 1 to 4, bits per byte": entry["tables"]["bitsPerByte"],
                                             "tables on all held-out tokens (corpus statistics)": entry["tables"]["wholeStream"],
                                             "gpt-pp minus the table that sees as many tokens, with 1, 2 and 3 in view, bits per byte": equal["networkMinusTable"],
                                             "its standard error, paired on the targets": equal["se"],
                                             "convention": "bits per token over the held-out set's bytes per token, as the trainer's val_bpb"}
        write_atomic(quoted, json.dumps(numbers, indent=1, ensure_ascii=False).encode("utf-8"))
    return entry


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True, help="a run directory (its model.pt) or a checkpoint file of a run")
    ap.add_argument("--name", default=None, help="the key of this model in the file (default: the run's name without -base)")
    ap.add_argument("--targets", type=int, default=12288, help="held-out tokens scored at every context")
    ap.add_argument("--batch-tokens", type=int, default=32768, help="tokens in a batch of windows")
    ap.add_argument("--table-tokens", type=int, default=200_000_000, help="training tokens the tables are counted on")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--minutes", type=float, default=None, help="stop by itself after this long, keeping what is measured; the same command then goes on")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(DATA))
    args = ap.parse_args()
    try:
        entry = export(Path(args.ckpt), args.name, Path(args.out), args.device, args.targets, args.batch_tokens, args.seed, args.table_tokens, args.minutes)
    except NotFinished as e:
        print(f"NOT FINISHED ({e}); nothing is written yet: run the same command again", flush=True)
        sys.exit(3)
    print(json.dumps({k: entry[k] for k in ("label", "checkpoint", "step", "k", "bitsPerByte", "se", "gain", "gainSe", "tables", "atEqualContext")}))


if __name__ == "__main__":
    main()
