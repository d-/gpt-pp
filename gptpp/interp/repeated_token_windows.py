"""A corpus with windows of repeated random tokens mixed in: the training data of the circuit models.

    python -m gptpp.interp.repeated_token_windows --data tinystories/bpe4k --name tinystories-repeats --fraction 0.1

Why. An induction head completes [A][B] ... [A] with [B]: it pays where text repeats itself in ways that its
bigrams do not already predict. Children's stories written in a small vocabulary repeat little of that kind (on
TinyStories, of the tokens whose predecessor occurred earlier in the context, 17 % continue as they did the last
time, and a one-layer model already predicts three quarters of those), and a two-layer attention-only model trained
on the stories alone grows only a faint induction head. The models of "In-context learning and induction heads"
(Olsson et al., 2022) were trained on web text, which is full of such repeats (names, numbers, code, quotations).
This script adds them in the plainest form: one training window in ten is replaced by the beginning-of-text token
followed by a block of 8 to 64 random tokens repeated to the end of the window, the kind of sequence the paper
uses to test for induction heads. The held-out stream is the corpus's own, unchanged, so the held-out loss of a
model trained on the mixture is comparable with one trained on the corpus alone.

What it writes, in $GPTPP_WORK/data/<name>/<tokenizer>/, is a dataset in the lab's format (Contract 1): the
training shards with the chosen windows overwritten in place, on the grid the trainer cuts windows on (window k of
a shard is tokens k * 256 to k * 256 + 255), val.bin copied, and meta.json with a `repeats` record: the fraction,
the block lengths, the seed and the counts; and repeats_pool.json, the token ids the random tokens were drawn from
(every token seen at least 1,000 times in training, less the 100 most frequent). The trainer shuffles windows on
that grid, so a model with context 256 sees each replaced window whole.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from .. import work_dir
from ..data import TOKEN_DTYPE, TokenData
from .circuits import token_pool

POOL_FILE = "repeats_pool.json"


def token_counts(data: TokenData) -> np.ndarray:
    """How often each token occurs in the training shards, (V,)."""
    counts = np.zeros(data.vocab_size, dtype=np.int64)
    for i in range(len(data.train_files)):
        counts += np.bincount(np.asarray(data.shard(i)), minlength=data.vocab_size)
    return counts


def repeated_window(pool: np.ndarray, bos_id: int, window: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """One window: the beginning-of-text token, then `block` tokens drawn without replacement from the pool,
    repeated until the window is full (the last copy is cut where the window ends)."""
    tokens = rng.choice(pool, size=block, replace=False)
    return np.concatenate(([bos_id], np.tile(tokens, (window - 1) // block + 1)))[:window]


def mix_shard(tokens: np.ndarray, pool: np.ndarray, bos_id: int, fraction: float, window: int, min_block: int,
              max_block: int, rng: np.random.Generator) -> tuple[np.ndarray, int, int]:
    """A copy of a shard with a random `fraction` of its windows replaced; also the number of windows and of
    windows replaced."""
    out = np.array(tokens, dtype=TOKEN_DTYPE)
    n_windows = (len(out) - 1) // window
    chosen = np.flatnonzero(rng.random(n_windows) < fraction)
    for k in chosen:
        block = int(rng.integers(min_block, max_block + 1))
        out[k * window:(k + 1) * window] = repeated_window(pool, bos_id, window, block, rng)
    return out, n_windows, len(chosen)


def write_mixture(source: TokenData, out: Path, name: str, fraction: float = 0.1, window: int = 256,
                  min_block: int = 8, max_block: int = 64, drop_most: int = 100, min_count: int = 1000,
                  seed: int = 0) -> dict:
    """Write the mixed dataset to `out` and return its meta."""
    out.mkdir(parents=True, exist_ok=True)
    counts = token_counts(source)
    pool = token_pool(torch.from_numpy(counts), drop_most, min_count).numpy()
    rng = np.random.default_rng(seed)
    n_windows = n_replaced = 0
    for i, path in enumerate(source.train_files):
        mixed, windows, replaced = mix_shard(np.asarray(source.shard(i)), pool.astype(TOKEN_DTYPE),
                                             int(source.meta["bos_id"]), fraction, window, min_block, max_block, rng)
        mixed.tofile(out / path.name)
        n_windows, n_replaced = n_windows + windows, n_replaced + replaced
    shutil.copyfile(source.val_file, out / source.val_file.name)
    meta = dict(source.meta)
    meta.update(
        dataset=name,
        val_docs_file=None,  # the held-out documents are the source's
        repeats={
            "source": str(source.dir), "fraction": fraction, "window": window, "min_block": min_block,
            "max_block": max_block, "seed": seed, "windows": n_windows, "windows_replaced": n_replaced,
            "drop_most": drop_most, "min_count": min_count, "pool_file": POOL_FILE, "pool_size": len(pool),
            "note": "a replaced window is <|bos|> and a block of random tokens of the pool, repeated to the end of "
                    "the window; val.bin is the source's, unchanged",
        },
    )
    (out / POOL_FILE).write_text(json.dumps(pool.tolist()) + "\n")
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def load_pool(data: TokenData) -> torch.Tensor:
    """The token ids random sequences are drawn from, for a dataset written by this script: the pool it stored.
    For any other dataset, the same rule applied to its own counts."""
    record = data.meta.get("repeats")
    if record:
        return torch.tensor(json.loads((data.dir / record["pool_file"]).read_text()))
    return token_pool(torch.from_numpy(token_counts(data)), drop_most=100, min_count=1000)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="tinystories/bpe4k", help="the corpus: <dataset>/<tokenizer> or a directory")
    parser.add_argument("--name", default="tinystories-repeats", help="name of the mixed dataset")
    parser.add_argument("--fraction", type=float, default=0.1, help="share of the training windows replaced")
    parser.add_argument("--window", type=int, default=256, help="the context length of the models to be trained")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    source = TokenData(args.data)
    out = work_dir() / "data" / args.name / source.dir.name
    meta = write_mixture(source, out, args.name, args.fraction, args.window, seed=args.seed)
    record = meta["repeats"]
    print(f"{out}: {record['windows_replaced']:,} of {record['windows']:,} windows of {args.window} tokens replaced "
          f"({record['windows_replaced'] / record['windows']:.4f}); pool of {record['pool_size']:,} tokens")


if __name__ == "__main__":
    main()
