"""Token shards on disk and the loaders that read them.

The format (Contract 1 of the lab), under $GPTPP_WORK/data/<dataset>/<tokenizer>/:

    train_0000.bin, train_0001.bin, ...   uint16 little-endian token ids, about 100M tokens a shard
    val.bin                               uint16, held-out documents
    meta.json                             vocab_size, bos_id, n_train_tokens, n_val_tokens, train_shards,
                                          bytes_per_token_val, ...

Every document starts with the BOS token and documents are concatenated in a shuffled order, so the stream is
cut into windows of seq_len tokens without regard to document boundaries: a window's targets are its tokens
shifted by one, and consecutive windows share one token so that every token is a target once.

    TrainLoader   one pass over the shards in order, without replacement. Inside a shard the windows are
                  visited in an order shuffled by the seed, so that a batch mixes documents from the whole
                  shard. A step's windows depend on the seed and on the number of windows a step takes, not on
                  how they are split among cards: rank r of n takes the r-th slice of each step.
    ValLoader     a fixed set of held-out windows, the same for every run on the same data at the same
                  seq_len, whatever the batch size or the number of cards.

The shards are memory-mapped. A batch is assembled on the CPU, while the card is still busy with the previous
step, into one page-locked (pinned) buffer that is allocated once, and copied to the card from there.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional, Sequence, Union

import numpy as np
import torch
from torch import Tensor

from . import work_dir

TOKEN_DTYPE = np.dtype("<u2")


class TokenData:
    """A tokenized dataset on disk: its meta, its training shards, its held-out tokens."""

    def __init__(self, name: Union[str, Path]):
        direct = Path(name)
        self.dir = direct if (direct / "meta.json").exists() else work_dir() / "data" / name
        if not (self.dir / "meta.json").exists():
            raise FileNotFoundError(f"no dataset at {direct} or {self.dir} (meta.json not found)")
        self.meta: dict = json.loads((self.dir / "meta.json").read_text())
        names = self.meta.get("train_shards") or sorted(p.name for p in self.dir.glob("train_*.bin"))
        self.train_files = [self.dir / name for name in names]
        self.val_file = self.dir / self.meta.get("val_file", "val.bin")
        if not self.train_files:
            raise FileNotFoundError(f"{self.dir} has no training shards")

    @property
    def vocab_size(self) -> int:
        return int(self.meta["vocab_size"])

    def shard(self, i: int) -> np.memmap:
        return np.memmap(self.train_files[i], dtype=TOKEN_DTYPE, mode="r")

    def val_tokens(self) -> np.memmap:
        return np.memmap(self.val_file, dtype=TOKEN_DTYPE, mode="r")


def _windows(tokens: np.ndarray, starts: np.ndarray, seq_len: int) -> np.ndarray:
    """The windows of seq_len + 1 tokens that begin at `starts`, (len(starts), seq_len + 1): the first seq_len
    columns are the inputs, the last seq_len the targets."""
    return tokens[starts[:, None] + np.arange(seq_len + 1)[None, :]]


class TrainLoader:
    """Batches (device_batch, seq_len) of inputs and targets for one rank of a training run."""

    def __init__(self, data: TokenData, seq_len: int, device_batch: int, rank: int = 0, world: int = 1,
                 seed: int = 0, device: Union[str, torch.device] = "cpu"):
        self.data, self.seq_len, self.device_batch = data, seq_len, device_batch
        self.rank, self.world, self.seed, self.device = rank, world, seed, torch.device(device)
        self.epoch = 0  # completed passes over the data
        self.shard_index = -1
        self.tokens: Optional[np.memmap] = None
        self.order = np.empty(0, dtype=np.int64)
        self.cursor = 0  # windows of the current shard already taken by all ranks together
        # Pinning memory takes milliseconds, so one buffer is pinned once and every batch passes through it. It
        # is filled through its NumPy view: a single-threaded copy, where PyTorch would wake a thread pool.
        self.staging = torch.empty(device_batch, seq_len + 1, dtype=torch.int64, pin_memory=self.device.type == "cuda")
        self.staging_view = self.staging.numpy()
        self._next_shard()

    def _next_shard(self) -> None:
        """Open the next shard that holds at least one batch for every rank, and shuffle its windows."""
        for _ in range(len(self.data.train_files) + 1):
            self.shard_index += 1
            if self.shard_index == len(self.data.train_files):
                self.shard_index, self.epoch = 0, self.epoch + 1
            self.tokens = self.data.shard(self.shard_index)
            n_windows = (len(self.tokens) - 1) // self.seq_len
            if n_windows >= self.world * self.device_batch:
                # The same permutation on every rank: it depends on the seed, the pass and the shard only.
                rng = np.random.default_rng([self.seed, self.epoch, self.shard_index])
                self.order = rng.permutation(n_windows)
                self.cursor = 0
                return
        raise ValueError(f"{self.data.dir}: no shard holds {self.world * self.device_batch} windows of "
                         f"{self.seq_len} tokens; use a smaller batch")

    def next_batch(self) -> tuple[Tensor, Tensor]:
        """The next inputs and targets of this rank. All ranks advance together: each call consumes
        world * device_batch windows, of which this rank reads its own slice."""
        if self.cursor + self.world * self.device_batch > len(self.order):
            self._next_shard()  # the few windows left over at the end of a shard are dropped
        first = self.cursor + self.rank * self.device_batch
        starts = self.order[first:first + self.device_batch] * self.seq_len
        self.cursor += self.world * self.device_batch
        self.staging_view[:] = _windows(self.tokens, starts, self.seq_len)
        # A blocking copy: when it returns the card has the batch and the staging buffer may be refilled.
        block = self.staging.to(self.device)
        return block[:, :-1].contiguous(), block[:, 1:].contiguous()


class ValLoader:
    """The fixed held-out batches of one rank: the first windows of val.bin, in order.

    The set is the first floor(val_tokens / seq_len) windows; rank r of n evaluates windows r, r + n, ... and
    the ranks' sums are added, so the set does not depend on the batch size or the number of cards. Batches
    have a constant shape (the compiled model is not retraced): the last one is padded with repeats whose
    targets are -1, which the loss ignores. The tokens live on the card: evaluation loads nothing.
    """

    def __init__(self, data: TokenData, seq_len: int, device_batch: int, rank: int = 0, world: int = 1,
                 val_tokens: float = 1048576.0, device: Union[str, torch.device] = "cpu"):
        tokens = data.val_tokens()
        self.n_windows = int(min(val_tokens, len(tokens) - 1)) // seq_len
        if self.n_windows < 1:
            raise ValueError(f"{data.val_file} is shorter than one window of {seq_len} tokens")
        mine = np.arange(rank, self.n_windows, world)
        self.batches: list[tuple[Tensor, Tensor, int]] = []
        for i in range(0, len(mine), device_batch):
            windows = mine[i:i + device_batch]
            padded = np.concatenate([windows, np.repeat(windows[:1], device_batch - len(windows))])
            block = torch.from_numpy(_windows(tokens, padded * seq_len, seq_len).astype(np.int64)).to(device)
            x, y = block[:, :-1].contiguous(), block[:, 1:].contiguous()
            y[len(windows):] = -1
            self.batches.append((x, y, len(windows) * seq_len))

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor, int]]:
        """Inputs, targets and the number of targets that count, batch by batch."""
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)


# ----------------------------------------------------------------------------------------------------------------
# A byte-level dataset for tests


class ByteTokenizer:
    """The simplest tokenizer: one token per byte of UTF-8, ids 0 to 255, and id 256 for BOS. It has the
    interface of the production tokenizer, so tests and the sampling CLI can use either."""

    vocab_size = 257
    bos_id = 256
    special = {"<|bos|>": 256}

    def encode(self, text: str, bos: bool = True) -> list[int]:
        return ([self.bos_id] if bos else []) + list(text.encode("utf-8"))

    def decode(self, ids: Sequence[int]) -> str:
        return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")

    def token_bytes(self, i: int) -> bytes:
        return bytes([i]) if i < 256 else b""


def write_byte_dataset(text_file: Union[str, Path], out_dir: Union[str, Path], val_frac: float = 0.05,
                       shard_tokens: int = 400_000, seed: int = 0) -> Path:
    """Write a text file as a byte-level dataset in the on-disk format above; return its directory.

    The text is split into documents at blank lines; the documents are shuffled by the seed, a fraction is
    held out, and each is written as BOS followed by its bytes. `out_dir` should be <root>/data/<dataset>/bytes
    so that the trainer finds it as --data <dataset>/bytes with GPTPP_WORK=<root>. This is for tests: the
    real datasets are prepared by scripts/prepare_*.py.
    """
    text = Path(text_file).read_text(encoding="utf-8")
    documents = [d for d in text.split("\n\n") if d.strip()]
    rng = np.random.default_rng(seed)
    documents = [documents[i] for i in rng.permutation(len(documents))]
    n_val = max(1, round(val_frac * len(documents)))
    val_docs, train_docs = documents[:n_val], documents[n_val:]

    tokenizer = ByteTokenizer()

    def encode(docs: list[str]) -> np.ndarray:
        return np.array([i for d in docs for i in tokenizer.encode(d, bos=True)], dtype=TOKEN_DTYPE)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train, val = encode(train_docs), encode(val_docs)
    shards = []
    for i, start in enumerate(range(0, len(train), shard_tokens)):
        shards.append(f"train_{i:04d}.bin")
        train[start:start + shard_tokens].tofile(out / shards[-1])
    val.tofile(out / "val.bin")
    with open(out.parent / "val_docs.jsonl", "w", encoding="utf-8") as f:
        for d in val_docs:
            f.write(json.dumps({"text": d}) + "\n")
    val_bytes = sum(len(d.encode("utf-8")) for d in val_docs)
    meta = {
        "dataset": out.parent.name, "tokenizer": "bytes", "vocab_size": tokenizer.vocab_size,
        "bos_id": tokenizer.bos_id, "n_train_tokens": int(len(train)), "n_val_tokens": int(len(val)),
        "train_shards": shards, "val_bytes": val_bytes, "val_docs": len(val_docs),
        "bytes_per_token_val": val_bytes / len(val), "source": str(text_file), "license": "public domain",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return out
