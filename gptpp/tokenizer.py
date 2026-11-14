"""The production tokenizer: the byte-pair encoding of `gptpp/bpe.py`, run by Hugging Face `tokenizers`.

`bpe.py` is the algorithm written to be read, and runs on one core. This module is the same
tokenizer at the speed the data pipeline needs (the Rust library trains and encodes on every core),
with the same token ids:

    ids 0..255                   the 256 byte values
    ids 256..256+M-1             the M merges, in the order learned
    the last ids                 the special tokens (`<|bos|>` first)

Training. The merges are learned by the library's `BpeTrainer`, which implements the textbook loop
(merge the most frequent pair, ties towards the smaller pair of ids). Two things are arranged so that
it learns exactly the merges of `bpe.train`:

- the text is cut into pieces here, by the same Python regular expression as in `bpe.py`, and
- each piece reaches the trainer as its UTF-8 bytes written as the characters U+0000..U+00FF, so the
  trainer's initial alphabet, sorted by code point, is the bytes in order and its ids are ours.

Encoding. The saved `tokenizer.json` is a standard byte-level BPE file (a regular expression split,
then the byte-level alphabet of GPT-2, then the merges) that any program using `tokenizers` or
`transformers` can load. The special tokens occupy ids in its vocabulary but are deliberately not
registered as "added tokens": the text `<|bos|>` inside a web page is encoded as seven ordinary
characters, never as the special token. Special ids enter a sequence only through `encode(..., bos=True)`
or by the caller writing them.

    tok = Tokenizer.load("bpe32k")            # by name under $GPTPP_WORK/data/tokenizers, or a path
    ids = tok.encode("text", bos=True)
    ids = tok.encode_batch(texts, bos=True)   # the fast path
    text = tok.decode(ids)
"""

from __future__ import annotations

import json
import multiprocessing
import os
import time
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

import regex
import tokenizers
from tokenizers import Regex, decoders, models, pre_tokenizers, trainers

from gptpp import bpe, work_dir
from gptpp.bpe import GPT4_PATTERN, SPECIAL_TOKENS, Pair


def tokenizer_dir(name_or_path: str | os.PathLike) -> Path:
    """A bare name means $GPTPP_WORK/data/tokenizers/<name>; anything with a separator is a path."""
    text = os.fspath(name_or_path)
    if os.sep in text or os.path.isdir(text):
        return Path(text)
    return work_dir() / "data" / "tokenizers" / text


def byte_alphabet() -> list[str]:
    """The character that stands for each byte in a byte-level `tokenizer.json` (the GPT-2 alphabet).

    A byte that is a printable, non-space Latin-1 character stands for itself; the 68 others (the
    control characters, the space, 0x7F..0xA0 and the soft hyphen 0xAD) are moved, in order, to
    U+0100 and up, so that every token of the vocabulary file is visible text.
    """
    visible = [b for b in range(256) if 0x21 <= b <= 0x7E or 0xA1 <= b <= 0xAC or 0xAE <= b <= 0xFF]
    moved = [b for b in range(256) if b not in visible]
    table = {b: chr(b) for b in visible} | {b: chr(0x100 + k) for k, b in enumerate(moved)}
    return [table[b] for b in range(256)]


# ----------------------------------------------------------------------------------------------------
# Training: pieces are cut in worker processes, the merges are learned by the Rust trainer
# ----------------------------------------------------------------------------------------------------

_split: regex.Pattern | None = None  # the compiled pattern of a worker process


def _init_worker(pattern: str) -> None:
    global _split
    _split = regex.compile(pattern)


def _cut(texts: list[str]) -> list[bytes]:
    """Each text as the UTF-8 bytes of its pieces, joined by 0xFF, a byte that UTF-8 never uses."""
    return [b"\xff".join(piece.encode("utf-8") for piece in _split.findall(text)) for text in texts]


def _batches(texts: Iterable[str], counter: dict[str, int], batch_chars: int = 1 << 20) -> Iterator[list[str]]:
    """Group the texts into batches of about a million characters, counting what goes by."""
    batch: list[str] = []
    size = 0
    for text in texts:
        batch.append(text)
        size += len(text)
        counter["docs"] += 1
        counter["chars"] += len(text)
        if size >= batch_chars:
            yield batch
            batch, size = [], 0
    if batch:
        yield batch


def train_merges(
    texts: str | Iterable[str],
    vocab_size: int,
    pattern: str = GPT4_PATTERN,
    special_tokens: Sequence[str] = SPECIAL_TOKENS,
    workers: int | None = None,
) -> tuple[list[Pair], dict[str, int]]:
    """Learn the merges of `bpe.train` with the library's trainer. Returns (merges, {"docs", "chars"})."""
    if isinstance(texts, str):
        texts = [texts]
    counter = {"docs": 0, "chars": 0}
    hf = tokenizers.Tokenizer(models.BPE())
    hf.pre_tokenizer = pre_tokenizers.CharDelimiterSplit("\xff")
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size - len(special_tokens),
        min_frequency=0,
        special_tokens=[],
        initial_alphabet=[chr(b) for b in range(256)],
        show_progress=False,
    )
    workers = workers or min(os.cpu_count() or 1, 64)
    context = multiprocessing.get_context("forkserver")  # safe even if this process already runs threads
    context.set_forkserver_preload(["gptpp.tokenizer"])
    with context.Pool(workers, _init_worker, (pattern,)) as pool:
        cut = (doc.decode("latin-1") for docs in pool.imap_unordered(_cut, _batches(texts, counter)) for doc in docs)
        hf.train_from_iterator(cut, trainer=trainer)

    # Read the merges back as pairs of ids, and check that the trainer's ids are the ones we promise.
    model = json.loads(hf.to_str())["model"]
    vocab: dict[str, int] = model["vocab"]
    assert all(vocab[chr(b)] == b for b in range(256)), "the trainer's alphabet is not the bytes in order"
    merges: list[Pair] = []
    for k, (left, right) in enumerate(model["merges"]):
        assert vocab[left + right] == 256 + k, "the trainer's merges are not numbered in the order learned"
        merges.append((vocab[left], vocab[right]))
    return merges, counter


# ----------------------------------------------------------------------------------------------------
# The tokenizer
# ----------------------------------------------------------------------------------------------------


def _build(merges: Sequence[Pair], pattern: str, special_tokens: Sequence[str]) -> tokenizers.Tokenizer:
    """A `tokenizers` object that encodes like `bpe.encode`, with our ids."""
    token_bytes = bpe.vocabulary(merges, special_tokens)
    assert len(set(token_bytes)) == len(token_bytes), "two tokens have the same bytes"
    alphabet = byte_alphabet()
    strings = ["".join(alphabet[b] for b in token) for token in token_bytes]
    model = models.BPE(
        vocab={string: i for i, string in enumerate(strings)},
        merges=[(strings[left], strings[right]) for left, right in merges],
    )
    hf = tokenizers.Tokenizer(model)
    hf.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(pattern), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    hf.decoder = decoders.ByteLevel()
    return hf


class Tokenizer:
    """Byte-level BPE with the gpt-pp special tokens (Contract 2 of the lab)."""

    def __init__(
        self,
        merges: Sequence[Pair],
        pattern: str = GPT4_PATTERN,
        special_tokens: Sequence[str] = SPECIAL_TOKENS,
        name: str = "",
        info: dict | None = None,
    ) -> None:
        self.name = name
        self.merges = [tuple(pair) for pair in merges]
        self.pattern = pattern
        self.info = dict(info or {})  # provenance: what the merges were trained on
        self._bytes = bpe.vocabulary(self.merges, special_tokens)
        self.vocab_size = len(self._bytes)
        self.special = {name: self.vocab_size - len(special_tokens) + k for k, name in enumerate(special_tokens)}
        self.bos_id = self.special.get("<|bos|>")
        self._hf = _build(self.merges, pattern, special_tokens)
        self._encode_batch = getattr(self._hf, "encode_batch_fast", self._hf.encode_batch)  # no offsets

    # -- making, saving and loading ---------------------------------------------------------------

    @classmethod
    def train(
        cls,
        texts: str | Iterable[str],
        vocab_size: int,
        name: str = "",
        pattern: str = GPT4_PATTERN,
        special_tokens: Sequence[str] = SPECIAL_TOKENS,
        trained_on: str = "",
        workers: int | None = None,
    ) -> "Tokenizer":
        """Learn a vocabulary of `vocab_size` tokens (bytes, merges and special tokens) from `texts`."""
        start = time.time()
        merges, counted = train_merges(texts, vocab_size, pattern, special_tokens, workers)
        info = {
            "trained_on": trained_on,
            "n_chars": counted["chars"],
            "n_docs": counted["docs"],
            "train_seconds": round(time.time() - start, 1),
        }
        return cls(merges, pattern, special_tokens, name, info)

    def save(self, name_or_path: str | os.PathLike | None = None) -> Path:
        """Write `tokenizer.json`, `merges.txt` and `meta.json`; returns the directory."""
        folder = tokenizer_dir(name_or_path or self.name)
        folder.mkdir(parents=True, exist_ok=True)
        self._hf.save(str(folder / "tokenizer.json"))
        bpe.save_merges(self.merges, str(folder / "merges.txt"))
        meta = {
            "name": self.name or folder.name,
            "vocab_size": self.vocab_size,
            "special_tokens": self.special,
            "bos_id": self.bos_id,
            "n_merges": len(self.merges),
            "pattern": self.pattern,
            "merges_file": "merges.txt",
            "ids": "0..255 the bytes; 256.. the merges in the order learned; the last ids the special tokens",
            "library": f"tokenizers {tokenizers.__version__}",
            **self.info,
        }
        (folder / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
        return folder

    @classmethod
    def load(cls, name_or_path: str | os.PathLike) -> "Tokenizer":
        """Load by name (under $GPTPP_WORK/data/tokenizers) or from a directory."""
        folder = tokenizer_dir(name_or_path)
        meta = json.loads((folder / "meta.json").read_text())
        merges = bpe.load_merges(str(folder / meta["merges_file"]))
        info = {key: meta[key] for key in ("trained_on", "n_chars", "n_docs", "train_seconds") if key in meta}
        tok = cls(merges, meta["pattern"], list(meta["special_tokens"]), meta["name"], info)
        assert tok.special == meta["special_tokens"] and tok.vocab_size == meta["vocab_size"]
        return tok

    # -- encoding and decoding --------------------------------------------------------------------

    def encode(self, text: str, bos: bool = False) -> list[int]:
        """Text to token ids, with `<|bos|>` in front if asked. The text is never read as special tokens."""
        ids = self._hf.encode(text, add_special_tokens=False).ids
        return [self.bos_id] + ids if bos else ids

    def encode_batch(self, texts: Sequence[str], bos: bool = False, threads: int | None = None) -> list[list[int]]:
        """Encode many texts at once.

        `threads=None` hands the whole batch to the library's thread pool (every core, unless the
        environment sets TOKENIZERS_PARALLELISM=false or RAYON_NUM_THREADS). `threads=1` encodes in
        the calling thread. `threads=k` never keeps more than k cores busy: the pool is given k
        texts at a time, texts of similar length together so that no core waits long for another.
        """
        texts = list(texts)
        if threads is None:
            batch = [encoding.ids for encoding in self._encode_batch(texts, add_special_tokens=False)]
        elif threads <= 1:
            batch = [self._hf.encode(text, add_special_tokens=False).ids for text in texts]
        else:
            batch = [[] for _ in texts]
            by_length = sorted(range(len(texts)), key=lambda i: len(texts[i]))
            for start in range(0, len(texts), threads):
                group = by_length[start : start + threads]
                encodings = self._encode_batch([texts[i] for i in group], add_special_tokens=False)
                for i, encoding in zip(group, encodings):
                    batch[i] = encoding.ids
        return [[self.bos_id] + ids for ids in batch] if bos else batch

    def decode(self, ids: Iterable[int]) -> str:
        """Token ids to text. Bytes that are not valid UTF-8 (a character cut in two) become U+FFFD."""
        return b"".join(self._bytes[i] for i in ids).decode("utf-8", errors="replace")

    def token_bytes(self, i: int) -> bytes:
        """The bytes of token i (for a special token, its name)."""
        return self._bytes[i]
