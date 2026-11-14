"""Byte-pair encoding from scratch, in plain Python.

A tokenizer is a reversible map between text and a sequence of integers. Byte-pair encoding (BPE)
builds one by compression. The algorithm is from Gage, "A new algorithm for data compression" (1994);
Sennrich, Haddow and Birch, "Neural machine translation of rare words with subword units" (2016)
brought it to language models; GPT-2 (2019) ran it on bytes instead of characters, so that every
string can be encoded and there is no unknown token.

    1. Write the corpus as UTF-8 bytes. The first 256 tokens are the 256 values of a byte.
    2. Count every pair of adjacent tokens. Give the most frequent pair a new id and replace the pair
       by that id everywhere.
    3. Repeat until the vocabulary has the size asked for.

The list of merges, in the order they were learned, is the whole tokenizer. To encode a text, apply
the same merges in the same order; to decode, concatenate the bytes of the tokens.

Two refinements, both standard since GPT-2:

- Pre-tokenization. A regular expression first cuts the text into pieces (a word with its leading
  space, a run of digits, punctuation, whitespace) and merges never cross a piece boundary. Training
  then needs only the table of distinct pieces with their counts, and no token glues the end of one
  word to the start of the next. The pattern here is the one of GPT-4 (`cl100k_base` in tiktoken).
- Special tokens. A few ids stand for structure rather than text (the beginning of a document, the
  turns of a conversation). They are never produced by encoding text, only added by the caller.

Token ids: 0..255 are the bytes, 256.. the merges in the order learned, and the special tokens take
the last ids of the vocabulary.

Ties between equally frequent pairs are broken towards the pair with the smaller ids. With that rule
the functions below learn exactly the merges of the production tokenizer (`gptpp/tokenizer.py`, built
on the Hugging Face `tokenizers` library) and encode exactly as it does; `tests/test_tokenizer.py`
checks both.
"""

from __future__ import annotations

import heapq
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence

import regex  # the standard `re` module has no \p{L} (any letter) and \p{N} (any digit) classes

# The split pattern of GPT-4, one alternative a line. It differs from tiktoken's `cl100k_base` in two
# ways. The possessive quantifiers (`?+`, `++`, `*+`), which only make the match faster, are written
# as greedy ones: the cut is the same. The alternative `\s++$`, a guard against slow matching of
# whitespace at the very end of a text, is left out: the cut changes only inside the last run of
# whitespace of a text, and only if that run holds a line break followed by more whitespace. Without
# these two constructs, which Oniguruma (the engine inside `tokenizers`) reads differently from
# Python's `regex`, the two engines cut every text alike.
GPT4_PATTERN = "|".join(
    [
        r"'(?i:[sdmt]|ll|ve|re)",  # contractions: 's 'd 'm 't 'll 've 're
        r"[^\r\n\p{L}\p{N}]?\p{L}+",  # a word, with the space (or other single character) before it
        r"\p{N}{1,3}",  # digits, at most three at a time
        r" ?[^\s\p{L}\p{N}]+[\r\n]*",  # punctuation and symbols, with the line breaks after them
        r"\s*[\r\n]",  # whitespace up to and including a line break
        r"\s+(?!\S)",  # whitespace, leaving the last space to the word that follows
        r"\s",  # a single whitespace character that nothing above took
    ]
)

# The special tokens of gpt-pp. `<|bos|>` begins every document; the other four delimit the turns of
# a conversation and are reserved for a later chat stage.
SPECIAL_TOKENS = ["<|bos|>", "<|user_start|>", "<|user_end|>", "<|assistant_start|>", "<|assistant_end|>"]

Pair = tuple[int, int]
Word = tuple[int, ...]


# ----------------------------------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------------------------------


def count_words(texts: str | Iterable[str], pattern: str = GPT4_PATTERN) -> dict[Word, int]:
    """Cut the texts into pieces and count them: {the UTF-8 bytes of a piece: its occurrences}."""
    if isinstance(texts, str):
        texts = [texts]
    split = regex.compile(pattern)
    pieces: Counter[str] = Counter()
    for text in texts:
        pieces.update(split.findall(text))
    return {tuple(piece.encode("utf-8")): n for piece, n in pieces.items()}


def count_pairs(words: dict[Word, int]) -> Counter[Pair]:
    """How often each pair of adjacent tokens occurs in the corpus."""
    counts: Counter[Pair] = Counter()
    for word, n in words.items():
        for pair in zip(word, word[1:]):
            counts[pair] += n
    return counts


def merge(ids: Sequence[int], pair: Pair, new_id: int) -> list[int]:
    """Replace each occurrence of `pair` in `ids` by `new_id`, scanning from the left."""
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def train(
    texts: str | Iterable[str],
    vocab_size: int,
    pattern: str = GPT4_PATTERN,
    special_tokens: Sequence[str] = SPECIAL_TOKENS,
) -> list[Pair]:
    """Learn the merges of a vocabulary of `vocab_size` tokens (bytes, merges and special tokens).

    Returns the merges in the order learned; merge number k creates the token 256 + k. This is the
    algorithm as one would say it aloud: every round recounts all the pairs. It is quadratic and
    meant for small corpora; `train_fast` learns the same merges and keeps the counts up to date
    instead.
    """
    words = count_words(texts, pattern)
    merges: list[Pair] = []
    for new_id in range(256, vocab_size - len(special_tokens)):
        counts = count_pairs(words)
        if not counts:
            break  # every word is a single token: there is nothing left to merge
        pair = min(counts, key=lambda p: (-counts[p], p))  # the most frequent; ties: the smallest ids
        merges.append(pair)
        words = {tuple(merge(word, pair, new_id)): n for word, n in words.items()}
    return merges


def train_fast(
    texts: str | Iterable[str],
    vocab_size: int,
    pattern: str = GPT4_PATTERN,
    special_tokens: Sequence[str] = SPECIAL_TOKENS,
) -> list[Pair]:
    """The same merges as `train`, without recounting everything at every round.

    A merge changes only the words that contain the merged pair, so it is enough to remember, for
    each pair, which words contain it, and to correct the counts of those words. The best pair comes
    from a heap. An entry of the heap can be out of date, but only in one direction: after a pair
    has entered the heap its count can only fall, because a merge creates no new neighbours among
    the old tokens. An entry whose count is stale is therefore pushed back with its true count, and
    the first entry found up to date is the true maximum.
    """
    table = count_words(texts, pattern)
    words = [list(word) for word in table]  # each distinct word as a list of token ids
    freqs = list(table.values())  # how often each word occurs
    counts: Counter[Pair] = Counter()  # pair -> occurrences in the corpus
    where: defaultdict[Pair, set[int]] = defaultdict(set)  # pair -> the words that contain it
    for i, word in enumerate(words):
        for pair in zip(word, word[1:]):
            counts[pair] += freqs[i]
            where[pair].add(i)
    heap = [(-n, pair) for pair, n in counts.items()]  # smallest first: highest count, then smallest ids
    heapq.heapify(heap)

    merges: list[Pair] = []
    n_merges = vocab_size - 256 - len(special_tokens)
    while len(merges) < n_merges and heap:
        negative_count, pair = heapq.heappop(heap)
        if -negative_count != counts[pair]:  # stale entry: push it back with its true count
            if counts[pair] > 0:
                heapq.heappush(heap, (-counts[pair], pair))
            continue
        new_id = 256 + len(merges)
        merges.append(pair)
        created: set[Pair] = set()  # the pairs that contain the new token
        for i in where.pop(pair):
            old, new = words[i], merge(words[i], pair, new_id)
            for p in zip(old, old[1:]):
                counts[p] -= freqs[i]
            for p in zip(new, new[1:]):
                counts[p] += freqs[i]
                where[p].add(i)
                if new_id in p:
                    created.add(p)
            words[i] = new
        for p in created:
            heapq.heappush(heap, (-counts[p], p))
    return merges


# ----------------------------------------------------------------------------------------------------
# Encoding and decoding
# ----------------------------------------------------------------------------------------------------


def vocabulary(merges: Sequence[Pair], special_tokens: Sequence[str] = SPECIAL_TOKENS) -> list[bytes]:
    """The bytes of every token, indexed by id: the 256 bytes, the merges, the special tokens."""
    vocab = [bytes([b]) for b in range(256)]
    for left, right in merges:
        vocab.append(vocab[left] + vocab[right])
    return vocab + [name.encode("utf-8") for name in special_tokens]


def encode_piece(piece: bytes, ranks: dict[Pair, int]) -> list[int]:
    """Encode one piece of text: apply the merges it contains, earliest learned first.

    `ranks` maps a pair to the id of its merge; a smaller id was learned earlier.
    """
    ids = list(piece)
    while len(ids) >= 2:
        pair = min(zip(ids, ids[1:]), key=lambda p: ranks.get(p, float("inf")))
        if pair not in ranks:
            break  # no pair of this piece was ever merged
        ids = merge(ids, pair, ranks[pair])
    return ids


def encode(text: str, merges: Sequence[Pair], pattern: str = GPT4_PATTERN) -> list[int]:
    """Text to token ids. Special tokens are never produced: text is always encoded as text."""
    ranks = {pair: 256 + k for k, pair in enumerate(merges)}
    ids: list[int] = []
    for piece in regex.findall(pattern, text):
        ids.extend(encode_piece(piece.encode("utf-8"), ranks))
    return ids


def decode(ids: Iterable[int], merges: Sequence[Pair], special_tokens: Sequence[str] = SPECIAL_TOKENS) -> str:
    """Token ids to text. Bytes that are not valid UTF-8 (a character cut in two) become U+FFFD."""
    vocab = vocabulary(merges, special_tokens)
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")


class BPE:
    """A trained tokenizer: the functions above with their tables built once, and the special tokens.

    It has the interface of the production `gptpp.tokenizer.Tokenizer`, so that the two can be set
    side by side.
    """

    def __init__(
        self,
        merges: Sequence[Pair],
        pattern: str = GPT4_PATTERN,
        special_tokens: Sequence[str] = SPECIAL_TOKENS,
    ) -> None:
        self.merges = [tuple(pair) for pair in merges]
        self.pattern = pattern
        self.vocab = vocabulary(self.merges, special_tokens)  # id -> bytes
        self.vocab_size = len(self.vocab)
        self.special = {name: self.vocab_size - len(special_tokens) + k for k, name in enumerate(special_tokens)}
        self.bos_id = self.special.get("<|bos|>")
        self._split = regex.compile(pattern)
        self._ranks = {pair: 256 + k for k, pair in enumerate(self.merges)}
        self._cache: dict[str, list[int]] = {}  # piece -> ids: most pieces of a text have been seen before

    @classmethod
    def train(
        cls,
        texts: str | Iterable[str],
        vocab_size: int,
        pattern: str = GPT4_PATTERN,
        special_tokens: Sequence[str] = SPECIAL_TOKENS,
    ) -> "BPE":
        return cls(train_fast(texts, vocab_size, pattern, special_tokens), pattern, special_tokens)

    def encode(self, text: str, bos: bool = False) -> list[int]:
        ids = [self.bos_id] if bos else []
        for piece in self._split.findall(text):
            if piece not in self._cache:
                self._cache[piece] = encode_piece(piece.encode("utf-8"), self._ranks)
            ids.extend(self._cache[piece])
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def token_bytes(self, i: int) -> bytes:
        return self.vocab[i]


# ----------------------------------------------------------------------------------------------------
# The merges as a file
# ----------------------------------------------------------------------------------------------------


def save_merges(merges: Sequence[Pair], path: str) -> None:
    """Write the merges as text: line k holds the two ids that merge into token 256 + k."""
    with open(path, "w", encoding="ascii") as f:
        f.write("# byte-pair merges: line k (from 0) is the pair of token ids merged into token 256 + k\n")
        for left, right in merges:
            f.write(f"{left} {right}\n")


def load_merges(path: str) -> list[Pair]:
    """Read a file written by `save_merges`."""
    merges: list[Pair] = []
    with open(path, encoding="ascii") as f:
        for line in f:
            if not line.startswith("#"):
                left, right = line.split()
                merges.append((int(left), int(right)))
    return merges
