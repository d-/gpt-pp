"""Models made by counting, for the exporters of part 1 ("Predicting the next token"): interpolated Kneser-Ney
n-gram models over any stream of symbols, the held-out stories as text, and the token table that the page ships.

The n-gram model is the one `scripts/corpus_stats.py` scores for the corpus statistics (the same recursion, the
same discounts), written here so that an exporter can ask for the probability of every symbol of a stream and not
only for the mean:

    order n, top level     counts of n-grams in the sample
    levels below           for a k-gram, the number of different symbols that precede it in the sample (its
                           continuation count), not how often it occurs
    under the unigrams     the uniform distribution over the vocabulary
    discount of a level    D = n1 / (n1 + 2 n2), with n1 and n2 the numbers of its k-grams counted once and twice
    one step               P(w | h) = (max(c(hw) - D, 0) + D * types(h) * P_lower(w | h')) / c(h) where h was seen,
                           P_lower(w | h') where it was not

Nothing is tuned on held-out text.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .. import work_dir


# ----------------------------------------------------------------------------------------------------------------
# Interpolated Kneser-Ney over a stream of small integers


def ngram_keys(stream: np.ndarray, n: int, bits: int) -> np.ndarray:
    """Every n-gram of a stream as one integer: its symbols side by side, `bits` bits each."""
    count = len(stream) - n + 1
    keys = stream[:count].astype(np.uint64)
    for k in range(1, n):
        keys = (keys << np.uint64(bits)) | stream[k:k + count].astype(np.uint64)
    return keys


def table(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The distinct values of `keys`, sorted, and the number of times each occurs."""
    keys = np.sort(keys)
    first = np.flatnonzero(np.concatenate(([True], keys[1:] != keys[:-1])))
    return keys[first], np.diff(first, append=len(keys))


def lookup(keys: np.ndarray, values: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """values[i] where keys[i] == query, and 0 where the query is not among the (sorted) keys."""
    index = np.minimum(np.searchsorted(keys, queries), len(keys) - 1)
    return np.where(keys[index] == queries, values[index], 0)


class Level:
    """One level of the model: counts of k-grams and of their contexts of k - 1 symbols."""

    def __init__(self, grams: np.ndarray, counts: np.ndarray, bits: int) -> None:
        self.grams, self.counts, self.bits = grams, counts, bits
        ones, twos = int((counts == 1).sum()), int((counts == 2).sum())
        self.discount = ones / (ones + 2 * twos) if ones and twos else 0.5       # Ney, Essen and Kneser (1994)
        contexts = grams >> np.uint64(bits)                                       # sorted, because the k-grams are
        first = np.flatnonzero(np.concatenate(([True], contexts[1:] != contexts[:-1])))
        self.contexts = contexts[first]
        self.totals = np.add.reduceat(counts, first)                              # how often each context was continued
        self.types = np.diff(first, append=len(grams))                            # by how many different symbols

    def probability(self, grams: np.ndarray, lower: np.ndarray) -> np.ndarray:
        """P(last symbol | the k - 1 before it), given the estimate `lower` of the level below."""
        count = lookup(self.grams, self.counts, grams)
        contexts = grams >> np.uint64(self.bits)
        total = lookup(self.contexts, self.totals, contexts)
        types = lookup(self.contexts, self.types, contexts)
        seen = total > 0
        p = lower.copy()
        p[seen] = (np.maximum(count[seen] - self.discount, 0) + self.discount * types[seen] * lower[seen]) / total[seen]
        return p


class KneserNey:
    """Interpolated Kneser-Ney models of orders 1 .. `orders` counted on one stream of symbols 0 .. vocab - 1."""

    def __init__(self, sample: np.ndarray, vocab: int, orders: int) -> None:
        self.vocab, self.orders = int(vocab), int(orders)
        self.bits = max(1, int(np.ceil(np.log2(vocab))))
        if self.bits * orders > 64:
            raise ValueError("an n-gram of this order does not fit one 64-bit key")
        self.raw: dict[int, Level] = {}
        self.continuation: dict[int, Level] = {}
        for n in range(1, orders + 1):
            grams, counts = table(ngram_keys(sample, n, self.bits))
            self.raw[n] = Level(grams, counts, self.bits)
            if n > 1:               # the (n - 1)-grams, each counted by the number of different symbols seen before it
                suffixes = grams & np.uint64((1 << (self.bits * (n - 1))) - 1)
                self.continuation[n - 1] = Level(*table(suffixes), self.bits)

    def probabilities(self, stream: np.ndarray, n: int, start: int | None = None) -> np.ndarray:
        """P(stream[i] | the n - 1 symbols before it) for i = start .. len - 1 (start >= n - 1; default orders - 1,
        so that every order scores the same symbols)."""
        start = self.orders - 1 if start is None else start
        if start < n - 1:
            raise ValueError("a symbol needs its n - 1 predecessors in the stream")
        p = np.full(len(stream) - start, 1.0 / self.vocab)
        for k in range(1, n + 1):
            level = self.raw[k] if k == n else self.continuation[k]
            p = level.probability(ngram_keys(stream[start - k + 1:], k, self.bits), p)
        return p

    def distribution(self, context: np.ndarray, n: int) -> np.ndarray:
        """The whole next-symbol distribution after `context` (its last n - 1 symbols are used): (vocab,)."""
        ctx = np.asarray(context[len(context) - (n - 1):] if n > 1 else context[:0], dtype=np.uint64)
        streams = np.concatenate((np.tile(ctx, (self.vocab, 1)), np.arange(self.vocab, dtype=np.uint64)[:, None]), axis=1)
        p = np.full(self.vocab, 1.0 / self.vocab)
        for k in range(1, n + 1):
            keys = np.zeros(self.vocab, dtype=np.uint64)
            for column in range(n - k, n):
                keys = (keys << np.uint64(self.bits)) | streams[:, column]
            p = (self.raw[k] if k == n else self.continuation[k]).probability(keys, p)
        return p


# ----------------------------------------------------------------------------------------------------------------
# The held-out stories as text


def story_texts(limit: int | None = None) -> list[str]:
    """The held-out TinyStories stories in the order of `val.bin`, as text."""
    out = []
    with open(work_dir() / "data" / "tinystories" / "val_docs.jsonl") as f:
        for line in f:
            out.append(json.loads(line)["text"])
            if limit is not None and len(out) >= limit:
                break
    return out


def plain(story: str) -> bool:
    """A story that a table of characters and a table of words can both read back exactly: ASCII, no blank at
    either end, words parted by one space or one line break."""
    return (story.isascii() and story == story.strip() and "  " not in story and " \n" not in story
            and "\n " not in story and "\n\n" not in story and "\t" not in story and "\r" not in story and len(story) > 0)


def held_out_tokens(dataset: str = "tinystories/bpe4k") -> tuple[np.ndarray, dict]:
    """The held-out token stream of a dataset and its metadata."""
    d = work_dir() / "data" / dataset
    meta = json.loads((d / "meta.json").read_text())
    return np.fromfile(d / meta.get("val_file", "val.bin"), dtype="<u2"), meta


def training_tokens(dataset: str, n: int) -> np.ndarray:
    """The first n training tokens of a dataset (its first shards)."""
    d = work_dir() / "data" / dataset
    meta = json.loads((d / "meta.json").read_text())
    parts, have = [], 0
    for name in meta["train_shards"]:
        part = np.memmap(d / name, dtype="<u2", mode="r")[: n - have]
        parts.append(np.asarray(part))
        have += len(part)
        if have >= n:
            break
    return np.concatenate(parts)


# ----------------------------------------------------------------------------------------------------------------
# A network scored as the evaluation scores (gptpp/evaluate.py), for a model already in hand


def window_bits(model, documents: list[list[int]], bos: int, window: int, batch_tokens: int = 1 << 15,
                device: str = "cpu") -> np.ndarray:
    """Bits spent on each document by a model, under the evaluation's protocol: the document's tokens are cut into
    consecutive windows, each `<|bos|>` followed by at most window - 1 tokens, no overlap; every token is scored
    once, from the tokens before it in its window. Returns one float64 a document."""
    import torch
    import torch.nn.functional as F

    pieces, owner = [], []
    for d, tokens in enumerate(documents):
        for start in range(0, len(tokens), window - 1):
            pieces.append([bos] + list(tokens[start:start + window - 1]))
            owner.append(d)
    bits = np.zeros(len(documents), dtype=np.float64)
    order = sorted(range(len(pieces)), key=lambda i: -len(pieces[i]))
    done = 0
    with torch.no_grad():
        while done < len(order):
            width = len(pieces[order[done]])
            rows = order[done:done + max(1, batch_tokens // width)]
            idx = torch.full((len(rows), width), bos, dtype=torch.long)
            target = torch.full((len(rows), width - 1), -100, dtype=torch.long)
            for r, i in enumerate(rows):
                piece = torch.as_tensor(pieces[i], dtype=torch.long)
                idx[r, :len(piece)] = piece
                target[r, :len(piece) - 1] = piece[1:]
            logits = model(idx.to(device))[:, :-1].float()
            nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1).to(device), reduction="none")
            nll = nll.view(len(rows), width - 1).double().sum(dim=1).cpu().numpy() / np.log(2.0)
            for r, i in enumerate(rows):
                bits[owner[i]] += nll[r]
            done += len(rows)
    return bits


# ----------------------------------------------------------------------------------------------------------------
# The table of token pairs as the page holds it


LOG_RANGE = 24.0        # a kept probability is stored as -log2 of it, on 16 bits over [0, LOG_RANGE]


class PairTable:
    """The unigram and bigram models of tokens exactly as the page computes them from `<name>.bin`:

        unigram      p1[w]: interpolated Kneser-Ney of order 1 (counts, one discount, the uniform distribution under)
        bigram       q(w | v) = (kept(v, w) + back[v] * lower[w]) / (sum of kept(v, .) + back[v])
                     kept    the discounted relative frequency max(c(vw) - D, 0) / c(v) of the pairs seen at least
                             `min_count` times, as stored (16 bits, logarithmic)
                     lower   the Kneser-Ney distribution under the pairs (continuation counts)
                     back    what the kept pairs of v leave, 1 - sum of kept(v, .), and never less than the
                             Kneser-Ney weight D * types(v) / c(v) (the rounding of the stored values could
                             otherwise eat it); the division makes every row sum to one again

    With min_count = 1 and no rounding this is the interpolated Kneser-Ney bigram of the corpus statistics.
    """

    def __init__(self, p1: np.ndarray, lower: np.ndarray, back: np.ndarray, start: np.ndarray, follow: np.ndarray,
                 code: np.ndarray) -> None:
        self.p1, self.lower, self.back = p1.astype(np.float32), lower.astype(np.float32), back.astype(np.float32)
        self.start, self.follow, self.code = start.astype(np.uint32), follow.astype(np.uint16), code.astype(np.uint16)
        self.vocab = len(p1)
        self.mass = np.zeros(self.vocab)                        # the kept mass of each row, summed in the order stored
        np.add.at(self.mass, np.repeat(np.arange(self.vocab), np.diff(self.start.astype(np.int64))), self.decode(self.code))
        self.norm = self.mass + self.back.astype(np.float64)

    @staticmethod
    def decode(code: np.ndarray) -> np.ndarray:
        return np.exp2(-code.astype(np.float64) * (LOG_RANGE / 65535.0))

    @classmethod
    def count(cls, train: np.ndarray, vocab: int, min_count: int) -> tuple["PairTable", dict]:
        pair = np.bincount(train[:-1].astype(np.int64) * vocab + train[1:], minlength=vocab * vocab).reshape(vocab, vocab)
        uni = np.bincount(train, minlength=vocab)

        def discount(counts: np.ndarray) -> float:
            ones, twos = int((counts == 1).sum()), int((counts == 2).sum())
            return ones / (ones + 2 * twos) if ones and twos else 0.5

        d1, d2 = discount(uni), discount(pair)
        p1 = (np.maximum(uni - d1, 0) + d1 * int((uni > 0).sum()) / vocab) / uni.sum()
        cont = (pair > 0).sum(axis=0)                           # by how many different tokens each token is preceded
        dc = discount(cont)
        lower = (np.maximum(cont - dc, 0) + dc * int((cont > 0).sum()) / vocab) / cont.sum()
        total = pair.sum(axis=1)
        keep = pair >= min_count
        rows, cols = np.nonzero(keep)                           # row by row, the followers of a row in the order of their ids
        kept = np.maximum(pair[rows, cols] - d2, 0) / total[rows]
        code = np.clip(np.rint(-np.log2(kept) * (65535.0 / LOG_RANGE)), 0, 65535).astype(np.uint16)
        start = np.concatenate(([0], np.cumsum(keep.sum(axis=1)))).astype(np.uint32)
        mass = np.zeros(vocab)
        np.add.at(mass, rows, cls.decode(code))
        types = (pair > 0).sum(axis=1)
        back = np.where(total > 0, np.maximum(1.0 - mass, d2 * types / np.maximum(total, 1)), 1.0)
        table = cls(p1, lower, back, start, cols, code)
        facts = {"tokens": int(len(train)), "pairs": int((pair > 0).sum()), "pairsSeenOnce": int((pair == 1).sum()),
                 "pairsKept": int(keep.sum()), "minCount": int(min_count), "occurrencesKept": float(pair[keep].sum() / pair.sum()),
                 "discounts": {"unigram": d1, "pairs": d2, "lower": dc}}
        return table, facts

    def kept(self, v: int) -> tuple[np.ndarray, np.ndarray]:
        a, b = int(self.start[v]), int(self.start[v + 1])
        return self.follow[a:b], self.decode(self.code[a:b])

    def distribution(self, v: int) -> np.ndarray:
        """q(. | v) in float64, as the page computes it (float32 back-off and lower, float64 sums)."""
        q = np.float64(self.back[v]) * self.lower.astype(np.float64)
        ids, p = self.kept(v)
        q[ids] += p
        return q / self.norm[v]

    def probabilities(self, stream: np.ndarray, start: int = 1) -> np.ndarray:
        """q(stream[i] | stream[i - 1]) for i = start .. len - 1."""
        a, b = stream[start - 1:-1].astype(np.int64), stream[start:].astype(np.int64)
        keys = (np.repeat(np.arange(self.vocab, dtype=np.int64), np.diff(self.start.astype(np.int64))) << 16) | self.follow
        values = self.decode(self.code)
        query = (a << 16) | b
        index = np.minimum(np.searchsorted(keys, query), len(keys) - 1)
        kept = np.where(keys[index] == query, values[index], 0.0)
        return (kept + self.back.astype(np.float64)[a] * self.lower.astype(np.float64)[b]) / self.norm[a]

    @classmethod
    def load(cls, name: str = "tinystories-token-pairs", out: Path | str | None = None) -> "PairTable":
        from .common import DATA
        out = Path(out) if out else DATA
        meta = json.loads((out / f"{name}.json").read_text())
        raw = (out / f"{name}.bin").read_bytes()
        arrays = {}
        for key, e in meta["layout"].items():
            arrays[key] = np.frombuffer(raw, dtype=e["dtype"], count=e["count"], offset=e["offset"])
        return cls(arrays["p1"], arrays["lower"], arrays["back"], arrays["start"], arrays["follow"], arrays["code"])
