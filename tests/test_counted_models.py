"""The counted models behind the figures of part 1 (gptpp/export/counted_models.py), on small synthetic streams:
the Kneser-Ney recursion against a direct computation from dictionaries of counts, distributions that sum to one,
n-grams scored one a row, and the table of token pairs as the page holds it."""
from __future__ import annotations

import collections

import numpy as np

from gptpp.export.counted_models import KneserNey, PairTable, plain


def stream(n: int, vocab: int, seed: int) -> np.ndarray:
    """A stream with structure: each symbol depends on the one before it."""
    rng = np.random.default_rng(seed)
    out = np.zeros(n, dtype=np.uint16)
    for i in range(1, n):
        out[i] = (out[i - 1] * 3 + rng.integers(0, 4)) % vocab if rng.random() < 0.8 else rng.integers(0, vocab)
    return out


def by_hand(sample: list[int], vocab: int, gram: tuple[int, ...]) -> float:
    """Interpolated Kneser-Ney of the order of `gram`, straight from the definition."""
    n = len(gram)
    raw = [collections.Counter(tuple(sample[i:i + k]) for i in range(len(sample) - k + 1)) for k in range(1, n + 1)]

    def level(k: int, top: bool) -> collections.Counter:
        if top:
            return raw[k - 1]
        return collections.Counter(g[1:] for g in raw[k])          # by how many different symbols each k-gram is preceded

    p = 1.0 / vocab
    for k in range(1, n + 1):
        counts = level(k, k == n)
        ones, twos = sum(1 for c in counts.values() if c == 1), sum(1 for c in counts.values() if c == 2)
        d = ones / (ones + 2 * twos) if ones and twos else 0.5
        context = gram[n - k:n - 1]
        followers = {g[-1]: c for g, c in counts.items() if g[:-1] == context}
        total = sum(followers.values())
        if total:
            p = (max(followers.get(gram[-1], 0) - d, 0) + d * len(followers) * p) / total
    return p


def test_kneser_ney_is_the_definition():
    vocab, sample, held = 11, stream(3000, 11, 0), stream(400, 11, 1)
    model = KneserNey(sample, vocab, 3)
    for n in (1, 2, 3):
        p = model.probabilities(held, n)                           # positions 2 .. len - 1
        for i in (2, 3, 57, 200, 399):
            want = by_hand(list(map(int, sample)), vocab, tuple(int(t) for t in held[i - n + 1:i + 1]))
            assert abs(p[i - 2] - want) < 1e-12, (n, i)


def test_distributions_sum_to_one_and_agree_with_scoring():
    vocab, sample, held = 13, stream(5000, 13, 2), stream(300, 13, 3)
    model = KneserNey(sample, vocab, 4)
    for n in (1, 2, 3, 4):
        q = model.distribution(held[:50], n)
        assert abs(q.sum() - 1) < 1e-9 and (q > 0).all()
        p = model.probabilities(held, n, 50)
        assert abs(p[0] - q[held[50]]) < 1e-12
        positions = np.arange(3, len(held))
        grams = held[positions[:, None] - (n - 1) + np.arange(n)[None, :]]
        assert np.abs(model.of_grams(grams) - model.probabilities(held, n, 3)).max() == 0


def test_pair_table_unpruned_is_the_bigram_and_pruned_rows_still_sum_to_one():
    vocab, sample, held = 17, stream(20000, 17, 4), stream(2000, 17, 5)
    whole, facts = PairTable.count(sample, vocab, 1)
    reference = KneserNey(sample, vocab, 2).probabilities(held, 2, 1)
    assert np.abs(np.log2(whole.probabilities(held, 1)) - np.log2(reference)).max() < 2e-3      # the 16-bit code of the kept pairs
    assert facts["pairsKept"] == facts["pairs"]
    pruned, facts = PairTable.count(sample, vocab, 20)
    assert 0 < facts["pairsKept"] < facts["pairs"] and 0 < facts["occurrencesKept"] < 1
    for v in range(vocab):
        q = pruned.distribution(v)
        assert abs(q.sum() - 1) < 1e-6 and (q > 0).all()
        ids, kept = pruned.kept(v)
        assert (np.diff(ids.astype(int)) > 0).all() and (kept > 0).all()
    p = pruned.probabilities(held, 1)
    assert np.allclose(p, [pruned.distribution(int(a))[int(b)] for a, b in zip(held[:-1], held[1:])])
    assert -np.log2(p).mean() >= -np.log2(whole.probabilities(held, 1)).mean() - 0.05              # dropping pairs costs, or nearly nothing


def test_plain_stories():
    assert plain("One day. Two.\nThree") and not plain("a  b") and not plain(" a") and not plain("a \nb") and not plain("café") and not plain("")
