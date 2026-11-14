"""The head-scores exporter (fig-03-02): its scores on patterns whose answer is known.

A stand-in model returns the same attention pattern for every head: an even spread over the tokens a reader may
read, or all the weight on one relative position. The scores must then be the yardstick the exporter writes
(`uniform_scores`) or exactly one.
"""
from types import SimpleNamespace

import pytest
import torch

from gptpp.export import head_scores as hs


class Stub:
    """Just enough of GPT for the exporter: n_layer, and run_with_cache giving one pattern for every head."""

    def __init__(self, pattern_of, layers: int = 2, heads: int = 3):
        self.cfg, self.heads, self.pattern_of = SimpleNamespace(n_layer=layers, n_head=heads), heads, pattern_of

    def run_with_cache(self, idx, names=None):
        B, T = idx.shape
        A = self.pattern_of(idx).expand(B, self.heads, T, T)
        return None, {name: A for name in names}


def uniform(idx):
    T = idx.size(1)
    return torch.tril(torch.ones(T, T)) / torch.arange(1, T + 1)[:, None]


def offset(k):
    """All the weight of reader i on token i - k (on token 0 when there is none)."""
    def pattern(idx):
        T = idx.size(1)
        A = torch.zeros(T, T)
        i = torch.arange(T)
        A[i, (i - k).clamp(min=0)] = 1.0
        return A
    return pattern


def test_an_even_spread_scores_the_yardstick():
    tokens, length, repeats = 40, 7, 3
    want = hs.uniform_scores(tokens, length, repeats)
    text = torch.zeros(5, tokens, dtype=torch.long)
    got = hs.story_scores(Stub(uniform), text, batch=2)
    for key in ("previous", "first", "self", "entropy", "distance"):
        assert got[key].shape == (2, 3)
        assert torch.allclose(got[key], torch.full((2, 3), want[key], dtype=torch.float64), atol=1e-6), key
    pool = torch.arange(10, 60)
    seqs = hs.repeated_sequences(pool, bos=99, length=length, repeats=repeats, n=6, seed=0)
    assert seqs.shape == (6, 1 + length * repeats) and bool((seqs[:, 0] == 99).all())
    assert torch.equal(seqs[:, 1:1 + length], seqs[:, 1 + length:1 + 2 * length])
    rep = hs.repeat_scores(Stub(uniform), seqs, length, repeats, batch=4)
    for key in ("duplicate", "induction"):
        assert torch.allclose(rep[key], torch.full((2, 3), want[key], dtype=torch.float64), atol=1e-6), key


@pytest.mark.parametrize("k, key", [(1, "previous"), (0, "self")])
def test_a_head_with_one_habit_scores_one_on_it(k, key):
    got = hs.story_scores(Stub(offset(k)), torch.zeros(3, 24, dtype=torch.long), batch=3)
    assert torch.allclose(got[key], torch.ones(2, 3, dtype=torch.float64))
    assert torch.allclose(got["entropy"], torch.zeros(2, 3, dtype=torch.float64))
    assert torch.allclose(got["distance"], torch.full((2, 3), float(k), dtype=torch.float64))
    other = "self" if key == "previous" else "previous"
    assert torch.allclose(got[other], torch.zeros(2, 3, dtype=torch.float64))


def test_duplicate_and_induction_on_repeats():
    length, repeats = 6, 3
    seqs = hs.repeated_sequences(torch.arange(100, 200), bos=7, length=length, repeats=repeats, n=4, seed=1)
    dup = hs.repeat_scores(Stub(offset(length)), seqs, length, repeats, batch=4)        # reads the copy one repeat back
    assert torch.allclose(dup["duplicate"], torch.ones(2, 3, dtype=torch.float64))
    assert torch.allclose(dup["induction"], torch.zeros(2, 3, dtype=torch.float64))
    ind = hs.repeat_scores(Stub(offset(length - 1)), seqs, length, repeats, batch=4)    # reads what followed that copy
    assert torch.allclose(ind["induction"], torch.ones(2, 3, dtype=torch.float64))
    assert torch.allclose(ind["duplicate"], torch.zeros(2, 3, dtype=torch.float64))
