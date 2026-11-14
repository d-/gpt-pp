"""The exporter of fig-01-04 (`gptpp.export.loss_against_context`): the targets it draws, the loss it reads at the
last position of a window, and a measurement stopped and gone on with, on a made-up stream and a small random model
(no checkpoint, no card, no data)."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from gptpp.config import ModelConfig
from gptpp.export.loss_against_context import NotFinished, choose_targets, last_token_loss, measure
from gptpp.model import GPT

BOS = 63


def made_up_stream(seed: int = 0) -> np.ndarray:
    """Documents of 5 to 60 tokens, each begun by BOS."""
    rng = np.random.default_rng(seed)
    parts = []
    for _ in range(200):
        parts.append(np.concatenate(([BOS], rng.integers(0, BOS, rng.integers(4, 60)))))
    return np.concatenate(parts).astype(np.uint16)


def test_targets_have_a_full_context_of_their_own_document():
    stream = made_up_stream()
    for context in (4, 16, 32):
        targets, share = choose_targets(stream, BOS, context, 300, seed=1)
        assert len(targets) == len(set(targets.tolist())) and (np.diff(targets) > 0).all() and 0 < share < 1
        for p in targets:
            window = stream[p - context + 1:p + 1]                  # the target and the context - 1 tokens before it
            assert BOS not in window, "the window may begin with the document's start token, never cross into another document"
        again, _ = choose_targets(stream, BOS, context, 300, seed=1)
        assert np.array_equal(targets, again)
    deep = (np.arange(len(stream)) - np.maximum.accumulate(np.where(stream == BOS, np.arange(len(stream)), 0)) >= 16).mean()
    assert abs(choose_targets(stream, BOS, 16, 10, 0)[1] - deep) < 1e-12


def test_the_last_position_is_read_as_the_whole_pass_reads_it():
    torch.manual_seed(0)
    model = GPT(ModelConfig(vocab_size=64, n_layer=2, n_head=2, d_model=32, block_size=16)).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn_like(p))                      # away from the initialization, where some projections are zero
    stream = made_up_stream(3)
    targets, _ = choose_targets(stream, BOS, 16, 40, seed=0)
    for k in (1, 3, 16):
        windows = torch.from_numpy(stream[targets[:, None] - k + np.arange(k)[None, :]].astype(np.int64))
        goal = torch.from_numpy(stream[targets].astype(np.int64))
        with torch.no_grad():
            whole = F.cross_entropy(model(windows)[:, -1].float(), goal, reduction="none")
        assert torch.allclose(last_token_loss(model, windows, goal), whole, atol=1e-6)
    bits = measure(model, stream, targets, [1, 4, 16], "cpu", batch_tokens=64)
    assert bits.shape == (3, len(targets)) and np.isfinite(bits).all() and (bits > 0).all()
    one = measure(model, stream, targets[:5], [4], "cpu", batch_tokens=4)             # one window at a time gives the same numbers
    assert np.allclose(one[0], bits[1, :5], atol=1e-5)


def test_a_measurement_that_was_stopped_goes_on_where_it_stopped(tmp_path):
    torch.manual_seed(1)
    model = GPT(ModelConfig(vocab_size=64, n_layer=2, n_head=2, d_model=32, block_size=16)).eval()
    stream = made_up_stream(5)
    targets, _ = choose_targets(stream, BOS, 16, 40, seed=0)
    ks, keep = [1, 4, 16], tmp_path / "kept" / "measured.npz"
    whole = measure(model, stream, targets, ks, "cpu", batch_tokens=64)
    with pytest.raises(NotFinished):                                # no time at all: stopped before the first window
        measure(model, stream, targets, ks, "cpu", 64, keep, "these weights", seconds=-1.0)
    kept = np.load(keep)
    assert (kept["done"] == 0).all() and str(kept["stamp"]) == "these weights"
    assert np.allclose(measure(model, stream, targets, ks, "cpu", 64, keep, "these weights"), whole, atol=1e-6)
    # a file as a stop in the middle of the second context leaves it: the first context done, 16 targets of the second
    kept = {k: np.load(keep)[k].copy() for k in ("targets", "ks", "bits", "done", "stamp")}
    kept["done"][:] = [len(targets), 16, 0]; kept["bits"][1, 16:] = 0; kept["bits"][2] = 0; kept["bits"][0] += 1.0
    np.savez(keep, **kept)
    again = measure(model, stream, targets, ks, "cpu", 64, keep, "these weights")
    assert np.allclose(again[0], whole[0] + 1.0, atol=1e-6), "what is kept is not measured again"
    assert np.allclose(again[1:], whole[1:], atol=1e-6), "and the rest is measured from where it stopped"
    assert (np.load(keep)["done"] == len(targets)).all()
    with pytest.raises(NotFinished):                                # other weights, other targets or other contexts: nothing is kept
        measure(model, stream, targets, ks, "cpu", 64, keep, "other weights", seconds=-1.0)
    assert (np.load(keep)["done"] == 0).all()
    with pytest.raises(NotFinished):
        measure(model, stream, targets[:30], ks, "cpu", 64, keep, "other weights", seconds=-1.0)
    # a context added later is measured alone: the contexts kept before keep their rows
    measure(model, stream, targets, ks, "cpu", 64, keep, "these weights")
    kept = {k: np.load(keep)[k].copy() for k in ("targets", "ks", "bits", "done", "stamp")}
    kept["bits"][1] += 1.0
    np.savez(keep, **kept)
    more = measure(model, stream, targets, [1, 3, 4, 16], "cpu", 64, keep, "these weights")
    assert np.allclose(more[[0, 3]], whole[[0, 2]], atol=1e-6) and np.allclose(more[2], whole[1] + 1.0, atol=1e-6)
    assert np.allclose(more[1], measure(model, stream, targets, [3], "cpu", 64)[0], atol=1e-6)
    assert np.array_equal(np.load(keep)["ks"], [1, 3, 4, 16]) and (np.load(keep)["done"] == len(targets)).all()
