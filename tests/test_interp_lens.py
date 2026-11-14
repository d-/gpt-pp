"""Tests of gptpp/interp/lens.py on a tiny random model: the lens at the last depth is the model, its quantities
are what their definitions say, and the statistics over windows are the table's averages.

    cd gpt-pp && $PY -m pytest -q tests/test_interp_lens.py
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from gptpp.config import ModelConfig
from gptpp.interp import lens
from gptpp.model import GPT


def tiny(**changes: object) -> GPT:
    base = dict(vocab_size=50, n_layer=3, n_head=4, d_model=32, block_size=24, init="gpt2")
    torch.manual_seed(0)
    model = GPT(ModelConfig().replace(**{**base, **changes})).eval()
    with torch.no_grad():                                  # larger weights: the depths differ, the distributions are not flat
        for p in list(model.blocks.parameters()) + [model.unembed.weight]:
            p.mul_(8.0)
    return model


@pytest.mark.parametrize("changes", [{}, dict(logit_softcap=0.0), dict(norm="layernorm", bias=True, tie_embeddings=True)])
def test_the_last_depth_is_the_model(changes):
    model = tiny(**changes)
    idx = torch.randint(0, 50, (2, 12))
    x = lens.streams(model, idx)
    assert x.shape == (4, 2, 12, 32)
    logp = lens.readout(model, x)
    assert torch.allclose(logp[-1], F.log_softmax(model(idx), dim=-1), atol=1e-5)
    assert torch.allclose(logp.exp().sum(-1), torch.ones(4, 2, 12), atol=1e-5)


def test_the_table_is_what_it_says():
    model = tiny()
    idx = torch.randint(0, 50, (10,))
    t = lens.lens_table(model, idx, top=3)
    logp = lens.readout(model, lens.streams(model, idx.view(1, -1))[:, 0]).detach().numpy()
    D, T = 4, 10
    assert t["top_ids"].shape == (D, T, 3) and t["kl_bits"].shape == (D, T)
    assert (t["final_id"] == logp[-1].argmax(-1)).all() and (t["top_ids"][-1, :, 0] == t["final_id"]).all()
    assert np.allclose(t["kl_bits"][-1], 0, atol=1e-6) and (t["kl_bits"][:-1] > 0).all()
    assert (t["rank_final"][-1] == 0).all()
    for d in range(D):
        for p in range(T):
            f = t["final_id"][p]
            assert np.isclose(t["p_final"][d, p], np.exp(logp[d, p, f]), atol=1e-6)
            assert t["rank_final"][d, p] == (logp[d, p] > logp[d, p, f]).sum()
            kl = (np.exp(logp[-1, p]) * (logp[-1, p] - logp[d, p])).sum() / np.log(2)
            assert np.isclose(t["kl_bits"][d, p], kl, atol=1e-5)
    for p in range(T):                                     # settling: the top token stays the final one from there on
        s, on = t["settles"][p], t["rank_final"][:, p] == 0
        assert on[s:].all() and (s == 0 or not on[s - 1])
        assert on[t["first_top"][p]] and not on[:t["first_top"][p]].any() and t["first_top"][p] <= s


def test_settling_depth_by_hand():
    on = torch.tensor([[1, 0, 1, 1], [0, 0, 0, 1], [1, 1, 1, 1], [1, 1, 0, 1]]).bool().T      # columns are positions
    assert lens.settling_depth(on).tolist() == [2, 3, 0, 3]


def test_statistics_are_the_tables_averages():
    model = tiny()
    windows = torch.randint(0, 50, (5, 9))
    s = lens.lens_statistics(model, windows, skip=1, batch=2)
    tables = [lens.lens_table(model, w, top=1) for w in windows]
    kl = np.stack([t["kl_bits"][:, 1:].mean(1) for t in tables], 1)                  # (D, windows)
    assert s["positions"] == 40 and s["windows"] == 5
    assert np.allclose(s["kl_bits"], kl.mean(1), atol=1e-5) and np.allclose(s["kl_se"], kl.std(1, ddof=1) / np.sqrt(5), atol=1e-5)
    assert np.allclose(s["top1"], np.stack([(t["rank_final"][:, 1:] == 0) for t in tables]).mean((0, 2)), atol=1e-6)
    settles = np.concatenate([t["settles"][1:] for t in tables])
    assert s["settle_median"] == np.median(settles) and s["settle_hist"].sum() == 40
    assert abs(s["bias_bits"][-1]) < 1e-9 and (s["bias_bits"][:-1] > 0).all() and s["top1"][-1] == 1.0


def test_the_tuned_lens_starts_as_the_logit_lens_and_fits():
    model = tiny()
    windows = torch.randint(0, 50, (24, 12))
    start = lens.fit_tuned_lens(model, windows, steps=0)
    assert start["a"].shape == (3, 32, 32) and start["b"].shape == (3, 32) and torch.equal(start["a"][1], torch.eye(32))
    x = lens.streams(model, windows[:2])
    for depth in range(3):                                 # at the identity the tuned lens is the logit lens
        tuned = lens.tuned_readout(model, x[depth], start["a"][depth], start["b"][depth])
        assert torch.allclose(tuned, lens.readout(model, x[depth]), atol=1e-4)
    before = lens.tuned_statistics(model, windows, start, batch=5)
    plain = lens.lens_statistics(model, windows, batch=5)
    assert np.allclose(before["kl_bits"], plain["kl_bits"], atol=1e-4) and np.allclose(before["top1"], plain["top1"], atol=1e-6)
    grads = [p.requires_grad for p in model.parameters()]
    fitted = lens.fit_tuned_lens(model, windows[:16], steps=40, batch=8, lr=5e-3)
    assert [p.requires_grad for p in model.parameters()] == grads, "the model is frozen while fitting and as it was after"
    after = lens.tuned_statistics(model, windows[16:], fitted, batch=4)           # on windows it was not fitted on
    held = lens.tuned_statistics(model, windows[16:], start, batch=4)
    assert (after["kl_bits"][:-1] < held["kl_bits"][:-1]).all() and abs(after["kl_bits"][-1]) < 1e-9 and after["top1"][-1] == 1.0
    assert fitted["loss"].shape == (40, 3) and (fitted["loss"][-5:].mean(0) < fitted["loss"][:5].mean(0)).all()
