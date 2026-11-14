"""The transcoders (gptpp/interp/transcoder.py): the stacked module against its one-layer readings, the code, what
is read of the model's MLPs and how their outputs are replaced, the windows of text, a short training that beats
the mean, the evaluation, and the statistics that name a latent.

    cd gpt-pp && $PY -m pytest -q tests/test_transcoder.py

Everything runs on the CPU, on a two-layer model with random weights and byte-level Shakespeare.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import tiny
from gptpp.data import TokenData
from gptpp.interp.transcoder import (StoryStream, TranscoderConfig, Transcoders, decode_sparse, describe, evaluate, fit_normalization,
                                     heldout_starts, load_transcoders, logits_with_mlps, loss_of, mlp_io, promoted, save_transcoders,
                                     story_starts, train, windows_at)
from gptpp.model import GPT

BOS = 256


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    """Small tensors on a machine of many cores: waking a thread pool costs more than the arithmetic."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)



@pytest.fixture(scope="module")
def model() -> GPT:
    torch.manual_seed(0)
    return GPT(tiny(vocab_size=257, init="gpt2")).eval()


@pytest.fixture(scope="module")
def data(byte_data: Path) -> TokenData:
    return TokenData(byte_data)


def random_transcoders(seed: int = 0) -> Transcoders:
    tcs = Transcoders(TranscoderConfig(d=32, n=64, k=4, layers=2), seed=seed).eval()
    torch.manual_seed(seed)
    with torch.no_grad():
        tcs.enc_bias.normal_(0, 0.3), tcs.bias.normal_(0, 0.3), tcs.in_mean.normal_(0, 0.3), tcs.out_mean.normal_(0, 0.3)
        tcs.out_scale.uniform_(0.5, 2.0)
    return tcs


@torch.no_grad()
def test_one_layer_readings_are_the_stacked_module():
    tcs, x = random_transcoders(), torch.randn(2, 7, 32)
    y, z = tcs(x)
    assert y.shape == (2, 7, 32) and z.shape == (2, 7, 64)
    for layer in range(2):
        E, offset = tcs.read(layer)
        W, b = tcs.write(layer)
        torch.testing.assert_close(tcs.layer_pre(layer, x[layer]), tcs.pre(x)[layer], atol=1e-5, rtol=0)
        torch.testing.assert_close(x[layer] @ E.T + offset, tcs.pre(x)[layer], atol=1e-5, rtol=0)
        torch.testing.assert_close(z[layer] @ W + b, y[layer], atol=1e-5, rtol=0)
        out, code = tcs.layer_forward(layer, x[layer])
        torch.testing.assert_close(out, y[layer], atol=1e-5, rtol=0)
        torch.testing.assert_close(code, z[layer])
    assert torch.allclose(tcs.dictionary.norm(dim=-1), torch.ones(2, 64)), "the rows of the dictionary start as unit vectors"


@torch.no_grad()
def test_the_code_keeps_the_k_largest_where_positive():
    tcs = random_transcoders()
    pre = torch.tensor([[[3.0, -1.0, 2.0, 0.5, 4.0, 1.0] + [-5.0] * 58, [-1.0] * 64]]).expand(2, 2, 64)
    z = tcs.code(pre)
    assert z[0, 0, :6].tolist() == [3.0, 0.0, 2.0, 0.0, 4.0, 1.0] and float(z[0, 0, 6:].abs().sum()) == 0
    assert float(z[0, 1].abs().sum()) == 0, "nothing positive: nothing active"
    x = torch.randn(2, 50, 32)
    z = tcs.encode(x)
    assert int((z > 0).sum(-1).max()) <= 4 and float(z.min()) >= 0
    top = tcs.pre(x).topk(4, dim=-1)
    torch.testing.assert_close(decode_sparse(tcs, torch.relu(top.values), top.indices) + tcs.bias[:, None, :], tcs.decode_normalized(z), atol=1e-5, rtol=0)


@torch.no_grad()
def test_what_is_read_of_the_mlps_is_what_the_model_computes(model: GPT):
    idx = torch.randint(0, 257, (3, 12))
    x, y = mlp_io(model, idx)
    logits, cache = model.run_with_cache(idx, ["resid_mid.*", "mlp_out.*"])
    for layer in range(2):
        torch.testing.assert_close(x[layer], model.blocks[layer].norm2(cache[f"resid_mid.{layer}"]), atol=1e-5, rtol=0)
        torch.testing.assert_close(y[layer], cache[f"mlp_out.{layer}"], atol=1e-5, rtol=0)
    torch.testing.assert_close(logits_with_mlps(model, idx, lambda layer, x_, y_: None), logits, atol=1e-5, rtol=0)
    torch.testing.assert_close(logits_with_mlps(model, idx, lambda layer, x_, y_: y_), logits, atol=1e-5, rtol=0)
    hooked = model.run_with_hooks(idx, {"mlp_out.0": lambda t: torch.zeros_like(t)})
    torch.testing.assert_close(logits_with_mlps(model, idx, lambda layer, x_, y_: torch.zeros_like(y_) if layer == 0 else None), hooked, atol=1e-5, rtol=0)
    assert not torch.allclose(hooked, logits)


def test_windows_begin_at_a_document_start(data: TokenData):
    shard = data.shard(0)
    starts = story_starts(shard, BOS, 16, np.random.default_rng(0), 40)
    assert len(starts) == 40 and np.all(shard[starts] == BOS)
    val = data.val_tokens()
    held = heldout_starts(val, BOS, 16, 25)
    assert len(held) == 25 and np.all(val[held] == BOS) and np.all(np.diff(held) >= 16), "in order, and no two overlap"
    assert held[0] == np.flatnonzero(np.asarray(val) == BOS)[0]
    idx = windows_at(val, held, 16, "cpu")
    assert idx.shape == (25, 16) and idx.dtype == torch.int64 and bool((idx[:, 0] == BOS).all())


def test_a_short_training_beats_the_mean_and_is_evaluated(model: GPT, data: TokenData, tmp_path: Path):
    torch.manual_seed(0)
    tcs = Transcoders(TranscoderConfig(d=32, n=128, k=8, layers=2), seed=0)
    stream = StoryStream(model, data, BOS, batch=256, buffer_windows=64, model_batch=32, seed=0, device="cpu")
    x, y, mark = stream.fill()
    assert x.shape == y.shape == (2, 64 * 16, 32) and mark.shape == (64 * 16,) and 0 < int(mark.sum()) < len(mark)
    fit_normalization(tcs, x, y, mark)
    normalized = tcs.normalize_out(y[:, ~mark])
    torch.testing.assert_close(normalized.pow(2).sum(-1).mean(1), torch.ones(2), atol=1e-4, rtol=0)
    assert float(normalized.mean(1).abs().max()) < 1e-4
    loss, fvu, fired = loss_of(tcs, x[:, :256], y[:, :256], None, 16)
    assert fvu.shape == (2,) and fired.shape == (2, 128) and fired.dtype == torch.bool and loss.requires_grad
    with_aux, _, _ = loss_of(tcs, x[:, :256], y[:, :256], [torch.arange(40), torch.arange(0)], 16)
    assert float(with_aux.detach()) > float(loss.detach()), "the dead latents' share of the error is added"
    log = train(tcs, stream, tokens=256 * 300, dead_after=20_000, log_every=100, say=lambda line: None)
    assert log[-1]["step"] == 299 and all(a > b for a, b in zip(log[0]["fvu"], log[-1]["fvu"]))
    torch.testing.assert_close(tcs.dictionary.norm(dim=-1), torch.ones(2, 128), atol=1e-5, rtol=0)
    tcs.eval()
    scores = evaluate(model, tcs, data.val_tokens(), BOS, n_windows=96, model_batch=32, device="cpu")
    assert all(0 < v < 0.9 for v in scores["fvu"]), "better than answering the mean"
    assert all(0 < v <= 8 for v in scores["l0"]) and all(0 <= v <= 1 for v in scores["dead"])
    H = scores["loss"]
    assert set(H) == {"clean", "replaced", "zero", "mean", "layer0", "layer1"} and all(np.isfinite(v) for v in H.values())
    both = windows_at(data.val_tokens(), heldout_starts(data.val_tokens(), BOS, 16, 96), 17, "cpu")
    with torch.no_grad():
        assert H["clean"] == pytest.approx(float(model(both[:, :-1], both[:, 1:])), rel=1e-5), "the clean loss is the model's own"
    assert 0 < scores["agreement"] <= 1 and np.isfinite(scores["loss_recovered"])
    assert len(scores["loss_increase_by_layer"]) == 2 and scores["tokens"] == 96 * 16
    # saved and loaded, they are the same function
    save_transcoders(tcs, tmp_path / "t.pt", {"run": "test"})
    again, info = load_transcoders(tmp_path / "t.pt")
    assert info == {"run": "test"} and again.cfg == tcs.cfg
    with torch.no_grad():
        torch.testing.assert_close(again(x[:, :64])[0], tcs(x[:, :64])[0])
    # what names a latent
    stats = describe(model, tcs, data.val_tokens(), BOS, n_windows=48, model_batch=16, device="cpu")
    assert stats["density"].shape == (2, 128) and stats["mass"].shape == (2, 128, 257) and len(stats["starts"]) == 48
    assert float(stats["density"].sum(1).max()) <= 8 + 1e-4, "at most k latents a token"
    live = int(stats["best"][0].argmax())
    window, at = int(stats["best_window"][0, live]), int(stats["best_pos"][0, live])
    with torch.no_grad():
        idx = windows_at(data.val_tokens(), stats["starts"][window:window + 1], 16, "cpu")
        z = tcs.encode(mlp_io(model, idx)[0].flatten(1, 2))
    assert float(z[0, at, live]) == pytest.approx(float(stats["best"][0, live]), rel=1e-4), "the passage where it fires most"
    assert float(stats["mass"][0, live, int(idx[0, at])]) > 0
    up, down = promoted(model, tcs, 0, torch.tensor([live, 3]), top=5)
    assert up.shape == down.shape == (2, 5) and not set(up[0].tolist()) & set(down[0].tolist())
