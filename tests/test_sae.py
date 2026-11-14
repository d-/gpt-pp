"""Tests of gptpp/interp/sae.py: the sparse autoencoder, its training and its evaluation, on small tensors on the CPU.

    cd gpt-pp && $PY -m pytest -q tests/test_sae.py

What is asserted: the two halves of the model around a layer compose to the model; a TopK code has k active latents
and decodes to the sum of k rows; the normalization and the geometric median; training recovers a planted
dictionary and keeps the rows of unit length; AuxK brings dead latents back and nothing else does; an L1 penalty
shrinks reconstructions and TopK does not; the evaluation scores a perfect dictionary and an empty one as such;
the stream of activations leaves out what it says; and the command line runs end to end on a small run.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gptpp.config import ModelConfig  # noqa: E402
from gptpp.interp import sae as S  # noqa: E402
from gptpp.model import GPT  # noqa: E402

BOS = 63


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    """Small tensors on a machine of many cores: waking a thread pool costs more than the arithmetic."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def small_model(seed: int = 0) -> GPT:
    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=64, n_layer=3, n_head=2, d_model=32, block_size=32, init="gpt2")
    return GPT(cfg).eval()


def token_stream(n: int, seed: int = 0) -> np.ndarray:
    """Random tokens with a document-start mark about every twenty tokens."""
    rng = np.random.default_rng(seed)
    tokens = rng.integers(0, BOS, size=n).astype("<u2")
    tokens[rng.random(n) < 0.05] = BOS
    return tokens


def planted(n_samples: int, d: int = 24, n_features: int = 48, active: int = 3, seed: int = 0):
    """Activations that are sums of `active` rows of a known dictionary with coefficients in [0.5, 1.5]."""
    g = torch.Generator().manual_seed(seed)
    rows = F.normalize(torch.randn(n_features, d, generator=g), dim=-1)
    which = torch.stack([torch.randperm(n_features, generator=g)[:active] for _ in range(n_samples)])
    coefficient = 0.5 + torch.rand(n_samples, active, generator=g)
    return (rows[which] * coefficient[..., None]).sum(1), rows


def fit(sae: S.SparseAutoencoder, x: torch.Tensor, steps: int, batch: int = 512, dead_after: int = 20_000) -> S.Trainer:
    S.fit_normalization([sae], x)
    trainer = S.Trainer(sae, steps, batch, dead_after)
    g = torch.Generator().manual_seed(1)
    for _ in range(steps):
        trainer.step(sae.normalize(x[torch.randint(len(x), (batch,), generator=g)]))
    return trainer


def test_the_two_halves_around_a_layer_compose_to_the_model():
    model = small_model()
    idx = torch.randint(0, 64, (3, 20))
    with torch.no_grad():
        logits, cache = model.run_with_cache(idx, names=["resid_post.*"])
        for layer in range(3):
            x = S.residual_after(model, idx, layer)
            assert torch.allclose(x, cache[f"resid_post.{layer}"], atol=1e-5)
            assert torch.allclose(S.logits_from(model, x, layer), logits, atol=1e-4)
        assert torch.allclose(S.logits_from(model, x, 2), model(idx), atol=1e-4)


def test_a_topk_code_has_k_active_latents_and_decodes_to_the_sum_of_k_rows():
    torch.manual_seed(0)
    sae = S.SparseAutoencoder(S.DictConfig(d=16, n=64, kind="topk", k=5))
    x = torch.randn(40, 16)
    z = sae.encode(x)
    assert z.shape == (40, 64) and (z >= 0).all() and ((z > 0).sum(-1) <= 5).all()
    top = F.relu(sae.pre(x)).topk(5, dim=-1)
    assert torch.allclose(S.decode_sparse(sae, top.values, top.indices), sae.decode(z), atol=1e-5)
    by_hand = sum(z[0, i] * sae.dictionary[i] for i in range(64)) + sae.bias
    assert torch.allclose(sae.decode(z)[0], by_hand, atol=1e-5)
    assert torch.allclose(sae.dictionary.norm(dim=-1), torch.ones(64), atol=1e-5), "rows start as unit vectors"
    assert torch.equal(sae.encoder, sae.dictionary), "the encoder starts as the dictionary"
    assert S.DictConfig(16, 64, "topk", 5).name == "topk-n64-k5" and S.DictConfig(16, 64, "relu", 0, 0.02).name == "relu-n64-l0.02"


def test_normalization_centers_scales_to_sqrt_d_and_restores():
    torch.manual_seed(0)
    x = 40 * torch.randn(5000, 16) + 7
    sae = S.SparseAutoencoder(S.DictConfig(d=16, n=32))
    S.fit_normalization([sae], x)
    xn = sae.normalize(x)
    assert xn.mean(0).abs().max() < 1e-4
    assert abs(float(xn.norm(dim=-1).mean()) - 4.0) < 1e-3, "the mean norm is sqrt(d)"
    assert abs(float(sae.variance) - float(xn.pow(2).sum(-1).mean())) < 1e-3
    assert torch.allclose(sae.restore(xn), x, atol=1e-3)
    assert torch.allclose(sae.directions(), sae.dictionary / sae.scale)


def test_the_geometric_median_ignores_a_far_outlier_and_the_mean_does_not():
    points = torch.cat([torch.randn(200, 2, generator=torch.Generator().manual_seed(0)) * 0.1, torch.tensor([[1000.0, 0.0]])])
    assert points.mean(0)[0] > 4
    assert S.geometric_median(points).norm() < 0.05
    square = torch.tensor([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])
    assert S.geometric_median(square).abs().max() < 1e-6


def test_training_recovers_a_planted_dictionary_with_unit_rows():
    x, rows = planted(20_000)
    torch.manual_seed(0)
    sae = S.SparseAutoencoder(S.DictConfig(d=24, n=48, kind="topk", k=3))
    trainer = fit(sae, x, steps=3000)
    assert torch.allclose(sae.dictionary.norm(dim=-1), torch.ones(48), atol=1e-4), "rows keep unit length"
    with torch.no_grad():
        xn = sae.normalize(x[:4000])
        fvu = float((xn - sae(xn)[0]).pow(2).sum() / xn.pow(2).sum())
        best = (sae.dictionary @ rows.T).max(0).values                         # for each planted row, its best learned row
    assert fvu < 0.08, fvu
    assert float(best.min()) > 0.98, "every planted direction is a row of the learned dictionary"
    assert trainer.lr == pytest.approx(2e-4 / math.sqrt(48 / 2 ** 14))
    assert trainer.lr_factor() == 0.0 and trainer.step_no == 3000, "the rate has decayed to zero at the last step"


def test_the_gradient_of_a_row_loses_its_component_along_the_row():
    x, _ = planted(2000)
    torch.manual_seed(0)
    sae = S.SparseAutoencoder(S.DictConfig(d=24, n=48, kind="topk", k=3))
    S.fit_normalization([sae], x)
    trainer = S.Trainer(sae, 10, 256, 10_000)
    seen = {}
    step = trainer.opt.step
    def spy():
        g = sae.dictionary.grad
        seen["along"] = float((g * sae.dictionary.detach()).sum(-1).abs().max()); seen["size"] = float(g.norm())
        step()
    trainer.opt.step = spy
    trainer.step(sae.normalize(x[:256]))
    assert seen["size"] > 0 and seen["along"] < 1e-5 * seen["size"] + 1e-7


def test_auxk_revives_dead_latents_and_without_it_they_stay_dead(monkeypatch):
    x, _ = planted(20_000)
    def run(aux_weight: float) -> float:
        monkeypatch.setattr(S, "AUX_WEIGHT", aux_weight)
        torch.manual_seed(0)
        sae = S.SparseAutoencoder(S.DictConfig(d=24, n=48, kind="topk", k=3))
        with torch.no_grad():
            sae.enc_bias[:24] = -6.0                     # half of the latents can never be among the top k
        fit(sae, x, steps=1500, dead_after=2_000)
        with torch.no_grad():
            z = sae.encode(sae.normalize(x[:4000]))
        return float(((z[:, :24] > 0).any(0)).float().mean())
    assert run(0.0) == 0.0, "no gradient reaches a latent that never fires"
    assert run(1 / 32) > 0.5


def test_an_l1_penalty_shrinks_reconstructions_and_topk_does_not():
    x, _ = planted(20_000)
    torch.manual_seed(0)
    relu = S.SparseAutoencoder(S.DictConfig(d=24, n=48, kind="relu", l1=0.05))
    topk = S.SparseAutoencoder(S.DictConfig(d=24, n=48, kind="topk", k=3))
    fit(relu, x, steps=2000); relu.fold_norms()
    fit(topk, x, steps=2000)
    assert torch.allclose(relu.dictionary.norm(dim=-1), torch.ones(48), atol=1e-4), "fold_norms leaves unit rows"
    @torch.no_grad()
    def gain(sae):
        xn = sae.normalize(x[:4000]); xhat = sae(xn)[0]
        return float(((xn - sae.bias) * (xhat - sae.bias)).sum() / (xhat - sae.bias).pow(2).sum())
    assert gain(relu) > 1.03, gain(relu)
    assert abs(gain(topk) - 1) < 0.02, gain(topk)


def test_clamping_a_latent_moves_the_stream_along_its_direction_only():
    torch.manual_seed(0)
    sae = S.SparseAutoencoder(S.DictConfig(d=16, n=64, kind="topk", k=5))
    S.fit_normalization([sae], 30 * torch.randn(2000, 16) + 3)
    x = 30 * torch.randn(4, 7, 16) + 3
    latent = int(sae.encode(sae.normalize(x.reshape(-1, 16))).sum(0).argmax())
    y, f = S.clamp(sae, x, latent, 9.0)
    assert y.shape == x.shape and f.shape == (4, 7) and (f > 0).any() and (f == 0).any(), "the latent fires on some tokens and not on others"
    u = sae.directions()[latent]
    step = (y - x).reshape(-1, 16)
    assert torch.allclose(step, (9.0 - f.reshape(-1))[:, None] * u, atol=1e-4), "x + (c - f) u"
    assert torch.allclose(S.clamp(sae, x, latent, 0.0)[0].reshape(-1, 16), x.reshape(-1, 16) - f.reshape(-1)[:, None] * u, atol=1e-4), "clamping at zero removes the latent's term"
    same, _ = S.clamp(sae, x[f > 0], latent, 0.0)
    assert not torch.allclose(same, x[f > 0])


def test_fold_norms_does_not_change_the_function():
    torch.manual_seed(0)
    sae = S.SparseAutoencoder(S.DictConfig(d=8, n=16, kind="relu", l1=0.1))
    with torch.no_grad():
        sae.dictionary *= torch.rand(16, 1) * 3 + 0.2
    x = torch.randn(10, 8)
    before = sae(x)[0].clone()
    sae.fold_norms()
    assert torch.allclose(sae(x)[0], before, atol=1e-5)


def test_harvest_leaves_out_the_mark_and_the_first_tokens_of_a_window():
    h = S.Harvest(layer=1, context=8, skip=2, bos_id=BOS)
    idx = torch.tensor([[1, 2, 3, BOS, 5, 6, 7, 8], [BOS, 1, 2, 3, 4, 5, BOS, 7]])
    assert h.keep(idx).tolist() == [[False, False, True, False, True, True, True, True],
                                    [False, False, True, True, True, True, False, True]]


def test_evaluation_scores_a_perfect_dictionary_and_an_empty_one():
    model = small_model()
    d = 32
    perfect = S.SparseAutoencoder(S.DictConfig(d=d, n=2 * d, kind="topk", k=d))
    empty = S.SparseAutoencoder(S.DictConfig(d=d, n=2 * d, kind="topk", k=d))
    with torch.no_grad():
        both = torch.cat([torch.eye(d), -torch.eye(d)])       # x = relu(x) - relu(-x): 2d latents, at most d active
        perfect.dictionary.copy_(both); perfect.encoder.copy_(both)
        empty.encoder.zero_(); empty.enc_bias.fill_(-1.0)
    harvest = S.Harvest(layer=1, context=32, skip=4, bos_id=BOS)
    a, b = S.evaluate(model, [perfect, empty], token_stream(32 * 40 + 1), harvest, n_splice_windows=16,
                      model_batch=8, device="cpu")
    assert a["fvu"] < 1e-9 and abs(a["loss_added"]) < 1e-4 and a["loss_recovered"] == pytest.approx(1.0, abs=1e-4)
    assert a["gain"] == pytest.approx(1.0, abs=1e-5) and a["l0"] <= d and a["dead"] == 0.0
    assert abs(a["fvu_rescaled"]) < 1e-6 and b["fvu_rescaled"] <= b["fvu"] + 1e-9, "the best gain can only lower the error"
    assert b["l0"] == 0.0 and b["dead"] == 1.0 and b["fvu"] > 0.5
    assert b["loss_spliced"] == pytest.approx(b["loss_zero"], abs=1e-5), "an empty code with zero bias and zero mean is the zero ablation"
    assert b["loss_recovered"] == pytest.approx(0.0, abs=1e-5)
    assert a["loss_zero"] > a["loss_clean"] and a["eval_tokens"] > 0 and sum(a["density_hist"]) == 2 * d
    assert float(perfect.threshold) > 0


def write_run(root: Path) -> Path:
    """A small run laid out as the trainer writes it, with its dataset, under a workspace `root`."""
    data = root / "data" / "toy" / "tok"
    data.mkdir(parents=True)
    token_stream(60_000, 1).tofile(data / "train_0000.bin")
    token_stream(8_000, 2).tofile(data / "val.bin")
    (data / "meta.json").write_text(json.dumps({"dataset": "toy", "tokenizer": "tok", "vocab_size": 64, "bos_id": BOS,
                                                "train_shards": ["train_0000.bin"], "val_file": "val.bin"}))
    run = root / "ckpt" / "toy-run"
    run.mkdir(parents=True)
    model = small_model()
    torch.save({"model": model.state_dict(), "model_config": model.cfg.to_dict(), "step": 7, "tokens": 0}, run / "ckpt_7.pt")
    (run / "config.json").write_text(json.dumps({"data_meta": {"dataset": "toy", "tokenizer": "tok"}}))
    return run


def test_the_training_stream_is_shuffled_seeded_and_leaves_out_the_mark(tmp_path, monkeypatch):
    monkeypatch.setenv("GPTPP_WORK", str(tmp_path))
    run = write_run(tmp_path)
    model, data, source = S.load_run(run, None, "cpu")
    assert source == {"run": "toy-run", "checkpoint": "ckpt_7.pt", "dataset": "toy", "tokenizer": "tok"}, "the latest checkpoint while model.pt is missing"
    harvest = S.Harvest(layer=1, context=32, skip=4, bos_id=BOS)
    def first(seed):
        return next(iter(S.TrainingStream(model, data, harvest, batch=64, buffer_windows=16, model_batch=8, seed=seed, device="cpu")))
    a, b, c = first(0), first(0), first(1)
    assert a.shape == (64, 32) and torch.equal(a, b) and not torch.equal(a, c)
    stream = S.TrainingStream(model, data, harvest, batch=64, buffer_windows=16, model_batch=8, seed=0, device="cpu")
    n = len(stream.fill())
    assert 16 * 28 * 0.85 < n <= 16 * 28, "every window gives its tokens after the first four, less the marks"


def test_the_command_line_trains_evaluates_saves_and_loads(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GPTPP_WORK", str(tmp_path))
    write_run(tmp_path)
    S.main(["--run", "toy-run", "--layer", "1", "--widths", "64", "--ks", "4", "--l1", "0.05", "--tokens", "30000", "--batch", "256",
            "--skip", "4", "--buffer", "4000", "--dead-after", "5000", "--splice-windows", "32", "--device", "cpu"])
    folder = S.dict_dir("toy-run", 1)
    assert sorted(p.name for p in folder.iterdir()) == ["relu-n64-l0.05.json", "relu-n64-l0.05.pt", "topk-n64-k4.json", "topk-n64-k4.pt"]
    record = json.loads((folder / "topk-n64-k4.json").read_text())
    assert record["checkpoint"] == "ckpt_7.pt" and record["layer"] == 1 and record["eval"]["l0"] == pytest.approx(4.0, abs=0.01)
    assert 0 < record["eval"]["fvu"] < 1 and record["log"][-1]["step"] == 30000 // 256 - 1
    sae = S.load(folder / "topk-n64-k4.pt", "cpu")
    assert sae.cfg.hook == "resid_post.1" and float(sae.threshold) == pytest.approx(record["eval"]["threshold"], rel=1e-5)
    x = torch.randn(5, 32) * 3
    assert sae.reconstruct(x).shape == x.shape
    relu = json.loads((folder / "relu-n64-l0.05.json").read_text())["eval"]
    assert relu["kind"] == "relu" and relu["l0"] > 0
    assert "clean loss" in capsys.readouterr().out
    before = record["eval"]
    S.main(["--run", "toy-run", "--layer", "1", "--skip", "4", "--splice-windows", "32", "--device", "cpu", "--evaluate-only"])
    again = json.loads((folder / "topk-n64-k4.json").read_text())
    assert again["eval"]["fvu"] == pytest.approx(before["fvu"], rel=1e-6) and again["log"] == record["log"], "scores are reproduced, the training log is kept"
