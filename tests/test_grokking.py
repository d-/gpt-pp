"""The Fourier measures of the modular-addition run (gptpp/interp/grokking.py), on cases worked by hand."""
import math

import torch
import torch.nn.functional as F

from gptpp.interp import grokking as g


def test_fourier_basis_is_orthonormal():
    for p in (7, 113):
        basis = g.fourier_basis(p)
        assert basis.shape == (p, p)
        assert torch.allclose(basis @ basis.T, torch.eye(p, dtype=torch.float64), atol=1e-12)


def test_embedding_power_finds_the_frequency_and_adds_up_to_the_squared_norm():
    p, d = 113, 8
    basis = g.fourier_basis(p)
    a = torch.arange(p, dtype=torch.float64)
    w_e = torch.zeros(p, d, dtype=torch.float64)
    w_e[:, 0] = torch.cos(2 * math.pi * 14 * a / p)          # frequency 14: a cosine in one coordinate,
    w_e[:, 1] = 0.5 * torch.sin(2 * math.pi * 14 * a / p)    # half a sine in another
    w_e[:, 2] = 2.0                                          # and a constant
    power = g.embedding_power(w_e, basis)
    assert power.shape == (57,)
    assert torch.isclose(power.sum(), w_e.pow(2).sum())
    assert torch.isclose(power[0], torch.tensor(4.0 * p, dtype=torch.float64))
    assert torch.isclose(power[14], torch.tensor((1 + 0.25) * p / 2, dtype=torch.float64))
    assert power[1:14].abs().max() < 1e-20 and power[15:].abs().max() < 1e-20
    assert g.key_frequencies(power) == [14]


def test_random_embedding_has_no_key_frequencies_to_speak_of():
    power = g.embedding_power(torch.randn(113, 128, generator=torch.Generator().manual_seed(0)), g.fourier_basis(113))
    assert len(g.key_frequencies(power)) > 40               # every frequency is within a factor of ten of the strongest


def test_progress_measures_on_the_exact_algorithm():
    """Logits alpha * cos(w k (a + b - c)) lie entirely along the waves of a + b: keeping only that part changes
    nothing, removing it leaves the uniform distribution."""
    p, k, alpha = 23, 5, 30.0
    tokens, answers = g.all_pairs(p)
    a, b, c = tokens[:, 0].double(), tokens[:, 1].double(), torch.arange(p, dtype=torch.float64)
    logits = alpha * torch.cos(2 * math.pi * k * (a[:, None] + b[:, None] - c[None, :]) / p)
    train = g.train_mask(p, 0.3, 0)
    assert int(train.sum()) == int(0.3 * p * p)
    directions = g.key_directions(p, [k], g.fourier_basis(p))
    assert torch.allclose(directions @ directions.T, torch.eye(4, dtype=torch.float64), atol=1e-12)
    waves = g.key_directions(p, [k], g.fourier_basis(p), products=False)             # cos and sin of w (a + b) lie in their span
    assert torch.allclose(waves @ waves.T, torch.eye(2, dtype=torch.float64), atol=1e-12)
    assert torch.allclose((waves @ directions.T) @ (directions @ waves.T), torch.eye(2, dtype=torch.float64), atol=1e-12)
    assert g.key_directions(113, [14, 35, 41, 42, 52], g.fourier_basis(113)).shape == (20, 113 * 113)   # the paper's 20 terms
    train_idx, test_idx = torch.where(train)[0], torch.where(~train)[0]
    restricted_test, restricted_all, excluded_train = g.progress_measures(logits, answers, train_idx, test_idx, directions).tolist()
    assert abs(restricted_all - F.cross_entropy(logits, answers).item()) < 1e-9
    assert abs(restricted_test - F.cross_entropy(logits[~train], answers[~train]).item()) < 1e-9
    assert abs(excluded_train - math.log(p)) < 1e-9
    assert torch.allclose(g.progress_measures(logits, answers, train_idx, test_idx, waves), g.progress_measures(logits, answers, train_idx, test_idx, directions))
    other = g.key_directions(p, [k + 1], g.fourier_basis(p))            # a frequency the logits do not use
    assert abs(g.progress_measures(logits, answers, train_idx, test_idx, other)[2].item() - F.cross_entropy(logits[train], answers[train]).item()) < 1e-9


def test_the_model_is_the_papers_shape_and_a_short_run_fits_its_training_pairs():
    model = g.build_model(113, seed=0)
    n = model.num_params()
    assert (model.cfg.n_layer, model.cfg.d_model, model.cfg.n_head, model.cfg.d_mlp) == (1, 128, 4, 512)
    assert n["embedding"] == 114 * 128 and n["position"] == 3 * 128 and n["norm"] == 0
    assert abs(model.embed.weight.std().item() - 1 / math.sqrt(128)) < 0.005
    torch.set_num_threads(2)
    result = g.run(p=11, steps=300, device="cpu", every=10 ** 9)
    history = result["history"]
    partial = g.run(p=11, steps=300, device="cpu", every=10 ** 9, budget=0.0)["partial"]      # stops at the first check
    resumed = g.run(p=11, steps=300, device="cpu", every=10 ** 9, resume=partial)["history"]
    assert partial["step"] == 50 and resumed["step"] == history["step"]
    assert max(abs(a - b) for a, b in zip(resumed["train_loss"], history["train_loss"])) < 1e-6
    assert history["step"][0] == 0 and history["step"][-1] == 300 and len(result["states"]) == len(history["step"])
    assert history["train_loss"][-1] < 0.05 * history["train_loss"][0]
    assert len(history["embedding_power"][0]) == 6 and len(history["excluded_train"]) == len(history["step"])
    assert len(history["restricted_test_sum"]) == len(history["step"]) and len(history["excluded_train_sum"]) == len(history["step"])
    assert g.first_step_at({"step": [0, 1, 2, 3], "acc": [0.0, 1.0, 0.5, 1.0]}, "acc", 0.99) == 3
