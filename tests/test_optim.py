"""Optimizers and schedules: the Newton-Schulz orthogonalization, Muon, Lion, parameter groups, learning rates."""

from __future__ import annotations

import math

import pytest
import torch
from conftest import tiny

from gptpp.config import TrainConfig, train_preset
from gptpp.model import GPT
from gptpp.optim import (NS_CLASSIC, NS_TUNED, Lion, Muon, build_optimizers, group_settings, lr_multiplier,
                         orthogonalize, set_lr)

# ----------------------------------------------------------------------------------------------------------------
# Newton-Schulz


def with_singular_values(values: torch.Tensor, rows: int, columns: int, seed: int = 0) -> tuple[torch.Tensor, ...]:
    """A (rows, columns) float64 matrix U diag(values) V^T with random orthonormal U and V."""
    generator = torch.Generator().manual_seed(seed)
    k = len(values)
    u, _ = torch.linalg.qr(torch.randn(rows, k, generator=generator, dtype=torch.float64))
    v, _ = torch.linalg.qr(torch.randn(columns, k, generator=generator, dtype=torch.float64))
    return u @ torch.diag(values) @ v.T, u, v


@pytest.mark.parametrize("shape", [(24, 24), (60, 24), (24, 60)])
def test_classic_newton_schulz_converges_to_the_orthogonal_factor(shape: tuple[int, int]) -> None:
    values = torch.linspace(0.05, 1.0, 24, dtype=torch.float64)
    g, u, v = with_singular_values(values, *shape)
    exact = u @ v.T  # the polar factor: the singular vectors of g with every singular value set to one
    assert torch.allclose(orthogonalize(g, steps=40, coefficients=NS_CLASSIC), exact, atol=1e-8)
    left, _, right = torch.linalg.svd(g, full_matrices=False)
    assert torch.allclose(left @ right, exact, atol=1e-8)


@pytest.mark.parametrize("shape", [(24, 24), (60, 24), (24, 60)])
def test_tuned_newton_schulz_keeps_the_singular_vectors(shape: tuple[int, int]) -> None:
    """Five tuned steps do not reach 1 exactly: they leave every singular value in [0.68, 1.21], with the
    singular vectors untouched."""
    values = torch.logspace(-2, 0, 24, dtype=torch.float64)  # a condition number of 100
    g, u, v = with_singular_values(values, *shape)
    out = orthogonalize(g, steps=5)
    core = u.T @ out @ v  # in the basis of g's singular vectors the result must be diagonal
    off_diagonal = core - torch.diag(core.diagonal())
    assert off_diagonal.abs().max() < 1e-9
    assert core.diagonal().min() > 0.68 and core.diagonal().max() < 1.21


def test_the_tuned_polynomial() -> None:
    """What five steps of s -> a s + b s^3 + c s^5 do to a singular value s in (0, 1]."""
    a, b, c = NS_TUNED
    s = torch.logspace(-6, 0, 100001, dtype=torch.float64)
    out = s.clone()
    for _ in range(5):
        out = a * out + b * out ** 3 + c * out ** 5
    converged = s >= 0.002
    assert out[converged].min() > 0.68 and out[converged].max() < 1.21
    assert torch.allclose(out[0] / s[0], torch.tensor(a ** 5, dtype=torch.float64), rtol=1e-6)  # tiny ones: times 485
    classic = s.clone()
    for _ in range(5):
        classic = NS_CLASSIC[0] * classic + NS_CLASSIC[1] * classic ** 3 + NS_CLASSIC[2] * classic ** 5
    assert classic[s < 0.01].max() < 0.35  # the classic iteration is still far from 1 there


def test_orthogonalize_batches_shapes_and_scale() -> None:
    torch.manual_seed(0)
    g = torch.randn(5, 12, 30, dtype=torch.float64)
    out = orthogonalize(g)
    assert out.shape == g.shape and out.dtype == g.dtype
    for i in range(5):
        assert torch.allclose(out[i], orthogonalize(g[i]), atol=1e-12)  # a batch is its matrices one by one
    assert torch.allclose(orthogonalize(g.mT), out.mT, atol=1e-12)  # tall is the transpose of wide
    assert torch.allclose(orthogonalize(1e4 * g), out, atol=1e-9)  # the size of the gradient does not matter,
    assert torch.allclose(orthogonalize(1e-4 * g), out, atol=1e-3)  # down to the 1e-7 that guards the division
    assert orthogonalize(torch.zeros(3, 4)).abs().max() == 0  # and a zero gradient gives a zero update


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_orthogonalize_in_bfloat16_on_the_gpu() -> None:
    torch.manual_seed(0)
    g = torch.randn(4, 256, 1024, device="cuda")
    out = orthogonalize(g)
    assert out.dtype == torch.float32
    singular = torch.linalg.svdvals(out.double())
    assert singular.min() > 0.6 and singular.max() < 1.3
    exact = orthogonalize(g.double().cpu())
    assert (out.double().cpu() - exact).abs().max() < 0.02 * exact.abs().max() + 0.01


# ----------------------------------------------------------------------------------------------------------------
# Muon


def matrix_parameter(rows: int, columns: int, seed: int) -> torch.nn.Parameter:
    generator = torch.Generator().manual_seed(seed)
    p = torch.nn.Parameter(torch.randn(rows, columns, generator=generator, dtype=torch.float64))
    p.grad = torch.randn(rows, columns, generator=generator, dtype=torch.float64)
    return p


@pytest.mark.parametrize("shape", [(16, 16), (48, 16), (16, 48)])
def test_muon_first_step(shape: tuple[int, int]) -> None:
    p = matrix_parameter(*shape, seed=0)
    before, grad = p.detach().clone(), p.grad.clone()
    Muon([p], lr=0.02, momentum=0.95).step()
    step = before - p.detach()
    scale = max(1.0, shape[0] / shape[1]) ** 0.5
    # From a zero momentum buffer the first update is the orthogonalized gradient itself.
    assert torch.allclose(step, 0.02 * scale * orthogonalize(grad), atol=1e-6)
    singular = torch.linalg.svdvals(step)
    assert 0.6 * 0.02 * scale < singular.min() and singular.max() < 1.25 * 0.02 * scale


def test_muon_ignores_the_size_of_the_gradient() -> None:
    steps = []
    for factor in (1.0, 1e3, 1e-3):
        p = matrix_parameter(12, 20, seed=1)
        start = p.detach().clone()
        optimizer = Muon([p], lr=0.05)
        for _ in range(3):
            p.grad = factor * (p.detach() @ torch.ones(20, 20, dtype=torch.float64))  # some function of p
            optimizer.step()
        steps.append(p.detach() - start)
    assert torch.allclose(steps[0], steps[1], atol=1e-8) and torch.allclose(steps[0], steps[2], atol=1e-6)


def test_muon_momentum_and_nesterov() -> None:
    g1 = matrix_parameter(8, 8, seed=2).grad
    g2 = matrix_parameter(8, 8, seed=3).grad
    for nesterov in (True, False):
        p = matrix_parameter(8, 8, seed=4)
        start = p.detach().clone()
        optimizer = Muon([p], lr=0.1, momentum=0.9, nesterov=nesterov)
        p.grad = g1.clone()
        optimizer.step()
        after_one = p.detach().clone()
        p.grad = g2.clone()
        optimizer.step()
        buffer1 = 0.1 * g1
        buffer2 = 0.9 * buffer1 + 0.1 * g2
        first = (0.1 * g1 + 0.9 * buffer1) if nesterov else buffer1
        second = (0.1 * g2 + 0.9 * buffer2) if nesterov else buffer2
        assert torch.allclose(start - after_one, 0.1 * orthogonalize(first), atol=1e-12)
        assert torch.allclose(after_one - p.detach(), 0.1 * orthogonalize(second), atol=1e-12)


def test_muon_weight_decay_is_decoupled() -> None:
    p = matrix_parameter(8, 8, seed=5)
    before = p.detach().clone()
    p.grad = torch.zeros_like(p)
    Muon([p], lr=0.1, weight_decay=0.5).step()
    assert torch.allclose(p.detach(), before * (1 - 0.1 * 0.5))  # a zero gradient: only the decay acts


def test_muon_stacks_matrices_of_equal_shape() -> None:
    shapes = [(8, 8), (8, 8), (32, 8), (8, 8), (8, 32), (32, 8)]
    together = [matrix_parameter(*shape, seed=i) for i, shape in enumerate(shapes)]
    alone = [matrix_parameter(*shape, seed=i) for i, shape in enumerate(shapes)]
    joint = Muon(together, lr=0.03)
    separate = [Muon([p], lr=0.03) for p in alone]
    for round_ in range(3):
        for i, (a, b) in enumerate(zip(together, alone)):
            a.grad = torch.sin(a.detach() * (round_ + 1) + i)
            b.grad = a.grad.clone()
        joint.step()
        for optimizer in separate:
            optimizer.step()
    for a, b in zip(together, alone):
        assert torch.allclose(a, b, atol=1e-12)


def test_muon_is_for_matrices() -> None:
    with pytest.raises(ValueError, match="matrices"):
        Muon([torch.nn.Parameter(torch.zeros(5))])


def test_muon_descends() -> None:
    """Least squares in a matrix: Muon with a decaying rate reaches the solution."""
    torch.manual_seed(0)
    target = torch.randn(10, 6, dtype=torch.float64)
    w = torch.nn.Parameter(torch.zeros(10, 6, dtype=torch.float64))
    optimizer = Muon([w], lr=0.1, momentum=0.9)
    losses = []
    for step in range(200):
        optimizer.param_groups[0]["lr"] = 0.1 * (1 - step / 200)
        loss = (w - target).pow(2).sum()
        (w.grad,) = torch.autograd.grad(loss, w)
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < 1e-3 * losses[0]


# ----------------------------------------------------------------------------------------------------------------
# Lion


def test_lion_moves_every_coordinate_by_the_learning_rate() -> None:
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(7, 5, dtype=torch.float64))
    before = p.detach().clone()
    g1 = torch.randn(7, 5, dtype=torch.float64)
    optimizer = Lion([p], lr=0.01, betas=(0.9, 0.99))
    p.grad = g1.clone()
    optimizer.step()
    assert torch.allclose(before - p.detach(), 0.01 * g1.sign())  # from zero momentum: the sign of the gradient
    momentum = 0.01 * g1
    g2 = torch.randn(7, 5, dtype=torch.float64)
    after_one = p.detach().clone()
    p.grad = g2.clone()
    optimizer.step()
    assert torch.allclose(after_one - p.detach(), 0.01 * (0.9 * momentum + 0.1 * g2).sign())
    q = torch.nn.Parameter(torch.ones(3, dtype=torch.float64))
    q.grad = torch.zeros(3, dtype=torch.float64)
    Lion([q], lr=0.1, weight_decay=0.2).step()
    assert torch.allclose(q.detach(), torch.full((3,), 1 - 0.1 * 0.2, dtype=torch.float64))


# ----------------------------------------------------------------------------------------------------------------
# Groups


def test_group_settings() -> None:
    new = group_settings(train_preset("gptpp-base"))
    assert new["matrices"] == dict(lr=0.02, weight_decay=0.0)
    assert new["embedding"]["lr"] == 0.3 and new["unembedding"]["lr"] == 0.008 and new["gains"]["lr"] == 0.01
    wide = group_settings(train_preset("gptpp-ladder"))  # the AdamW rates scale with 1/sqrt(width / 768)
    assert wide["embedding"]["lr"] == pytest.approx(0.3 * math.sqrt(768 / 512), abs=1e-4)
    assert wide["matrices"]["lr"] == 0.02
    old = group_settings(train_preset("gpt2-2019-base"))
    assert {name: g["lr"] for name, g in old.items()} == dict.fromkeys(old, 6e-4)  # one rate for everything
    assert group_settings(train_preset("gpt2-2019-ladder"))["embedding"]["lr"] == 2.4e-3  # tuned on short runs
    assert {name: g["weight_decay"] for name, g in old.items()} == \
        dict(embedding=0.1, unembedding=0.1, matrices=0.1, gains=0.0)  # decay on matrices only
    # rates quoted for a reference batch scale with the square root of the batch
    cfg = train_preset("gptpp-base").replace(lr_ref_batch=524288.0)
    scaled = group_settings(cfg, tokens_per_step=131072)
    assert scaled["matrices"]["lr"] == pytest.approx(0.01) and scaled["embedding"]["lr"] == pytest.approx(0.15)
    assert group_settings(cfg)["matrices"]["lr"] == 0.02


@pytest.mark.parametrize("optimizer", ["adamw", "adam", "sgd", "lion", "muon+adamw"])
@pytest.mark.parametrize("recipe", ["gptpp", "gpt2-2019"])
def test_build_optimizers_cover_every_parameter_once(optimizer: str, recipe: str) -> None:
    changes = dict(init="gpt2") if recipe == "gptpp" else dict(
        pos="learned", norm="layernorm", bias=True, tie_embeddings=True, qk_norm=False, embed_norm=False)
    model = GPT(tiny(**changes))
    cfg = TrainConfig(optimizer=optimizer, lr=0.02, lr_embedding=0.3, lr_unembedding=0.004, lr_gains=None,
                      weight_decay=0.1, wd_embedding=0.01)
    optimizers = build_optimizers(model, cfg)
    groups = [group for o in optimizers for group in o.param_groups]
    seen = [id(p) for group in groups for p in group["params"]]
    assert sorted(seen) == sorted(id(p) for p in model.parameters())
    by_name = {group["name"]: group for group in groups}
    assert by_name["matrices"]["lr"] == by_name["matrices"]["base_lr"] == 0.02
    assert by_name["matrices"]["weight_decay"] == 0.1
    if "gains" in by_name:
        assert by_name["gains"]["lr"] == 0.02 and by_name["gains"]["weight_decay"] == 0.0
    assert ("unembedding" in by_name) == (recipe == "gptpp")
    if recipe == "gptpp":
        assert by_name["embedding"]["lr"] == 0.3 and by_name["embedding"]["weight_decay"] == 0.01
        assert by_name["unembedding"]["lr"] == 0.004
    else:  # tied: the embedding's group owns the one matrix, trained at the unembedding's rate and decay
        assert by_name["embedding"]["lr"] == 0.004 and by_name["embedding"]["weight_decay"] == 0.0
    if optimizer == "muon+adamw":
        adamw, muon = optimizers
        assert isinstance(muon, Muon) and isinstance(adamw, torch.optim.AdamW)
        assert [g["name"] for g in muon.param_groups] == ["matrices"]
        assert {id(p) for p in muon.param_groups[0]["params"]} == {id(p) for p in model.param_groups()["matrices"]}
    else:
        assert len(optimizers) == 1
    set_lr(optimizers, 0.25)
    assert by_name["matrices"]["lr"] == 0.005 and by_name["embedding"]["lr"] == 0.25 * by_name["embedding"]["base_lr"]
    assert by_name["matrices"]["base_lr"] == 0.02


@pytest.mark.parametrize("optimizer, rates", [
    ("adamw", dict(lr=3e-3)), ("adam", dict(lr=3e-3)), ("sgd", dict(lr=0.3, momentum=0.9)),
    ("lion", dict(lr=2e-3, beta1=0.9, beta2=0.99)), ("muon+adamw", dict(lr=0.02, lr_embedding=0.3, lr_unembedding=0.004)),
])
def test_every_optimizer_trains_a_small_model(optimizer: str, rates: dict) -> None:
    """Forty steps on one fixed batch: the loss must fall well below its starting value of log(50)."""
    torch.manual_seed(0)
    cfg = tiny(init="gpt2" if optimizer != "muon+adamw" else "zero_proj")
    model = GPT(cfg)
    train_cfg = TrainConfig(optimizer=optimizer, lr_embedding=None, lr_unembedding=None, lr_gains=None).replace(**rates)
    optimizers = build_optimizers(model, train_cfg)
    generator = torch.Generator().manual_seed(1)
    idx = torch.randint(0, cfg.vocab_size, (8, cfg.block_size + 1), generator=generator)
    x, y = idx[:, :-1], idx[:, 1:]
    first = model(x, y).item()
    for _ in range(40):
        loss = model(x, y)
        loss.backward()
        for o in optimizers:
            o.step()
            o.zero_grad(set_to_none=True)
    assert abs(first - math.log(cfg.vocab_size)) < 0.2
    assert model(x, y).item() < first - 1.0, optimizer


# ----------------------------------------------------------------------------------------------------------------
# Schedules


def schedule(name: str, **changes: float) -> TrainConfig:
    return TrainConfig(schedule=name, warmup_tokens=100.0, final_lr_frac=0.1, cooldown_frac=0.4).replace(**changes)


@pytest.mark.parametrize("name", ["cosine", "linear", "constant", "trapezoid"])
def test_warmup_is_linear_from_zero(name: str) -> None:
    cfg = schedule(name)
    assert lr_multiplier(0, 1000, cfg) == 0.0
    assert lr_multiplier(25, 1000, cfg) == pytest.approx(0.25)
    assert lr_multiplier(99.999, 1000, cfg) == pytest.approx(1.0, abs=1e-4)
    assert lr_multiplier(100, 1000, cfg) == pytest.approx(1.0)
    no_warmup = schedule(name, warmup_tokens=0.0)
    assert lr_multiplier(0, 1000, no_warmup) == 1.0


def test_schedule_shapes() -> None:
    total = 1000.0
    cosine, linear, constant, trapezoid = (schedule(n) for n in ("cosine", "linear", "constant", "trapezoid"))
    assert lr_multiplier(total, total, cosine) == pytest.approx(0.1)
    assert lr_multiplier(550, total, cosine) == pytest.approx(0.1 + 0.9 * 0.5)  # halfway down the cosine
    assert lr_multiplier(total, total, linear) == pytest.approx(0.1)
    assert lr_multiplier(550, total, linear) == pytest.approx(0.1 + 0.9 * 0.5)
    assert lr_multiplier(325, total, linear) == pytest.approx(0.1 + 0.9 * 0.75)
    assert all(lr_multiplier(t, total, constant) == 1.0 for t in (100, 500, 1000))
    # trapezoid: flat until the last 40% of the budget, then a straight line down to the final fraction
    assert all(lr_multiplier(t, total, trapezoid) == 1.0 for t in (100, 300, 599, 600))
    assert lr_multiplier(800, total, trapezoid) == pytest.approx(0.1 + 0.9 * 0.5)
    assert lr_multiplier(total, total, trapezoid) == pytest.approx(0.1)
    assert lr_multiplier(2 * total, total, trapezoid) == pytest.approx(0.1)  # past the end: the final value
    # a cooldown that would begin inside the warmup begins when the warmup ends
    assert lr_multiplier(100, total, schedule("trapezoid", cooldown_frac=1.0)) == pytest.approx(1.0)
    for cfg in (cosine, linear, trapezoid):  # never increasing after the warmup, always within [final, 1]
        values = [lr_multiplier(t, total, cfg) for t in range(100, 1001, 10)]
        assert all(a >= b - 1e-12 for a, b in zip(values, values[1:]))
        assert min(values) == pytest.approx(0.1) and max(values) == 1.0


def test_schedules_are_functions_of_tokens_not_steps() -> None:
    """The same budget cut into steps of two different sizes gives the same rate at the same token count."""
    cfg = train_preset("gpt2-2019-ladder")
    total = 150e6
    for tokens in (1e6, 8e6, 40e6, 149e6):
        small_steps = lr_multiplier(round(tokens / 16384) * 16384, total, cfg)
        large_steps = lr_multiplier(round(tokens / 262144) * 262144, total, cfg)
        assert small_steps == pytest.approx(large_steps, abs=2e-2)
    assert lr_multiplier(4e6, total, cfg) == pytest.approx(0.5)  # half of the 8M-token warmup


def test_one_line_changed_rules_train_the_matrices_and_keep_the_recipes_adamw_for_the_rest():
    """optimizer=X+adamw: the hidden matrices under X, the tables and gains under the recipe's AdamW with the
    recipe's betas whatever beta1 and beta2 say; every parameter is in exactly one group; a step moves all."""
    import torch
    from gptpp.config import ModelConfig, TrainConfig
    from gptpp.model import GPT
    from gptpp.optim import Lion, build_optimizers

    torch.manual_seed(0)
    model = GPT(ModelConfig(n_layer=2, n_head=2, d_model=32, vocab_size=64, block_size=16))
    recipe = TrainConfig()
    for name, kind in (("adamw+adamw", torch.optim.AdamW), ("adam+adamw", torch.optim.Adam), ("lion+adamw", Lion),
                       ("sgd+adamw", torch.optim.SGD)):
        cfg = TrainConfig(optimizer=name, lr=1e-3, beta1=0.9, beta2=0.99, weight_decay=0.0)
        rest, rule = build_optimizers(model, cfg)
        assert type(rest) is torch.optim.AdamW and type(rule) is kind
        assert all(g["betas"] == (recipe.beta1, recipe.beta2) for g in rest.param_groups)
        assert [g["name"] for g in rule.param_groups] == ["matrices"] and rule.param_groups[0]["lr"] == 1e-3
        assert {g["name"] for g in rest.param_groups} <= {"embedding", "unembedding", "gains"}
        seen = [id(p) for o in (rest, rule) for g in o.param_groups for p in g["params"]]
        assert len(seen) == len(set(seen)) == len(list(model.parameters()))
        before = [p.detach().clone() for p in model.parameters()]
        idx = torch.randint(0, 64, (4, 16))
        for _ in range(3):   # the writers to the stream start at zero, so a matrix behind one moves from the second step on
            logits = model(idx[:, :-1])
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), idx[:, 1:].reshape(-1))
            loss.backward()
            for o in (rest, rule):
                o.step(); o.zero_grad()
        assert all(not torch.equal(a, p) for a, p in zip(before, model.parameters()))
