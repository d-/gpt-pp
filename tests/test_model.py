"""The model: causality, the two paths, hooks, cached decoding, positions, sampling, configuration.

Most tests run over the whole grid of variants in conftest.VARIANTS, on the CPU in float64, with every
parameter randomized so that no component is silent.
"""

from __future__ import annotations

import json
import math

import pytest
import torch
import torch.nn.functional as F
from conftest import tiny

from gptpp.config import (ModelConfig, TrainConfig, apply_overrides, preset, preset_names, train_preset)
from gptpp.generate import generate, sample, truncate
from gptpp.model import GPT, Hooks, KVCache, alibi_slopes, rotary_table, rotate, sinusoidal_table

TOL = 1e-5  # logits are returned in float32


def build(cfg: ModelConfig, seed: int = 0) -> GPT:
    """A model in float64 whose every parameter is random and of order one: matrices at 1/sqrt(fan_in), gains
    and scales near 1, biases around 0."""
    torch.manual_seed(seed)
    model = GPT(cfg)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim == 2:
                p.normal_(0.0, 1.0 / math.sqrt(p.shape[1]))
            elif name.endswith("bias"):
                p.normal_(0.0, 0.3)
            else:  # gains, the slope of dynamic tanh, learned attention scales
                p.mul_(0.7 + 0.6 * torch.rand_like(p))
    return model.double().eval()


def tokens(cfg: ModelConfig, batch: int = 3, length: int | None = None, seed: int = 1) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, cfg.vocab_size, (batch, length or cfg.block_size), generator=generator)


# ----------------------------------------------------------------------------------------------------------------
# The contract of forward


def test_forward_shapes_and_loss(variant: ModelConfig) -> None:
    model, idx = build(variant), tokens(variant)
    logits = model(idx)
    assert logits.shape == (3, variant.block_size, variant.vocab_size)
    assert logits.dtype == torch.float32
    targets = tokens(variant, seed=2)
    loss = model(idx, targets)
    expected = F.cross_entropy(logits.view(-1, variant.vocab_size), targets.view(-1))
    assert loss.ndim == 0 and torch.allclose(loss, expected, atol=1e-6)
    # A target of -1 is ignored: the loss is the mean over the other positions.
    masked = targets.clone()
    masked[:, :5] = -1
    kept = F.cross_entropy(logits[:, 5:].reshape(-1, variant.vocab_size), targets[:, 5:].reshape(-1))
    assert torch.allclose(model(idx, masked), kept, atol=1e-6)


def test_causality(variant: ModelConfig) -> None:
    """Changing a token never changes the logits of earlier positions, on either path."""
    model, idx = build(variant), tokens(variant)
    paths = {"fast": model, "explicit": lambda x: model.run_with_cache(x, names=[])[0]}
    for name, path in paths.items():
        reference = path(idx)
        for t in (1, 7, variant.block_size - 1):
            changed = idx.clone()
            changed[:, t] = (changed[:, t] + 1) % variant.vocab_size
            logits = path(changed)
            assert torch.equal(logits[:, :t], reference[:, :t]), f"{name} path: position {t} leaks into the past"
            assert (logits[:, t:] - reference[:, t:]).abs().max() > 1e-3, f"{name} path: token {t} has no effect"


def test_fast_path_equals_explicit_path(variant: ModelConfig) -> None:
    model, idx = build(variant), tokens(variant)
    fast = model(idx)
    explicit, cache = model.run_with_cache(idx)
    assert torch.allclose(fast, explicit, atol=TOL)
    assert torch.allclose(cache["logits"], fast, atol=TOL)
    assert set(cache) == {name for name in model.hook_names() if not name.endswith(".result")}


def test_two_paths_agree_in_training_mode_without_dropout(variant: ModelConfig) -> None:
    model, idx = build(variant).train(), tokens(variant)
    assert torch.allclose(model(idx), model.forward_explicit(idx), atol=TOL)


# ----------------------------------------------------------------------------------------------------------------
# The cache: what each named activation is


def test_cache_shapes(variant: ModelConfig) -> None:
    cfg = variant
    model, idx = build(cfg), tokens(cfg, batch=2)
    _, cache = model.run_with_cache(idx, names=["*"])
    B, T, H, Hkv, dh, d = 2, cfg.block_size, cfg.n_head, cfg.kv_heads, cfg.d_head, cfg.d_model
    for l in range(cfg.n_layer):
        assert cache[f"attn.{l}.q"].shape == (B, H, T, dh)
        assert cache[f"attn.{l}.k"].shape == (B, Hkv, T, dh)
        assert cache[f"attn.{l}.v"].shape == (B, Hkv, T, dh)
        assert cache[f"attn.{l}.scores"].shape == (B, H, T, T)
        assert cache[f"attn.{l}.pattern"].shape == (B, H, T, T)
        assert cache[f"attn.{l}.z"].shape == (B, H, T, dh)
        assert cache[f"attn.{l}.result"].shape == (B, H, T, d)
        for name in ("resid_pre", "attn_out", "resid_mid", "resid_post"):
            assert cache[f"{name}.{l}"].shape == (B, T, d)
        if cfg.use_mlp:
            assert cache[f"mlp.{l}.pre"].shape == (B, T, cfg.d_mlp)
            assert cache[f"mlp.{l}.post"].shape == (B, T, cfg.d_mlp)
            assert cache[f"mlp_out.{l}"].shape == (B, T, d)
        else:
            assert f"mlp_out.{l}" not in cache
    assert cache["embed"].shape == cache["final_norm"].shape == (B, T, d)
    assert cache["logits"].shape == (B, T, cfg.vocab_size)


def test_cache_is_the_residual_decomposition(variant: ModelConfig) -> None:
    """The stream is the embedding plus what every sublayer adds (pre-norm), each activation is what its name
    says, and the attention pattern is a causal distribution."""
    cfg = variant
    model, idx = build(cfg), tokens(cfg, batch=2)
    _, c = model.run_with_cache(idx, names=["*"])
    close = lambda a, b: torch.allclose(a, b, atol=1e-9)  # noqa: E731
    assert close(c["resid_pre.0"], c["embed"])
    future = torch.ones(cfg.block_size, cfg.block_size).triu(diagonal=1).bool()
    for l, block in enumerate(model.blocks):
        pre, mid, post = c[f"resid_pre.{l}"], c[f"resid_mid.{l}"], c[f"resid_post.{l}"]
        if cfg.norm_placement == "pre":
            assert close(mid, pre + c[f"attn_out.{l}"])
            assert close(post, mid + c[f"mlp_out.{l}"] if cfg.use_mlp else mid)
        else:
            assert close(mid, block.norm1(pre + c[f"attn_out.{l}"]))
            assert close(post, block.norm2(mid + c[f"mlp_out.{l}"]) if cfg.use_mlp else mid)
        if l + 1 < cfg.n_layer:
            assert close(c[f"resid_pre.{l + 1}"], post)

        pattern, scores, z = c[f"attn.{l}.pattern"], c[f"attn.{l}.scores"], c[f"attn.{l}.z"]
        assert close(pattern.sum(-1), torch.ones_like(pattern.sum(-1)))
        assert (pattern[..., future] == 0).all() and (pattern >= 0).all()
        assert torch.isinf(scores[..., future]).all() and torch.isfinite(scores[..., ~future]).all()
        assert close(pattern, scores.softmax(-1))
        _, v = block.attn.share(c[f"attn.{l}.k"], c[f"attn.{l}.v"])
        assert close(z, pattern @ v)
        attn = block.attn
        merged = z.transpose(1, 2).reshape(2, cfg.block_size, -1)
        assert close(c[f"attn_out.{l}"], attn.o(merged))
        by_head = c[f"attn.{l}.result"].sum(1) + (attn.o.bias if attn.o.bias is not None else 0)
        assert close(by_head, c[f"attn_out.{l}"])
        if cfg.use_mlp:
            assert close(c[f"mlp_out.{l}"], block.mlp.proj(c[f"mlp.{l}.post"]))
            if block.mlp.gate is None:
                assert close(c[f"mlp.{l}.post"], block.mlp.act(c[f"mlp.{l}.pre"]))
    assert close(c["final_norm"], model.final_norm(c[f"resid_post.{cfg.n_layer - 1}"]))
    assert torch.allclose(c["logits"], model.unembed_logits(c["final_norm"]), atol=TOL)


def test_cache_selection() -> None:
    model = build(tiny(init="gpt2"))
    idx = tokens(model.cfg)
    _, cache = model.run_with_cache(idx, names=["attn.*.pattern", "resid_post.1"])
    assert set(cache) == {"attn.0.pattern", "attn.1.pattern", "resid_post.1"}
    _, cache = model.run_with_cache(idx, names=lambda name: name.startswith("mlp_out"))
    assert set(cache) == {"mlp_out.0", "mlp_out.1"}
    _, cache = model.run_with_cache(idx, names=["attn.0.result"])
    assert set(cache) == {"attn.0.result"}
    with pytest.raises(KeyError):
        model.run_with_cache(idx, names=["resid_post.9"])
    with pytest.raises(KeyError):
        model.run_with_hooks(idx, {"resid_pst.0": lambda x: x})


def test_hooks_intervene(variant: ModelConfig) -> None:
    cfg = variant
    model = build(cfg)
    clean, corrupted = tokens(cfg, seed=1), tokens(cfg, seed=2)
    clean_logits, clean_cache = model.run_with_cache(clean)
    last = cfg.n_layer - 1

    # A hook that returns None, or its input, changes nothing.
    seen = []
    same = model.run_with_hooks(clean, {"resid_mid.0": lambda x: seen.append(x.shape), "attn.1.z": lambda x: x})
    assert torch.equal(same, clean_logits) and seen == [(3, cfg.block_size, cfg.d_model)]

    # Ablation: zeroing the output of an attention layer changes the logits.
    ablated = model.run_with_hooks(clean, {"attn_out.0": torch.zeros_like})
    assert (ablated - clean_logits).abs().max() > 1e-3

    # Activation patching: the whole stream of the clean run, patched into the corrupted run at any depth,
    # restores the clean logits, because nothing downstream reads the tokens again.
    for name in ("embed", f"resid_pre.{last}", f"resid_mid.{last}", f"resid_post.{last}", "final_norm"):
        patched = model.run_with_hooks(corrupted, {name: lambda x, name=name: clean_cache[name]})
        assert torch.allclose(patched, clean_logits, atol=TOL), name

    # Patching the pattern of one layer only moves the logits part of the way.
    partial = model.run_with_hooks(corrupted, {"attn.0.pattern": lambda x: clean_cache["attn.0.pattern"]})
    assert (partial - clean_logits).abs().max() > 1e-3

    # Hooks and the cache together: the cache records what the intervention produced.
    _, cache = model.run_with_cache(clean, names=["attn_out.0", "resid_mid.0"], hooks={"attn_out.0": torch.zeros_like})
    assert cache["attn_out.0"].abs().max() == 0
    if cfg.norm_placement == "pre":
        assert torch.equal(cache["resid_mid.0"], clean_cache["resid_pre.0"])


def test_cache_carries_gradients() -> None:
    """Cached activations are nodes of the autograd graph, as attribution methods need."""
    model, idx = build(tiny(init="gpt2")), tokens(tiny())
    logits, cache = model.run_with_cache(idx, names=["resid_post.0"])
    (gradient,) = torch.autograd.grad(logits[:, -1].logsumexp(-1).sum(), cache["resid_post.0"])
    assert gradient.shape == cache["resid_post.0"].shape and gradient.abs().max() > 0


# ----------------------------------------------------------------------------------------------------------------
# KV-cached decoding


def test_cached_decoding_equals_full_forward(variant: ModelConfig) -> None:
    model, idx = build(variant), tokens(variant, batch=2)
    full = model(idx)
    # one token at a time
    kv = KVCache(variant.n_layer)
    stepwise = torch.cat([model.forward_explicit(idx[:, t:t + 1], kv=kv) for t in range(idx.size(1))], dim=1)
    assert kv.length == idx.size(1)
    assert torch.allclose(stepwise, full, atol=TOL)
    # a prompt of five tokens, then chunks of three, two and the rest
    kv = KVCache(variant.n_layer)
    pieces = [model.forward_explicit(piece, kv=kv) for piece in idx.split([5, 3, 2, idx.size(1) - 10], dim=1)]
    assert torch.allclose(torch.cat(pieces, dim=1), full, atol=TOL)
    # the cache holds one key and one value per key/value head and token
    assert kv.keys[0].shape == (2, variant.kv_heads, idx.size(1), variant.d_head)


# ----------------------------------------------------------------------------------------------------------------
# Positions


def test_rotation_depends_on_relative_position_only() -> None:
    torch.manual_seed(0)
    dh, n = 16, 64
    cos, sin = (t.double() for t in rotary_table(n, dh, 10000.0))
    q = torch.randn(dh, dtype=torch.float64).expand(1, 1, n, dh)
    k = torch.randn(dh, dtype=torch.float64).expand(1, 1, n, dh)
    scores = (rotate(q, cos, sin) @ rotate(k, cos, sin).transpose(-2, -1))[0, 0]  # the same q and k everywhere
    for offset in range(-n + 1, n):
        diagonal = scores.diagonal(offset)
        assert torch.allclose(diagonal, diagonal[0].expand_as(diagonal), atol=1e-6), offset
    assert scores.diagonal(0).std() < 1e-6 and scores[0].std() > 1e-2  # constant along diagonals, not across
    # a rotation: norms are kept, and position 0 is the identity
    x = torch.randn(2, 3, n, dh, dtype=torch.float64)
    assert torch.allclose(rotate(x, cos, sin).norm(dim=-1), x.norm(dim=-1))
    assert torch.allclose(rotate(x, cos, sin)[:, :, 0], x[:, :, 0])


@pytest.mark.parametrize("changes", [dict(), dict(qk_norm=False), dict(n_kv_head=2), dict(norm_placement="post")])
def test_rotary_model_scores_are_functions_of_the_offset(changes: dict) -> None:
    """With every token the same, queries and keys do not depend on the position, so the scores of a rotary
    model must be constant along each diagonal, in every layer and head."""
    cfg = tiny(init="gpt2", pos="rope", **changes)
    model = build(cfg)
    idx = torch.full((1, cfg.block_size), 7)
    _, cache = model.run_with_cache(idx, names=["attn.*.scores"])
    for name, scores in cache.items():
        for offset in range(-cfg.block_size + 1, 1):
            diagonal = scores.diagonal(offset, dim1=-2, dim2=-1)
            assert torch.allclose(diagonal, diagonal[..., :1].expand_as(diagonal), atol=1e-8), (name, offset)
        assert scores[0, 0, -1, :].std() > 1e-4  # and they do vary with the offset


@pytest.mark.parametrize("pos", ["rope", "alibi", "none"])
def test_relative_models_ignore_absolute_position(pos: str) -> None:
    """Decoding a sequence as if it began at position 37 gives the same logits: only offsets matter."""
    cfg = tiny(init="gpt2", pos=pos)
    model, idx = build(cfg), tokens(cfg)
    first, last = 37, 37 + idx.size(1)
    model.extend_positions(last)
    rot = model.rotation(first, last)
    bias = model.causal_bias(first, last)[..., first:]  # the keys are the sequence's own tokens
    x = model.embed_tokens(idx, first)
    for layer, block in enumerate(model.blocks):
        x = block.forward_explicit(x, rot, bias, Hooks(), layer)
    logits = model.unembed_logits(model.final_norm(x))
    assert torch.allclose(logits, model(idx), atol=TOL)


@pytest.mark.parametrize("pos", ["learned", "sinusoidal"])
def test_absolute_models_depend_on_absolute_position(pos: str) -> None:
    cfg = tiny(init="gpt2", pos=pos, block_size=32)
    model, idx = build(cfg), tokens(cfg, length=8)
    assert (model.embed_tokens(idx, 0) - model.embed_tokens(idx, 5)).abs().max() > 1e-3


@pytest.mark.parametrize("pos", ["sinusoidal", "rope", "alibi", "none"])
def test_sequences_longer_than_block_size(pos: str) -> None:
    cfg = tiny(init="gpt2", pos=pos)
    model = build(cfg)
    idx = tokens(cfg, length=3 * cfg.block_size)
    logits = model(idx)
    assert logits.shape == (3, 3 * cfg.block_size, cfg.vocab_size) and torch.isfinite(logits).all()
    assert torch.allclose(logits, model.forward_explicit(idx), atol=TOL)
    # the first block_size positions are computed exactly as before the tables were extended
    assert torch.allclose(logits[:, :cfg.block_size], model(idx[:, :cfg.block_size]), atol=TOL)


def test_learned_positions_end_at_block_size() -> None:
    cfg = tiny(pos="learned")
    model = build(cfg)
    with pytest.raises(ValueError, match="block_size"):
        model(tokens(cfg, length=cfg.block_size + 1))


def test_position_tables() -> None:
    table = sinusoidal_table(100, 32)
    assert table.shape == (100, 32) and table.abs().max() <= 1
    assert torch.allclose(table[0, 0::2], torch.zeros(16)) and torch.allclose(table[0, 1::2], torch.ones(16))
    assert torch.allclose(table[5, 0], torch.sin(torch.tensor(5.0)))  # the fastest coordinate has frequency 1
    slopes = alibi_slopes(8)
    assert torch.allclose(slopes, torch.tensor([2.0 ** -k for k in range(1, 9)], dtype=torch.float64))
    cfg = tiny(init="gpt2", pos="alibi", qk_norm=False)
    model = build(cfg)
    bias = model.causal_bias(0, 6)
    assert bias.shape == (cfg.n_head, 6, 6)
    assert torch.allclose(bias[0, 5, :3].double(), -alibi_slopes(cfg.n_head)[0] * torch.tensor([5.0, 4.0, 3.0]).double())
    assert torch.isinf(bias[:, 2, 3:]).all() and (bias.diagonal(dim1=1, dim2=2) == 0).all()


# ----------------------------------------------------------------------------------------------------------------
# Attention variants, the MLP, the two ends


def test_grouped_query_attention() -> None:
    full = build(tiny(init="gpt2", n_kv_head=4))
    assert full.blocks[0].attn.k.weight.shape == (32, 32)
    default = build(tiny(init="gpt2"))
    default.load_state_dict(full.state_dict())
    idx = tokens(default.cfg)
    assert torch.equal(default(idx), full(idx))  # n_kv_head = n_head is ordinary attention
    grouped = build(tiny(init="gpt2", n_kv_head=2))
    assert grouped.blocks[0].attn.k.weight.shape == (16, 32) and grouped.blocks[0].attn.v.weight.shape == (16, 32)
    assert grouped.num_params()["attention"] < default.num_params()["attention"]
    # heads 0 and 1 share a key head, so with equal queries they have equal patterns
    with torch.no_grad():
        w = grouped.blocks[0].attn.q.weight
        w[8:16] = w[0:8]
    _, cache = grouped.run_with_cache(idx, names=["attn.0.pattern"])
    pattern = cache["attn.0.pattern"]
    assert torch.allclose(pattern[:, 0], pattern[:, 1]) and not torch.allclose(pattern[:, 0], pattern[:, 2])


@pytest.mark.parametrize("learned", [False, True])
def test_qk_norm_bounds_the_scores(learned: bool) -> None:
    cfg = tiny(init="gpt2", qk_scale=7.0, qk_scale_learned=learned, pos="rope")
    torch.manual_seed(0)
    model = GPT(cfg).double().eval()
    with torch.no_grad():
        for block in model.blocks:  # huge queries and keys: without normalization the softmax would saturate
            block.attn.q.weight.normal_(0, 30.0)
            block.attn.k.weight.normal_(0, 30.0)
    _, cache = model.run_with_cache(tokens(cfg), names=["attn.*.scores"])
    for scores in cache.values():  # a score is 7 times a cosine
        finite = scores[torch.isfinite(scores)]
        assert finite.abs().max() <= 7.0 + 1e-6 and finite.abs().max() > 0.5 * 7.0
    assert ("blocks.0.attn.qk_factor" in dict(model.named_parameters())) == learned
    if learned:  # the learned factor multiplies the scale, head by head
        with torch.no_grad():
            model.blocks[0].attn.qk_factor.copy_(torch.tensor([1.0, 2.0, 0.5, 1.0]))
        _, scaled = model.run_with_cache(tokens(cfg), names=["attn.0.scores"])
        before, after = cache["attn.0.scores"], scaled["attn.0.scores"]
        finite = torch.isfinite(before[:, 0])
        for head, factor in enumerate((1.0, 2.0, 0.5, 1.0)):
            assert torch.allclose(after[:, head][finite], factor * before[:, head][finite], atol=1e-9)


def test_gated_mlp_keeps_the_parameter_count() -> None:
    for width in (128, 512, 768):
        plain = ModelConfig(d_model=width, act="gelu")
        gated = ModelConfig(d_model=width, act="swiglu")
        assert plain.d_mlp == 4 * width and gated.d_mlp == round(8 * width / 3)
        assert abs(3 * gated.d_mlp - 2 * plain.d_mlp) <= 1  # three matrices against two
    model = build(tiny(init="gpt2", act="swiglu"))
    x = torch.randn(2, 5, 32, dtype=torch.float64)
    mlp = model.blocks[0].mlp
    assert torch.allclose(mlp(x), mlp.proj(F.silu(mlp.gate(x)) * mlp.fc(x)))


def test_activations() -> None:
    from gptpp.model import ACTIVATIONS
    x = torch.linspace(-3, 3, 13, dtype=torch.float64)
    assert torch.equal(ACTIVATIONS["relu"](x), x.clamp(min=0))
    assert torch.equal(ACTIVATIONS["relu2"](x), x.clamp(min=0) ** 2)
    assert torch.allclose(ACTIVATIONS["silu"](x), x * torch.sigmoid(x))
    assert torch.allclose(ACTIVATIONS["gelu"](x), x * 0.5 * (1 + torch.erf(x / math.sqrt(2))))


def test_norms() -> None:
    from gptpp.model import DynamicTanh, LayerNorm, RMSNorm, make_norm
    x = torch.randn(4, 7, 32, dtype=torch.float64) * 5 + 2
    rms = RMSNorm(32, gain=False)(x)
    assert torch.allclose(rms.pow(2).mean(-1), torch.ones(4, 7, dtype=torch.float64), atol=1e-6)
    assert not list(RMSNorm(32, gain=False).parameters()) and len(list(RMSNorm(32, gain=True).parameters())) == 1
    ln = LayerNorm(32, bias=True).double()(x)
    assert torch.allclose(ln.mean(-1), torch.zeros(4, 7, dtype=torch.float64), atol=1e-6)
    assert torch.allclose(ln.var(-1, unbiased=False), torch.ones(4, 7, dtype=torch.float64), atol=1e-4)
    dyt = DynamicTanh(32, alpha=0.5, bias=True).double()
    assert torch.allclose(dyt(x), torch.tanh(0.5 * x))
    assert isinstance(make_norm(tiny(norm="none")), torch.nn.Identity)
    # rmsnorm is scale invariant, dynamic tanh is not
    assert torch.allclose(RMSNorm(32, gain=False)(3 * x), rms, atol=1e-6)
    assert not torch.allclose(dyt(3 * x), dyt(x))


def test_soft_cap_bounds_the_logits() -> None:
    cfg = tiny(init="gpt2", logit_softcap=15.0)
    model = build(cfg)
    with torch.no_grad():
        model.unembed.weight.mul_(100.0)
    logits = model(tokens(cfg))
    assert logits.abs().max() <= 15.0 and logits.abs().max() > 14.0
    uncapped = build(tiny(init="gpt2", logit_softcap=0.0))
    with torch.no_grad():
        uncapped.unembed.weight.mul_(100.0)
    assert uncapped(tokens(cfg)).abs().max() > 15.0


def test_tied_embeddings_share_one_matrix() -> None:
    tied, untied = GPT(tiny(tie_embeddings=True)), GPT(tiny())
    assert tied.unembed.weight is tied.embed.weight
    assert untied.unembed.weight is not untied.embed.weight
    assert tied.num_params()["unembedding"] == 0
    assert untied.num_params()["total"] - tied.num_params()["total"] == 50 * 32
    assert sum(p.numel() for p in tied.parameters()) == tied.num_params()["total"]


# ----------------------------------------------------------------------------------------------------------------
# Initialization


def test_zero_proj_init_makes_every_block_the_identity() -> None:
    cfg = tiny(vocab_size=300)
    torch.manual_seed(0)
    model = GPT(cfg).eval()
    for block in model.blocks:
        assert block.attn.o.weight.abs().max() == 0 and block.mlp.proj.weight.abs().max() == 0
        bound = 1 / math.sqrt(cfg.d_model)
        assert block.attn.q.weight.abs().max() <= bound and block.attn.q.weight.std() > 0.4 * bound
    idx = tokens(cfg)
    logits, cache = model.run_with_cache(idx)
    assert torch.equal(cache["resid_post.1"], cache["embed"])  # the model starts as embed -> unembed
    assert abs(model(idx, idx).item() - math.log(cfg.vocab_size)) < 0.01  # and predicts uniformly
    assert abs(model.embed.weight.std().item() - 1.0) < 0.05 and model.unembed.weight.std() < 0.002


def test_gpt2_init() -> None:
    cfg = tiny(init="gpt2", d_model=64, n_head=4, vocab_size=2000, bias=True, norm="layernorm", pos="learned")
    torch.manual_seed(0)
    model = GPT(cfg)
    block = model.blocks[0]
    for weight in (model.embed.weight, model.pos_embed.weight, model.unembed.weight, block.attn.q.weight, block.mlp.fc.weight):
        assert abs(weight.std().item() - 0.02) < 0.004
    for weight in (block.attn.o.weight, block.mlp.proj.weight):  # N = 4 residual layers
        assert abs(weight.std().item() - 0.02 / math.sqrt(4)) < 0.002
    assert block.attn.q.bias.abs().max() == 0 and (block.norm1.gain == 1).all()
    attention_only = GPT(tiny(init="gpt2", d_model=64, use_mlp=False))
    assert abs(attention_only.blocks[0].attn.o.weight.std().item() - 0.02 / math.sqrt(2)) < 0.003


# ----------------------------------------------------------------------------------------------------------------
# Parameter counts and groups, checkpoints


def test_param_counts_of_the_presets() -> None:
    totals = {}
    for name in preset_names():
        model = GPT(preset(name))
        counts = model.num_params()
        assert list(counts) == ["embedding", "position", "unembedding", "attention", "mlp", "norm", "total"]
        assert counts["total"] == sum(v for k, v in counts.items() if k != "total")
        assert counts["total"] == sum(p.numel() for p in model.parameters())
        totals[name] = counts["total"]
    assert totals["gptpp-base"] == 135_266_304 and totals["gpt2-2019-base"] == 111_008_256
    assert totals["gptpp-ladder"] == 58_720_256 and totals["gpt2-2019-ladder"] == 42_259_456
    pico = GPT(preset("gptpp-pico")).num_params()
    assert pico == {"embedding": 4096 * 128, "position": 0, "unembedding": 4096 * 128, "attention": 6 * 4 * 128 * 128,
                    "mlp": 6 * 2 * 4 * 128 * 128, "norm": 0, "total": 2_228_224}
    cfg = preset("gptpp-ladder")
    assert (cfg.n_layer, cfg.d_model, cfg.n_head, cfg.block_size, cfg.vocab_size) == (8, 512, 8, 512, 32768)
    cfg = preset("gptpp")
    assert cfg == preset("gptpp-base")
    assert (cfg.n_layer, cfg.d_model, cfg.n_head, cfg.block_size, cfg.vocab_size) == (12, 768, 12, 1024, 32768)


def test_the_2019_preset_has_the_parameter_count_of_gpt2_small() -> None:
    """With GPT-2's vocabulary, the 2019 recipe at the base size is GPT-2 small: 124,439,808 parameters."""
    counts = GPT(preset("gpt2-2019").replace(vocab_size=50257)).num_params()
    assert counts["total"] == 124_439_808
    assert counts["embedding"] == 50257 * 768 and counts["position"] == 1024 * 768 and counts["unembedding"] == 0


def test_param_groups_partition_the_parameters(variant: ModelConfig) -> None:
    model = GPT(variant)
    groups = model.param_groups()
    assert set(groups) == {"embedding", "unembedding", "matrices", "gains"}
    grouped = [id(p) for group in groups.values() for p in group]
    assert sorted(grouped) == sorted(id(p) for p in model.parameters())
    assert all(p.ndim == 2 for p in groups["matrices"])
    assert all(p.ndim == 1 for p in groups["gains"])
    per_block = 4 + (0 if not variant.use_mlp else 3 if variant.act in ("swiglu", "geglu") else 2)
    assert len(groups["matrices"]) == variant.n_layer * per_block
    assert (len(groups["unembedding"]) == 0) == variant.tie_embeddings
    assert len(groups["embedding"]) == (2 if variant.pos == "learned" else 1)


def test_checkpoint_round_trip(tmp_path, variant: ModelConfig) -> None:
    from gptpp.train import save_checkpoint
    model, idx = build(variant).float(), tokens(variant)
    save_checkpoint(tmp_path / "model.pt", model, step=12, tokens=3456)
    for path in (tmp_path / "model.pt", tmp_path):  # the file, or the run directory that holds it
        loaded, cfg = GPT.from_checkpoint(path, device="cpu")
        assert cfg == variant and not loaded.training
        assert torch.equal(loaded(idx), model(idx))
    checkpoint = torch.load(tmp_path / "model.pt", weights_only=True)
    assert set(checkpoint) == {"model", "model_config", "step", "tokens"} and checkpoint["step"] == 12
    assert not any(name in key for key in checkpoint["model"] for name in ("rope", "alibi", "pos_table"))
    # keys saved from a compiled, data-parallel wrapper still load
    checkpoint["model"] = {f"module._orig_mod.{key}": value for key, value in checkpoint["model"].items()}
    torch.save(checkpoint, tmp_path / "wrapped.pt")
    loaded, _ = GPT.from_checkpoint(tmp_path / "wrapped.pt", device="cpu")
    assert torch.equal(loaded(idx), model(idx))


# ----------------------------------------------------------------------------------------------------------------
# Dropout


def test_dropout_only_in_training_mode() -> None:
    cfg = tiny(init="gpt2", dropout=0.5)
    model, idx = build(cfg), tokens(cfg)
    assert torch.equal(model(idx), model(idx))
    assert torch.allclose(model(idx), model.forward_explicit(idx), atol=TOL)
    model.train()
    torch.manual_seed(0)
    a = model(idx)
    b = model(idx)
    assert not torch.allclose(a, b)
    assert not torch.allclose(model.forward_explicit(idx), model.forward_explicit(idx))


# ----------------------------------------------------------------------------------------------------------------
# Sampling


def test_truncation_rules() -> None:
    probs = torch.tensor([[0.5, 0.2, 0.15, 0.1, 0.05], [0.2, 0.2, 0.2, 0.2, 0.2]])
    assert torch.equal(truncate(probs), probs)
    top2 = truncate(probs[:1], top_k=2)
    assert torch.allclose(top2, torch.tensor([[5 / 7, 2 / 7, 0, 0, 0]]))
    # the smallest set of most likely tokens whose probability reaches p
    assert (truncate(probs[:1], top_p=0.6) > 0).sum() == 2
    assert (truncate(probs[:1], top_p=0.5) > 0).sum() == 1
    assert (truncate(probs[:1], top_p=0.86) > 0).sum() == 4
    assert (truncate(probs[:1], top_p=1e-9) > 0).sum() == 1  # the most likely token always survives
    # at least min_p times the probability of the most likely token
    assert (truncate(probs[:1], min_p=0.25) > 0).sum() == 3
    assert (truncate(probs[1:], min_p=0.99) > 0).sum() == 5
    # together: a token must pass every rule
    both = truncate(probs[:1], top_k=4, top_p=0.6, min_p=0.05)
    assert (both > 0).sum() == 2 and torch.allclose(both.sum(-1), torch.ones(1))
    shuffled = truncate(probs[:1, [3, 0, 4, 1, 2]], top_p=0.6)
    assert torch.equal(shuffled > 0, torch.tensor([[False, True, False, True, False]]))


def test_sample() -> None:
    logits = torch.tensor([[0.0, 3.0, 1.0, -2.0]]).expand(4000, 4)
    assert (sample(logits, temperature=0.0) == 1).all()
    assert (sample(logits, top_k=1) == 1).all()
    generator = torch.Generator().manual_seed(0)
    drawn = sample(logits, temperature=1.0, generator=generator).flatten()
    frequency = torch.bincount(drawn, minlength=4) / 4000
    assert torch.allclose(frequency, logits[0].softmax(-1), atol=0.03)
    cold = sample(logits, temperature=0.25, generator=generator).flatten()
    assert (cold == 1).float().mean() > 0.99


@pytest.mark.parametrize("pos", ["rope", "learned", "alibi"])
def test_generate(pos: str) -> None:
    cfg = tiny(init="gpt2", pos=pos)
    model = build(cfg)
    prompt = tokens(cfg, batch=2, length=4)
    # greedy decoding with the cache is greedy decoding by repeated full forward passes
    out = model.generate(prompt, max_new_tokens=10, temperature=0.0)
    slow = prompt
    for _ in range(10):
        slow = torch.cat([slow, model(slow)[:, -1].argmax(-1, keepdim=True)], dim=1)
    assert torch.equal(out, slow)
    # a seed fixes the sample; the prompt is kept; tokens are in the vocabulary
    a = model.generate(prompt, max_new_tokens=10, temperature=1.0, top_k=20, top_p=0.95, min_p=0.01, seed=3)
    b = generate(model, prompt, 10, temperature=1.0, top_k=20, top_p=0.95, min_p=0.01, seed=3)
    c = model.generate(prompt, max_new_tokens=10, temperature=1.0, seed=4)
    assert torch.equal(a, b) and not torch.equal(a, c)
    assert a.shape == (2, 14) and torch.equal(a[:, :4], prompt) and a.max() < cfg.vocab_size
    # past the context length the window slides: the cache is rebuilt from the most recent half
    long = model.generate(prompt, max_new_tokens=3 * cfg.block_size, temperature=0.0)
    n = cfg.block_size
    assert long.shape == (2, 4 + 3 * n)
    assert torch.equal(long[:, n], model(long[:, :n])[:, -1].argmax(-1))  # the last token decoded from a full window
    recent = long[:, n + 1 - n // 2:n + 1]  # then the cache restarts from the most recent half
    assert torch.equal(long[:, n + 1], model(recent)[:, -1].argmax(-1))
    assert not model.training


def test_generate_restores_training_mode() -> None:
    model = build(tiny(init="gpt2")).train()
    model.generate(tokens(model.cfg, batch=1, length=2), max_new_tokens=2)
    assert model.training


# ----------------------------------------------------------------------------------------------------------------
# Configuration


def test_config_json_round_trip() -> None:
    for name in preset_names():
        for cfg in (preset(name), train_preset(name)):
            assert type(cfg).from_json(cfg.to_json()) == cfg
            assert type(cfg).from_dict(json.loads(json.dumps(cfg.to_dict()))) == cfg
    cfg = ModelConfig(n_kv_head=3)
    assert ModelConfig.from_json(cfg.to_json()).n_kv_head == 3


def test_config_from_dict_is_strict() -> None:
    values = ModelConfig().to_dict()
    with pytest.raises(ValueError, match="unknown"):
        ModelConfig.from_dict({**values, "n_embd": 768})
    with pytest.raises(ValueError, match="must be int"):
        ModelConfig.from_dict({**values, "n_layer": "12"})
    with pytest.raises(ValueError, match="must be bool"):
        ModelConfig.from_dict({**values, "bias": 1})
    with pytest.raises(ValueError, match="must be float"):
        ModelConfig.from_dict({**values, "dropout": True})
    del values["act"]
    with pytest.warns(UserWarning, match="act"):
        assert ModelConfig.from_dict(values).act == ModelConfig().act
    assert ModelConfig.from_dict({**ModelConfig().to_dict(), "mlp_mult": 4}).mlp_mult == 4.0  # JSON has one number type


@pytest.mark.parametrize("changes", [
    dict(d_model=100, n_head=12), dict(n_head=12, n_kv_head=5), dict(pos="relative"), dict(norm="batchnorm"),
    dict(norm_placement="mid"), dict(act="tanh"), dict(init="xavier"), dict(dropout=1.0), dict(attn_kind="differential"),
    dict(logit_softcap=-1.0), dict(n_layer=0), dict(pos="rope", d_model=36, n_head=4), dict(qk_scale=0.0),
])
def test_invalid_model_configs_are_rejected(changes: dict) -> None:
    with pytest.raises(ValueError):
        ModelConfig().replace(**changes)


def test_overrides_are_type_checked() -> None:
    model_cfg, train_cfg = preset("gptpp-ladder"), train_preset("gptpp-ladder")
    m, t = apply_overrides([model_cfg, train_cfg], [
        "act=gelu", "n_layer=4", "qk_norm=false", "n_kv_head=2", "logit_softcap=0", "lr=1e-3", "tokens=2.5e6",
        "optimizer=adamw", "lr_embedding=none", "train.seed=7", "model.dropout=0.1", "compile=no"])
    assert (m.act, m.n_layer, m.qk_norm, m.n_kv_head, m.logit_softcap, m.dropout) == ("gelu", 4, False, 2, 0.0, 0.1)
    assert (t.lr, t.tokens, t.optimizer, t.lr_embedding, t.seed, t.compile) == (1e-3, 2.5e6, "adamw", None, 7, False)
    assert isinstance(m.logit_softcap, float) and isinstance(m.n_layer, int)
    assert apply_overrides([model_cfg, train_cfg], []) == [model_cfg, train_cfg]
    for bad in ("n_layer=4.5", "qk_norm=maybe", "lr=fast", "nonsense=1", "act", "=3", "act=tanh", "n_kv_head=3",
                "optim.lr=1", "model.lr=1", "train.act=gelu", "n_layer=none", "seed=1.0"):
        with pytest.raises(ValueError):
            apply_overrides([model_cfg, train_cfg], [bad])


def test_presets_are_the_two_recipes() -> None:
    old, new = preset("gpt2-2019-ladder"), preset("gptpp-ladder")
    assert (old.pos, old.norm, old.norm_placement, old.act, old.bias, old.tie_embeddings, old.init) == \
        ("learned", "layernorm", "pre", "gelu", True, True, "gpt2")
    assert not old.qk_norm and old.logit_softcap == 0 and not old.embed_norm
    assert (new.pos, new.norm, new.norm_placement, new.act, new.bias, new.tie_embeddings, new.init) == \
        ("rope", "rmsnorm_fixed", "pre", "relu2", False, False, "zero_proj")
    assert new.qk_norm and new.logit_softcap > 0 and new.mlp_mult == 4.0
    assert train_preset("gpt2-2019-ladder").optimizer == "adamw" and train_preset("gptpp-ladder").optimizer == "muon+adamw"
    old_train = train_preset("gpt2-2019-base")
    assert (old_train.beta1, old_train.beta2, old_train.weight_decay, old_train.schedule) == (0.9, 0.95, 0.1, "cosine")
    with pytest.raises(ValueError, match="unknown preset"):
        preset("gptpp-huge")
    assert isinstance(train_preset("gptpp-pico"), TrainConfig)


# ----------------------------------------------------------------------------------------------------------------
# bfloat16 autocast, as in training: on the CPU, and on the GPU when there is one


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
@pytest.mark.parametrize("changes", [dict(), dict(pos="alibi"), dict(pos="learned", qk_norm=False, norm="layernorm"),
                                     dict(n_kv_head=2, act="swiglu"), dict(norm="dyt", pos="sinusoidal"),
                                     dict(qk_scale_learned=True)])
def test_two_paths_agree_under_bfloat16_autocast(changes: dict, device: str) -> None:
    cfg = tiny(init="gpt2", d_model=64, block_size=64, vocab_size=500, **changes)
    model = build(cfg).float().to(device)
    idx = tokens(cfg).to(device)
    exact = model(idx)
    assert torch.allclose(exact, model.forward_explicit(idx), atol=1e-4)
    with torch.autocast(device, dtype=torch.bfloat16):
        fast, explicit = model(idx), model.forward_explicit(idx)
        loss = model(idx, idx)
    assert fast.dtype == explicit.dtype == loss.dtype == torch.float32
    assert (fast - explicit).abs().max() < 0.15 and (fast - exact).abs().max() < 0.3  # bfloat16 keeps 8 bits
    assert abs(loss.item() - F.cross_entropy(exact.view(-1, 500), idx.view(-1)).item()) < 0.05
