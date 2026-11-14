"""Tests of gptpp/interp/circuits.py: the framework's identities hold to rounding on small attention-only models
without normalization, and the scores give the values that can be worked out by hand.

    cd gpt-pp && $PY -m pytest -q tests/test_interp_circuits.py

Everything runs on the CPU in a few seconds. The models are random (GPT-2 initialization, with the attention
matrices scaled up so that the patterns are far from uniform): the identities are algebra, not a property of
training.
"""

from __future__ import annotations

import math

import pytest
import torch

from gptpp.config import ModelConfig
from gptpp.interp import circuits as C
from gptpp.model import GPT


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    """Tiny tensors on a many-core machine: waking the thread pool costs more than the arithmetic."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def exact_model(n_layer: int, seed: int = 0, double: bool = False, **changes: object) -> GPT:
    """An attention-only model on which the algebra is exact: 4 heads of 8, 32 wide, 50 tokens, context 24.
    `double=True` runs it in float64, so that an identity can be checked far below float32's rounding."""
    base = dict(vocab_size=50, n_layer=n_layer, n_head=4, d_model=32, block_size=24, use_mlp=False, norm="none",
                embed_norm=False, qk_norm=False, logit_softcap=0.0, init="gpt2")
    torch.manual_seed(seed)
    model = GPT(ModelConfig().replace(**{**base, **changes})).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.mul_(25.0)  # std 0.5: scores of order one, so every softmax is far from uniform
    return model.double() if double else model


def sequence(model: GPT, batch: int = 3, length: int = 20, seed: int = 1) -> torch.Tensor:
    return torch.randint(0, model.cfg.vocab_size, (batch, length), generator=torch.Generator().manual_seed(seed))


# ---- the path expansion


@pytest.mark.parametrize("n_layer", [1, 2, 3])
def test_the_path_expansion_reproduces_the_logits(n_layer: int) -> None:
    model = exact_model(n_layer, double=True)
    idx = sequence(model)
    paths = C.path_expansion(model, idx)
    assert len(paths.heads) == (1 + model.cfg.n_head) ** n_layer
    assert paths.heads[0] == () and all(len(set(l for l, _ in path)) == len(path) for path in paths.heads)
    _, cache = model.run_with_cache(idx, names=[f"resid_post.{n_layer - 1}"])
    stream = cache[f"resid_post.{n_layer - 1}"]
    assert (paths.stream.sum(dim=0) - stream).abs().max().item() < 1e-10 * stream.abs().max().item()
    logits = model(idx).double()  # the model returns its logits in float32
    scale = logits.abs().max().item()
    assert scale > 1.0, "the test model must have logits of order one for the comparison to mean anything"
    assert (paths.logits().sum(dim=0) - logits).abs().max().item() < 1e-6 * scale


def test_the_path_expansion_holds_in_float32_to_rounding() -> None:
    model = exact_model(2)
    idx = sequence(model)
    logits = model(idx).double()
    error = (C.path_expansion(model, idx).logits().sum(dim=0) - logits).abs().max().item()
    assert error < 1e-4 * logits.abs().max().item()


def test_the_paths_are_the_direct_path_the_heads_and_the_virtual_heads() -> None:
    model = exact_model(2)
    idx = sequence(model, batch=1)
    patterns = C.attention_patterns(model, idx)
    paths = C.path_expansion(model, idx, patterns)
    assert [paths.order(p) for p in range(len(paths.heads))].count(1) == 8
    assert [paths.order(p) for p in range(len(paths.heads))].count(2) == 16
    # The direct path is the bigram table read at the tokens.
    direct = paths.logits()[0, 0]  # (T, V)
    assert torch.allclose(direct, C.direct_path(model)[:, idx[0]].T, atol=1e-9)
    # A first-layer head's term is its pattern times its OV circuit's columns at the source tokens.
    p = paths.heads.index(((0, 2),))
    table = C.ov_circuit(model, 0, 2)  # (V, V)
    expected = patterns[0][0, 2].double() @ table[:, idx[0]].T
    assert torch.allclose(paths.logits()[p, 0], expected, atol=1e-8)
    # A virtual head: the product of two patterns and of two OV matrices.
    p = paths.heads.index(((0, 1), (1, 3)))
    product = C.unembedding(model) @ C.ov_matrix(model, 1, 3) @ C.ov_matrix(model, 0, 1) @ C.embedding(model).T
    expected = (patterns[1][0, 3].double() @ patterns[0][0, 1].double()) @ product[:, idx[0]].T
    assert torch.allclose(paths.logits()[p, 0], expected, atol=1e-7)
    # by_order adds up to the same stream.
    assert torch.allclose(paths.by_order().sum(dim=0), paths.stream.sum(dim=0), atol=1e-9)


def test_the_path_expansion_refuses_a_model_it_is_not_exact_for() -> None:
    for changes in (dict(norm="rmsnorm_fixed"), dict(use_mlp=True), dict(logit_softcap=15.0), dict(bias=True),
                    dict(qk_norm=True), dict(pos="learned")):
        model = exact_model(1, **changes)
        assert not C.is_exact(model)
        with pytest.raises(ValueError):
            C.path_expansion(model, sequence(model))
    assert C.is_exact(exact_model(2)) and C.is_exact(exact_model(1, pos="none"))


def test_the_orders_add_up_to_the_paths_and_the_last_loss_is_the_models() -> None:
    model = exact_model(2, double=True)
    idx = sequence(model)
    paths = C.path_expansion(model, idx)
    orders = C.stream_by_order(model, idx)
    assert orders.shape[0] == 3 and torch.allclose(orders, paths.by_order(), atol=1e-9)
    three = exact_model(3, double=True)
    assert torch.allclose(C.stream_by_order(three, sequence(three)), C.path_expansion(three, sequence(three)).by_order(), atol=1e-8)
    losses = C.loss_by_order(model, idx)
    assert len(losses) == 3
    assert losses[-1] == pytest.approx(model(idx[:, :-1], idx[:, 1:]).item(), rel=1e-5)
    direct = torch.log_softmax(C.direct_path(model)[:, idx[:, :-1]].permute(1, 2, 0), dim=-1)
    assert losses[0] == pytest.approx(-direct.gather(2, idx[:, 1:, None]).mean().item(), rel=1e-9)


def test_patching_the_real_stream_changes_nothing_and_patching_another_inputs_stream_replaces_the_heads() -> None:
    model = exact_model(2, double=True)
    real, other = sequence(model, seed=1), sequence(model, seed=2)
    clean = model(real)
    _, cache = model.run_with_cache(real, names=["resid_post.0"])
    same = C.patch_inputs(model, real, 1, [0, 2], query=cache["resid_post.0"], key=cache["resid_post.0"], value=cache["resid_post.0"])
    assert torch.allclose(same, clean, atol=1e-5)
    assert torch.equal(C.patch_inputs(model, real, 1, [0, 2]), clean)
    # All three inputs of heads 0 and 2 from the other input: their outputs are those of the other run (the
    # scores depend on position only through offsets), and the rest of the stream is the real one.
    _, alt = model.run_with_cache(other, names=["resid_post.0", "attn.1.result"])
    _, ref = model.run_with_cache(real, names=["resid_post.0", "attn.1.result"])
    patched = C.patch_inputs(model, real, 1, [0, 2], query=alt["resid_post.0"], key=alt["resid_post.0"], value=alt["resid_post.0"])
    stream = ref["resid_post.0"] + ref["attn.1.result"][:, [1, 3]].sum(dim=1) + alt["attn.1.result"][:, [0, 2]].sum(dim=1)
    expected = stream @ C.unembedding(model).T
    assert (patched.double() - expected).abs().max().item() < 1e-5 * expected.abs().max().item()
    # Only the value patched: the pattern stays, the output changes.
    _, v_only = model.run_with_cache(real, names=["attn.1.pattern"])
    hooked = C.patch_inputs(model, real, 1, [1], value=alt["resid_post.0"])
    assert not torch.allclose(hooked, clean, atol=1e-3)


def test_the_share_explained_is_one_for_the_whole_stream_and_zero_for_none_of_it() -> None:
    model = exact_model(2)
    pool = torch.arange(5, 45)
    real = C.repeated_random_tokens(6, 5, 2, pool, bos_id=49, seed=0)
    other = C.repeated_random_tokens(6, 5, 2, pool, bos_id=49, seed=1)
    out = C.induction_paths_explained(model, real, other, induction=[0, 1], previous=2, length=5, repeats=2)
    assert set(out) >= {"value", "query", "key", "all", "loss.clean", "loss.none"}
    assert out["loss.clean"] == pytest.approx(C.repeated_loss(C.loss_by_position(model, real), 5, 2)["repeated"], rel=1e-5)
    # With the other input equal to the real one every path carries the real stream: nothing is lost.
    identical = C.induction_paths_explained(model, real, real, induction=[0, 1], previous=2, length=5, repeats=2)
    assert identical["loss.none"] == pytest.approx(identical["loss.clean"], rel=1e-5)
    assert C.baseline(256) == 0.0625


# ---- zero layers


def test_the_zero_layer_model_fits_the_bigram_table_as_far_as_its_rank_allows() -> None:
    generator = torch.Generator().manual_seed(0)
    vocab = 12
    counts = (torch.randperm(vocab * vocab, generator=generator).view(vocab, vocab) + 1).double()  # no ties
    counts[3] = 0  # a token that never occurs is left out of every average
    wide_e, wide_u, losses = C.fit_zero_layer(counts, d_model=12, steps=1500, lr=0.05)
    assert losses[-1] < losses[0]
    wide = C.bigram_agreement(wide_u @ wide_e.T, counts)
    assert wide["loss"] == pytest.approx(losses[-1], abs=1e-3)
    assert wide["kl"] < 2e-3 and wide["top1"] == 1.0  # full rank: the table of log conditionals itself
    narrow_e, narrow_u, _ = C.fit_zero_layer(counts, d_model=1, steps=1500, lr=0.05)
    narrow = C.bigram_agreement(narrow_u @ narrow_e.T, counts)
    assert narrow["entropy"] == pytest.approx(wide["entropy"]) and narrow["kl"] > 5 * wide["kl"]
    # By hand: a table that is exactly log p has no divergence; one source alone is weighted as one.
    exact = torch.log(counts.clamp_min(1e-9) / counts.sum(dim=1, keepdim=True).clamp_min(1)).T
    assert abs(C.bigram_agreement(exact, counts)["kl"]) < 1e-9
    two = torch.tensor([[3.0, 1.0], [1.0, 1.0]])
    uniform = C.bigram_agreement(torch.zeros(2, 2), two, sources=[0])
    assert uniform["loss"] == pytest.approx(math.log(2)) and uniform["entropy"] == pytest.approx(0.5623351446)


# ---- the QK and OV circuits


@pytest.mark.parametrize("pos", ["rope", "none"])
def test_the_qk_circuit_is_the_first_layers_score(pos: str) -> None:
    model = exact_model(1, pos=pos)
    idx = sequence(model, batch=1, length=12)
    _, cache = model.run_with_cache(idx, names=["attn.0.scores"])
    scores = cache["attn.0.scores"][0].double()  # (H, T, T)
    for head in range(model.cfg.n_head):
        for offset in (0, 1, 2, 5, 9):
            table = C.qk_circuit(model, 0, head, offset=offset)
            for i in range(offset, idx.size(1)):
                expected = table[idx[0, i], idx[0, i - offset]].item()
                assert abs(scores[head, i, i - offset].item() - expected) < 1e-4 * max(1.0, abs(expected))
    centered = C.qk_circuit(model, 0, 0, offset=1, center=True)
    assert centered.mean(dim=1).abs().max().item() < 1e-9
    # The same number through the d x d matrix.
    e = C.embedding(model)
    assert torch.allclose(e @ C.qk_matrix(model, 0, 1, offset=3) @ e.T, C.qk_circuit(model, 0, 1, offset=3), atol=1e-9)


def test_rotary_scores_depend_on_the_offset_and_the_turn_is_a_rotation() -> None:
    model = exact_model(1)
    r = C.relative_rotation(model, 3)
    assert torch.allclose(r @ r.T, torch.eye(model.cfg.d_head, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(C.relative_rotation(model, 2) @ C.relative_rotation(model, 5), C.relative_rotation(model, 7),
                          atol=1e-12)
    assert not torch.allclose(C.qk_circuit(model, 0, 0, offset=1), C.qk_circuit(model, 0, 0, offset=2))
    assert torch.equal(C.relative_rotation(exact_model(1, pos="none"), 4), torch.eye(8, dtype=torch.float64))


def test_the_ov_circuit_selects_tokens_and_matches_the_heads_output() -> None:
    model = exact_model(1)
    full = C.ov_circuit(model, 0, 1)
    assert full.shape == (50, 50)
    part = C.ov_circuit(model, 0, 1, sources=[3, 7, 9], outputs=[0, 49])
    assert torch.allclose(part, full[[0, 49]][:, [3, 7, 9]], atol=1e-12)
    # attn.0.result through the unembedding is the pattern times the OV columns.
    idx = sequence(model, batch=1)
    _, cache = model.run_with_cache(idx, names=["attn.0.result", "attn.0.pattern"])
    through = cache["attn.0.result"][0, 1].double() @ C.unembedding(model).T  # (T, V)
    expected = cache["attn.0.pattern"][0, 1].double() @ full[:, idx[0]].T
    assert torch.allclose(through, expected, atol=1e-4 * expected.abs().max().item())


# ---- copying


def test_the_eigenvalues_are_those_of_the_whole_table() -> None:
    model = exact_model(1)
    small = C.ov_eigenvalues(model, 0, 0)
    assert small.shape == (model.cfg.d_head,)
    big = torch.linalg.eigvals(C.ov_circuit(model, 0, 0))
    big = big[big.abs().argsort(descending=True)][: model.cfg.d_head]
    small = small[small.abs().argsort(descending=True)]
    assert torch.allclose(big.abs(), small.abs(), rtol=1e-6, atol=1e-8)
    assert abs(big.sum().real.item() - small.sum().real.item()) < 1e-6 * small.abs().sum().item()
    # The sum of the eigenvalues is the trace: the total by which tokens raise their own logits.
    assert abs(small.sum().real.item() - C.ov_circuit(model, 0, 0).diagonal().sum().item()) < 1e-8 * small.abs().sum().item()


def test_the_copying_statistic_by_hand() -> None:
    assert C.copying_statistic(torch.tensor([1.0, 2.0, 3.0], dtype=torch.complex128)) == 1.0
    assert C.copying_statistic(torch.tensor([-1.0, -2.0], dtype=torch.complex128)) == -1.0
    assert C.copying_statistic(torch.tensor([3.0, -1.0], dtype=torch.complex128)) == 0.5
    assert abs(C.copying_statistic(torch.tensor([1j, -1j, 2.0]))) - 0.5 < 1e-12  # a conjugate pair adds nothing
    assert C.copying_statistic(torch.zeros(4, dtype=torch.complex128)) == 0.0  # an output matrix at zero


def copier(sign: float = 1.0) -> GPT:
    """A one-layer model whose head 0 copies: U = E, and W_O W_V is `sign` times a projection."""
    model = exact_model(1)
    with torch.no_grad():
        model.unembed.weight.copy_(model.embed.weight)
        attn = model.blocks[0].attn
        attn.v.weight[:8].zero_()
        attn.v.weight[:8, :8] = torch.eye(8)
        attn.o.weight[:, :8].zero_()
        attn.o.weight[:8, :8] = sign * torch.eye(8)
    return model


def test_a_head_built_to_copy_scores_as_one_and_a_random_head_does_not() -> None:
    tokens = list(range(50))
    model = copier()
    assert C.copying_statistic(C.ov_eigenvalues(model, 0, 0)) == pytest.approx(1.0, abs=1e-9)
    assert C.copying_score(model, 0, 0, tokens) > 5 / 50  # well over the 1 / 50 of chance
    assert C.self_rank(model, 0, 0, tokens).float().median().item() <= 2
    against = copier(sign=-1.0)
    assert C.copying_statistic(C.ov_eigenvalues(against, 0, 0)) == pytest.approx(-1.0, abs=1e-9)
    assert C.copying_score(against, 0, 0, tokens) < 1 / 50
    random = [C.copying_statistic(C.ov_eigenvalues(exact_model(1, seed=s), 0, h)) for s in range(6) for h in range(4)]
    assert abs(sum(random) / len(random)) < 0.15 and max(abs(r) for r in random) < 0.7
    assert abs(C.copying_score(exact_model(1, seed=3), 0, 2, tokens) - 1 / 50) < 0.03


# ---- composition


def test_composition_scores_are_bounded_and_do_not_depend_on_the_basis_of_the_stream() -> None:
    model = exact_model(2)
    scores = C.composition(model, (0, 1), (1, 2))
    assert set(scores) == {"q", "k", "v"} and all(0 <= value <= 1 for value in scores.values())
    # Rotate the residual stream: every writer W -> R W, every reader W -> W R^T. The function is unchanged, and
    # so is every score.
    rotation = torch.linalg.qr(torch.randn(32, 32, generator=torch.Generator().manual_seed(5)))[0]
    turned = exact_model(2)
    with torch.no_grad():
        turned.embed.weight.copy_(model.embed.weight @ rotation.T)
        turned.unembed.weight.copy_(model.unembed.weight @ rotation.T)
        for a, b in zip(model.blocks, turned.blocks):
            for name in ("q", "k", "v"):
                getattr(b.attn, name).weight.copy_(getattr(a.attn, name).weight @ rotation.T)
            b.attn.o.weight.copy_(rotation @ a.attn.o.weight)
    idx = sequence(model)
    assert torch.allclose(model(idx), turned(idx), atol=2e-3)
    for name, value in C.composition(turned, (0, 1), (1, 2)).items():
        assert value == pytest.approx(scores[name], rel=1e-5)
    table = C.composition_table(model, 0, 1)
    assert table["k"].shape == (4, 4) and table["k"][1, 2].item() == pytest.approx(scores["k"])


def test_the_baseline_is_one_over_the_square_root_of_the_width() -> None:
    for d_model, d_head in ((64, 8), (128, 32), (256, 32)):
        baseline = C.composition_baseline(d_model, d_head, draws=60)
        assert baseline == pytest.approx(1 / math.sqrt(d_model), rel=0.04)
    # The permuted quantity of the 2021 figure sits elsewhere: about 1 / sqrt(d_head).
    assert C.composition_baseline(128, 32, draws=60, cyclic=True) == pytest.approx(1 / math.sqrt(32), rel=0.08)


def test_a_key_built_to_read_a_heads_output_composes_with_it_and_the_permuted_norm_differs() -> None:
    model = exact_model(2)
    with torch.no_grad():
        first, second = model.blocks[0].attn, model.blocks[1].attn
        # Head (1, 0)'s key reads exactly the 8 directions head (0, 0) writes; its query reads 8 others.
        first.o.weight[:, :8].zero_()
        first.o.weight[8:16, :8] = torch.eye(8)
        second.k.weight[:8].zero_()
        second.k.weight[:8, 8:16] = torch.eye(8)
        second.q.weight[:8].zero_()
        second.q.weight[:8, 16:24] = torch.eye(8)
    built = C.composition(model, (0, 0), (1, 0))
    assert built["k"] == pytest.approx(1 / math.sqrt(8), rel=1e-6)  # |W_K W_O| / (|W_K| |W_O|) for two 8-dimensional identities
    assert built["k"] > 1.9 * C.composition_baseline(32, 8, draws=50)  # chance is 1 / sqrt(32)
    assert built["k"] > 1.9 * C.composition(model, (0, 1), (1, 0))["k"]
    assert built["q"] < 1e-9  # the query reads none of it
    # The 2021 figure's quantity, every norm taken on the cyclically permuted product, is another number: here
    # the key and the query read orthogonal directions, W_K W_Q^T is zero, and the permuted score is 0 by the
    # convention for a zero denominator although the key reads all that the first head writes.
    assert C.composition(model, (0, 0), (1, 0), cyclic=True)["k"] == 0.0
    # On generic heads the two differ as well: a Frobenius norm is not invariant under cyclic permutation.
    generic = exact_model(2, seed=2)
    assert abs(C.composition(generic, (0, 0), (1, 0), cyclic=True)["k"] - C.composition(generic, (0, 0), (1, 0))["k"]) > 0.02
    # FW's eigenvalue test of the same edge is defined and has d_head values.
    assert C.k_composition_eigenvalues(model, (0, 0), (1, 0)).shape == (8,)


def test_a_head_that_writes_nothing_composes_with_nothing() -> None:
    model = exact_model(2, init="zero_proj")
    assert C.composition(model, (0, 0), (1, 0)) == {"q": 0.0, "k": 0.0, "v": 0.0}
    assert C.copying_statistic(C.ov_eigenvalues(model, 0, 0)) == 0.0


# ---- scores on activations


def test_repeated_random_tokens() -> None:
    pool = C.token_pool(torch.arange(100, 0, -1) * 50, drop_most=10, min_count=1000)
    assert pool.tolist() == list(range(10, 81))  # counts 5000, 4950, ...: the first ten dropped, then down to 1000
    seq = C.repeated_random_tokens(5, 6, 3, pool, bos_id=99, seed=4)
    assert seq.shape == (5, 19) and (seq[:, 0] == 99).all()
    assert torch.equal(seq[:, 1:7], seq[:, 7:13]) and torch.equal(seq[:, 1:7], seq[:, 13:19])
    assert all(len(set(row.tolist())) == 6 for row in seq[:, 1:7]), "no token twice in a block"
    assert torch.equal(seq, C.repeated_random_tokens(5, 6, 3, pool, bos_id=99, seed=4))
    assert not torch.equal(seq, C.repeated_random_tokens(5, 6, 3, pool, bos_id=99, seed=5))


def test_prefix_matching_and_previous_token_scores_by_hand() -> None:
    length, repeats = 4, 3
    T = 1 + length * repeats
    induction = torch.zeros(1, 1, T, T)
    previous = torch.zeros(1, 1, T, T)
    uniform = torch.zeros(1, 1, T, T)
    for i in range(T):
        uniform[0, 0, i, : i + 1] = 1 / (i + 1)
        previous[0, 0, i, max(i - 1, 0)] = 1
        if i >= 1 + length:
            induction[0, 0, i, i - length + 1] = 1  # the token after the latest earlier occurrence
        else:
            induction[0, 0, i, 0] = 1
    assert C.prefix_matching_score(induction, length, repeats).item() == pytest.approx(1.0)
    assert C.prefix_matching_score(previous, length, repeats).item() == 0.0
    expected = sum(len([k for k in range(1, repeats) if i - k * length >= 1]) / (i + 1) for i in range(1 + length, T)) / (T - 1 - length)
    assert C.prefix_matching_score(uniform, length, repeats).item() == pytest.approx(expected)
    assert C.previous_token_score(previous).item() == pytest.approx(1.0)
    assert C.previous_token_score(induction).item() < 0.2
    # Two heads, two sequences: one score a head.
    both = torch.cat((induction, previous), dim=1).repeat(2, 1, 1, 1)
    assert C.prefix_matching_score(both, length, repeats).tolist() == pytest.approx([1.0, 0.0])


def test_losses_by_position_and_the_scores_built_on_them() -> None:
    model = exact_model(1)
    idx = sequence(model, batch=4, length=20)
    losses = C.loss_by_position(model, idx)
    assert losses.shape == (4, 19)
    assert losses.mean().item() == pytest.approx(model(idx[:, :-1], idx[:, 1:]).item(), rel=1e-5)
    table = 20.0 - torch.arange(19.0).repeat(4, 1)  # the loss on the n-th token is 22 - n: later tokens are easier
    assert C.in_context_learning_score(table, early=5, late=18) == pytest.approx(13.0)
    assert C.in_context_learning_score(-table, early=5, late=18) == pytest.approx(-13.0)
    seq_losses = torch.zeros(2, 12)  # 1 + 4 * 3 tokens
    seq_losses[:, 1:4] = 8.0  # targets 2..4 of the first block
    seq_losses[:, 4] = 5.0  # the first token of the second block: nothing predicts it
    seq_losses[:, 5:12] = 0.5
    assert C.repeated_loss(seq_losses, 4, 3) == {"first": 8.0, "repeated": 0.5}


def test_repeat_statistics_and_the_rise_interval_by_hand() -> None:
    #          A  B  C  A  B  A  C      positions 3, 4, 5 hold a token seen before; after 3 (A) comes B as before,
    windows = torch.tensor([[1, 2, 3, 1, 2, 1, 3]])  # after 4 (B) comes A, not C; after 5 (A) comes C, not B
    stats = C.repeat_statistics(windows)
    assert stats == {"seen": 3 / 6, "continues": 1 / 3}
    assert C.repeat_statistics(windows, exclude=[1]) == {"seen": 1 / 6, "continues": 0.0}
    tokens = [0, 1e6, 2e6, 4e6, 8e6, 16e6]
    rise = C.rise_interval(tokens, [0.03, 0.03, 0.05, 0.50, 0.90, 0.93])
    assert rise["start"] == 4e6 and rise["end"] == 8e6 and rise["factor"] == 2.0  # 10 %: 0.12; 90 %: 0.84
    assert rise["from"] == 0.03 and rise["to"] == 0.93
    fall = C.rise_interval(tokens, [9.0, 9.0, 8.9, 5.0, 0.5, 0.1])  # a loss that falls is read the same way
    assert fall["start"] == 4e6 and fall["end"] == 8e6


def test_a_mixed_shard_keeps_the_text_and_replaces_whole_windows(tmp_path) -> None:
    import numpy as np

    from gptpp.interp import repeated_token_windows as R

    rng = np.random.default_rng(0)
    pool = np.arange(100, 200, dtype=np.uint16)
    window = R.repeated_window(pool, bos_id=7, window=32, block=5, rng=rng)
    assert window.shape == (32,) and window[0] == 7 and len(set(window[1:6].tolist())) == 5
    assert np.array_equal(window[1:6], window[6:11]) and np.array_equal(window[1:27], window[6:32])
    text = np.arange(3, 3 + 32 * 40 + 9, dtype=np.uint16) % 90  # tokens below 100: never from the pool
    mixed, n_windows, n_replaced = R.mix_shard(text, pool, 7, fraction=0.25, window=32, min_block=4, max_block=12, rng=rng)
    assert n_windows == 40 and 3 <= n_replaced <= 20 and mixed.shape == text.shape
    replaced = [k for k in range(n_windows) if mixed[k * 32] == 7 and mixed[k * 32 + 1] >= 100]
    assert len(replaced) == n_replaced
    for k in range(n_windows):
        if k in replaced:
            assert (mixed[k * 32 + 1:(k + 1) * 32] >= 100).all()
        else:
            assert np.array_equal(mixed[k * 32:(k + 1) * 32], text[k * 32:(k + 1) * 32])
    assert np.array_equal(mixed[40 * 32:], text[40 * 32:]), "what follows the last whole window is untouched"


# ---- ablations


def test_a_pattern_preserving_ablation_removes_exactly_the_paths_through_the_head() -> None:
    model = exact_model(2)
    idx = sequence(model, batch=2)
    patterns = C.attention_patterns(model, idx)
    paths = C.path_expansion(model, idx, patterns)
    for head in ((0, 1), (1, 3)):
        hooks = C.ablation_hooks([head], "zero", patterns=patterns)
        ablated = model.run_with_hooks(idx, hooks).double()
        kept = [p for p, path in enumerate(paths.heads) if head not in path]
        assert len(kept) < len(paths.heads)
        expected = paths.logits()[kept].sum(dim=0)
        assert (ablated - expected).abs().max().item() < 2e-5 * expected.abs().max().item()


def test_a_full_ablation_changes_later_patterns_and_a_mean_ablation_uses_the_mean() -> None:
    model = exact_model(2)
    idx = sequence(model, batch=6)
    clean = C.attention_patterns(model, idx)
    zeroed = C.attention_patterns(model, idx, C.ablation_hooks([(0, 0)], "zero"))
    assert torch.equal(clean[0], zeroed[0]), "the ablated head's own layer keeps its patterns"
    assert (clean[1] - zeroed[1]).abs().max().item() > 1e-3, "the next layer recomputes its attention"
    means = C.mean_head_outputs(model, idx)
    assert means[0].shape == (4, 20, 8)
    _, cache = model.run_with_cache(idx, names=["attn.0.z"], hooks=C.ablation_hooks([(0, 2)], "mean", means))
    assert torch.allclose(cache["attn.0.z"][:, 2], means[0][2].expand(6, -1, -1))
    _, untouched = model.run_with_cache(idx, names=["attn.0.z"])
    assert torch.equal(cache["attn.0.z"][:, 1], untouched["attn.0.z"][:, 1])
    with pytest.raises(ValueError):
        C.ablation_hooks([(0, 0)], "mean")
    with pytest.raises(ValueError):
        C.ablation_hooks([(0, 0)], "median")


def test_head_scores_has_every_score_for_every_head() -> None:
    model = exact_model(2)
    pool = torch.arange(5, 45)
    scores = C.head_scores(model, sequence(model, batch=4), pool, bos_id=49, length=5, repeats=2, n_sequences=8,
                           copy_sample=10, copy_draws=4)
    for name in ("prefix_matching", "previous_token", "copying_statistic", "copying_score"):
        assert len(scores[name]) == 2 and all(len(layer) == 4 for layer in scores[name])
    assert scores["repeated_loss"] > 0 and scores["first_loss"] > 0
