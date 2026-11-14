"""The attribution graph (gptpp/interp/attribution_graph.py): the frozen model is the model on the prompt, every
node's incoming edges and constant add up to its value, edges respect the order of the computation, influence is
the walk it is said to be, pruning and its scores, and what the edges predict of an ablation.

    cd gpt-pp && $PY -m pytest -q tests/test_attribution_graph.py

The identities are tested in double precision on small models with random weights and random transcoders: they hold
for any transcoder, since the error nodes carry whatever it gets wrong.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from conftest import tiny
from gptpp.interp.attribution_graph import (EMBEDDING, ERROR, LATENT, OUTPUT, Graph, ablate, build_graph, frozen_forward,
                                             influence, linear_effects, normalize, prune, record, scores, spearman, top_outputs)
from gptpp.interp.transcoder import TranscoderConfig, Transcoders
from gptpp.model import GPT

@pytest.fixture(scope="module", autouse=True)
def one_thread():
    """Small tensors on a machine of many cores: waking a thread pool costs more than the arithmetic."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


RECIPES = {
    "pico's recipe": {},
    "learned gain": dict(norm="rmsnorm"),
    "the 2019 recipe": dict(pos="learned", norm="layernorm", embed_norm=False, act="gelu", qk_norm=False, bias=True,
                            tie_embeddings=True, logit_softcap=0.0),
    "alibi, grouped keys and values": dict(pos="alibi", n_kv_head=2, qk_norm=False),
}


def setup(changes: dict, seed: int = 0, n_layer: int = 3) -> tuple[GPT, Transcoders, torch.Tensor]:
    """A small model whose every weight is random (so that patterns, denominators and MLPs all matter), random
    transcoders with every constant set, and a prompt, in double precision."""
    torch.manual_seed(seed)
    model = GPT(tiny(n_layer=n_layer, init="gpt2", **changes)).double().eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(torch.randn_like(p) * (0.3 if p.dim() > 1 else 0.1) + (1.0 if name.endswith("gain") else 0.0))
    tcs = Transcoders(TranscoderConfig(d=32, n=96, k=5, layers=n_layer), seed=seed + 1).double().eval()
    with torch.no_grad():
        tcs.enc_bias.normal_(0, 0.3), tcs.bias.normal_(0, 0.3), tcs.in_mean.normal_(0, 0.3), tcs.out_mean.normal_(0, 0.3)
        tcs.out_scale.uniform_(0.5, 2.0)
    return model, tcs, torch.randint(0, 50, (9,))


@pytest.mark.parametrize("recipe", list(RECIPES))
def test_exact_on_the_prompt_and_linear_between_nodes(recipe: str):
    model, tcs, idx = setup(RECIPES[recipe])
    graph = build_graph(model, tcs, idx)
    # the frozen model, given the recorded embeddings and MLP outputs, is the model on this prompt
    assert graph.check["frozen_rel"] < 1e-12
    # the value of every latent and output is the sum of its incoming edges plus its constant
    assert graph.check["edges_rel"] < 1e-12
    inputs = (graph.kind == LATENT) | (graph.kind == OUTPUT)
    np.testing.assert_allclose(graph.A.sum(1)[inputs] + graph.constant[inputs], graph.value[inputs], rtol=0, atol=1e-10)
    assert np.all(graph.A[~inputs] == 0), "embeddings and errors have no inputs"


def test_the_graph_holds_what_the_model_computed():
    model, tcs, idx = setup({})
    graph, rec = build_graph(model, tcs, idx), record(model, idx)
    torch.set_grad_enabled(False)
    L, T = model.cfg.n_layer, len(idx)
    assert graph.tokens == idx.tolist()
    assert (graph.kind == EMBEDDING).sum() == T and (graph.kind == ERROR).sum() == L * T
    # a latent node for every active latent, with the transcoder's own activation and pre-activation
    for l in range(L):
        pre = tcs.layer_pre(l, rec.reads[l])
        code = tcs.code(pre)
        mine = (graph.kind == LATENT) & (graph.layer == l)
        assert mine.sum() == int((code > 0).sum()) <= T * tcs.cfg.k
        np.testing.assert_allclose(graph.activation[mine], code[graph.pos[mine], graph.index[mine]].numpy(), atol=1e-12)
        np.testing.assert_allclose(graph.value[mine], pre[graph.pos[mine], graph.index[mine]].numpy(), atol=1e-12)
        # the error node is what the transcoder's output lacks of the MLP's
        error = rec.mlp_out[l] - tcs.layer_forward(l, rec.reads[l])[0]
        np.testing.assert_allclose(graph.activation[(graph.kind == ERROR) & (graph.layer == l)], error.norm(dim=-1).detach().numpy(), atol=1e-12)
    # the outputs: the most probable next tokens, their probabilities, and their logits before the cap, less the mean
    out = graph.kind == OUTPUT
    logits = model(idx[None])[0, -1].double()
    assert graph.index[out].tolist() == top_outputs(logits.softmax(-1))
    np.testing.assert_allclose(graph.activation[out], logits.softmax(-1)[graph.index[out]].numpy(), atol=1e-6)
    cap = model.cfg.logit_softcap
    raw = cap * torch.atanh((logits / cap).clamp(-1 + 1e-12, 1 - 1e-12))
    np.testing.assert_allclose(graph.value[out], (raw - raw.mean())[graph.index[out]].numpy(), atol=1e-4)
    assert np.all(graph.activation[out][:-1] >= graph.activation[out][1:]) and graph.weights().sum() == pytest.approx(graph.activation[out].sum())
    torch.set_grad_enabled(True)


def test_edges_follow_the_order_of_the_computation():
    model, tcs, idx = setup({})
    graph = build_graph(model, tcs, idx)
    t, s = np.nonzero(graph.A)
    assert np.all(graph.pos[s] <= graph.pos[t]), "nothing is read from a later position"
    later = graph.layer[s] >= graph.layer[t]
    assert not later.any(), "a node reads only what earlier blocks wrote (an embedding is block -1, an output block L)"
    assert np.all(graph.kind[s] != OUTPUT)
    # a latent of the first block at the first position reads the first embedding alone
    first = np.flatnonzero((graph.kind == LATENT) & (graph.layer == 0) & (graph.pos == 0))
    assert len(first) and np.all(np.count_nonzero(graph.A[first], axis=1) == 1) and np.all(graph.A[first, 0] != 0)


def test_top_outputs():
    p = torch.tensor([0.5, 0.3, 0.1, 0.06, 0.04])
    assert top_outputs(p, 0.95) == [0, 1, 2, 3] and top_outputs(p, 0.5) == [0] and top_outputs(p, 0.95, most=2) == [0, 1]
    assert top_outputs(torch.linspace(2, 1, 30) / 45) == list(range(10)), "at most ten"


def toy_graph(A: np.ndarray, kind: list[int], activation: list[float]) -> Graph:
    n = len(kind)
    return Graph(tokens=[0], kind=np.array(kind), layer=np.zeros(n, dtype=int), pos=np.zeros(n, dtype=int), index=np.zeros(n, dtype=int),
                 activation=np.array(activation, dtype=float), value=np.zeros(n), constant=np.zeros(n), A=np.array(A, dtype=float), check={})


def test_influence_is_the_walk_from_the_output_to_the_inputs():
    # embedding 0 and error 1 feed latent 2 (3 : 1 in size); latents 2 and the embedding feed the output 3 (1 : 1)
    A = np.zeros((4, 4))
    A[2, 0], A[2, 1], A[3, 2], A[3, 0] = 3.0, -1.0, 2.0, -2.0
    g = toy_graph(A, [EMBEDDING, ERROR, LATENT, OUTPUT], [1, 1, 1, 0.8])
    At = normalize(g.A)
    np.testing.assert_allclose(At[2], [0.75, 0.25, 0, 0])
    np.testing.assert_allclose(At[3], [0.5, 0, 0.5, 0])
    iota = influence(At, g.weights())
    # from the output: half the walks go to the embedding at once, half to the latent, of which 3/4 on to the embedding
    np.testing.assert_allclose(iota, 0.8 * np.array([0.5 + 0.5 * 0.75, 0.5 * 0.25, 0.5, 0.0]))
    s = scores(At, g.kind, g.weights())
    assert s["to_tokens"] == pytest.approx(0.875) and s["to_errors"] == pytest.approx(0.125) and s["pruned"] == pytest.approx(0)
    assert s["replacement"] == pytest.approx(0.875)
    # completeness: the latent (weight 0.4) has 3/4 of its input from a node that is no error, the output (0.8) all of it
    assert s["completeness"] == pytest.approx((0.4 * 0.75 + 0.8 * 1.0) / 1.2)
    # with the edge from the embedding to the latent pruned, the walks that took it are lost, and count as error
    edges = g.A != 0
    edges[2, 0] = False
    cut = scores(At, g.kind, g.weights(), edges)
    assert cut["to_tokens"] == pytest.approx(0.5) and cut["to_errors"] == pytest.approx(0.125) and cut["pruned"] == pytest.approx(0.375)
    assert cut["replacement"] == pytest.approx(0.5)


def test_influence_on_a_real_graph_and_pruning():
    model, tcs, idx = setup({}, seed=2)
    graph = build_graph(model, tcs, idx)
    At, w = normalize(graph.A), graph.weights()
    iota = influence(At, w)
    leaves = (graph.kind == EMBEDDING) | (graph.kind == ERROR)
    assert np.all(iota >= -1e-12) and iota[graph.kind == OUTPUT].max() < 1e-12
    assert iota[leaves].sum() == pytest.approx(w.sum()), "every walk ends at an embedding or an error"
    full = scores(At, graph.kind, w)
    assert full["to_tokens"] + full["to_errors"] == pytest.approx(1) and 0 < full["replacement"] < 1 and 0 < full["completeness"] <= 1
    everything = prune(graph, 1.0, 1.0)  # all but the latents that no walk reaches
    assert np.all(everything.nodes[iota > 1e-15]) and everything.scores["replacement"] == pytest.approx(full["replacement"])
    assert everything.scores["completeness"] == pytest.approx(full["completeness"])
    kept = [prune(graph, share, 0.98) for share in (0.5, 0.8, 0.95)]
    counts = [int((p.nodes & (graph.kind == LATENT)).sum()) for p in kept]
    assert counts[0] <= counts[1] <= counts[2] < (graph.kind == LATENT).sum()
    for p in kept:
        latent = p.nodes & (graph.kind == LATENT)
        assert np.all(p.edges.any(0)[latent]) and np.all(p.edges.any(1)[latent]), "a kept latent has an edge in and an edge out"
        assert not p.edges[~p.nodes].any() and not p.edges[:, ~p.nodes].any()
        assert p.scores["replacement"] <= full["replacement"] + 1e-12, "what is pruned counts as error"
        assert p.scores["to_tokens"] + p.scores["to_errors"] + p.scores["pruned"] == pytest.approx(1)
    assert kept[0].scores["replacement"] <= kept[2].scores["replacement"] + 1e-12


def test_linear_effects_by_hand():
    # s -> m -> out and s -> out; a_s = 1, a_m = 2
    A = np.zeros((3, 3))
    A[1, 0], A[2, 1], A[2, 0] = 2.0, 3.0, 0.5
    g = toy_graph(A, [LATENT, LATENT, OUTPUT], [1.0, 2.0, 1.0])
    E = linear_effects(g)
    np.testing.assert_allclose(E[:, 0], [0, -2.0, -(0.5 + 1.5 * 2.0)])  # one unit less of s: m falls by 2, the output by 0.5 + 1.5 * 2
    np.testing.assert_allclose(E[:, 1], [0, 0, -1.5])
    np.testing.assert_allclose(E[:, 2], 0)


def test_linear_effects_are_the_frozen_model_with_one_latent_taken_out():
    """The same prediction made the long way: take the latent's write out of the frozen model and carry the change
    forward layer by layer, every later latent moving with its pre-activation."""
    model, tcs, idx = setup({}, seed=3)
    graph, rec = build_graph(model, tcs, idx), record(model, idx)
    L = model.cfg.n_layer
    E = linear_effects(graph)
    s = int(np.flatnonzero((graph.kind == LATENT) & (graph.layer == 0))[3])
    with torch.no_grad():
        writes = [m.clone()[None] for m in rec.mlp_out]
        writes[0][0, graph.pos[s]] -= graph.activation[s] * tcs.write(0)[0][graph.index[s]]
        change = np.zeros(graph.n)
        for l in range(1, L):
            reads, _ = frozen_forward(model, rec, rec.embed[None], writes)
            mine = np.flatnonzero((graph.kind == LATENT) & (graph.layer == l))
            now = tcs.layer_pre(l, reads[l][0])[graph.pos[mine], graph.index[mine]].numpy()
            change[mine] = now - graph.value[mine]
            for node, delta in zip(mine, change[mine]):
                writes[l][0, graph.pos[node]] += delta * tcs.write(l)[0][graph.index[node]]
        _, logits = frozen_forward(model, rec, rec.embed[None], writes)
        out = np.flatnonzero(graph.kind == OUTPUT)
        centered = (logits[0, -1] - logits[0, -1].mean())[graph.index[out]].numpy()
        change[out] = centered - graph.value[out]
    np.testing.assert_allclose(graph.activation[s] * E[:, s], change, atol=1e-10)
    assert np.abs(change).max() > 1e-6


def test_ablation_in_the_model():
    model, tcs, idx = setup({}, seed=4)
    graph = build_graph(model, tcs, idx)
    latents = np.flatnonzero(graph.kind == LATENT)
    last = int(latents[-1])
    nodes = [int(latents[0]), int(latents[len(latents) // 2]), last]
    changes = ablate(model, tcs, graph, nodes, batch=2)
    assert changes.shape == (3, graph.n)
    for row, s in zip(changes, nodes):
        assert np.all(row[(graph.kind == EMBEDDING) | (graph.kind == ERROR)] == 0)
        assert np.abs(row[(graph.kind == LATENT) & (graph.layer <= graph.layer[s])]).max() < 1e-12, "nothing at or before its own block moves"
        assert np.abs(row[graph.kind == OUTPUT]).max() > 0
    # by hand, for the last one: its write taken out of the last MLP's output at its position
    l, at = int(graph.layer[last]), int(graph.pos[last])
    def take_out(x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        x[0, at] -= graph.activation[last] * tcs.write(l)[0][graph.index[last]]
        return x
    _, cache = model.run_with_cache(idx[None], ["final_norm"], {f"mlp_out.{l}": take_out})
    logits = model.unembed(cache["final_norm"][0, -1])
    out = graph.kind == OUTPUT
    np.testing.assert_allclose(changes[2][out], ((logits - logits.mean())[graph.index[out]]).detach().numpy() - graph.value[out], atol=1e-10)
    with pytest.raises(ValueError):
        ablate(model, tcs, graph, [0])


def test_what_is_not_supported_says_so():
    for changes in (dict(norm_placement="post", norm="layernorm"), dict(norm="dyt"), dict(use_mlp=False)):
        model = GPT(tiny(init="gpt2", **changes)).double().eval()
        with pytest.raises(NotImplementedError):
            record(model, torch.randint(0, 50, (5,)))


def test_spearman():
    assert spearman(np.array([1, 2, 3, 4]), np.array([10, 20, 30, 40])) == pytest.approx(1)
    assert spearman(np.array([1, 2, 3, 4]), np.array([4, 3, 2, 1])) == pytest.approx(-1)
    assert spearman(np.array([1, 2, 3, 4, 5]), np.array([2, 1, 4, 3, 5])) == pytest.approx(0.8)
    assert spearman(np.array([1, 1, 2, 2]), np.array([1, 2, 3, 4])) == pytest.approx(0.8944271909999159)  # ties share their mean rank
