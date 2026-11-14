"""Attribution graphs: one prompt's forward pass rewritten over transcoder latents, and what each node owes the others.

A transcoder (transcoder.py) stands in for an MLP only approximately, and a model with all of its MLPs replaced
drifts from the model. For one prompt the stand-in can be made exact (Ameisen et al., "Circuit Tracing: Revealing
Computational Graphs in Language Models", 2025). The local replacement model of a prompt is the model with

    every MLP output      written as its transcoder's output on this prompt (the rows of the active latents times
                          their activations, plus the transcoder's constant) plus an error vector, the recorded
                          difference from the true output
    every attention       pattern held at the values recorded on this prompt
    every normalization   dividing by the denominator recorded on this prompt

Its stream and its logits equal the model's on the prompt, and with those three things constant the stream at every
layer and position is a linear (affine) function of the token embeddings, the latents' activations and the error
vectors. So the pre-activation h_t of a latent t, and the logit of an output token less the mean logit, is a sum of
edges from the nodes upstream, plus a constant:

    h_t = sum over s of A[t, s] + constant_t,        A[t, s] = a_s  u_s^T  J  e_t

a_s u_s is what node s writes into the stream (a latent's activation times its row of the dictionary; an embedding
or an error vector itself), e_t the direction node t reads (a latent's encoder row; for an output token the
gradient of its logit less the mean logit, in the final stream), and J the Jacobian of the stream where t reads
with respect to the stream where s writes, the patterns, denominators and MLP outputs being constants: an edge is a
decoder, a frozen Jacobian and an encoder. The constant holds what is no node: the latent's own offset and the
constants of the transcoders upstream (and, in a model that has them, its biases).

    nodes        an embedding a position; a latent for every active latent at every layer and position; an error a
                 layer and position; an output for each of the most probable next tokens (as many as reach 95% of
                 the probability, at most ten)
    the output   is the logit BEFORE the soft cap: cap * tanh(. / cap) is one more nonlinearity, and it is neither a
                 pattern nor a denominator. The logit before it is linear in the final stream, so the identity above
                 is exact for it; the cap is increasing, so the order of the tokens is the same; and an
                 intervention is measured on the same quantity. The weights of the outputs are the model's own
                 probabilities, after the cap

What is exact, and tested: `Graph.check` holds the largest difference between the frozen model and the model on
the prompt, and the largest difference between a node's incoming edges plus its constant and its value.

Influence (the paper's appendix on pruning): with A~[t, s] = |A[t, s]| / sum over s' of |A[t, s']| and w the
outputs' probabilities, iota = w (A~ + A~^2 + ...). A walk that starts at an output drawn by w and steps from a node
to one of its inputs, to s with probability A~[t, s], passes through s with probability iota_s; it can stop only at
an embedding or an error. Pruning keeps the nodes that carry a share of the influence (0.8), then the edges that
carry a share of what is left (0.98). The replacement score is the share of walks that end at embeddings, not at
errors; completeness is the share of every node's input that does not come from errors, weighted by the node's
influence. On a pruned graph what was pruned counts as error.

A graph is a statement about the local replacement model. `ablate` tests it in the model: the write of one latent
is taken out of its MLP's output at its position, the model is run on with its own attention and its own MLPs, and
the change of the outputs and of the later latents is compared with what the edges predict (`linear_effects`).

    graph = build_graph(model, tcs, torch.tensor(tokenizer.encode(prompt, bos=True)))
    pruned = prune(graph, 0.8, 0.98)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from gptpp.interp.transcoder import Transcoders
from gptpp.model import GPT, LayerNorm, RMSNorm

EMBEDDING, LATENT, ERROR, OUTPUT = 0, 1, 2, 3
KINDS = ("embedding", "latent", "error", "output")


# ----------------------------------------------------------------------------------------------------------------
# The forward pass of one prompt, and the model with its patterns and denominators held


def denominator(norm: nn.Module, x: Tensor) -> Tensor:
    """What a normalization divides x by, (..., 1): the one thing in it that is not linear in x."""
    if isinstance(norm, RMSNorm):
        return (x.pow(2).mean(-1, keepdim=True) + 1e-6).sqrt()
    if isinstance(norm, LayerNorm):
        return (x.var(-1, unbiased=False, keepdim=True) + 1e-5).sqrt()
    if isinstance(norm, nn.Identity):
        return torch.ones_like(x[..., :1])
    raise NotImplementedError(f"{type(norm).__name__} has no denominator to hold constant")


def frozen_norm(norm: nn.Module, x: Tensor, denom: Tensor) -> Tensor:
    """The normalization with its denominator given: a linear map of x (plus the bias of a LayerNorm)."""
    if isinstance(norm, RMSNorm):
        y = x / denom
        return y if norm.gain is None else y * norm.gain
    if isinstance(norm, LayerNorm):
        y = (x - x.mean(-1, keepdim=True)) / denom * norm.gain
        return y if norm.bias is None else y + norm.bias
    return x


@dataclass
class Record:
    """A model's forward pass on one prompt: what the frozen model holds constant, and what it must reproduce."""

    idx: Tensor  # (T,) the prompt's tokens
    embed: Tensor  # (T, d) what enters the first block
    patterns: list[Tensor]  # per block (H, T, T)
    denom1: list[Tensor]  # per block (T, 1): the denominator of the norm before attention
    denom2: list[Tensor]  # per block (T, 1): of the norm before the MLP
    denom_final: Tensor  # (T, 1)
    reads: list[Tensor]  # per block (T, d): what the MLP reads
    mlp_out: list[Tensor]  # per block (T, d): what the MLP writes
    logits: Tensor  # (V,) at the last position, before the soft cap
    probs: Tensor  # (V,) the model's next-token probabilities (after the cap)


@torch.no_grad()
def record(model: GPT, idx: Tensor) -> Record:
    cfg = model.cfg
    if cfg.norm_placement != "pre" or not cfg.use_mlp or cfg.norm == "dyt":
        raise NotImplementedError("attribution graphs are built for pre-norm models with MLPs and a norm that divides")
    L = cfg.n_layer
    names = ["embed", "resid_pre.*", "resid_mid.*", "attn.*.pattern", "mlp_out.*", f"resid_post.{L - 1}", "final_norm", "logits"]
    logits, cache = model.run_with_cache(idx[None], names)
    blocks = model.blocks
    return Record(
        idx=idx, embed=cache["embed"][0],
        patterns=[cache[f"attn.{l}.pattern"][0] for l in range(L)],
        denom1=[denominator(blocks[l].norm1, cache[f"resid_pre.{l}"][0]) for l in range(L)],
        denom2=[denominator(blocks[l].norm2, cache[f"resid_mid.{l}"][0]) for l in range(L)],
        denom_final=denominator(model.final_norm, cache[f"resid_post.{L - 1}"][0]),
        reads=[blocks[l].norm2(cache[f"resid_mid.{l}"][0]) for l in range(L)],
        mlp_out=[cache[f"mlp_out.{l}"][0] for l in range(L)],
        logits=model.unembed(cache["final_norm"][0, -1]),
        probs=logits[0, -1].double().softmax(-1),
    )


def frozen_forward(model: GPT, rec: Record, embed: Tensor, writes: list[Tensor]) -> tuple[list[Tensor], Tensor]:
    """The model with the patterns and denominators of `rec`, given what enters the first block (B, T, d) and what
    each MLP writes (B, T, d): what each MLP reads (B, T, d) and the logits before the cap (B, T, V). Every
    operation is affine in `embed` and `writes`; with the recorded embeddings and MLP outputs it is the model."""
    x, reads = embed, []
    B, T, _ = x.shape
    for l, block in enumerate(model.blocks):
        attn = block.attn
        v = attn.v(frozen_norm(block.norm1, x, rec.denom1[l])).view(B, T, attn.n_kv_head, attn.d_head).transpose(1, 2)
        if attn.n_kv_head != attn.n_head:
            v = v.repeat_interleave(attn.n_head // attn.n_kv_head, dim=1)
        z = rec.patterns[l] @ v  # (H, T, T) @ (B, H, T, dh): each head's recorded average of the values
        x = x + attn.o(z.transpose(1, 2).reshape(B, T, -1))
        reads.append(frozen_norm(block.norm2, x, rec.denom2[l]))
        x = x + writes[l]
    return reads, model.unembed(frozen_norm(model.final_norm, x, rec.denom_final))


# ----------------------------------------------------------------------------------------------------------------
# The graph


@dataclass
class Graph:
    """The attribution graph of a prompt. Nodes are in the order embeddings, latents (by layer, then position),
    errors (by layer, then position), outputs (most probable first).

    A[t, s] is what node s contributes to node t: to the pre-activation of a latent, or to an output's logit less
    the mean logit. Rows of embeddings and errors are zero (they have no inputs). `value[t] = A[t].sum() + constant[t]`.
    """

    tokens: list[int]
    kind: np.ndarray  # (N,) EMBEDDING, LATENT, ERROR or OUTPUT
    layer: np.ndarray  # (N,) the block of a latent or an error; -1 for an embedding; the number of blocks for an output
    pos: np.ndarray  # (N,) the position
    index: np.ndarray  # (N,) the latent's index in its transcoder; an output's token; a prompt token for an embedding; -1 for an error
    activation: np.ndarray  # (N,) a latent's activation; the norm of an embedding or an error vector; an output's probability
    value: np.ndarray  # (N,) a latent's pre-activation; an output's logit less the mean logit (before the cap); 0 otherwise
    constant: np.ndarray  # (N,) what of the value is no edge
    A: np.ndarray  # (N, N) float64
    check: dict  # the largest errors of the two identities, absolute and relative

    @property
    def n(self) -> int:
        return len(self.kind)

    def weights(self) -> np.ndarray:
        """The probabilities of the outputs, zero elsewhere: where the walks of the influence start."""
        return np.where(self.kind == OUTPUT, self.activation, 0.0)

    def edges(self) -> int:
        return int((self.A != 0).sum())


def top_outputs(probs: Tensor, mass: float = 0.95, most: int = 10) -> list[int]:
    """The most probable tokens, as many as reach `mass` of the probability, at most `most`."""
    order = probs.argsort(descending=True)[:most]
    reached = (probs[order].cumsum(0) >= mass).nonzero()
    return order[: int(reached[0]) + 1].tolist() if len(reached) else order.tolist()


def build_graph(model: GPT, tcs: Transcoders, idx: Tensor, mass: float = 0.95, most: int = 10, batch: int = 256) -> Graph:
    """The attribution graph of the prompt `idx` (T,) for the model's next token after it.

    One backward pass through the frozen model per target (a latent or an output), `batch` targets at a time:
    copies of the prompt are laid side by side, each target's reading direction is applied to its own copy, and
    the gradient at every place a node writes is read off; an edge is that gradient times what the node wrote.
    """
    rec = record(model, idx)
    L, T, d = model.cfg.n_layer, len(idx), model.cfg.d_model
    dtype, device = rec.embed.dtype, rec.embed.device
    with torch.no_grad():
        pre = [tcs.layer_pre(l, rec.reads[l]) for l in range(L)]
        code = [tcs.code(p) for p in pre]
        read = [tuple(t.detach() for t in tcs.read(l)) for l in range(L)]  # (E, offset) per layer
        write = [tuple(t.detach() for t in tcs.write(l)) for l in range(L)]  # (W, b) per layer
        error = [rec.mlp_out[l] - (code[l] @ write[l][0] + write[l][1]) for l in range(L)]
    outputs = top_outputs(rec.probs, mass, most)

    # ---- the nodes
    where = [code[l].nonzero() for l in range(L)]  # per layer (S_l, 2): position, latent; in order of position
    n_latent = [len(w) for w in where]
    first_latent = T + np.concatenate([[0], np.cumsum(n_latent)])  # the first node of each layer's latents
    first_error, first_output = int(first_latent[-1]), int(first_latent[-1]) + L * T
    N = first_output + len(outputs)
    kind, layer, pos, index = (np.zeros(N, dtype=np.int64) for _ in range(4))
    activation, value = np.zeros(N), np.zeros(N)
    kind[:T], layer[:T], pos[:T], index[:T] = EMBEDDING, -1, np.arange(T), idx.cpu().numpy()
    activation[:T] = rec.embed.norm(dim=-1).cpu().numpy()
    for l in range(L):
        s = slice(int(first_latent[l]), int(first_latent[l + 1]))
        p, i = where[l][:, 0], where[l][:, 1]
        kind[s], layer[s], pos[s], index[s] = LATENT, l, p.cpu().numpy(), i.cpu().numpy()
        activation[s], value[s] = code[l][p, i].cpu().numpy(), pre[l][p, i].cpu().numpy()
        e = slice(first_error + l * T, first_error + (l + 1) * T)
        kind[e], layer[e], pos[e], index[e] = ERROR, l, np.arange(T), -1
        activation[e] = error[l].norm(dim=-1).cpu().numpy()
    centered = rec.logits - rec.logits.mean()
    o = slice(first_output, N)
    kind[o], layer[o], pos[o], index[o] = OUTPUT, L, T - 1, outputs
    activation[o], value[o] = rec.probs[outputs].cpu().numpy(), centered[outputs].cpu().numpy()

    # ---- what each node writes, as (T, d) or (S_l, d) vectors
    wrote = [code[l][where[l][:, 0], where[l][:, 1], None] * write[l][0][where[l][:, 1]] for l in range(L)]

    # ---- the constants: the frozen model run on the transcoders' constants alone
    with torch.no_grad():
        reads0, logits0 = frozen_forward(model, rec, torch.zeros(1, T, d, dtype=dtype, device=device),
                                         [write[l][1].expand(1, T, d) for l in range(L)])
    constant = np.zeros(N)
    for l in range(L):
        s = slice(int(first_latent[l]), int(first_latent[l + 1]))
        p, i = where[l][:, 0], where[l][:, 1]
        constant[s] = ((reads0[l][0, p] * read[l][0][i]).sum(-1) + read[l][1][i]).cpu().numpy()
    constant[o] = (logits0[0, -1] - logits0[0, -1].mean())[outputs].cpu().numpy()

    # ---- the edges: a backward pass per target
    A = np.zeros((N, N))
    targets = np.concatenate([np.arange(T, first_error), np.arange(first_output, N)])
    for start in range(0, len(targets), batch):
        chunk = targets[start:start + batch]
        B = len(chunk)
        embed = rec.embed.expand(B, T, d).clone().requires_grad_(True)
        writes = [rec.mlp_out[l].expand(B, T, d).clone().requires_grad_(True) for l in range(L)]
        reads, logits = frozen_forward(model, rec, embed, writes)
        rows = torch.arange(B, device=device)
        total = torch.zeros((), dtype=dtype, device=device)
        for l in range(L):
            mine = torch.from_numpy(layer[chunk] == l).to(device) & torch.from_numpy(kind[chunk] == LATENT).to(device)
            if mine.any():
                at = torch.from_numpy(pos[chunk]).to(device)[mine]
                latent = torch.from_numpy(index[chunk]).to(device)[mine]
                total = total + (reads[l][rows[mine], at] * read[l][0][latent]).sum()
        out = torch.from_numpy(kind[chunk] == OUTPUT).to(device)
        if out.any():
            last = logits[rows[out], -1]
            token = torch.from_numpy(index[chunk]).to(device)[out]
            total = total + (last.gather(1, token[:, None])[:, 0] - last.mean(-1)).sum()
        grads = torch.autograd.grad(total, [embed] + writes, allow_unused=True)
        g_embed, g_writes = grads[0], grads[1:]
        with torch.no_grad():
            if g_embed is not None:
                A[chunk, :T] = (g_embed * rec.embed).sum(-1).double().cpu().numpy()
            for l in range(L):
                if g_writes[l] is None:
                    continue
                s = slice(int(first_latent[l]), int(first_latent[l + 1]))
                A[chunk, s] = (g_writes[l][:, where[l][:, 0]] * wrote[l]).sum(-1).double().cpu().numpy()
                A[chunk, first_error + l * T:first_error + (l + 1) * T] = (g_writes[l] * error[l]).sum(-1).double().cpu().numpy()

    # ---- the two identities
    with torch.no_grad():
        reads_again, logits_again = frozen_forward(model, rec, rec.embed[None], [m[None] for m in rec.mlp_out])
        frozen_abs = max(max(float((reads_again[l][0] - rec.reads[l]).abs().max()) for l in range(L)),
                         float((logits_again[0, -1] - rec.logits).abs().max()))
        frozen_scale = max(max(float(rec.reads[l].abs().max()) for l in range(L)), float(rec.logits.abs().max()))
    has_inputs = (kind == LATENT) | (kind == OUTPUT)
    gap = np.abs(A.sum(1) + constant - value)[has_inputs]
    scale = np.abs(A).sum(1)[has_inputs] + np.abs(constant[has_inputs]) + 1e-30  # the size of what is summed, not of the sum: the terms cancel
    check = {"frozen_abs": frozen_abs, "frozen_rel": frozen_abs / frozen_scale, "edges_abs": float(gap.max()), "edges_rel": float((gap / scale).max())}
    return Graph(tokens=idx.tolist(), kind=kind, layer=layer, pos=pos, index=index, activation=activation, value=value,
                 constant=constant, A=A, check=check)


# ----------------------------------------------------------------------------------------------------------------
# Influence, pruning, scores


def normalize(A: np.ndarray) -> np.ndarray:
    """A~[t, s] = |A[t, s]| / sum over s' of |A[t, s']|: the share of node t's input that comes from s."""
    absolute = np.abs(A)
    total = absolute.sum(1, keepdims=True)
    return np.divide(absolute, total, out=np.zeros_like(absolute), where=total > 0)


def influence(At: np.ndarray, w: np.ndarray) -> np.ndarray:
    """iota = w (A~ + A~^2 + ...) = w ((I - A~)^-1 - I): the probability that a walk from the outputs to the inputs
    passes through each node. Every edge runs from a lower layer to a higher one, so the sum is finite."""
    n = len(w)
    return w @ (np.linalg.inv(np.eye(n) - At) - np.eye(n))


@dataclass
class Pruned:
    nodes: np.ndarray  # (N,) bool: the nodes kept
    edges: np.ndarray  # (N, N) bool: the edges kept, [t, s]
    influence: np.ndarray  # (N,) of the full graph
    scores: dict  # of what is kept: replacement, completeness, the shares of walks ending at tokens, at errors, pruned


def scores(At: np.ndarray, kind: np.ndarray, w: np.ndarray, edges: Optional[np.ndarray] = None) -> dict:
    """Where the walks of the influence end, and how much of each node's input is accounted for.

        to_tokens, to_errors   the share of walks that end at an embedding, at an error node
        pruned                 the share that reaches an edge that was pruned (0 on the full graph)
        replacement            to_tokens / (to_tokens + to_errors + pruned): what was pruned counts as error
        completeness           the share of a node's input that comes from kept nodes other than errors, averaged
                               over the latents and outputs with the weights iota + w
    """
    kept = At if edges is None else At * edges
    iota = influence(kept, w)
    total = w.sum()
    to_tokens, to_errors = iota[kind == EMBEDDING].sum() / total, iota[kind == ERROR].sum() / total
    weight = (iota + w) * ((kind == LATENT) | (kind == OUTPUT))
    explained = kept[:, kind != ERROR].sum(1)
    return {"to_tokens": float(to_tokens), "to_errors": float(to_errors), "pruned": float(max(0.0, 1 - to_tokens - to_errors)),
            "replacement": float(to_tokens), "completeness": float((weight * explained).sum() / weight.sum())}


def prune(graph: Graph, node_share: float = 0.8, edge_share: float = 0.98) -> Pruned:
    """Keep the latents that carry `node_share` of the influence, then the edges that carry `edge_share` of what is
    left; embeddings, errors and outputs are never pruned as nodes, and a latent left without an edge into it or
    out of it goes too."""
    kind, w = graph.kind, graph.weights()
    At = normalize(graph.A)
    iota = influence(At, w)
    inner = kind != OUTPUT
    order = np.argsort(-np.where(inner, iota, -1.0))[: inner.sum()]
    share = np.cumsum(iota[order]) / iota[inner].sum()
    nodes = np.zeros(graph.n, dtype=bool)
    nodes[order[: int(np.searchsorted(share, node_share)) + 1]] = True
    nodes |= kind != LATENT
    left = At * nodes[:, None] * nodes[None, :]
    score = influence(left, w) + w
    edge_score = left * score[:, None]
    flat = np.sort(edge_score[edge_score > 0])[::-1]
    reached = np.searchsorted(np.cumsum(flat) / flat.sum(), edge_share)
    edges = edge_score >= flat[min(reached, len(flat) - 1)]
    while True:  # a latent needs an edge in and an edge out
        lonely = nodes & (kind == LATENT) & ~(edges.any(0) & edges.any(1))
        if not lonely.any():
            break
        nodes &= ~lonely
        edges &= nodes[:, None] & nodes[None, :]
    return Pruned(nodes=nodes, edges=edges, influence=iota, scores=scores(At, kind, w, edges))


# ----------------------------------------------------------------------------------------------------------------
# Interventions: what the edges predict, and what the model does


def linear_effects(graph: Graph) -> np.ndarray:
    """E[t, s]: the change of node t's value (a latent's pre-activation, an output's centered logit) that the graph
    predicts for one unit less of latent s, every other latent answering along its edges: with V[t, s] = A[t, s] /
    a_s the direct effect of one unit of s on t, E = -(V + V^2 + ...) restricted to paths through latents. A latent
    in between is taken to move with its pre-activation, which is right while it stays among its token's k
    largest. Columns of nodes that are not latents are zero."""
    is_latent = graph.kind == LATENT
    V = np.zeros_like(graph.A)
    V[:, is_latent] = graph.A[:, is_latent] / graph.activation[is_latent]
    through = V * is_latent[:, None]  # steps that arrive at a latent, which passes them on
    return -V @ np.linalg.inv(np.eye(graph.n) - through)


@torch.no_grad()
def ablate(model: GPT, tcs: Transcoders, graph: Graph, nodes: list[int], batch: int = 64) -> np.ndarray:
    """Take latent s out of the model, for each s of `nodes` (latents of `graph`): subtract its write, its
    activation times its row of the dictionary, from the output of its MLP at its position, and run the model on
    from there with its own attention, normalizations and MLPs. Returns (len(nodes), N): the change of the value of
    every node of the graph, the pre-activation of a latent (read by its transcoder's encoder on what its MLP now
    reads) and the centered logit of an output, before the cap; zero for embeddings and errors."""
    L, T = model.cfg.n_layer, len(graph.tokens)
    device, dtype = model.embed.weight.device, model.embed.weight.dtype
    idx = torch.tensor(graph.tokens, device=device)
    changes = np.zeros((len(nodes), graph.n))
    latents = [np.flatnonzero((graph.kind == LATENT) & (graph.layer == l)) for l in range(L)]
    outputs = np.flatnonzero(graph.kind == OUTPUT)
    for start in range(0, len(nodes), batch):
        chunk = nodes[start:start + batch]
        B = len(chunk)
        delta = [torch.zeros(B, T, model.cfg.d_model, dtype=dtype, device=device) for _ in range(L)]
        for b, s in enumerate(chunk):
            if graph.kind[s] != LATENT:
                raise ValueError(f"node {s} is not a latent")
            l = int(graph.layer[s])
            delta[l][b, int(graph.pos[s])] = graph.activation[s] * tcs.write(l)[0][int(graph.index[s])].to(dtype)
        hooks = {f"mlp_out.{l}": (lambda x, l=l: x - delta[l]) for l in range(L)}
        _, cache = model.run_with_cache(idx[None].expand(B, T), ["resid_mid.*", "final_norm"], hooks)
        for l in range(L):
            if len(latents[l]):
                reads = model.blocks[l].norm2(cache[f"resid_mid.{l}"])
                at, i = torch.from_numpy(graph.pos[latents[l]]).to(device), torch.from_numpy(graph.index[latents[l]]).to(device)
                E, offset = tcs.read(l)
                now = (reads[:, at] * E[i].to(dtype)).sum(-1) + offset[i].to(dtype)
                changes[start:start + B, latents[l]] = now.double().cpu().numpy() - graph.value[latents[l]]
        logits = model.unembed(cache["final_norm"][:, -1])
        centered = logits - logits.mean(-1, keepdim=True)
        token = torch.from_numpy(graph.index[outputs]).to(device)
        changes[start:start + B, outputs] = centered[:, token].double().cpu().numpy() - graph.value[outputs]
    return changes


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """The rank correlation of two samples: the correlation of their ranks, ties sharing the mean rank."""
    def ranks(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v, kind="stable")
        r = np.empty(len(v))
        r[order] = np.arange(len(v))
        for value in np.unique(v):
            same = v == value
            r[same] = r[same].mean()
        return r
    if len(x) < 2:
        return float("nan")
    rx, ry = ranks(np.asarray(x, dtype=float)), ranks(np.asarray(y, dtype=float))
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")
