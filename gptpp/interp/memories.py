"""The MLP as a key-value memory: what triggers a neuron, and what its output promotes.

An MLP of the model is MLP(x) = sum_i m_i(x) v_i with m_i(x) = act(k_i . x): neuron i matches its input weights
k_i (row i of the first matrix, the key) against the normalized stream x, and adds its output weights v_i (column
i of the second matrix, the value) in proportion to the match, the memory coefficient m_i. Two papers read the
two sides:

    keys     Geva, Schuster, Berant and Levy, "Transformer feed-forward layers are key-value memories", EMNLP 2021
             (arXiv:2012.14913): for a neuron, the positions of training text with the largest coefficient (its
             top triggers, each read as the prefix that ends there). Annotators found patterns in them, shallow
             ones in the lower layers (the prefixes end in the same word), semantic ones in the upper layers.
    values   the same paper, Section 4, and Geva, Caciularu, Wang and Goldberg, "Transformer feed-forward layers
             build predictions by promoting concepts in the vocabulary space", EMNLP 2022 (arXiv:2203.14680): the
             value read through the unembedding, W_U v_i, is a fixed score for every token of the vocabulary;
             when the neuron fires, the tokens at the top of that ranking gain probability and those at the
             bottom lose it. The agreement rate of a layer is the share of its neurons whose top-scoring token is
             the token that follows the neuron's top trigger.

What this module measures, on any model of the lab whose MLP is not gated:

    value_scores(model, layer, neurons)       W_U v_i for the chosen neurons, (n, V)
    promoted(scores, k)                       the k tokens with the largest and with the smallest score
    top_triggers(model, windows, neurons, k)  the k positions of the text with the largest coefficient, for every
                                              chosen neuron of every chosen layer, in one pass over the text
    same_last_token(tokens)                   the share of a neuron's top triggers that end in one token (an
                                              automatic stand-in for the annotators' "shallow pattern")
    neuron_measures(...)                      per neuron: that share, whether the top-scoring token follows the
                                              top trigger (the papers' agreement), and the share of the top
                                              triggers that are followed by one of the value's top tokens
    layer_statistics(measures)                their means over the neurons of a layer that fire at all

A coefficient is the hook `mlp.l.post`. With ReLU or squared ReLU it is never negative, so a neuron pushes its
ranking one way only. Normalizing a lone vector by an RMS norm without gain rescales it, so the ranking W_U v_i is
the ranking after the final normalization too; a model with a layer norm would need the vector normalized first.
"""
from __future__ import annotations

from typing import Iterable, Optional

import torch
from torch import Tensor

from ..model import GPT


class _Stop(Exception):
    """Raised by the hook of the last layer wanted: nothing after it is needed."""


def value_vectors(model: GPT, layer: int, neurons: Optional[Tensor] = None) -> Tensor:
    """The values of a layer, one row a neuron: columns of the MLP's output matrix, (n, d)."""
    w = model.blocks[layer].mlp.proj.weight.float().T
    return w if neurons is None else w[neurons.to(w.device)]


@torch.no_grad()
def value_scores(model: GPT, layer: int, neurons: Optional[Tensor] = None) -> Tensor:
    """W_U v_i for the neurons of a layer: (n, V), the static score of every token under each value."""
    return value_vectors(model, layer, neurons) @ model.unembed.weight.float().T


def promoted(scores: Tensor, k: int = 30) -> dict[str, Tensor]:
    """The two ends of each ranking in scores (n, V): `top_ids`, `top_scores` (the tokens a firing neuron
    promotes, largest first) and `bottom_ids`, `bottom_scores` (those it suppresses, smallest first), each (n, k)."""
    top, bottom = scores.topk(k, dim=-1), (-scores).topk(k, dim=-1)
    return {"top_ids": top.indices, "top_scores": top.values, "bottom_ids": bottom.indices, "bottom_scores": -bottom.values}


@torch.no_grad()
def top_triggers(model: GPT, windows: Iterable[Tensor], neurons: dict[int, Tensor], k: int = 25,
                 autocast: bool = False) -> dict[int, dict[str, Tensor]]:
    """One pass over text: for every neuron listed in `neurons` (layer -> indices), the k positions with the
    largest memory coefficient.

    `windows` yields batches (B, T) of token ids, all with the same T; window w, position t of the whole stream is
    numbered w * T + t in the order the batches arrive. Returns, per layer, `coef` (n, k) the coefficients, largest
    first, and `window`, `pos` (n, k) where each occurred; a neuron that fired on fewer than k positions has
    zeros (ReLU family) in its tail. The forward pass stops after the last layer asked for.
    """
    device = model.embed.weight.device
    layers = sorted(neurons)
    best = {l: None for l in layers}                       # (values (k, n), flat positions (k, n))
    state = {"offset": 0, "T": 0}

    def collect(layer: int):
        cols = neurons[layer].to(device)

        def fn(x: Tensor) -> None:
            B, T, _ = x.shape
            acts = x[..., cols].reshape(B * T, -1).float()                           # (B T, n)
            vals, rows = acts.topk(min(k, B * T), dim=0)
            where = rows + state["offset"]
            if best[layer] is not None:
                vals, where = torch.cat((best[layer][0], vals)), torch.cat((best[layer][1], where))
                vals, keep = vals.topk(min(k, vals.size(0)), dim=0)
                where = where.gather(0, keep)
            best[layer] = (vals, where)
            if layer == layers[-1]:
                raise _Stop

        return fn

    hooks = {f"mlp.{l}.post": collect(l) for l in layers}
    for batch in windows:
        batch = batch.to(device)
        state["T"] = batch.size(1)
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast and device.type == "cuda"):
                model.run_with_hooks(batch, hooks)
        except _Stop:
            pass
        state["offset"] += batch.numel()
    T = state["T"]
    return {l: {"coef": best[l][0].T.contiguous().cpu(), "window": (best[l][1].T // T).cpu(), "pos": (best[l][1].T % T).cpu()}
            for l in layers}


def same_last_token(tokens: Tensor) -> tuple[Tensor, Tensor]:
    """tokens (n, k): the token at each of a neuron's top triggers. Returns the share of the k that are the
    neuron's commonest trigger token, (n,), and that token, (n,)."""
    mode = tokens.mode(dim=1).values
    return (tokens == mode.unsqueeze(1)).float().mean(1), mode


def neuron_measures(trigger_tokens: Tensor, next_tokens: Tensor, coef: Tensor, top_ids: Tensor) -> dict[str, Tensor]:
    """The measures of each of n neurons from its k top triggers and its value's ranking.

        trigger_tokens (n, k)   the token at each top trigger, strongest first
        next_tokens    (n, k)   the token that follows it in the text (-1 where the text ends)
        coef           (n, k)   the memory coefficients
        top_ids        (n, m)   the m top-scoring tokens of W_U v_i, best first

    ->  same_last  (n,)  share of the triggers that end in the neuron's commonest trigger token; `last_token` (n,)
        agrees     (n,)  whether the value's top token is the token after the top trigger (the papers' agreement)
        follows    (n,)  share of the triggers that are followed by one of the value's m top tokens
        alive      (n,)  whether the neuron fired on at least k positions (its weakest top trigger is positive)
    """
    share, mode = same_last_token(trigger_tokens)
    follows = (next_tokens.unsqueeze(-1) == top_ids.unsqueeze(1)).any(-1).float().mean(1)
    return {"same_last": share, "last_token": mode, "agrees": next_tokens[:, 0] == top_ids[:, 0], "follows": follows,
            "alive": coef[:, -1] > 0}


def layer_statistics(measures: dict[str, Tensor]) -> dict[str, float]:
    """A layer's means over its sampled neurons that are alive: `same_last` (mean share of top triggers ending in
    one token), `agreement` (share of neurons whose top token follows their top trigger), `follows` (mean share of
    top triggers followed by a top token of the value), with `neurons` sampled and `alive` among them."""
    alive = measures["alive"]
    n = int(alive.sum())
    mean = lambda x: float(x[alive].float().mean()) if n else float("nan")
    return {"same_last": mean(measures["same_last"]), "agreement": mean(measures["agrees"]),
            "follows": mean(measures["follows"]), "neurons": int(alive.numel()), "alive": n}
