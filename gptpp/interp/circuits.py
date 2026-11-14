"""Circuits: an attention-only transformer read from its weights, and the scores that find its heads.

"A mathematical framework for transformer circuits" (Elhage et al., 2021; FW below) rewrites a transformer so that
its weights can be read. Every layer reads the residual stream and adds to it, so the stream at the end is a sum,
and with the attention patterns held fixed everything else is linear in the one-hot tokens. A head h does two
separable things, each a d x d matrix of rank at most d_head:

    W_QK = W_Q^T R W_K / sqrt(d_head)   where it looks: the score of a destination vector x against a source vector y
                                        is x^T W_QK y (R turns the key by the distance between them: rotary positions)
    W_OV = W_O W_V                      what looking does: a source vector y, attended to, adds W_OV y at the destination

Between the embedding E (V x d, one row a token) and the unembedding U (V x d) these become tables over the
vocabulary, the QK circuit E W_QK E^T [destination token, source token] and the OV circuit U W_OV E^T [output token,
source token], and the logits of an attention-only model without normalization are exactly a sum over paths:

    0 heads   U E^T                             the direct path: a table of bigrams (fit_zero_layer trains the
                                                model that is nothing else; bigram_agreement holds a table to counts)
    1 head    A^h (x) U W_OV^h E^T              "source ... destination -> output": skip-trigrams
    2 heads   A^h2 A^h1 (x) U W_OV^h2 W_OV^h1 E^T   and, inside the patterns of the second layer, composition

`path_expansion` computes those terms from the weights and the recorded patterns and nothing else; the test of
this module checks that they add up to the model's logits. The scores:

    from the weights      ov_eigenvalues, copying_statistic (FW: sum of eigenvalues over sum of their absolute values),
                          copying_score and self_rank (does a token raise its own logit), composition (FW's Q-, K- and
                          V-composition as corrected in 2023, with the random baseline), k_composition_eigenvalues
    from activations      prefix_matching_score and previous_token_score ("In-context learning and induction
                          heads", Olsson et al., 2022; IH below), on repeated random tokens and on text;
                          in_context_learning_score; repeated_loss; loss_by_order (FW's term importance);
                          ablations of heads, with patterns recomputed or frozen; patch_inputs and
                          induction_paths_explained (path patching into the second layer)

Conventions. Matrices are stored as PyTorch stores them: E and U have one row a token; W_Q, W_K, W_V are
(d_head, d), W_O is (d, d_head). A head is (layer, head), both counted from zero. Patterns are (B, H, T, T) with
the destination (the query's position) first and the source second; every row sums to one. Weight arithmetic is
done in float64 on the device of the weights.

What is exact and what is not. For a model with use_mlp=False, norm="none", embed_norm=False, bias=False,
qk_norm=False, logit_softcap=0 and positions that never enter the stream (rotary, or none), every identity here is
exact and `is_exact(model)` is true. On any other model of the lab the tables and scores are still defined from
the same matrices (biases, normalization and the MLPs are ignored, as FW does), and the activation scores are
measured on the real forward pass, so they hold for every model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from ..model import GPT, HookFn

Head = tuple[int, int]  # (layer, head)
Tokens = Optional[Union[Tensor, Sequence[int]]]  # a selection of token ids; None = the whole vocabulary


# ----------------------------------------------------------------------------------------------------------------
# The weights


def is_exact(model: GPT) -> bool:
    """True if the framework's algebra is exact for this model: attention only, nothing between the stream and
    its readers, position only in the scores."""
    cfg = model.cfg
    return (not cfg.use_mlp and cfg.norm == "none" and not cfg.embed_norm and not cfg.bias and not cfg.qk_norm
            and cfg.logit_softcap == 0 and cfg.pos in ("rope", "none") and cfg.dropout == 0)


def embedding(model: GPT) -> Tensor:
    """E, (V, d): row a is what token a writes into the residual stream."""
    return model.embed.weight.detach().double()


def unembedding(model: GPT) -> Tensor:
    """U, (V, d): the logit of token c is the dot product of row c with the final residual stream."""
    return model.unembed.weight.detach().double()


@dataclass(frozen=True)
class HeadWeights:
    """The four matrices of one head: q, k, v are (d_head, d) and read the stream; o is (d, d_head) and writes."""

    q: Tensor
    k: Tensor
    v: Tensor
    o: Tensor


def head_weights(model: GPT, layer: int, head: int) -> HeadWeights:
    """The slices of the layer's four matrices that belong to one head. With grouped-query attention a key/value
    head is shared by a group of query heads."""
    attn = model.blocks[layer].attn
    dh = attn.d_head
    kv = head // (attn.n_head // attn.n_kv_head)
    rows, kv_rows = slice(head * dh, (head + 1) * dh), slice(kv * dh, (kv + 1) * dh)
    return HeadWeights(q=attn.q.weight.detach().double()[rows], k=attn.k.weight.detach().double()[kv_rows],
                       v=attn.v.weight.detach().double()[kv_rows], o=attn.o.weight.detach().double()[:, rows])


def relative_rotation(model: GPT, offset: float) -> Tensor:
    """R(offset), (d_head, d_head): the turn a key gets relative to a query `offset` positions after it.

    With rotary positions the model turns the query by its position and the key by its own; only the difference
    survives the dot product, so the score of a destination `offset` tokens after its source is
    q^T R(offset) k / sqrt(d_head), with R the block rotation by -offset * frequency in every pair of coordinates
    (i, i + d_head / 2). Any other position scheme does not touch queries and keys: R is the identity.
    """
    dh = model.cfg.d_head
    if model.cfg.pos != "rope" or offset == 0:
        return torch.eye(dh, dtype=torch.float64, device=model.embed.weight.device)
    half = dh // 2
    frequency = model.cfg.rope_base ** (-torch.arange(0, dh, 2, dtype=torch.float64) / dh)
    angle = offset * frequency
    rotation = torch.zeros(dh, dh, dtype=torch.float64)
    index = torch.arange(half)
    rotation[index, index] = angle.cos()
    rotation[index, index + half] = angle.sin()
    rotation[index + half, index] = -angle.sin()
    rotation[index + half, index + half] = angle.cos()
    return rotation.to(model.embed.weight.device)


def ov_matrix(model: GPT, layer: int, head: int) -> Tensor:
    """W_OV = W_O W_V, (d, d): a source vector y, attended to with weight one, adds W_OV y to the destination."""
    w = head_weights(model, layer, head)
    return w.o @ w.v


def qk_matrix(model: GPT, layer: int, head: int, offset: float = 0) -> Tensor:
    """W_QK = W_Q^T R(offset) W_K / sqrt(d_head), (d, d): the score of a destination vector x against a source
    vector y that stands `offset` tokens before it is x^T W_QK y. The 1/sqrt(d_head) of the model is folded in."""
    w = head_weights(model, layer, head)
    return w.q.T @ relative_rotation(model, offset) @ w.k / math.sqrt(model.cfg.d_head)


def fit_zero_layer(bigram: Tensor, d_model: int, steps: int = 1500, lr: float = 0.03, seed: int = 0,
                   device: Union[str, torch.device] = "cpu") -> tuple[Tensor, Tensor, list[float]]:
    """The zero-layer transformer of a corpus: an embedding E (V, d) and an unembedding U (V, d), untied and
    without biases, trained so that softmax(U E[a]) predicts the token after a. Returns (E, U, losses).

    A zero-layer model sees one token and predicts the next, so its loss over a corpus,
    -sum_{a,c} n(a, c) log softmax(U E[a])[c] / N, depends on the corpus only through the bigram counts
    n(a, c) [previous, next]: it can be trained on that table, every pair of the corpus in every step, with no
    sampling. An unconstrained table of logits would reach the conditional entropy of the bigrams exactly (the
    logit of c after a equal to log p(c | a) plus a constant per a); with d < V the model finds the nearest
    table of rank d in that cross-entropy sense (FW, "Zero-layer transformers"). Adam, full batch.
    """
    generator = torch.Generator().manual_seed(seed)
    vocab = bigram.size(0)
    counts = bigram.to(device=device, dtype=torch.float32)
    weight = counts / counts.sum()
    embed = (0.02 * torch.randn(vocab, d_model, generator=generator)).to(device).requires_grad_()
    unembed = (0.02 * torch.randn(vocab, d_model, generator=generator)).to(device).requires_grad_()
    optimizer = torch.optim.Adam([embed, unembed], lr=lr, betas=(0.9, 0.99))
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=lr / 20)
    losses = []
    with torch.enable_grad():  # a caller that measures under no_grad still trains here
        for _ in range(steps):
            loss = -(weight * F.log_softmax(embed @ unembed.T, dim=1)).sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            schedule.step()
            losses.append(loss.item())
    return embed.detach().double(), unembed.detach().double(), losses


def bigram_agreement(table: Tensor, bigram: Tensor, sources: Tokens = None) -> dict[str, float]:
    """How well a table of logits [output token, source token] predicts the token after each source token, held
    against the corpus's bigram counts [previous, next]; over `sources`, weighted by how often each occurs.

        loss      the cross-entropy of softmax(table column) under the corpus's next-token distribution, in nats
        entropy   the entropy of that distribution: the least any table can reach
        kl        loss - entropy
        top1      the share of sources (unweighted) whose highest logit is the corpus's most frequent successor
    """
    ids = torch.arange(bigram.size(0)) if sources is None else torch.as_tensor(sources, dtype=torch.long)
    counts = bigram[ids].double().to(table.device)  # (n, V)
    seen = counts.sum(dim=1) > 0
    counts, columns = counts[seen], _rows(table.T, ids.to(table.device)[seen.to(ids.device)])  # (n, V) each
    p = counts / counts.sum(dim=1, keepdim=True)
    weight = counts.sum(dim=1) / counts.sum()
    logq = F.log_softmax(columns.double(), dim=1)
    loss = (-(p * logq).sum(dim=1) * weight).sum().item()
    entropy = (-torch.xlogy(p, p).sum(dim=1) * weight).sum().item()
    top1 = (columns.argmax(dim=1) == counts.argmax(dim=1)).double().mean().item()
    return {"loss": loss, "entropy": entropy, "kl": loss - entropy, "top1": top1}


def _rows(matrix: Tensor, tokens: Tokens) -> Tensor:
    if tokens is None:
        return matrix
    return matrix[torch.as_tensor(tokens, dtype=torch.long, device=matrix.device)]


# ----------------------------------------------------------------------------------------------------------------
# The tables over the vocabulary


def direct_path(model: GPT, sources: Tokens = None, outputs: Tokens = None) -> Tensor:
    """U E^T, [output token, source token]: the logits a token gives the next token with no head in between.

    It is all there is of a zero-layer model, whose best table under cross-entropy is the log of the bigram
    probabilities up to a constant per source (FW, "Zero-layer transformers"); in a deeper model it is the same
    table less what the heads supply.
    """
    return _rows(unembedding(model), outputs) @ _rows(embedding(model), sources).T


def ov_circuit(model: GPT, layer: int, head: int, sources: Tokens = None, outputs: Tokens = None) -> Tensor:
    """The OV circuit U W_OV E^T, [output token, source token]: the change in each logit per unit of attention
    paid to a source token (FW, "Splitting attention head terms into query-key and output-value circuits")."""
    w = head_weights(model, layer, head)
    return (_rows(unembedding(model), outputs) @ w.o) @ (w.v @ _rows(embedding(model), sources).T)


def qk_circuit(model: GPT, layer: int, head: int, offset: float = 1, dests: Tokens = None, sources: Tokens = None,
               center: bool = False) -> Tensor:
    """The QK circuit E W_QK E^T, [destination token, source token]: the attention score, before the softmax, of
    a destination token for a source token `offset` positions before it.

    A softmax ignores a constant added to a row, so scores are comparable within a row only: `center=True`
    subtracts each row's mean over the sources given.
    """
    w = head_weights(model, layer, head)
    queries = _rows(embedding(model), dests) @ w.q.T
    keys = _rows(embedding(model), sources) @ w.k.T
    table = queries @ relative_rotation(model, offset) @ keys.T / math.sqrt(model.cfg.d_head)
    return table - table.mean(dim=1, keepdim=True) if center else table


# ----------------------------------------------------------------------------------------------------------------
# Copying, from the weights


def ov_eigenvalues(model: GPT, layer: int, head: int) -> Tensor:
    """The d_head eigenvalues that the OV circuit U W_O W_V E^T can have besides zero, complex (d_head,).

    The circuit is V x V but has rank at most d_head, and the nonzero eigenvalues of a product AB are those of
    BA: they are read off the d_head x d_head matrix (W_V E^T)(U W_O).
    """
    w = head_weights(model, layer, head)
    small = (w.v @ embedding(model).T) @ (unembedding(model) @ w.o)
    return torch.linalg.eigvals(small)


def copying_statistic(eigenvalues: Tensor) -> float:
    """FW's summary of an OV circuit, sum(eigenvalues) / sum(|eigenvalues|), in [-1, 1].

    An eigenvector with a positive eigenvalue is a set of tokens that, attended to, raises the logits of those
    same tokens: copying. The numerator is the trace, the total by which tokens raise their own logits. A random
    matrix has eigenvalues spread evenly around zero and scores near 0; FW counted 10 copying heads of 12 in its
    one-layer model. A matrix with only zero eigenvalues (an output matrix initialized at zero) scores 0.
    """
    total = eigenvalues.abs().sum().item()
    return eigenvalues.sum().real.item() / total if total > 0 else 0.0


def k_composition_eigenvalues(model: GPT, first: Head, second: Head, offset: float = 0) -> Tensor:
    """Eigenvalues of E W_QK^second W_OV^first E^T, [destination token, token before the source], complex (d_head,).

    This is the table behind an induction head (FW, "How induction heads work"): the key of the second head
    reads what the first head wrote, and if the first is a previous-token head, the score of destination token b
    for a source whose predecessor is token a is entry [b, a]. Positive eigenvalues mean "match the same token".
    """
    w1, w2 = head_weights(model, *first), head_weights(model, *second)
    e = embedding(model)
    rotation = relative_rotation(model, offset)
    # E w2.q^T R w2.k w1.o w1.v E^T has the nonzero eigenvalues of (R w2.k w1.o)(w1.v E^T E w2.q^T).
    small = (rotation @ w2.k @ w1.o) @ ((w1.v @ e.T) @ (e @ w2.q.T))
    return torch.linalg.eigvals(small)


def copying_score(model: GPT, layer: int, head: int, tokens: Tokens) -> float:
    """The share of the logit a head raises, within a sample of tokens, that falls on the attended token itself.

    After IH's "copying" evaluator, which is described but not written as a formula; this is our definition.
    Take the OV circuit's columns for the sample, measure every logit from the column's mean over the whole
    vocabulary, keep the increases, and restrict the outputs to the sample: the score is the sum of the diagonal
    over the sum of all entries. 1 means attending to a token raises that token and no other of the sample; a
    head that raises tokens at random scores about 1 / len(tokens).
    """
    ids = torch.as_tensor(tokens, dtype=torch.long, device=model.embed.weight.device)
    columns = ov_circuit(model, layer, head, sources=ids)  # (V, n)
    raised = F.relu(columns - columns.mean(dim=0, keepdim=True))[ids]  # (n, n): sample outputs by sample sources
    total = raised.sum().item()
    return raised.diagonal().sum().item() / total if total > 0 else 0.0


def self_rank(model: GPT, layer: int, head: int, tokens: Tokens) -> Tensor:
    """For each token of the sample, the rank of its own logit in its OV column (0 = attending to the token
    raises that token more than any other of the vocabulary). FW's alternative to the eigenvalue summary."""
    ids = torch.as_tensor(tokens, dtype=torch.long, device=model.embed.weight.device)
    columns = ov_circuit(model, layer, head, sources=ids)
    own = columns[ids, torch.arange(len(ids), device=ids.device)]
    return (columns > own).sum(dim=0)


# ----------------------------------------------------------------------------------------------------------------
# Composition


def _frobenius(*factors: Tensor) -> float:
    product = factors[0]
    for factor in factors[1:]:
        product = product @ factor
    return torch.linalg.matrix_norm(product).item()


def _ratio(numerator: float, denominator: float) -> float:
    """A composition score; 0 when a matrix of the denominator is zero (an output matrix initialized at zero
    writes nothing, so nothing composes with it)."""
    return numerator / denominator if denominator > 0 else 0.0


def composition(model: GPT, first: Head, second: Head, offset: float = 0, cyclic: bool = False) -> dict[str, float]:
    """FW's composition scores of an earlier head `first` with a later head `second`, each in [0, 1]:

        q   |W_QK^T W_OV| / (|W_QK^T| |W_OV|)   the query of the second reads what the first wrote
        k   |W_QK   W_OV| / (|W_QK|   |W_OV|)   the key of the second reads it
        v   |W_OV'  W_OV| / (|W_OV'|  |W_OV|)   the value of the second reads it

    with Frobenius norms, W_QK and W_OV' of the second head and W_OV of the first. A score is the share of what
    the first head writes that lands where the second reads, weighted by singular values; subtract
    `composition_baseline` to see what exceeds chance. With rotary positions W_QK depends on the distance between
    destination and source; `offset` chooses it (0: no turn).

    `cyclic=True` reproduces the quantity of FW's 2021 figure, corrected in a note added to the paper in 2023:
    every Frobenius norm of a product was computed on the product cyclically permuted to its d_head x d_head
    form (|W_K W_O W_V W_Q^T| for the key). Traces and eigenvalues survive that permutation; a Frobenius norm
    does not. The permuted quantity is large only when the key reads the first head's output AND the query
    reads the same space as its value, which is the induction circuit: the figure showed it and hid the rest.
    """
    w1, w2 = head_weights(model, *first), head_weights(model, *second)
    rotation = relative_rotation(model, offset)
    if cyclic:
        ov = _frobenius(w1.v, w1.o)
        qk = _frobenius(rotation @ w2.k, w2.q.T)
        ov2 = _frobenius(w2.v, w2.o)
        return {"q": _ratio(_frobenius(w2.q, w1.o, w1.v, w2.k.T, rotation.T), qk * ov),
                "k": _ratio(_frobenius(rotation @ w2.k, w1.o, w1.v, w2.q.T), qk * ov),
                "v": _ratio(_frobenius(w2.v, w1.o, w1.v, w2.o), ov2 * ov)}
    ov = w1.o @ w1.v
    qk = w2.q.T @ rotation @ w2.k
    ov2 = w2.o @ w2.v
    norm_ov, norm_qk, norm_ov2 = _frobenius(ov), _frobenius(qk), _frobenius(ov2)
    return {"q": _ratio(_frobenius(qk.T, ov), norm_qk * norm_ov),
            "k": _ratio(_frobenius(qk, ov), norm_qk * norm_ov),
            "v": _ratio(_frobenius(ov2, ov), norm_ov2 * norm_ov)}


def composition_table(model: GPT, lower: int, upper: int, offset: float = 0, cyclic: bool = False) -> dict[str, Tensor]:
    """The three scores for every head of layer `lower` with every head of layer `upper`: {"q", "k", "v"}, each
    (heads of lower, heads of upper), before the baseline is subtracted."""
    n_head = model.cfg.n_head
    table = {name: torch.zeros(n_head, n_head, dtype=torch.float64) for name in ("q", "k", "v")}
    for h1 in range(n_head):
        for h2 in range(n_head):
            scores = composition(model, (lower, h1), (upper, h2), offset, cyclic)
            for name, value in scores.items():
                table[name][h1, h2] = value
    return table


def baseline(d_model: int) -> float:
    """The baseline subtracted from a composition score: 1 / sqrt(d_model), 0.0625 at width 256.

    If what one head writes stands in general position to what another reads (a uniformly random rotation Q of
    the stream between them), E |M2 Q M1|^2 = |M2|^2 |M1|^2 / d: the root mean square of the score is
    1 / sqrt(d), whatever the two matrices are. `composition_baseline` measures the same level on Gaussian
    matrices."""
    return 1 / math.sqrt(d_model)


def composition_baseline(d_model: int, d_head: int, draws: int = 200, seed: int = 0, cyclic: bool = False) -> float:
    """The composition score of heads whose four matrices are Gaussian noise of the same shapes, averaged over
    `draws`: what FW subtracts ("the empirical expected amount for random matrices of the same shapes").

    For Gaussian factors E|W_Q^T W_K W_O W_V|^2 = d^3 dh^2 s^8 and |W_Q^T W_K|^2 |W_O W_V|^2 is about
    d^4 dh^2 s^8, so the score concentrates at 1 / sqrt(d_model) whatever d_head is, and the three scores share
    one baseline. The cyclically permuted quantity has another, about 1 / sqrt(d_head). Neither score reaches 1
    in practice: two heads whose d_head directions match perfectly and evenly score 1 / sqrt(d_head).
    """
    generator = torch.Generator().manual_seed(seed)
    total = 0.0
    for _ in range(draws):
        q, k, v = (torch.randn(d_head, d_model, generator=generator, dtype=torch.float64) for _ in range(3))
        o = torch.randn(d_model, d_head, generator=generator, dtype=torch.float64)
        if cyclic:
            total += _frobenius(k, o, v, q.T) / (_frobenius(k, q.T) * _frobenius(v, o))
        else:
            total += _frobenius(q.T @ k, o @ v) / (_frobenius(q.T @ k) * _frobenius(o @ v))
    return total / draws


# ----------------------------------------------------------------------------------------------------------------
# The path expansion


@dataclass
class Paths:
    """The residual stream at the end of an attention-only model as a sum over paths.

    `heads[p]` is the path of term p: () for the direct path, ((0, 3),) for one head, ((0, 3), (1, 5)) for the
    second-layer head 5 reading what the first-layer head 3 wrote (a virtual head). `stream[p]` is (B, T, d): what
    that path adds to the final stream, computed from the embedding, the OV matrices and the recorded patterns.
    """

    heads: list[tuple[Head, ...]]
    stream: Tensor  # (paths, B, T, d)
    unembed: Tensor  # (V, d)

    def logits(self, position: Optional[int] = None, outputs: Tokens = None) -> Tensor:
        """Each path's contribution to the logits: (paths, B, T, n_outputs), or (paths, B, n_outputs) at one position."""
        stream = self.stream if position is None else self.stream[:, :, position]
        return stream @ _rows(self.unembed, outputs).T

    def order(self, p: int) -> int:
        return len(self.heads[p])

    def by_order(self) -> Tensor:
        """The stream summed within each order: (orders, B, T, d); order n holds the paths through n heads."""
        orders = max(len(path) for path in self.heads) + 1
        out = torch.zeros((orders,) + self.stream.shape[1:], dtype=self.stream.dtype, device=self.stream.device)
        for p, path in enumerate(self.heads):
            out[len(path)] += self.stream[p]
        return out


@torch.no_grad()
def attention_patterns(model: GPT, idx: Tensor, hooks: Optional[dict[str, HookFn]] = None) -> list[Tensor]:
    """The pattern of every layer on idx (B, T): a list of (B, H, T, T), destination first."""
    _, cache = model.run_with_cache(idx, names=["attn.*.pattern"], hooks=hooks)
    return [cache[f"attn.{layer}.pattern"] for layer in range(model.cfg.n_layer)]


@torch.no_grad()
def path_expansion(model: GPT, idx: Tensor, patterns: Optional[Sequence[Tensor]] = None) -> Paths:
    """Every path from the tokens to the final residual stream of an exact attention-only model (FW, "Path
    expansion of logits"), with the attention patterns frozen at their values on idx (B, T).

    The stream after layer l is the stream before it plus, for every head, A^h (x) W_OV^h applied to it. Applying
    that to a sum of paths gives the same paths and each of them extended by each head of the layer, so a model of
    L layers of H heads has (1 + H)^L paths: 1 + H for one layer, 1 + 2H + H^2 for two. Only the embedding, the
    OV matrices and the patterns are used; that the terms add up to the model's own logits is the test.
    """
    if not is_exact(model):
        raise ValueError("the path expansion is exact only for an attention-only model without normalization, "
                         "biases, QK normalization or a logit soft-cap (see is_exact)")
    if patterns is None:
        patterns = attention_patterns(model, idx)
    heads: list[tuple[Head, ...]] = [()]
    terms = [embedding(model)[idx]]
    for layer in range(model.cfg.n_layer):
        new_heads, new_terms = [], []
        for head in range(model.cfg.n_head):
            ov, pattern = ov_matrix(model, layer, head), patterns[layer][:, head].double()
            for path, term in zip(heads, terms):
                new_heads.append(path + ((layer, head),))
                new_terms.append(pattern @ (term @ ov.T))
        heads, terms = heads + new_heads, terms + new_terms
    return Paths(heads, torch.stack(terms), unembedding(model))


@torch.no_grad()
def stream_by_order(model: GPT, idx: Tensor, patterns: Optional[Sequence[Tensor]] = None) -> Tensor:
    """The final residual stream of an exact attention-only model split by the number of heads a path runs
    through: (orders, B, T, d), order 0 the embedding, order n the sum of all paths through n heads, with the
    patterns frozen at their values on idx. The same sums as path_expansion(...).by_order(), without forming the
    paths one by one: a head of layer l turns what was order n before the layer into order n + 1."""
    if not is_exact(model):
        raise ValueError("the expansion by order is exact only for a model for which is_exact is true")
    if patterns is None:
        patterns = attention_patterns(model, idx)
    orders = [embedding(model)[idx]]
    for layer in range(model.cfg.n_layer):
        added = [torch.zeros_like(orders[0]) for _ in range(len(orders) + 1)]
        for head in range(model.cfg.n_head):
            ov, pattern = ov_matrix(model, layer, head), patterns[layer][:, head].double()
            for n, term in enumerate(orders):
                added[n + 1] += pattern @ (term @ ov.T)
        orders = [term + added[n] for n, term in enumerate(orders)] + [added[-1]]
    return torch.stack(orders)


@torch.no_grad()
def loss_by_order(model: GPT, idx: Tensor) -> list[float]:
    """FW's "term importance": the next-token loss on idx (B, T), in nats, when the logits are built from the
    paths through at most n heads, for n = 0 .. n_layer, with every pattern frozen at its value in the real
    forward pass. Entry 0 is the direct path alone; the last entry is the model's own loss. The gain of order n
    is entry n - 1 minus entry n (for n = 0, log(vocabulary) minus entry 0)."""
    stream = stream_by_order(model, idx)
    unembed, targets = unembedding(model), idx[:, 1:]
    losses, partial = [], torch.zeros_like(stream[0])
    for term in stream:
        partial = partial + term
        logp = F.log_softmax(partial[:, :-1] @ unembed.T, dim=-1)
        losses.append(-logp.gather(2, targets[:, :, None]).mean().item())
    return losses


# ----------------------------------------------------------------------------------------------------------------
# Path patching into one layer


@torch.no_grad()
def patch_inputs(model: GPT, idx: Tensor, layer: int, heads: Sequence[int], query: Optional[Tensor] = None,
                 key: Optional[Tensor] = None, value: Optional[Tensor] = None) -> Tensor:
    """Logits on idx (B, T) when chosen heads of one layer compute their queries, keys or values from streams
    given by the caller, (B, T, d) each, in place of the stream the model built (None leaves that input as it
    is). Everything else runs as usual: the heads' patterns and outputs follow from the patched inputs, the
    other heads of the layer and all later layers read what results.

    This is path patching for an attention-only model without normalization (Wang et al., 2022, arXiv:2211.00593;
    Goldowsky-Dill et al., 2023, arXiv:2304.05969). Its stream is a sum of what the embedding and each earlier
    head wrote, so "the head's key sees this path from the real input and every other path from another input"
    is a stream assembled from the two runs' terms and handed to the key alone.
    """
    attn = model.blocks[layer].attn
    which = list(heads)

    def swap(name: str, stream: Optional[Tensor], pick: int) -> dict[str, HookFn]:
        if stream is None:
            return {}
        projected = attn.project(model.blocks[layer].norm1(stream.to(model.embed.weight.dtype)))[pick]

        def hook(x: Tensor) -> Tensor:
            x = x.clone()
            x[:, which] = projected[:, which]
            return x
        return {f"attn.{layer}.{name}": hook}

    hooks = {**swap("q", query, 0), **swap("k", key, 1), **swap("v", value, 2)}
    return model.run_with_hooks(idx, hooks) if hooks else model(idx)


@torch.no_grad()
def induction_paths_explained(model: GPT, reference: Tensor, other: Tensor, induction: Sequence[int],
                              previous: int, length: int, repeats: int) -> dict[str, float]:
    """How much of what the induction heads do for the loss is carried by the three paths of the minimal circuit,
    for a two-layer attention-only model: the share explained, in the sense of Goldowsky-Dill et al. (2023).

    `reference` and `other` are (B, T) repeated random sequences with different tokens; `induction` the
    second-layer heads, `previous` the first-layer previous-token head. In each experiment the induction heads
    read the real stream only along the paths named and the other input's stream along every other path:

        value      the value reads the token's own embedding (and what the first layer wrote, from `other`)
        query      the query reads the token's own embedding
        key        the key reads what the previous-token head wrote (the embedding and the other first-layer
                   heads from `other`)
        all        the three together: the minimal induction circuit
        none       all three inputs from `other`: the heads tell nothing of the real sequence. Its cost is the
                   whole effect of the heads

    The loss is the mean over the repeated tokens of `reference`. Share explained = 1 - (cost of the experiment)
    / (cost of `none`), a cost being the loss less the clean loss. Returned with the losses themselves.
    """
    names = ["embed", "attn.0.result", "resid_post.0"]
    _, ref = model.run_with_cache(reference, names=names)
    _, alt = model.run_with_cache(other, names=names)
    real = ref["resid_post.0"]
    embedding_only = ref["embed"] + (alt["resid_post.0"] - alt["embed"])  # the token itself; the first layer from `other`
    through_previous = alt["resid_post.0"] - alt["attn.0.result"][:, previous] + ref["attn.0.result"][:, previous]
    experiments = {
        "clean": dict(),
        "none": dict(query=alt["resid_post.0"], key=alt["resid_post.0"], value=alt["resid_post.0"]),
        "value": dict(value=embedding_only),
        "query": dict(query=embedding_only),
        "key": dict(key=through_previous),
        "all": dict(query=embedding_only, key=through_previous, value=embedding_only),
    }
    targets = reference[:, 1:]
    loss = {}
    for name, inputs in experiments.items():
        logits = patch_inputs(model, reference, 1, induction, **inputs)
        by_position = -F.log_softmax(logits[:, :-1].float(), dim=-1).gather(2, targets[:, :, None]).squeeze(2)
        loss[name] = repeated_loss(by_position, length, repeats)["repeated"]
    effect = loss["none"] - loss["clean"]
    out = {f"loss.{name}": value for name, value in loss.items()}
    for name in ("value", "query", "key", "all"):
        out[name] = 1 - (loss[name] - loss["clean"]) / effect if effect > 0 else float("nan")
    return out


# ----------------------------------------------------------------------------------------------------------------
# Scores measured on activations


def token_pool(counts: Tensor, drop_most: int = 100, min_count: int = 1000) -> Tensor:
    """Token ids to draw random sequences from: IH draws them "excluding the most common and the least common
    tokens". The `drop_most` most frequent are left out, and every token seen fewer than `min_count` times in
    training (a token the model has hardly met has no embedding to match)."""
    counts = torch.as_tensor(counts)
    order = counts.argsort(descending=True)
    keep = order[drop_most:]
    return keep[counts[keep] >= min_count].sort().values


def repeated_random_tokens(n: int, length: int, repeats: int, pool: Tensor, bos_id: int, seed: int = 0) -> Tensor:
    """n sequences (n, 1 + length * repeats): the beginning-of-text token, then `length` tokens drawn without
    replacement from the pool, the same block `repeats` times. On such a sequence nothing but the earlier copy
    predicts the next token (IH, "Head activation evaluators": 25 tokens, 4 times)."""
    generator = torch.Generator().manual_seed(seed)
    pool = torch.as_tensor(pool)
    blocks = torch.stack([pool[torch.randperm(len(pool), generator=generator)[:length]] for _ in range(n)])
    return torch.cat((torch.full((n, 1), bos_id, dtype=blocks.dtype), blocks.repeat(1, repeats)), dim=1)


def prefix_matching_score(pattern: Tensor, length: int, repeats: int) -> Tensor:
    """IH's prefix-matching score of each head, (H,), from patterns (B, H, T, T) on repeated random tokens.

    For every destination in the second and later repeats, add the attention it pays to the token that FOLLOWED
    each earlier occurrence of its own token (position i - k * length + 1 for k = 1, 2, ...), and average over
    destinations and sequences. An induction head scores near 1; a head with a uniform pattern near
    (repeats - 1) / mean context, a few hundredths. IH's text is ambiguous about which neighbour of the earlier
    occurrence is meant; its figure, FW's mechanism and the usual implementations agree on the one after.
    """
    total = torch.zeros(pattern.shape[:2], dtype=torch.float64, device=pattern.device)
    count = 0
    for i in range(1 + length, 1 + length * repeats):
        earlier = [i - k * length + 1 for k in range(1, repeats) if i - k * length >= 1]
        total += pattern[:, :, i, earlier].double().sum(dim=-1)
        count += 1
    return (total / count).mean(dim=0)


def previous_token_score(pattern: Tensor) -> Tensor:
    """IH's previous-token score of each head, (H,): the mean attention from token i to token i - 1, over the
    positions i >= 1 of patterns (B, H, T, T) measured on text."""
    return pattern.diagonal(offset=-1, dim1=-2, dim2=-1).double().mean(dim=(0, 2))


@torch.no_grad()
def loss_by_position(model: GPT, idx: Tensor, hooks: Optional[dict[str, HookFn]] = None) -> Tensor:
    """-log p(token i + 1 | tokens up to i) for i = 0 .. T - 2, (B, T - 1), in nats; with `hooks`, under an
    intervention (run_with_hooks)."""
    logits = model.run_with_hooks(idx, hooks) if hooks else model(idx)
    logp = F.log_softmax(logits[:, :-1].float(), dim=-1)
    return -logp.gather(2, idx[:, 1:, None]).squeeze(2)


def in_context_learning_score(losses: Tensor, early: int, late: int) -> float:
    """The in-context learning score from losses (B, T - 1): the mean loss on the `early`-th token of the context
    minus the mean loss on the `late`-th, positive when the later token is the easier one: the more positive, the
    more the model gains from its context. IH takes the 50th and the 500th token of 512 and states the difference
    with both signs; this is the positive one. Tokens are counted from one, and the loss on the n-th token is
    column n - 2 (the first token is never predicted)."""
    return (losses[:, early - 2].double().mean() - losses[:, late - 2].double().mean()).item()


def repeated_loss(losses: Tensor, length: int, repeats: int) -> dict[str, float]:
    """The mean loss on the two kinds of token of repeated random sequences, from losses (B, T - 1):

        first      the tokens of the first block after its first: random, nothing predicts them
        repeated   the tokens of the later blocks whose predecessor has been seen before, which is all of them
                   but the first token of the second block: the earlier copy predicts each exactly
    """
    first = losses[:, 1:length]  # targets at positions 2 .. length
    repeated = losses[:, length + 1:length * repeats]  # targets at positions length + 2 .. length * repeats
    return {"first": first.double().mean().item(), "repeated": repeated.double().mean().item()}


def repeat_statistics(windows: Tensor, exclude: Sequence[int] = ()) -> dict[str, float]:
    """How much a text repeats itself in the way an induction head can use, from windows (B, T) of it:

        seen        the share of positions (all but the last of each window) whose token occurred earlier in the window
        continues   among those, the share whose next token is the one that followed the latest earlier occurrence:
                    the cases [A][B] ... [A] -> [B] that induction predicts

    Tokens in `exclude` (the beginning-of-text token) are never counted as the current token.
    """
    seen = continues = total = 0
    for row in windows.tolist():
        last: dict[int, int] = {}
        for i, token in enumerate(row[:-1]):
            total += 1
            if token in last and token not in exclude:
                seen += 1
                continues += row[last[token] + 1] == row[i + 1]
            last[token] = i
    return {"seen": seen / total, "continues": continues / max(seen, 1)}


def rise_interval(tokens: Sequence[float], values: Sequence[float], low: float = 0.1, high: float = 0.9) -> dict:
    """When a score that starts at values[0] and ends at values[-1] makes its change, read off checkpoints:
    the first checkpoint at which it has covered the fraction `low` of the way and the first at which it has
    covered `high`, as {"start", "end" (in the units of `tokens`), "factor" (end / start), "from", "to" (the
    score at the two ends)}. A change within a factor of two in training time is abrupt on a log axis; one
    spread over a factor of ten is gradual."""
    first, final = values[0], values[-1]

    def reached(fraction: float) -> float:
        level = first + fraction * (final - first)
        for t, v in zip(tokens, values):
            if (v - level) * (final - first) >= 0 and t > 0:
                return t
        return tokens[-1]

    start, end = reached(low), reached(high)
    return {"start": start, "end": end, "factor": end / start if start > 0 else float("inf"), "from": first, "to": final}


# ----------------------------------------------------------------------------------------------------------------
# Ablations


@torch.no_grad()
def mean_head_outputs(model: GPT, idx: Tensor) -> list[Tensor]:
    """The mean over sequences of `attn.l.z` at every position: a list of (H, T, d_head), one a layer. What a
    head is replaced by under mean ablation: its average output on inputs of the same kind, which keeps the
    stream on its usual scale and removes what the head says about this input."""
    _, cache = model.run_with_cache(idx, names=["attn.*.z"])
    return [cache[f"attn.{layer}.z"].mean(dim=0) for layer in range(model.cfg.n_layer)]


def ablation_hooks(heads: Sequence[Head], mode: str = "zero", means: Optional[Sequence[Tensor]] = None,
                   patterns: Optional[Sequence[Tensor]] = None) -> dict[str, HookFn]:
    """Hooks for run_with_hooks or run_with_cache that remove heads.

        mode "zero"   the head's output is replaced by zeros
        mode "mean"   by its mean output at each position (`means` from mean_head_outputs)

    Without `patterns` the ablation is full: every later layer recomputes its attention from the changed stream.
    With `patterns` (a list of (B, H, T, T), one a layer, recorded on the same input) every pattern is forced to
    its recorded value, IH's "pattern-preserving" ablation: only what flows through values changes, and in an
    attention-only model that subtracts exactly the paths through the ablated heads.
    """
    if mode not in ("zero", "mean"):
        raise ValueError(f"mode must be 'zero' or 'mean', not {mode!r}")
    if mode == "mean" and means is None:
        raise ValueError("mean ablation needs `means` (mean_head_outputs)")
    by_layer: dict[int, list[int]] = {}
    for layer, head in heads:
        by_layer.setdefault(layer, []).append(head)
    hooks: dict[str, HookFn] = {}

    def remove(layer: int, which: list[int]) -> HookFn:
        def hook(z: Tensor) -> Tensor:
            z = z.clone()
            for head in which:
                z[:, head] = 0 if mode == "zero" else means[layer][head, :z.size(2)].to(z.dtype)
            return z
        return hook

    def force(pattern: Tensor) -> HookFn:
        return lambda _: pattern

    for layer, which in by_layer.items():
        hooks[f"attn.{layer}.z"] = remove(layer, which)
    if patterns is not None:
        for layer, pattern in enumerate(patterns):
            hooks[f"attn.{layer}.pattern"] = force(pattern)
    return hooks


# ----------------------------------------------------------------------------------------------------------------
# All the scores of a model, in one call


@torch.no_grad()
def head_scores(model: GPT, text: Tensor, pool: Tensor, bos_id: int, length: int = 25, repeats: int = 4,
                n_sequences: int = 64, seed: int = 0, copy_sample: int = 25, copy_draws: int = 40) -> dict:
    """Every per-head score of this module for every head of a model: a dictionary of lists indexed
    [layer][head] (and `repeated_loss`, `first_loss` for the model as a whole).

        prefix_matching   on `n_sequences` sequences of `length` random tokens repeated `repeats` times
        previous_token    on the windows of `text` (B, T)
        copying_statistic the eigenvalue summary of the OV circuit
        copying_score     our weight-only version of IH's copying score, averaged over `copy_draws` samples of
                          `copy_sample` tokens of the pool
    """
    device = model.embed.weight.device
    sequences = repeated_random_tokens(n_sequences, length, repeats, pool, bos_id, seed).to(device)
    on_random = attention_patterns(model, sequences)
    on_text = attention_patterns(model, text.to(device))
    generator = torch.Generator().manual_seed(seed + 1)
    samples = [torch.as_tensor(pool)[torch.randperm(len(pool), generator=generator)[:copy_sample]]
               for _ in range(copy_draws)]
    out: dict = {name: [] for name in ("prefix_matching", "previous_token", "copying_statistic", "copying_score")}
    for layer in range(model.cfg.n_layer):
        out["prefix_matching"].append(prefix_matching_score(on_random[layer], length, repeats).tolist())
        out["previous_token"].append(previous_token_score(on_text[layer]).tolist())
        out["copying_statistic"].append([copying_statistic(ov_eigenvalues(model, layer, head))
                                         for head in range(model.cfg.n_head)])
        out["copying_score"].append([sum(copying_score(model, layer, head, sample) for sample in samples) / copy_draws
                                     for head in range(model.cfg.n_head)])
    losses = repeated_loss(loss_by_position(model, sequences), length, repeats)
    out["repeated_loss"], out["first_loss"] = losses["repeated"], losses["first"]
    return out
