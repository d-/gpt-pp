"""The logit lens: the model's own readout, applied to the residual stream before the last layer.

A pre-norm transformer keeps one stream from the embedding to the end, and each block adds to it. The prediction
is read from the stream after the last block: final normalization, unembedding, the cap on the logits if the
model has one, softmax. The logit lens (nostalgebraist, "interpreting GPT: the logit lens", 2020) applies that
same readout to the stream after the embedding and after every earlier block:

    lens_l(x) = softmax(cap(W_U N(x_l)))        l = 0 (after the embedding), 1 ... L (after block l)

It is the model with every later block's contribution set to zero, so at depth L it is the model's own
distribution. What it answers: if the computation stopped here, what would the readout say? It does not show how
the answer is computed, and an empty lens is not an empty stream: nothing obliges an early block to keep
information where the unembedding reads it. The tuned lens (Belrose et al., 2023, arXiv:2303.08112) follows the
drift of the stream's basis: an affine map per depth, started at the identity and fitted, with the model frozen,
to minimize D(final || lens). Here the map acts on the stream as every block of a pre-norm model reads it,
normalized: tuned_l(x) = softmax(cap(W_U N(A_l N(x_l) + b_l))). With a normalization that has no parameters this
is the paper's lens up to the scale of the bias.

    fit_tuned_lens(model, windows)       a tuned lens: one affine map per depth, fitted to the final distribution
    tuned_statistics(model, windows, lens)   its relative entropy and top-token agreement by depth on other text
    streams(model, idx)                  the stream at every depth, (L + 1, B, T, d)
    readout(model, x)                    log-probabilities of the model's readout applied to any stream (..., d)
    tuned_table(model, idx, lens)        lens_table through a tuned lens
    lens_table(model, idx, top=5)        one passage: at every depth and position the top tokens, the probability
                                         and the rank the lens gives the model's final prediction, the relative
                                         entropy from the final distribution, and the depth at which each
                                         position settles
    lens_statistics(model, windows)      the same quantities averaged over many held-out positions, by depth

Conventions. Depth 0 is the stream after the embedding (and its normalization, if the model has one); depth l is
the stream after block l, counted from one, which is the hook `resid_post.{l - 1}`. The final prediction of a
position is the token the model itself ranks first there (the argmax at depth L). A position settles at the first
depth from which the lens's top token stays the final prediction up to the end; that depth is L at the latest.
The relative entropy is D(final || lens) in bits: the direction in which a lens is fitted and reported (Belrose
et al.), zero at depth L.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from ..model import GPT

LN2 = math.log(2.0)


def depth_names(model: GPT) -> list[str]:
    """The hook of the stream at each depth: after the embedding, then after each block."""
    return ["embed"] + [f"resid_post.{l}" for l in range(model.cfg.n_layer)]


@torch.no_grad()
def streams(model: GPT, idx: Tensor) -> Tensor:
    """The residual stream of idx (B, T) at every depth, (L + 1, B, T, d)."""
    names = depth_names(model)
    _, cache = model.run_with_cache(idx, names=names)
    return torch.stack([cache[name] for name in names])


def readout(model: GPT, x: Tensor) -> Tensor:
    """Log-probabilities (..., V) of the model's own readout applied to a stream x (..., d): the final
    normalization, the unembedding, the soft-cap on the logits when the model has one, the log-softmax."""
    return F.log_softmax(model.unembed_logits(model.final_norm(x)), dim=-1)


def settling_depth(on_top: Tensor) -> Tensor:
    """on_top (D, ...) says at each depth whether the lens's top token is the final prediction (true at the last
    depth by definition). Returns (...) the first depth from which it stays true to the end."""
    depth = on_top.size(0)
    # the number of consecutive depths, counted back from the end, at which it holds
    run = torch.cumprod(on_top.flip(0).long(), dim=0).sum(0)
    return depth - run


def _table(model: GPT, x: Tensor, top: int, marginal: Tensor | None = None) -> dict[str, Tensor]:
    """x (D, N, d), the stream of N positions at every depth -> the lens's quantities at every depth and position.
    One depth is read out at a time, so the memory is that of a few (N, V) arrays. `marginal` (D, V), if given,
    accumulates the sum over positions of each depth's distribution."""
    final = readout(model, x[-1]).float()                          # (N, V): the model's own distribution
    best_final = final.topk(top, dim=-1)
    final_id, p_model = best_final.indices[:, 0], final.exp()
    rows: dict[str, list[Tensor]] = {"top_ids": [], "top_p": [], "p_final": [], "rank_final": [], "kl_bits": []}
    for depth in range(x.size(0)):
        last = depth == x.size(0) - 1
        logp = final if last else readout(model, x[depth]).float()
        best = best_final if last else logp.topk(top, dim=-1)
        logp_final = logp.gather(-1, final_id.unsqueeze(-1))       # (N, 1)
        rank = torch.count_nonzero(logp > logp_final, dim=-1)
        # two tokens can tie for the top in float32 where the lens is nearly flat: the lens's top token is then the
        # one it lists first, and the other has rank 1
        rank = torch.where((rank == 0) & (best.indices[:, 0] != final_id), torch.ones_like(rank), rank)
        rows["top_ids"].append(best.indices)
        rows["top_p"].append(best.values.exp())
        rows["p_final"].append(logp_final.squeeze(-1).exp())
        rows["rank_final"].append(rank)                             # 0: it is the lens's top token
        rows["kl_bits"].append((p_model * (final - logp)).sum(-1) / LN2)            # D(final || lens)
        if marginal is not None:
            marginal[depth] += logp.exp().sum(0).double()
    out = {key: torch.stack(value) for key, value in rows.items()}
    on_top = out["rank_final"] == 0
    out["final_id"] = final_id
    out["settles"] = settling_depth(on_top)
    out["first_top"] = on_top.long().argmax(0)                     # the last depth is always on top
    return out


@torch.no_grad()
def lens_table(model: GPT, idx: Tensor, top: int = 5) -> dict[str, np.ndarray]:
    """The lens on one passage idx (T,) or (1, T). With D = L + 1 depths:

        top_ids, top_p   (D, T, top)   the lens's `top` likeliest next tokens at each depth and position
        final_id         (T,)          the model's own prediction at each position (the top token at depth L)
        p_final          (D, T)        the probability each depth gives that prediction
        rank_final       (D, T)        how many tokens the depth ranks above it (0: it is the lens's top token)
        kl_bits          (D, T)        D(final || lens) in bits
        settles          (T,)          the first depth from which the lens's top token stays the final prediction
        first_top        (T,)          the first depth at which it is the top token at all
    """
    idx = idx.view(1, -1)
    return {key: value.cpu().numpy() for key, value in _table(model, streams(model, idx)[:, 0], top).items()}


@torch.no_grad()
def lens_statistics(model: GPT, windows: Tensor, skip: int = 1, batch: int = 8) -> dict[str, np.ndarray | int | float]:
    """The lens averaged over held-out text: windows (N, T) of tokens, every position from `skip` on (the first
    token of a window has no context to read). By depth, with D = L + 1:

        kl_bits       (D,)   mean D(final || lens) in bits            kl_se   its standard error over windows
        top1          (D,)   share of positions at which the lens's top token is the final prediction
        p_final       (D,)   mean probability the lens gives the final prediction
        bias_bits     (D,)   D(mean final || mean lens): how far the lens's marginal distribution over tokens is
                             from the model's own (the bias of Belrose et al., Section 2)
        settle_hist   (D,)   how many positions settle at each depth     settle_median   the median settling depth
        positions, windows   the counts
    """
    n_depth, device = model.cfg.n_layer + 1, model.embed.weight.device
    kl_by_window, top1, p_final, hist = [], torch.zeros(n_depth), torch.zeros(n_depth), torch.zeros(n_depth)
    marginal = torch.zeros(n_depth, model.cfg.vocab_size, dtype=torch.float64, device=device)
    settle_all, positions = [], 0
    for start in range(0, windows.size(0), batch):
        x = streams(model, windows[start:start + batch].to(device))[:, :, skip:]     # (D, B, T - skip, d)
        B, T = x.shape[1], x.shape[2]
        table = _table(model, x.reshape(n_depth, B * T, -1), top=1, marginal=marginal)
        kl_by_window.append(table["kl_bits"].view(n_depth, B, T).mean(-1).cpu())    # (D, B)
        top1 += (table["rank_final"] == 0).float().sum(-1).cpu()
        p_final += table["p_final"].sum(-1).cpu()
        hist += torch.bincount(table["settles"], minlength=n_depth).float().cpu()
        settle_all.append(table["settles"].cpu())
        positions += B * T
    kl = torch.cat(kl_by_window, dim=1)                                              # (D, N)
    marginal = marginal.cpu() / positions
    bias = (marginal[-1] * (marginal[-1].clamp_min(1e-30).log() - marginal.clamp_min(1e-30).log())).sum(-1) / LN2
    n = kl.size(1)
    return {"kl_bits": kl.mean(1).numpy(), "kl_se": (kl.std(1) / math.sqrt(n)).numpy() if n > 1 else np.zeros(n_depth),
            "top1": (top1 / positions).numpy(), "p_final": (p_final / positions).numpy(), "bias_bits": bias.numpy(),
            "settle_hist": hist.numpy(), "settle_median": float(np.median(torch.cat(settle_all).numpy())),
            "positions": positions, "windows": n}


def _unit(x: Tensor) -> Tensor:
    """A stream as a pre-norm block reads it: divided by the root mean square of its coordinates."""
    return F.rms_norm(x.float(), (x.size(-1),), eps=1e-6)


def tuned_readout(model: GPT, x: Tensor, a: Tensor, b: Tensor) -> Tensor:
    """Log-probabilities of the tuned lens of one depth: the model's readout of A N(x) + b, for x (..., d)."""
    return readout(model, _unit(x) @ a.T + b)


def tuned_streams(x: Tensor, tuned: dict[str, Tensor]) -> Tensor:
    """x (D, N, d), the stream at every depth -> what the tuned lens hands to the readout at every depth: each
    depth's stream, normalized, through its map; the last depth as it is."""
    mapped = [_unit(x[depth]) @ tuned["a"][depth].T + tuned["b"][depth] for depth in range(x.size(0) - 1)]
    return torch.stack(mapped + [x[-1].float()])


@torch.no_grad()
def tuned_table(model: GPT, idx: Tensor, tuned: dict[str, Tensor], top: int = 5) -> dict[str, np.ndarray]:
    """lens_table through a tuned lens: the same quantities, each depth read through its fitted map."""
    idx = idx.view(1, -1)
    return {key: value.cpu().numpy() for key, value in _table(model, tuned_streams(streams(model, idx)[:, 0], tuned), top).items()}


def fit_tuned_lens(model: GPT, windows: Tensor, steps: int = 200, batch: int = 8, lr: float = 1e-3, skip: int = 1,
                   seed: int = 0) -> dict[str, Tensor]:
    """A tuned lens for depths 0 to L - 1: `a` (L, d, d) and `b` (L, d), each map started at the identity and
    fitted by Adam (learning rate decaying linearly to zero over `steps` steps of `batch` windows) to minimize the
    mean over positions of D(final || lens) in nats. The model is frozen; windows (N, T) are token ids, positions
    before `skip` are not used. Returns also `loss` (steps, L): the relative entropy in bits by step and depth."""
    device, L, d = model.embed.weight.device, model.cfg.n_layer, model.cfg.d_model
    a = torch.eye(d, device=device).repeat(L, 1, 1).requires_grad_(True)
    b = torch.zeros(L, d, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([a, b], lr=lr)
    frozen = [(p, p.requires_grad) for p in model.parameters()]
    for p, _ in frozen:
        p.requires_grad_(False)
    generator = torch.Generator().manual_seed(seed)
    history = []
    try:
        for step in range(steps):
            rows = torch.randint(0, windows.size(0), (batch,), generator=generator)
            with torch.no_grad():
                x = streams(model, windows[rows].to(device))[:, :, skip:].reshape(L + 1, -1, d)
                final = readout(model, x[-1]).float()
                p_final = final.exp()
            for group in optimizer.param_groups:
                group["lr"] = lr * (1 - step / steps)
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for depth in range(L):                                    # one depth at a time: a (positions, V) graph each
                loss = (p_final * (final - tuned_readout(model, x[depth], a[depth], b[depth]).float())).sum(-1).mean()
                loss.backward()
                losses.append(float(loss.detach()) / LN2)
            optimizer.step()
            history.append(losses)
    finally:
        for p, flag in frozen:
            p.requires_grad_(flag)
    return {"a": a.detach(), "b": b.detach(), "loss": torch.tensor(history)}


@torch.no_grad()
def tuned_statistics(model: GPT, windows: Tensor, tuned: dict[str, Tensor], skip: int = 1, batch: int = 8) -> dict[str, np.ndarray]:
    """The tuned lens on text it was not fitted on, by depth (the last depth is the model): `kl_bits` (D,) the
    mean D(final || lens) in bits, `kl_se` its standard error over windows, `top1` (D,) the share of positions at
    which the lens's top token is the model's final prediction."""
    device, L, d = model.embed.weight.device, model.cfg.n_layer, model.cfg.d_model
    kl_by_window, top1, positions = [], torch.zeros(L + 1), 0
    for start in range(0, windows.size(0), batch):
        x = streams(model, windows[start:start + batch].to(device))[:, :, skip:]
        B, T = x.shape[1], x.shape[2]
        final = readout(model, x[-1].reshape(B * T, d)).float()
        final_id, p_final = final.argmax(-1), final.exp()
        rows = []
        for depth in range(L + 1):
            logp = final if depth == L else tuned_readout(model, x[depth].reshape(B * T, d), tuned["a"][depth], tuned["b"][depth]).float()
            rows.append(((p_final * (final - logp)).sum(-1) / LN2).view(B, T).mean(-1).cpu())
            top1[depth] += (logp.argmax(-1) == final_id).float().sum().cpu()
        kl_by_window.append(torch.stack(rows))
        positions += B * T
    kl = torch.cat(kl_by_window, dim=1)
    n = kl.size(1)
    return {"kl_bits": kl.mean(1).numpy(), "kl_se": (kl.std(1) / math.sqrt(n)).numpy() if n > 1 else np.zeros(L + 1),
            "top1": (top1 / positions).numpy()}
