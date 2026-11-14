"""Activation patching and direct logit attribution on a clean and a corrupted prompt.

Two prompts that differ in one token and in their answer: "The capital of France is" (Paris) and "The capital of
Italy is" (Rome). The metric is the logit difference at the last position,

    LD = logit(clean answer) - logit(corrupted answer),

positive on the clean prompt and negative on the corrupted one. It is taken before the model's cap on its logits,
where it is a linear function of the normalized final stream: LD = (W_U[a] - W_U[b]) . N(x_L).

Activation patching (Vig et al., 2020; Geiger et al., 2021; Meng et al., 2022; practice in Zhang and Nanda, 2023,
and Heimersheim and Nanda, 2024). Run the model on one prompt with one node overwritten by its value on the
other, and report where the metric lands between the two runs:

    effect = (LD_patched - LD_corrupted) / (LD_clean - LD_corrupted)        0: as corrupted, 1: as clean

A node is the stream entering a layer at one position (`resid_pre`), or the output of a layer's attention or MLP
at one position (`attn_out`, `mlp_out`). Denoising writes clean values into the corrupted run and finds what is
sufficient to restore the answer; noising writes corrupted values into the clean run and finds what is necessary.

Direct logit attribution. The final stream is a sum, x_L = embedding + sum of what each head and each MLP wrote,
and the final normalization divides it by a scale rho that depends on the whole sum. With rho frozen at its value
on this input LD is linear in the sum, so each component has a share

    DLA_c = (W_U[a] - W_U[b]) . o_c / rho

and the shares of all components and of the embedding add up to LD exactly. A share is the last edge of every
path and no more: a head that matters by steering a later head has none.

    logit_difference(model, idx, a, b, hooks)      LD at the last position, one value a row of idx
    patch_map(model, source, target, a, b, site)   LD with node (layer, position) of the run on `target` overwritten
                                                   by its value on `source`, for every layer and position, (L, T)
    patching_effects(model, clean, corrupt, a, b)  both directions, the three sites, normalized as above
    direct_attribution(model, idx, a, b)           the shares: heads (L, H), MLPs (L,), embedding; their sum; rho
    attention_from(model, idx)                     every head's attention from the last position, (L, H, T)
    head_sources(model, idx, a, b)                 each head's share split by the position it read, (L, H, T)

Layers and heads are counted from zero here, as the hooks are; positions are positions of idx, `<|bos|>` included
if idx has it.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from ..model import GPT, LayerNorm, RMSNorm

SITES = ("resid_pre", "attn_out", "mlp_out")


def answer_direction(model: GPT, a: int, b: int) -> Tensor:
    """W_U[a] - W_U[b], (d,): the direction of the normalized final stream that LD reads."""
    w = model.unembed.weight
    return (w[a] - w[b]).float()


@torch.no_grad()
def logit_difference(model: GPT, idx: Tensor, a: int, b: int, hooks: Optional[dict] = None, pos: int = -1) -> Tensor:
    """logit(a) - logit(b) at position `pos` of each row of idx (B, T), before the cap on the logits, (B,).
    `hooks` are applied as in GPT.run_with_hooks, so this is also the metric of a patched run."""
    _, cache = model.run_with_cache(idx, names=["final_norm"], hooks=hooks)
    return cache["final_norm"][:, pos].float() @ answer_direction(model, a, b)


@torch.no_grad()
def patch_map(model: GPT, source: Tensor, target: Tensor, a: int, b: int, site: str = "resid_pre",
              batch: int = 64) -> Tensor:
    """LD of the run on `target` (T,) when the node `site`.l at position t is overwritten by its value in the run
    on `source` (T,), for every layer l and position t: (L, T). One row of a batch a node, so a map costs
    L * T / batch forward passes."""
    source, target = source.view(1, -1), target.view(1, -1)
    if source.shape != target.shape:
        raise ValueError(f"the two prompts must have the same number of tokens: {source.size(1)} and {target.size(1)}")
    L, T = model.cfg.n_layer, source.size(1)
    names = [f"{site}.{l}" for l in range(L)]
    _, cache = model.run_with_cache(source, names=names)
    cells = [(l, t) for l in range(L) for t in range(T)]
    out = torch.empty(L * T)
    for start in range(0, len(cells), batch):
        chunk = cells[start:start + batch]
        hooks = {}
        for l in sorted({l for l, _ in chunk}):
            rows = torch.tensor([r for r, (cl, _) in enumerate(chunk) if cl == l])
            at = torch.tensor([t for cl, t in chunk if cl == l])

            def overwrite(x: Tensor, rows: Tensor = rows, at: Tensor = at, name: str = names[l]) -> Tensor:
                x = x.clone()
                rows, at = rows.to(x.device), at.to(x.device)
                x[rows, at] = cache[name][0, at].to(x.dtype)
                return x

            hooks[names[l]] = overwrite
        out[start:start + len(chunk)] = logit_difference(model, target.expand(len(chunk), -1), a, b, hooks).cpu()
    return out.view(L, T)


@torch.no_grad()
def patching_effects(model: GPT, clean: Tensor, corrupt: Tensor, a: int, b: int, sites: tuple[str, ...] = SITES,
                     batch: int = 64) -> dict:
    """Both directions of patching at every (layer, position), for each site, normalized so that 0 is the
    corrupted run and 1 the clean one:

        ld_clean, ld_corrupt     the metric on the two prompts (floats)
        denoise[site]  (L, T)    clean values written into the corrupted run: the share of the answer restored
        noise[site]    (L, T)    corrupted values written into the clean run: the share of the answer that is left
    """
    ld_clean = float(logit_difference(model, clean.view(1, -1), a, b))
    ld_corrupt = float(logit_difference(model, corrupt.view(1, -1), a, b))
    gap = ld_clean - ld_corrupt
    out = {"ld_clean": ld_clean, "ld_corrupt": ld_corrupt, "denoise": {}, "noise": {}}
    for site in sites:
        out["denoise"][site] = (patch_map(model, clean, corrupt, a, b, site, batch) - ld_corrupt) / gap
        out["noise"][site] = (patch_map(model, corrupt, clean, a, b, site, batch) - ld_corrupt) / gap
    return out


def frozen_readout(model: GPT, x_final: Tensor, a: int, b: int) -> tuple[Tensor, float, float]:
    """The final normalization at the stream x_final (d,), frozen: (u, constant, rho) with LD = u . x + constant
    for every x, when the normalization keeps the scale rho it has at x_final. For an RMS norm u is the answer
    direction (times the gain) over rho; a layer norm also centers, and its bias gives the constant."""
    direction, norm = answer_direction(model, a, b), model.final_norm
    x = x_final.float()
    if isinstance(norm, RMSNorm):
        rho = float((x.square().mean() + 1e-6).sqrt())
        gain = 1.0 if norm.gain is None else norm.gain.float()
        return direction * gain / rho, 0.0, rho
    if isinstance(norm, LayerNorm):
        rho = float((x.var(unbiased=False) + 1e-5).sqrt())
        u = direction * norm.gain.float() / rho
        constant = 0.0 if norm.bias is None else float(direction @ norm.bias.float())
        return u - u.mean(), constant, rho
    if isinstance(norm, torch.nn.Identity) and model.cfg.norm_placement == "pre":
        return direction, 0.0, 1.0
    raise NotImplementedError("direct attribution needs a pre-norm model whose final normalization is linear once "
                              "its scale is frozen (RMS norm, layer norm or none)")


@torch.no_grad()
def direct_attribution(model: GPT, idx: Tensor, a: int, b: int, pos: int = -1) -> dict:
    """The share of each component in LD = logit(a) - logit(b) at position `pos` of idx (T,), the scale of the
    final normalization frozen:

        heads  (L, H)    what each head wrote at the position           mlp  (L,)   what each MLP wrote there
        embed  float     the embedding's share                          bias float  output biases and the norm's bias
        total  float     their sum, which is LD before the cap          rho  float  the frozen scale
    """
    idx = idx.view(1, -1)
    L = model.cfg.n_layer
    names = ["embed", "final_norm"] + [f"attn.{l}.result" for l in range(L)] + [f"attn_out.{l}" for l in range(L)]
    names += [f"mlp_out.{l}" for l in range(L)] if model.cfg.use_mlp else []
    names += [f"resid_post.{L - 1}"]
    _, cache = model.run_with_cache(idx, names=names)
    u, constant, rho = frozen_readout(model, cache[f"resid_post.{L - 1}"][0, pos], a, b)
    heads = torch.stack([cache[f"attn.{l}.result"][0, :, pos].float() @ u for l in range(L)])          # (L, H)
    attn = torch.stack([cache[f"attn_out.{l}"][0, pos].float() @ u for l in range(L)])                 # (L,)
    mlp = torch.stack([cache[f"mlp_out.{l}"][0, pos].float() @ u for l in range(L)]) if model.cfg.use_mlp else torch.zeros(L)
    embed = float(cache["embed"][0, pos].float() @ u)
    bias = float((attn - heads.sum(1)).sum()) + constant             # attention's output bias, if the model has one
    total = embed + float(heads.sum()) + float(mlp.sum()) + bias
    return {"heads": heads.cpu(), "mlp": mlp.cpu(), "embed": embed, "bias": bias, "total": total, "rho": rho,
            "ld": float(cache["final_norm"][0, pos].float() @ answer_direction(model, a, b))}


@torch.no_grad()
def attention_from(model: GPT, idx: Tensor, pos: int = -1) -> Tensor:
    """Every head's attention from position `pos` of idx (T,) over the positions it can see: (L, H, T), each row
    summing to one."""
    idx = idx.view(1, -1)
    _, cache = model.run_with_cache(idx, names=["attn.*.pattern"])
    return torch.stack([cache[f"attn.{l}.pattern"][0, :, pos].float() for l in range(model.cfg.n_layer)]).cpu()


@torch.no_grad()
def head_sources(model: GPT, idx: Tensor, a: int, b: int, pos: int = -1) -> Tensor:
    """Each head's share of LD split by the position it read: (L, H, T), entry (l, h, j) being the attention of
    head h of layer l from `pos` to j times what the value at j, through the head's output matrix, adds to LD.
    Summed over j it is direct_attribution's `heads`."""
    idx = idx.view(1, -1)
    L, H = model.cfg.n_layer, model.cfg.n_head
    names = [f"attn.{l}.pattern" for l in range(L)] + [f"attn.{l}.v" for l in range(L)] + [f"resid_post.{L - 1}"]
    _, cache = model.run_with_cache(idx, names=names)
    u, _, _ = frozen_readout(model, cache[f"resid_post.{L - 1}"][0, pos], a, b)
    out = []
    for l in range(L):
        attn = model.blocks[l].attn
        _, v = attn.share(cache[f"attn.{l}.v"], cache[f"attn.{l}.v"])                 # (1, H, T, dh)
        w_o = attn.o.weight.float().view(-1, H, attn.d_head)                            # (d, H, dh)
        through = torch.einsum("d,dhk->hk", u, w_o)                                     # (H, dh): u through each head's O
        out.append(cache[f"attn.{l}.pattern"][0, :, pos].float() * torch.einsum("htk,hk->ht", v[0].float(), through))
    return torch.stack(out).cpu()
