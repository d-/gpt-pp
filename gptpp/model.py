"""A GPT in one file, with every component switchable between its historical variants.

Reading order: normalization, position, attention, the MLP, the block, the model. Each variant says in its
docstring where it comes from and why the field moved to it; docs/DESIGN.md collects the choices.

Shapes: B sequences, T tokens, d = d_model, H heads, dh = d_head = d / H, V vocabulary.

The model computes one function along two paths:

    forward            the fast path. Attention is one fused kernel, nothing is recorded, and torch.compile
                       turns the whole thing into a single graph. Training and evaluation use it.
    forward_explicit   attention written out (scores, pattern, per-head outputs), every activation handed to
                       a hook that may record or replace it. Analysis and KV-cached decoding use it.

The activations are named in the residual convention of "A mathematical framework for transformer circuits"
(Elhage et al., 2021): the residual stream is `resid_pre.l`, `resid_mid.l`, `resid_post.l` around the attention
and the MLP of block l, and each sublayer adds `attn_out.l` or `mlp_out.l` to it.
"""

from __future__ import annotations

import fnmatch
import math
import re
from pathlib import Path
from typing import Callable, Iterable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import GATED, ModelConfig

Rotation = Optional[tuple[Tensor, Tensor]]  # (cos, sin) of the rotary angles, each (T, dh / 2)
HookFn = Callable[[Tensor], Optional[Tensor]]  # receives an activation; returns its replacement, or None


# ----------------------------------------------------------------------------------------------------------------
# Normalization


class LayerNorm(nn.Module):
    """Layer normalization (Ba, Kiros and Hinton, 2016, arXiv:1607.06450).

    Each token's vector is centered and divided by its standard deviation over the d coordinates, then scaled
    by a learned gain and shifted by a learned bias. It uses no statistics across the batch or across time,
    which suits sequences of varying length, where batch normalization is awkward; it is the norm of the
    Transformer (2017) and of GPT-2 (2019).
    """

    def __init__(self, d: int, bias: bool):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(d)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(x, (x.size(-1),), self.gain, self.bias, 1e-5)


class RMSNorm(nn.Module):
    """Root-mean-square normalization (Zhang and Sennrich, 2019, arXiv:1910.07467): x / sqrt(mean(x^2)).

    LayerNorm without the centering, which its authors found to contribute little: what matters is that the
    output does not depend on the scale of the input. It is cheaper and is the norm of Llama (2023). With
    `gain=False` it has no parameters at all (`rmsnorm_fixed`): a gain is redundant when a linear layer follows,
    and a layer without parameters needs no place in the optimizer.
    """

    def __init__(self, d: int, gain: bool):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(d)) if gain else None

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), self.gain, eps=1e-6)


class DynamicTanh(nn.Module):
    """Dynamic tanh (Zhu, Chen, He, LeCun and Liu, 2025, arXiv:2503.10622): gain * tanh(alpha * x) + bias.

    No statistics of the activations are computed. The authors observed that trained LayerNorms map their
    input through an S-shaped curve (linear for most coordinates, squashing the few extreme ones) and replaced
    the normalization by that curve, with one learned slope alpha per layer.
    """

    def __init__(self, d: int, alpha: float, bias: bool):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor([alpha]))
        self.gain = nn.Parameter(torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(d)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        x = self.gain * torch.tanh(self.alpha * x)
        return x if self.bias is None else x + self.bias


def make_norm(cfg: ModelConfig) -> nn.Module:
    """The normalization layer chosen by `cfg.norm`; `none` is the identity."""
    if cfg.norm == "layernorm":
        return LayerNorm(cfg.d_model, cfg.bias)
    if cfg.norm == "rmsnorm":
        return RMSNorm(cfg.d_model, gain=True)
    if cfg.norm == "rmsnorm_fixed":
        return RMSNorm(cfg.d_model, gain=False)
    if cfg.norm == "dyt":
        return DynamicTanh(cfg.d_model, cfg.dyt_alpha, cfg.bias)
    return nn.Identity()


# ----------------------------------------------------------------------------------------------------------------
# Position
#
# `learned` (GPT, 2018; GPT-2, 2019) is a free vector per position added to the token embedding: simple, and as
# good as anything at the training length, but a position beyond block_size has no vector. `none` gives the
# model no position signal at all; a causal model can still infer position from the mask, since the number of
# tokens it can see differs at every position (Haviv et al., 2022, arXiv:2203.16634). The other three follow.


def sinusoidal_table(n_positions: int, d: int) -> Tensor:
    """Sinusoidal position vectors (Vaswani et al., 2017, arXiv:1706.03762), (n_positions, d).

    Coordinates 2i and 2i+1 hold the sine and cosine of position / 10000^(2i/d). A shift of the position is a
    rotation of each (sin, cos) pair, a linear map, which was meant to make attention by relative offset easy
    to learn; and the table needs no parameters and extends to any length.
    """
    position = torch.arange(n_positions, dtype=torch.float64).unsqueeze(1)
    frequency = 10000.0 ** (-torch.arange(0, d, 2, dtype=torch.float64) / d)
    table = torch.zeros(n_positions, d, dtype=torch.float64)
    table[:, 0::2] = torch.sin(position * frequency)
    table[:, 1::2] = torch.cos(position * frequency)
    return table.float()


def rotary_table(n_positions: int, d_head: int, base: float) -> tuple[Tensor, Tensor]:
    """Cosines and sines of the rotary angles, each (n_positions, d_head / 2): pair i of a head's coordinates
    turns by position * base^(-2i / d_head), so the wavelengths grow geometrically from 2 pi towards 2 pi base."""
    frequency = base ** (-torch.arange(0, d_head, 2, dtype=torch.float64) / d_head)
    angle = torch.outer(torch.arange(n_positions, dtype=torch.float64), frequency)
    return angle.cos().float(), angle.sin().float()


def rotate(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotary position embedding (Su et al., 2021, arXiv:2104.09864) of queries or keys x (B, H, T, dh).

    The coordinates are taken in pairs (i, i + dh/2) and each pair is rotated by the angle of its position.
    Rotating the query at position m and the key at position n leaves a dot product that depends on m - n
    only, so position enters where it is used, in the attention score, as a relative offset, and the residual
    stream carries no position vector. It is the default of open models since Llama (2023).
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).to(x.dtype)


def alibi_slopes(n_head: int) -> Tensor:
    """Per-head slopes of ALiBi (Press, Smith and Lewis, 2021, arXiv:2108.12409): 2^(-8/H), 2^(-16/H), ..., 2^-8.

    ALiBi adds no position vector either: it subtracts slope * distance from every attention score, a fixed
    recency bias that is strong in some heads and nearly absent in others. Its authors built it to extrapolate:
    a model trained at one length can be run at longer ones.
    """
    return 2.0 ** (-8.0 * torch.arange(1, n_head + 1, dtype=torch.float64) / n_head)


def alibi_table(n_positions: int, n_head: int) -> Tensor:
    """The ALiBi bias of every (query, key) pair, (H, n_positions, n_positions): -slope * (i - j) where the key
    j is not after the query i, and -inf where it is (the causal mask is folded in)."""
    position = torch.arange(n_positions, dtype=torch.float64)
    distance = position[:, None] - position[None, :]
    bias = -alibi_slopes(n_head)[:, None, None] * distance
    return bias.masked_fill(distance < 0, float("-inf")).float()


# ----------------------------------------------------------------------------------------------------------------
# Attention


class KVCache:
    """The keys and values of every layer for the tokens decoded so far.

    Under a causal mask the key and value of a token never change once computed, so decoding one more token
    needs one new query against the stored keys: linear work per token instead of quadratic.
    """

    def __init__(self, n_layer: int):
        self.keys: list[Optional[Tensor]] = [None] * n_layer
        self.values: list[Optional[Tensor]] = [None] * n_layer
        self.length = 0  # tokens whose keys and values are stored

    def extend(self, layer: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Store the keys and values of the new tokens (B, Hkv, T_new, dh); return those of all tokens."""
        if self.keys[layer] is not None:
            k = torch.cat((self.keys[layer], k), dim=2)
            v = torch.cat((self.values[layer], v), dim=2)
        self.keys[layer], self.values[layer] = k, v
        return k, v


class Hooks:
    """What forward_explicit does with each named activation: hand it to a function that may replace it, and
    keep it if it was asked for."""

    def __init__(self, fns: Optional[dict[str, HookFn]] = None, keep: Optional[Callable[[str], bool]] = None):
        self.fns = fns or {}
        self.keep = keep
        self.cache: dict[str, Tensor] = {}

    def wants(self, name: str) -> bool:
        return name in self.fns or (self.keep is not None and self.keep(name))

    def __call__(self, name: str, x: Tensor) -> Tensor:
        fn = self.fns.get(name)
        if fn is not None:
            replacement = fn(x)
            if replacement is not None:
                x = replacement
        if self.keep is not None and self.keep(name):
            self.cache[name] = x
        return x


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention (Vaswani et al., 2017), with the variants that followed.

    Every position emits a query, a key and a value per head; the score of a query against each earlier key,
    through a softmax, decides how much of that key's value is copied to the query's position.

    Grouped-query attention (Ainslie et al., 2023, arXiv:2305.13245; with a single key/value head it is the
    multi-query attention of Shazeer, 2019, arXiv:1911.02150): `n_kv_head` key/value heads are shared by
    groups of query heads. Keys and values are what a decoder caches for every past token, so sharing them
    shrinks the cache and the memory traffic of generation, at a small cost in quality.

    QK normalization (Henry et al., 2020, arXiv:2010.04245): queries and keys are normalized before the dot
    product, so a score is `qk_scale` times the cosine of their angle. The scale is the inverse temperature
    of the softmax: no score exceeds it, and no head can saturate its softmax by growing the norm of its
    weights. It is a constant, the same for every head; with `qk_scale_learned` each head multiplies it by a
    learned factor that starts at one.

    Q, K, V and O are four separate matrices. With AdamW that is the same model as GPT-2's fused matrix; Muon
    orthogonalizes each matrix on its own and works better when they are separate.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_head, self.n_kv_head, self.d_head = cfg.n_head, cfg.kv_heads, cfg.d_head
        self.q = nn.Linear(cfg.d_model, self.n_head * self.d_head, bias=cfg.bias)
        self.k = nn.Linear(cfg.d_model, self.n_kv_head * self.d_head, bias=cfg.bias)
        self.v = nn.Linear(cfg.d_model, self.n_kv_head * self.d_head, bias=cfg.bias)
        self.o = nn.Linear(self.n_head * self.d_head, cfg.d_model, bias=cfg.bias)
        self.qk_norm = cfg.qk_norm
        # Queries and keys normalized to unit root-mean-square have norm sqrt(dh), and scores are divided by
        # sqrt(dh): the score of such a pair is sqrt(dh) * cos. This gain on the query makes it qk_scale * cos.
        self.qk_gain = cfg.qk_scale / math.sqrt(self.d_head)
        self.qk_factor = nn.Parameter(torch.ones(self.n_head)) if cfg.qk_norm and cfg.qk_scale_learned else None
        self.dropout = cfg.dropout

    def project(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Queries (B, H, T, dh), keys and values (B, Hkv, T, dh): three linear maps of the residual stream."""
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_kv_head, self.d_head).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_kv_head, self.d_head).transpose(1, 2)
        return q, k, v

    def position(self, q: Tensor, k: Tensor, rot: Rotation) -> tuple[Tensor, Tensor]:
        """Normalize queries and keys (QK normalization) and turn them by their positions (rotary)."""
        if self.qk_norm:
            q = F.rms_norm(q, (self.d_head,), eps=1e-6) * self.qk_gain
            k = F.rms_norm(k, (self.d_head,), eps=1e-6)
            if self.qk_factor is not None:
                q = q * self.qk_factor.view(-1, 1, 1).to(q.dtype)
        if rot is not None:
            q, k = rotate(q, *rot), rotate(k, *rot)
        return q, k

    def share(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Grouped-query attention: each key/value head serves n_head / n_kv_head query heads."""
        if self.n_kv_head != self.n_head:
            group = self.n_head // self.n_kv_head
            k, v = k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)
        return k, v

    def forward(self, x: Tensor, rot: Rotation, bias: Optional[Tensor]) -> Tensor:
        """The fast path: softmax(q k^T / sqrt(dh) + mask) v in one fused kernel (FlashAttention where it applies).
        `bias` is the ALiBi table, which contains the causal mask; without it the kernel masks causally itself."""
        B, T, _ = x.shape
        q, k, v = self.project(x)
        q, k = self.position(q, k, rot)
        k, v = self.share(k, v)
        z = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None if bias is None else bias.to(q.dtype),
            is_causal=bias is None,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.o(z.transpose(1, 2).reshape(B, T, -1))

    def forward_explicit(self, x: Tensor, rot: Rotation, bias: Tensor, hook: Hooks, layer: int,
                         kv: Optional[KVCache] = None) -> Tensor:
        """The same attention written out, each step handed to `hook`.

        `bias` (T, T_keys) or (H, T, T_keys) is added to the scores: -inf on the keys a query may not see, plus
        the ALiBi term if any. With a KV cache, x holds only the new tokens and the keys and values of the
        earlier ones come from the cache.
        """
        B, T, _ = x.shape
        q, k, v = self.project(x)
        q = hook(f"attn.{layer}.q", q)
        k = hook(f"attn.{layer}.k", k)
        v = hook(f"attn.{layer}.v", v)
        q, k = self.position(q, k, rot)
        if kv is not None:
            k, v = kv.extend(layer, k, v)
        k, v = self.share(k, v)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head) + bias  # (B, H, T, T_keys)
        scores = hook(f"attn.{layer}.scores", scores)
        pattern = hook(f"attn.{layer}.pattern", scores.softmax(dim=-1))  # each row sums to one
        z = pattern if self.dropout == 0 else F.dropout(pattern, self.dropout, self.training)
        z = hook(f"attn.{layer}.z", z @ v)  # (B, H, T, dh): each head's weighted average of values
        if hook.wants(f"attn.{layer}.result"):
            # What each head writes into the residual stream, (B, H, T, d): the output matrix applied to z one
            # head at a time. Their sum over heads (plus the bias) is attn_out. Recorded, not replaceable.
            w_o = self.o.weight.view(-1, self.n_head, self.d_head)
            hook(f"attn.{layer}.result", torch.einsum("bhtk,dhk->bhtd", z, w_o.to(z.dtype)))
        return self.o(z.transpose(1, 2).reshape(B, T, -1))


# ----------------------------------------------------------------------------------------------------------------
# The MLP


def relu(x: Tensor) -> Tensor:
    """Rectified linear unit, max(0, x): the activation of the original Transformer (Vaswani et al., 2017)."""
    return F.relu(x)


def gelu(x: Tensor) -> Tensor:
    """Gaussian error linear unit (Hendrycks and Gimpel, 2016, arXiv:1606.08415): x times the probability that
    a standard normal variable is below x. A smooth ReLU that lets small negative inputs through, so no unit
    has an exactly zero gradient; GPT (2018), BERT (2018) and GPT-2 (2019) made it the default."""
    return F.gelu(x)


def silu(x: Tensor) -> Tensor:
    """SiLU or swish, x * sigmoid(x) (Elfwing et al., 2017, arXiv:1702.03118; found again by automated search
    in Ramachandran et al., 2017, arXiv:1710.05941). Nearly the same curve as GELU; it is the gate of SwiGLU."""
    return F.silu(x)


def relu2(x: Tensor) -> Tensor:
    """Squared ReLU, max(0, x)^2 (So et al., 2021, arXiv:2109.08668, where an evolutionary search over
    Transformer programs kept it as its most effective change). As cheap as ReLU, with a continuous derivative
    at zero and a response that grows with the size of the match between input and neuron."""
    return F.relu(x).square()


ACTIVATIONS: dict[str, Callable[[Tensor], Tensor]] = {
    "relu": relu, "gelu": gelu, "silu": silu, "relu2": relu2, "swiglu": silu, "geglu": gelu,
}


class MLP(nn.Module):
    """The feed-forward layer: proj(act(fc(x))), d -> mlp_mult * d -> d, applied at every position.

    Read as a key-value memory (Geva et al., 2020, arXiv:2012.14913): each hidden unit matches its input
    weights (a key) against the residual stream and adds its output weights (a value) in proportion.

    The gated variants, SwiGLU and GEGLU (Shazeer, 2020, arXiv:2002.05202, after the gated linear unit of
    Dauphin et al., 2017, arXiv:1612.08083), compute proj(act(gate(x)) * fc(x)): one linear map of the input,
    through the nonlinearity, gates another, so each hidden unit is a product of two views of its input. They
    have three matrices instead of two, so the hidden width is scaled by 2/3 to keep the parameter count; at
    equal parameters they gave the lowest perplexity in Shazeer's comparison, and PaLM (2022) and Llama (2023)
    adopted SwiGLU.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, cfg.d_mlp, bias=cfg.bias)
        self.gate = nn.Linear(cfg.d_model, cfg.d_mlp, bias=cfg.bias) if cfg.act in GATED else None
        self.proj = nn.Linear(cfg.d_mlp, cfg.d_model, bias=cfg.bias)
        self.act = ACTIVATIONS[cfg.act]

    def forward(self, x: Tensor) -> Tensor:
        if self.gate is None:
            return self.proj(self.act(self.fc(x)))
        return self.proj(self.act(self.gate(x)) * self.fc(x))

    def forward_explicit(self, x: Tensor, hook: Hooks, layer: int) -> Tensor:
        """`mlp.l.pre` is what enters the nonlinearity (the gate, in a gated unit); `mlp.l.post` the neurons."""
        if self.gate is None:
            post = self.act(hook(f"mlp.{layer}.pre", self.fc(x)))
        else:
            post = self.act(hook(f"mlp.{layer}.pre", self.gate(x))) * self.fc(x)
        return self.proj(hook(f"mlp.{layer}.post", post))


# ----------------------------------------------------------------------------------------------------------------
# The block


class Block(nn.Module):
    """Attention, then the MLP, each added to the residual stream.

    Where the normalization goes: `post` (the Transformer, 2017) normalizes after the addition,
    x = norm(x + f(x)), so every layer rescales the whole stream. `pre` (GPT-2, 2019; analyzed by Xiong et al.,
    2020, arXiv:2002.04745) normalizes only what the sublayer reads, x = x + f(norm(x)): the stream itself is
    an untouched sum from the embedding to the last layer, gradients reach every layer directly, and deep
    stacks train without the long warmup that post-norm needs. An attention-only model (`use_mlp=False`) has
    no second half.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.pre_norm = cfg.norm_placement == "pre"
        self.norm1 = make_norm(cfg)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = make_norm(cfg) if cfg.use_mlp else None
        self.mlp = MLP(cfg) if cfg.use_mlp else None
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor, rot: Rotation, bias: Optional[Tensor]) -> Tensor:
        if self.pre_norm:
            x = x + self.drop(self.attn(self.norm1(x), rot, bias))
            if self.mlp is not None:
                x = x + self.drop(self.mlp(self.norm2(x)))
        else:
            x = self.norm1(x + self.drop(self.attn(x, rot, bias)))
            if self.mlp is not None:
                x = self.norm2(x + self.drop(self.mlp(x)))
        return x

    def forward_explicit(self, x: Tensor, rot: Rotation, bias: Tensor, hook: Hooks, layer: int,
                         kv: Optional[KVCache] = None) -> Tensor:
        x = hook(f"resid_pre.{layer}", x)
        a = self.attn.forward_explicit(self.norm1(x) if self.pre_norm else x, rot, bias, hook, layer, kv)
        a = self.drop(hook(f"attn_out.{layer}", a))
        x = hook(f"resid_mid.{layer}", x + a if self.pre_norm else self.norm1(x + a))
        if self.mlp is not None:
            m = self.mlp.forward_explicit(self.norm2(x) if self.pre_norm else x, hook, layer)
            m = self.drop(hook(f"mlp_out.{layer}", m))
            x = x + m if self.pre_norm else self.norm2(x + m)
        return hook(f"resid_post.{layer}", x)


# ----------------------------------------------------------------------------------------------------------------
# The model


class GPT(nn.Module):
    """Token ids in, next-token logits out: embed, n_layer blocks, final norm, unembed."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.block_size, cfg.d_model) if cfg.pos == "learned" else None
        self.embed_norm = make_norm(cfg) if cfg.embed_norm else None
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        # With post-norm every block ends in a norm; with pre-norm the stream needs one before it is read out.
        self.final_norm = make_norm(cfg) if cfg.norm_placement == "pre" else nn.Identity()
        self.unembed = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            # Weight tying (Press and Wolf, 2017, arXiv:1608.05859; Inan et al., 2017, arXiv:1611.01462): one
            # matrix reads tokens in and writes them out. It saves V * d parameters and was standard through
            # GPT-2. Untied, the two roles get their own geometry and their own learning rates.
            self.unembed.weight = self.embed.weight
        self.init_weights()
        # Position tables are computed, not learned: buffers kept out of the checkpoint, extended on demand.
        self.n_positions = 0
        for name in ("pos_table", "rope_cos", "rope_sin", "alibi"):
            self.register_buffer(name, None, persistent=False)
        self.extend_positions(cfg.block_size)

    @torch.no_grad()
    def init_weights(self) -> None:
        """Two schemes.

        `gpt2` (Radford et al., 2019): every weight is drawn from N(0, 0.02^2), and the matrices that write
        into the residual stream are scaled down by 1/sqrt(N), N the number of residual layers, so that the
        stream's variance at the output does not grow with depth.

        `zero_proj`: the matrices that write into the residual stream start at zero, so every block starts
        as the identity and the network as a bigram model (embed, unembed); the blocks then grow in. This is
        the idea of Fixup (Zhang, Dauphin and Ma, 2019, arXiv:1901.09321) and ReZero (Bachlechner et al.,
        2020, arXiv:2003.04887). The other hidden matrices are uniform with bound 1/sqrt(fan_in); an untied
        embedding is N(0, 1), the scale of the normalized stream it feeds, and an untied unembedding starts
        near zero, at N(0, 0.001^2), so the first predictions are uniform.
        """
        cfg = self.cfg
        gpt2 = cfg.init == "gpt2"
        self.embed_std = 0.02 if gpt2 or cfg.tie_embeddings else 1.0
        nn.init.normal_(self.embed.weight, std=self.embed_std)
        if self.pos_embed is not None:
            nn.init.normal_(self.pos_embed.weight, std=self.embed_std)
        if not cfg.tie_embeddings:
            nn.init.normal_(self.unembed.weight, std=0.02 if gpt2 else 0.001)
        n_residual = cfg.n_layer * (2 if cfg.use_mlp else 1)
        for block in self.blocks:
            writers = [block.attn.o] + ([block.mlp.proj] if block.mlp is not None else [])
            for module in block.modules():
                if not isinstance(module, nn.Linear):
                    continue
                writes = any(module is w for w in writers)
                if gpt2:
                    nn.init.normal_(module.weight, std=0.02 / math.sqrt(n_residual) if writes else 0.02)
                elif writes:
                    nn.init.zeros_(module.weight)
                else:
                    bound = 1 / math.sqrt(module.in_features)
                    nn.init.uniform_(module.weight, -bound, bound)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ---- positions

    def extend_positions(self, n: int) -> None:
        """Make the position tables cover n positions. A model with sinusoidal, rotary, ALiBi or no positions
        accepts sequences longer than it was trained on, which is how length generalization is measured; a
        learned table cannot be extended."""
        if n <= self.n_positions:
            return
        cfg, device = self.cfg, self.embed.weight.device
        if cfg.pos == "learned" and n > cfg.block_size:
            raise ValueError(f"learned positions end at block_size {cfg.block_size}; got a sequence of {n}")
        if cfg.pos == "sinusoidal":
            # The 2017 model added unit-amplitude sinusoids to unit-variance embeddings; keep that ratio.
            self.pos_table = (sinusoidal_table(n, cfg.d_model) * self.embed_std).to(device)
        elif cfg.pos == "rope":
            cos, sin = rotary_table(n, cfg.d_head, cfg.rope_base)
            self.rope_cos, self.rope_sin = cos.to(device), sin.to(device)
        elif cfg.pos == "alibi":
            self.alibi = alibi_table(n, cfg.n_head).to(device)
        self.n_positions = n

    def rotation(self, t0: int, t1: int) -> Rotation:
        """The rotary (cos, sin) of positions [t0, t1), or None for a model without rotary positions."""
        return (self.rope_cos[t0:t1], self.rope_sin[t0:t1]) if self.cfg.pos == "rope" else None

    def causal_bias(self, t0: int, t1: int) -> Tensor:
        """What the explicit path adds to the scores of queries at positions [t0, t1) against keys at [0, t1):
        -inf where the key is after the query, and the ALiBi term if any."""
        if self.cfg.pos == "alibi":
            return self.alibi[:, t0:t1, :t1]
        return torch.full((t1 - t0, t1), float("-inf"), device=self.embed.weight.device).triu(diagonal=t0 + 1)

    # ---- the two ends

    def embed_tokens(self, idx: Tensor, t0: int = 0) -> Tensor:
        """Token vectors, plus position vectors for `learned` and `sinusoidal`, for tokens at positions t0, t0+1, ...

        Fixed sinusoids added before an embedding norm fade as training lengthens the token rows (the norm
        divides both by the rows' scale). With `pos_after_norm` they are added after the norm instead, at unit
        amplitude on a unit-rms token vector, so that position keeps a third of the energy whatever the table does.
        """
        x = self.embed(idx)
        after = self.cfg.pos == "sinusoidal" and self.cfg.pos_after_norm
        if self.cfg.pos == "learned":
            x = x + self.pos_embed.weight[t0:t0 + idx.size(1)]
        elif self.cfg.pos == "sinusoidal" and not after:
            x = x + self.pos_table[t0:t0 + idx.size(1)]
        if self.embed_norm is not None:
            x = self.embed_norm(x)
        if after:
            x = x + self.pos_table[t0:t0 + idx.size(1)] / self.embed_std
        return F.dropout(x, self.cfg.dropout, self.training) if self.cfg.dropout > 0 else x

    def unembed_logits(self, x: Tensor) -> Tensor:
        """Logits in float32. The soft-cap cap * tanh(logits / cap) bounds them smoothly (the clipping of Bello
        et al., 2016, arXiv:1611.09940, used on output logits by Gemma 2, 2024, arXiv:2408.00118): the model
        cannot spend its capacity on ever more extreme logits for tokens it is already sure of."""
        logits = self.unembed(x).float()
        cap = self.cfg.logit_softcap
        return cap * torch.tanh(logits / cap) if cap > 0 else logits

    # ---- the fast path

    def forward(self, idx: Tensor, targets: Optional[Tensor] = None) -> Tensor:
        """idx (B, T) token ids -> logits (B, T, V); or, with targets (B, T), the mean cross-entropy in nats
        (a target of -1 is ignored)."""
        T = idx.size(1)
        self.extend_positions(T)
        x = self.embed_tokens(idx)
        rot = self.rotation(0, T)
        bias = self.alibi[:, :T, :T] if self.cfg.pos == "alibi" else None
        for block in self.blocks:
            x = block(x, rot, bias)
        logits = self.unembed_logits(self.final_norm(x))
        if targets is None:
            return logits
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)

    # ---- the explicit path: hooks and cached decoding

    def forward_explicit(self, idx: Tensor, hook: Optional[Hooks] = None, kv: Optional[KVCache] = None) -> Tensor:
        """The same function with attention written out and every activation handed to `hook`.

        With a KV cache, idx holds only the tokens that follow those already in the cache, and the logits
        returned are those of the new tokens.
        """
        hook = hook or Hooks()
        t0 = 0 if kv is None else kv.length
        t1 = t0 + idx.size(1)
        self.extend_positions(t1)
        x = hook("embed", self.embed_tokens(idx, t0))
        rot, bias = self.rotation(t0, t1), self.causal_bias(t0, t1)
        for layer, block in enumerate(self.blocks):
            x = block.forward_explicit(x, rot, bias, hook, layer, kv)
        if kv is not None:
            kv.length = t1
        x = hook("final_norm", self.final_norm(x))
        return hook("logits", self.unembed_logits(x))

    def hook_names(self) -> list[str]:
        """Every activation name of this model, in the order of the computation."""
        names = ["embed"]
        for l in range(self.cfg.n_layer):
            names += [f"resid_pre.{l}"]
            names += [f"attn.{l}.{part}" for part in ("q", "k", "v", "scores", "pattern", "z", "result")]
            names += [f"attn_out.{l}", f"resid_mid.{l}"]
            if self.cfg.use_mlp:
                names += [f"mlp.{l}.pre", f"mlp.{l}.post", f"mlp_out.{l}"]
            names += [f"resid_post.{l}"]
        return names + ["final_norm", "logits"]

    def run_with_cache(self, idx: Tensor, names: Union[None, Iterable[str], Callable[[str], bool]] = None,
                       hooks: Optional[dict[str, HookFn]] = None) -> tuple[Tensor, dict[str, Tensor]]:
        """Logits and the named activations, computed on the explicit path with the same numbers as forward.

        `names` selects what is kept: None keeps everything except the large per-head `attn.l.result`; a list
        keeps the names it lists, with shell wildcards allowed ("attn.*.pattern", "resid_post.*"); a function
        of the name decides for itself. `hooks` are applied as in run_with_hooks, so an intervention and its
        effects can be recorded in one pass. The cached tensors are part of the autograd graph: call under
        torch.no_grad() unless gradients with respect to activations are wanted.

            q, k (before normalization and rotation) and v: (B, H or Hkv, T, dh)
            scores (before the softmax, mask included), pattern: (B, H, T, T)
            z: (B, H, T, dh), each head's output before the output matrix; result: (B, H, T, d), after it
            everything else: (B, T, d), except mlp.l.pre and mlp.l.post (B, T, d_mlp) and logits (B, T, V)
        """
        valid = self.hook_names()
        if names is None:
            def keep(name: str) -> bool:
                return not name.endswith(".result")
        elif callable(names):
            keep = names
        else:
            patterns = list(names)
            for pattern in patterns:
                if not fnmatch.filter(valid, pattern):
                    raise KeyError(f"no activation is named {pattern!r}; see GPT.hook_names()")
            kept = {name for pattern in patterns for name in fnmatch.filter(valid, pattern)}
            keep = kept.__contains__
        hook = Hooks(self._checked(hooks, valid), keep)
        logits = self.forward_explicit(idx, hook)
        return logits, hook.cache

    def run_with_hooks(self, idx: Tensor, hooks: dict[str, HookFn]) -> Tensor:
        """Logits when each named activation is passed through its function: fn(tensor) returns the tensor to
        use instead, or None to leave it. This is ablation, activation patching and steering in one call:

            model.run_with_hooks(idx, {"resid_post.3": lambda x: x + direction})
        """
        return self.forward_explicit(idx, Hooks(self._checked(hooks, self.hook_names())))

    @staticmethod
    def _checked(hooks: Optional[dict[str, HookFn]], valid: list[str]) -> dict[str, HookFn]:
        for name in hooks or {}:
            if name not in valid:
                raise KeyError(f"no activation is named {name!r}; see GPT.hook_names()")
        return dict(hooks or {})

    @torch.no_grad()
    def generate(self, idx: Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None,
                 top_p: Optional[float] = None, min_p: Optional[float] = None, seed: Optional[int] = None) -> Tensor:
        """Continue idx (B, T) by max_new_tokens sampled tokens, decoding with the KV cache; see generate.py."""
        from .generate import generate

        return generate(self, idx, max_new_tokens, temperature, top_k, top_p, min_p, seed)

    # ---- bookkeeping

    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        """The four kinds of parameter that optimizers treat differently: the embedding (with learned
        positions), the unembedding (empty when tied), the hidden matrices, and the gains and biases."""
        embedding = [self.embed.weight] + ([self.pos_embed.weight] if self.pos_embed is not None else [])
        unembedding = [] if self.cfg.tie_embeddings else [self.unembed.weight]
        matrices = [m.weight for m in self.blocks.modules() if isinstance(m, nn.Linear)]
        taken = {id(p) for p in embedding + unembedding + matrices}
        gains = [p for p in self.parameters() if id(p) not in taken]
        return {"embedding": embedding, "unembedding": unembedding, "matrices": matrices, "gains": gains}

    def num_params(self) -> dict[str, int]:
        """Parameter counts by part. A tied unembedding counts zero: its matrix is the embedding's."""
        def count(parameters: Iterable[nn.Parameter]) -> int:
            return sum(p.numel() for p in parameters)

        n = {
            "embedding": self.embed.weight.numel(),
            "position": self.pos_embed.weight.numel() if self.pos_embed is not None else 0,
            "unembedding": 0 if self.cfg.tie_embeddings else self.unembed.weight.numel(),
            "attention": count(p for block in self.blocks for p in block.attn.parameters()),
            "mlp": count(p for block in self.blocks if block.mlp is not None for p in block.mlp.parameters()),
        }
        total = count(self.parameters())
        n["norm"] = total - sum(n.values())
        n["total"] = total
        return n

    @classmethod
    def from_checkpoint(cls, path: Union[str, Path],
                        device: Union[str, torch.device] = "cuda") -> tuple["GPT", ModelConfig]:
        """Load a checkpoint written by the trainer (`model.pt`, `ckpt_<step>.pt`, or the run directory that
        holds `model.pt`): {"model": state_dict, "model_config": dict, "step", "tokens"}. The model is returned
        on `device`, in evaluation mode, with its configuration."""
        path = Path(path)
        if path.is_dir():
            path = path / "model.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        cfg = ModelConfig.from_dict(checkpoint["model_config"])
        model = cls(cfg)
        # A state dict saved from a compiled or data-parallel wrapper carries prefixes on every key.
        state = {re.sub(r"^(module\.|_orig_mod\.)+", "", key): value for key, value in checkpoint["model"].items()}
        model.load_state_dict(state)
        return model.to(device).eval(), cfg
