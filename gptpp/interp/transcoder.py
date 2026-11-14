"""Transcoders: a wide, sparse stand-in for each MLP of a model, trained to imitate it.

An MLP reads the normalized residual stream x at one token and adds y = MLP(x) to the stream. A sparse autoencoder
(sae.py) describes what is in an activation; it cannot say what a layer does with it, because between the
features it finds before an MLP and those after it stands the MLP itself, a dense nonlinear map. A transcoder
(Dunefsky, Chlenski and Nanda, 2024, arXiv:2406.11944) replaces that map by a second network with one hidden layer,
much wider than the MLP and sparse,

    z   = TopK_k(ReLU(E (x - m) + c))      the code: n numbers, at most k of them not zero
    y^  = z D + b                          a sum of at most k rows of D (n x d), plus a constant

trained so that y^ is close to y. Latent i reads the direction E_i of the MLP's input and writes the direction D_i
into the stream, and nothing else happens in between: with every MLP replaced in this way, and the attention
patterns and normalization denominators of one prompt held fixed, the model is linear from latent to latent, which
is what an attribution graph needs (Ameisen et al., "Circuit Tracing", 2025; attribution_graph.py).

Here there is one transcoder per MLP (per-layer transcoders; a cross-layer transcoder would let a latent write to
every later layer), with TopK sparsity (Gao et al., 2024, arXiv:2406.04093). All of them are trained at once on the
same tokens: every tensor of `Transcoders` has a leading dimension over the model's layers.

    data            stories read from their start: windows of the training text that begin at a document-start
                    mark, a context long. Every token counts, the mark included, since the stand-in must work
                    wherever the model is run. The mark sees itself alone, so its MLP input is one fixed vector
                    and its output a constant many times longer than any other token's: the statistics below and
                    the reported errors leave the mark out, and its error is reported apart
    normalization   the input is centered by its dataset mean m; the output is centered by its mean and divided
                    by one constant per layer, so that the mean squared norm of what is left is 1. The rows of D
                    are unit vectors in those units
    initialization  D and E: random unit rows, independently (input and output are different spaces, so E cannot
                    start as D as in an autoencoder); b and c: zero
    loss            |y - y^|^2 in normalized units (the fraction of variance unexplained, summed over layers),
                    plus 1/32 of the auxiliary error of AuxK: the k_aux = d / 2 dead latents with the largest
                    pre-activations are asked to reconstruct what the live ones left over
    optimizer       Adam (0.9, 0.999), epsilon 6.25e-10, a short warmup, linear decay to zero over the last fifth;
                    after every step the rows of D are renormalized. The learning rate is 8 times the rule of Gao
                    et al., 2e-4 / sqrt(n / 2^14), which is tuned for runs a hundred times longer: on pico at 10
                    million tokens the unexplained variance fell from 0.09 to 0.30 by layer (at the rule) to 0.02
                    to 0.07 (at 8 times), and at 20 times a third of the latents of the late layers died

Evaluation, on held-out stories:

    fvu             per layer, sum |y - y^|^2 / sum |y - mean|^2 over the tokens that are not the mark
    l0, dead        the mean number of active latents a token; the fraction of latents that never fire
    spliced loss    the model's next-token loss with one MLP replaced by its transcoder, and with all of them
                    replaced at once (the replacement model, in which the errors of early layers are read by the
                    later ones); against the clean loss and the loss with every MLP output set to zero, and to
                    its mean
    agreement       how often the replacement model's most likely next token is the model's own

    python -m gptpp.interp.transcoder --run pico --n 2048 --k 16 --tokens 40e6

trains the transcoders of every layer, evaluates them and writes `<name>.pt` and `<name>.json` under
$GPTPP_WORK/transcoders/<run>/.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from gptpp import work_dir
from gptpp.data import TokenData
from gptpp.interp.sae import autocast, load_run
from gptpp.model import GPT

AUX_WEIGHT = 1 / 32  # the weight of the auxiliary loss (Gao et al., appendix B.1)
DENSE = 0.1  # a latent that fires on more than this fraction of tokens is called dense


# ----------------------------------------------------------------------------------------------------------------
# The MLPs of a model: what each reads and writes, and the model with their outputs replaced


def mlp_modules(model: GPT) -> list[nn.Module]:
    return [block.mlp for block in model.blocks]


@torch.no_grad()
def mlp_io(model: GPT, idx: Tensor) -> tuple[Tensor, Tensor]:
    """What every MLP reads and writes on a batch of windows: (x, y), each (L, B, T, d) in float32. x is the
    normalized stream the MLP is given, y what it adds to the stream. One pass through the blocks on the fast
    path (the unembedding, the largest product of a small model, is not needed), in the precision the model was
    trained in, with a hook on each MLP."""
    seen: list[tuple[Tensor, Tensor]] = []
    hooks = [mlp.register_forward_hook(lambda module, args, out: seen.append((args[0].float(), out.float())))
             for mlp in mlp_modules(model)]
    try:
        with autocast(idx.device):
            T = idx.size(1)
            model.extend_positions(T)
            stream, rot = model.embed_tokens(idx), model.rotation(0, T)
            bias = model.alibi[:, :T, :T] if model.cfg.pos == "alibi" else None
            for block in model.blocks:
                stream = block(stream, rot, bias)
    finally:
        for hook in hooks:
            hook.remove()
    return torch.stack([x for x, _ in seen]), torch.stack([y for _, y in seen])


@torch.no_grad()
def logits_with_mlps(model: GPT, idx: Tensor, replace: Callable[[int, Tensor, Tensor], Optional[Tensor]]) -> Tensor:
    """The model's logits when the output of MLP l is `replace(l, x, y)` (x its input, y its own output; None
    leaves it). Later layers read the replaced stream, so the errors of a stand-in are carried forward."""
    hooks = [mlp.register_forward_hook(lambda module, args, out, layer=layer: replace(layer, args[0], out))
             for layer, mlp in enumerate(mlp_modules(model))]
    try:
        with autocast(idx.device):
            return model(idx).float()
    finally:
        for hook in hooks:
            hook.remove()


# ----------------------------------------------------------------------------------------------------------------
# The transcoders


@dataclass(frozen=True)
class TranscoderConfig:
    """The transcoders of one model: one per MLP, all of the same shape."""

    d: int  # the width of the stream
    n: int  # the latents of each transcoder: rows of its dictionary
    k: int  # the active latents a token
    layers: int  # the number of MLPs

    @property
    def name(self) -> str:
        return f"topk-n{self.n}-k{self.k}"


class Transcoders(nn.Module):
    """x -> z = TopK(ReLU(E (x - m) + c)) -> y^ = (z D + b) * scale + mean, for every MLP of a model at once.

    `encoder` (E) and `dictionary` (D) are (L, n, d), `enc_bias` (c) is (L, n), `bias` (b) is (L, d). The inputs
    and outputs of the methods are the model's own activations (raw units); the normalization lives in the
    buffers `in_mean`, `out_mean`, `out_scale`.
    """

    def __init__(self, cfg: TranscoderConfig, seed: int = 0):
        super().__init__()
        self.cfg = cfg
        g = torch.Generator().manual_seed(seed)
        shape = (cfg.layers, cfg.n, cfg.d)
        self.dictionary = nn.Parameter(F.normalize(torch.randn(shape, generator=g), dim=-1))
        self.encoder = nn.Parameter(F.normalize(torch.randn(shape, generator=g), dim=-1))
        self.enc_bias = nn.Parameter(torch.zeros(cfg.layers, cfg.n))
        self.bias = nn.Parameter(torch.zeros(cfg.layers, cfg.d))
        self.register_buffer("in_mean", torch.zeros(cfg.layers, cfg.d))  # the dataset mean of the MLP inputs
        self.register_buffer("out_mean", torch.zeros(cfg.layers, cfg.d))  # and of the MLP outputs
        self.register_buffer("out_scale", torch.ones(cfg.layers))  # normalized output = (raw - out_mean) / out_scale

    # ---- every layer at once: x and y are (L, B, d)

    def pre(self, x: Tensor) -> Tensor:
        """Pre-activations (L, B, n): how much of each latent's input direction the MLP's input holds."""
        return torch.baddbmm(self.enc_bias[:, None, :], x - self.in_mean[:, None, :], self.encoder.transpose(1, 2))

    def code(self, pre: Tensor) -> Tensor:
        """The k largest positive pre-activations of each token, zeros elsewhere."""
        top = F.relu(pre).topk(self.cfg.k, dim=-1, sorted=False)
        return torch.zeros_like(pre).scatter_(-1, top.indices, top.values)

    def encode(self, x: Tensor) -> Tensor:
        return self.code(self.pre(x))

    def normalize_out(self, y: Tensor) -> Tensor:
        return (y - self.out_mean[:, None, :]) / self.out_scale[:, None, None]

    def decode_normalized(self, z: Tensor) -> Tensor:
        return torch.baddbmm(self.bias[:, None, :], z, self.dictionary)

    def decode(self, z: Tensor) -> Tensor:
        return self.decode_normalized(z) * self.out_scale[:, None, None] + self.out_mean[:, None, :]

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """MLP inputs (L, B, d) -> the stand-in's outputs (L, B, d) and its code (L, B, n)."""
        z = self.encode(x)
        return self.decode(z), z

    # ---- one layer, as plain matrices in the model's units: what attribution reads

    def read(self, layer: int) -> tuple[Tensor, Tensor]:
        """(E, offset) of one layer with pre-activation = x E^T + offset: (n, d) and (n,). The offset is the
        latent's own constant, c - E m."""
        E = self.encoder[layer]
        return E, self.enc_bias[layer] - E @ self.in_mean[layer]

    def write(self, layer: int) -> tuple[Tensor, Tensor]:
        """(W, b) of one layer with y^ = z W + b: (n, d) and (d,). Row i of W is what one unit of latent i adds
        to the stream; b is the transcoder's constant output."""
        return self.dictionary[layer] * self.out_scale[layer], self.bias[layer] * self.out_scale[layer] + self.out_mean[layer]

    def layer_pre(self, layer: int, x: Tensor) -> Tensor:
        """Pre-activations of one layer's latents for inputs (..., d) -> (..., n)."""
        E, offset = self.read(layer)
        return x @ E.T + offset

    def layer_forward(self, layer: int, x: Tensor) -> tuple[Tensor, Tensor]:
        """One layer on inputs (..., d): its output (..., d) and its code (..., n)."""
        z = self.code(self.layer_pre(layer, x))
        W, b = self.write(layer)
        return z @ W + b, z

    def replacer(self) -> Callable[[int, Tensor, Tensor], Tensor]:
        """For logits_with_mlps: every MLP's output replaced by its transcoder's."""
        return lambda layer, x, y: self.layer_forward(layer, x.float())[0].to(y.dtype)


def save_transcoders(tcs: Transcoders, path: Path, info: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": tcs.state_dict(), "config": asdict(tcs.cfg), "info": info}, path)


def load_transcoders(path: str | Path, device: str | torch.device = "cpu") -> tuple[Transcoders, dict]:
    """The transcoders written by this module (`<name>.pt`), in evaluation mode, with what was recorded of them."""
    blob = torch.load(path, map_location="cpu", weights_only=True)
    tcs = Transcoders(TranscoderConfig(**blob["config"]))
    tcs.load_state_dict(blob["state"])
    return tcs.to(device).eval(), blob["info"]


def default_path(run: str, name: str) -> Path:
    return work_dir() / "transcoders" / run / f"{name}.pt"


# ----------------------------------------------------------------------------------------------------------------
# The text: stories read from their start mark


def story_starts(tokens: np.ndarray, bos_id: int, context: int, rng: np.random.Generator, n: int) -> np.ndarray:
    """n places of a token stream where a document begins and a whole context follows, drawn at random."""
    starts = []
    while len(starts) < n:
        at = int(rng.integers(0, len(tokens) - 2 * context - 4096))
        marks = np.flatnonzero(tokens[at:at + 4096] == bos_id)
        if len(marks):
            starts.append(at + int(marks[0]))
    return np.array(starts)


def heldout_starts(tokens: np.ndarray, bos_id: int, context: int, n: int) -> np.ndarray:
    """The first n windows of a token stream that begin at a document start and do not overlap, in order."""
    marks = np.flatnonzero(np.asarray(tokens) == bos_id)
    starts, end = [], 0
    for at in marks:
        if at >= end and at + context + 1 <= len(tokens):
            starts.append(int(at))
            end = at + context
            if len(starts) == n:
                break
    return np.array(starts)


def windows_at(tokens: np.ndarray, starts: np.ndarray, length: int, device: str | torch.device) -> Tensor:
    block = tokens[starts[:, None] + np.arange(length)[None, :]]
    return torch.from_numpy(block.astype(np.int64)).to(device)


class StoryStream:
    """The MLPs' inputs and outputs on training stories, token by token in shuffled order.

    A buffer of `buffer_windows` stories is run through the model; the tokens of all of them are shuffled together
    and handed out in batches of `batch` as (x, y, is_mark): (L, B, d), (L, B, d) and (B,) bool. Then the buffer
    is filled again, from another place of another shard.
    """

    def __init__(self, model: GPT, data: TokenData, bos_id: int, batch: int, buffer_windows: int = 256,
                 model_batch: int = 64, seed: int = 0, device: str | torch.device = "cuda"):
        self.model, self.data, self.bos_id = model, data, bos_id
        self.batch, self.buffer_windows, self.model_batch, self.device = batch, buffer_windows, model_batch, device
        self.context = model.cfg.block_size
        self.rng = np.random.default_rng(seed)
        self.torch_rng = torch.Generator(device=device).manual_seed(seed)

    def fill(self) -> tuple[Tensor, Tensor, Tensor]:
        shard = self.data.shard(int(self.rng.integers(len(self.data.train_files))))
        starts = story_starts(shard, self.bos_id, self.context, self.rng, self.buffer_windows)
        xs, ys, marks = [], [], []
        for i in range(0, len(starts), self.model_batch):
            idx = windows_at(shard, starts[i:i + self.model_batch], self.context, self.device)
            x, y = mlp_io(self.model, idx)
            xs.append(x.flatten(1, 2)), ys.append(y.flatten(1, 2)), marks.append((idx == self.bos_id).flatten())
        x, y, mark = torch.cat(xs, 1), torch.cat(ys, 1), torch.cat(marks)
        order = torch.randperm(x.size(1), generator=self.torch_rng, device=self.device)
        return x[:, order], y[:, order], mark[order]

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor, Tensor]]:
        while True:
            x, y, mark = self.fill()
            for i in range(0, x.size(1) - self.batch + 1, self.batch):
                yield x[:, i:i + self.batch], y[:, i:i + self.batch], mark[i:i + self.batch]


# ----------------------------------------------------------------------------------------------------------------
# Training


@torch.no_grad()
def fit_normalization(tcs: Transcoders, x: Tensor, y: Tensor, mark: Tensor) -> None:
    """The means and the output scale of every layer from a sample (L, N, d), the document-start mark left out."""
    x, y = x[:, ~mark], y[:, ~mark]
    tcs.in_mean.copy_(x.mean(1))
    tcs.out_mean.copy_(y.mean(1))
    tcs.out_scale.copy_((y - tcs.out_mean[:, None, :]).pow(2).sum(-1).mean(1).sqrt())


def decode_sparse(tcs: Transcoders, values: Tensor, indices: Tensor) -> Tensor:
    """z D from the active latents alone, (L, B, k) values and indices -> (L, B, d): the sum of k rows of each
    layer's dictionary, without the product with the n - k zeros."""
    return torch.stack([F.embedding_bag(indices[layer], tcs.dictionary[layer], per_sample_weights=values[layer], mode="sum")
                        for layer in range(tcs.cfg.layers)])


def loss_of(tcs: Transcoders, x: Tensor, y: Tensor, dead: Optional[list[Tensor]], k_aux: int) -> tuple[Tensor, Tensor, Tensor]:
    """The loss of every layer's transcoder on a batch, summed; the error of each layer as a fraction of its
    variance (L,); and which latents fired (L, n). `dead` lists the dead latents of each layer (their indices),
    or is None when there are none.

    AuxK (Gao et al., 2024): the k_aux dead latents with the largest pre-activations reconstruct what the live
    ones left over. That error is a constant here (detached): the auxiliary term must not move the live latents.
    """
    with autocast(x.device):  # the one large product, n latents by B tokens by d, in half precision on a card
        pre = tcs.pre(x)
        top = pre.topk(tcs.cfg.k, dim=-1, sorted=False)
    values = F.relu(top.values.float())  # the k largest, where positive: the code, without its zeros
    error = tcs.normalize_out(y) - (decode_sparse(tcs, values, top.indices) + tcs.bias[:, None, :])
    fvu = error.pow(2).sum(-1).mean(1)
    loss = fvu.sum()
    fired = torch.zeros(tcs.cfg.layers, tcs.cfg.n, device=x.device).scatter_add_(1, top.indices.flatten(1), (values > 0).float().flatten(1)) > 0
    if dead is not None:
        target = error.detach()
        spread = (target - target.mean(1, keepdim=True)).pow(2).sum(-1).mean(1)
        for layer, rows in enumerate(dead):  # layer by layer, the dead latents alone: their pre-activations again
            if not len(rows):
                continue
            aux_pre = (x[layer] - tcs.in_mean[layer]) @ tcs.encoder[layer][rows].T + tcs.enc_bias[layer][rows]
            if len(rows) > k_aux:  # keep each token's k_aux largest: those at or above its k_aux-th value
                kth = aux_pre.detach().sort(dim=-1, descending=True).values[:, k_aux - 1:k_aux]
                aux_pre = aux_pre.masked_fill(aux_pre < kth, 0.0)
            left = target[layer] - F.relu(aux_pre) @ tcs.dictionary[layer][rows]
            loss = loss + AUX_WEIGHT * (left.pow(2).sum(-1).mean() / spread[layer]).nan_to_num(0.0)
    return loss, fvu.detach(), fired


def train(tcs: Transcoders, stream: StoryStream, tokens: float, dead_after: int = 1_000_000, lr_scale: float = 8.0,
          log_every: int = 500, say=print) -> list[dict]:
    """Train the transcoders of every layer on `tokens` tokens of the stream; returns the log."""
    steps = int(tokens) // stream.batch
    batches = iter(stream)
    fit_normalization(tcs, *stream.fill())
    lr = lr_scale * 2e-4 / math.sqrt(tcs.cfg.n / 2 ** 14)
    opt = torch.optim.Adam(tcs.parameters(), lr=lr, betas=(0.9, 0.999), eps=6.25e-10)
    warmup, decay_from = min(1000, steps // 20), int(0.8 * steps)
    since_fired = torch.zeros(tcs.cfg.layers, tcs.cfg.n, dtype=torch.long, device=tcs.dictionary.device)
    dead, k_aux, log, start = None, max(1, tcs.cfg.d // 2), [], time.time()
    harvest = 0.0
    for step in range(steps):
        t0 = time.time()
        x, y, _ = next(batches)
        harvest += time.time() - t0
        loss, fvu, fired = loss_of(tcs, x, y, dead, k_aux)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():  # a unit row can only turn: remove the part of its gradient that would lengthen it
            g, rows = tcs.dictionary.grad, tcs.dictionary
            g -= (g * rows).sum(-1, keepdim=True) * rows
        factor = (step + 1) / warmup if step < warmup else 1.0 if step < decay_from else (steps - step) / (steps - decay_from)
        for group in opt.param_groups:
            group["lr"] = lr * factor
        opt.step()
        with torch.no_grad():
            tcs.dictionary /= tcs.dictionary.norm(dim=-1, keepdim=True)
        since_fired += x.size(1)
        since_fired.masked_fill_(fired, 0)
        if step % 64 == 0:  # which latents are dead is looked up once in 64 steps: asking the card for a number makes the step wait
            is_dead = since_fired > dead_after
            dead = [row.nonzero().flatten() for row in is_dead] if bool(is_dead.any()) else None
        if step % log_every == 0 or step == steps - 1:
            dead_now = (since_fired > dead_after).float().mean(1)
            log.append({"step": step, "tokens": (step + 1) * stream.batch, "fvu": [round(float(v), 4) for v in fvu],
                        "dead": [round(float(v), 4) for v in dead_now]})
            say(f"step {step:6d}/{steps}  {time.time() - start:5.0f}s  fvu by layer " + " ".join(f"{float(v):.3f}" for v in fvu)
                + f"  dead {float(dead_now.mean()):.3f}  (reading the model: {harvest:.0f}s)")
    return log


# ----------------------------------------------------------------------------------------------------------------
# Evaluation on held-out stories


@torch.no_grad()
def evaluate(model: GPT, tcs: Transcoders, tokens: np.ndarray, bos_id: int, n_windows: int = 1024, model_batch: int = 32,
             device: str | torch.device = "cuda") -> dict:
    """The transcoders on held-out stories: per layer the error, the sparsity and the dead and dense latents; and
    the model's next-token loss with its MLPs replaced, one at a time and all at once."""
    L, context = tcs.cfg.layers, model.cfg.block_size
    starts = heldout_starts(tokens, bos_id, context, n_windows)
    zeros = lambda *shape: torch.zeros(*shape, dtype=torch.float64, device=device)
    err, tot, err_mark, tot_mark, active, fired = zeros(L), zeros(L), zeros(L), zeros(L), zeros(L), zeros(L, tcs.cfg.n)
    sum_y, n_tok, n_mark = zeros(L, tcs.cfg.d), 0, 0
    nll = {name: 0.0 for name in ["clean", "replaced", "zero", "mean"] + [f"layer{l}" for l in range(L)]}
    agree = same_as_model = n_pred = 0
    # the mean output of the held-out tokens, for the variance: a first pass
    for i in range(0, len(starts), model_batch):
        idx = windows_at(tokens, starts[i:i + model_batch], context, device)
        _, y = mlp_io(model, idx)
        keep = (idx != bos_id).flatten()
        sum_y += y.flatten(1, 2)[:, keep].double().sum(1)
        n_tok += int(keep.sum())
    mean_y = (sum_y / n_tok).float()
    for i in range(0, len(starts), model_batch):
        both = windows_at(tokens, starts[i:i + model_batch], context + 1, device)
        idx, targets = both[:, :-1], both[:, 1:]
        x, y = mlp_io(model, idx)
        x, y, mark = x.flatten(1, 2), y.flatten(1, 2), (idx == bos_id).flatten()
        y_hat, z = tcs(x)
        sq, var = (y - y_hat).pow(2).sum(-1).double(), (y - mean_y[:, None, :]).pow(2).sum(-1).double()
        err += sq[:, ~mark].sum(1); tot += var[:, ~mark].sum(1)
        err_mark += sq[:, mark].sum(1); tot_mark += var[:, mark].sum(1)
        on = (z > 0).float()
        active += on.sum((1, 2)).double(); fired += on.sum(1).double()
        n_mark += int(mark.sum())

        def loss(logits: Tensor) -> float:
            return float(F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten(), reduction="sum"))

        clean = model(idx)
        replaced = logits_with_mlps(model, idx, tcs.replacer())
        nll["clean"] += loss(clean)
        nll["replaced"] += loss(replaced)
        nll["zero"] += loss(logits_with_mlps(model, idx, lambda layer, x_, y_: torch.zeros_like(y_)))
        nll["mean"] += loss(logits_with_mlps(model, idx, lambda layer, x_, y_: mean_y[layer].to(y_.dtype).expand_as(y_)))
        for only in range(L):
            nll[f"layer{only}"] += loss(logits_with_mlps(model, idx, lambda layer, x_, y_, only=only: tcs.layer_forward(layer, x_.float())[0].to(y_.dtype) if layer == only else None))
        agree += int((replaced.argmax(-1) == clean.argmax(-1)).sum())
        same_as_model += int((clean.argmax(-1) == targets).sum())
        n_pred += targets.numel()
    n_all = n_tok + n_mark
    H = {name: value / n_pred for name, value in nll.items()}
    density = fired / n_all
    return {
        "windows": int(len(starts)), "tokens": int(n_all), "marks": int(n_mark),
        "fvu": [float(v) for v in err / tot], "fvu_mark": [float(v) for v in err_mark / tot_mark.clamp_min(1e-30)],
        "l0": [float(v) for v in active / n_all],
        "dead": [float(v) for v in (fired == 0).double().mean(1)], "dense": [float(v) for v in (density > DENSE).double().mean(1)],
        "loss": H,
        "loss_recovered": (H["zero"] - H["replaced"]) / (H["zero"] - H["clean"]),
        "loss_recovered_mean": (H["mean"] - H["replaced"]) / (H["mean"] - H["clean"]),
        "loss_increase_by_layer": [H[f"layer{l}"] - H["clean"] for l in range(L)],
        "agreement": agree / n_pred, "model_accuracy": same_as_model / n_pred,
    }


# ----------------------------------------------------------------------------------------------------------------
# What a latent is about: the tokens it fires on, the tokens it promotes, a passage where it fires most


@torch.no_grad()
def describe(model: GPT, tcs: Transcoders, tokens: np.ndarray, bos_id: int, n_windows: int = 512, model_batch: int = 32,
             device: str | torch.device = "cuda") -> dict:
    """Statistics of every latent on held-out stories, for naming it:

        density      (L, n) the fraction of tokens it fires on
        mass         (L, n, V) its summed activation by the token it fires on: the tokens it reads
        best         (L, n) its largest activation, with `best_window` and `best_pos`: where (an index into `starts`
                     and a position of that window)
        starts       the windows read
    The tokens a latent promotes need no text: they are the largest entries of its row of the dictionary through
    the unembedding (`promoted`).
    """
    L, n, V, context = tcs.cfg.layers, tcs.cfg.n, model.cfg.vocab_size, model.cfg.block_size
    starts = heldout_starts(tokens, bos_id, context, n_windows)
    fired = torch.zeros(L, n, device=device)
    mass = torch.zeros(L, n, V, device=device)
    best = torch.zeros(L, n, device=device)
    best_at = torch.zeros(L, n, dtype=torch.long, device=device)
    for i in range(0, len(starts), model_batch):
        idx = windows_at(tokens, starts[i:i + model_batch], context, device)
        x, _ = mlp_io(model, idx)
        z = tcs.encode(x.flatten(1, 2))  # (L, B T, n)
        flat = idx.flatten()
        fired += (z > 0).float().sum(1)
        for layer in range(L):
            mass[layer].index_add_(1, flat, z[layer].T)
        top = z.max(1)
        better = top.values > best
        best = torch.where(better, top.values, best)
        best_at = torch.where(better, top.indices + i * context, best_at)
    return {"density": fired / (len(starts) * context), "mass": mass, "best": best,
            "best_window": best_at // context, "best_pos": best_at % context, "starts": starts}


@torch.no_grad()
def promoted(model: GPT, tcs: Transcoders, layer: int, latents: Tensor, top: int = 4,
             allowed: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
    """The tokens whose logits one unit of each given latent raises and lowers most by its direct path to the
    unembedding (its row of the dictionary, read by the unembedding, less the mean over the vocabulary): two
    (len(latents), top) tensors of token ids. Later layers also read the latent; this is the direct effect only.
    `allowed` (V, bool) restricts the tokens named, for instance to those that occur in text at all: the
    unembedding rows of tokens the model never met are large and mean nothing."""
    W, _ = tcs.write(layer)
    effect = W[latents] @ model.unembed.weight.float().T
    effect = effect - effect.mean(-1, keepdim=True)
    if allowed is not None:
        return effect.masked_fill(~allowed, -torch.inf).topk(top, dim=-1).indices, (-effect).masked_fill(~allowed, -torch.inf).topk(top, dim=-1).indices
    return effect.topk(top, dim=-1).indices, (-effect).topk(top, dim=-1).indices


# ----------------------------------------------------------------------------------------------------------------
# Command line


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="a run directory, or its name under $GPTPP_WORK/ckpt")
    ap.add_argument("--n", type=int, default=2048, help="latents of each transcoder")
    ap.add_argument("--k", type=int, default=16, help="active latents a token")
    ap.add_argument("--tokens", type=float, default=40e6)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr-scale", type=float, default=8.0, help="a factor on the learning rate 2e-4 / sqrt(n / 2^14)")
    ap.add_argument("--eval-windows", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None, help="the folder to write into (default $GPTPP_WORK/transcoders/<run>)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    model, data, info = load_run(args.run, device=args.device)
    model = model.float()
    cfg = TranscoderConfig(d=model.cfg.d_model, n=args.n, k=args.k, layers=model.cfg.n_layer)
    tcs = Transcoders(cfg, seed=args.seed).to(args.device)
    stream = StoryStream(model, data, data.meta["bos_id"], args.batch, seed=args.seed, device=args.device)
    start = time.time()
    log = train(tcs, stream, args.tokens, lr_scale=args.lr_scale)
    seconds = time.time() - start
    tcs.eval()
    scores = evaluate(model, tcs, data.val_tokens(), data.meta["bos_id"], args.eval_windows, device=args.device)
    info = dict(info, name=cfg.name, tokens=int(args.tokens), batch=args.batch, lr_scale=args.lr_scale, seed=args.seed, train_seconds=round(seconds, 1), eval=scores)
    path = Path(args.out) / f"{cfg.name}.pt" if args.out else default_path(info["run"], cfg.name)
    save_transcoders(tcs, path, info)
    path.with_suffix(".json").write_text(json.dumps(dict(info, config=asdict(cfg), log=log), indent=1))
    print(json.dumps({key: scores[key] for key in ("fvu", "fvu_mark", "l0", "dead", "loss", "loss_recovered", "loss_recovered_mean", "agreement")}))
    if torch.cuda.is_available():
        print(f"peak memory {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GB, training {seconds:.0f} s")
    print("wrote", path)


if __name__ == "__main__":
    main()
