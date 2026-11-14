"""Sparse dictionaries on the residual stream: the TopK autoencoder, its training and its evaluation.

The superposition hypothesis (Elhage et al., 2022, arXiv:2209.10652) says that a layer of d units carries many
more than d features, each a direction of the layer's space, few of them active on any one token. A sparse
autoencoder looks for those directions. It rewrites an activation x as a sum of a few rows of a dictionary of
n > d unit directions,

    z   = TopK_k(ReLU(E (x - b) + c))      the code: n numbers, at most k of them not zero
    x^  = z D + b                          the reconstruction: a sum of at most k rows of D (n x d)

and is trained so that x^ is close to x. The rows of D are the candidate features, and z says which of them a
token carries and how strongly.

Two ways of making z sparse are implemented, on the same architecture, so that they can be compared:

    topk    keep the k largest activations and zero the rest (Makhzani and Frey, 2013, arXiv:1312.5663; Gao et
            al., 2024, arXiv:2406.04093). Sparsity is set, not tuned, and the values that are kept are trained
            by the reconstruction alone.
    relu    keep every positive activation and add lambda * sum_i z_i |D_i| to the loss (Bricken et al., 2023;
            the decoder norm in the penalty is from Conerly et al., April 2024). The penalty that makes the
            code sparse also pulls every active value towards zero: the reconstruction comes out too short
            ("shrinkage"), which is what TopK and the gated and JumpReLU variants of 2024 were made to avoid.

The recipe of `topk` follows Gao et al. (appendices A and B) at small scale:

    data            the residual stream after one block, at every token except the document-start mark and the
                    first tokens of a window; the dataset mean is subtracted and one constant scales the
                    activations so that their mean norm is sqrt(d)
    initialization  b: the geometric median of a sample; D: random unit rows; E = D (at the start only);
                    c = 0
    loss            |x - x^|^2 / (the mean of |x|^2) + 1/32 of the same for the auxiliary reconstruction below
    dead latents    a latent that has not fired for `dead_after` tokens is dead. The k_aux = d / 2 dead latents
                    with the largest pre-activations are asked to reconstruct the error x - x^ (AuxK), which
                    gives them a gradient that brings them back
    optimizer       Adam (0.9, 0.999), epsilon 6.25e-10, no weight decay, learning rate 2e-4 / sqrt(n / 2^14)
                    (the largest stable rate falls as 1 / sqrt(n)), a short warmup, linear decay to zero over
                    the last fifth
    every step      the gradient of D loses its component along each row, and the rows are renormalized

Evaluation, on held-out text:

    fvu             the fraction of variance unexplained (the normalized error), sum |x - x^|^2 / sum |x - mean|^2,
                    the mean being that of the held-out activations
    l0              the mean number of active latents a token (k for topk)
    loss recovered  the model's next-token loss with x^ spliced in place of x, H*, against the clean loss H and
                    the loss with the stream replaced by zero, H0: (H0 - H*) / (H0 - H). The zero baseline is
                    weak (Bricken et al. say so), which flatters the ratio; the raw increase H* - H and the
                    ratio against the dataset-mean baseline are reported beside it
    dead, dense     the fraction of latents that never fire on the held-out tokens, and the fraction that fire
                    on more than one token in ten
    gain            the scalar g that minimizes |x - b - g (x^ - b)|^2: 1 when reconstructions have the right
                    length, above 1 when they are too short (shrinkage); and `fvu_rescaled`, the fvu if every
                    reconstruction were lengthened by g, which says how much of the error is shrinkage alone

    python -m gptpp.interp.sae --run <run dir> --layer 2 --widths 1024,4096 --ks 16,32 --l1 0.02 --tokens 100e6

trains one dictionary per (width, k) and per (width, lambda) on the same stream of activations, evaluates each on
held-out text, and writes `<name>.pt` and `<name>.json` under $GPTPP_WORK/sae/<run>-L<layer>/.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from gptpp import work_dir
from gptpp.data import TokenData
from gptpp.model import GPT

AUX_WEIGHT = 1 / 32  # the weight of the auxiliary loss (Gao et al., appendix B.1)
DENSE = 0.1  # a latent that fires on more than this fraction of tokens is called dense


# ----------------------------------------------------------------------------------------------------------------
# The model around the layer: the stream after block `layer`, and the rest of the model from there


def residual_after(model: GPT, idx: Tensor, layer: int) -> Tensor:
    """The residual stream after block `layer` (the activation `resid_post.{layer}`), (B, T, d), computed on the
    model's fast path and stopping there: harvesting needs neither the later blocks nor the attention patterns."""
    T = idx.size(1)
    model.extend_positions(T)
    x = model.embed_tokens(idx)
    rot = model.rotation(0, T)
    bias = model.alibi[:, :T, :T] if model.cfg.pos == "alibi" else None
    for block in model.blocks[: layer + 1]:
        x = block(x, rot, bias)
    return x


def logits_from(model: GPT, x: Tensor, layer: int) -> Tensor:
    """The model's logits when the stream after block `layer` is x: the remaining blocks, the final norm, the
    unembedding. With x = residual_after(model, idx, layer) these are the model's own logits."""
    T = x.size(1)
    rot = model.rotation(0, T)
    bias = model.alibi[:, :T, :T] if model.cfg.pos == "alibi" else None
    for block in model.blocks[layer + 1:]:
        x = block(x, rot, bias)
    return model.unembed_logits(model.final_norm(x))


def load_run(run: str | Path, ckpt: Optional[str] = None, device: str = "cuda") -> tuple[GPT, TokenData, dict]:
    """A trained run: its model, its tokenized dataset, and what was loaded.

    `run` is a run directory (or a name under $GPTPP_WORK/ckpt). Without `ckpt` the final `model.pt` is used, or,
    while the run is still training, its latest `ckpt_<step>.pt`.
    """
    folder = Path(run) if Path(run).is_dir() else work_dir() / "ckpt" / str(run)
    if ckpt is None:
        steps = sorted(folder.glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
        path = folder / "model.pt" if (folder / "model.pt").exists() else steps[-1]
    else:
        path = folder / ckpt
    model, _ = GPT.from_checkpoint(path, device=device)
    meta = json.loads((folder / "config.json").read_text())["data_meta"]
    data = TokenData(f"{meta['dataset']}/{meta['tokenizer']}")
    return model, data, {"run": folder.name, "checkpoint": path.name, "dataset": meta["dataset"],
                         "tokenizer": meta["tokenizer"]}


# ----------------------------------------------------------------------------------------------------------------
# Activations


@dataclass(frozen=True)
class Harvest:
    """Where activations are read: which block, how long a window, and which tokens count.

    The token streams are documents laid end to end, each begun by the document-start mark, and cut into windows
    without regard to where documents begin. Two kinds of position are left out. The mark itself: its activation
    is several times longer than any other token's and would dominate a squared error (Gao et al., appendix F.2,
    report the same of the first position in GPT-2). And the first `skip` tokens of a window, which the model
    reads without their context only because the window was cut there.
    """

    layer: int
    context: int
    skip: int
    bos_id: int

    def keep(self, idx: Tensor) -> Tensor:
        """The tokens of a batch of windows (B, T) whose activations count, as a mask (B, T)."""
        mask = idx != self.bos_id
        mask[:, : self.skip] = False
        return mask


def autocast(device: torch.device | str) -> torch.autocast:
    """The precision the models were trained in: bfloat16 on a card, float32 on the CPU."""
    on_card = torch.device(device).type == "cuda"
    return torch.autocast("cuda" if on_card else "cpu", dtype=torch.bfloat16, enabled=on_card)


@torch.no_grad()
def activations(model: GPT, idx: Tensor, harvest: Harvest) -> Tensor:
    """The stream after the harvested block for a batch of windows, (B, T, d) in float32."""
    with autocast(idx.device):
        return residual_after(model, idx, harvest.layer).float()


def window_batches(tokens: np.ndarray, context: int, batch: int, n_windows: Optional[int] = None,
                   device: str = "cuda") -> Iterator[tuple[int, Tensor]]:
    """The first windows of a token stream in order, `batch` at a time: (index of the first window, ids (B, T))."""
    total = (len(tokens) - 1) // context
    total = total if n_windows is None else min(total, n_windows)
    for first in range(0, total, batch):
        rows = np.arange(first, min(first + batch, total))
        block = tokens[rows[:, None] * context + np.arange(context)[None, :]]
        yield first, torch.from_numpy(block.astype(np.int64)).to(device)


class TrainingStream:
    """Activations of training text in shuffled order, a batch at a time, never the same window twice in a row.

    A buffer of `buffer_windows` windows, drawn at random places of a random shard, is run through the model; the
    activations that count are shuffled across windows and handed out in batches; then the buffer is filled
    again. A batch therefore mixes tokens of thousands of documents, which dictionary learning needs: the tokens
    of one document are strongly correlated (Bricken et al., 2023).
    """

    def __init__(self, model: GPT, data: TokenData, harvest: Harvest, batch: int, buffer_windows: int,
                 model_batch: int, seed: int = 0, device: str = "cuda"):
        self.model, self.data, self.harvest = model, data, harvest
        self.batch, self.buffer_windows, self.model_batch, self.device = batch, buffer_windows, model_batch, device
        self.rng = np.random.default_rng(seed)
        self.torch_rng = torch.Generator(device=device).manual_seed(seed)
        self.tokens_read = 0

    def fill(self) -> Tensor:
        """One buffer of activations (N, d), shuffled."""
        shard = self.data.shard(int(self.rng.integers(len(self.data.train_files))))
        starts = np.sort(self.rng.integers(0, len(shard) - self.harvest.context, size=self.buffer_windows))
        kept = []
        for i in range(0, len(starts), self.model_batch):
            rows = starts[i:i + self.model_batch]
            block = shard[rows[:, None] + np.arange(self.harvest.context)[None, :]]
            idx = torch.from_numpy(block.astype(np.int64)).to(self.device)
            kept.append(activations(self.model, idx, self.harvest)[self.harvest.keep(idx)])
        x = torch.cat(kept)
        return x[torch.randperm(len(x), generator=self.torch_rng, device=self.device)]

    def __iter__(self) -> Iterator[Tensor]:
        while True:
            x = self.fill()
            for i in range(0, len(x) - self.batch + 1, self.batch):
                self.tokens_read += self.batch
                yield x[i:i + self.batch]


def geometric_median(x: Tensor, steps: int = 64) -> Tensor:
    """The point with the least summed distance to the rows of x, by Weiszfeld's iteration: a center that a few
    far outliers cannot move, which is why Bricken et al. (2023) start the autoencoder's bias there."""
    m = x.mean(0)
    for _ in range(steps):
        w = 1 / (x - m).norm(dim=-1).clamp_min(1e-8)
        m = (w[:, None] * x).sum(0) / w.sum()
    return m


# ----------------------------------------------------------------------------------------------------------------
# The autoencoder


@dataclass(frozen=True)
class DictConfig:
    """One dictionary: what it reads, how large it is, how it is made sparse."""

    d: int  # the width of the activations
    n: int  # the number of latents: rows of the dictionary
    kind: str = "topk"  # "topk" or "relu"
    k: int = 32  # topk: the active latents a token
    l1: float = 0.0  # relu: the weight lambda of the L1 penalty
    hook: str = ""  # the activation it reads, e.g. "resid_post.2"

    @property
    def name(self) -> str:
        return f"topk-n{self.n}-k{self.k}" if self.kind == "topk" else f"relu-n{self.n}-l{self.l1:g}"


class SparseAutoencoder(nn.Module):
    """x -> z = sparse(ReLU(E (x - b) + c)) -> x^ = z D + b, on activations normalized by `mean` and `scale`.

    The rows of `dictionary` (D, n x d) are the directions; `encoder` (E, n x d) reads how much of each a token
    carries. E starts as D and is then free: the best direction to detect a feature along is not the direction
    it is written in when other features overlap with it. `bias` (b) is subtracted before encoding and added
    after decoding.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        rows = F.normalize(torch.randn(cfg.n, cfg.d), dim=-1)
        if cfg.kind == "relu":
            rows = 0.1 * rows  # Conerly et al. (April 2024): short rows at the start, no constraint afterwards
        self.dictionary = nn.Parameter(rows)
        self.encoder = nn.Parameter(rows.clone())
        self.enc_bias = nn.Parameter(torch.zeros(cfg.n))
        self.bias = nn.Parameter(torch.zeros(cfg.d))
        self.register_buffer("mean", torch.zeros(cfg.d))  # the dataset mean of the raw activations
        self.register_buffer("scale", torch.ones(()))  # normalized = (raw - mean) * scale
        self.register_buffer("variance", torch.ones(()))  # the mean of |normalized|^2: the error of predicting the mean
        self.register_buffer("threshold", torch.zeros(()))  # the mean smallest activation TopK keeps (set by evaluate)

    # ---- raw activations <-> the normalized space the autoencoder works in

    def normalize(self, x: Tensor) -> Tensor:
        return (x - self.mean) * self.scale

    def restore(self, x: Tensor) -> Tensor:
        return x / self.scale + self.mean

    # ---- the autoencoder, on normalized activations

    def pre(self, x: Tensor) -> Tensor:
        """Pre-activations (B, n): how much of each direction the encoder reads in x."""
        return (x - self.bias) @ self.encoder.T + self.enc_bias

    def encode(self, x: Tensor) -> Tensor:
        """The code z (B, n). topk: the k largest positive activations of each token, zeros elsewhere."""
        acts = F.relu(self.pre(x))
        if self.cfg.kind == "relu":
            return acts
        top = acts.topk(self.cfg.k, dim=-1)
        return torch.zeros_like(acts).scatter_(-1, top.indices, top.values)

    def decode(self, z: Tensor) -> Tensor:
        return z @ self.dictionary + self.bias

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Normalized activations (B, d) -> their reconstruction and their code."""
        z = self.encode(x)
        return self.decode(z), z

    def reconstruct(self, x: Tensor) -> Tensor:
        """Raw activations (..., d) -> raw reconstructions: what is spliced into the model."""
        flat = self.normalize(x.reshape(-1, x.size(-1)))
        return self.restore(self.forward(flat)[0]).reshape(x.shape)

    def directions(self) -> Tensor:
        """What one unit of each latent adds to the model's stream, (n, d): the rows of D in raw units."""
        return self.dictionary / self.scale

    @torch.no_grad()
    def fold_norms(self) -> None:
        """Make every row of D a unit vector by moving its norm into the encoder (Conerly et al., April 2024): the
        function is unchanged, and an activation then means the same length in every latent."""
        norm = self.dictionary.norm(dim=-1).clamp_min(1e-8)
        self.dictionary /= norm[:, None]
        self.encoder *= norm[:, None]
        self.enc_bias *= norm


@torch.no_grad()
def clamp(sae: SparseAutoencoder, x: Tensor, latent: int, value: float) -> tuple[Tensor, Tensor]:
    """Clamp one latent at `value` on raw activations x (..., d): x + (value - f_i(x)) u_i, where f_i(x) is the
    latent's activation as the dictionary reads x and u_i its direction in the units of the stream (Templeton et
    al., 2024). The latent's term in the reconstruction becomes value * u_i; everything else in x, the part the
    dictionary does not reconstruct included, is left as it is. Returns the clamped activations and f_i(x)."""
    flat = x.reshape(-1, x.size(-1)).float()
    f = sae.encode(sae.normalize(flat))[:, latent]
    out = flat + (value - f)[:, None] * sae.directions()[latent]
    return out.reshape(x.shape).to(x.dtype), f.reshape(x.shape[:-1])


def decode_sparse(sae: SparseAutoencoder, values: Tensor, indices: Tensor) -> Tensor:
    """x^ from the k active latents of each token alone, (B, k) values and indices: the sum of k rows of D. The
    same numbers as sae.decode on the scattered code, without the product with the n - k zeros."""
    return F.embedding_bag(indices, sae.dictionary, per_sample_weights=values, mode="sum") + sae.bias


def topk_loss(sae: SparseAutoencoder, x: Tensor, dead: Optional[Tensor], k_aux: int) -> tuple[Tensor, Tensor, Tensor]:
    """The loss of a TopK autoencoder on a batch (Gao et al., 2024): the reconstruction error as a fraction of the
    variance, plus AUX_WEIGHT times the auxiliary error. Returns the loss, the fraction of variance unexplained,
    and which latents fired (n, bool). `dead` marks the dead latents (n, bool), or is None when there are none.

    AuxK: the k_aux dead latents with the largest pre-activations reconstruct what the live ones left over,
    e = x - x^. A dead latent is in nobody's top k, so the main loss gives it no gradient; this term does, in the
    direction that makes it useful. e is a constant here (detached): the auxiliary term must not move the live
    latents.
    """
    pre = sae.pre(x)
    top = F.relu(pre).topk(sae.cfg.k, dim=-1, sorted=False)
    error = x - decode_sparse(sae, top.values, top.indices)
    fvu = error.pow(2).sum(-1).mean() / sae.variance
    fired = torch.zeros(sae.cfg.n, device=x.device).index_add_(0, top.indices.flatten(), (top.values > 0).float().flatten()) > 0
    loss = fvu
    if dead is not None:
        aux = pre.masked_fill(~dead, -torch.inf).topk(min(k_aux, sae.cfg.n), dim=-1, sorted=False)
        values = torch.where(torch.isfinite(aux.values), aux.values, 0.0)  # fewer dead latents than k_aux: the rest add nothing
        target = error.detach()
        left = target - F.embedding_bag(aux.indices, sae.dictionary, per_sample_weights=values, mode="sum")
        spread = (target - target.mean(0)).pow(2).sum(-1).mean()
        loss = loss + AUX_WEIGHT * (left.pow(2).sum(-1).mean() / spread).nan_to_num(0.0)
    return loss, fvu.detach(), fired


def relu_loss(sae: SparseAutoencoder, x: Tensor, l1: float) -> tuple[Tensor, Tensor, Tensor]:
    """The loss of a ReLU autoencoder with an L1 penalty: the same reconstruction term plus
    l1 * sum_i z_i |D_i|. The norm of the row is in the penalty (Conerly et al., April 2024) because without it
    the penalty is escaped by shrinking z and growing D, which changes nothing in x^."""
    z = F.relu(sae.pre(x))
    error = x - sae.decode(z)
    fvu = error.pow(2).sum(-1).mean() / sae.variance
    penalty = (z * sae.dictionary.norm(dim=-1)).sum(-1).mean()
    return fvu + l1 * penalty, fvu.detach(), (z > 0).any(0)


class Trainer:
    """One dictionary being trained: its optimizer, its schedule, and the count of tokens since each latent fired.

    A step asks the card for no number back (the loss stays a tensor, and whether any latent is dead is looked up
    once in 64 steps), so that the steps of many dictionaries on one batch queue up without waiting.
    """

    def __init__(self, sae: SparseAutoencoder, steps: int, batch: int, dead_after: int, lr_scale: float = 1.0):
        self.sae, self.steps, self.batch, self.dead_after = sae, steps, batch, dead_after
        self.lr = lr_scale * 2e-4 / math.sqrt(sae.cfg.n / 2 ** 14)
        self.opt = torch.optim.Adam(sae.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=6.25e-10)
        self.warmup = min(1000, steps // 20)
        self.step_no = 0
        self.since_fired = torch.zeros(sae.cfg.n, dtype=torch.long, device=sae.dictionary.device)
        self.any_dead = False
        self.k_aux = max(1, sae.cfg.d // 2)
        self.log: list[dict] = []

    def lr_factor(self) -> float:
        """Linear warmup, constant, then linear decay to zero over the last fifth of the steps."""
        s, decay_from = self.step_no, int(0.8 * self.steps)
        if s < self.warmup:
            return (s + 1) / self.warmup
        return 1.0 if s < decay_from else max(0.0, (self.steps - s) / (self.steps - decay_from))

    def step(self, x: Tensor) -> Tensor:
        """One step on a batch of normalized activations (B, d); returns the fraction of variance unexplained."""
        sae, topk = self.sae, self.sae.cfg.kind == "topk"
        if topk:
            dead = self.since_fired > self.dead_after if self.any_dead else None
            loss, fvu, fired = topk_loss(sae, x, dead, self.k_aux)
        else:
            # The penalty is warmed up over the first twentieth of training, so that latents are not pushed to
            # zero before they have found a direction (Conerly et al., April 2024).
            warm = min(1.0, (self.step_no + 1) / max(1, self.steps // 20))
            loss, fvu, fired = relu_loss(sae, x, sae.cfg.l1 * warm)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        if topk:
            # A unit row can only turn: remove the part of its gradient that would lengthen it (Bricken et al.).
            with torch.no_grad():
                g, rows = sae.dictionary.grad, sae.dictionary
                g -= (g * rows).sum(-1, keepdim=True) * rows
        for group in self.opt.param_groups:
            group["lr"] = self.lr * self.lr_factor()
        self.opt.step()
        if topk:
            with torch.no_grad():
                sae.dictionary /= sae.dictionary.norm(dim=-1, keepdim=True)
        self.since_fired += x.size(0)
        self.since_fired.masked_fill_(fired, 0)
        self.step_no += 1
        if self.step_no % 64 == 0:
            self.any_dead = bool((self.since_fired > self.dead_after).any())
        return fvu

    def dead_fraction(self) -> float:
        return float((self.since_fired > self.dead_after).float().mean())


@torch.no_grad()
def fit_normalization(saes: Sequence[SparseAutoencoder], sample: Tensor) -> None:
    """Set each dictionary's normalization and starting bias from a sample of raw activations (N, d): the dataset
    mean, the constant that makes the mean norm sqrt(d), the variance left after centering, and the geometric
    median of the normalized sample as the bias."""
    mean = sample.mean(0)
    scale = math.sqrt(sample.size(1)) / (sample - mean).norm(dim=-1).mean()
    normalized = (sample - mean) * scale
    variance = normalized.pow(2).sum(-1).mean()
    median = geometric_median(normalized[:32768])
    for sae in saes:
        sae.mean.copy_(mean)
        sae.scale.copy_(scale)
        sae.variance.copy_(variance)
        sae.bias.copy_(median)


def train(saes: Sequence[SparseAutoencoder], stream: TrainingStream, tokens: float, dead_after: int = 1_000_000,
          lr_scale: float = 1.0, log_every: int = 500, say=print) -> list[Trainer]:
    """Train every dictionary on the same batches of the stream, `tokens` activations each."""
    steps = int(tokens) // stream.batch
    batches = iter(stream)
    first = next(batches)
    fit_normalization(saes, torch.cat([first] + [next(batches) for _ in range(31)]))
    trainers = [Trainer(sae, steps, stream.batch, dead_after, lr_scale) for sae in saes]
    start = time.time()
    for step in range(steps):
        x = next(batches)
        fvus = [t.step(t.sae.normalize(x)) for t in trainers]
        if step % log_every == 0 or step == steps - 1:
            for t, fvu in zip(trainers, fvus):
                t.log.append({"step": step, "tokens": (step + 1) * stream.batch, "fvu": float(fvu), "dead": t.dead_fraction()})
            line = "  ".join(f"{t.sae.cfg.name} {float(fvu):.3f}/{t.dead_fraction():.2f}" for t, fvu in zip(trainers, fvus))
            say(f"step {step:6d}/{steps}  {time.time() - start:5.0f}s  fvu/dead  {line}")
    for sae in saes:
        if sae.cfg.kind == "relu":
            sae.fold_norms()
    return trainers


# ----------------------------------------------------------------------------------------------------------------
# Evaluation on held-out text


@torch.no_grad()
def spliced_loss(model: GPT, idx: Tensor, x: Tensor, replacement: Tensor, harvest: Harvest) -> Tensor:
    """The summed next-token loss (nats) of a batch of windows when the stream after the harvested block is
    `replacement` at the tokens that count and the model's own x elsewhere; the targets are the tokens that
    follow the tokens that count. A few windows at a time: the logits of a window are context x vocabulary numbers."""
    keep = harvest.keep(idx)
    targets = idx[:, 1:].masked_fill(~keep[:, :-1], -1)
    per = max(1, (1 << 27) // (idx.size(1) * model.cfg.vocab_size))
    total = torch.zeros((), device=idx.device, dtype=torch.float64)
    for i in range(0, idx.size(0), per):
        rows = slice(i, i + per)
        with autocast(idx.device):
            logits = logits_from(model, torch.where(keep[rows, :, None], replacement[rows], x[rows]), harvest.layer).float()
        total += F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), targets[rows].reshape(-1), ignore_index=-1,
                                 reduction="sum")
    return total


@torch.no_grad()
def evaluate(model: GPT, saes: Sequence[SparseAutoencoder], tokens: np.ndarray, harvest: Harvest,
             n_windows: Optional[int] = None, n_splice_windows: int = 2048, model_batch: int = 64,
             device: str = "cuda", chunk: int = 4096) -> list[dict]:
    """Every dictionary on the same held-out windows: fvu, l0, dead and dense latents, the density of every
    latent, the gain that undoes shrinkage, and (on the first `n_splice_windows`) the model's loss with the
    reconstruction spliced in. Sets each dictionary's `threshold`."""
    n, d = len(saes), saes[0].cfg.d
    err = torch.zeros(n, device=device, dtype=torch.float64)  # sum |x - x^|^2
    own = torch.zeros(n, device=device, dtype=torch.float64)  # sum |x - b|^2
    dot = torch.zeros(n, device=device, dtype=torch.float64)  # sum <x - b, x^ - b>
    rec = torch.zeros(n, device=device, dtype=torch.float64)  # sum |x^ - b|^2
    active = torch.zeros(n, device=device, dtype=torch.float64)  # sum of active latents
    least = torch.zeros(n, device=device, dtype=torch.float64)  # sum over tokens of the smallest active value
    fires = [torch.zeros(s.cfg.n, device=device, dtype=torch.float64) for s in saes]
    square = torch.zeros((), device=device, dtype=torch.float64)  # sum |x|^2, in normalized units
    first_moment = torch.zeros(d, device=device, dtype=torch.float64)  # sum x, in normalized units
    spliced = torch.zeros(n, device=device, dtype=torch.float64)
    clean = zero = mean_abl = torch.zeros((), device=device, dtype=torch.float64)
    count = n_targets = 0
    for first, idx in window_batches(tokens, harvest.context, model_batch, n_windows, device):
        x = activations(model, idx, harvest)
        keep = harvest.keep(idx)
        flat = x[keep]
        count += len(flat)
        splice = first < n_splice_windows
        if splice:
            n_targets += int(keep[:, :-1].sum())
            clean = clean + spliced_loss(model, idx, x, x, harvest)
            zero = zero + spliced_loss(model, idx, x, torch.zeros_like(x), harvest)
            mean_abl = mean_abl + spliced_loss(model, idx, x, saes[0].mean.expand_as(x), harvest)
        for j, sae in enumerate(saes):
            normalized = sae.normalize(flat)
            if j == 0:
                square += normalized.pow(2).sum()
                first_moment += normalized.sum(0)
            parts = []
            for i in range(0, len(normalized), chunk):  # a few thousand tokens at a time: a code is n numbers a token
                xn = normalized[i:i + chunk]
                xhat, z = sae(xn)
                on = z > 0
                err[j] += (xn - xhat).pow(2).sum()
                own[j] += (xn - sae.bias).pow(2).sum()
                dot[j] += ((xn - sae.bias) * (xhat - sae.bias)).sum()
                rec[j] += (xhat - sae.bias).pow(2).sum()
                active[j] += on.sum()
                least[j] += z.masked_fill(~on, torch.inf).amin(-1).nan_to_num(posinf=0.0).sum()
                fires[j] += on.sum(0)
                parts.append(xhat)
            if splice:
                replacement = torch.zeros_like(x)
                replacement[keep] = sae.restore(torch.cat(parts))
                spliced[j] += spliced_loss(model, idx, x, replacement, harvest)
    H, H0, Hm = float(clean) / n_targets, float(zero) / n_targets, float(mean_abl) / n_targets
    total = square - first_moment.pow(2).sum() / count  # sum |x - mean|^2 about the mean of these held-out activations
    out = []
    for j, sae in enumerate(saes):
        density = (fires[j] / count).cpu().numpy()
        Hs = float(spliced[j]) / n_targets
        gain = float(dot[j] / rec[j]) if float(rec[j]) > 0 else 1.0  # a dictionary that reconstructs nothing has no gain
        sae.threshold.fill_(float(least[j]) / count)
        logd = np.log10(np.maximum(density[density > 0], 1e-12))
        out.append({
            **asdict(sae.cfg), "name": sae.cfg.name,
            "fvu": float(err[j] / total), "l0": float(active[j]) / count, "gain": gain,
            "fvu_rescaled": float((own[j] - gain * dot[j]) / total),
            "dead": float((density == 0).mean()), "dense": float((density > DENSE).mean()),
            "threshold": float(sae.threshold),
            "loss_clean": H, "loss_spliced": Hs, "loss_zero": H0, "loss_mean": Hm,
            "loss_added": Hs - H, "loss_recovered": (H0 - Hs) / (H0 - H), "loss_recovered_mean": (Hm - Hs) / (Hm - H),
            "density_hist": np.histogram(logd, bins=28, range=(-7.0, 0.0))[0].tolist(),
            "eval_tokens": count, "splice_targets": n_targets,
        })
    return out


# ----------------------------------------------------------------------------------------------------------------
# Files


def dict_dir(run_name: str, layer: int) -> Path:
    """Where the dictionaries of a run's layer are kept: $GPTPP_WORK/sae/<run>-L<layer>/."""
    return work_dir() / "sae" / f"{run_name}-L{layer}"


def save(sae: SparseAutoencoder, folder: Path, record: dict) -> None:
    """Write `<name>.pt` (the weights and the configuration) and `<name>.json` (how it was trained and what it
    scored), so that an exporter can list a folder's dictionaries without loading them."""
    folder.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(sae.cfg), "state": sae.state_dict()}, folder / f"{sae.cfg.name}.pt")
    (folder / f"{sae.cfg.name}.json").write_text(json.dumps(record, indent=1) + "\n")


def load(path: str | Path, device: str = "cuda") -> SparseAutoencoder:
    """A dictionary written by `save`, in evaluation mode on `device`."""
    blob = torch.load(path, map_location="cpu", weights_only=True)
    sae = SparseAutoencoder(DictConfig(**blob["config"]))
    sae.load_state_dict(blob["state"])
    return sae.to(device).eval()


def describe(score: dict) -> str:
    """One line of a dictionary's scores."""
    return (f"{score['name']:22s} fvu {score['fvu']:.4f} ({score['fvu_rescaled']:.4f} rescaled by {score['gain']:.3f})  "
            f"l0 {score['l0']:6.1f}  loss +{score['loss_added']:.4f} (recovered {score['loss_recovered']:.4f}, "
            f"against the mean {score['loss_recovered_mean']:.4f})  dead {score['dead']:.4f}  dense {score['dense']:.4f}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Train and evaluate sparse dictionaries on a run's residual stream.")
    ap.add_argument("--run", required=True, help="a run directory, or a run's name under $GPTPP_WORK/ckpt")
    ap.add_argument("--ckpt", default=None, help="a checkpoint file of the run (default: model.pt, or the latest)")
    ap.add_argument("--layer", type=int, required=True, help="read the stream after this block (resid_post.<layer>)")
    ap.add_argument("--widths", default="4096", help="latents, comma separated")
    ap.add_argument("--ks", default="32", help="active latents a token for the TopK dictionaries, comma separated")
    ap.add_argument("--l1", default="", help="L1 weights for ReLU dictionaries at each width, comma separated")
    ap.add_argument("--tokens", type=float, default=100e6, help="activations each dictionary is trained on")
    ap.add_argument("--batch", type=int, default=2048, help="activations a step")
    ap.add_argument("--context", type=int, default=0, help="tokens a window (default: the model's block size)")
    ap.add_argument("--skip", type=int, default=16, help="leave out the first tokens of every window")
    ap.add_argument("--buffer", type=float, default=1e6, help="activations shuffled together")
    ap.add_argument("--model-batch", type=int, default=0, help="windows a forward pass (default: about 32k tokens)")
    ap.add_argument("--dead-after", type=float, default=1e6, help="tokens without firing after which a latent is dead")
    ap.add_argument("--lr-scale", type=float, default=4.0, help="multiplies the learning rate 2e-4 / sqrt(n / 2^14)")
    ap.add_argument("--eval-windows", type=int, default=0, help="held-out windows for evaluation (default: all)")
    ap.add_argument("--splice-windows", type=int, default=2048, help="held-out windows for the spliced loss")
    ap.add_argument("--log-every", type=int, default=500, help="steps between lines of the log")
    ap.add_argument("--evaluate-only", action="store_true",
                    help="train nothing: evaluate again every dictionary already in the folder and rewrite its scores")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="folder for the dictionaries (default: $GPTPP_WORK/sae/<run>-L<layer>)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model, data, source = load_run(args.run, args.ckpt, args.device)
    context = args.context or model.cfg.block_size
    harvest = Harvest(args.layer, context, args.skip, int(data.meta["bos_id"]))
    model_batch = args.model_batch or max(1, 32768 // context)
    folder = Path(args.out) if args.out else dict_dir(source["run"], args.layer)
    if args.evaluate_only:
        names = sorted(p.stem for p in folder.glob("*.pt"))
        saes = [load(folder / f"{name}.pt", args.device) for name in names]
        scores = evaluate(model, saes, data.val_tokens(), harvest, args.eval_windows or None, args.splice_windows,
                          model_batch, args.device)
        for sae, score in zip(saes, scores):
            record = json.loads((folder / f"{sae.cfg.name}.json").read_text())
            save(sae, folder, {**record, **source, "eval": score})
            print(describe(score), flush=True)
        return
    torch.manual_seed(args.seed)
    configs = [DictConfig(model.cfg.d_model, n, "topk", k, 0.0, f"resid_post.{args.layer}")
               for n in map(int, args.widths.split(",")) for k in map(int, filter(None, args.ks.split(",")))]
    configs += [DictConfig(model.cfg.d_model, n, "relu", 0, l1, f"resid_post.{args.layer}")
                for n in map(int, args.widths.split(",")) for l1 in map(float, filter(None, args.l1.split(",")))]
    saes = [SparseAutoencoder(c).to(args.device) for c in configs]
    stream = TrainingStream(model, data, harvest, args.batch, max(1, int(args.buffer) // context), model_batch,
                            args.seed, args.device)
    print(f"{source['run']} ({source['checkpoint']}), resid_post.{args.layer}, {len(saes)} dictionaries, "
          f"{args.tokens:.3g} tokens each, windows of {context}", flush=True)
    start = time.time()
    trainers = train(saes, stream, args.tokens, int(args.dead_after), args.lr_scale, args.log_every,
                     say=lambda s: print(s, flush=True))
    train_seconds = time.time() - start
    for sae in saes:
        sae.eval()
    scores = evaluate(model, saes, data.val_tokens(), harvest, args.eval_windows or None, args.splice_windows,
                      model_batch, args.device)
    for sae, trainer, score in zip(saes, trainers, scores):
        record = {**source, "layer": args.layer, "context": context, "skip": args.skip, "train_tokens": args.tokens,
                  "batch": args.batch, "lr": trainer.lr, "dead_after": args.dead_after, "seed": args.seed,
                  "train_seconds_all": round(train_seconds, 1), "eval": score, "log": trainer.log}
        save(sae, folder, record)
        print(describe(score), flush=True)
    print(f"clean loss {scores[0]['loss_clean']:.4f}, zero {scores[0]['loss_zero']:.3f}, mean {scores[0]['loss_mean']:.3f}; "
          f"{scores[0]['eval_tokens']} held-out tokens; trained in {train_seconds:.0f} s; "
          f"peak memory {torch.cuda.max_memory_allocated() / 1e9 if args.device == 'cuda' else 0:.1f} GB; wrote {folder}")


if __name__ == "__main__":
    main()
