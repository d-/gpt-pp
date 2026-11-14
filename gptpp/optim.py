"""Optimizers and learning-rate schedules.

The parameters of a GPT fall into four groups that are worth treating differently (GPT.param_groups):

    embedding      a lookup table: each row is updated only when its token occurs
    unembedding    the last linear map, straight into the loss
    matrices       the hidden linear maps of attention and the MLPs
    gains          gains and biases: norms, linear biases, learned attention scales

`build_optimizers` gives each group its own learning rate and weight decay and returns the optimizers chosen by
TrainConfig.optimizer:

    sgd          momentum SGD: one step size for every coordinate.
    adam         Adam (Kingma and Ba, 2014, arXiv:1412.6980): each coordinate's step is divided by a running
                 root-mean-square of its gradient. Weight decay, if any, is the old L2 penalty added to the
                 gradient, which Adam then rescales along with it.
    adamw        AdamW (Loshchilov and Hutter, 2017, arXiv:1711.05101): the decay is taken out of the
                 gradient and applied to the weights directly, so that it shrinks every weight at the same
                 relative rate. The optimizer of GPT-2-era recipes and still the baseline.
    lion         Lion (Chen et al., 2023, arXiv:2302.06675): the sign of an interpolation of gradient and
                 momentum; found by program search, it keeps one buffer instead of Adam's two.
    muon+adamw   Muon (Jordan et al., 2024) on the hidden matrices, AdamW on everything else.
    X+adamw      for X in adamw, adam, lion, sgd: the recipe with one line changed. The hidden matrices are trained
                 by X at the rate `lr` (with `weight_decay`, `beta1`, `beta2`, `momentum` as given); the embedding,
                 the unembedding and the gains keep the recipe's AdamW at their own rates, with the recipe's betas
                 and epsilon whatever `beta1` and `beta2` say. This is the fair way to compare a rule with Muon on
                 the 2026 model: with one rate for every group (the plain names above) the normalized embedding,
                 which needs a rate hundreds of times larger than a hidden matrix, does not train.

Schedules are functions of the number of tokens seen (`lr_multiplier`), not of the step count.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor

from .config import TrainConfig

GROUPS = ("embedding", "unembedding", "matrices", "gains")

# Coefficients (a, b, c) of the Newton-Schulz step  X <- a X + b (X X^T) X + c (X X^T)^2 X.
NS_CLASSIC = (2.0, -1.5, 0.5)  # converges to singular values of exactly 1, slowly for the small ones
NS_TUNED = (3.4445, -4.7750, 2.0315)  # Jordan et al.: a steeper start (a = 3.44); ends in [0.68, 1.20], not at 1


def orthogonalize(g: Tensor, steps: int = 5, coefficients: tuple[float, float, float] = NS_TUNED) -> Tensor:
    """Push every singular value of g (..., m, n) towards one, leaving its singular vectors alone.

    If g = U S V^T, the result approximates U V^T, the semi-orthogonal matrix nearest to g. No decomposition
    is computed: one Newton-Schulz step maps g to a g + b (g g^T) g + c (g g^T)^2 g = U (a S + b S^3 + c S^5) V^T,
    an odd polynomial applied to the singular values only, and a few steps of a well-chosen polynomial send
    every s in (0, 1] near 1. Dividing by the Frobenius norm first puts all singular values in [0, 1]. The
    whole thing is a handful of matrix products, which run in bfloat16 on a GPU.

    With the tuned coefficients, five steps send every singular value above about 0.0015 of the Frobenius norm
    into [0.68, 1.20] rather than exactly to 1, which costs nothing in training (Jordan et al., 2024), and
    multiply the smaller ones by up to a^5 = 485.
    """
    a, b, c = coefficients
    x = g.bfloat16() if g.is_cuda else g
    tall = x.size(-2) > x.size(-1)
    if tall:  # work with the wide orientation so that the Gram matrix x x^T is the small one
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        gram = x @ x.mT
        x = a * x + (b * gram + c * gram @ gram) @ x
    if tall:
        x = x.mT
    return x.to(g.dtype)


class Muon(torch.optim.Optimizer):
    """Muon: momentum, orthogonalized by Newton-Schulz (Jordan et al., 2024,
    https://kellerjordan.github.io/posts/muon/; the reference implementations are modded-nanogpt and nanochat).

    For a weight matrix W with gradient G, at each step:

        M <- momentum * M + (1 - momentum) * G           a running average of gradients
        U <- orthogonalize(G + momentum * (M - G))       Nesterov: the average, looked ahead by one step
        W <- W - lr * sqrt(max(1, rows / columns)) * U

    Adam divides each coordinate of the update by its own scale. Muon instead equalizes the update's singular
    values: the raw gradient of a hidden matrix is close to low rank, a few directions dominate it, and
    orthogonalizing gives every direction of the update the same size. The step is then the steepest descent
    direction under the spectral norm (Bernstein and Newhouse, 2024, arXiv:2409.20325), the norm that measures
    how much a matrix can change the vectors passing through it; `lr` is that change per step, whatever the
    size of the gradient. The factor sqrt(rows / columns) keeps the size of the change in the output per unit
    of input the same for matrices that widen the stream.

    It is for hidden matrices only: embeddings, the unembedding, gains and biases stay with AdamW. Matrices of
    the same shape are stacked and orthogonalized in one batched call.
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95, nesterov: bool = True, ns_steps: int = 5,
                 weight_decay: float = 0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                                      weight_decay=weight_decay))
        for group in self.param_groups:
            if any(p.ndim != 2 for p in group["params"]):
                raise ValueError("Muon is for matrices: every parameter must have two dimensions")
        self._momentum: dict[tuple[int, torch.Size], Tensor] = {}

    @torch.no_grad()
    def step(self, closure=None) -> None:
        for index, group in enumerate(self.param_groups):
            by_shape: dict[torch.Size, list[Tensor]] = {}
            for p in group["params"]:
                by_shape.setdefault(p.shape, []).append(p)
            for shape, params in by_shape.items():
                grad = torch.stack([p.grad for p in params])  # (K, rows, columns)
                buffer = self._momentum.setdefault((index, shape), torch.zeros_like(grad))
                buffer.lerp_(grad, 1 - group["momentum"])
                update = grad.lerp_(buffer, group["momentum"]) if group["nesterov"] else buffer
                update = orthogonalize(update, group["ns_steps"])
                step_size = group["lr"] * max(1.0, shape[0] / shape[1]) ** 0.5
                for p, u in zip(params, update.unbind(0)):
                    if group["weight_decay"] > 0:
                        p.mul_(1 - group["lr"] * group["weight_decay"])  # decoupled, as in AdamW
                    p.add_(u, alpha=-step_size)


class Lion(torch.optim.Optimizer):
    """Lion (Chen et al., 2023, "Symbolic discovery of optimization algorithms", arXiv:2302.06675).

        update <- sign(beta1 * M + (1 - beta1) * G)
        M      <- beta2 * M + (1 - beta2) * G
        W      <- W * (1 - lr * weight_decay) - lr * update

    Every coordinate moves by exactly lr each step. The paper recommends a learning rate three to ten times
    smaller than AdamW's and a weight decay as many times larger, with betas (0.9, 0.99).
    """

    def __init__(self, params, lr: float = 1e-4, betas: tuple[float, float] = (0.9, 0.99), weight_decay: float = 0.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None) -> None:
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["momentum"] = torch.zeros_like(p)
                momentum = state["momentum"]
                update = momentum.lerp(p.grad, 1 - beta1).sign_()
                momentum.lerp_(p.grad, 1 - beta2)
                if group["weight_decay"] > 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"])


def group_settings(cfg: TrainConfig, tokens_per_step: Optional[float] = None) -> dict[str, dict[str, float]]:
    """Learning rate and weight decay of each parameter group.

    A rate left as None follows `lr`. Gains and biases are never decayed. If `lr_ref_batch` is set, the rates
    are quoted for a step of that many tokens and are multiplied by sqrt(tokens_per_step / lr_ref_batch): a
    larger batch gives a less noisy gradient and supports a larger step, by about the square root for
    sign-like optimizers such as Adam and Muon.
    """
    scale = 1.0
    if cfg.lr_ref_batch > 0 and tokens_per_step is not None:
        scale = math.sqrt(tokens_per_step / cfg.lr_ref_batch)

    def rate(value: Optional[float]) -> float:
        return scale * (cfg.lr if value is None else value)

    return {
        "embedding": dict(lr=rate(cfg.lr_embedding), weight_decay=cfg.wd_embedding),
        "unembedding": dict(lr=rate(cfg.lr_unembedding), weight_decay=cfg.wd_unembedding),
        "matrices": dict(lr=rate(cfg.lr), weight_decay=cfg.weight_decay),
        "gains": dict(lr=rate(cfg.lr_gains), weight_decay=0.0),
    }


def build_optimizers(model, cfg: TrainConfig, tokens_per_step: Optional[float] = None) -> list[torch.optim.Optimizer]:
    """The optimizers of a run. Every parameter group carries its name and its `base_lr`, the peak rate that
    `set_lr` multiplies by the schedule.

    When the embedding and the unembedding are one tied matrix, it is trained with the unembedding's rate and
    decay: it feeds the loss directly, and the much larger rate that suits a lookup table makes it diverge.
    """
    params = model.param_groups()
    settings = group_settings(cfg, tokens_per_step)
    if not params["unembedding"]:
        settings["embedding"] = settings["unembedding"]

    def groups(names: tuple[str, ...]) -> list[dict]:
        return [dict(params=params[name], name=name, base_lr=settings[name]["lr"], **settings[name])
                for name in names if params[name]]

    fused = all(p.is_cuda for group in params.values() for p in group)
    betas = (cfg.beta1, cfg.beta2)
    if cfg.optimizer == "adamw":
        return [torch.optim.AdamW(groups(GROUPS), betas=betas, eps=cfg.eps, fused=fused)]
    if cfg.optimizer == "adam":
        return [torch.optim.Adam(groups(GROUPS), betas=betas, eps=cfg.eps, fused=fused)]
    if cfg.optimizer == "sgd":
        return [torch.optim.SGD(groups(GROUPS), momentum=cfg.momentum, nesterov=cfg.nesterov and cfg.momentum > 0)]
    if cfg.optimizer == "lion":
        return [Lion(groups(GROUPS), betas=betas)]
    if cfg.optimizer == "muon+adamw":
        adamw = torch.optim.AdamW(groups(("embedding", "unembedding", "gains")), betas=betas, eps=cfg.eps, fused=fused)
        muon = Muon(groups(("matrices",)), momentum=cfg.momentum, nesterov=cfg.nesterov, ns_steps=cfg.ns_steps)
        return [adamw, muon]
    if cfg.optimizer.endswith("+adamw"):
        recipe = TrainConfig()  # the tables and gains keep the recipe's AdamW; beta1, beta2 and eps of cfg go to the matrices' rule
        rest = torch.optim.AdamW(groups(("embedding", "unembedding", "gains")), betas=(recipe.beta1, recipe.beta2),
                                 eps=recipe.eps, fused=fused)
        matrices, rule = groups(("matrices",)), cfg.optimizer[: -len("+adamw")]
        if rule == "adamw":
            return [rest, torch.optim.AdamW(matrices, betas=betas, eps=cfg.eps, fused=fused)]
        if rule == "adam":
            return [rest, torch.optim.Adam(matrices, betas=betas, eps=cfg.eps, fused=fused)]
        if rule == "lion":
            return [rest, Lion(matrices, betas=betas)]
        if rule == "sgd":
            return [rest, torch.optim.SGD(matrices, momentum=cfg.momentum, nesterov=cfg.nesterov and cfg.momentum > 0)]
    raise ValueError(f"unknown optimizer {cfg.optimizer!r}")


def lr_multiplier(tokens: float, total_tokens: float, cfg: TrainConfig) -> float:
    """The learning rate after `tokens` of a budget of `total_tokens`, as a fraction of its peak.

    A linear warmup from zero over `warmup_tokens` (the first steps of Adam divide by a second-moment estimate
    built from very few gradients, and are best taken small), then one of:

        constant    1
        cosine      half a cosine from 1 down to final_lr_frac at the end of the budget
        linear      a straight line from 1 down to final_lr_frac
        trapezoid   1 until the last `cooldown_frac` of the budget, then a straight line down to final_lr_frac
    """
    tokens = min(max(tokens, 0.0), total_tokens)
    if tokens < cfg.warmup_tokens:
        return tokens / cfg.warmup_tokens
    if cfg.schedule == "constant":
        return 1.0
    if cfg.schedule == "trapezoid":
        start = max(cfg.warmup_tokens, total_tokens * (1 - cfg.cooldown_frac))
    else:
        start = cfg.warmup_tokens
    progress = (tokens - start) / max(total_tokens - start, 1e-9)  # 0 when the decay begins, 1 at the end
    if progress <= 0:
        return 1.0
    if cfg.schedule == "cosine":
        decay = 0.5 * (1 + math.cos(math.pi * progress))
    else:  # linear, and the falling edge of the trapezoid
        decay = 1 - progress
    return cfg.final_lr_frac + (1 - cfg.final_lr_frac) * decay


def set_lr(optimizers: list[torch.optim.Optimizer], multiplier: float) -> None:
    """Set every group's learning rate to its peak times the schedule's multiplier."""
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["lr"] = group["base_lr"] * multiplier
