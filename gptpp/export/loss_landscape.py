"""The loss around a trained model on a plane that holds its training path: `loss-landscape.json` for fig-06-07.

    CUDA_VISIBLE_DEVICES=3 python -m gptpp.export.loss_landscape --ckpt $GPTPP_WORK/ckpt/pico [--grid 61] [--tokens 262144]

A run saved with `--save-log N` leaves the weights at the initialization, at N log-spaced steps and at the end.
Each is one point of a space with as many dimensions as the model has parameters (2.2 million for pico); the
path of training is the polyline through them. To look at the loss around the trained model one has to choose
two directions. Random ones (Li et al., 2018, "Visualizing the loss landscape of neural nets",
arXiv:1712.09913, Section 4) show the basin but not the path, which is almost orthogonal to any random plane.
The plane used here contains the path's two ends and as much of the rest as a plane can:

    origin   the trained weights
    e1       the unit vector from the trained weights to the initialization
    e2       the first principal direction of what is left of the checkpoints' differences from the trained
             weights once their e1 component is taken out (uncentered, as in Li et al., Section 7, who take the
             principal components of those differences; fixing e1 instead keeps both ends exactly on the plane,
             so the surface shows the loss the run started from and the loss it ended at)

    theta(a, b) = theta_final + a e1 + b e2;   the initialization is (L, 0) and the trained model (0, 0)

Distances. Parameter space is measured after one change of scale: every row of the embedding matrix is divided
by its root mean square. The model applies a parameter-free RMSNorm to each token's embedding before anything
reads it, so the length of a row changes nothing the model computes, and it is not small: under the 2026
recipe (AdamW at rate 0.73 on the embedding, no weight decay) the rows of pico grow 45-fold, and that growth
alone is 99.9% of the raw distance from initialization to trained weights. Measured raw, the plane would show
the size of the embedding against everything else. This is the point of Li et al.'s filter normalization
(scale-invariant weights must be compared at one scale), applied to the one exact per-row invariance of this
model; it is done only when the model has it (`embed_norm`, `norm=rmsnorm_fixed`, no additive positions).

What is written (assets/data/gpt/loss-landscape.json, about 15 kB):

    grid     nx x ny held-out losses z[j][i] at a = a_lo + i h, b = b_lo + j h (one spacing h for both axes, in
             units of L; the path lies inside with a margin), on the first `--tokens` held-out tokens, in the
             precision the model trains in. If the grid would take longer than `--seconds`, the number of tokens
             is halved until it does not
    path     for every checkpoint: step, tokens seen, its coordinates (a, b), its distance from the plane, its
             own loss, and the loss of the point of the plane under it (its projection); these two on all the
             held-out tokens the trainer evaluates on (`--path-tokens`), so that the two ends carry the losses
             the run logged
    share    how much of the path the plane holds, three ways (see `shares`)
    pca      for comparison: the shares of the first principal components taken freely (Li et al.'s plane)
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import torch

from ..data import TokenData, ValLoader
from ..model import GPT
from .common import CKPT, DATA, write_json

NAME = "loss-landscape"


def load_state(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return {re.sub(r"^(module\.|_orig_mod\.)+", "", k): v for k, v in checkpoint["model"].items()}


def row_scale_is_free(cfg: dict) -> bool:
    """Is the length of an embedding row ignored by the model? Yes when a parameter-free RMSNorm reads the
    embedding and nothing is added to it before (learned and sinusoidal positions are added before the norm)."""
    return bool(cfg.get("embed_norm")) and cfg.get("norm") == "rmsnorm_fixed" and cfg.get("pos") in ("rope", "alibi", "none") \
        and not cfg.get("tie_embeddings")


def flatten(state: dict[str, torch.Tensor], keys: list[str], unit_rows: bool) -> torch.Tensor:
    parts = []
    for key in keys:
        w = state[key].double()
        if unit_rows and key == "embed.weight":
            w = w / w.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-12)
        parts.append(w.reshape(-1))
    return torch.cat(parts)


def plane(points: torch.Tensor) -> dict:
    """The plane through the last point (the trained weights) and the first (the initialization) that holds the
    most of the others. points: (n, d), in training order. Returns e1, e2 (unit, orthogonal), L, and the
    coordinates (a, b) and distance from the plane of every point."""
    diff = points - points[-1]
    length = diff[0].norm()
    e1 = diff[0] / length
    a = diff @ e1
    rest = diff - a[:, None] * e1[None, :]
    _, s, vt = torch.linalg.svd(rest[:-1], full_matrices=False)
    e2 = vt[0]
    b = diff @ e2
    if b[b.abs().argmax()] < 0:  # the path bows towards positive b
        e2, b = -e2, -b
    off = (diff - a[:, None] * e1 - b[:, None] * e2).norm(dim=1)
    return {"e1": e1, "e2": e2, "L": float(length), "a": a, "b": b, "off": off, "rest_singular": s}


def shares(points: torch.Tensor, e1: torch.Tensor, e2: torch.Tensor) -> dict:
    """How much of the path lies in the plane.

        displacement   sum_i |P d_i|^2 / sum_i |d_i|^2 with d_i the difference of checkpoint i from the trained
                       weights and P the projection on the plane: the quantity principal components maximize
                       (Li et al. report it as the share of variation their two axes capture)
        steps          sum_i |P s_i|^2 / sum_i |s_i|^2 with s_i the move from one checkpoint to the next
        length         length of the projected polyline over the length of the polyline through the checkpoints
    """
    diff = points - points[-1]
    proj = lambda v: torch.stack([v @ e1, v @ e2], dim=1)  # noqa: E731
    moves = points[1:] - points[:-1]
    pd, pm = proj(diff[:-1]), proj(moves)
    return {
        "displacement": float(pd.pow(2).sum() / diff[:-1].pow(2).sum()),
        "displacement_e1": float(pd[:, 0].pow(2).sum() / diff[:-1].pow(2).sum()),
        "steps": float(pm.pow(2).sum() / moves.pow(2).sum()),
        "length": float(pm.norm(dim=1).sum() / moves.norm(dim=1).sum()),
        "path_length_over_L": float(moves.norm(dim=1).sum() / diff[0].norm()),
    }


def pca_shares(points: torch.Tensor, k: int = 4) -> dict:
    diff = (points - points[-1])[:-1]
    _, s, vt = torch.linalg.svd(diff, full_matrices=False)
    var = s.pow(2) / s.pow(2).sum()
    moves = points[1:] - points[:-1]
    first = float(((moves @ vt[0]).pow(2) + (moves @ vt[1]).pow(2)).sum() / moves.pow(2).sum())
    return {"components": [float(v) for v in var[:k]], "two": float(var[:2].sum()), "steps_two": first,
            "init_off_plane_over_L": float((diff[0] - (diff[0] @ vt[0]) * vt[0] - (diff[0] @ vt[1]) * vt[1]).norm() / diff[0].norm())}


class Evaluator:
    """The held-out loss of the model with a given flat parameter vector."""

    def __init__(self, model: GPT, keys: list[str], shapes: list[torch.Size], data: TokenData, seq_len: int,
                 batch: int, tokens: int, device: torch.device):
        self.model, self.keys, self.shapes, self.device = model, keys, shapes, device
        self.loader = ValLoader(data, seq_len, batch, 0, 1, tokens, device)
        self.autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")
        self.params = dict(model.named_parameters())
        self.n_tokens = sum(n for _, _, n in self.loader)

    @torch.no_grad()
    def __call__(self, flat: torch.Tensor) -> float:
        offset = 0
        for key, shape in zip(self.keys, self.shapes):
            n = math.prod(shape)
            self.params[key].copy_(flat[offset:offset + n].view(shape))
            offset += n
        total, count = 0.0, 0
        for x, y, n in self.loader:
            with self.autocast:
                loss = self.model(x, y)
            total += float(loss) * n
            count += n
        return total / count


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", default=str(CKPT / "pico"), help="a run directory saved with --save-log")
    ap.add_argument("--grid", type=int, default=61, help="points along the axis from the trained weights to the initialization")
    ap.add_argument("--seconds", type=float, default=420.0, help="time the grid may take; fewer tokens are used if it would take longer")
    ap.add_argument("--tokens", type=int, default=262144, help="held-out tokens every loss of the grid is measured on")
    ap.add_argument("--path-tokens", type=int, default=0, help="held-out tokens for the checkpoints (default: the trainer's)")
    ap.add_argument("--margin", type=float, default=0.25, help="room around the path, in units of L")
    ap.add_argument("--batch", type=int, default=128, help="sequences per forward pass")
    ap.add_argument("--raw", action="store_true", help="measure distances on the raw weights (no row rescaling)")
    ap.add_argument("--out", default=str(DATA))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    run = Path(args.ckpt)
    config = json.loads((run / "config.json").read_text())
    model_cfg, derived = config["model_config"], config["derived"]
    device = torch.device(args.device)

    # ---- the checkpoints as points
    files = {int(re.fullmatch(r"ckpt_(\d+)\.pt", p.name).group(1)): p for p in run.glob("ckpt_*.pt")}
    files[derived["steps"]] = run / "model.pt"
    steps = sorted(files)
    states = [load_state(files[s]) for s in steps]
    model, _ = GPT.from_checkpoint(run / "model.pt", device)
    keys = [k for k, _ in model.named_parameters()]
    shapes = [p.shape for _, p in model.named_parameters()]
    unit_rows = row_scale_is_free(model_cfg) and not args.raw
    points = torch.stack([flatten(s, keys, unit_rows) for s in states])
    raw = torch.stack([flatten(s, keys, False) for s in states])
    pl = plane(points)
    e1, e2, L = pl["e1"], pl["e2"], pl["L"]
    share, share_raw = shares(points, e1, e2), None
    pr = plane(raw)
    share_raw = shares(raw, pr["e1"], pr["e2"])
    embed = slice(0, states[0]["embed.weight"].numel()) if keys[0] == "embed.weight" else None
    raw_embed = float((raw[0] - raw[-1])[embed].pow(2).sum() / (raw[0] - raw[-1]).pow(2).sum()) if embed else None
    rms = [float(s["embed.weight"].double().pow(2).mean().sqrt()) for s in (states[0], states[-1])]

    # ---- the grid: one spacing for both axes, the path inside with a margin
    a, b = pl["a"] / L, pl["b"] / L
    nx = args.grid
    a_lo, a_hi = float(a.min()) - args.margin, float(a.max()) + args.margin
    h = (a_hi - a_lo) / (nx - 1)
    b_lo = float(b.min()) - args.margin
    ny = math.ceil((float(b.max()) + args.margin - b_lo) / h - 1e-9) + 1
    b_hi = b_lo + (ny - 1) * h
    data = TokenData(config["train_config"]["data"])
    origin = points[-1].to(device)
    d1, d2 = (e1 * L).to(device), (e2 * L).to(device)
    tokens = args.tokens
    while True:  # time a few points; if the whole grid would not fit in the time allowed, use half the tokens
        evaluate = Evaluator(model, keys, shapes, data, derived["seq_len"], args.batch, tokens, device)
        evaluate(origin.float())
        start = time.time()
        for k in range(8):
            evaluate((origin + (a_lo + k * h) * d1).float())
        each = (time.time() - start) / 8
        if each * (nx * ny + 2 * len(steps)) <= args.seconds or tokens <= 16384:
            break
        tokens //= 2
    print(f"{nx} x {ny} points, spacing {h:.4f} L, {evaluate.n_tokens:,} held-out tokens, about {each * nx * ny:.0f} s", flush=True)
    start = time.time()
    z = np.zeros((ny, nx))
    for j in range(ny):
        bj = b_lo + j * h
        for i in range(nx):
            z[j, i] = evaluate((origin + (a_lo + i * h) * d1 + bj * d2).float())
        if j % 6 == 0 or j == ny - 1:
            print(f"row {j + 1}/{ny}: b = {bj:+.3f} L, loss {z[j].min():.3f} to {z[j].max():.3f}, {time.time() - start:.0f} s", flush=True)

    # ---- the checkpoints: their own loss, and the loss of the plane under them, on the trainer's held-out tokens
    grid_tokens = evaluate.n_tokens
    path_tokens = args.path_tokens or int(config["train_config"]["val_tokens"])
    evaluate = Evaluator(model, keys, shapes, data, derived["seq_len"], args.batch, path_tokens, device)
    path = []
    for k, step in enumerate(steps):
        own = evaluate(points[k].to(device).float())
        under = evaluate((origin + float(a[k]) * d1 + float(b[k]) * d2).float())
        path.append({"step": step, "tokens": step * derived["tokens_per_step"], "a": float(a[k]), "b": float(b[k]),
                     "off": float(pl["off"][k] / L), "loss": own, "plane_loss": under})
    log = [json.loads(line) for line in (run / "log.jsonl").read_text().splitlines()]
    logged = {r["step"]: r["val_loss"] for r in log if r["kind"] == "val"}
    out = {
        "run": config["run"], "preset": config["preset"], "data": config["train_config"]["data"],
        "params": config["params"]["total"], "vocab_size": model_cfg["vocab_size"], "uniform_loss": math.log(model_cfg["vocab_size"]),
        "steps": derived["steps"], "tokens": derived["tokens"], "checkpoints": len(steps),
        "held_out_tokens": grid_tokens, "path_held_out_tokens": evaluate.n_tokens, "logged_final_loss": logged.get(derived["steps"]),
        "unit_rows": unit_rows, "L": L, "L_raw": pr["L"], "raw_embedding_share_of_distance": raw_embed,
        "embedding_rms": {"init": rms[0], "trained": rms[1]},
        "grid": {"nx": nx, "ny": ny, "h": h, "a": [a_lo, a_hi], "b": [b_lo, b_hi],
                 "z": [[float(v) for v in row] for row in z], "min": float(z.min()), "max": float(z.max())},
        "path": path, "share": share, "share_raw": share_raw, "pca": pca_shares(points), "pca_raw": pca_shares(raw),
        "note": "a, b and off are in units of L, the distance from the initialization to the trained weights; "
                "z[j][i] is the held-out loss at a = a[0] + i h, b = b[0] + j h, on held_out_tokens tokens; the losses of "
                "the path are on path_held_out_tokens tokens",
    }
    Path(args.out).mkdir(parents=True, exist_ok=True)
    size = write_json(Path(args.out) / f"{NAME}.json", out)
    print(f"{NAME}.json: {size / 1e3:.1f} kB; L = {L:.1f} ({'unit rows' if unit_rows else 'raw'}), plane holds "
          f"{100 * share['displacement']:.1f}% of the displacement, {100 * share['steps']:.1f}% of the steps; "
          f"loss {path[0]['loss']:.4f} -> {path[-1]['loss']:.4f} (logged {out['logged_final_loss']})")


if __name__ == "__main__":
    main()
