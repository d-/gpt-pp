"""One Muon step on a real gradient: the gradient of two hidden matrices of a small model on one batch of held-out
text, early in training and at its end, with the update Adam makes of the same gradient. Writes
`assets/data/gpt/muon-gradient-matrices.bin`, `.entries.json` and `.json` for the figure `gpt-muon-orthogonalizes`.

    python -m gptpp.export.muon_gradient_matrices --ckpt <run directory> [--early 115] [--layer 2]
        [--batches 16] [--device cpu] [--out <dir>]

`--ckpt` is the directory of a run (`model.pt`, `ckpt_<step>.pt`, `config.json`): the early checkpoint is
`ckpt_<--early>.pt`, the final one `model.pt`. For each of the two checkpoints and each of the two matrices (the
input matrix of the MLP of block `--layer`, `blocks.<layer>.mlp.fc.weight`, and the query matrix of its attention,
`blocks.<layer>.attn.q.weight`) the exporter backpropagates the mean cross-entropy of `--batches` batches of the
run's own shape (256 windows of 256 tokens for pico), taken in order from the held-out tokens, in float32 on the
CPU, and keeps:

    gradient   G, the gradient on the first batch, divided by the root mean square of its entries (`unit`) so that
               float16 holds it at full relative precision; the figure normalizes it anyway
    adam       Adam's update of the same gradient with its momentum left out: G / (sqrt(v) + eps), entry by entry,
               where v is the mean of the squared gradients of all `--batches` batches at the same weights (the
               second moment at its stationary value) and eps is the run's

Both are float16, rows by columns as PyTorch stores the weight (fan-out by fan-in). The page computes everything
else: the singular values (a Jacobi decomposition), the Newton-Schulz steps, the stable ranks. So that its test
can compare, the JSON also holds what the lab computes from the same float16 values:

    generated_by   "gptpp.export.muon_gradient_matrices"
    model          {run, n_layer, d_model, n_head, vocab_size, block_size, optimizer, momentum, ns_steps}
    batch          {windows, tokens (per window), total, source}
    adam           {batches, eps, rule}
    newton_schulz  {coefficients: [a, b, c], steps}: gptpp.optim.NS_TUNED and the run's ns_steps
    checkpoints    [{key: "early" | "final", file, step, tokens, loss (mean cross-entropy of the first batch)}]
    matrices       [{key: "mlp" | "attention", parameter, block, rows, cols}]
    cases          {"<checkpoint>.<matrix>": {
                      gradient, adam     the manifest keys of the two arrays
                      unit               root mean square of the gradient's entries (stored = G / unit)
                      frobenius          the Frobenius norm of G
                      sigma              singular values of G / ||G||_F, largest first (numpy, float64)
                      after              what `gptpp.optim.orthogonalize` leaves of each of them after ns_steps steps:
                                         u_i^T O v_i for the singular vectors of G, in the order of `sigma`
                      adam_sigma         singular values of the Adam update / its Frobenius norm, largest first
                      stable_rank        {gradient, update, adam, sign, adam_momentum}: sum of squared singular
                                         values over the largest squared; `sign` is of sign(G), Adam's first step
                                         from empty moments; `adam_momentum` is of Adam with the run's betas after
                                         the same batches, the first batch last
                      sigma_shift        largest relative change of a singular value above 0.001 caused by float16
                    }}

Runs in about a minute for pico on the CPU. Run it again when the model is retrained; the module and its test read
shapes, steps and numbers from these files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..data import TokenData, ValLoader
from ..model import GPT
from ..optim import NS_TUNED, orthogonalize
from .common import DATA, Blob, write_json

NAME = "muon-gradient-matrices"
MATRICES = (("mlp", "blocks.{layer}.mlp.fc.weight"), ("attention", "blocks.{layer}.attn.q.weight"))


def stable_rank(sigma: np.ndarray) -> float:
    """Sum of squared singular values over the largest squared: how many directions a matrix spreads its size over."""
    s2 = np.asarray(sigma, dtype=np.float64) ** 2
    return float(s2.sum() / s2.max())


def batch_gradients(model: GPT, batches: list, names: list[str]) -> tuple[dict[str, np.ndarray], list[float]]:
    """The gradient of each named parameter on each batch at fixed weights, (batches, rows, cols), and the losses."""
    params = dict(model.named_parameters())
    out: dict[str, list[np.ndarray]] = {name: [] for name in names}
    losses = []
    for x, y, _ in batches:
        model.zero_grad(set_to_none=True)
        loss = model(x, y)
        loss.backward()
        losses.append(loss.item())
        for name in names:
            out[name].append(params[name].grad.detach().double().cpu().numpy().copy())
    return {name: np.stack(g) for name, g in out.items()}, losses


def adam_with_momentum(grads: np.ndarray, beta1: float, beta2: float, eps: float) -> np.ndarray:
    """Adam's bias-corrected update after seeing the batches from the last to the first, so that the first batch's
    gradient is the newest."""
    m, v = np.zeros_like(grads[0]), np.zeros_like(grads[0])
    for g in grads[::-1]:
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g * g
    n = len(grads)
    return (m / (1 - beta1 ** n)) / (np.sqrt(v / (1 - beta2 ** n)) + eps)


def case(grads: np.ndarray, eps: float, ns_steps: int, betas: tuple[float, float]) -> tuple[np.ndarray, np.ndarray, dict]:
    """The two float16 arrays of one matrix at one checkpoint, and the numbers the lab computes from them."""
    g = grads[0]
    unit = float(np.sqrt((g ** 2).mean()))
    stored = (g / unit).astype(np.float16)
    adam = (g / (np.sqrt((grads ** 2).mean(axis=0)) + eps)).astype(np.float16)

    x = stored.astype(np.float64)
    x /= np.linalg.norm(x)
    u, sigma, vt = np.linalg.svd(x, full_matrices=False)
    exact = np.linalg.svd(g / np.linalg.norm(g), compute_uv=False)
    update = orthogonalize(torch.from_numpy(stored.astype(np.float32)), ns_steps).double().numpy()
    after = np.einsum("ik,ij,kj->k", u, update, vt)                      # u_k^T O v_k
    a = adam.astype(np.float64)
    adam_sigma = np.linalg.svd(a / np.linalg.norm(a), compute_uv=False)
    big = sigma > 1e-3
    numbers = {
        "unit": unit, "frobenius": float(np.linalg.norm(g)),
        "sigma": sigma.tolist(), "after": after.tolist(), "adam_sigma": adam_sigma.tolist(),
        "stable_rank": {
            "gradient": stable_rank(sigma), "update": stable_rank(after), "adam": stable_rank(adam_sigma),
            "sign": stable_rank(np.linalg.svd(np.sign(g), compute_uv=False)),
            "adam_momentum": stable_rank(np.linalg.svd(adam_with_momentum(grads, *betas, eps), compute_uv=False)),
        },
        "sigma_shift": float(np.abs(sigma[big] / exact[big] - 1).max()),
    }
    return stored, adam, numbers


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True, help="the run directory: model.pt, ckpt_<step>.pt, config.json")
    ap.add_argument("--early", type=int, default=115, help="the step of the early checkpoint")
    ap.add_argument("--layer", type=int, default=2, help="the block whose matrices are taken (0 is next to the embedding)")
    ap.add_argument("--batches", type=int, default=16, help="batches behind Adam's second moment")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(DATA))
    args = ap.parse_args()

    run = Path(args.ckpt)
    config = json.loads((run / "config.json").read_text())
    train = config["train_config"]
    names = [pattern.format(layer=args.layer) for _, pattern in MATRICES]
    data = TokenData(train["data"])
    blob = Blob(NAME, args.out)
    checkpoints, cases, shapes, loader = [], {}, {}, None
    for key, file in (("early", f"ckpt_{args.early}.pt"), ("final", "model.pt")):
        model, cfg = GPT.from_checkpoint(run / file, device=args.device)
        if loader is None:                      # the same held-out batches for both checkpoints
            loader = list(ValLoader(data, cfg.block_size, train["device_batch"], device=args.device,
                                    val_tokens=args.batches * train["device_batch"] * cfg.block_size))[:args.batches]
        grads, losses = batch_gradients(model, loader, names)
        saved = torch.load(run / file, map_location="cpu", weights_only=True)
        checkpoints.append({"key": key, "file": file, "step": int(saved["step"]), "tokens": int(saved["tokens"]),
                            "loss": losses[0]})
        for (matrix, _), name in zip(MATRICES, names):
            stored, adam, numbers = case(grads[name], train["eps"], train["ns_steps"], (train["beta1"], train["beta2"]))
            shapes[matrix] = stored.shape
            entry = {"gradient": f"{NAME}.{key}.{matrix}.gradient", "adam": f"{NAME}.{key}.{matrix}.adam"}
            where = f"{name} of {config['run']} at step {saved['step']}"
            blob.put(entry["gradient"], stored, f"gradient of {where} on one batch, divided by its root mean square entry")
            blob.put(entry["adam"], adam, f"Adam's update of that gradient, momentum left out: G / (sqrt(v) + eps), {where}")
            cases[f"{key}.{matrix}"] = {**entry, **numbers}
            rank = numbers["stable_rank"]
            print(f"{key:5s} {name:26s} loss {losses[0]:.3f}  stable rank: gradient {rank['gradient']:.2f}, "
                  f"update {rank['update']:.1f}, adam {rank['adam']:.2f}  (float16 shift {numbers['sigma_shift']:.1e})")

    mc = config["model_config"]
    bytes_json = write_json(Path(args.out) / f"{NAME}.json", {
        "generated_by": "gptpp.export.muon_gradient_matrices",
        "model": {"run": config["run"], "n_layer": mc["n_layer"], "d_model": mc["d_model"], "n_head": mc["n_head"],
                  "vocab_size": mc["vocab_size"], "block_size": mc["block_size"], "optimizer": train["optimizer"],
                  "momentum": train["momentum"], "ns_steps": train["ns_steps"]},
        "batch": {"windows": train["device_batch"], "tokens": mc["block_size"],
                  "total": train["device_batch"] * mc["block_size"],
                  "source": f"the first windows of the held-out tokens of {train['data']}"},
        "adam": {"batches": args.batches, "eps": train["eps"],
                 "rule": "G / (sqrt(v) + eps) entry by entry, v the mean squared gradient over the batches"},
        "newton_schulz": {"coefficients": list(NS_TUNED), "steps": train["ns_steps"]},
        "checkpoints": checkpoints,
        "matrices": [{"key": matrix, "parameter": name, "block": args.layer, "rows": int(shapes[matrix][0]),
                      "cols": int(shapes[matrix][1])} for (matrix, _), name in zip(MATRICES, names)],
        "cases": cases,
    })
    print(f"wrote {NAME}.bin ({blob.save()} bytes), {NAME}.json ({bytes_json} bytes) in {args.out}")


if __name__ == "__main__":
    main()
