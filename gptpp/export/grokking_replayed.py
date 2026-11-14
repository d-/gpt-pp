"""Export for fig-08-04 `gpt-grokking-replayed`: the modular-addition run, step by step.

    python -m gptpp.interp.grokking --out $GPTPP_WORK/ckpt/grokking --budget 520      # the run (repeat until it ends)
    python -m gptpp.export.grokking_replayed --run $GPTPP_WORK/ckpt/grokking           # this file: a few seconds, no card

From the run's `history.json` and `states.pt`, at each of its log-spaced steps:

    the curves     train and test loss and accuracy, the norm of the weights, restricted loss (held-out pairs) and
                   excluded loss (training pairs) as Nanda et al. define them (the key-frequency terms of the
                   logits' Fourier transform over a and b kept alone, or removed), the share of the embedding's
                   power held by the key frequencies
    the spectrum   the power of the embedding at each of the (p - 1) / 2 frequencies
    the points     where each of the p numbers stands in a few fixed directions of the residual stream: for every
                   key frequency k, the plane spanned by the cosine and sine components of the TRAINED embedding at
                   k (two orthonormal directions); and one direction the trained model has given up, the largest
                   component of the frequency that was strongest among those it dropped, taken at the step of its
                   peak and made orthogonal to the planes. The directions never change: only the embedding moves.
                   Early on the numbers project to a cloud in each plane; in the trained model the plane of
                   frequency k holds a circle with the number a at angle 2 pi k a / p, and the dropped direction
                   holds nothing.
    the circle     for each plane and step, the share of the plane's variance that is at the plane's own frequency
                   (1 when the numbers stand exactly on an ellipse traversed at that frequency)

Writes `grokking-replayed.json` (the curves and the run's settings) and `grokking-replayed.bin` with the points as
float16 (steps, p, 2 * key frequencies + 1) and the spectrum as bytes (steps, 1 + (p - 1) / 2), and their entries
for the manifest.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from ..interp.grokking import first_step_at, fourier_basis
from .common import DATA, Blob, write_json

NAME = "grokking-replayed"


def planes_of(embedding: Tensor, basis: Tensor, frequencies: list[int]) -> Tensor:
    """For each frequency, two orthonormal directions of the residual stream that span the cosine and the sine
    component of the embedding (p, d) at that frequency: (len(frequencies), 2, d)."""
    coefficient = basis @ embedding
    out = []
    for k in frequencies:
        c, s = coefficient[2 * k - 1], coefficient[2 * k]
        u = c / c.norm()
        v = s - (s @ u) * u
        out.append(torch.stack((u, v / v.norm())))
    return torch.stack(out)


def circle_share(points: Tensor, basis: Tensor, k: int) -> Tensor:
    """The share of the variance of points (..., p, 2) that is at frequency k: the squared length of their
    components along the cosine and the sine of k, over that along all the basis functions but the constant."""
    square = (basis @ points).pow(2).sum(dim=-1)          # (..., p): one entry a basis function
    return (square[..., 2 * k - 1] + square[..., 2 * k]) / square[..., 1:].sum(dim=-1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="the directory written by gptpp.interp.grokking")
    ap.add_argument("--out", default=str(DATA))
    args = ap.parse_args()
    run, out = Path(args.run), Path(args.out)
    config = json.loads((run / "config.json").read_text())
    history = json.loads((run / "history.json").read_text())
    saved = torch.load(run / "states.pt", map_location="cpu", weights_only=True)
    assert saved["steps"] == history["step"]
    p, key = config["p"], config["key_frequencies"]
    basis = fourier_basis(p)
    embedding = torch.stack([state["embed.weight"][:p].double() for state in saved["states"]])   # (steps, p, d)
    power = torch.tensor(history["embedding_power"], dtype=torch.float64)                       # (steps, 1 + (p - 1) / 2)

    planes = planes_of(embedding[-1], basis, key)                                                # (K, 2, d)
    others = [k for k in range(1, p // 2 + 1) if k not in key]
    dropped = others[int(power[:, others].max(dim=0).values.argmax())]
    peak = int(power[:, dropped].argmax())
    component = basis @ embedding[peak]
    z = max((component[2 * dropped - 1], component[2 * dropped]), key=lambda v: float(v.norm()))
    flat = planes.reshape(-1, planes.size(-1))                                                   # (2K, d), nearly orthonormal
    z = z - torch.linalg.lstsq(flat.T, z.unsqueeze(1)).solution.squeeze(1) @ flat                # orthogonal to every plane
    directions = torch.cat((flat, (z / z.norm()).unsqueeze(0)))                                  # (2K + 1, d)
    points = embedding @ directions.T                                                            # (steps, p, 2K + 1)

    in_plane = points[..., : 2 * len(key)].reshape(points.size(0), p, len(key), 2).permute(0, 2, 1, 3)   # (steps, K, p, 2)
    share = torch.stack([circle_share(in_plane[:, i], basis, k) for i, k in enumerate(key)], dim=1)       # (steps, K)
    radius = in_plane[-1].norm(dim=-1).mean(dim=-1)                                              # (K,) of the trained model
    key_share = power[:, key].sum(dim=1) / power.sum(dim=1)

    blob = Blob(NAME, out)
    blob.put(f"{NAME}.points", points.numpy().astype(np.float16),
             f"(steps, {p} numbers, {2 * len(key)} plane coordinates then 1 dropped direction): the embedding in fixed directions")
    top = float(power[:, 1:].max())
    blob.put_u8(f"{NAME}.spectrum", power.numpy(), "(steps, constant then frequencies 1 .. (p - 1) / 2): power of the embedding",
                lo=0.0, hi=top)
    size = blob.save()
    model = config["model_config"]
    meta = {
        "p": p, "fraction": config["fraction"], "n_train": config["n_train"], "n_test": config["n_test"],
        "steps": config["steps"], "lr": config["lr"], "weight_decay": config["weight_decay"], "betas": config["betas"],
        "seed": config["seed"], "data_seed": config["data_seed"], "init": config["init"], "seconds": config["seconds"],
        "model": {"layers": model["n_layer"], "width": model["d_model"], "heads": model["n_head"],
                  "mlp": int(model["mlp_mult"] * model["d_model"]), "act": model["act"], "params": config["params"]["total"]},
        "key": key, "dropped": dropped, "dropped_peak_step": history["step"][peak],
        "train_acc_99_step": config["train_acc_99_step"], "test_acc_99_step": config["test_acc_99_step"],
        "train_acc_100_step": first_step_at(history, "train_acc", 1.0),
        "measures": "restricted: the loss on the held-out pairs of the logits' constant term and their 4 terms a key "
                    "frequency in the Fourier transform over (a, b) (Nanda et al., Section 5.1: 20 terms for 5 "
                    "frequencies); excluded: the loss on the training pairs of the logits without those terms",
        "step": history["step"],
        "train_loss": history["train_loss"], "test_loss": history["test_loss"],
        "train_acc": history["train_acc"], "test_acc": history["test_acc"], "weight_norm": history["weight_norm"],
        "restricted": history["restricted_test"], "excluded": history["excluded_train"],
        "key_share": key_share.tolist(), "power_total": power.sum(dim=1).tolist(),   # the squared norm of the embedding
        "circle": share.T.tolist(), "radius": radius.tolist(),
        "dropped_spread": points[..., -1].std(dim=1).tolist(), "spectrum_top": top,
        "keys": {"points": f"{NAME}.points", "spectrum": f"{NAME}.spectrum"},
    }
    json_size = write_json(out / f"{NAME}.json", meta)
    last = points.size(0) - 1
    print(f"key frequencies {key}, dropped {dropped} (peak at step {history['step'][peak]}); "
          f"radius of the circles {[round(float(r), 3) for r in radius]}; share on the circle at the end "
          f"{[round(float(s), 4) for s in share[last]]}, at step 0 {[round(float(s), 4) for s in share[0]]}; "
          f"spread along the dropped direction {float(points[0, :, -1].std()):.3f} -> {float(points[last, :, -1].std()):.4f}")
    print(f"wrote {NAME}.bin ({size:,} bytes) and {NAME}.json ({json_size:,} bytes): {points.size(0)} steps")


if __name__ == "__main__":
    main()
