"""Export for fig-08-06 `gpt-addition-as-rotation`: the frequencies and amplitudes of the trained modular-addition
model, read off its logits.

    python -m gptpp.export.addition_as_rotation [--run $GPTPP_WORK/ckpt/grokking] [--out <dir>]

A few seconds on a processor; no card. The run is the one fig-08-04 replays (`python -m gptpp.interp.grokking`).

The trained model's logits L(a, b, c) over all p * p pairs and the p candidate answers are projected on the waves
of Proposition 8.3.2 of the series: for each frequency k = 1 .. (p - 1) / 2,

    alpha_k = (2 / p^3) * sum over a, b, c of L(a, b, c) cos(w_k (a + b - c)),      w_k = 2 pi k / p,

the least-squares coefficient of cos(w_k (a + b - c)) in the logits (these waves are orthogonal over the p^3
points, each with squared norm p^3 / 2), and beta_k the same with the sine, which the algorithm predicts to be
zero. Reported with them: the share of the variance of the logits (the mean over c removed for each pair, since a
softmax ignores it) that the cosine waves of the key frequencies hold, and that all (p - 1) / 2 hold; the accuracy
of the model on all pairs; and the accuracy and the margins of the formula itself with the trained amplitudes.

Writes `addition-as-rotation.json` (under two kilobytes).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from .. import work_dir
from ..interp.grokking import all_pairs, answer_logits, build_model
from .common import DATA, write_json

NAME = "addition-as-rotation"


def export(run: Path, out: Path) -> dict:
    config = json.loads((run / "config.json").read_text())
    saved = torch.load(run / "states.pt", map_location="cpu", weights_only=True)
    p = int(config["p"])
    model = build_model(p, config["seed"], init=config["init"])
    model.load_state_dict(saved["states"][-1])
    model.eval()
    tokens, answers = all_pairs(p)
    with torch.no_grad():
        logits = answer_logits(model, tokens, p)                      # (p * p, p), float64
    accuracy = float((logits.argmax(dim=1) == answers).double().mean())
    a, b = tokens[:, 0].double(), tokens[:, 1].double()
    c = torch.arange(p, dtype=torch.float64)
    phase = (a + b)[:, None] - c[None, :]                             # a + b - c, (p * p, p)
    centered = logits - logits.mean(dim=1, keepdim=True)
    total = float(centered.pow(2).sum())
    half = (p - 1) // 2
    alpha, beta = [], []
    for k in range(1, half + 1):
        w = 2 * math.pi * k / p
        alpha.append(float((logits * torch.cos(w * phase)).sum() * 2 / p ** 3))
        beta.append(float((logits * torch.sin(w * phase)).sum() * 2 / p ** 3))
    key = [int(k) for k in config["key_frequencies"]]
    key = sorted(key, key=lambda k: -alpha[k - 1])                    # the strongest wave first

    def share(ks: list[int]) -> float:
        return float(sum(alpha[k - 1] ** 2 for k in ks) * p ** 3 / 2 / total)

    formula = sum(alpha[k - 1] * torch.cos(2 * math.pi * k / p * phase) for k in key)
    formula_accuracy = float((formula.argmax(dim=1) == answers).double().mean())
    top = sum(alpha[k - 1] for k in key)
    wrong = formula.clone()
    wrong[torch.arange(p * p), answers] = -float("inf")
    data = {
        "name": NAME, "p": p, "key": key, "alpha": [alpha[k - 1] for k in key], "beta": [beta[k - 1] for k in key],
        "alpha_all": alpha, "variance_explained_key": share(key), "variance_explained_all": share(list(range(1, half + 1))),
        "model_accuracy": accuracy, "formula_accuracy": formula_accuracy, "formula_top": top, "formula_largest_wrong": float(wrong.max()),
        "model": {"layers": 1, "width": int(config["model_config"]["d_model"]), "heads": int(config["model_config"]["n_head"]),
                  "steps": int(config["steps"]), "fraction": float(config["fraction"]), "weight_decay": float(config["weight_decay"]),
                  "seed": int(config["seed"]), "embedding_key_frequencies": [int(k) for k in config["key_frequencies"]]},
        "note": "alpha_k: the coefficient of cos(2 pi k (a + b - c) / p) in the trained model's logits over all pairs and candidates; "
                "key: the frequencies of its embedding, strongest wave first",
    }
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / f"{NAME}.json", data)
    return {k: data[k] for k in ("p", "key", "alpha", "beta", "variance_explained_key", "variance_explained_all", "model_accuracy",
                                 "formula_accuracy", "formula_top", "formula_largest_wrong")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default=str(work_dir() / "ckpt" / "grokking"))
    ap.add_argument("--out", default=str(DATA))
    args = ap.parse_args()
    print(json.dumps(export(Path(args.run), Path(args.out)), indent=1))


if __name__ == "__main__":
    main()
