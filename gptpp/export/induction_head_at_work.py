"""Export for fig-09-03 `gpt-induction-head-at-work`: the induction circuit of the two-layer attention-only model on
rows of random tokens given twice, with every head ablated in turn.

    python -m gptpp.export.induction_head_at_work [--two circ-attn2-repeats] [--device cpu]

From the final model, with gptpp/interp/circuits.py:

    heads         the induction head (the second-layer head with the highest prefix-matching score), every
                  second-layer head scoring above 0.5, and the previous-token head: the first-layer head with the
                  largest K-composition score with the induction head
    scores        of every head: prefix matching (25 tokens, 4 times), previous token on stories and on the rows,
                  copying score, positivity of the OV circuit
    composition   the Q-, K- and V-composition scores of the 64 pairs less 1 / sqrt(d), and the K score computed
                  with the permuted product of the 2021 figure, less its own baseline
    sequences     a few rows of 16 random word tokens given twice: the pattern of every head; under the zero and
                  the mean ablation of each first-layer head, the patterns of the second layer; under every
                  ablation (of all 16 heads), the loss at each token and the model's first choice with its
                  probability; each second-layer head's direct effect on the logit of the token that came; and, for
                  the induction head, its score on the stripe split by what the key read (Proposition 9.3.2: the
                  token's own embedding, or what each first-layer head wrote there)
    meanLoss      the loss on the first copy and on the repeated tokens under every ablation, over 256 such rows
                  drawn from the whole pool

An ablation is full: the head's output is replaced by zeros, or by its mean over the 256 rows at each position,
and the layers after it recompute. Writes `induction-head-at-work.json` and `.bin`.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..interp import circuits as C
from .circuit_runs import TWO, Run, held_out, is_word, pool_for, token_names
from .common import DATA, Blob, write_json

NAME = "induction-head-at-work"
SHOWN = {"length": 16, "repeats": 2, "sequences": 4, "seed": 11}
MEAN = {"sequences": 256, "seed": 5}
SCORE = {"length": 25, "repeats": 4, "sequences": 64, "seed": 0}
INDUCTION = 0.5


@torch.no_grad()
def key_terms(model, cache: dict, head: tuple[int, int], length: int) -> torch.Tensor:
    """For a second-layer head and every destination i of the second copy: the score q_i . R k_j / sqrt(d_head)
    split by what the key read at the source j, (T, 2, 1 + H). The parts are the token's own embedding, then what
    each first-layer head wrote at j; they add up to the head's score at (i, j) (Proposition 9.3.2, the terms
    without and with K-composition). Row 0: at the stripe source j = i - length + 1. Row 1: the mean over the
    other sources 1 .. i, what the same part gives a key that should not match."""
    w = C.head_weights(model, *head)
    stream, results = cache["resid_post.0"][0].double(), cache["attn.0.result"][0].double()  # (T, d), (H, T, d)
    parts = torch.cat((cache["embed"][0].double()[None], results))  # (1 + H, T, d)
    T = stream.size(0)
    out = torch.zeros(T, 2, parts.size(0), dtype=torch.float64)
    for i in range(length + 1, T):
        query = w.q @ stream[i]
        at = torch.stack([(parts[:, j] @ w.k.T) @ (C.relative_rotation(model, i - j).T @ query) for j in range(1, i + 1)])
        at = at / math.sqrt(model.cfg.d_head)  # (i sources, 1 + H)
        stripe = i - length  # the index of source i - length + 1 in the list of sources 1 .. i
        out[i, 0] = at[stripe]
        out[i, 1] = (at.sum(dim=0) - at[stripe]) / (at.size(0) - 1)
    return out


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--two", default=TWO)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=str(DATA))
    args = parser.parse_args()
    torch.set_num_threads(min(16, torch.get_num_threads()))
    run = Run(args.two, args.device)
    model, cfg, bos = run.model(), run.cfg, int(run.data.meta["bos_id"])
    device, H = model.embed.weight.device, cfg.n_head
    names, pool = token_names(), pool_for([run])
    L, R = SHOWN["length"], SHOWN["repeats"]
    T = 1 + L * R

    # ---- the heads, from their scores
    scores = C.head_scores(model, held_out(256), pool, bos, SCORE["length"], SCORE["repeats"], SCORE["sequences"], SCORE["seed"])
    prefix = scores["prefix_matching"]
    induction = max(range(H), key=lambda h: prefix[1][h])
    base = C.baseline(cfg.d_model)
    composition = {k: (v - base) for k, v in C.composition_table(model, 0, 1).items()}
    permuted_base = C.composition_baseline(cfg.d_model, cfg.d_head, draws=100, cyclic=True)
    permuted = C.composition_table(model, 0, 1, cyclic=True)["k"] - permuted_base
    previous = int(composition["k"][:, induction].argmax())

    # ---- the rows: shown ones of word tokens, measured ones of the whole pool
    words = torch.tensor([t for t in pool.tolist() if is_word(names[t]) and names[t].startswith(" ") and len(names[t]) >= 4])
    shown = C.repeated_random_tokens(SHOWN["sequences"], L, R, words, bos, SHOWN["seed"]).to(device)
    many = C.repeated_random_tokens(MEAN["sequences"], L, R, pool, bos, MEAN["seed"]).to(device)
    means = C.mean_head_outputs(model, many)
    previous_rows = C.previous_token_score(C.attention_patterns(model, many)[0]).tolist()
    prefix_rows = [C.prefix_matching_score(p, L, R).tolist() for p in C.attention_patterns(model, many)]

    conditions = [{"key": "none"}] + [{"layer": layer, "head": head, "mode": mode}
                                      for layer in range(cfg.n_layer) for head in range(H) for mode in ("zero", "mean")]

    def hooks_of(condition: dict):
        if condition.get("key") == "none":
            return None
        return C.ablation_hooks([(condition["layer"], condition["head"])], condition["mode"], means)

    mean_loss = {"repeated": [], "first": []}
    for condition in conditions:
        halves = C.repeated_loss(C.loss_by_position(model, many, hooks_of(condition)), L, R)
        mean_loss["repeated"].append(halves["repeated"])
        mean_loss["first"].append(halves["first"])

    # ---- the shown rows under every condition
    K = shown.size(0)
    patterns = torch.stack(C.attention_patterns(model, shown), dim=1).reshape(K, 2 * H, T, T)  # (K, 16, T, T)
    ablated = torch.zeros(K, 2 * H, H, T, T)
    loss = torch.zeros(K, len(conditions), T - 1)
    pred = torch.zeros(K, len(conditions), T, dtype=torch.long)
    pred_p = torch.zeros(K, len(conditions), T)
    for c, condition in enumerate(conditions):
        hooks = hooks_of(condition)
        logits, cache = model.run_with_cache(shown, names=["attn.1.pattern"], hooks=hooks)
        p = F.softmax(logits.float(), dim=-1)
        pred_p[:, c], pred[:, c] = (t.cpu() for t in p.max(dim=-1))
        loss[:, c] = C.loss_by_position(model, shown, hooks).cpu()
        if c > 0 and condition["layer"] == 0:
            ablated[:, 2 * condition["head"] + (condition["mode"] == "mean")] = cache["attn.1.pattern"].cpu()
    dla = torch.zeros(K, H, T)
    terms = torch.zeros(K, T, 2, 1 + H)
    unembed = C.unembedding(model)
    for k in range(K):
        _, cache = model.run_with_cache(shown[k:k + 1], names=["embed", "attn.0.result", "attn.1.result", "resid_post.0"])
        through = cache["attn.1.result"][0].double() @ unembed.T  # (H, T, V)
        through = through - through.mean(dim=-1, keepdim=True)
        for i in range(T - 1):
            dla[k, :, i] = through[:, i, shown[k, i + 1]].cpu()
        terms[k] = key_terms(model, cache, (1, induction), L)

    blob = Blob(NAME, args.out)
    u8 = lambda a: np.rint(a.numpy() * 255).astype(np.uint8)  # noqa: E731
    blob.put(f"{NAME}.patterns", u8(patterns.cpu()), "attention patterns of every head on the shown rows: (rows, heads of layer 0 "
             "then layer 1, destination, source)", lo=0.0, hi=1.0)
    blob.put(f"{NAME}.ablated", u8(ablated), "patterns of the second layer when a first-layer head is ablated: (rows, 2 * head + "
             "(0 zero | 1 mean), second-layer head, destination, source)", lo=0.0, hi=1.0)
    blob.put(f"{NAME}.predp", u8(pred_p), "probability of the model's first choice: (rows, conditions, position)", lo=0.0, hi=1.0)
    blob.put(f"{NAME}.loss", loss.numpy().astype(np.float16), "loss in nats on the token after each position: (rows, conditions, position)")
    blob.put(f"{NAME}.dla", dla.numpy().astype(np.float16), "what each second-layer head adds to the logit of the token that came, "
             "from the mean over the vocabulary: (rows, head, position)")
    blob.put(f"{NAME}.keyterms", terms.numpy().astype(np.float16), "the induction head's score split by what the key read: (rows, "
             "destination, 0 at the stripe source | 1 mean over the other sources, the embedding then each first-layer head)")
    size = blob.save()

    used = sorted(set(shown.flatten().tolist()) | set(pred.flatten().tolist()))
    out = {
        "model": run.describe(), "length": L, "repeats": R, "positions": T,
        "heads": {"induction": [1, induction], "previous": [0, previous],
                  "inductionAll": [h for h in range(H) if prefix[1][h] > INDUCTION], "threshold": INDUCTION},
        "scores": {"prefix": prefix, "prefixOnRows": prefix_rows, "previousOnText": scores["previous_token"],
                   "previousOnRows": [previous_rows], "copying": scores["copying_score"], "positivity": scores["copying_statistic"],
                   "protocol": SCORE},
        "composition": {"baseline": base, "k": composition["k"].tolist(), "q": composition["q"].tolist(),
                        "v": composition["v"].tolist(), "kPermuted": permuted.tolist(), "permutedBaseline": permuted_base},
        "conditions": conditions,
        "meanLoss": dict(mean_loss, sequences=MEAN["sequences"], uniform=math.log(len(pool)), pool=len(pool)),
        "sequences": [{"tokens": shown[k].tolist(), "pred": pred[k].tolist()} for k in range(K)],
        "names": {str(t): names[t] for t in used},
    }
    json_size = write_json(Path(args.out) / f"{NAME}.json", out)
    print(f"wrote {NAME}.json ({json_size:,} bytes) and {NAME}.bin ({size:,} bytes); induction head 1.{induction}, previous-token "
          f"head 0.{previous}; repeated loss {mean_loss['repeated'][0]:.3f}, with 0.{previous} zeroed "
          f"{mean_loss['repeated'][1 + 2 * previous]:.3f}, mean-ablated {mean_loss['repeated'][2 + 2 * previous]:.3f}")


if __name__ == "__main__":
    main()
