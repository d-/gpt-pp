"""The two vocabulary tables of a trained model, measured: the helpers that the exporters of part 2 share, and the
numbers that the post of part 2 quotes about gpt-pp's tables.

    python -m gptpp.export.embedding_tables --ckpt $GPTPP_WORK/ckpt/gptpp-base/model.pt

A model has an input embedding E (one row a token, read by lookup) and an unembedding U (one row a token, applied to
the final state). This script reads both from a checkpoint, with the token counts of the training set from the
dataset's `stats.json`, and writes `results/interp/part-2.json`, keyed by the words of the post's placeholders:

    the number of three-digit strings that are single tokens of the tokenizer
    the rank correlation of row norm with log frequency, for E and for U (tokens that occur at least once)
    the mean cosine of random pairs of rows, raw and after the mean row is subtracted, for E and for U
    the five nearest neighbors by cosine of a few tokens, in each table
    the answers of the offset rule to five analogy questions, in each table, inputs excluded

Everything runs on the CPU in a few seconds. The checkpoint may be `model.pt`, a `ckpt_<step>.pt` or a run folder.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import torch

from .. import repo_dir
from .common import WORK, write_json

RESULTS = repo_dir() / "results" / "interp"
QUESTIONS = [  # a is to a* as b is to what: (a, a*, b), the tokens with their leading spaces
    (" man", " king", " woman"), (" France", " Paris", " Germany"), (" walk", " walked", " run"),
    (" good", " better", " bad"), (" big", " bigger", " small"),
]
NEIGHBORS_OF = [" king", " Paris", " 7"]


# ----------------------------------------------------------------------------------------------------------------
# What the exporters of part 2 share


def load_tables(path: str | Path) -> dict:
    """The embedding and the unembedding of a checkpoint as float64 arrays (the same array twice when tied), with
    what the checkpoint says of itself."""
    path = Path(path)
    if path.is_dir():
        path = path / "model.pt"
    ck = torch.load(path, map_location="cpu", weights_only=True)
    state = {re.sub(r"^(module\.|_orig_mod\.)+", "", k): v for k, v in ck["model"].items()}
    cfg = dict(ck["model_config"])
    E = state["embed.weight"].double().numpy()
    U = E if cfg.get("tie_embeddings") else state["unembed.weight"].double().numpy()
    return {"E": E, "U": U, "config": cfg, "step": int(ck.get("step", -1)), "tokens": int(ck.get("tokens", -1)),
            "run": path.parent.name, "file": path.name, "final": path.name == "model.pt"}


def model_info(t: dict) -> dict:
    cfg = t["config"]
    return {"run": t["run"], "checkpoint": t["file"], "final": t["final"], "step": t["step"], "tokens": t["tokens"],
            "layers": cfg["n_layer"], "width": cfg["d_model"], "vocabulary": cfg["vocab_size"],
            "tied": bool(cfg.get("tie_embeddings")), "embedNorm": bool(cfg.get("embed_norm"))}


def corpus_counts(dataset: str = "fineweb-edu/bpe32k") -> tuple[np.ndarray, dict]:
    """Occurrences of every token id in the training set, and the dataset's statistics."""
    stats = json.loads((repo_dir() / "results" / "data" / dataset / "stats.json").read_text())
    rf = stats["rank_frequency"]
    counts = np.zeros(stats["vocab_size"], dtype=np.int64)
    counts[np.asarray(rf["ids"])] = np.asarray(rf["counts"], dtype=np.int64)
    return counts, stats


def by_rank(counts: np.ndarray) -> np.ndarray:
    """Token ids from the most frequent to the least; equal counts by id (the order of `stats.json`)."""
    return np.lexsort((np.arange(len(counts)), -counts))


def unit(W: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(W, axis=1, keepdims=True)
    return W / np.where(n > 0, n, 1.0)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).statistic)


def mean_cosine(W: np.ndarray, pairs: int = 200_000, seed: int = 0) -> dict:
    """The mean cosine of random pairs of distinct rows, before and after the mean row is subtracted."""
    rng = np.random.default_rng(seed)
    i, j = rng.integers(0, len(W), pairs), rng.integers(0, len(W), pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    mean = W.mean(0)
    a, c = unit(W), unit(W - mean)
    return {"raw": float((a[i] * a[j]).sum(1).mean()), "centered": float((c[i] * c[j]).sum(1).mean()),
            "normOfMeanRow": float(np.linalg.norm(mean)), "meanRowNorm": float(np.linalg.norm(W, axis=1).mean()),
            "pairs": int(keep.sum()), "seed": seed}


def spectrum(W: np.ndarray) -> dict:
    """Singular values of the table and of the centered table; the participation ratio of the latter."""
    s = np.linalg.svd(W, compute_uv=False)
    sc = np.linalg.svd(W - W.mean(0), compute_uv=False)
    p = sc ** 2 / np.sum(sc ** 2)
    return {"top": [float(v) for v in s[:3]], "bottom": [float(v) for v in s[-2:]],
            "participationRatioCentered": float(1.0 / np.sum(p ** 2)), "dimensions": int(W.shape[1]),
            "varianceShareTop1Centered": float(p[0]), "varianceShareTop3Centered": float(p[:3].sum())}


def norms_by_frequency(W: np.ndarray, counts: np.ndarray) -> dict:
    """Row norm against frequency over the tokens that occur: the rank correlation with log count, and the mean
    norm in each tenth of those tokens from the most to the least frequent."""
    norm = np.linalg.norm(W, axis=1)
    seen = counts > 0
    order = np.argsort(-counts[seen], kind="stable")
    return {"spearman": spearman(norm[seen], np.log(counts[seen])),
            "meanByDecile": [float(part.mean()) for part in np.array_split(norm[seen][order], 10)],
            "meanNeverSeen": float(norm[~seen].mean()) if (~seen).any() else None, "neverSeen": int((~seen).sum()),
            "mean": float(norm.mean()), "min": float(norm.min()), "max": float(norm.max())}


def neighbors(Wn: np.ndarray, rows: Sequence[int], k: int, chunk: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """For each token of `rows`, its k nearest other tokens by cosine among all rows of Wn (unit rows), nearest
    first -> (ids, cosines), both len(rows) x k."""
    rows = np.asarray(rows)
    ids, cos = np.empty((len(rows), k), dtype=np.int64), np.empty((len(rows), k))
    for at in range(0, len(rows), chunk):
        r = rows[at:at + chunk]
        c = Wn[r] @ Wn.T
        c[np.arange(len(r)), r] = -np.inf
        part = np.argpartition(-c, k, axis=1)[:, :k]
        val = np.take_along_axis(c, part, 1)
        order = np.argsort(-val, axis=1, kind="stable")
        ids[at:at + chunk], cos[at:at + chunk] = np.take_along_axis(part, order, 1), np.take_along_axis(val, order, 1)
    return ids, cos


def analogy(Wn: np.ndarray, a: int, a_star: int, b: int, k: int = 4, exclude: bool = True) -> tuple[list[int], list[float]]:
    """The offset rule on unit rows: the k tokens nearest by cosine to a* - a + b, the three inputs left out or not."""
    q = Wn[a_star] - Wn[a] + Wn[b]
    score = Wn @ (q / np.linalg.norm(q))
    if exclude:
        score[[a, a_star, b]] = -np.inf
    top = np.argsort(-score, kind="stable")[:k]
    return [int(i) for i in top], [float(score[i]) for i in top]


# ----------------------------------------------------------------------------------------------------------------
# The numbers of part 2


def single_token(tok, text: str) -> Optional[int]:
    ids = tok.encode(text)
    return ids[0] if len(ids) == 1 else None


def digit_strings(encode, digits: Iterable[int] = (1, 2, 3, 4)) -> dict:
    """How many of the 10^k strings of k digits (leading zeros included) are one token."""
    return {str(k): {"single": sum(1 for n in range(10 ** k) if len(encode(str(n).zfill(k))) == 1), "of": 10 ** k}
            for k in digits}


def part_2(ckpt: str | Path, tokenizer: str = "bpe32k", dataset: str = "fineweb-edu/bpe32k") -> dict:
    from ..tokenizer import Tokenizer

    tok = Tokenizer.load(tokenizer)
    t = load_tables(ckpt)
    counts, _ = corpus_counts(dataset)
    name = lambda i: tok.decode([int(i)])  # noqa: E731
    tables = {"input embedding": t["E"], "unembedding": t["U"]}
    units = {k: unit(W) for k, W in tables.items()}
    out: dict = {"_model": model_info(t), "_tokenizer": tokenizer, "_dataset": dataset}

    digits = digit_strings(tok.encode)
    out["number of the 1,000 three-digit strings that are single tokens of gpt-pp's tokenizer"] = digits["3"]["single"]
    out["_digit strings that are single tokens, by length"] = digits

    corr = {k: norms_by_frequency(W, counts) for k, W in tables.items()}
    out["rank correlation between row norm and log frequency over the rows of gpt-pp's trained input embedding"] = corr["input embedding"]["spearman"]
    out["the same for its unembedding (rank correlation between row norm and log frequency)"] = corr["unembedding"]["spearman"]
    out["_row norms by frequency"] = corr

    cosines = {k: mean_cosine(W) for k, W in tables.items()}
    out["mean cosine between random pairs of rows of gpt-pp's trained input embedding, raw and after subtracting the mean row"] = {
        "raw": cosines["input embedding"]["raw"], "centered": cosines["input embedding"]["centered"]}
    out["the same for its unembedding (mean cosine between random pairs of rows, raw and after subtracting the mean row)"] = {
        "raw": cosines["unembedding"]["raw"], "centered": cosines["unembedding"]["centered"]}
    out["_mean cosines, with the norm of the mean row and the mean row norm"] = cosines
    out["_spectrum"] = {k: spectrum(W) for k, W in tables.items()}

    near: dict = {}
    for word in NEIGHBORS_OF:
        ids, asked = tok.encode(word), word
        note = None
        if len(ids) != 1:  # the pattern gives a number no leading space: " 7" is a space and "7"
            asked = word.strip()
            note = f"{word!r} is {len(ids)} tokens of this tokenizer ({'|'.join(name(i) for i in ids)}); the neighbors are those of {asked!r}"
            ids = tok.encode(asked)
        entry: dict = {"token": asked, "id": ids[0] if len(ids) == 1 else None}
        if note:
            entry["note"] = note
        if len(ids) == 1:
            for k, Wn in units.items():
                nid, ncos = neighbors(Wn, [ids[0]], 5)
                entry[k] = [{"token": name(i), "cosine": round(float(c), 4)} for i, c in zip(nid[0], ncos[0])]
        near[word] = entry
    out['the five nearest neighbors by cosine of " king", " Paris" and " 7" in each of the two tables'] = near

    answers: dict = {}
    for a, a_star, b in QUESTIONS:
        ids = [single_token(tok, w) for w in (a, a_star, b)]
        key = f"{a_star.strip()} - {a.strip()} + {b.strip()}"
        if any(i is None for i in ids):
            answers[key] = {"note": "an input is not a single token of this tokenizer"}
            continue
        entry = {}
        for k, Wn in units.items():
            top, score = analogy(Wn, *ids, k=4, exclude=True)
            raw, _ = analogy(Wn, *ids, k=2, exclude=False)
            entry[k] = {"answer": name(top[0]), "top4": [name(i) for i in top], "cosines": [round(s, 4) for s in score],
                        "inputsNotExcluded": [name(i) for i in raw]}
        answers[key] = entry
    out["answers of rule (2.9), inputs excluded, to the five analogy questions of Example 2.6.3 in each of gpt-pp's two tables"] = answers
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", default=str(WORK / "ckpt" / "gptpp-base"), help="model.pt, a ckpt_<step>.pt or a run folder")
    ap.add_argument("--tokenizer", default="bpe32k")
    ap.add_argument("--dataset", default="fineweb-edu/bpe32k")
    ap.add_argument("--out", default=str(RESULTS / "part-2.json"))
    args = ap.parse_args()
    res = part_2(args.ckpt, args.tokenizer, args.dataset)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    size = write_json(args.out, res, compact=False)
    m = res["_model"]
    print(f"{args.out}: {size} bytes, from {m['run']}/{m['checkpoint']} (step {m['step']}, {m['tokens']:,} tokens, final: {m['final']})")
    for k, v in res.items():
        if not k.startswith("_"):
            print(f"  {k}: {json.dumps(v, ensure_ascii=False)[:260]}")


if __name__ == "__main__":
    main()
