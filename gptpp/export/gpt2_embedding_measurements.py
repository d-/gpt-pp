"""The released GPT-2 small, measured: its tokenizer on digits and on a few strings, its token frequencies on
WikiText-103, and the geometry of its one embedding matrix (it ties: the same rows are read at the input and
applied at the output). These are the numbers part 2 of the series quotes about GPT-2.

    HF_HUB_OFFLINE=1 python -m gptpp.export.gpt2_embedding_measurements [--out results/interp/gpt2-embeddings.json]

Read from the local Hugging Face cache (nothing is downloaded with HF_HUB_OFFLINE=1): the model `openai-community/gpt2`
and the dataset `Salesforce/wikitext`, configuration `wikitext-103-raw-v1`; the tokenizers come from `tiktoken`
(`gpt2`, and `cl100k_base`, `o200k_base` for the comparison on digits).

    tokenizer   how many of the 10^k strings of k digits are one token (k = 1..4, leading zeros included), for the
                three tokenizers; how a few strings are cut; the id of " SolidGoldMagikarp"
    zipf        GPT-2 tokens of the first training file of WikiText-103, joined and encoded as one text: the count
                of every token, the least-squares exponent of count against rank over three windows of ranks, the
                tokens that never occur, the share of the text in the most frequent 100 and 1,000, the entropy
    matrix      row norms (mean, least, greatest; by tenth of the tokens that occur, from the most to the least
                frequent; of the tokens that never occur; rank correlation with log frequency); the mean cosine of
                200,000 random pairs of rows before and after the mean row is subtracted; the singular values at
                both ends; the participation ratio of the centered matrix; the tokens nearest the centroid and the
                place of " SolidGoldMagikarp" among them
    analogies   the offset rule on unit rows, five questions, the three inputs left out and not

About two minutes on one core, most of it the tokenization of 59 million tokens.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np

from .. import repo_dir
from .common import write_json
from .embedding_tables import QUESTIONS, analogy, digit_strings, mean_cosine, norms_by_frequency, spectrum, unit

MODEL = "openai-community/gpt2"
STRINGS = ["strawberry", " strawberry", " SolidGoldMagikarp", "1234567", "3.14159", "dog", " dog", "dog."]
WINDOWS = [(1, 1000), (10, 1000), (10, 10_000), (1000, 30_000)]


def wikitext_train_first_file() -> tuple[str, str]:
    import pandas as pd

    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    found = sorted(glob.glob(os.path.join(hub, "datasets--Salesforce--wikitext", "snapshots", "*", "wikitext-103-raw-v1", "train-*.parquet")))
    if not found:
        raise SystemExit("WikiText-103 is not in the Hugging Face cache: load Salesforce/wikitext, wikitext-103-raw-v1, once")
    return "".join(pd.read_parquet(found[0])["text"].tolist()), Path(found[0]).name


def zipf_exponent(counts: np.ndarray, lo: int, hi: int) -> float:
    c = np.sort(counts[counts > 0])[::-1].astype(float)
    r = np.arange(1, len(c) + 1)
    m = (r >= lo) & (r <= min(hi, len(c)))
    return float(-np.polyfit(np.log(r[m]), np.log(c[m]), 1)[0])


def measure() -> dict:
    import tiktoken
    import torch
    from transformers import AutoModelForCausalLM

    enc = tiktoken.get_encoding("gpt2")
    others = {"cl100k_base": tiktoken.get_encoding("cl100k_base"), "o200k_base": tiktoken.get_encoding("o200k_base")}
    name = lambda i: enc.decode([int(i)])  # noqa: E731
    out: dict = {"model": MODEL, "tokenizer": "tiktoken gpt2", "vocabulary": enc.n_vocab}

    # ---- the tokenizer
    out["digitStrings"] = {"gpt2": digit_strings(enc.encode)}
    for key, e in others.items():
        out["digitStrings"][key] = digit_strings(e.encode)
    out["strings"] = {s: {"gpt2": [name(i) for i in enc.encode(s)], "ids": enc.encode(s),
                          **{key: len(e.encode(s)) for key, e in others.items()}} for s in STRINGS}

    # ---- frequencies on WikiText-103
    t0 = time.time()
    text, file = wikitext_train_first_file()
    ids = np.asarray(enc.encode(text, disallowed_special=()))
    counts = np.bincount(ids, minlength=enc.n_vocab)
    p = counts[counts > 0] / counts.sum()
    desc = np.sort(counts)[::-1]
    out["zipf"] = {
        "text": f"WikiText-103 (raw), {file}, joined and encoded as one text", "tokens": int(len(ids)), "bytes": len(text.encode("utf-8")),
        "exponents": {f"ranks {lo} to {hi}": zipf_exponent(counts, lo, hi) for lo, hi in WINDOWS},
        "neverSeen": int((counts == 0).sum()), "shareTop100": float(desc[:100].sum() / counts.sum()),
        "shareTop1000": float(desc[:1000].sum() / counts.sum()), "entropyBits": float(-(p * np.log2(p)).sum()),
        "top10": [[name(i), int(counts[i])] for i in np.argsort(-counts, kind="stable")[:10]], "seconds": round(time.time() - t0, 1),
    }

    # ---- the matrix
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True)
    E = model.get_input_embeddings().weight.detach()
    U = model.get_output_embeddings().weight.detach()
    W = E.double().numpy()
    out["matrix"] = {"shape": list(W.shape), "tied": bool(E.data_ptr() == U.data_ptr() or torch.equal(E, U)), "parameters": int(W.size),
                     "shareOfModel": float(W.size / sum(q.numel() for q in model.parameters()))}
    out["norms"] = norms_by_frequency(W, counts)
    out["cosine"] = mean_cosine(W)
    out["spectrum"] = spectrum(W)
    dist = np.linalg.norm(W - W.mean(0), axis=1)
    place = np.argsort(np.argsort(dist, kind="stable"), kind="stable")
    glitch = enc.encode(" SolidGoldMagikarp")
    out["centroid"] = {"nearest": [name(i) for i in np.argsort(dist, kind="stable")[:12]],
                       "distance": {"least": float(dist.min()), "median": float(np.median(dist)), "greatest": float(dist.max())}}
    if len(glitch) == 1:
        out["centroid"][" SolidGoldMagikarp"] = {"id": glitch[0], "placeByDistanceToCentroid": int(place[glitch[0]]) + 1, "of": int(len(dist)),
                                                 "distance": float(dist[glitch[0]]), "occurrences": int(counts[glitch[0]])}

    # ---- analogies
    Wn = unit(W)
    out["analogies"] = {}
    for a, a_star, b in QUESTIONS:
        t = [enc.encode(w) for w in (a, a_star, b)]
        key = f"{a_star.strip()} - {a.strip()} + {b.strip()}"
        if any(len(x) != 1 for x in t):
            out["analogies"][key] = {"note": "an input is not a single token"}
            continue
        top, score = analogy(Wn, t[0][0], t[1][0], t[2][0], k=4, exclude=True)
        raw, _ = analogy(Wn, t[0][0], t[1][0], t[2][0], k=2, exclude=False)
        out["analogies"][key] = {"answer": name(top[0]), "top4": [name(i) for i in top], "cosines": [round(s, 4) for s in score],
                                 "inputsNotExcluded": [name(i) for i in raw]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(repo_dir() / "results" / "interp" / "gpt2-embeddings.json"))
    args = ap.parse_args()
    res = measure()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    size = write_json(args.out, res, compact=False)
    print(f"{args.out}: {size} bytes")
    print(json.dumps(res, ensure_ascii=False)[:3000])


if __name__ == "__main__":
    main()
