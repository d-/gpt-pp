"""Export for fig-01-03 `gpt-prediction-is-compression`: what an arithmetic coder needs to code three sentences under
three predictors, and what the same predictors and two general-purpose compressors make of the whole held-out text.

    python -m gptpp.export.prediction_is_compression [--out <dir>]

An arithmetic coder narrows an interval, token by token, to the slice that a predictor reserves for the token that
comes. With the vocabulary in the order of its ids, the slice of token t is [C(t - 1), C(t)) / C(V - 1), where C is
the running sum of the predictor's probabilities, taken in double precision in the order of the ids. For each
sentence the page offers, each predictor and each token, this file holds the two ends of the token's slice, exactly
(the page's coder computes with them as exact fractions), and the sixteen widest slices of the step, for drawing.

    unigram   the table's frequencies of tokens (`token_pair_table.py`)
    bigram    the table of token pairs the page ships: it sees the token before (`<|bos|>` before the first)
    pico      the network the page runs (the shipped weights, probabilities in single precision as in the page):
              it sees the sentence so far, from `<|bos|>`

The page computes the fourth predictor, the uniform one, by itself, and all four live for a sentence the reader
types. The corpus figures (bits per byte on the held-out stories) are read from the lab's statistics, from the
table's export and from the measurement that chose the form of pico's weights.

Writes `prediction-is-compression.json`.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from .. import repo_dir, work_dir
from ..tokenizer import Tokenizer
from .browser_model import page_model
from .common import DATA, write_atomic
from .counted_models import PairTable

NAME = "prediction-is-compression"
WIDEST, MAX_TOKENS = 16, 22
SENTENCES = [
    ("lucy", "a story", "Once upon a time, there was a little girl named Lucy. She had a black cat named Shadow."),
    ("garden", "another", "Tom and Mia are playing in the garden. They have a big pot and some seeds."),
    ("shannon", "not a story", "There is no reverse on a motorcycle. A friend of mine found this out."),
]


def step(p: np.ndarray, token: int, name) -> dict:
    """The slice of `token` under the distribution p, and the widest slices of the step."""
    running = np.cumsum(p.astype(np.float64))
    total = running[-1]
    ends = np.concatenate(([0.0], running / total))                 # ends[v]: where the slice of token v begins
    widest = np.sort(np.argsort(-p, kind="stable")[:WIDEST])
    return {"c0": float(ends[token]), "c1": float(ends[token + 1]),
            "slices": [[name(int(v)), float(f"{ends[v]:.6g}"), float(f"{ends[v + 1] - ends[v]:.6g}")] for v in widest]}


def export(out: Path) -> dict:
    tok = Tokenizer.load("bpe4k")
    model, cfg = page_model("pico", out)
    table = PairTable.load(out=out)
    name = lambda i: "(the end)" if i == tok.bos_id else tok.decode([i])
    sentences = []
    for key, label, text in SENTENCES:
        ids = tok.encode(text)
        if len(ids) > MAX_TOKENS:
            raise SystemExit(f"{key}: {len(ids)} tokens, the figure has room for {MAX_TOKENS}")
        seq = [tok.bos_id] + ids
        with torch.no_grad():
            pico = torch.softmax(model(torch.tensor([seq]))[0, :-1].float(), dim=-1).numpy()
        item = {"key": key, "label": label, "text": text, "bytes": len(text.encode("utf-8")), "tokens": ids, "texts": [tok.decode([i]) for i in ids]}
        for predictor, dist in (("unigram", lambda k: table.p1.astype(np.float64)), ("bigram", lambda k: table.distribution(seq[k])), ("pico", lambda k: pico[k])):
            steps = [step(dist(k), t, name) for k, t in enumerate(ids)]
            item[predictor] = {"c0": [s["c0"] for s in steps], "c1": [s["c1"] for s in steps], "slices": [s["slices"] for s in steps]}
        sentences.append(item)

    pairs = json.loads((out / "tinystories-token-pairs.json").read_text())
    stats = json.loads((repo_dir() / "results" / "data" / "tinystories" / "bpe4k" / "stats.json").read_text())
    by_name = {r["name"].split(" ")[0]: r for r in stats["compressors"]["results"]}
    meta = json.loads((work_dir() / "data" / "tinystories" / "bpe4k" / "meta.json").read_text())
    pico_meta = json.loads((out / "pico.json").read_text())
    q = pico_meta["quantization"]
    corpus = {"bytes": int(meta["val_bytes"]), "tokens": int(meta["n_val_tokens"]),
              "bitsPerByte": {"gzip": by_name["gzip"]["bits_per_byte"], "xz": by_name["xz"]["bits_per_byte"],
                              "uniform": pairs["heldOut"]["bitsPerByte"]["uniform"], "unigram": pairs["heldOut"]["bitsPerByte"]["unigram"],
                              "bigram": pairs["heldOut"]["bitsPerByte"]["bigram"],
                              "pico": q["loss"][q["chosen"]] / math.log(2.0) / float(meta["bytes_per_token_val"])},
              "picoTokens": int(q["tokens"]),
              "note": "compressors: gzip -9 and xz -9e on the held-out stories as one stream; tables: every held-out token; "
                      "pico: the shipped weights on the trainer's held-out windows"}
    data = {"name": NAME, "vocabulary": table.vocab, "bos": tok.bos_id, "widest": WIDEST, "maxTokens": MAX_TOKENS, "sentences": sentences, "corpus": corpus}
    write_atomic(out / f"{NAME}.json", json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    report = {"corpus": corpus, "sentences": {}}
    for s in sentences:
        report["sentences"][s["key"]] = {"tokens": len(s["tokens"]), "bytes": s["bytes"], **{k: round(float(-np.log2(np.array(s[k]["c1"]) - np.array(s[k]["c0"])).sum()), 2) for k in ("unigram", "bigram", "pico")}}
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(DATA))
    print(json.dumps(export(Path(ap.parse_args().out)), indent=1))


if __name__ == "__main__":
    main()
