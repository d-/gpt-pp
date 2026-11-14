"""Export for the figures of part 1 that predict with tables of tokens (fig-01-02 `gpt-guessing-game`, fig-01-03
`gpt-prediction-is-compression`): the unigram and the bigram model of TinyStories tokens, small enough for a page.

    python -m gptpp.export.token_pair_table [--tokens 200000000] [--min-count 100] [--out <dir>]

Counted on the first 200 million training tokens, as the n-gram models of the corpus statistics are. The unigram
model is theirs (interpolated Kneser-Ney of order 1). The bigram model is theirs with one change that makes it
fit a page: only the pairs seen at least `--min-count` times are kept, and what the others held goes to the
back-off, the Kneser-Ney distribution under the pairs (`counted_models.PairTable` has the formulas). Both models are
then scored on the held-out stream as the corpus statistics score (every token from the fourth on, bits per byte
in the trainer's convention), next to the unpruned bigram, so that the page can say what its table is worth.

Writes `tinystories-token-pairs.bin` (little-endian: three float32 vectors of the vocabulary's size, the row
starts as uint32, then for every kept pair the id of the second token as uint16 and -log2 of its discounted
relative frequency on 16 bits), `tinystories-token-pairs.json` (the layout, the counts, the scores, a check
vector) and the entries for the manifest.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .common import DATA, Blob, write_atomic
from .counted_models import LOG_RANGE, PairTable, held_out_tokens, training_tokens

NAME = "tinystories-token-pairs"
DATASET = "tinystories/bpe4k"


def export(out: Path, tokens: int, min_count: int) -> dict:
    held, meta = held_out_tokens(DATASET)
    vocab, per_token = int(meta["vocab_size"]), float(meta["bytes_per_token_val"])
    train = training_tokens(DATASET, tokens)
    table, facts = PairTable.count(train, vocab, min_count)
    whole, _ = PairTable.count(train, vocab, 1)
    start = 3                                                # as the corpus statistics: the same tokens for every order
    def bits(p: np.ndarray) -> float:
        return float(-np.log2(p).mean() / per_token)
    scores = {"unigram": bits(table.p1.astype(np.float64)[held[start:]]), "bigram": bits(table.probabilities(held, start)),
              "bigramUnpruned": bits(whole.probabilities(held, start)), "uniform": float(np.log2(vocab) / per_token)}

    blob = Blob(NAME, out)
    layout = {}
    def put(key: str, array: np.ndarray, dtype: str, note: str) -> None:
        raw = np.ascontiguousarray(array.astype(dtype))
        entry = blob.put(f"{NAME}.{key}", np.frombuffer(raw.tobytes(), dtype=np.uint8), note)
        layout[key] = {"offset": entry["offset"], "count": int(raw.size), "dtype": dtype}
    put("p1", table.p1, "<f4", "the probability of each token by its frequency (Kneser-Ney, order 1): float32 a token")
    put("lower", table.lower, "<f4", "the distribution under the pairs (continuation counts): float32 a token")
    put("back", table.back, "<f4", "the weight of the back-off after each token: float32 a token")
    put("start", table.start, "<u4", "where the followers of each token start among the kept pairs: uint32, vocabulary + 1 of them")
    put("follow", table.follow, "<u2", "the second token of every kept pair, row by row: uint16")
    put("code", table.code, "<u2", f"-log2 of the discounted relative frequency of every kept pair, 16 bits over 0 .. {LOG_RANGE:g}")
    size = blob.save()

    check_tokens = [int(t) for t in held[:24]]               # a check vector: the first held-out tokens under both models
    check = {"tokens": check_tokens,
             "unigramBits": [float(-np.log2(np.float64(table.p1[t]))) for t in check_tokens[1:]],
             "bigramBits": [float(-np.log2(table.distribution(a)[b])) for a, b in zip(check_tokens[:-1], check_tokens[1:])],
             "rowSums": [float(table.distribution(t).sum()) for t in check_tokens[:6]]}
    data = {"name": NAME, "vocabulary": vocab, "bos": int(meta["bos_id"]), "logRange": LOG_RANGE, "bytes": size, "layout": layout,
            "counted": {"dataset": "TinyStories, the first training tokens", **facts},
            "heldOut": {"tokens": int(len(held) - start), "bytesPerToken": per_token, "bitsPerByte": scores},
            "check": check}
    write_atomic(out / f"{NAME}.json", json.dumps(data, separators=(",", ":")).encode("utf-8"))
    return {"bytes": size, **facts, "bitsPerByte": scores}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tokens", type=int, default=200_000_000, help="training tokens the pairs are counted on")
    ap.add_argument("--min-count", type=int, default=100, help="a pair is kept when it was seen at least this often")
    ap.add_argument("--out", default=str(DATA))
    args = ap.parse_args()
    print(json.dumps(export(Path(args.out), args.tokens, args.min_count), indent=1))


if __name__ == "__main__":
    main()
