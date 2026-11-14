"""Export for fig-02-03 `gpt-zipf-tail-of-tokens`: how often each token of the vocabulary occurs, and what training
did to its two rows.

    python -m gptpp.export.zipf_and_embedding_norms --ckpt $GPTPP_WORK/ckpt/gptpp-base/model.pt

Read: the token counts of the training set (the dataset's `stats.json`: every token of the 9.9 billion counted);
the checkpoint (`model.pt`, a `ckpt_<step>.pt` or a run folder); and the run's `ckpt_0.pt`, the model as it was
initialized (`--init` names another file). Written, one value a token id:

    zipf-tail-of-tokens.bin    count           occurrences in the training set, uint32 little-endian (four bytes a token)
                               embed.norm      the length of the token's row of the input embedding
                               embed.moved     the distance of that row from where initialization put it
                               unembed.norm, unembed.moved   the same for the unembedding           (float16)
    zipf-tail-of-tokens.json   the corpus (tokens, the fitted exponent and its window, the entropy), the checkpoint
                               (run, step, tokens read, whether it is the finished model), the lengths at
                               initialization, and per table the rank correlations with log frequency and the number
                               of rows that are exactly where they started

The figure draws from the arrays and computes the rest (ranks, the cover of the top k, the trend) itself.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .common import DATA, WORK, Blob, write_json
from .embedding_tables import by_rank, corpus_counts, load_tables, model_info, spearman

NAME = "zipf-tail-of-tokens"


def table(W: np.ndarray, W0: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    norm, moved = np.linalg.norm(W, axis=1), np.linalg.norm(W - W0, axis=1)
    seen = counts > 0
    logc = np.log(counts[seen])
    info = {"spearmanNorm": spearman(norm[seen], logc), "spearmanMoved": spearman(moved[seen], logc),
            "atInit": int((moved == 0).sum()), "normAtInit": float(np.linalg.norm(W0, axis=1).mean()),
            "normMean": float(norm.mean()), "movedMean": float(moved.mean()),
            "movedMeanNeverSeen": float(moved[~seen].mean()) if (~seen).any() else None}
    return norm, moved, info


def export(ckpt: str, init: str | None, dataset: str, out: Path) -> dict:
    t = load_tables(ckpt)
    init_path = Path(init) if init else (Path(ckpt) if Path(ckpt).is_dir() else Path(ckpt).parent) / "ckpt_0.pt"
    t0 = load_tables(init_path)
    if t0["step"] != 0:
        raise SystemExit(f"{init_path} is not the initialization (step {t0['step']})")
    counts, stats = corpus_counts(dataset)
    if not (by_rank(counts) == np.asarray(stats["rank_frequency"]["ids"])).all():
        raise SystemExit("the ranks of stats.json are not (count descending, id ascending): the figure assumes that order")
    if counts.max() >= 2 ** 32:
        raise SystemExit("a count does not fit in 32 bits")
    blob = Blob(NAME, out)
    blob.put(f"{NAME}.count", counts.astype("<u4").view(np.uint8).reshape(len(counts), 4),
             "occurrences of each token id in the training set: uint32 little-endian, four bytes a token")
    info = {}
    for key, W, W0, label in (("embed", t["E"], t0["E"], "input embedding"), ("unembed", t["U"], t0["U"], "unembedding")):
        norm, moved, info[key] = table(W, W0, counts)
        blob.put(f"{NAME}.{key}.norm", norm.astype(np.float16), f"length of each token's row of the {label}")
        blob.put(f"{NAME}.{key}.moved", moved.astype(np.float16), f"distance of each token's row of the {label} from its initialization")
    size = blob.save()
    fit = stats["rank_frequency"]["zipf_fit"]
    meta = {
        "name": NAME, "dataset": stats["dataset"], "tokenizer": stats["tokenizer"], "vocabulary": int(stats["vocab_size"]),
        "tokens": int(stats["tokens"]["train"]), "neverSeen": int(stats["unigram"]["tokens_never_seen"]),
        "entropyBits": float(stats["unigram"]["entropy_bits_per_token"]),
        "zipf": {"exponent": float(fit["exponent"]), "log10CountAtRank1": float(fit["log10_count_at_rank_1"]), "from": 10, "to": 1000},
        "model": dict(model_info(t), label="gpt-pp", init=init_path.name),
        "embed": info["embed"], "unembed": info["unembed"], "bytes": size,
    }
    write_json(out / f"{NAME}.json", meta)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", default=str(WORK / "ckpt" / "gptpp-base"), help="model.pt, a ckpt_<step>.pt or a run folder")
    ap.add_argument("--init", default=None, help="the initialization (default: ckpt_0.pt beside the checkpoint)")
    ap.add_argument("--dataset", default="fineweb-edu/bpe32k")
    ap.add_argument("--out", default=str(DATA))
    args = ap.parse_args()
    meta = export(args.ckpt, args.init, args.dataset, Path(args.out))
    print(json.dumps({k: meta[k] for k in ("model", "embed", "unembed", "bytes")}))


if __name__ == "__main__":
    main()
