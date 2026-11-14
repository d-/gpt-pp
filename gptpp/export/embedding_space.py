"""Export for fig-02-04 `gpt-embedding-space-3d`: the tokens of a model as points in three principal directions of
each of its two vocabulary tables, with every listed token's nearest neighbors in the full space.

    python -m gptpp.export.embedding_space --ckpt $GPTPP_WORK/ckpt/gptpp-base/model.pt [--cloud 4000] [--listed 16384] [--k 10]

For the input embedding and for the unembedding, each row is first scaled to unit length (the model normalizes the
row it looks up, and a cosine ignores length anyway). Then:

    the picture    the principal directions of the `cloud` most frequent tokens (their unit rows, centered on their
                   mean) are found, and EVERY token is written as its three coordinates along the first three: the
                   figure draws the cloud, and can light any neighbor wherever it lies
    the neighbors  for every token, its k nearest other tokens by cosine among all the rows, in all d dimensions,
                   with the cosines. They are written in two files, so that a page loads the rarer half of the
                   vocabulary only when a reader asks for one of its tokens

    embedding-space.bin         ids             every token id, from the most frequent to the least (uint16 LE)
                                <table>.xyz     V x 3 float16, <table> = embed | unembed
                                <table>.near    listed x k token ids (uint16 LE, two bytes each): the neighbors of the
                                                `listed` most frequent tokens, in the order of `ids`
                                <table>.cos     listed x k cosines as bytes over [0, 1]
    embedding-space-rare.bin    <table>.near, <table>.cos    the same for the other V - listed tokens
    embedding-space.json        the checkpoint, the counts above, and per table the share of the cloud's variance
                                that each of the three directions carries

Runs on the CPU in about ten seconds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .common import DATA, WORK, Blob, write_json
from .embedding_tables import by_rank, corpus_counts, load_tables, model_info, neighbors, unit

NAME = "embedding-space"


def picture(Wn: np.ndarray, cloud: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """Coordinates of every unit row along the first three principal directions of the cloud, and the share of
    the cloud's variance in each. A direction's sign is fixed so that its cubed coordinates over the cloud sum to
    a positive number: an export of a later checkpoint then does not flip the picture by chance."""
    mean = Wn[cloud].mean(0)
    _, s, vt = np.linalg.svd(Wn[cloud] - mean, full_matrices=False)
    xyz = (Wn - mean) @ vt[:3].T
    for k in range(3):
        if np.sum(xyz[cloud, k] ** 3) < 0:
            xyz[:, k] = -xyz[:, k]
    share = s ** 2 / np.sum(s ** 2)
    return xyz, [float(v) for v in share[:3]]


def export(ckpt: str, cloud_n: int, listed_n: int, k: int, dataset: str, out: Path) -> dict:
    t = load_tables(ckpt)
    counts, stats = corpus_counts(dataset)
    order = by_rank(counts)
    cloud, V = order[:cloud_n], len(order)
    blob, rare = Blob(NAME, out), Blob(f"{NAME}-rare", out)
    blob.put(f"{NAME}.ids", order.astype("<u2").view(np.uint8).reshape(V, 2),
             "every token id from the most frequent to the least (uint16 little-endian); the first `cloud` are the drawn cloud")
    tables = {}
    for key, W, label in (("embed", t["E"], "input embedding"), ("unembed", t["U"], "unembedding")):
        Wn = unit(W)
        xyz, share = picture(Wn, cloud)
        near, cos = neighbors(Wn, order, k)
        cos8 = np.clip(np.rint(cos * 255.0), 0, 255).astype(np.uint8)
        blob.put(f"{NAME}.{key}.xyz", xyz.astype(np.float16),
                 f"every token's unit row of the {label} along the first three principal directions of the cloud")
        for b, part, rows, who in ((blob, slice(0, listed_n), listed_n, f"the {listed_n} most frequent tokens"),
                                   (rare, slice(listed_n, V), V - listed_n, f"the {V - listed_n} least frequent tokens")):
            b.put(f"{b.name}.{key}.near", near[part].astype("<u2").view(np.uint8).reshape(rows, k, 2),
                  f"the {k} nearest tokens by cosine, in all dimensions, of each of {who} in the {label} (uint16 little-endian ids)")
            b.put(f"{b.name}.{key}.cos", cos8[part], f"their cosines in the {label}", lo=0.0, hi=1.0)
        tables[key] = {"variance": share, "varianceTop3": float(sum(share)), "meanUnitRowNorm": float(np.linalg.norm(Wn[cloud].mean(0))),
                       "cosineNearest": {"mean": float(cos[:, 0].mean()), "tenth": float(cos[:, -1].mean())}}
    size, rare_size = blob.save(), rare.save()
    meta = {"name": NAME, "dataset": stats["dataset"], "tokenizer": stats["tokenizer"], "vocabulary": int(stats["vocab_size"]),
            "dimensions": int(t["E"].shape[1]), "cloud": cloud_n, "listed": listed_n, "k": k,
            "model": dict(model_info(t), label="gpt-pp"), "embed": tables["embed"], "unembed": tables["unembed"], "bytes": size,
            "rare": {"file": f"{NAME}-rare", "tokens": V - listed_n, "bytes": rare_size}}
    write_json(out / f"{NAME}.json", meta)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", default=str(WORK / "ckpt" / "gptpp-base"), help="model.pt, a ckpt_<step>.pt or a run folder")
    ap.add_argument("--cloud", type=int, default=4000, help="the most frequent tokens drawn as the cloud; the directions are theirs")
    ap.add_argument("--listed", type=int, default=16384, help="the most frequent tokens whose neighbors go in the first file")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--dataset", default="fineweb-edu/bpe32k")
    ap.add_argument("--out", default=str(DATA))
    args = ap.parse_args()
    print(json.dumps(export(args.ckpt, args.cloud, args.listed, args.k, args.dataset, Path(args.out))))


if __name__ == "__main__":
    main()
