"""Export the directions of three dictionaries of growing size as points in three dimensions, with the families that
join them: what fig-10-03 (`gpt-feature-space-3d`) draws.

    python -m gptpp.export.feature_space --run <run dir> --layer 2 \
        --dicts topk-n1024-k16,topk-n2048-k16,topk-n4096-k16 [--threshold 0.5] [--default 0:123] [--out <dir>]

The projection. A dictionary's directions live in the d dimensions of the residual stream and are close to
orthogonal to one another, so their three leading principal components keep only a few percent of their variance
(the number is written into the export as `pca3`): a linear picture of them is a featureless ball. The points are
therefore placed by UMAP (McInnes, Healy and Melville, 2018, arXiv:1802.03426) on cosine distance, fitted once on
the directions of all three dictionaries together, which keeps directions with a high cosine near one another.
What the picture preserves is neighborhood; distances between far points and the overall shape mean nothing.

The families. Dictionaries of different sizes trained on the same activations find related directions: a latent
of the smaller often has several latents of the larger near it, which share its territory ("feature splitting",
Bricken et al., 2023). An edge joins a latent of one dictionary to each of its nearest latents in the next larger
one whose cosine with it is at least `threshold`, at most `--children` of them.

Files: `feature-space-<model>.json` and `.bin`, and `feature-space.json`, the index of the models exported so far.

    json  model, dicts [{ name, n, k, first, count }] (points are numbered through the dictionaries in order),
          projection, pca3, threshold, edges, snippet { left, right, len }, vocab, showcase (families chosen by hand to
          show first: [{ dict, id, note }]), split (how many latents of each dictionary have 0, 1, 2, 3 and more
          children)
    bin   .xyz u8 [points, 3] in [-1, 1] · .edge u8 [edges, 4] (parent, child: two bytes each, low first, as numbers
          within their own dictionaries) · .cos u8 [edges] in [threshold, 1] · .level u8 [edges] (0: first to second
          dictionary, 1: second to third) · .tok u8 [points, len, 2] · .act u8 [points, len] · .at u8 [points]
          (each point's strongest example) · .auto u8 [points, 3] (the token it fires on most and its share) ·
          .density u8 [points] (log10 of the share of tokens it fires on, over [-7, 0])
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gptpp.export.common import DATA, Blob, write_json
from gptpp.export.feature_browser import NONE, Vocabulary, run_facts, update_index
from gptpp.interp import dashboard as B
from gptpp.interp import sae as S

LEFT, RIGHT = 7, 2  # a point's example is short: ten tokens


def project(rows: np.ndarray, seed: int = 0, n_neighbors: int = 15, min_dist: float = 0.08) -> tuple[np.ndarray, dict]:
    """Unit directions (N, d) -> points (N, 3) inside the unit ball, by UMAP on cosine distance."""
    import umap

    points = umap.UMAP(n_components=3, metric="cosine", n_neighbors=n_neighbors, min_dist=min_dist,
                       random_state=seed).fit_transform(rows)
    points = points - points.mean(0)
    points = points / np.quantile(np.linalg.norm(points, axis=1), 0.995)
    norm = np.linalg.norm(points, axis=1, keepdims=True)
    points = np.where(norm > 1, points / norm, points)  # the few outliers are drawn on the sphere
    return points, {"method": "UMAP", "metric": "cosine", "n_neighbors": n_neighbors, "min_dist": min_dist, "seed": seed}


def pca_share(rows: np.ndarray, k: int = 3) -> float:
    """The share of the directions' variance that their k leading principal components keep."""
    s = np.linalg.svd(rows - rows.mean(0), compute_uv=False)
    return float((s[:k] ** 2).sum() / (s ** 2).sum())


def edges_between(small: np.ndarray, large: np.ndarray, threshold: float, children: int) -> list[tuple[int, int, float]]:
    """(parent, child, cosine) for every latent of the smaller dictionary and each of its `children` nearest latents
    of the larger whose cosine is at least `threshold`."""
    cos = small @ large.T
    order = np.argsort(-cos, axis=1)[:, :children]
    return [(i, int(j), float(cos[i, j])) for i in range(len(small)) for j in order[i] if cos[i, j] >= threshold]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--dicts", required=True, help="three dictionaries by name, smallest first")
    ap.add_argument("--threshold", type=float, default=0.5, help="the least cosine of an edge")
    ap.add_argument("--children", type=int, default=4, help="the most edges from one latent to the next dictionary")
    ap.add_argument("--families", default=None, help="families to show first, with a note each: a JSON list of { dict, id, note } "
                    "(default: gptpp/export/curated/<model>-L<layer>-families.json)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(DATA))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    run_dir = Path(args.run) if Path(args.run).is_dir() else S.work_dir() / "ckpt" / args.run
    names = args.dicts.split(",")
    boards = [B.open_dashboards(str(run_dir), args.layer, name, args.ckpt, args.device) for name in names]
    record = json.loads((S.dict_dir(run_dir.name, args.layer) / f"{names[-1]}.json").read_text())
    model = run_facts(run_dir, args.layer, record)
    rows = [D.rows / np.linalg.norm(D.rows, axis=1, keepdims=True) for D in boards]
    points, projection = project(np.concatenate(rows), args.seed)
    first = np.cumsum([0] + [len(r) for r in rows])
    edge, cosine, level, split = [], [], [], []
    for q in range(len(rows) - 1):
        found = edges_between(rows[q], rows[q + 1], args.threshold, args.children)
        edge += [(i, j) for i, j, _ in found]
        cosine += [c for _, _, c in found]
        level += [q] * len(found)
        kids = np.bincount([i for i, _, _ in found], minlength=len(rows[q]))
        split.append([int((kids == 0).sum()), int((kids == 1).sum()), int((kids == 2).sum()), int((kids >= 3).sum())])
    vocab = Vocabulary(boards[0].tokenizer)
    length, N = LEFT + RIGHT + 1, int(first[-1])
    tok = np.full((N, length), NONE, dtype=np.uint16)
    act, at = np.zeros((N, length), dtype=np.uint8), np.zeros(N, dtype=np.uint8)
    auto, density = np.zeros((N, 3), dtype=np.uint8), np.zeros(N, dtype=np.uint8)
    for q, D in enumerate(boards):
        profile = D.profile()
        for i in range(D.sae.cfg.n):
            p = int(first[q]) + i
            if D.count[i] == 0:
                auto[p] = (255, 255, 0)
                continue
            t, _ = D.top(i, 1)[0]
            ids, values, focus = D.snippet(i, t, LEFT, RIGHT)
            tok[p, : len(ids)] = [vocab(x) for x in ids]
            act[p, : len(ids)] = np.clip(np.rint(255 * values / float(D.peak[i])), 0, 255)
            at[p] = focus
            v = vocab(profile["token"][i])
            auto[p] = (v & 255, v >> 8, int(round(255 * profile["purity"][i])))
            density[p] = int(np.clip(np.rint(255 * (np.log10(D.density[i]) + 7) / 7), 0, 255))
    name = f"feature-space-{model['label'].replace('-', '')}"
    blob = Blob(name, Path(args.out))
    blob.put_u8(f"{name}.xyz", points, "every latent's direction as a point, by UMAP on cosine distance", lo=-1.0, hi=1.0)
    e = np.asarray(edge, dtype=np.uint16).reshape(-1, 2)
    blob.put(f"{name}.edge", np.stack([e[:, 0] & 255, e[:, 0] >> 8, e[:, 1] & 255, e[:, 1] >> 8], axis=-1).astype(np.uint8),
             "edges: a latent and one of its nearest latents in the next larger dictionary, as numbers within their dictionaries, two bytes each")
    blob.put_u8(f"{name}.cos", np.asarray(cosine), "edges: the cosine of the two directions", lo=args.threshold, hi=1.0)
    blob.put(f"{name}.level", np.asarray(level, dtype=np.uint8), "edges: 0 joins the first dictionary to the second, 1 the second to the third")
    blob.put(f"{name}.tok", np.stack([tok & 255, tok >> 8], axis=-1).astype(np.uint8), "each latent's strongest example: tokens as indices into vocab, two bytes; 65535 is no token")
    blob.put(f"{name}.act", act, "each latent's activation on the tokens of its example, as a fraction of its peak", lo=0.0, hi=1.0)
    blob.put(f"{name}.at", at, "the index of the token the example is about")
    blob.put(f"{name}.auto", auto, "the token each latent fires on most (vocab index, two bytes; 65535 for a dead latent) and its share of the firings (byte / 255)")
    blob.put(f"{name}.density", density, "log10 of the share of held-out tokens each latent fires on", lo=-7.0, hi=0.0)
    size = blob.save()
    chosen = Path(args.families) if args.families else Path(__file__).parent / "curated" / f"{model['label'].replace('-', '')}-L{args.layer}-families.json"
    showcase = json.loads(chosen.read_text()) if chosen.exists() else []
    if not showcase:  # nobody chose: the latent of the smallest dictionary with the most relatives in the next
        showcase = [{"dict": 0, "id": int(np.argmax(np.bincount([i for (i, _), l in zip(edge, level) if l == 0], minlength=len(rows[0])))), "note": ""}]
    doc = {"model": model, "dicts": [{"name": n, "n": len(r), "k": D.sae.cfg.k, "first": int(f), "count": len(r)} for n, r, D, f in zip(names, rows, boards, first)],
           "projection": projection, "pca3": pca_share(np.concatenate(rows)), "threshold": args.threshold, "children": args.children,
           "edges": len(edge), "split": split, "snippet": {"left": LEFT, "right": RIGHT, "len": length}, "heldout": boards[0].codes.counted,
           "vocab": vocab.strings, "showcase": showcase}
    js = write_json(Path(args.out) / f"{name}.json", doc)
    update_index(Path(args.out) / "feature-space.json", model["label"], name)
    print(f"{name}: {N} points, {len(edge)} edges at cosine >= {args.threshold}; children per latent (0, 1, 2, 3+): {split}; "
          f"three principal components keep {doc['pca3']:.1%}; json {js / 1e3:.0f} kB, bin {size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
