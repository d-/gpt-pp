"""Export for fig-02-07 `gpt-softmax-bottleneck`: a table of next-token probabilities of full rank, and the closest
tables that logits of width d can give.

    python -m gptpp.export.softmax_bottleneck [--tokens 200000000] [--n 300] [--iters 3000] [--out <dir>]

About three minutes on eight processor threads; no card.

The true table. Pairs of adjacent tokens are counted on the first `--tokens` training tokens of TinyStories under
pico's tokenizer (as the corpus statistics count their n-grams). The `--n` most frequent tokens are the contexts
c_1 .. c_n and also the candidates x_1 .. x_n, in the order of their frequency. P*(x_j | c_i) is the interpolated
Kneser-Ney bigram estimate of the corpus statistics (one discount D = n1 / (n1 + 2 n2), continuation counts below,
the uniform distribution under those), restricted to the n candidates and renormalized: the probability of the next
token given the last one and given that the next is one of the n. Every entry is positive. A = ln P* is stored as
float32, and the true table is DEFINED as the softmax of the stored rows, so that the page and this file agree to
rounding. A context weighs pi_i, its share of the counted contexts, in every mean.

The fits. For each width d of a ladder, H (n x d) and W (n x d) minimize the divergence

    D(d) = sum_i pi_i KL( P*(. | c_i) || softmax(H W^T)_i )        (bits a token)

by L-BFGS in float64, started from the truncated singular value decomposition of the row-centered table and, from
the second width on, also from the previous width's solution with the next singular directions of what it leaves;
the better of the two ends is kept. The factors are stored as float16 and D(d) is then computed again on the
stored factors: that is the number the page computes and the figure prints. Two further stops need no training:
d = n - 2, the row-centered table minus its weakest singular direction (a table of rank n - 2: one short), stored
as that one direction in float32; and d = n - 1, the row-centered table itself, which has rank n - 1 and is exact
(the softmax ignores the row means, which is why the bound of the theorem is rank(A) - 1 and not rank(A)).

For comparison the file also holds the divergence of the least-squares fit at every d from 0 to n - 1: the
truncated singular value decomposition of the row-centered table, which is the best rank-d approximation of the
logits in the Frobenius norm over all row shifts, and is NOT what a model trained on cross-entropy finds.

Writes `softmax-bottleneck.json` (tokens, weights, the ladder with its divergences, the least-squares curve, the
reference levels), `softmax-bottleneck.bin` (the table as float32, the factors as float16, the weakest direction
as float32) and the entries for the manifest.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .. import work_dir
from ..tokenizer import Tokenizer
from .common import DATA, Blob, write_json
from .counted_models import training_tokens

NAME = "softmax-bottleneck"
DATASET = "tinystories/bpe4k"
LADDER = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
LN2 = float(np.log(2.0))


def discount(counts: np.ndarray) -> float:
    ones, twos = int((counts == 1).sum()), int((counts == 2).sum())
    return ones / (ones + 2 * twos) if ones and twos else 0.5


def true_table(train: np.ndarray, vocab: int, n: int) -> dict:
    """The n most frequent tokens, the Kneser-Ney bigram table among them (rows sum to one), and the weights."""
    pair = np.bincount(train[:-1].astype(np.int64) * vocab + train[1:], minlength=vocab * vocab).reshape(vocab, vocab)
    uni = np.bincount(train, minlength=vocab)
    ids = np.argsort(-uni, kind="stable")[:n]
    d2 = discount(pair)
    cont = (pair > 0).sum(axis=0).astype(np.float64)          # by how many different tokens each token is preceded
    dc = discount(cont)
    lower = (np.maximum(cont - dc, 0) + dc * int((cont > 0).sum()) / vocab) / cont.sum()
    total, types = pair.sum(axis=1).astype(np.float64), (pair > 0).sum(axis=1)
    full = (np.maximum(pair[ids] - d2, 0) + d2 * types[ids, None] * lower[None, :]) / total[ids, None]
    kept = full[:, ids].sum(axis=1)                            # the mass of the n candidates after each context
    p = full[:, ids] / kept[:, None]
    sub = pair[np.ix_(ids, ids)]
    return {"ids": ids, "p": p, "pi": total[ids] / total[ids].sum(), "kept": kept, "count": uni[ids], "pairs": sub,
            "discount": d2, "discount_lower": dc, "pairs_counted": int(pair.sum()), "pairs_seen": int((pair > 0).sum()),
            "coverage_contexts": float(total[ids].sum() / total.sum())}


def softmax_rows(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def divergence(p: np.ndarray, pi: np.ndarray, z: np.ndarray) -> tuple[float, np.ndarray]:
    """Mean divergence in bits of softmax(z) from p, weighted by pi, and the divergence of each row."""
    zs = z - z.max(axis=1, keepdims=True)
    logq = zs - np.log(np.exp(zs).sum(axis=1, keepdims=True))
    rows = (p * (np.log(p) - logq)).sum(axis=1) / LN2
    return float((pi * rows).sum()), rows


def train_width(p: torch.Tensor, pi: torch.Tensor, h0: np.ndarray, w0: np.ndarray, iters: int) -> tuple[np.ndarray, np.ndarray, float, int]:
    """L-BFGS on the divergence from a start (h0, w0): -> H, W, divergence in bits, iterations."""
    h = torch.tensor(np.ascontiguousarray(h0), dtype=torch.float64).requires_grad_(True)
    w = torch.tensor(np.ascontiguousarray(w0), dtype=torch.float64).requires_grad_(True)
    entropy = (pi * (p * torch.log(p)).sum(dim=1)).sum()

    def loss() -> torch.Tensor:
        return (entropy - (pi * (p * torch.log_softmax(h @ w.T, dim=1)).sum(dim=1)).sum()) / LN2

    opt = torch.optim.LBFGS([h, w], lr=1.0, max_iter=iters, max_eval=int(iters * 1.25), history_size=50,
                            tolerance_grad=1e-11, tolerance_change=1e-15, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        opt.zero_grad()
        value = loss()
        value.backward()
        return value

    opt.step(closure)
    with torch.no_grad():
        final = float(loss())
    return h.detach().numpy(), w.detach().numpy(), final, int(opt.state_dict()["state"][0]["n_iter"])


def export(out: Path, tokens: int, n: int, iters: int, threads: int) -> dict:
    t0 = time.time()
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    d = work_dir() / "data" / DATASET
    meta = json.loads((d / "meta.json").read_text())
    vocab, bos = int(meta["vocab_size"]), int(meta["bos_id"])
    tab = true_table(training_tokens(DATASET, tokens), vocab, n)
    tok = Tokenizer.load("bpe4k")
    names = ["<|bos|>" if int(i) == bos else tok.token_bytes(int(i)).decode("utf-8", errors="replace") for i in tab["ids"]]

    a32 = np.log(tab["p"]).astype(np.float32)                 # what the page holds
    p = softmax_rows(a32.astype(np.float64))                  # the true table, as the page computes it
    a = np.log(p)
    pi = tab["pi"]
    entropy = float(-(pi * (p * np.log2(p)).sum(axis=1)).sum())
    unigram = (pi[:, None] * p).sum(axis=0)
    centered = a - a.mean(axis=1, keepdims=True)
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    rank = int((s > s[0] * 1e-10).sum())
    if rank != n - 1:
        raise SystemExit(f"the row-centered table has rank {rank}, not {n - 1}: the figure's stops assume full rank")

    least = [divergence(p, pi, np.zeros_like(a))[0]]          # d = 0: the uniform guess
    z = np.zeros_like(a)
    for k in range(n - 1):                                    # the least-squares fit at every width, one direction at a time
        z = z + s[k] * np.outer(u[:, k], vt[k])
        least.append(divergence(p, pi, z)[0])

    blob = Blob(NAME, out)
    blob.put(f"{NAME}.table", a32, f"A = ln P*: {n} contexts by {n} tokens, natural logarithms, float32; the true table is the softmax of each row")
    pt, pit = torch.tensor(p), torch.tensor(pi)
    stops, prev = [], None
    for width in [w for w in LADDER if w < n - 2]:
        starts = [(u[:, :width] * np.sqrt(s[:width]), vt[:width].T * np.sqrt(s[:width]))]
        if prev is not None:                                  # the previous width's solution, widened by what it leaves
            hp, wp = prev
            zp = hp @ wp.T
            rest = centered - (zp - zp.mean(axis=1, keepdims=True))
            ur, sr, vtr = np.linalg.svd(rest, full_matrices=False)
            extra = width - hp.shape[1]
            starts.append((np.concatenate([hp, ur[:, :extra] * np.sqrt(sr[:extra])], axis=1),
                           np.concatenate([wp, vtr[:extra].T * np.sqrt(sr[:extra])], axis=1)))
        runs = [train_width(pt, pit, h0, w0, iters) for h0, w0 in starts]
        best = int(np.argmin([r[2] for r in runs]))
        h, w, trained, steps = runs[best]
        prev = (h, w)
        scale = np.sqrt(np.abs(h).max(axis=0) / np.abs(w).max(axis=0))     # balance the two factors column by column
        h16, w16 = (h / scale).astype(np.float16), (w * scale).astype(np.float16)
        zq = h16.astype(np.float64) @ w16.astype(np.float64).T
        bits, rows = divergence(p, pi, zq)
        blob.put(f"{NAME}.h{width}", h16, f"H at width {width}: one row a context, float16")
        blob.put(f"{NAME}.w{width}", w16, f"W at width {width}: one row a token, float16")
        stops.append({"d": width, "kind": "trained", "bits": bits, "bits_float64": trained, "bits_other_start": [r[2] for r in runs][1 - best] if len(runs) > 1 else None,
                      "start": "svd" if best == 0 else "previous", "iterations": steps, "least_squares_bits": least[width],
                      "top1": int((zq.argmax(axis=1) == a.argmax(axis=1)).sum()), "worst_bits": float(rows.max())})
        print(f"d = {width:3d}: {bits:.3e} bits stored ({trained:.3e} before rounding; start {stops[-1]['start']}, {steps} iterations; "
              f"least squares {least[width]:.3e}); {time.time() - t0:.0f} s", flush=True)

    weakest = np.concatenate([u[:, n - 2], vt[n - 2], [s[n - 2]]]).astype(np.float32)        # the last direction of the rank n - 1 table
    blob.put(f"{NAME}.weakest", weakest, f"the weakest singular direction of the row-centered table: u ({n}), v ({n}), then its singular value, float32")
    a64 = a32.astype(np.float64)
    c32 = a64 - a64.mean(axis=1, keepdims=True)
    wk = weakest.astype(np.float64)
    z_short = c32 - wk[2 * n] * np.outer(wk[:n], wk[n:2 * n])
    bits_short, rows_short = divergence(p, pi, z_short)
    stops.append({"d": n - 2, "kind": "one short", "bits": bits_short, "least_squares_bits": least[n - 2],
                  "top1": int((z_short.argmax(axis=1) == a.argmax(axis=1)).sum()), "worst_bits": float(rows_short.max())})
    bits_exact, rows_exact = divergence(p, pi, c32)
    stops.append({"d": n - 1, "kind": "exact", "bits": bits_exact, "least_squares_bits": least[n - 1], "top1": n, "worst_bits": float(rows_exact.max())})
    size = blob.save()

    data = {
        "name": NAME, "n": n, "rank": rank + 1, "bos": int(np.flatnonzero(tab["ids"] == bos)[0]) if bos in tab["ids"] else -1,
        "tokens": names, "ids": [int(i) for i in tab["ids"]],
        "pi": [float(f"{v:.6g}") for v in pi],
        "entropy_bits": entropy, "uniform_bits": least[0], "unigram_bits": divergence(p, pi, np.tile(np.log(unigram), (n, 1)))[0],
        "stops": stops, "least_squares_bits": least, "singular_values": [float(f"{v:.5g}") for v in s[: n - 1]],
        "counted": {"dataset": "TinyStories, the first training tokens, pico's tokenizer (4,096 tokens)", "tokens": int(tokens),
                    "pairs": tab["pairs_counted"], "pairs_seen": tab["pairs_seen"], "discount": tab["discount"], "discount_lower": tab["discount_lower"],
                    "pairs_among_n_never_seen": float((tab["pairs"] == 0).mean()), "contexts_share_of_text": tab["coverage_contexts"],
                    "next_token_among_n": float((pi * tab["kept"]).sum())},
        "bytes": size, "seconds": round(time.time() - t0, 1),
    }
    write_json(out / f"{NAME}.json", data)
    return {"bytes": size, "json_bytes": (out / f"{NAME}.json").stat().st_size, "entropy_bits": entropy, "uniform_bits": least[0], "unigram_bits": data["unigram_bits"],
            "stops": [(st["d"], float(f"{st['bits']:.4g}")) for st in stops], "seconds": data["seconds"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tokens", type=int, default=200_000_000, help="training tokens the pairs are counted on")
    ap.add_argument("--n", type=int, default=300, help="contexts and candidates: the n most frequent tokens")
    ap.add_argument("--iters", type=int, default=3000, help="L-BFGS iterations a start")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=str(DATA))
    args = ap.parse_args()
    print(json.dumps(export(Path(args.out), args.tokens, args.n, args.iters, args.threads), indent=1))


if __name__ == "__main__":
    main()
