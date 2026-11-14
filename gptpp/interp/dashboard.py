"""Feature dashboards: what the latents of a sparse dictionary fire on, and what they write.

A latent of a dictionary (gptpp/interp/sae.py) is a direction of the residual stream and a number per token. Whether
it deserves a name is decided by looking, and the looking has a standard form since "Towards Monosemanticity"
(Bricken et al., 2023): for each latent,

    density          the fraction of held-out tokens it fires on, and its place among all latents
    histogram        of its nonzero activations, in equal intervals from zero to its largest
    top examples     the tokens it fires on most strongly, each with the text around it and the latent's
                     activation on every token of that text
    samples          tokens drawn at random among those it fires on, at several levels of activation. The top
                     examples flatter a latent: what it means at a tenth of its peak is where a second meaning hides
    logit effects    the tokens whose logits its direction raises and lowers most when read through the final norm
                     and the unembedding (the direct path only: what later blocks make of it is not in this number)
    neighbors        the latents whose directions are closest to its own

One pass over the held-out stream records every nonzero activation of every latent (`collect`), sorted by latent,
and everything above is read from that record (`Dashboards`).

    python -m gptpp.interp.dashboard --run <run dir> --layer 2 --dict topk-n4096-k16 [--features 3,14,159] [--survey 40]

prints dashboards as text: of the latents named, or a survey of the dictionary by kind (one token, many tokens, a
place in the document, a burst inside one document) from which a curated list is chosen by hand.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from gptpp.interp import sae as S
from gptpp.model import GPT, LayerNorm, RMSNorm

LEFT, RIGHT = 12, 5  # tokens of text shown before and after the token a snippet is about (LEFT at most the harvest's skip)
LEVELS = 4  # activation levels sampled: quarters of the range from zero to the latent's peak
BINS = 12  # intervals of the activation histogram


@dataclass
class Codes:
    """Every nonzero activation of every latent on the held-out stream, sorted by latent and then by token.

    The activations of latent i are act[start[i]:start[i + 1]], on the tokens tok[start[i]:start[i + 1]] (indices
    into the held-out token stream). `counted` is the number of tokens that were read (see sae.Harvest).
    """

    start: np.ndarray  # (n + 1,) int64
    tok: np.ndarray  # (nnz,) int64
    act: np.ndarray  # (nnz,) float32
    counted: int

    def of(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        a, b = self.start[i], self.start[i + 1]
        return self.tok[a:b], self.act[a:b]

    def save(self, path: Path) -> None:
        np.savez(path, start=self.start, tok=self.tok.astype(np.uint32), act=self.act.astype(np.float16),
                 counted=np.int64(self.counted))

    @classmethod
    def load(cls, path: Path) -> "Codes":
        z = np.load(path)
        return cls(z["start"], z["tok"].astype(np.int64), z["act"].astype(np.float32), int(z["counted"]))


@torch.no_grad()
def collect(model: GPT, sae: S.SparseAutoencoder, tokens: np.ndarray, harvest: S.Harvest, model_batch: int = 64,
            n_windows: Optional[int] = None, device: str = "cuda", chunk: int = 4096) -> Codes:
    """Run the model and the dictionary over the held-out stream and keep every nonzero activation."""
    latents, where, values, counted = [], [], [], 0
    for first, idx in S.window_batches(tokens, harvest.context, model_batch, n_windows, device):
        keep = harvest.keep(idx)
        x = sae.normalize(S.activations(model, idx, harvest)[keep])
        position = (first * harvest.context + torch.arange(idx.numel(), device=device).view_as(idx))[keep]
        counted += len(x)
        for i in range(0, len(x), chunk):
            z = sae.encode(x[i:i + chunk])
            row, latent = z.nonzero(as_tuple=True)
            latents.append(latent.to(torch.int32).cpu().numpy())
            where.append(position[i:i + chunk][row].cpu().numpy())
            values.append(z[row, latent].cpu().numpy())
    latent, tok, act = np.concatenate(latents), np.concatenate(where), np.concatenate(values)
    order = np.argsort(latent, kind="stable")  # stable: within a latent the tokens stay in the order of the stream
    start = np.zeros(sae.cfg.n + 1, dtype=np.int64)
    np.cumsum(np.bincount(latent, minlength=sae.cfg.n), out=start[1:])
    return Codes(start, tok[order].astype(np.int64), act[order].astype(np.float32), counted)


def cached_codes(folder: Path, name: str, model: GPT, sae: S.SparseAutoencoder, tokens: np.ndarray,
                 harvest: S.Harvest, model_batch: int = 64, device: str = "cuda") -> Codes:
    """The record of a dictionary, from `<name>.codes.npz` beside it if that is newer than the dictionary."""
    path, weights = folder / f"{name}.codes.npz", folder / f"{name}.pt"
    if path.exists() and path.stat().st_mtime >= weights.stat().st_mtime:
        return Codes.load(path)
    codes = collect(model, sae, tokens, harvest, model_batch, device=device)
    codes.save(path)
    return codes


def through_final_norm(model: GPT, directions: torch.Tensor) -> torch.Tensor:
    """Directions of the stream as the unembedding sees them: the final norm's linear part applied (centering and
    gain for a LayerNorm, gain for an RMSNorm with one, nothing otherwise). The division by the stream's length
    is left out: it rescales all logit effects of a token by the same positive number."""
    norm = model.final_norm
    if isinstance(norm, LayerNorm):
        return (directions - directions.mean(-1, keepdim=True)) * norm.gain
    if isinstance(norm, RMSNorm) and norm.gain is not None:
        return directions * norm.gain
    return directions


class Dashboards:
    """The dashboards of one dictionary, read from its record on the held-out stream."""

    def __init__(self, model: GPT, sae: S.SparseAutoencoder, codes: Codes, tokens: np.ndarray, harvest: S.Harvest,
                 tokenizer=None):
        self.model, self.sae, self.codes, self.tokens, self.harvest, self.tokenizer = model, sae, codes, tokens, harvest, tokenizer
        n = sae.cfg.n
        self.count = np.diff(codes.start)
        self.density = self.count / codes.counted
        self.peak = np.zeros(n, dtype=np.float32)
        alive = self.count > 0
        self.peak[alive] = np.maximum.reduceat(codes.act, codes.start[:-1][alive])
        # where every token stands in its document: tokens since the last document-start mark
        marks = np.flatnonzero(np.asarray(tokens) == harvest.bos_id)
        self.doc_start = np.zeros(len(tokens), dtype=np.int64)
        self.doc_start[marks] = marks
        self.doc_start = np.maximum.accumulate(self.doc_start)
        self.doc_end = np.full(len(tokens), len(tokens), dtype=np.int64)  # the next mark after each token
        self.doc_end[marks] = marks
        self.doc_end = np.minimum.accumulate(self.doc_end[::-1])[::-1]
        with torch.no_grad():
            rows = sae.dictionary
            self.logit_effect = (through_final_norm(model, rows) @ model.unembed.weight.T.float()).cpu().numpy()  # (n, V)
            self.cosine = (rows @ rows.T).cpu().numpy() if n <= 16384 else None
            self.rows = rows.cpu().numpy()

    # ---- one latent

    def histogram(self, i: int) -> list[int]:
        """Counts of its nonzero activations in BINS equal intervals from zero to its peak."""
        _, act = self.codes.of(i)
        return np.histogram(act, bins=BINS, range=(0.0, max(float(self.peak[i]), 1e-9)))[0].tolist()

    def top(self, i: int, m: int = 20) -> list[tuple[int, float]]:
        """Its m strongest activations as (token index, activation), no two within one snippet of each other."""
        tok, act = self.codes.of(i)
        out: list[tuple[int, float]] = []
        for j in np.argsort(-act, kind="stable")[: 40 * m]:
            if all(abs(int(tok[j]) - t) > LEFT + RIGHT for t, _ in out):
                out.append((int(tok[j]), float(act[j])))
                if len(out) == m:
                    break
        return out

    def samples(self, i: int, per_level: int = 3, seed: int = 0) -> list[list[tuple[int, float]]]:
        """Random activating tokens at each of LEVELS levels of activation: level q holds the tokens whose
        activation is between q / LEVELS and (q + 1) / LEVELS of the peak; strongest level first."""
        tok, act = self.codes.of(i)
        rng = np.random.default_rng([seed, i])
        level = np.minimum((act / max(float(self.peak[i]), 1e-9) * LEVELS).astype(np.int64), LEVELS - 1)
        out = []
        for q in range(LEVELS - 1, -1, -1):
            members = np.flatnonzero(level == q)
            chosen = rng.choice(members, size=min(per_level, len(members)), replace=False) if len(members) else []
            out.append([(int(tok[j]), float(act[j])) for j in sorted(chosen, key=lambda j: -act[j])])
        return out

    def snippet(self, i: int, t: int, left: int = LEFT, right: int = RIGHT) -> tuple[np.ndarray, np.ndarray, int]:
        """The text around token t as the model read it: token ids, latent i's activation on each, and the index
        of t among them. The text stays inside t's window and inside t's document."""
        window = (t // self.harvest.context) * self.harvest.context
        a = max(t - left, window, int(self.doc_start[t]) + (self.tokens[self.doc_start[t]] == self.harvest.bos_id))
        b = min(t + right + 1, window + self.harvest.context, int(self.doc_end[t]), len(self.tokens))
        tok, act = self.codes.of(i)
        lo, hi = np.searchsorted(tok, a), np.searchsorted(tok, b)
        values = np.zeros(b - a, dtype=np.float32)
        values[tok[lo:hi] - a] = act[lo:hi]
        return np.asarray(self.tokens[a:b]).astype(np.int64), values, t - a

    def fires_on(self, i: int, m: int = 5) -> list[tuple[int, float]]:
        """The tokens it fires on most often: (token id, share of its firings)."""
        tok, _ = self.codes.of(i)
        if not len(tok):
            return []
        ids, counts = np.unique(np.asarray(self.tokens)[tok], return_counts=True)
        order = np.argsort(-counts)[:m]
        return [(int(ids[j]), float(counts[j] / len(tok))) for j in order]

    def logits(self, i: int, m: int = 10) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
        """The m tokens whose logits its direction raises most, and the m it lowers most: (token id, effect)."""
        e = self.logit_effect[i]
        up, down = np.argsort(-e)[:m], np.argsort(e)[:m]
        return [(int(j), float(e[j])) for j in up], [(int(j), float(e[j])) for j in down]

    def neighbors(self, i: int, m: int = 5) -> list[tuple[int, float]]:
        """The m latents whose directions have the largest cosine with its own."""
        c = self.cosine[i] if self.cosine is not None else self.rows @ self.rows[i]
        order = [j for j in np.argsort(-c)[: m + 1] if j != i][:m]
        return [(int(j), float(c[j])) for j in order]

    # ---- all latents: the numbers a survey sorts by

    def profile(self) -> dict[str, np.ndarray]:
        """For every latent: `purity` (the share of its firings on its most frequent token), `token` (that token),
        `place` (the mean number of tokens since the document began, where it fires), `early` (the share of its
        firings in the first 12 tokens of a document), `burst` (its firings per document it fires in)."""
        n, tokens = self.sae.cfg.n, np.asarray(self.tokens)
        purity, token, place, early, burst = (np.zeros(n) for _ in range(5))
        for i in np.flatnonzero(self.count > 0):
            tok, _ = self.codes.of(i)
            ids, counts = np.unique(tokens[tok], return_counts=True)
            j = int(np.argmax(counts))
            purity[i], token[i] = counts[j] / len(tok), ids[j]
            since = tok - self.doc_start[tok]
            place[i], early[i] = since.mean(), (since <= 12).mean()
            burst[i] = len(tok) / len(np.unique(self.doc_start[tok]))
        return {"purity": purity, "token": token.astype(np.int64), "place": place, "early": early, "burst": burst}

    # ---- as text

    def show(self, piece: int) -> str:
        text = self.tokenizer.decode([piece]) if self.tokenizer is not None else str(piece)
        return text.replace("\n", "\\n")

    def line(self, i: int, t: int) -> str:
        """A snippet on one line: the token it is about in [[double brackets]], other firing tokens in [brackets]."""
        ids, values, at = self.snippet(i, t)
        parts = []
        for j, (piece, v) in enumerate(zip(ids, values)):
            s = self.show(int(piece))
            parts.append(f"[[{s}]]" if j == at else f"[{s}]" if v > 0 else s)
        return "".join(parts)

    def text(self, i: int, top: int = 6, per_level: int = 2) -> str:
        """The dashboard of latent i as a few lines of text."""
        if self.count[i] == 0:
            return f"#{i}  dead: never fired on {self.codes.counted:,} held-out tokens"
        up, down = self.logits(i, 6)
        fires = "  ".join(f"{self.show(p)!r} {share:.0%}" for p, share in self.fires_on(i, 4))
        near = " ".join(f"#{j}({c:.2f})" for j, c in self.neighbors(i, 4))
        lines = [f"#{i}  density {self.density[i]:.2e}  peak {self.peak[i]:.1f}  fires on: {fires}",
                 "   raises: " + " ".join(repr(self.show(p)) for p, _ in up) + "   lowers: " + " ".join(repr(self.show(p)) for p, _ in down),
                 f"   near: {near}   histogram: {self.histogram(i)}"]
        lines += [f"   {a:6.1f}  {self.line(i, t)}" for t, a in self.top(i, top)]
        for q, level in enumerate(self.samples(i, per_level)):
            lines += [f"   {a:6.1f}~ {self.line(i, t)}" for t, a in level if q > 0]
        return "\n".join(lines)


def open_dashboards(run: str, layer: int, name: str, ckpt: Optional[str] = None, device: str = "cuda",
                    folder: Optional[str] = None, context: int = 0, skip: int = 16) -> Dashboards:
    """The dashboards of the dictionary `name` of a run's layer, with its record made or read from the cache."""
    from gptpp.generate import load_tokenizer

    model, data, source = S.load_run(run, ckpt, device)
    where = Path(folder) if folder else S.dict_dir(source["run"], layer)
    sae = S.load(where / f"{name}.pt", device)
    context = context or model.cfg.block_size
    harvest = S.Harvest(layer, context, skip, int(data.meta["bos_id"]))
    tokens = data.val_tokens()
    codes = cached_codes(where, name, model, sae, tokens, harvest, max(1, 32768 // context), device)
    return Dashboards(model, sae, codes, tokens, harvest, load_tokenizer(source["tokenizer"]))


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Print feature dashboards of a dictionary as text.")
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--dict", required=True, help="the dictionary's name, e.g. topk-n4096-k16")
    ap.add_argument("--features", default="", help="latents to print, comma separated")
    ap.add_argument("--survey", type=int, default=0, help="print this many latents of each kind")
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    D = open_dashboards(args.run, args.layer, args.dict, args.ckpt, args.device)
    alive = D.count > 0
    print(f"{args.dict}: {int(alive.sum())} of {len(alive)} latents fire on {D.codes.counted:,} held-out tokens; "
          f"median density {np.median(D.density[alive]):.2e}")
    for i in map(int, filter(None, args.features.split(","))):
        print(D.text(i, args.top))
    if args.survey:
        p = D.profile()
        common = alive & (D.count >= 200)
        rng = np.random.default_rng(0)
        kinds = {
            "one token (purity above 0.9)": np.flatnonzero(common & (p["purity"] > 0.9)),
            "many tokens (purity below 0.2)": np.flatnonzero(common & (p["purity"] < 0.2)),
            "the start of a document": np.flatnonzero(common)[np.argsort(-p["early"][common])][: 3 * args.survey],
            "late in a document": np.flatnonzero(common)[np.argsort(-p["place"][common])][: 3 * args.survey],
            "bursts inside one document": np.flatnonzero(common)[np.argsort(-p["burst"][common])][: 3 * args.survey],
        }
        for kind, members in kinds.items():
            chosen = rng.choice(members, size=min(args.survey, len(members)), replace=False)
            print(f"\n===== {kind}: {len(members)} latents; {len(chosen)} of them\n")
            for i in chosen:
                print(D.text(int(i), args.top))


if __name__ == "__main__":
    main()
