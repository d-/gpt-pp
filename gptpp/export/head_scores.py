"""Export for fig-03-02 `gpt-all-heads-of-pico`: what every attention head measurably does, as seven scores
(and the held-out passages that fig-03-02 and fig-03-08 `gpt-attention-masks-zoo` show the model reading).

    python -m gptpp.export.head_scores --ckpt $GPTPP_WORK/ckpt/pico
    python -m gptpp.export.head_scores --ckpt $GPTPP_WORK/ckpt/pico --name pico --documents 512 --tokens 128

The scores label the heads of the model that runs in the page, so they are measured on the page's own weights:
when the site's data folder holds `<name>.bin` (written by `gptpp.export.browser_model`), the model is built from
it (`page_model`), 8-bit matrices and all; `--float32` measures the checkpoint instead. Run this again whenever
`browser_model` has been run again for the same name.

Two batches go through the model on the explicit path, and the attention pattern A of every head (a row a reading
token, a column a token read, each row summing to one) is averaged into scores.

On held-out stories (the first `--tokens` tokens of the first `--documents` held-out documents that are at least
that long; every row begins with `<|bos|>`), over the reading positions i >= 2, so that "the token before" and
"the first token" are never the same token:

    previous    A[i, i - 1]: the weight on the token just before the reader          (Olsson et al., 2022)
    first       A[i, 0]: the weight on the start token, where a head rests when it has nothing to read
    self        A[i, i]: the weight on the reading token itself
    distance    how far back the attention to the text lies, in tokens: the sum of A[i, j] (i - j) over the text
                tokens j >= 1, divided by the weight that went to the text (the start token is left out, or a
                head that rests there would look long-range)
    entropy     the entropy of the row, in bits: 0 for a head that reads one token, log2(i + 1) for a uniform one

On repeated random tokens (`<|bos|>`, then `--length` tokens drawn uniformly from the tokens that occur in the
training set, special tokens excluded, then the same tokens again, `--repeats` times in all), over the reading
positions of the second and later repeats, where the token at i also stood at i - L, i - 2L, ...:

    duplicate   the weight on the earlier occurrences of the reading token: the sum over k of A[i, i - k L]
    induction   the weight on the tokens that followed those occurrences: the sum over k of A[i, i - k L + 1]
                (the prefix-matching score of Olsson et al., "In-context learning and induction heads", 2022)

`uniform` holds what a head that spreads its weight evenly over the tokens it may read would score on each: the
yardstick a score is read against.

Also written, for the figures that show this model reading a text: a few held-out stories (their first
`--passage-tokens` tokens, as text and as ids), the first of the held-out set, in its order, that are plain text
and whose cut ends a sentence. Never from the training set.

Writes `head-scores-<name>.json`: the set-up, the seven scores as layers x heads (indices from 0), the yardstick
and the passages. No arrays, so there is nothing for the manifest. A few seconds on the CPU for pico.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from ..data import TokenData
from ..model import GPT
from ..tokenizer import Tokenizer
from .common import DATA, write_json

NAME = "head-scores"
SCORES = ("previous", "first", "self", "distance", "entropy", "duplicate", "induction")


def story_openings(data: TokenData, tokens: int, documents: int) -> Tensor:
    """The first `tokens` tokens of the first `documents` held-out documents that have that many: (documents, tokens).
    Every row begins with `<|bos|>` and stays inside one document."""
    stream = np.asarray(data.val_tokens()).astype(np.int64)
    starts = np.flatnonzero(stream == data.meta["bos_id"])
    lengths = np.diff(np.append(starts, len(stream)))
    chosen = starts[lengths >= tokens][:documents]
    return torch.from_numpy(stream[chosen[:, None] + np.arange(tokens)[None, :]])


def drawable_tokens(data: TokenData) -> Tensor:
    """The ids a random sequence is drawn from: every token that occurs in the training set, except the special ones."""
    stats = json.loads((data.dir / "stats.json").read_text())["rank_frequency"]
    special = set(range(data.vocab_size - 5, data.vocab_size))
    return torch.tensor(sorted(i for i, count in zip(stats["ids"], stats["counts"]) if count > 0 and i not in special))


def repeated_sequences(pool: Tensor, bos: int, length: int, repeats: int, n: int, seed: int) -> Tensor:
    """n sequences `<|bos|>` r_1 .. r_L, the L tokens `repeats` times, the r drawn uniformly from the pool."""
    generator = torch.Generator().manual_seed(seed)
    first = pool[torch.randint(len(pool), (n, length), generator=generator)]
    return torch.cat([torch.full((n, 1), bos)] + [first] * repeats, dim=1)


@torch.no_grad()
def patterns(model: GPT, sequences: Tensor, batch: int):
    """The attention patterns of every head, batch by batch: yields (B, layers, heads, T, T) float64."""
    names = [f"attn.{l}.pattern" for l in range(model.cfg.n_layer)]
    for start in range(0, sequences.size(0), batch):
        _, cache = model.run_with_cache(sequences[start:start + batch], names=names)
        yield torch.stack([cache[name] for name in names], dim=1).double()


def story_scores(model: GPT, sequences: Tensor, batch: int) -> dict[str, Tensor]:
    """previous, first, self, distance and entropy of every head on text: each (layers, heads)."""
    T = sequences.size(1)
    i = torch.arange(2, T)
    back = (torch.arange(T)[:, None] - torch.arange(T)[None, :]).clamp(min=0).double()     # i - j
    total = {k: 0.0 for k in ("previous", "first", "self", "entropy", "reach", "text")}
    for A in patterns(model, sequences, batch):
        rows = A[..., 2:, :]                                                                # the readers i >= 2
        total["previous"] += A[..., i, i - 1].sum(dim=(0, -1))
        total["first"] += rows[..., 0].sum(dim=(0, -1))
        total["self"] += A[..., i, i].sum(dim=(0, -1))
        total["entropy"] += -(rows * torch.log2(rows.clamp(min=1e-300))).sum(dim=(0, -2, -1))
        total["reach"] += (rows[..., 1:] * back[2:, 1:]).sum(dim=(0, -2, -1))
        total["text"] += rows[..., 1:].sum(dim=(0, -2, -1))
    n = sequences.size(0) * (T - 2)
    out = {k: total[k] / n for k in ("previous", "first", "self", "entropy")}
    out["distance"] = total["reach"] / total["text"]
    return out


def repeat_scores(model: GPT, sequences: Tensor, length: int, repeats: int, batch: int) -> dict[str, Tensor]:
    """duplicate and induction of every head on repeated random tokens: each (layers, heads)."""
    total = {"duplicate": 0.0, "induction": 0.0}
    count = 0
    for A in patterns(model, sequences, batch):
        for rho in range(1, repeats):                                    # the readers of repeat rho + 1
            i = 1 + rho * length + torch.arange(length)
            for k in range(1, rho + 1):
                total["duplicate"] += A[..., i, i - k * length].sum(dim=(0, -1))
                total["induction"] += A[..., i, i - k * length + 1].sum(dim=(0, -1))
        count += A.size(0) * (repeats - 1) * length
    return {k: v / count for k, v in total.items()}


def uniform_scores(tokens: int, length: int, repeats: int) -> dict[str, float]:
    """What a head that spreads its weight evenly over the tokens it may read would score: the yardstick."""
    i = np.arange(2, tokens, dtype=np.float64)
    one = float(np.mean(1.0 / (i + 1.0)))                                 # one key among the i + 1 a reader sees
    readers = [(1 + rho * length + m, rho) for rho in range(1, repeats) for m in range(length)]
    match = float(np.mean([rho / (i + 1.0) for i, rho in readers]))       # rho earlier occurrences among i + 1 keys
    reach = float(np.sum((i * (i - 1.0) / 2.0) / (i + 1.0)) / np.sum(i / (i + 1.0)))       # text tokens 1 .. i, each 1 / (i + 1)
    return {"previous": one, "first": one, "self": one, "distance": reach, "entropy": float(np.mean(np.log2(i + 1.0))),
            "duplicate": match, "induction": match}


def passages(data: TokenData, tok: Tokenizer, tokens: int, count: int) -> list[dict]:
    """`count` held-out stories for the figures to read: the first of the held-out set, in its order, that are at
    least `tokens` tokens long, plain (letters, digits, spaces and the common punctuation only, so that every
    token shows as itself) and whose first `tokens` tokens end a sentence. Each as those tokens: the ids (the start
    token first) and their text."""
    stream = np.asarray(data.val_tokens()).astype(np.int64)
    starts = np.flatnonzero(stream == data.meta["bos_id"])
    lengths = np.diff(np.append(starts, len(stream)))
    plain = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?'\"-")
    out = []
    for k, (start, n) in enumerate(zip(starts, lengths)):
        if n < tokens:
            continue
        ids = stream[start:start + tokens].tolist()
        text = tok.decode(ids[1:])
        if set(text) <= plain and text[-1] in ".!?" and tok.encode(text) == ids[1:]:
            out.append({"document": int(k), "tokens": ids, "text": text})
        if len(out) == count:
            break
    return out


def export(ckpt: Path, out: Path, name: str, documents: int, tokens: int, sequences: int, length: int, repeats: int,
           batch: int, seed: int, device: str, passage_tokens: int, passage_count: int, float32: bool) -> dict:
    started = time.time()
    run = ckpt.parent if ckpt.is_file() else ckpt
    page = not float32 and (out / f"{name}.bin").exists() and (out / f"{name}.json").exists()
    if page:                                     # the weights the page runs: the scores then describe the heads a reader sees
        from .browser_model import page_model
        model, cfg = page_model(name, out, device)
        form = json.loads((out / f"{name}.json").read_text()).get("form", "as shipped")
    else:
        model, cfg = GPT.from_checkpoint(ckpt, device=device)
        form = "float32"
    config = json.loads((run / "config.json").read_text())
    data = TokenData(config["train_config"]["data"])
    tok = Tokenizer.load(data.meta["tokenizer"])
    text = story_openings(data, tokens, documents).to(device)
    pool = drawable_tokens(data)
    repeated = repeated_sequences(pool, data.meta["bos_id"], length, repeats, sequences, seed).to(device)
    scores = story_scores(model, text, batch)
    scores.update(repeat_scores(model, repeated, length, repeats, batch))
    result = {
        "name": f"{NAME}-{name}", "model": name, "layers": cfg.n_layer, "heads": cfg.n_head,
        "weights": f"the page's ({form})" if page else "the checkpoint's (float32)",
        "stories": {"documents": int(text.size(0)), "tokens": tokens, "readers": "positions 2 and later", "source": "held-out TinyStories"},
        "repeats": {"sequences": sequences, "length": length, "repeats": repeats, "seed": seed, "pool": int(len(pool)),
                    "readers": "the second and later repeats"},
        "order": list(SCORES),
        "scores": {k: scores[k].cpu().numpy().round(4).tolist() for k in SCORES},
        "uniform": uniform_scores(tokens, length, repeats),
        "passages": passages(data, tok, passage_tokens, passage_count),
    }
    write_json(out / f"{NAME}-{name}.json", result)
    result["seconds"] = time.time() - started
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True, help="model.pt, a ckpt_<step>.pt, or the run directory")
    ap.add_argument("--out", default=str(DATA))
    ap.add_argument("--name", default=None, help="the model's name in the file name (default: the run's directory)")
    ap.add_argument("--documents", type=int, default=512)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--sequences", type=int, default=256)
    ap.add_argument("--length", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8, help="CPU threads (the machine is shared)")
    ap.add_argument("--passage-tokens", type=int, default=64)
    ap.add_argument("--passage-count", type=int, default=4)
    ap.add_argument("--float32", action="store_true", help="measure the checkpoint, not the weights the page runs")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    ckpt = Path(a.ckpt)
    name = a.name or (ckpt.parent if ckpt.is_file() else ckpt).name
    r = export(ckpt, Path(a.out), name, a.documents, a.tokens, a.sequences, a.length, a.repeats, a.batch, a.seed, a.device,
               a.passage_tokens, a.passage_count, a.float32)
    print(f"{r['name']}.json: weights {r['weights']}; {r['layers']} x {r['heads']} heads, {r['stories']['documents']} stories of {r['stories']['tokens']} tokens, "
          f"{r['repeats']['sequences']} repeated sequences; {r['seconds']:.1f} s")
    for k in SCORES:
        print(f"  {k:10s}", " | ".join(" ".join(f"{v:5.2f}" for v in row) for row in r["scores"][k]))
    for p in r["passages"]:
        print(f"  passage (held-out document {p['document']}): {p['text'][:110]!r}")


if __name__ == "__main__":
    main()
