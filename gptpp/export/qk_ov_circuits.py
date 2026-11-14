"""Export for fig-09-02 `gpt-qk-ov-circuits`: every head of the attention-only models as its two tables over a
chosen set of tokens, and the eigenvalues of its OV circuit, before training and after.

    python -m gptpp.export.qk_ov_circuits [--one circ-attn1] [--two circ-attn2-repeats] [--device cpu]

For each model, at its initialization (the checkpoint of step 0) and at the end of training, and for every head
(gptpp/interp/circuits.py):

    ov        the OV circuit U W_OV E^T cut down to the chosen tokens, [output token, source token], each column
              measured from its mean over the whole vocabulary, divided by the largest entry in absolute value (kept
              in `scale`) and stored as bytes
    q, k      the queries W_Q E[token] and the keys W_K E[token] of the chosen tokens, (tokens, d_head) each: the page
              turns the key by the distance asked for and takes the dot product, as the model does, so the QK circuit
              of any offset is exact (rotary positions: one table for each offset)
    eig       the d_head eigenvalues of the OV circuit, (real, imaginary), divided by the largest modulus
    positivity   sum of the eigenvalues over the sum of their moduli (Definition 9.2.9 of the post)

The tokens are about sixty frequent ones, in groups a reader can name (names, pronouns, verbs, nouns, adjectives,
function words, punctuation); a candidate is kept only if it is a single token of the vocabulary. For a head of the
second layer the tables are those of the terms without composition: what the head does with the token embeddings
themselves. Writes `qk-ov-circuits.json` and `.bin`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ..interp import circuits as C
from ..tokenizer import Tokenizer
from .circuit_runs import ONE_STORIES, TWO, Run
from .common import DATA, Blob, write_json

NAME = "qk-ov-circuits"
THRESHOLD = 0.8  # a head is counted as copying above this positivity
KINDS = [
    ("names", [" Lily", " Tim", " Tom", " Sue", " Max", " Mia", " Ben", " Sam", " Lucy", " Sara"]),
    ("pronouns", [" he", " she", " they", " it", " I", " you", " her", " his", " we", " them"]),
    ("verbs", [" was", " said", " went", " saw", " played", " wanted", " loved", " found", " had", " asked"]),
    ("nouns", [" ball", " park", " dog", " cat", " tree", " mom", " box", " bird", " toy", " car"]),
    ("adjectives", [" big", " little", " happy", " sad", " red", " new", " good", " scared", " small", " old"]),
    ("function words", [" the", " a", " and", " to", " of", " in", " with", " but", " that", " on"]),
    ("punctuation", [".", ",", "!", "?", " \"", "'s", ".\n", ":", ";"]),
]
PER_KIND = 8


def chosen_tokens(tokenizer: Tokenizer) -> tuple[list[int], list[dict]]:
    """The token ids in order, and the groups as {name, from, to} (indices into that list)."""
    ids, groups = [], []
    for name, words in KINDS:
        single = [tokenizer.encode(w)[0] for w in words if len(tokenizer.encode(w)) == 1][:PER_KIND]
        groups.append({"name": name, "from": len(ids), "to": len(ids) + len(single)})
        ids += single
    return ids, groups


@torch.no_grad()
def tables(model, ids: list[int]) -> dict:
    """ov (heads, n, n), scale (heads), q and k (heads, n, d_head), eig (heads, d_head, 2), positivity, by head with
    the layers one after another."""
    cfg, tokens = model.cfg, torch.tensor(ids, device=model.embed.weight.device)
    out: dict = {"ov": [], "scale": [], "q": [], "k": [], "eig": [], "positivity": [], "eigScale": []}
    for layer in range(cfg.n_layer):
        for head in range(cfg.n_head):
            columns = C.ov_circuit(model, layer, head, sources=tokens)  # (V, n)
            cut = (columns - columns.mean(dim=0, keepdim=True))[tokens]
            scale = max(cut.abs().max().item(), 1e-12)
            w = C.head_weights(model, layer, head)
            embed = C.embedding(model)[tokens]
            values = C.ov_eigenvalues(model, layer, head)
            modulus = max(values.abs().max().item(), 1e-12)
            out["ov"].append((cut / scale).cpu().numpy())
            out["scale"].append(scale)
            out["q"].append((embed @ w.q.T).cpu().numpy())
            out["k"].append((embed @ w.k.T).cpu().numpy())
            out["eig"].append(torch.stack((values.real, values.imag), dim=1).cpu().numpy() / modulus)
            out["eigScale"].append(modulus)
            out["positivity"].append(C.copying_statistic(values))
    return out


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--one", default=ONE_STORIES, help="the one-layer model")
    parser.add_argument("--two", default=TWO, help="the two-layer model")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=str(DATA))
    args = parser.parse_args()
    tokenizer = Tokenizer.load("bpe4k")
    ids, groups = chosen_tokens(tokenizer)
    blob, models, checks = Blob(NAME, args.out), [], []
    for key, name in (("one", args.one), ("two", args.two)):
        run = Run(name, args.device)
        first, last = run.checkpoints()[0], run.checkpoints()[-1]
        states = [("initial", first[0], tables(run.model(first[1]), ids)), ("trained", last[0], tables(run.model(), ids))]
        stack = lambda field, dtype: np.stack([np.stack(t[field]) for _, _, t in states]).astype(dtype)  # noqa: E731
        blob.put(f"{NAME}.ov.{key}", np.rint((stack("ov", np.float64) + 1) * 127.5).astype(np.uint8),
                 "OV circuit over the chosen tokens, each head divided by its largest entry: (state, head, output, source)", lo=-1.0, hi=1.0)
        blob.put(f"{NAME}.q.{key}", stack("q", np.float16), "queries of the chosen tokens: (state, head, token, d_head)")
        blob.put(f"{NAME}.k.{key}", stack("k", np.float16), "keys of the chosen tokens: (state, head, token, d_head)")
        blob.put(f"{NAME}.eig.{key}", stack("eig", np.float16), "eigenvalues of the OV circuit over their largest modulus: "
                 "(state, head, d_head, real | imaginary)")
        # A few entries of the QK circuit computed here, for the test of the page's own arithmetic.
        trained = run.model()
        for head, dest, source, offset in ((0, 3, 17, 1), (run.cfg.n_head - 1, 9, 0, 2), (2, 40, 48, 7), (5, 20, 33, 30)):
            layer, h = divmod(head + (run.cfg.n_layer - 1) * run.cfg.n_head, run.cfg.n_head)
            score = C.qk_circuit(trained, layer, h, offset=offset, dests=[ids[dest]], sources=[ids[source]])[0, 0].item()
            checks.append({"model": key, "state": 1, "head": layer * run.cfg.n_head + h, "dest": dest, "source": source,
                           "offset": offset, "score": score})
        describe = run.describe()
        describe.update(key=key, states=[{"key": s, "step": step, "tokens": step * run.tokens_per_step,
                                          "positivity": t["positivity"], "ovScale": t["scale"], "eigScale": t["eigScale"]}
                                         for s, step, t in states])
        models.append(describe)
        for s, _, t in states:
            print(f"{run.name} {s}: positivity " + " ".join(f"{p:+.2f}" for p in t["positivity"]))
    size = blob.save()
    out = {"threshold": THRESHOLD, "tokens": {"ids": ids, "names": [tokenizer.decode([i]) for i in ids], "kinds": groups},
           "models": models, "checks": checks}
    json_size = write_json(Path(args.out) / f"{NAME}.json", out)
    print(f"wrote {NAME}.json ({json_size:,} bytes) and {NAME}.bin ({size:,} bytes); {len(ids)} tokens in {len(groups)} kinds")


if __name__ == "__main__":
    main()
