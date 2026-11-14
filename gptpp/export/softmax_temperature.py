"""Export for fig-02-06 `gpt-softmax-temperature`: what PyTorch says at the positions the figure opens on.

    python -m gptpp.export.softmax_temperature [--out <dir>]

The figure runs pico and pico-2019 in the page. This script runs the same two models in PyTorch, on exactly the
weights the page loads (`browser_model.page_model`: the 8-bit matrices of `<name>.bin`), on the figure's preset
sentences, at the last token of each, by the two paths the figure offers:

    whole     the model's own logits there
    direct    the embedding of the token, normalized and unembedded with no block between

and writes `softmax-temperature-test.json`: per preset, model and path the eight likeliest tokens with their logits
and their probabilities at temperature 1 and at the figure's default temperature, the entropy at a few
temperatures, the share of the thirty likeliest tokens, and the rank of the token itself among the answers of the
direct path. No page loads the file. The site's test of the figure compares what the page computes with it, and
takes from it every number it would otherwise have to type.

Run it again whenever pico or pico-2019 is exported again. CPU, a few seconds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..tokenizer import Tokenizer
from .browser_model import page_model
from .common import DATA, write_json

NAME = "softmax-temperature-test"
PRESETS = {  # the figure's (assets/js/gpt/gpt-softmax-temperature-figure.js keeps the same texts; its test compares)
    "named": "Once upon a time, there was a little girl named",
    "very": "The little dog was very",
    "upon": "Once upon a",
    "place": "One day, Tim and his mom went to the",
}
MODELS = ("pico", "pico-2019")
TEMPERATURE, ROWS, SHOWN = 0.8, 30, 8
TEMPERATURES = (0.1, 0.5, 0.8, 1.0, 2.0, 10.0)


def entropy_bits(p: torch.Tensor) -> float:
    p = p[p > 0]
    return float(-(p * torch.log2(p)).sum())


@torch.no_grad()
def at_last(model, ids: list[int]) -> dict:
    idx = torch.tensor([ids])
    whole = model(idx)[0, -1].double()
    direct = model.unembed_logits(model.final_norm(model.embed_tokens(idx)))[0, -1].double()
    out = {}
    for path, z in (("whole", whole), ("direct", direct)):
        p1, pT = torch.softmax(z, 0), torch.softmax(z / TEMPERATURE, 0)
        top = torch.topk(z, ROWS).indices
        order = torch.argsort(z, descending=True, stable=True)
        out[path] = {
            "top": top[:SHOWN].tolist(), "logits": z[top[:SHOWN]].tolist(), "p1": p1[top[:SHOWN]].tolist(), "pT": pT[top[:SHOWN]].tolist(),
            "entropy1": entropy_bits(p1), "entropyT": entropy_bits(pT), "lead1": float(p1[top].sum()), "leadT": float(pT[top].sum()),
            "entropy": {str(T): entropy_bits(torch.softmax(z / T, 0)) for T in TEMPERATURES},
            "selfRank": int((order == ids[-1]).nonzero()[0, 0]),
        }
    return out


def export(out: Path) -> dict:
    tok = Tokenizer.load("bpe4k")
    models = {name: page_model(name)[0] for name in MODELS}
    res = {"name": NAME, "temperature": TEMPERATURE, "rows": ROWS, "tied": {name: bool(m.cfg.tie_embeddings) for name, m in models.items()},
           "presets": {}, "results": {},
           "reference": "PyTorch (gptpp.model.GPT) on the weights the page loads, float64 softmax; row = the last token of the sentence"}
    for key, text in PRESETS.items():
        ids = tok.encode(text, bos=True)
        res["presets"][key] = {"text": text, "ids": ids, "last": tok.decode([ids[-1]])}
        res["results"][key] = {}
        for name, model in models.items():
            r = at_last(model, ids)
            for path in r:
                r[path]["tokens"] = [tok.decode([i]) for i in r[path]["top"]]
            res["results"][key][name] = r
    size = write_json(out / f"{NAME}.json", res)
    d = res["results"]["named"]
    return {"bytes": size, "named": {m: {p: [d[m][p]["tokens"][0], round(d[m][p]["p1"][0], 4), round(d[m][p]["entropy1"], 3)] for p in ("whole", "direct")} for m in MODELS}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(DATA))
    print(json.dumps(export(Path(ap.parse_args().out)), ensure_ascii=False))


if __name__ == "__main__":
    main()
