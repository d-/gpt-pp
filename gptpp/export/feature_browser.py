"""Export the dashboards of a dictionary's features: what fig-10-02 (`gpt-feature-browser`) draws.

    python -m gptpp.export.feature_browser --run <run dir> --layer 2 --dict topk-n4096-k16 [--random 60] [--out <dir>]

Two lists of features are written. `curated` is chosen by hand and named by what the top examples and the logit
effects show; it is read from `gptpp/export/curated/<model>-L<layer>-<dict>.json`, a list of
`{ id, name, kind, note }`. `random` is a seeded draw of latents that nobody chose, with no names: it is there so
that a reader can judge how many latents of the dictionary can be read at all.

Files: `feature-browser-<model>.json` and `.bin` (<model> is `pico` or `gptpp`), and `feature-browser.json`, the index
of the models exported so far (`{ models: { label: file }, order, default }`; gpt-pp is the default once it exists).

    json  model, dict (its scores), snippet (tokens before and after), density_hist (all latents, log10),
          removal_examples (how many of a feature's strongest firings its removal is measured at),
          vocab (the strings of every token the file names), lists { curated, random } (indices into features),
          kinds, features: [{ id, name, kind, note, density, peak, count, hist (activation histogram),
          fires [[token, share]], up / down [[token, logit effect of its direction read out directly]],
          held_up / held_down [[token, change of its logit when the feature is removed where it fires most]],
          near [[latent, cosine]],
          top [first row, rows], levels [[first row, rows] x 4, strongest first] }]
    bin   .tok   u8 [rows, length, 2]  the tokens of each snippet as indices into vocab, low byte first (65535: none)
          .act   u8 [rows, length]     the feature's activation on each token, as a fraction of its peak
          .at    u8 [rows]             which token of the snippet the example is about
          .auto  u8 [latents, 3]       for every latent of the dictionary: the token it fires on most (vocab index,
                                       two bytes) and the share of its firings on it (byte), for naming neighbors
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gptpp.export.common import DATA, Blob, write_json
from gptpp.export.sparsity_against_fidelity import DATASETS, model_label
from gptpp.interp import dashboard as B
from gptpp.interp import sae as S

KINDS = [
    {"key": "word", "label": "a word in its senses"},
    {"key": "token", "label": "one token"},
    {"key": "role", "label": "a role in the sentence"},
    {"key": "topic", "label": "a topic, a kind of thing"},
    {"key": "place", "label": "a place in the text"},
    {"key": "next", "label": "what comes next"},
    {"key": "mixed", "label": "more than one thing"},
]
TOP, PER_LEVEL = 10, 2  # snippets a feature: its strongest examples, and random ones at each level of activation
REMOVED = 64  # the strongest firings (no two in one snippet) at which a feature is removed to measure what it holds up
NONE = 65535


class Vocabulary:
    """The strings of the tokens a file names, numbered in the order they are first asked for."""

    def __init__(self, tokenizer):
        self.tokenizer, self.index, self.strings = tokenizer, {}, []

    def __call__(self, token: int) -> int:
        token = int(token)
        if token not in self.index:
            self.index[token] = len(self.strings)
            self.strings.append(self.tokenizer.decode([token]))
        return self.index[token]


def feature_entry(D: B.Dashboards, i: int, vocab: Vocabulary, rows: list, seed: int) -> dict:
    """The dashboard of latent i as JSON, its snippets appended to `rows` as (tokens, activations, focus)."""
    def add(examples) -> list[int]:
        first = len(rows)
        for t, _ in examples:
            ids, values, at = D.snippet(i, t)
            rows.append(([vocab(p) for p in ids], values / max(float(D.peak[i]), 1e-9), at))
        return [first, len(rows) - first]

    up, down = D.logits(i, 10)
    held_up, held_down = D.held(i, 10, REMOVED) if D.count[i] else ([], [])
    entry = {"id": i, "density": float(D.density[i]), "peak": float(D.peak[i]), "count": int(D.count[i]),
             "hist": D.histogram(i) if D.count[i] else [0] * B.BINS,
             "fires": [[vocab(p), share] for p, share in D.fires_on(i, 4)],
             "up": [[vocab(p), e] for p, e in up], "down": [[vocab(p), e] for p, e in down],
             "held_up": [[vocab(p), e] for p, e in held_up], "held_down": [[vocab(p), e] for p, e in held_down],
             "near": [[j, c] for j, c in D.neighbors(i, 5)]}
    entry["top"] = add(D.top(i, TOP)) if D.count[i] else [len(rows), 0]
    entry["levels"] = [add(level) for level in D.samples(i, PER_LEVEL, seed)] if D.count[i] else [[len(rows), 0]] * B.LEVELS
    return entry


def export(D: B.Dashboards, record: dict, model: dict, curated: list[dict], n_random: int, seed: int, name: str,
           out: Path) -> dict:
    vocab, rows, features, lists = Vocabulary(D.tokenizer), [], [], {"curated": [], "random": []}
    for c in curated:
        entry = feature_entry(D, int(c["id"]), vocab, rows, seed)
        entry.update(name=c["name"], kind=c["kind"], note=c.get("note", ""))
        lists["curated"].append(len(features))
        features.append(entry)
    rng = np.random.default_rng(seed)
    for i in sorted(rng.choice(D.sae.cfg.n, size=min(n_random, D.sae.cfg.n), replace=False).tolist()):
        lists["random"].append(len(features))
        features.append(feature_entry(D, i, vocab, rows, seed))
    length = B.LEFT + B.RIGHT + 1
    tok = np.full((len(rows), length), NONE, dtype=np.uint16)
    act = np.zeros((len(rows), length), dtype=np.uint8)
    at = np.zeros(len(rows), dtype=np.uint8)
    for r, (ids, values, focus) in enumerate(rows):
        tok[r, : len(ids)] = ids
        act[r, : len(ids)] = np.clip(np.rint(values * 255), 0, 255)
        at[r] = focus
    profile = D.profile()
    auto = np.zeros((D.sae.cfg.n, 3), dtype=np.uint8)
    for i in range(D.sae.cfg.n):
        v = vocab(profile["token"][i]) if D.count[i] else NONE
        auto[i] = (v & 255, v >> 8, int(round(255 * profile["purity"][i])))
    blob = Blob(name, out)
    blob.put(f"{name}.tok", np.stack([tok & 255, tok >> 8], axis=-1).astype(np.uint8),
             "snippets: tokens as indices into vocab, low byte then high byte; 65535 is no token")
    blob.put(f"{name}.act", act, "snippets: the feature's activation on each token as a fraction of its peak", lo=0.0, hi=1.0)
    blob.put(f"{name}.at", at, "snippets: the index of the token each example is about")
    blob.put(f"{name}.auto", auto, "every latent: the token it fires on most (vocab index, two bytes) and its share of the firings (byte / 255)")
    size = blob.save()
    alive = D.density > 0
    e = record["eval"]
    doc = {
        "model": model,
        "dict": {"name": e["name"], "n": e["n"], "k": e["k"], "fvu": e["fvu"], "loss_clean": e["loss_clean"],
                 "loss_spliced": e["loss_spliced"], "loss_recovered": e["loss_recovered"], "dead": e["dead"], "dense": e["dense"],
                 "heldout": D.codes.counted, "train_tokens": record["train_tokens"]},
        "snippet": {"left": B.LEFT, "right": B.RIGHT, "len": length, "top": TOP, "per_level": PER_LEVEL, "levels": B.LEVELS},
        "density_hist": {"lo": -7.0, "hi": 0.0,
                         "counts": np.histogram(np.log10(D.density[alive]), bins=28, range=(-7.0, 0.0))[0].tolist()},
        "removal_examples": REMOVED,
        "vocab": vocab.strings, "lists": lists, "kinds": KINDS, "seed": seed, "features": features,
    }
    return {"json": write_json(out / f"{name}.json", doc), "bin": size, "rows": len(rows), "features": len(features)}


def run_facts(run_dir: Path, layer: int, record: dict) -> dict:
    """What a figure says about the model a dictionary was trained on."""
    config = json.loads((run_dir / "config.json").read_text())["model_config"]
    return {"label": model_label(run_dir), "layers": config["n_layer"], "width": config["d_model"], "after": layer + 1,
            "dataset": DATASETS.get(record["dataset"], record["dataset"]), "checkpoint": record["checkpoint"]}


def update_index(path: Path, label: str, file: str) -> dict:
    """The small file that says which models have an export, so that a figure asks only for files that exist:
    { models: { label: file name }, order, default }. gpt-pp is the default view once it is there."""
    index = json.loads(path.read_text()) if path.exists() else {"models": {}}
    index["models"][label] = file
    index["order"] = [m for m in ("gpt-pp", "pico") if m in index["models"]]
    index["default"] = index["order"][0]
    write_json(path, index)
    return index


def curated_file(label: str, layer: int, name: str) -> Path:
    return Path(__file__).parent / "curated" / f"{label.replace('-', '')}-L{layer}-{name}.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--dict", required=True, help="the dictionary's name, e.g. topk-n4096-k16")
    ap.add_argument("--curated", default=None, help="the hand-made list (default: gptpp/export/curated/<model>-L<layer>-<dict>.json)")
    ap.add_argument("--random", type=int, default=60, help="latents drawn at random for the second list")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(DATA))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    run_dir = Path(args.run) if Path(args.run).is_dir() else S.work_dir() / "ckpt" / args.run
    D = B.open_dashboards(str(run_dir), args.layer, args.dict, args.ckpt, args.device)
    record = json.loads((S.dict_dir(run_dir.name, args.layer) / f"{args.dict}.json").read_text())
    model = run_facts(run_dir, args.layer, record)
    path = Path(args.curated) if args.curated else curated_file(model["label"], args.layer, args.dict)
    curated = json.loads(path.read_text()) if path.exists() else []
    name = f"feature-browser-{model['label'].replace('-', '')}"
    res = export(D, record, model, curated, args.random, args.seed, name, Path(args.out))
    update_index(Path(args.out) / "feature-browser.json", model["label"], name)
    print(f"{name}: {len(curated)} curated and {args.random} random features, {res['rows']} snippets; "
          f"json {res['json'] / 1e3:.0f} kB, bin {res['bin'] / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
