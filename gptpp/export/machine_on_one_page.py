"""The numbers of the site's figure `gpt-machine-on-one-page` (the root page's blueprint of the whole model): for the
2019 recipe and for the 2026 recipe at the base size, the parameter count of every part as this repository counts
it, the shapes, one real text carried from bytes to tokens (and, with a checkpoint, to the model's next-token
distribution), the nine changes between the two recipes, and the Section of the series that explains each part.

    python -m gptpp.export.machine_on_one_page [--ckpt <model.pt | ckpt_<step>.pt | run directory>] [--text "..."]
                                                [--device cpu] [--out <dir>]

Without `--ckpt` the file has no next-token distribution (`example.next` is null) and the figure draws the machine
without it. With it (the final `model.pt` of the main run, or any checkpoint of it while it trains) the model reads
the text once on the CPU and the five likeliest next tokens are stored. Nothing else is measured: the parameter
counts are those of a freshly built model of each preset (`GPT(preset(name))`), counted tensor by tensor and
checked against `GPT.num_params()`.

The Sections are read from the drafts of the posts (`$GPTPP_SITE/_posts/gpt/*-gpt-NN.md`): a part's title, and its
`## ` headings in order, which the site numbers N.1, N.2, ... A part of the machine names the heading that explains
it by a few words of the heading (`EXPLAINED` below); if a heading is renamed and the words no longer match, the link
falls back to the part's page and the script says so. Run it again when the posts' headings change.

The file (`machine-on-one-page.json`), field by field:

    generated_by   "gptpp.export.machine_on_one_page"
    model          {n_layer, n_head, d_model, d_head, d_mlp, vocab_size, block_size}: the shape both recipes share
    recipes        {"2019": {...}, "2026": {...}}, each:
      preset         "gpt2-2019-base" | "gptpp-base"
      config         the fields that differ between the recipes: {pos, norm, embed_norm, act, qk_norm, qk_scale,
                     bias, tie_embeddings, logit_softcap, init, optimizer, rope_base}
      params         GPT.num_params(): {embedding, position, unembedding, attention, mlp, norm, total}
      parts          parameters of each part of the figure: {embed, pos, norm0, norm1, qkv, wo, norm2, win, wout,
                     normf, unembed}; the parts inside a block (norm1 ... wout) are for ONE block
      block          parameters of one block; blocks: of all of them
      init_std       the standard deviation the writers of a block (wo, wout) start with (0 for zero)
    example        {text, n_bytes, tokens (the pieces, as text), ids, bos (true: the model also reads its
                   beginning-of-text token first), T (tokens the model reads), next: null | [{token, p}] (the five
                   likeliest next tokens), from: null | {run, step, tokens (seen in training)}}
    changes        the nine steps from the 2019 recipe to the 2026 recipe, in the order of the ladder's rungs:
                   [{step, rung (the ladder's name of the step), parts (keys of the figure's parts it changes), what,
                   year, origin}]
    outline        {"1": {title, heads: [..]}, ...}: the Sections; heads is empty while a post has no draft
    explained      {part: [{section, k (0: the page), label, href}]}: where each part is explained
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .common import DATA, SITE, write_json

NAME = "machine-on-one-page"
TEXT = "Water is made of hydrogen and"
TITLES = {1: "Predicting the next token", 2: "Tokens and embeddings", 3: "Attention", 4: "Position", 5: "The block",
          6: "Optimization", 7: "Training gpt-pp", 8: "What training learns", 9: "Circuits", 10: "Features",
          11: "Frontier, reading and recipe"}

# where each part of the figure is explained: (Section, words of the heading), in order of importance
EXPLAINED = {
    "text": [(1, "One distribution")],
    "bytes": [(2, "bytes to tokens")],
    "tokens": [(2, "bytes to tokens"), (2, "Zipf")],
    "embed": [(2, "embedding table"), (2, "High-dimensional")],
    "pos": [(4, "Absolute positions")],
    "norm0": [(5, "Normalization")],
    "stream": [(5, "residual stream")],
    "norm1": [(5, "Normalization")],
    "qkv": [(3, "The mechanism")],
    "score": [(3, "scores are scaled"), (4, "Rotary")],
    "pattern": [(3, "The mechanism"), (3, "Three readings")],
    "mix": [(3, "The mechanism"), (3, "Decoding")],
    "wo": [(5, "Initialization")],
    "norm2": [(5, "Normalization")],
    "win": [(5, "MLP and its activation")],
    "act": [(5, "MLP and its activation")],
    "wout": [(5, "What the MLP stores"), (5, "Initialization")],
    "biases": [(5, "2019 block beside")],
    "normf": [(5, "Normalization")],
    "unembed": [(2, "unembedding")],
    "cap": [(5, "2019 block beside")],
    "softmax": [(2, "unembedding")],
    "sample": [(7, "Sampling"), (3, "Decoding")],
    "loss": [(1, "One distribution"), (1, "compression")],
    "optimizer": [(6, "Steepest descent"), (6, "Adam and AdamW")],
    "run": [(7, "recipe and the run"), (7, "Evaluation")],
    "learns": [(8, "")],
    "circuits": [(9, "framework"), (9, "induction head")],
    "features": [(10, "Neurons that mean"), (10, "monosemanticity")],
    "frontier": [(11, "field settled"), (11, "models behind")],
}

# the nine steps of the ladder (experiments/small.jsonl), with what each changes in the figure and where it comes
# from (docs/DESIGN.md, section 2: the sources of each decision)
CHANGES = [
    ("no-biases", ["biases"], "no biases", 2022, "PaLM, Chowdhery et al., 2022"),
    ("rotary", ["pos", "score"], "rotary positions", 2021, "Su et al., 2021"),
    ("rmsnorm", ["norm1", "norm2", "normf"], "RMS norm without parameters", 2019, "Zhang and Sennrich, 2019; the gain dropped in 2024"),
    ("relu2", ["act"], "squared ReLU", 2021, "So et al., 2021"),
    ("qk-norm", ["score"], "cosine scores", 2020, "Henry et al., 2020"),
    ("zero-init", ["wo", "wout"], "writers start at zero", 2019, "Zhang, Dauphin and Ma, 2019"),
    ("muon", ["optimizer"], "Muon for the hidden matrices", 2024, "Jordan et al., 2024"),
    ("untied", ["unembed", "norm0"], "untied unembedding, norm after the embedding", 2024, "modded-nanogpt, November 2024"),
    ("softcap", ["cap"], "soft cap on the logits", 2024, "Gemma 2, 2024, after Bello et al., 2016"),
]


def count_parts(preset_name: str) -> dict:
    """Parameters of each part of a freshly built model, tensor by tensor."""
    from ..config import preset, train_preset
    from ..model import GPT

    cfg = preset(preset_name)
    model = GPT(cfg)
    parts = dict.fromkeys(["embed", "pos", "norm0", "norm1", "qkv", "wo", "norm2", "win", "wout", "normf", "unembed"], 0)
    where = [(r"^embed\.", "embed"), (r"^pos_embed\.", "pos"), (r"^embed_norm\.", "norm0"), (r"^blocks\.0\.norm1\.", "norm1"),
             (r"^blocks\.0\.attn\.[qkv]\.", "qkv"), (r"^blocks\.0\.attn\.o\.", "wo"), (r"^blocks\.0\.norm2\.", "norm2"),
             (r"^blocks\.0\.mlp\.fc\.", "win"), (r"^blocks\.0\.mlp\.proj\.", "wout"), (r"^final_norm\.", "normf"),
             (r"^unembed\.", "unembed")]
    total_named, block, other_blocks = 0, 0, 0
    for name, p in model.named_parameters():
        total_named += p.numel()
        if re.match(r"^blocks\.(?!0\.)", name):
            other_blocks += p.numel()
            continue
        key = next((k for pattern, k in where if re.match(pattern, name)), None)
        if key is None:
            raise ValueError(f"{preset_name}: no part of the figure holds the tensor {name}")
        parts[key] += p.numel()
        if name.startswith("blocks.0."):
            block += p.numel()
    n = model.num_params()
    in_block = ["norm1", "qkv", "wo", "norm2", "win", "wout"]
    assert block == sum(parts[k] for k in in_block) and other_blocks == block * (cfg.n_layer - 1)
    assert total_named == n["total"] == sum(parts[k] for k in parts if k not in in_block) + block * cfg.n_layer
    assert n["attention"] == (parts["qkv"] + parts["wo"]) * cfg.n_layer and n["mlp"] == (parts["win"] + parts["wout"]) * cfg.n_layer
    assert n["norm"] == (parts["norm1"] + parts["norm2"]) * cfg.n_layer + parts["normf"] + parts["norm0"]
    assert n["embedding"] == parts["embed"] and n["position"] == parts["pos"] and n["unembedding"] == parts["unembed"]
    t = train_preset(preset_name)
    init_std = 0.0 if cfg.init == "zero_proj" else 0.02 / (2 * cfg.n_layer) ** 0.5
    return {"preset": preset_name,
            "config": {"pos": cfg.pos, "norm": cfg.norm, "embed_norm": cfg.embed_norm, "act": cfg.act, "qk_norm": cfg.qk_norm,
                       "qk_scale": cfg.qk_scale, "bias": cfg.bias, "tie_embeddings": cfg.tie_embeddings,
                       "logit_softcap": cfg.logit_softcap, "init": cfg.init, "optimizer": t.optimizer, "rope_base": cfg.rope_base},
            "params": n, "parts": parts, "block": block, "blocks": block * cfg.n_layer, "init_std": init_std,
            "shape": {"n_layer": cfg.n_layer, "n_head": cfg.n_head, "d_model": cfg.d_model, "d_head": cfg.d_head, "d_mlp": cfg.d_mlp,
                      "vocab_size": cfg.vocab_size, "block_size": cfg.block_size}}


def example(text: str, ckpt: str | None, device: str) -> dict:
    from ..tokenizer import Tokenizer

    tok = Tokenizer.load("bpe32k")
    ids = tok.encode(text)
    out = {"text": text, "n_bytes": len(text.encode("utf-8")), "tokens": [tok.decode([i]) for i in ids], "ids": ids, "bos": True,
           "T": len(ids) + 1, "next": None, "from": None}
    if ckpt:
        import torch
        from ..model import GPT

        path = Path(ckpt)
        path = path / "model.pt" if path.is_dir() else path
        model, cfg = GPT.from_checkpoint(path, device=device)
        meta = torch.load(path, map_location="cpu", weights_only=True)
        with torch.no_grad():
            logits = model(torch.tensor([tok.encode(text, bos=True)], device=device))
        p = torch.softmax(logits[0, -1].float(), dim=-1)
        top = torch.topk(p, 5)
        out["next"] = [{"token": tok.decode([int(i)]), "p": float(v)} for v, i in zip(top.values, top.indices)]
        out["from"] = {"run": path.parent.name, "step": int(meta.get("step", 0)), "tokens": int(meta.get("tokens", 0)), "final": path.name == "model.pt"}
        assert cfg.vocab_size == tok.vocab_size
    return out


def outline(site: Path) -> dict:
    """The Sections of the series: the title and the `## ` headings of each post that has a draft."""
    out = {}
    for n, title in TITLES.items():
        heads = []
        posts = sorted((site / "_posts" / "gpt").glob(f"*-gpt-{n:02d}.md"))
        if posts:
            body = re.sub(r"```.*?```", "", posts[0].read_text(encoding="utf-8"), flags=re.S)
            heads = [h.strip() for h in re.findall(r"^## (.+)$", body, flags=re.M)]
            m = re.search(r'^title: "Keys and values, \d+\. (.+)"$', body, flags=re.M)
            if m:
                title = m.group(1)
        out[str(n)] = {"title": title, "heads": heads}
    return out


def explained(sections: dict) -> tuple[dict, list[str]]:
    out, missing = {}, []
    for part, wants in EXPLAINED.items():
        links = []
        for n, words in wants:
            s = sections[str(n)]
            k = next((i + 1 for i, h in enumerate(s["heads"]) if words and words.lower() in h.lower()), 0)
            if words and not k and s["heads"]:
                missing.append(f"{part}: no heading of Section {n} has the words {words!r}")
            links.append({"section": n, "k": k, "label": s["heads"][k - 1] if k else s["title"],
                          "href": f"/gpt-{n:02d}" + (f"#section-{n}-{k}" if k else "")})
        out[part] = links
    return out, missing


def build(text: str, ckpt: str | None, device: str, site: Path) -> tuple[dict, list[str]]:
    recipes = {"2019": count_parts("gpt2-2019-base"), "2026": count_parts("gptpp-base")}
    shape = recipes["2026"].pop("shape")
    assert recipes["2019"].pop("shape") == shape, "the two recipes share one shape"
    sections = outline(site)
    links, missing = explained(sections)
    changes = [{"step": k + 1, "rung": rung, "parts": parts, "what": what, "year": year, "origin": origin}
               for k, (rung, parts, what, year, origin) in enumerate(CHANGES)]
    return {"generated_by": "gptpp.export.machine_on_one_page", "model": shape, "recipes": recipes,
            "example": example(text, ckpt, device), "changes": changes, "outline": sections, "explained": links}, missing


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", default=None, help="a checkpoint of the main run: the model's next-token distribution on the text")
    ap.add_argument("--text", default=TEXT)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--site", default=str(SITE))
    ap.add_argument("--out", default=str(DATA))
    args = ap.parse_args()
    data, missing = build(args.text, args.ckpt, args.device, Path(args.site))
    size = write_json(Path(args.out) / f"{NAME}.json", data, )
    ex = data["example"]
    print(f"{NAME}.json: {size:,} bytes")
    for year, r in data["recipes"].items():
        print(f"  {year} {r['preset']:15} total {r['params']['total']:,}  one block {r['block']:,}  parts " +
              " ".join(f"{k} {v:,}" for k, v in r["parts"].items() if v))
    print(f"  text {ex['text']!r}: {ex['n_bytes']} bytes, tokens {ex['tokens']}" +
          (f"; next {[(q['token'], round(q['p'], 3)) for q in ex['next']]} from {ex['from']}" if ex["next"] else "; no checkpoint given"))
    drafted = [n for n, s in data["outline"].items() if s["heads"]]
    print(f"  Sections with a draft: {', '.join(drafted)}" + ("" if not missing else "\n  " + "\n  ".join(missing)))


if __name__ == "__main__":
    main()
