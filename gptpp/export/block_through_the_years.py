"""The transformer block from 2017 to 2026: the open models of the census as configurations of one block, and the
year in which each choice became the default, as data for the site's figure `gpt-block-through-the-years` (part 11).

    python -m gptpp.export.block_through_the_years [--dossier <frontier-2026.md>] [--out <dir>] [--show]

Nothing is measured here. The source is the research dossier `frontier-2026.md`, section 2: its table of 36 open
models (each cell read from a published configuration, card or report, with the row's source) and its timeline of
when each choice first appeared and when it became the default. The table is read with the reader of
`gptpp.export.open_models_table` (the figure `gpt-what-open-models-use` draws the same rows), which also builds the
series' two rows, the 2019 recipe and gpt-pp, from this repository's presets.

This script adds two things, both by rules over the dossier's words and never typed in per model:

  1. each model's cells as the state of the parts of one block (`f`, below), so that the figure can draw the block
     of any model. A part the dossier's cell does not state is null: the figure leaves it blank;
  2. the block of each year from 2017 to 2026 (`blocks`). It starts as the transformer of 2017 (the table's first
     row), takes GPT-2's cells in 2019 (the second row), and after that changes only when a row of the dossier's
     timeline says a choice became the default in open models in that year. A timeline row whose "default" cell is
     a year (with or without examples in brackets) changes the block (scope "all"); one that names a condition ("for
     large MoE", "contested", "at 1M context") is kept beside the block as a variant (scope "some"); one that says
     "not a default" is a note (scope "none").

The file (`block-through-the-years.json`), field by field:

    generated_by   "gptpp.export.block_through_the_years"
    source         {dossier, section, rows, timeline, read}
    years          [2017, ..., 2026]
    parts          the parts of the block in the order drawn: [{key, label, column}]; column is the column of the
                   dossier's table the part is read from
    models         in the order of release, then the series' two. Each:
      key, name      an identifier; the name as the dossier writes it
      released       "YYYY-MM"; date [year, month]; own: null | "gpt2-2019" | "gptpp"
      vocabulary, context    as written
      source         the row's source, as written (without the dossier's own cross-references)
      cells          {column: the cell as written} for norm, act, pos, attn, emb, moe, opt
      f              the state of the parts, read from the cells:
        norm           "layernorm" | "rmsnorm" | null          place   "post" | "pre" | "pre+post" | null
        act            "relu" | "gelu" | "swiglu" | "geglu" | "gated" | "relu2" | null
        pos            "sinusoidal" | "learned" | "rope" | null    nope, partial   true where the cell says so
        heads          "mha" | "gqa" | "latent" | "compressed" | null      q, kv   head counts where written
        pattern        "full" | "window" | "sparse"            out     "plain" | "sink" | "gate"
        score          "dot" | "qknorm" | "softcap" | null: null where no cell states it (configuration files rarely
                       show the score); "dot" also for a model older than the first other score of the timeline
        recurrent      null | {kind, n, of}: layers that are not attention (n of `of`; null where not written)
        add            "identity" | "replaced"                 add_text  the words of the cell that say so
        tied           true | false
        moe            null | {experts, active, shared, also_dense}
        mtp            true where a multi-token prediction layer is written
    events         the dossier's timeline, one a row: {choice, part, set (the state it sets), scope ("all" | "some" |
                   "none"), year (when it became the default, or was proposed if it never did), first_year, first,
                   default, replaced, qualifier}
    blocks         {year: {f (as above, plus bias, base, vocab, routing where the timeline sets them), since {state key:
                   {year, who}}, new [event indices that changed the block this year], some [event indices kept as
                   variants up to this year], tie {untied, of}: how many of the models released this year untie the
                   embedding (f.tied is null when they disagree), notes [event indices: not a default, this year], first [event indices
                   first seen this year, the default only later], rows [models released this year, by key]}}
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from .common import DATA, WORK, write_json
from .open_models_table import own_rows, read_dossier, shown_source

NAME = "block-through-the-years"
YEARS = list(range(2017, 2027))
PARTS = [("emb", "embedding", "emb"), ("pos", "position", "pos"), ("norm", "normalization", "norm"), ("attn", "attention", "attn"),
         ("add", "the residual add", "norm"), ("mlp", "the MLP and its activation", "act"), ("moe", "dense or experts", "moe"),
         ("tie", "unembedding", "emb"), ("opt", "optimizer", "opt")]
COLUMNS = ["norm", "act", "pos", "attn", "emb", "moe", "opt"]

# The dossier's timeline rows, by their name there: the part of the block each belongs to, and the state it sets.
TIMELINE = {
    "pre-norm": ("norm", {"place": "pre"}),
    "RMSNorm": ("norm", {"norm": "rmsnorm"}),
    "SwiGLU / GeGLU": ("mlp", {"act": "swiglu"}),
    "RoPE": ("pos", {"pos": "rope"}),
    "larger RoPE base (5e5 to 1e7)": ("pos", {"base": "large"}),
    "grouped-query attention": ("attn", {"heads": "gqa"}),
    "no biases": ("mlp", {"bias": False}),
    "vocabulary above 100K": ("emb", {"vocab": "above 100K"}),
    "multi-head latent attention": ("attn", {"heads": "latent"}),
    "fine-grained MoE with shared expert": ("moe", {"moe": True}),
    "sigmoid routing, auxiliary-loss-free balance": ("moe", {"routing": "sigmoid"}),
    "multi-token prediction head": ("tie", {"mtp": True}),
    "local:global interleave": ("attn", {"pattern": "window"}),
    "QK-norm": ("attn", {"score": "qknorm"}),
    "attention sink / output gate": ("attn", {"out": "gate"}),
    "NoPE in every fourth layer": ("pos", {"nope": True}),
    "Muon for hidden matrices": ("opt", {"opt": "muon"}),
    "learned sparse attention (indexer)": ("attn", {"pattern": "sparse"}),
    "hybrid linear attention, 3:1": ("attn", {"recurrent": True}),
    "replaced residual (HC, mHC, AttnRes)": ("add", {"add": "replaced"}),
    "per-layer and n-gram embeddings": ("emb", {"extra": True}),
}
_YEAR = re.compile(r"\b((?:19|20)\d{2})\b")
_RECURRENT = [("DeltaNet", "Gated DeltaNet"), ("Delta Attention", "Kimi Delta Attention"), ("KDA", "Kimi Delta Attention"),
              ("Mamba", "Mamba-2"), ("SSM", "a state-space layer"), ("short conv", "a short convolution")]


# ------------------------------------------------------------------------------------------------ a model's cells as a block
def features(cells: dict[str, str]) -> dict:
    """The state of the block's parts, read from a row's cells (the texts as the dossier wrote them)."""
    norm, act, pos, attn, emb, moe = (cells[k] for k in ("norm", "act", "pos", "attn", "emb", "moe"))
    f: dict = {}
    # normalization: its kind, its placement, and what the cell says about the residual connection
    f["norm"] = None if "not visible" in norm else "rmsnorm" if "RMSNorm" in norm else "layernorm" if "LayerNorm" in norm else None
    head = norm.split(";")[0]
    f["place"] = None if f["norm"] is None else "pre+post" if re.search(r"pre \+ post|sandwich", head) else \
        "post" if re.search(r"\bpost\b", head) else "pre" if re.search(r"\bpre\b", head) else None
    m = re.search(r"residual replaced by mHC|Attention Residuals \(AttnRes\)|iHC hyper-connections|'Gated Residual' on widened residual streams", norm)
    f["add"], f["add_text"] = ("replaced", m.group(0)) if m else ("identity", None)
    # the MLP's activation
    f["act"] = None if "not visible" in act else "swiglu" if "SwiGLU" in act else "geglu" if "GeGLU" in act else "gated" if "SiTU" in act else \
        "relu2" if re.search(r"squared ReLU|ReLU²", act) else "gelu" if "GELU" in act else "relu" if "ReLU" in act else None
    # position
    if re.search(r"not visible|none visible|rope_theta: null", pos):
        f["pos"] = None
    else:
        f["pos"] = "sinusoidal" if "sinusoidal" in pos else "learned" if "learned" in pos else "rope" if re.search(r"RoPE|rotary", pos) else None
    f["nope"] = "NoPE" in pos
    f["partial"] = bool(re.search(r"partial|p-RoPE|% of dims", pos))
    # attention: how keys and values are kept, which keys a layer sees, what stands at the output, the score
    body = re.sub(r"\([^)]*:[^)]*\)", "", attn)        # a bracket with a colon speaks of another size or version of the model, or of what was removed
    f["heads"] = "compressed" if re.search(r"CSA|HCA|compressed KV", body) else "latent" if "MLA" in body else \
        "gqa" if re.search(r"GQA|MQA", body) else "mha" if "MHA" in body else None
    m = re.search(r"(?:GQA|MHA) (\d+)q?/(\d+)(?:kv)?", body)
    n = re.search(r"\((\d+) heads\)", body)
    f["q"], f["kv"] = (int(m.group(1)), int(m.group(2))) if m else (int(n.group(1)), int(n.group(1)) if f["heads"] == "mha" else None) if n else (None, None)
    f["pattern"] = "sparse" if re.search(r"[Ss]parse|indexer|indexed|DSA", body) else \
        "window" if re.search(r"local|window|chunk|SWA|banded", body) or "chunked local attn" in pos else "full"
    f["out"] = "sink" if "sink" in body else "gate" if re.search(r"gated (?:attention|MLA|DSA)", body, re.I) else "plain"
    # the score: query-key normalization is written in the normalization column, a cap on the logits and a plain scaled
    # dot product in the attention column; a row that writes none is null (see build for the rows older than any other score)
    f["score"] = "qknorm" if "QK-norm" in norm or "cos(q, k)" in attn else "softcap" if re.search(r"soft-capping \(attn", attn) else \
        "dot" if "q·k" in attn else None
    f["recurrent"] = None
    for word, label in _RECURRENT:
        if word in body:
            clause = re.sub(r"^[^:]*hybrid[^:]*:\s*", "", body.split(";")[0])
            parts = re.split(r" \+ | : ", clause)
            counts = [int(c.group(1)) if (c := re.match(r"\s*(\d+)\b", p)) else None for p in parts]
            mine = next((n for p, n in zip(parts, counts) if word in p), None)
            f["recurrent"] = {"kind": label, "n": mine, "of": sum(counts) if all(n is not None for n in counts) else None}
            break
    # embeddings and unembedding; dense or experts
    f["tied"] = not emb.strip().startswith("untied")
    f["moe"] = None
    if "MoE" in moe:
        e = re.search(r"MoE[, ]+(\d+)", moe)
        a = re.search(r"top-(\d+)|(\d+) active", moe)
        s = re.search(r"\+ ?(\d+) shared", moe)
        f["moe"] = {"experts": int(e.group(1)) if e else None, "active": int(a.group(1) or a.group(2)) if a else None,
                    "shared": int(s.group(1)) if s else 1 if "(+shared)" in moe else 0 if "no shared" in moe else None,
                    "also_dense": moe.strip().startswith("dense")}
    f["mtp"] = "MTP" in moe
    return f


# ------------------------------------------------------------------------------------------------ the timeline as events
def events(timeline: dict[str, dict]) -> list[dict]:
    out = []
    for choice, (part, state) in TIMELINE.items():
        if choice not in timeline:
            raise ValueError(f"the dossier's timeline has no row {choice!r}")
        row = timeline[choice]
        default = row["default"]
        if re.search(r"not a default", default):
            scope, qualifier = "none", ""
        elif re.fullmatch(r"\d{4}(?: to \d{4})?(?: \([^)]*\))?", default):
            scope, qualifier = "all", ""
        else:
            scope = "some"
            qualifier = re.sub(r"\s*\([^)]*\)\s*$", "", re.sub(r"^\d{4}(?: to \d{4})?,?\s*", "", default)).strip()
        years_default, years_first = _YEAR.findall(default), _YEAR.findall(row["first"])
        first_year = int(min(years_first)) if years_first else None
        year = int(years_default[0]) if years_default else first_year
        if year is None:
            raise ValueError(f"{choice}: no year")
        out.append({"choice": choice, "part": part, "set": state, "scope": scope, "year": year, "first_year": first_year,
                    "first": row["first"], "default": default, "replaced": row["replaced"], "qualifier": qualifier})
    extra = sorted(set(timeline) - set(TIMELINE))
    if extra:
        raise ValueError(f"rows of the dossier's timeline this script does not know: {extra}")
    return out


def blocks(models: list[dict], evs: list[dict]) -> dict:
    """The block of each year: 2017 from the table's first row, GPT-2's cells in 2019, then the timeline's defaults."""
    first, second = models[0], models[1]
    if first["date"][0] != 2017 or second["date"][0] != 2019:
        raise ValueError("the table no longer begins with the transformer of 2017 and GPT-2")
    f = dict(first["f"])
    f.update(bias=None, base=None, vocab=None, routing=None)
    since = {k: {"year": 2017, "who": first["name"]} for k in f}
    some: list[int] = []
    out = {}
    for year in YEARS:
        new = []
        if year == 2019:                                   # GPT-2: what its row says and the 2017 row did not
            for k, v in second["f"].items():
                if f.get(k) != v:
                    f[k] = v
                    since[k] = {"year": year, "who": second["name"]}
        for i, e in enumerate(evs):
            if e["year"] != year:
                continue
            if e["scope"] == "all":
                for k, v in e["set"].items():
                    f[k] = v
                    since[k] = {"year": year, "who": e["choice"]}
                new.append(i)
            elif e["scope"] == "some":
                some.append(i)
        # tying is not a row of the timeline: a year's block is tied or untied when all the models the census has of that
        # year agree, and neither (null) when they do not; `tie` counts them
        mine = [m for m in models if m["date"][0] == year and not m["own"]]
        untied = sum(not m["f"]["tied"] for m in mine)
        if mine:
            tied = True if untied == 0 else False if untied == len(mine) else None
            if tied != f["tied"]:
                f["tied"] = tied
                since["tied"] = {"year": year, "who": f"the {len(mine)} models of {year}" if len(mine) > 1 else mine[0]["name"]}
        out[str(year)] = {"f": dict(f), "since": {k: dict(v) for k, v in since.items()}, "new": new, "some": list(some),
                          "tie": {"untied": untied, "of": len(mine)},
                          "notes": [i for i, e in enumerate(evs) if e["scope"] == "none" and e["year"] == year],
                          "first": [i for i, e in enumerate(evs) if e["first_year"] == year and e["year"] > year],
                          "rows": [m["key"] for m in models if m["date"][0] == year and not m["own"]]}
    return out


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def build(dossier: Path) -> dict:
    d = read_dossier(dossier)
    rows = sorted(d["rows"], key=lambda r: (r["date"][0], r["date"][1], r["order"])) + own_rows()
    by_name = {r["model"]: r for r in rows}
    models = []
    for r in rows:
        cells = {k: r["cells"][k]["text"] for k in COLUMNS}
        for k, text in list(cells.items()):               # a cell that only refers to another row ("as V3") is read as that row's
            m = re.match(r"as (V[\d.]+)\b", text)
            if m:
                other = [o for o in by_name if o is not r["model"] and re.search(r"\b" + re.escape(m.group(1)) + r"\b", o)
                         and r["model"].split("-")[0] == o.split("-")[0] and not re.match(r"as ", by_name[o]["cells"][k]["text"])]
                if len(other) != 1:
                    raise ValueError(f"{r['model']}: cannot resolve {text!r}")
                cells[k] = f"{text}: {by_name[other[0]]['cells'][k]['text']}"
        models.append({"key": _key(r["model"]), "name": r["model"], "released": r["released"], "date": r["date"], "own": r["own"],
                       "vocabulary": r["vocabulary"], "context": r["context"], "source": r["source"] or shown_source(""),
                       "cells": cells, "f": features(cells)})
    if len({m["key"] for m in models}) != len(models):
        raise ValueError("two models share a key")
    evs = events(d["timeline"])
    # a model released before the timeline's first other score (query-key normalization, by its "first seen" cell) scores by
    # the scaled dot product of the transformer: there was nothing else to choose
    other = next(e["first_year"] for e in evs if e["set"].get("score"))
    for m in models:
        if m["f"]["score"] is None and m["date"][0] < other:
            m["f"]["score"] = "dot"
    return {"generated_by": "gptpp.export.block_through_the_years",
            "source": {"dossier": dossier.name, "section": "2. What the open models use", "rows": len(d["rows"]),
                       "timeline": len(evs), "read": "14 November 2026"},
            "years": YEARS, "parts": [{"key": k, "label": lab, "column": col} for k, lab, col in PARTS],
            "models": models, "events": evs, "blocks": blocks(models, evs)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dossier", default=str(WORK / "research" / "frontier-2026.md"))
    ap.add_argument("--out", default=str(DATA))
    ap.add_argument("--show", action="store_true", help="print every model's block and every year's")
    args = ap.parse_args()
    data = build(Path(args.dossier))
    size = write_json(Path(args.out) / f"{NAME}.json", data)
    print(f"{NAME}.json: {len(data['models'])} models, {len(data['events'])} timeline rows, {size:,} bytes")
    for y in data["years"]:
        b = data["blocks"][str(y)]
        ev = data["events"]
        print(f"  {y}  block changes: {[ev[i]['choice'] for i in b['new']] or '-'}  variants so far: {len(b['some'])}  "
              f"notes: {[ev[i]['choice'] for i in b['notes']] or '-'}  first seen: {[ev[i]['choice'] for i in b['first']] or '-'}  models: {len(b['rows'])}")
    if args.show:
        for m in data["models"]:
            f = m["f"]
            rec = f["recurrent"]
            print(f"{m['released']} {m['name'][:28]:28} {str(f['norm']):9} {str(f['place']):8} {str(f['act']):6} {str(f['pos']):10} "
                  f"{str(f['heads']):10} {str(f['q']):>3}/{str(f['kv']):<3} {f['pattern']:6} {f['out']:5} {str(f['score']):7} {f['add']:8} "
                  f"{'tied' if f['tied'] else 'untied':6} {'moe ' + str(f['moe']['experts']) + '/' + str(f['moe']['active']) if f['moe'] else 'dense':12} "
                  f"{(rec['kind'] + ' ' + str(rec['n']) + '/' + str(rec['of'])) if rec else ''}{' nope' if f['nope'] else ''}{' partial' if f['partial'] else ''}{' mtp' if f['mtp'] else ''}")
        for y in data["years"]:
            print(y, data["blocks"][str(y)]["f"])


if __name__ == "__main__":
    main()
