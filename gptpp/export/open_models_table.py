"""What the open models use: the table of open model families by design choice, as data for the site's figure
`gpt-what-open-models-use` (part 11), with the two models of the series added as rows.

    python -m gptpp.export.open_models_table [--dossier <frontier-2026.md>] [--out <dir>]

Nothing is measured here. The table is the one of the research dossier `frontier-2026.md` (section 2, "What the open
models use"): 36 rows, each cell read from a model's published configuration, card or report, with the row's source.
This script reads that Markdown table, keeps every cell's text as written, and adds to each cell a class (which of a
few named choices it is) by the rules below, so that the figure can draw a column as bands. It also copies the
dossier's own counts (its table "Counts over the 24 rows released from January 2025 on") and its timeline of when
each choice became the default, and it appends two rows built from this repository's presets: the 2019 recipe
(`gpt2-2019-base`) and gpt-pp (`gptpp-base`).

The file (`what-open-models-use.json`), field by field:

    generated_by   "gptpp.export.open_models_table"
    source         {dossier, section, rows (of the dossier's table), read}
    since          [2025, 1]: the month from which the counts run
    columns        the seven design choices, in the order drawn. Each:
      key            "norm" | "act" | "pos" | "attn" | "emb" | "moe" | "opt"
      label, short   the column's name, and its name in four letters for a narrow screen
      classes        [{key, label, tone, timeline}]: the choices a cell can be. tone 0: the design of 2017 to 2019;
                     tone 1: what replaced it; tone 2: a later departure; tone -1: not visible in the sources.
                     timeline: null, or the dossier's timeline row that dates the choice, {choice, first, default
                     (when it became the default in open models), replaced}; clauses the dossier marks "(memory)"
                     (not checked against a source) are left out
      newer          the class keys counted in the column's count line (the tone 1 choices)
      newer_label    how the count line names them ("RMSNorm", "a gated MLP", ...)
    rows           the models in the order of release (year, month; the table's order within a month), then the
                   series' two. Each:
      model          the name as the dossier writes it
      released       "YYYY-MM", and date: [year, month]
      since          true when released in or after `since`; the series' two rows are never counted
      own            null | "gpt2-2019" | "gptpp"
      vocabulary, context    as written in the dossier (strings)
      source         the row's source as written, without the dossier's own cross-references
      cells          {column key: {text (as written), cls (a class key of the column), short (at most 20 characters,
                     for the cell), doubt (true when the dossier says the cell was not re-checked)}}
    counts         {rows: how many rows are counted, by_column: {column: {class: count}}, newer: {column: count}}
    dossier_counts the dossier's own count table: [{feature, rows, of}]
    checks         [{feature, dossier, here, equal, note}]: the dossier's counts beside the same counts made from the
                   classes of this file, where a class says the same thing as a feature

The classes are rules over the cell's words (the functions `_norm`, `_act`, ... below), not judgments: a cell that
does not name a choice is "not visible".
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from .common import DATA, WORK, write_json

NAME = "what-open-models-use"
SINCE = (2025, 1)

# The dossier's table columns, by a word of their header, in the order of the table.
HEADERS = [("model", "model"), ("released", "released"), ("norm", "normalization"), ("act", "MLP activation"),
           ("pos", "position"), ("attn", "attention"), ("emb", "embeddings"), ("vocabulary", "vocabulary"),
           ("moe", "dense or MoE"), ("context", "context"), ("opt", "optimizer"), ("source", "source")]

COLUMNS = [
    # classes: (key, label, tone, the dossier's timeline row that dates the choice or None)
    dict(key="norm", label="normalization", short="norm", newer=["rmsnorm"], newer_label="RMSNorm",
         classes=[("layernorm", "LayerNorm", 0, None), ("rmsnorm", "RMSNorm", 1, "RMSNorm"),
                  ("unknown", "not visible", -1, None)]),
    dict(key="act", label="MLP activation", short="act", newer=["swiglu", "geglu", "gated"], newer_label="a gated MLP",
         classes=[("relu", "ReLU", 0, None), ("gelu", "GELU", 0, None), ("swiglu", "SwiGLU", 1, "SwiGLU / GeGLU"),
                  ("geglu", "GeGLU", 1, "SwiGLU / GeGLU"), ("gated", "another gated unit", 1, "SwiGLU / GeGLU"),
                  ("relu2", "squared ReLU", 2, None), ("unknown", "not visible", -1, None)]),
    dict(key="pos", label="position", short="pos", newer=["rope"], newer_label="rotary positions",
         classes=[("sinusoidal", "sinusoidal, absolute", 0, None), ("learned", "learned, absolute", 0, None),
                  ("rope", "rotary (RoPE)", 1, "RoPE"), ("unknown", "not visible", -1, None)]),
    dict(key="attn", label="attention", short="attn", newer=["grouped", "latent"], newer_label="shared keys and values",
         classes=[("mha", "full multi-head", 0, None), ("grouped", "grouped-query", 1, "grouped-query attention"),
                  ("latent", "latent or compressed keys and values", 1, "multi-head latent attention"),
                  ("hybrid", "hybrid: recurrent or convolution layers beside attention", 2, "hybrid linear attention, 3:1")]),
    dict(key="emb", label="embeddings", short="emb", newer=["untied"], newer_label="untied embeddings",
         classes=[("tied", "tied", 0, None), ("untied", "untied", 1, None)]),
    dict(key="moe", label="dense or experts", short="MoE", newer=["moe"], newer_label="a mixture of experts",
         classes=[("dense", "dense", 0, None), ("moe", "mixture of experts", 1, "fine-grained MoE with shared expert")]),
    dict(key="opt", label="optimizer", short="opt", newer=["muon"], newer_label="Muon",
         classes=[("adam", "Adam", 0, None), ("adamw", "AdamW", 0, None), ("muon", "Muon", 1, "Muon for hidden matrices"),
                  ("unknown", "not reported", -1, None)]),
]


# ------------------------------------------------------------------------------------------------ the classes
def _norm(t: str) -> tuple[str, str]:
    if re.search(r"not visible", t):
        return "unknown", "not visible"
    kind = "rmsnorm" if "RMSNorm" in t else "layernorm" if "LayerNorm" in t else "unknown"
    if kind == "unknown":
        return kind, "not visible"
    head = t.split(";")[0]
    place = "pre+post" if re.search(r"pre \+ post|sandwich", head) else "post" if re.search(r"\bpost\b", head) \
        else "pre" if re.search(r"\bpre\b", head, re.I) else ""
    return kind, ("RMSNorm" if kind == "rmsnorm" else "LayerNorm") + (" " + place if place else "")


def _act(t: str) -> tuple[str, str]:
    if "not visible" in t:
        return "unknown", "not visible"
    if "SwiGLU" in t:
        return "swiglu", "SwiGLU" + (", clamped" if "clamp" in t else "")
    if "GeGLU" in t:
        return "geglu", "GeGLU"
    if "SiTU" in t:
        return "gated", "SiTU, gated"
    if re.search(r"squared ReLU|ReLU²", t):
        return "relu2", "ReLU²"
    if "GELU" in t:
        return "gelu", "GELU"
    if "ReLU" in t:
        return "relu", "ReLU"
    return "unknown", "not visible"


def _pos(t: str) -> tuple[str, str]:
    if re.search(r"not visible|none visible|rope_theta: null", t):
        return "unknown", "none visible" if "none visible" in t else "not visible"
    if "sinusoidal" in t:
        return "sinusoidal", "sinusoidal"
    if "learned" in t:
        return "learned", "learned"
    if re.search(r"RoPE|rotary", t):
        if "NoPE" in t:
            return "rope", "RoPE + NoPE layers"
        if re.search(r"partial|p-RoPE|% of dims", t):
            return "rope", "RoPE, partial"
        if "decoupled" in t:
            return "rope", "RoPE, decoupled"
        return "rope", "RoPE"
    return "unknown", "not visible"


def _attn(t: str) -> tuple[str, str]:
    body = re.sub(r"\([^)]*removed[^)]*\)", "", t)                      # "(sliding window of v0.1 removed: ...)"
    extra = " + sparse" if re.search(r"[Ss]parse|indexer|indexed|DSA", body) else \
        " + window" if re.search(r"local|window|chunk|SWA|banded", body) else ""
    if re.search(r"DeltaNet|KDA|Delta Attention|SSM|Mamba|conv\b|convolution|linear", body):
        return "hybrid", "hybrid"
    if re.search(r"MLA|compressed", body):
        return "latent", ("MLA" if "MLA" in body else "compressed") + extra
    m = re.search(r"GQA (\d+)q?/(\d+)(?:kv)?", body)
    if m or re.search(r"GQA|MQA|share K and V", body):
        return "grouped", ("GQA " + m.group(1) + "/" + m.group(2) if m else "GQA") + extra
    if "MHA" in body:
        return "mha", "MHA" + extra
    return "mha", "MHA"


def _emb(t: str) -> tuple[str, str]:
    return ("untied", "untied") if t.strip().startswith("untied") else ("tied", "tied")


def _moe(t: str) -> tuple[str, str]:
    if "MoE" in t:
        m = re.search(r"MoE,? (\d+)", t)
        return "moe", ("dense, MoE" if t.strip().startswith("dense") else "MoE") + (" " + m.group(1) if m else "")
    return "dense", "dense"


def _opt(t: str) -> tuple[str, str]:
    if t.strip().startswith("not "):
        return "unknown", "not reported"
    if "Muon" in t:
        return "muon", "MuonClip" if "MuonClip" in t else "Muon + AdamW" if "AdamW" in t else "Muon"
    if "AdamW" in t:
        return "adamw", "AdamW"
    if "Adam" in t:
        return "adam", "Adam"
    return "unknown", "not reported"


RULES = {"norm": _norm, "act": _act, "pos": _pos, "attn": _attn, "emb": _emb, "moe": _moe, "opt": _opt}


def classify(key: str, text: str) -> dict:
    cls, short = RULES[key](text)
    doubt = bool(re.search(r"not re-?checked|\(memory\)|not verified", text)) and cls != "unknown"
    return {"text": text, "cls": cls, "short": short + ("?" if doubt else ""), "doubt": doubt}


# ------------------------------------------------------------------------------------------------ the dossier
def _tables(lines: list[str]) -> list[tuple[int, list[str], list[list[str]]]]:
    """Every Markdown table: (line of its header, header cells, rows of cells)."""
    out, i = [], 0
    while i < len(lines) - 1:
        if lines[i].startswith("|") and re.match(r"^\|\s*:?-", lines[i + 1]):
            header = [c.strip() for c in lines[i].strip().strip("|").split("|")]
            rows, j = [], i + 2
            while j < len(lines) and lines[j].startswith("|"):
                rows.append([c.strip() for c in lines[j].strip().strip("|").split("|")])
                j += 1
            out.append((i, header, rows))
            i = j
        else:
            i += 1
    return out


def shown_source(text: str) -> str:
    """The row's source as a reader can use it: the dossier's own cross-references (its numbered passages "(P3)",
    "the optimizers dossier") are taken out; a report the dossier cites through another dossier keeps its name."""
    parts = []
    for part in text.split(";"):
        part = re.sub(r"\s*\(P\d+[^)]*\)", "", part).strip()
        m = re.search(r"dossier(?: \(([^)]*)\))?", part)
        if m:
            part = (m.group(1) or "").strip()
        if part:
            parts.append(part)
    return "; ".join(parts)


def read_dossier(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8").split("\n")
    tables = _tables(lines)
    models = next(t for t in tables if t[1][0] == "model" and t[1][1].startswith("released"))
    counts = next(t for t in tables if t[1][:2] == ["feature", "rows"])
    timeline = next(t for t in tables if t[1][0] == "choice" and t[1][1] == "first")
    index = {}
    for key, word in HEADERS:
        hits = [k for k, h in enumerate(models[1]) if h.lower().startswith(word.lower())]
        if len(hits) != 1:
            raise ValueError(f"the dossier's table has {len(hits)} columns named like {word!r}")
        index[key] = hits[0]
    rows = []
    for n, cells in enumerate(models[2]):
        if len(cells) != len(models[1]):
            raise ValueError(f"row {n + 1} of the dossier's table has {len(cells)} cells, not {len(models[1])}")
        m = re.match(r"(\d{4})-(\d{2})", cells[index["released"]])
        if not m:
            raise ValueError(f"no date in {cells[index['released']]!r}")
        rows.append({"order": n, "model": cells[index["model"]], "date": [int(m.group(1)), int(m.group(2))],
                     "released": f"{m.group(1)}-{m.group(2)}", "released_text": cells[index["released"]], "own": None,
                     "vocabulary": cells[index["vocabulary"]], "context": cells[index["context"]],
                     "source": shown_source(cells[index["source"]]),
                     "cells": {c["key"]: classify(c["key"], cells[index[c["key"]]]) for c in COLUMNS}})
    # a cell that only refers to another row ("as V3") takes that row's class and keeps its own words
    for r in rows:
        for key, cell in r["cells"].items():
            m = re.match(r"as (V[\d.]+)\b", cell["text"])
            if m:
                other = [o for o in rows if o is not r and re.search(r"\b" + re.escape(m.group(1)) + r"\b", o["model"])
                         and r["model"].split("-")[0] == o["model"].split("-")[0]]
                if len(other) != 1:
                    raise ValueError(f"{r['model']}: cannot resolve {cell['text']!r}")
                cell.update(cls=other[0]["cells"][key]["cls"], short=other[0]["cells"][key]["short"].split(" ")[0])
    dossier_counts = []
    for cells in counts[2]:
        m = re.match(r"(\d+) of (\d+)", cells[1])
        dossier_counts.append({"feature": cells[0], "rows": int(m.group(1)), "of": int(m.group(2))})
    tl = {}
    for cells in timeline[2]:
        first = "; ".join(s.strip() for s in cells[1].split(";") if "(memory)" not in s)
        clean = lambda t: re.sub(r"\s*\((?:[^()]*\bmemory|`[^`]*`)\)", "", t).strip()
        tl[cells[0]] = {"choice": cells[0], "first": clean(first), "default": clean(cells[2]), "replaced": clean(cells[3])}
    return {"rows": rows, "dossier_counts": dossier_counts, "timeline": tl}


# ------------------------------------------------------------------------------------------------ the series' rows
def own_rows() -> list[dict]:
    """The 2019 recipe and gpt-pp, written from this repository's presets in the words of the dossier's table."""
    from ..config import preset, train_preset

    def thousands(n: int) -> str:
        return f"{n:,}"

    out = []
    for own, name, label, source in (
            ("gpt2-2019", "gpt2-2019-base", "the 2019 recipe, trained here",
             "this series: GPT-2's design at the shape and on the data of gpt-pp (Section 7)"),
            ("gptpp", "gptpp-base", "gpt-pp", "this series: the recipe and model card of Section 11")):
        m, t = preset(name), train_preset(name)
        norm = {"layernorm": "LayerNorm" + (" with gain and bias" if m.bias else " with gain"),
                "rmsnorm": "RMSNorm with a gain", "rmsnorm_fixed": "RMSNorm without parameters"}[m.norm]
        norm += f", {m.norm_placement} + final norm" + ("; also after the embedding" if m.embed_norm else "") \
            + ("; QK-norm" if m.qk_norm else "")
        act = {"gelu": "GELU", "relu2": "squared ReLU (ReLU²)", "relu": "ReLU"}[m.act]
        pos = {"learned": "learned absolute", "rope": f"RoPE θ={m.rope_base:.0e}".replace("e+0", "e")}[m.pos]
        attn = f"full MHA ({m.n_head} heads)" if m.kv_heads == m.n_head else f"GQA {m.n_head}/{m.kv_heads}"
        attn += f"; scores {m.qk_scale:g} · cos(q, k)" if m.qk_norm else "; scores q·k / sqrt(d_head)"
        attn += f"; final logits soft-capped at {m.logit_softcap:g}" if m.logit_softcap else ""
        emb = "tied" if m.tie_embeddings else "untied"
        if t.optimizer == "muon+adamw":
            opt = f"Muon on the hidden matrices, AdamW ({t.beta1:g}, {t.beta2:g}) on the embeddings; no weight decay"
        else:
            opt = f"AdamW ({t.beta1:g}, {t.beta2:g}), wd {t.weight_decay:g}, warmup and cosine decay"
        texts = {"norm": norm, "act": act, "pos": pos, "attn": attn, "emb": emb, "moe": "dense", "opt": opt}
        out.append({"order": 1000 + len(out), "model": label, "date": [2026, 11], "released": "2026-11",
                    "released_text": "2026-11 (this series)", "own": own, "vocabulary": thousands(m.vocab_size),
                    "context": thousands(m.block_size), "source": source,
                    "cells": {c["key"]: classify(c["key"], texts[c["key"]]) for c in COLUMNS}})
    return out


# ------------------------------------------------------------------------------------------------ the file
def build(dossier: Path) -> dict:
    d = read_dossier(dossier)
    rows = sorted(d["rows"], key=lambda r: (r["date"][0], r["date"][1], r["order"])) + own_rows()
    for r in rows:
        r["since"] = r["own"] is None and tuple(r["date"]) >= SINCE
        del r["order"]
    counted = [r for r in rows if r["since"]]
    by_column, newer = {}, {}
    columns = []
    for c in COLUMNS:
        keys = [k[0] for k in c["classes"]]
        for r in rows:
            if r["cells"][c["key"]]["cls"] not in keys:
                raise ValueError(f"{r['model']}: class {r['cells'][c['key']]['cls']!r} is not a class of {c['key']}")
        by_column[c["key"]] = {k: sum(r["cells"][c["key"]]["cls"] == k for r in counted) for k in keys}
        newer[c["key"]] = sum(by_column[c["key"]][k] for k in c["newer"])
        missing = [t for _, _, _, t in c["classes"] if t and t not in d["timeline"]]
        if missing:
            raise ValueError(f"the dossier's timeline has no row {missing}")
        columns.append({"key": c["key"], "label": c["label"], "short": c["short"],
                        "classes": [{"key": k, "label": lab, "tone": tone, "timeline": d["timeline"][t] if t else None}
                                    for k, lab, tone, t in c["classes"]],
                        "newer": c["newer"], "newer_label": c["newer_label"]})
    # the dossier's counts beside ours, where a class says what a feature says
    same = {"RMSNorm": newer["norm"], "gated MLP (SwiGLU or GeGLU)": newer["act"],
            "rotary position embedding in some layers": newer["pos"], "untied embeddings": newer["emb"],
            "mixture of experts": newer["moe"],
            "hybrid (linear-recurrent, SSM or convolution layers mixed with attention)": by_column["attn"]["hybrid"],
            "Muon reported": newer["opt"]}
    named = sum("MoE" in r["cells"]["moe"]["text"] for r in counted)
    checks = []
    for f in d["dossier_counts"]:
        if f["feature"] in same:
            here = same[f["feature"]]
            note = ""
            if here != f["rows"] and f["feature"] == "mixture of experts":
                refs = [r["model"] for r in counted if r["cells"]["moe"]["cls"] == "moe" and "MoE" not in r["cells"]["moe"]["text"]]
                note = (f"{named} cells of the column say MoE; " + ", ".join(refs) +
                        " refers to an earlier row that is a mixture of experts and is counted here")
            checks.append({"feature": f["feature"], "dossier": f["rows"], "here": here, "equal": here == f["rows"], "note": note})
    return {"generated_by": "gptpp.export.open_models_table",
            "source": {"dossier": dossier.name, "section": "2. What the open models use", "rows": len(d["rows"]),
                       "read": "14 November 2026"},
            "since": list(SINCE), "columns": columns, "rows": rows,
            "counts": {"rows": len(counted), "by_column": by_column, "newer": newer},
            "dossier_counts": d["dossier_counts"], "checks": checks}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dossier", default=str(WORK / "research" / "frontier-2026.md"))
    ap.add_argument("--out", default=str(DATA))
    ap.add_argument("--show", action="store_true", help="print every row's classes")
    args = ap.parse_args()
    data = build(Path(args.dossier))
    size = write_json(Path(args.out) / f"{NAME}.json", data)
    print(f"{NAME}.json: {len(data['rows'])} rows ({data['source']['rows']} of the dossier, {data['counts']['rows']} "
          f"counted since {SINCE[0]}), {size:,} bytes")
    for c in data["columns"]:
        print(f"  {c['key']:5} {c['newer_label']:24} {data['counts']['newer'][c['key']]:2} of {data['counts']['rows']}   "
              + "  ".join(f"{k} {v}" for k, v in data["counts"]["by_column"][c["key"]].items() if v))
    for ch in data["checks"]:
        print(f"  check  {ch['feature'][:52]:52} dossier {ch['dossier']:2}  here {ch['here']:2}  {'' if ch['equal'] else 'DIFFERENT: ' + ch['note']}")
    if args.show:
        for r in data["rows"]:
            print(f"{r['released']} {r['model'][:30]:30} " + " | ".join(f"{r['cells'][c['key']]['short']:18}" for c in data["columns"]))


if __name__ == "__main__":
    main()
