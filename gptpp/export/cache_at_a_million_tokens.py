"""Where the memory goes when the context grows: the layers of nine models by what each keeps for the tokens it has
read, as data for the site's figure `gpt-cache-at-a-million-tokens` (part 11).

    python -m gptpp.export.cache_at_a_million_tokens [--dossier <frontier-2026.md>] [--configs <folder>] [--out <dir>]

Nothing is measured here. Every number is read from a model's published configuration file, in the copy the research
dossier `frontier-2026.md` was written from (its folder `_scratch/frontier-2026/hfconfig`, fetched from the Hugging
Face Hub on 14 November 2026); the release month and the context come from the dossier's table (section 2), and the
script checks that the context of the table is the one in the configuration. gpt-pp's row is built from this
repository's preset. The figure computes bytes from these numbers in the page; this script computes none.

What a layer keeps while a model decodes, by the kind of the layer (the dossier's statements 5.2, 5.3 and 5.6):

    full        every token: 2 g d numbers a token for g key-value heads of width d; g d when keys and values are one
                tensor; d_c + d_R when the cache is a latent of d_c numbers and one rotary key of d_R (latent attention)
    window      the last w tokens only: 2 g d numbers for each
    recurrent   one matrix a head whatever the context: heads x d_v x d_k numbers (the state S of the gated delta rule)
    compressed  one entry for every r tokens. The configuration gives r for every layer and not what an entry holds,
                so such a model is listed with its layers and without bytes ("priced": false)

The file (`cache-at-a-million-tokens.json`), field by field:

    generated_by      "gptpp.export.cache_at_a_million_tokens"
    source            {dossier, table, statements, configs, read}
    bytes_per_number  2: the cache is counted in 16-bit floats, as the series counts it (Section 3.7)
    contexts          the stops of the figure's slider: 1,024 to 1,048,576 tokens, doubling
    kinds             {full, window, recurrent, compressed}: {label, keeps}
    models            in the order of release, gpt-pp last. Each:
      key, name       an identifier, and the name as the dossier writes it
      released        "YYYY-MM" (the dossier's table)
      context         tokens (`max_position_embeddings` of the configuration; block_size for gpt-pp)
      repo            the repository on the Hub whose config.json was read; null for gpt-pp
      layers          the number of layers
      pattern         one character a layer, in order: the index of its group in `groups`
      groups          the kinds of layer the model has. Each:
        kind            "full" | "window" | "recurrent" | "compressed"
        count           how many layers
        per_token       numbers kept for one token (full, window); null otherwise
        window          tokens kept (window); null otherwise
        state           numbers kept whatever the context (recurrent); null otherwise
        ratio           tokens to one entry (compressed); null otherwise
        what            the layer in words, with its numbers
        formula         the arithmetic, as text
        fields          {configuration field: value}: what was read, and from where
      priced          false when a group's bytes cannot be had from the configuration
      note            null, or what a reader must know about the row
      own             true for gpt-pp
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from .common import DATA, WORK, write_json

NAME = "cache-at-a-million-tokens"
CONTEXTS = [2 ** e for e in range(10, 21)]
KINDS = {
    "full": {"label": "full attention", "keeps": "every token"},
    "window": {"label": "sliding window", "keeps": "the last tokens only"},
    "recurrent": {"label": "recurrent", "keeps": "one matrix a head, whatever the context"},
    "compressed": {"label": "compressed", "keeps": "one entry for a block of tokens"},
}


def _group(kind: str, count: int, what: str, formula: str, fields: dict, per_token=None, window=None, state=None, ratio=None) -> dict:
    return {"kind": kind, "count": count, "per_token": per_token, "window": window, "state": state, "ratio": ratio,
            "what": what, "formula": formula, "fields": fields}


def _pattern(kinds: list[str], order: list[str]) -> str:
    """One character a layer: the index of the layer's kind in `order`."""
    return "".join(str(order.index(k)) for k in kinds)


# ------------------------------------------------------------------------------------------------ one recipe a model
def uniform_gqa(c: dict) -> tuple[list[dict], str, str | None]:
    """Every layer attends to the whole context; g key-value heads of width hidden_size / heads."""
    g, d, n = c["num_key_value_heads"], c["hidden_size"] // c["num_attention_heads"], c["num_hidden_layers"]
    fields = {"num_hidden_layers": n, "num_key_value_heads": g, "hidden_size": c["hidden_size"], "num_attention_heads": c["num_attention_heads"]}
    return [_group("full", n, f"{g} key-value heads of width {d}", f"2 × {g} × {d}", fields, per_token=2 * g * d)], "0" * n, None


def explicit_head_dim(c: dict) -> tuple[list[dict], str, str | None]:
    """Every layer attends to the whole context (`attn_type_list` all 1); the head width is in the file."""
    kinds = Counter(c["attn_type_list"])
    if set(kinds) != {1} or len(c["attn_type_list"]) != c["num_hidden_layers"]:
        raise ValueError("attn_type_list is not all 1")
    g, d, n = c["num_key_value_heads"], c["head_dim"], c["num_hidden_layers"]
    fields = {"num_hidden_layers": n, "attn_type_list": "1 for every layer", "num_key_value_heads": g, "head_dim": d}
    return [_group("full", n, f"{g} key-value heads of width {d}", f"2 × {g} × {d}", fields, per_token=2 * g * d)], "0" * n, None


def latent(c: dict) -> tuple[list[dict], str, str | None]:
    """Latent attention in every layer: one latent and one rotary key a token, for all heads."""
    dc, dr, n = c["kv_lora_rank"], c["qk_rope_head_dim"], c["num_hidden_layers"]
    fields = {"num_hidden_layers": n, "kv_lora_rank": dc, "qk_rope_head_dim": dr}
    return [_group("full", n, f"a latent of {dc} numbers and a rotary key of {dr}", f"{dc} + {dr}", fields, per_token=dc + dr)], "0" * n, None


def window_and_full(c: dict) -> tuple[list[dict], str, str | None]:
    """`layer_types` names each layer sliding_attention or full_attention; both share the key-value heads."""
    types, g, d, w = c["layer_types"], c["num_key_value_heads"], c["head_dim"], c["sliding_window"]
    count = Counter(types)
    if set(count) != {"sliding_attention", "full_attention"} or len(types) != c["num_hidden_layers"]:
        raise ValueError(f"unexpected layer_types {dict(count)}")
    fields = {"layer_types": dict(count), "num_key_value_heads": g, "head_dim": d}
    groups = [_group("window", count["sliding_attention"], f"the last {w} tokens, {g} key-value heads of width {d}", f"2 × {g} × {d}",
                     dict(fields, sliding_window=w), per_token=2 * g * d, window=w),
              _group("full", count["full_attention"], f"{g} key-value heads of width {d}", f"2 × {g} × {d}", fields, per_token=2 * g * d)]
    return groups, _pattern(types, ["sliding_attention", "full_attention"]), None


def gemma4(c: dict) -> tuple[list[dict], str, str | None]:
    """Sliding layers with their own key-value heads; global layers with fewer, wider heads whose keys and values are
    one tensor (`attention_k_eq_v`)."""
    types, w = c["layer_types"], c["sliding_window"]
    count = Counter(types)
    if set(count) != {"sliding_attention", "full_attention"} or len(types) != c["num_hidden_layers"] or not c["attention_k_eq_v"]:
        raise ValueError("not the layout this recipe reads")
    g, d, gg, gd = c["num_key_value_heads"], c["head_dim"], c["num_global_key_value_heads"], c["global_head_dim"]
    groups = [_group("window", count["sliding_attention"], f"the last {w:,} tokens, {g} key-value heads of width {d}", f"2 × {g} × {d}",
                     {"layer_types": dict(count), "sliding_window": w, "num_key_value_heads": g, "head_dim": d}, per_token=2 * g * d, window=w),
              _group("full", count["full_attention"], f"{gg} key-value heads of width {gd}, keys and values one tensor", f"{gg} × {gd}",
                     {"layer_types": dict(count), "num_global_key_value_heads": gg, "global_head_dim": gd, "attention_k_eq_v": True}, per_token=gg * gd)]
    return groups, _pattern(types, ["sliding_attention", "full_attention"]), None


def delta_and_full(c: dict) -> tuple[list[dict], str, str | None]:
    """`layer_types` names each layer linear_attention (the gated delta rule) or full_attention."""
    types = c["layer_types"]
    count = Counter(types)
    if set(count) != {"linear_attention", "full_attention"} or len(types) != c["num_hidden_layers"]:
        raise ValueError(f"unexpected layer_types {dict(count)}")
    hv, dv, dk = c["linear_num_value_heads"], c["linear_value_head_dim"], c["linear_key_head_dim"]
    g, d = c["num_key_value_heads"], c["head_dim"]
    groups = [_group("recurrent", count["linear_attention"], f"{hv} states of {dv} × {dk} numbers", f"{hv} × {dv} × {dk}",
                     {"layer_types": dict(count), "linear_num_value_heads": hv, "linear_value_head_dim": dv, "linear_key_head_dim": dk}, state=hv * dv * dk),
              _group("full", count["full_attention"], f"{g} key-value heads of width {d}", f"2 × {g} × {d}",
                     {"layer_types": dict(count), "num_key_value_heads": g, "head_dim": d}, per_token=2 * g * d)]
    return groups, _pattern(types, ["linear_attention", "full_attention"]), None


def kimi(c: dict) -> tuple[list[dict], str, str | None]:
    """`linear_attn_config` lists the recurrent layers and the latent-attention layers by number, from 1."""
    la, n = c["linear_attn_config"], c["num_hidden_layers"]
    kda, full = set(la["kda_layers"]), set(la["full_attn_layers"])
    if kda & full or kda | full != set(range(1, n + 1)):
        raise ValueError("the two lists do not cover the layers once each")
    h, d, dc, dr = la["num_heads"], la["head_dim"], c["kv_lora_rank"], c["qk_rope_head_dim"]
    groups = [_group("recurrent", len(kda), f"{h} states of {d} × {d} numbers", f"{h} × {d} × {d}",
                     {"linear_attn_config.kda_layers": f"{len(kda)} layers", "linear_attn_config.num_heads": h, "linear_attn_config.head_dim": d}, state=h * d * d),
              _group("full", len(full), f"a latent of {dc} numbers and a rotary key of {dr}", f"{dc} + {dr}",
                     {"linear_attn_config.full_attn_layers": f"{len(full)} layers", "kv_lora_rank": dc, "qk_rope_head_dim": dr}, per_token=dc + dr)]
    note = None
    if c.get("mla_use_nope"):
        note = (f"The configuration also sets mla_use_nope; if its attention layers keep no rotary key, a token costs {dc} numbers a layer, "
                f"not {dc + dr}.")
    return groups, "".join("0" if i in kda else "1" for i in range(1, n + 1)), note


def compressed(c: dict) -> tuple[list[dict], str, str | None]:
    """`compress_ratios` gives, for each layer, how many tokens go to one cached entry. What an entry holds is not in
    the file, so the layers are listed and not priced."""
    n, w = c["num_hidden_layers"], c["sliding_window"]
    ratios = c["compress_ratios"][:n]
    count = Counter(ratios)
    if 0 in count or len(ratios) != n:
        raise ValueError("a hidden layer without a ratio")
    order = sorted(count, reverse=True)
    groups = [_group("compressed", count[r], f"one entry for every {r} tokens, and a window of {w}", f"t ÷ {r} entries",
                     {"compress_ratios": {str(k): v for k, v in sorted(count.items())}, "sliding_window": w,
                      "num_key_value_heads": c["num_key_value_heads"], "head_dim": c["head_dim"]}, ratio=r, window=w) for r in order]
    note = ("Every layer compresses its cache: one entry for a block of tokens. The configuration gives the size of the block for each "
            "layer and not what an entry holds, so no bytes are drawn. Its card claims a tenth of the cache of its predecessor at a million tokens.")
    return groups, _pattern(ratios, order), note


MODELS = [
    # key, name in the figure, the start of its row in the dossier's table, the copy of its config, its repository, recipe
    ("llama-3-8b", "Llama 3 8B", "Llama 3 8B", "unsloth_llama-3-8b", "unsloth/llama-3-8b", uniform_gqa),
    ("deepseek-v3", "DeepSeek-V3", "DeepSeek-V3 671B", "deepseek-ai_DeepSeek-V3", "deepseek-ai/DeepSeek-V3", latent),
    ("gpt-oss-120b", "gpt-oss-120b", "gpt-oss-120b", "openai_gpt-oss-120b", "openai/gpt-oss-120b", window_and_full),
    ("minimax-m2", "MiniMax-M2", "MiniMax-M2", "MiniMaxAI_MiniMax-M2", "MiniMaxAI/MiniMax-M2", explicit_head_dim),
    ("qwen3.5-27b", "Qwen3.5-27B", "Qwen3.5 27B", "Qwen_Qwen3.5-27B", "Qwen/Qwen3.5-27B", delta_and_full),
    ("gemma-4-31b", "Gemma 4 31B", "Gemma 4 31B", "unsloth_gemma-4-31B-it", "unsloth/gemma-4-31B-it", gemma4),
    ("deepseek-v4-pro", "DeepSeek-V4-Pro", "DeepSeek-V4-Pro", "deepseek-ai_DeepSeek-V4-Pro", "deepseek-ai/DeepSeek-V4-Pro", compressed),
    ("kimi-k3", "Kimi K3", "Kimi K3", "moonshotai_Kimi-K3", "moonshotai/Kimi-K3", kimi),
]


# ------------------------------------------------------------------------------------------------ the dossier's table
def table_rows(dossier: Path) -> dict[str, dict]:
    """The rows of the dossier's table of models: {model cell: {released: 'YYYY-MM', context: the cell as written}}."""
    lines = dossier.read_text(encoding="utf-8").split("\n")
    head = next(i for i, l in enumerate(lines) if l.startswith("| model | released"))
    names = [c.strip() for c in lines[head].strip().strip("|").split("|")]
    ctx = next(i for i, n in enumerate(names) if n.startswith("context"))
    out = {}
    for line in lines[head + 2:]:
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        m = re.match(r"(\d{4})-(\d{2})", cells[1])
        out[cells[0]] = {"released": f"{m.group(1)}-{m.group(2)}", "context": cells[ctx]}
    return out


def own_row() -> dict:
    """gpt-pp, from this repository's preset: full attention in every layer, a key head and a value head for each
    query head."""
    from ..config import preset

    m = preset("gptpp-base")
    g, d = m.kv_heads, m.d_model // m.n_head
    fields = {"n_layer": m.n_layer, "n_head": m.n_head, "key-value heads": g, "d_model": m.d_model}
    return {"key": "gpt-pp", "name": "gpt-pp", "released": "2026-11", "context": m.block_size, "repo": None, "layers": m.n_layer,
            "pattern": "0" * m.n_layer,
            "groups": [_group("full", m.n_layer, f"{g} key-value heads of width {d}", f"2 × {g} × {d}", fields, per_token=2 * g * d)],
            "priced": True, "note": None, "own": True}


def build(dossier: Path, configs: Path) -> dict:
    rows = table_rows(dossier)
    models = []
    for key, name, row, file, repo, recipe in MODELS:
        hit = [r for r in rows if r.startswith(row)]
        if len(hit) != 1:
            raise ValueError(f"{len(hit)} rows of the dossier's table begin with {row!r}")
        c = json.loads((configs / f"{file}.json").read_text(encoding="utf-8"))
        c = c.get("text_config", c)
        groups, pattern, note = recipe(c)
        context = c["max_position_embeddings"]
        in_table = [int(n.replace(",", "")) for n in re.findall(r"\d[\d,]*", rows[hit[0]]["context"])]
        if context not in in_table:
            raise ValueError(f"{name}: the configuration's context {context} is not in the dossier's cell {rows[hit[0]]['context']!r}")
        if len(pattern) != c["num_hidden_layers"] or sum(g["count"] for g in groups) != len(pattern):
            raise ValueError(f"{name}: the groups do not add up to its layers")
        models.append({"key": key, "name": name, "released": rows[hit[0]]["released"], "context": context, "repo": repo,
                       "layers": len(pattern), "pattern": pattern, "groups": groups,
                       "priced": all(g["kind"] != "compressed" for g in groups), "note": note, "own": False})
    models.sort(key=lambda m: m["released"])
    models.append(own_row())
    return {"generated_by": "gptpp.export.cache_at_a_million_tokens",
            "source": {"dossier": dossier.name, "table": "2. What the open models use", "statements": "5.2, 5.3, 5.6",
                       "configs": "config.json of each repository, as fetched for the dossier", "read": "14 November 2026"},
            "bytes_per_number": 2, "contexts": CONTEXTS, "kinds": KINDS, "models": models}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dossier", default=str(WORK / "research" / "frontier-2026.md"))
    ap.add_argument("--configs", default=str(WORK / "research" / "_scratch" / "frontier-2026" / "hfconfig"))
    ap.add_argument("--out", default=str(DATA))
    args = ap.parse_args()
    data = build(Path(args.dossier), Path(args.configs))
    size = write_json(Path(args.out) / f"{NAME}.json", data)
    print(f"{NAME}.json: {len(data['models'])} models, {size:,} bytes")
    top = data["contexts"][-1]
    for m in data["models"]:
        kept = 0
        for g in m["groups"]:
            if g["kind"] == "full":
                kept += g["count"] * g["per_token"] * top
            elif g["kind"] == "window":
                kept += g["count"] * g["per_token"] * min(top, g["window"])
            elif g["kind"] == "recurrent":
                kept += g["count"] * g["state"]
        layers = ", ".join(f"{g['count']} {g['kind']}" for g in m["groups"])
        total = f"{kept * data['bytes_per_number'] / 2 ** 30:8.1f} GiB at {top:,} tokens" if m["priced"] else "   not priced"
        print(f"  {m['released']}  {m['name']:16} {m['layers']:3} layers ({layers}); context {m['context']:,}; {total}")


if __name__ == "__main__":
    main()
