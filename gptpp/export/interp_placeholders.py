"""The numbers of the lens, patching and memory analyses that the posts quote, keyed by the posts' own placeholders.

    python -m gptpp.export.interp_placeholders            (after the three exporters; reads what they wrote)

Parts 9 and 5 of the series hold placeholders `@@...@@` for values that only these analyses produce. This script
reads the three exports the figures draw (`logit-lens.json`, `activation-patching.json`, `neurons-as-memories.json`
in the site's data folder), so that a number in the prose is the number in the figure, and writes
`results/interp/lens-patching-memories.json`:

    placeholders   {"<the placeholder's own words>": {"post", "value" and/or "text", "how", ...}}
    more           measured values for the posts' RESULTS notes: the route of the answer and the heads that carry
                   it, noising against denoising, which tokens settle early, the lens by depth on held-out text,
                   the neurons' measures by block
    unfilled       placeholders found in the two posts that concern these analyses and are not filled here

Conventions. In every text blocks, heads and neurons are counted from 1 ("block 9, head 6", "block 3, neuron
1,417"); fields named `*_index` count from 0, as the code and the data do. "After the first layer" is the stream
after block 1 (hook `resid_post.0`). Logit differences and attributions are taken before the model's cap on its
logits; the lens applies the model's whole readout, cap included, so that after the last block it is the model.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from .. import repo_dir
from .common import DATA, SITE, write_json

P9 = "gpt-09"
P5 = "gpt-05"
KEYS = {
    "pair": "the clean prompt, the corrupted prompt and the two answer tokens of the patching example on gpt-pp",
    "ld_clean": "logit difference of gpt-pp on the clean prompt of the patching example",
    "ld_corrupt": "logit difference of gpt-pp on the corrupted prompt of the patching example",
    "passage": "the passage of the lens example on gpt-pp and its length in tokens",
    "settle": "median over positions of the first layer from which the logit lens's top token stays equal to gpt-pp's final prediction",
    "kl_first": "mean relative entropy in bits from gpt-pp's final distribution to the logit lens after the first layer",
    "kl_11": "mean relative entropy in bits from gpt-pp's final distribution to the logit lens after layer 11",
    "lower": "a neuron of a lower layer of gpt-pp, with its layer and index, three of its trigger prefixes and the top tokens of its value vector",
    "upper": "a neuron of an upper layer of gpt-pp, with its layer and index, three of its trigger prefixes and the top tokens of its value vector",
}


def q(text: str) -> str:
    return "“" + text.replace("\n", "↵") + "”"


def patching(d: dict) -> tuple[dict, dict]:
    v, p = d["vocab"], d["pairs"][0]
    tok = [v[i] for i in p["tokens"]]
    S, A = np.array(p["denoise"]["resid_pre"]), np.array(p["denoise"]["attn_out"])
    L, c, e = S.shape[0], p["changed"], len(tok) - 1
    hold = next((l for l in range(L) if S[l, c] < 0.5), L)                       # blocks 1..hold restore at least half from the changed token
    arrive = next((l + 1 for l in range(L) if (S[l:, e] >= 0.5).all()), L + 1)
    heads, attention = np.array(p["heads"]), np.array(p["attention"])
    order = np.dstack(np.unravel_index(np.argsort(-np.abs(heads).ravel()), heads.shape))[0][:5]
    top = [{"head": f"block {l + 1}, head {h + 1}", "block_index": int(l), "head_index": int(h), "direct_effect": float(heads[l, h]),
            "attends_most_to": tok[int(attention[l, h].argmax())], "attention_there": float(attention[l, h].max()),
            "attention_to_start_of_text": float(p["attention_start"][l][h])} for l, h in order]
    noising = {}
    for site in d["sites"]:
        den, noi = np.array(p["denoise"][site]), 1 - np.array(p["noise"][site])
        noising[site] = {"largest_gap": float(np.abs(noi - den).max()), "correlation": float(np.corrcoef(den.ravel(), noi.ravel())[0, 1]),
                         "note": "share destroyed by noising a cell (1 - effect) against share restored by denoising it, over all cells"}
    out = {
        KEYS["pair"]: {"post": P9, "text": f"{q(p['clean'])}, answer {q(v[p['answer']])}, and {q(p['corrupt'])}, answer {q(v[p['answer_corrupt']])}",
                       "clean": p["clean"], "corrupted": p["corrupt"], "answer_clean": v[p["answer"]], "answer_corrupted": v[p["answer_corrupt"]],
                       "tokens": len(tok), "changed_token": tok[c], "changed_position": c + 1, "p_clean": p["p_clean"], "p_corrupted": p["p_corrupt"],
                       "how": "the first pair of the patching export (its kind: " + p["kind"] + "): one token differs, both answers are single tokens, and each is "
                              "the model's top token after its prompt; the model reads <|bos|> before the prompt, which is not counted"},
        KEYS["ld_clean"]: {"post": P9, "value": round(p["ld_clean"], 2), "how": "logit of the clean answer less logit of the corrupted answer at the last position of the "
                           "clean prompt, before the cap on the logits: (W_U[a] - W_U[b]) . N(x_L)"},
        KEYS["ld_corrupt"]: {"post": P9, "value": round(p["ld_corrupt"], 2), "how": "the same difference at the last position of the corrupted prompt"},
    }
    more = {
        "pairs": [{"key": x["key"], "kind": x["kind"], "clean": x["clean"], "corrupted": x["corrupt"], "ld_clean": x["ld_clean"], "ld_corrupted": x["ld_corrupt"]} for x in d["pairs"]],
        "route": {"text": f"restoring the stream entering a block at {q(tok[c])} restores at least half of the logit difference up to block {hold}; at the last position "
                          f"it does from block {arrive} on",
                  "last_block_fed_by_the_changed_token": hold, "first_block_fed_by_the_last_position": arrive,
                  "stream_at_changed_token_by_block": S[:, c].round(3).tolist(), "stream_at_last_position_by_block": S[:, e].round(3).tolist(),
                  "attention_output_at_last_position_by_block": A[:, e].round(3).tolist(),
                  "mlp_output_at_changed_token_by_block": np.array(p["denoise"]["mlp_out"])[:, c].round(3).tolist()},
        "largest_direct_attributions": top,
        "attribution_sums": {"heads": float(heads.sum()), "mlps": float(np.sum(p["mlp"])), "embedding": p["embed"], "all": p["total"],
                             "mlp_by_block": [round(x, 3) for x in p["mlp"]]},
        "noising_against_denoising": noising,
        "not_kept": d.get("missed", []),
    }
    return out, more


def lens(d: dict) -> tuple[dict, dict]:
    v, p, H = d["vocab"], d["passages"][0], d["held_out"]
    tok, L = [v[i] for i in p["tokens"]], d["model"]["layers"]
    kl = np.array(p["kl_bits"])
    out = {
        KEYS["passage"]: {"post": P9, "text": f"{q(p['text'])}, {len(tok)} tokens", "passage": p["text"], "tokens": len(tok), "predicts": v[p["answer"]],
                          "probability": p["answer_p"], "how": "the first passage of the lens export (its kind: " + p["kind"] + "), kept because the model's top token "
                          "after it is the expected one; the model reads <|bos|> before it, which is not counted and not drawn"},
        KEYS["settle"]: {"post": P9, "value": p["settle_median"], "of": L, "by_position": [{"token": t, "predicts": v[f], "settles_after_block": int(s)}
                                                                                         for t, f, s in zip(tok, p["final"], p["settles"])],
                         "held_out_median": H["settle_median"],
                         "how": "for each position of the passage, the first depth (0 after the embedding, l after block l) from which the lens's top token stays the "
                                "model's final prediction at that position up to the last block; the median over the positions"},
        KEYS["kl_first"]: {"post": P9, "value": round(float(kl[1].mean()), 2), "held_out": round(H["kl_bits"][1], 2),
                           "how": "D(final || lens) in bits after block 1, averaged over the positions of the passage; `held_out` is the same over "
                                  f"{H['positions']} held-out positions (the first {H['window']} tokens of {H['windows']} held-out documents), the curve of the figure"},
        KEYS["kl_11"]: {"post": P9, "value": round(float(kl[L - 1].mean()), 2), "held_out": round(H["kl_bits"][L - 1], 2),
                        "how": f"the same after block {L - 1}"},
    }
    more = {
        "passages": [{"key": x["key"], "kind": x["kind"], "text": x["text"], "predicts": v[x["answer"]], "probability": x["answer_p"],
                      "settle_median": x["settle_median"], "answer_settles_after_block": x["settles"][-1]} for x in d["passages"]],
        "held_out_by_depth": {"depth": list(range(L + 1)), "kl_bits": H["kl_bits"], "kl_se": H["kl_se"], "top_token_already_final": H["top1"],
                              "probability_of_final_prediction": H["p_final"], "bias_bits": H["bias_bits"], "rms_of_stream": H["rms"],
                              "positions_settling_at_depth": H["settle_hist"], "positions": H["positions"],
                              **({"tuned_lens_kl_bits": H["tuned_kl_bits"], "tuned_lens_kl_se": H["tuned_kl_se"], "tuned_lens_top_token_already_final": H["tuned_top1"],
                                  "tuned_lens_fit": H["tuned"]} if "tuned_kl_bits" in H else {})},
        "note": "the lens puts the final prediction on top long before it gives it any probability: the last block multiplies the size of the stream "
                f"(root mean square {H['rms'][L - 1]:.0f} before it, {H['rms'][L]:.0f} after), so the direction the readout sees is mostly written by the last block"
                + (f". A tuned lens (an affine map of the normalized stream per block, fitted on {H['tuned']['positions']} other held-out positions to minimize "
                   f"D(final || lens)) is {H['tuned_kl_bits'][1]:.2f} bits from the final distribution after block 1, {H['tuned_kl_bits'][L // 2]:.2f} after block {L // 2} "
                   f"and {H['tuned_kl_bits'][L - 1]:.2f} after block {L - 1}, against {H['kl_bits'][1]:.2f}, {H['kl_bits'][L // 2]:.2f} and {H['kl_bits'][L - 1]:.2f} for the "
                   "logit lens: at the early blocks of this recipe the logit lens needs the tuned lens" if "tuned_kl_bits" in H else ""),
    }
    return out, more


def memories(d: dict) -> tuple[dict, dict]:
    v, K = d["vocab"], d["triggers"]

    def describe(l: int, n: dict, why: str) -> dict:
        triggers = [{"prefix": ("…" + ("".join(v[i] for i in t[1]) + v[t[2]]).replace("\n", "↵")[-60:]), "trigger": v[t[2]], "then": "".join(v[i] for i in t[3]),
                     "coefficient": t[0]} for t in n["triggers"][:3]]
        top = [v[t[0]] for t in n["top"][:8]]
        ends = round(n["same_last"] * K)
        hits = round(n["follows"] * K)
        share = f"all {K} of its strongest triggers end" if ends == K else f"{ends} of its {K} strongest triggers end"
        text = (f"block {l + 1}, neuron {n['neuron'] + 1:,}: {share} in {q(v[n['last_token']])} ("
                + " / ".join(t["prefix"] for t in triggers) + "); its value promotes " + ", ".join(q(t) for t in top)
                + f"; {'all' if hits == K else hits} of the {K} triggers are followed by one of its {d['top']} top tokens")
        return {"post": P5, "text": text, "block": l + 1, "neuron": n["neuron"] + 1, "block_index": l, "neuron_index": n["neuron"], "triggers": triggers,
                "top_tokens": top, "same_last": n["same_last"], "commonest_trigger_token": v[n["last_token"]], "follows": n["follows"], "agrees": n["agrees"], "how": why}

    L = len(d["layers"])
    low = [(n["same_last"], n["follows"], -l, l, k) for l in range(L // 3) for k, n in enumerate(d["layers"][l]["shown"])]
    high = [(n["follows"], n["max"], l, k) for l in range(L - L // 3, L) for k, n in enumerate(d["layers"][l]["shown"]) if n["same_last"] <= 0.5]
    lo, hi = max(low), max(high)
    text = f"{d['text']['tokens']:,} training tokens (the start of the first shard, in windows of {d['text']['window']}), {d['sample']} neurons sampled a block"
    out = {
        KEYS["lower"]: describe(lo[3], d["layers"][lo[3]]["shown"][lo[4]],
                                f"of the neurons exported for blocks 1 to {L // 3}, the one whose {K} strongest triggers most often end in one token (ties: the "
                                f"value that continues more of them); triggers over {text}; value: W_U v, the {d['top']} largest scores"),
        KEYS["upper"]: describe(hi[2], d["layers"][hi[2]]["shown"][hi[3]],
                                f"of the neurons exported for blocks {L - L // 3 + 1} to {L} whose strongest triggers do not mostly end in one token (at most half), "
                                "the one whose triggers are most often followed by one of its value's top tokens"),
    }
    cols = ["same_last", "same_last_se", "shallow", "agreement", "agreement_se", "follows", "follows_se", "agreement_shuffled", "follows_shuffled", "alive"]
    more = {"by_block": {"block": list(range(1, L + 1)), **{c: [row[c] for row in d["layers"]] for c in cols}}, "chance": d["chance"],
            "note": "same_last: mean share of a neuron's strongest triggers that end in its commonest trigger token (an automatic stand-in for the annotators' "
                    "shallow patterns); agreement: share of neurons whose top-scoring token is the token after their strongest trigger (the papers' agreement "
                    "rate); follows: mean share of strongest triggers followed by one of the value's top tokens; *_shuffled: the same with values dealt to "
                    "other neurons of the block; chance.same_last: for positions drawn at random. gpt-pp's units are squared ReLUs: coefficients are never negative",
            "text": d["text"], "sample": d["sample"], "triggers": K, "top": d["top"]}
    return out, more


def placeholders_in(post: Path) -> list[str]:
    return re.findall(r"@@(.+?)@@", post.read_text()) if post.exists() else []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--out", default=str(repo_dir() / "results" / "interp" / "lens-patching-memories.json"))
    args = ap.parse_args()
    load = lambda name: json.loads((Path(args.data) / f"{name}.json").read_text())
    exports = {"activation-patching": load("activation-patching"), "logit-lens": load("logit-lens"), "neurons-as-memories": load("neurons-as-memories")}
    filled, more = {}, {}
    for name, fn in (("activation-patching", patching), ("logit-lens", lens), ("neurons-as-memories", memories)):
        a, b = fn(exports[name])
        filled.update(a)
        more[name] = b
    posts = {P9: SITE / "_posts" / "gpt" / "2026-11-14-gpt-09.md", P5: SITE / "_posts" / "gpt" / "2026-11-14-gpt-05.md"}
    found = {part: placeholders_in(path) for part, path in posts.items()}
    missing = [k for k, x in filled.items() if found[x["post"]] and k not in found[x["post"]]]
    ours = re.compile(r"lens|patch|neuron|relative entropy|logit difference", re.I)
    unfilled = [{"post": part, "placeholder": k} for part, keys in found.items() for k in keys if ours.search(k) and k not in filled]
    out = {"models": {name: {k: e["model"][k] for k in ("run", "checkpoint", "step", "tokens", "final")} for name, e in exports.items()},
           "conventions": "blocks, heads and neurons are counted from 1 in every text; fields named *_index count from 0. Logit differences and attributions are "
                          "taken before the cap on the logits; the lens applies the model's whole readout, cap included. 'After the first layer' is after block 1.",
           "placeholders": filled, "more": more, "unfilled": unfilled, "not_found_in_the_posts": missing}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    size = write_json(args.out, out, compact=False)
    print(f"{args.out}: {size / 1e3:.0f} kB, {len(filled)} placeholders; unfilled {unfilled}; not found in the posts {missing}")
    for k, x in filled.items():
        print(f"  [{x['post']}] {k}\n      -> {x.get('text', x.get('value'))}")


if __name__ == "__main__":
    main()
