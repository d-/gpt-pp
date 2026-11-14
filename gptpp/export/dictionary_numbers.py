"""Write the numbers of part 10's two examples of dictionaries, keyed by the post's own placeholders.

    python -m gptpp.export.dictionary_numbers [--out results/interp/dictionaries.json]

The post "Features" has an example of dictionaries on gpt-pp and pico and an example of pico with one latent
clamped, each with placeholders for numbers that only the lab produces. This script gathers them from what the
other steps wrote, so that nothing in the file is typed by hand except the judgments of a reader:

    the records of the dictionaries   $GPTPP_WORK/sae/<run>-L<layer>/<name>.json   (gptpp.interp.sae)
    the dictionary the figures use    assets/data/gpt/sparsity-against-fidelity.json  (its `used`, by model)
    its decoder directions            $GPTPP_WORK/sae/<run>-L<layer>/<name>.pt
    the random latents, judged        gptpp/export/curated/<model>-L<layer>-<name>-judged.json
    the clamp                         assets/data/gpt/steering-pico.json            (gptpp.export.steering_pico)

Every entry is `{ value, text, how }`: the number, the words to print in its place, and how it was computed. An
entry whose source does not exist yet has value null and says what is missing. Blocks are counted from 1 in `text`,
as the post counts them; latents keep their own indices.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gptpp import repo_dir, work_dir
from gptpp.export.common import DATA
from gptpp.interp import sae as S

RUNS = {"gpt-pp": "gptpp-base", "pico": "pico"}


def thousands(n: float) -> str:
    return f"{int(round(n)):,}"


def millions(n: float) -> str:
    return f"{n / 1e6:.1f} million" if n < 1e7 else f"{n / 1e6:.0f} million"


def cosines(path: Path, device: str = "cpu", chunk: int = 2048) -> dict:
    """The mean absolute cosine between two different decoder directions of a dictionary, and the median over
    latents of the cosine with the nearest other direction."""
    rows = torch.nn.functional.normalize(S.load(path, device).dictionary.detach(), dim=-1)
    n, total, nearest = len(rows), 0.0, []
    for i in range(0, n, chunk):
        c = rows[i:i + chunk] @ rows.T
        c[torch.arange(len(c)), torch.arange(i, i + len(c))] = 0.0  # a direction with itself does not count
        total += float(c.abs().sum())
        nearest.append(c.max(dim=1).values.cpu())
    return {"mean_abs": total / (n * (n - 1)), "median_nearest": float(torch.cat(nearest).median()), "n": n, "d": rows.size(1)}


def entry(value, text: str, how: str) -> dict:
    return {"value": value, "text": text, "how": how}


def missing(what: str) -> dict:
    return entry(None, "", f"not computed yet: {what}")


def family(model: str, doc: dict) -> tuple[dict | None, list[dict], Path | None]:
    """A model's entry of the figure's export, the records of its dictionaries, and their folder."""
    e = doc["models"].get(model)
    if e is None:
        return None, [], None
    folder = S.dict_dir(RUNS[model], e["after"] - 1)
    return e, [json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))], folder


def dictionaries(doc: dict, device: str) -> dict:
    """The placeholders of the example of dictionaries on gpt-pp and pico."""
    out = {}
    g, records, folder = family("gpt-pp", doc)
    p, _, _ = family("pico", doc)
    if p is None:
        raise SystemExit("pico's dictionaries have not been exported: run gptpp.export.sparsity_against_fidelity for pico first")
    pico_used = next(d for d in p["dicts"] if d["name"] == p["used"])
    out["width of pico's dictionary"] = entry(pico_used["n"], thousands(pico_used["n"]), "the dictionary of pico that the figures read (sparsity-against-fidelity.json, pico, used)")
    out["block of pico that its dictionary reads"] = entry(p["after"], str(p["after"]), "the residual stream after this block, counted from 1 (resid_post.%d in the lab's names)" % (p["after"] - 1))
    keys = ["block of gpt-pp that the dictionaries read", "dictionary widths trained on gpt-pp", "values of k trained on gpt-pp", "training vectors per gpt-pp dictionary",
            "other kinds of dictionary trained, and their width", "its width", "its k", "random latents examined", "random latents that could be given one label",
            "mean absolute cosine between its decoder directions", "median cosine to the nearest neighbor", "held-out tokens used for evaluation", "its normalized error",
            "its loss recovered", "gpt-pp's cross-entropy, nats per token", "the same with the reconstruction spliced in", "share of its latents that are dead"]
    if g is None:
        for k in keys:
            out[k] = missing("the dictionaries of gpt-pp (train them with gptpp.interp.sae on the run gptpp-base, then export)")
        return out
    used = next(d for d in g["dicts"] if d["name"] == g["used"])
    topk = [d for d in g["dicts"] if d["kind"] == "topk"]
    relu = [d for d in g["dicts"] if d["kind"] == "relu"]
    widths, ks = sorted({d["n"] for d in topk}), sorted({d["k"] for d in topk})
    out["block of gpt-pp that the dictionaries read"] = entry(g["after"], str(g["after"]), "the residual stream after this block, counted from 1 (resid_post.%d in the lab's names)" % (g["after"] - 1))
    out["dictionary widths trained on gpt-pp"] = entry(widths, "{" + ", ".join(thousands(n) for n in widths) + "}", "the widths n of the TopK dictionaries in the folder of gpt-pp's dictionaries")
    out["values of k trained on gpt-pp"] = entry(ks, "{" + ", ".join(str(k) for k in ks) + "}", "the values of k of those dictionaries; every width was trained at every k" if len(topk) == len(widths) * len(ks) else "the values of k of those dictionaries (not every width at every k: see `also`)")
    out["training vectors per gpt-pp dictionary"] = entry(g["train_tokens"], millions(g["train_tokens"]), "activation vectors (one a token) each dictionary was trained on, drawn from random windows of the training text; the same stream for every dictionary")
    if relu:
        n_relu = sorted({d["n"] for d in relu})
        out["other kinds of dictionary trained, and their width"] = entry(
            {"kind": "relu+l1", "n": n_relu, "l0": [round(d["l0"], 1) for d in sorted(relu, key=lambda d: d["l0"])]},
            f"{len(relu)} ReLU autoencoders with an L1 penalty (the architecture of Definition 10.2.2 with the decoder norm in the penalty), n = {', '.join(thousands(n) for n in n_relu)}, "
            f"at penalties that leave between {min(d['l0'] for d in relu):.0f} and {max(d['l0'] for d in relu):.0f} latents active a token",
            "the ReLU dictionaries in the same folder, trained on the same activations for the same number of steps")
    else:
        out["other kinds of dictionary trained, and their width"] = entry(None, "none", "no ReLU dictionary was trained on gpt-pp")
    out["its width"] = entry(used["n"], thousands(used["n"]), "the dictionary of gpt-pp that the figures read (sparsity-against-fidelity.json, gpt-pp, used)")
    out["its k"] = entry(used["k"], str(used["k"]), "the same dictionary")
    judged = Path(__file__).parent / "curated" / f"gptpp-L{g['after'] - 1}-{g['used']}-judged.json"
    if judged.exists():
        j = json.loads(judged.read_text())
        one = sum(1 for v in j["latents"].values() if v["verdict"] == "one")
        out["random latents examined"] = entry(len(j["latents"]), str(len(j["latents"])), "the random list of the feature browser: a seeded draw from all latents of the dictionary, nobody chose them")
        out["random latents that could be given one label"] = entry(one, str(one), j["how"])
    else:
        out["random latents examined"] = missing(f"the judgments file {judged.name}")
        out["random latents that could be given one label"] = missing(f"the judgments file {judged.name}")
    c = cosines(folder / f"{g['used']}.pt", device)
    out["mean absolute cosine between its decoder directions"] = entry(c["mean_abs"], f"{c['mean_abs']:.3f}", f"the mean of |u_i . u_j| over all pairs of different latents of the dictionary (n = {c['n']}, d = {c['d']}); two random directions in {c['d']} dimensions have sqrt(2 / (pi d)) = {np.sqrt(2 / (np.pi * c['d'])):.3f}")
    out["median cosine to the nearest neighbor"] = entry(c["median_nearest"], f"{c['median_nearest']:.2f}", "for each latent the largest cosine u_i . u_j with another latent of the same dictionary; the median over latents")
    out["held-out tokens used for evaluation"] = entry(g["eval_tokens"], millions(g["eval_tokens"]), "held-out tokens whose activations were read: every token of the held-out split except the document-start marks and the first 16 tokens of each window; the loss is measured on the first %s of them" % thousands(g["splice_targets"]))
    out["its normalized error"] = entry(used["fvu"], f"{used['fvu']:.3f}", "sum |x - x^|^2 / sum |x - mean|^2 over those tokens, the mean being theirs (Definition 10.3.3)")
    out["its loss recovered"] = entry(used["loss_recovered"], f"{100 * used['loss_recovered']:.1f}%", f"(H0 - H*) / (H0 - H) with the stream replaced by zero for H0 = {g['loss_zero']:.2f} nats; against the dataset mean instead of zero ({g['loss_mean']:.2f} nats) it is {100 * used['loss_recovered_mean']:.1f}%")
    out["gpt-pp's cross-entropy, nats per token"] = entry(g["loss_clean"], f"{g['loss_clean']:.3f}", "gpt-pp's next-token loss on the same held-out windows, on the targets that follow the tokens read")
    out["the same with the reconstruction spliced in"] = entry(used["loss_spliced"], f"{used['loss_spliced']:.3f}", "the same loss when the stream after the block is replaced by the dictionary's reconstruction at every token read")
    out["share of its latents that are dead"] = entry(used["dead"], f"{100 * used['dead']:.1f}%" if used["dead"] else "none", f"latents that are active on none of the {millions(g['eval_tokens'])} held-out tokens (the post's definition says 10^7: the held-out split is smaller)")
    return out


def steering() -> dict:
    """The placeholders of the example of pico with one latent clamped."""
    path = DATA / "steering-pico.json"
    keys = ["index of the steered latent", "what the steered latent fires on", "clamp multiple", "stories sampled per condition", "word or theme counted in the stories",
            "share of clamped stories with it", "share of unclamped stories with it", "pico's held-out loss with the clamp", "pico's held-out loss, nats per token"]
    if not path.exists():
        return {k: missing("steering-pico.json (run gptpp.export.steering_pico)") for k in keys}
    doc = json.loads(path.read_text())
    f = doc["features"][doc["default"]["feature"]]
    r = f["strengths"][doc["default"]["strength"]]
    words = ", ".join(f"\"{w}\"" for w in f["words"][:-1]) + f" or \"{f['words'][-1]}\""
    weights = "computed on the 8-bit weights that the page runs, not on the checkpoint"
    return {
        "index of the steered latent": entry(f["id"], str(f["id"]), "the default feature of the steering figure; its index in pico's dictionary"),
        "what the steered latent fires on": entry(f["name"], f"the words for water and ways across a landscape (\"{f['name']}\")" if f["id"] == 3183 else f"\"{f['name']}\"",
                                                 "named by hand from its top examples and its most frequent tokens on held-out text (the feature browser shows them)"),
        "clamp multiple": entry(doc["multiples"][doc["default"]["strength"]], f"{doc['multiples'][doc['default']['strength']]:g}",
                                f"the latent is held at {r['value']:.2f} = this multiple of its largest activation on held-out text ({f['peak']:.2f}), at every token after the document-start mark, as in (10.10)"),
        "stories sampled per condition": entry(doc["stories"], str(doc["stories"]), f"stories of at most {doc['sampling']['steps']} tokens continued from \"{doc['openings'][doc['measured_opening']]['text']}\" at temperature {doc['sampling']['temperature']:g}, seeds 0 to {doc['stories'] - 1}; the same seeds with and without the clamp; {weights}"),
        "word or theme counted in the stories": entry(f["words"], f"One of the words {words}", "a story counts if a word of the list begins a word of it (so \"rivers\" counts for \"river\"); the list was fixed before sampling"),
        "share of clamped stories with it": entry(r["theme"], f"{100 * r['theme']:.0f}%", "the share of the clamped stories that count"),
        "share of unclamped stories with it": entry(r["theme_plain"], f"{100 * r['theme_plain']:.0f}%", "the share of the stories written without the clamp, from the same opening on the same seeds, that count"),
        "pico's held-out loss with the clamp": entry(r["loss"], f"{r['loss']:.2f}", f"pico's next-token loss on {doc['windows']} held-out windows of 256 tokens with the latent clamped at every token read; {weights}"),
        "pico's held-out loss, nats per token": entry(doc["loss_clean"], f"{doc['loss_clean']:.2f}", f"the same loss without the clamp, on the same windows; {weights} (the checkpoint itself: see `also`)"),
    }


def also(doc: dict) -> dict:
    """What the two examples do not ask for and a writer of results may want: pico's dictionary, the grid, the clamp at every strength."""
    out = {"models": {}}
    for model, e in doc["models"].items():
        used = next(d for d in e["dicts"] if d["name"] == e["used"])
        out["models"][model] = {
            "used": used, "after_block": e["after"], "train_tokens": e["train_tokens"], "eval_tokens": e["eval_tokens"], "splice_targets": e["splice_targets"],
            "loss_clean": e["loss_clean"], "loss_zero": e["loss_zero"], "loss_mean": e["loss_mean"], "checkpoint": e["checkpoint"],
            "grid": [{k: d[k] for k in ("name", "kind", "n", "k", "l0", "fvu", "fvu_rescaled", "gain", "loss_spliced", "loss_recovered", "dead", "dense")} for d in e["dicts"]],
        }
    path = DATA / "steering-pico.json"
    if path.exists():
        s = json.loads(path.read_text())
        out["clamp"] = {"loss_as_trained": s["loss_clean"], "surprise_of_its_own_stories": s["surprise_plain"], "dictionary_error_on_the_page_weights": s["dict"]["fvu_page"],
                        "features": [{"latent": f["id"], "name": f["name"], "words": f["words"], "largest_activation": f["peak"], "strengths": f["strengths"]} for f in s["features"]]}
    out["differences from the post's definitions"] = [
        "Definition 10.3.2 writes TopK as TopK_k(W_enc (x - b_dec)). The lab's dictionaries are TopK_k(ReLU(W_enc (x - b_dec) + b_enc)): an encoder bias, and a ReLU before the k largest are kept, as in the released code of the TopK paper and in the dictionary_learning library. With the ReLU a token has at most k active latents, not exactly k (the measured L0 is within 0.01 of k), and no activation is negative, which Definition 10.2.1 requires.",
        "Definition 10.3.3 calls a latent dead if it is active on none of 10^7 tokens. The lab counts over the held-out tokens read (see `eval_tokens` by model): about 9.8 million for gpt-pp and 5.1 million for pico, whose held-out split is smaller.",
        "The dictionary reads activations with the dataset mean subtracted and scaled by one constant (mean norm sqrt(d)); activations f_i and the clamp value c are in those units, and the direction u_i of (10.10) is the unit decoder row divided by that constant. The document-start mark and the first 16 tokens of every window are neither read nor replaced.",
    ]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(repo_dir() / "results" / "interp" / "dictionaries.json"))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    doc = json.loads((DATA / "sparsity-against-fidelity.json").read_text())
    result = {
        "about": "The numbers of Examples 10.3.4 and 10.3.6 of the post \"Features\", keyed by the post's placeholders. Each: value, text (what to print), how (how it was computed). "
                 "Written by gptpp.export.dictionary_numbers from the lab's records; blocks are counted from 1 in text.",
        "example 10.3.4 (dictionaries on gpt-pp and pico)": dictionaries(doc, args.device),
        "example 10.3.6 (pico with one latent clamped)": steering(),
        "also": also(doc),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    todo = [k for part in list(result.values())[1:3] for k, v in part.items() if v["value"] is None and v["text"] == ""]
    print(f"wrote {out}: {sum(len(p) for p in list(result.values())[1:3])} placeholders, {len(todo)} not computed yet")
    for k in todo:
        print("  missing:", k)


if __name__ == "__main__":
    main()
