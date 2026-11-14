"""Attribution graphs of pico on a few prompts, for the figure `gpt-attribution-graph` (fig-10-06) and for
Example 10.4.5 of the post.

    python -m gptpp.export.attribution_graph --ckpt $GPTPP_WORK/ckpt/pico/model.pt \
        [--transcoders $GPTPP_WORK/transcoders/pico/topk-n2048-k16.pt] [--out <site data dir>] [--results <dir>]

For every prompt of PROMPTS (pico's most probable next token is the one the text calls for in each): the
attribution graph (gptpp/interp/attribution_graph.py), its two exactness checks, its influence, its pruning at
several shares of the influence, and an intervention test in pico itself: each of the most influential latents is
taken out of the model's own forward pass, and the change of the outputs and of the later latents is set against
what the graph's edges predict. Runs on the CPU in about a minute.

Writes

    <out>/attribution-graph.json     the graphs of the prompts the figure shows (FIGURE), pruned to the loosest
                                     share: nodes with what fires them and what they promote, edges as shares of
                                     their target's input, the scores at every share, the intervention scatter;
                                     the transcoders' fidelity; a line for every prompt tried
    <results>/attribution-graph.json the numbers of Example 10.4.5, keyed by the post's placeholders, each with a
                                     note on how it was computed

A reader sees blocks counted from 1; this file keeps layers from 0, and the figure adds one.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gptpp import repo_dir
from gptpp.data import TokenData
from gptpp.export.common import CKPT, DATA, WORK, write_json
from gptpp.interp import attribution_graph as ag
from gptpp.interp.transcoder import describe, load_transcoders
from gptpp.model import GPT
from gptpp.tokenizer import Tokenizer

# (key, the prompt, the token the text calls for, what the prediction depends on)
PROMPTS = [
    ("she", "Lily was a little girl. One day,", " she", "a pronoun that agrees with the character"),
    ("key", "Tim found a shiny key. He picked up the", " key", "the object named earlier"),
    ("Ben", "Sara and Ben went to the park. Sara and", " Ben", "the name that followed the same words before"),
    ("Lily", "Lily gave the toy to Ben. Ben gave the toy back to", " Lily", "who gave it first"),
    ("he", "Tim was a little boy. One day,", " he", "a pronoun that agrees with the character"),
    ("cat", "Lily had a cat. Lily loved her", " cat", "the object named earlier"),
    ("ball", "Mia found a red ball. She picked up the", " ball", "the object named earlier"),
    ("frog", "Tom saw a frog. He wanted to catch the", " frog", "the object named earlier"),
    ("Mia", "Tom and Mia are friends. One day, Tom and", " Mia", "the name that followed the same words before"),
    ("Tim", "A boy named Tim had a dog. One day,", " Tim", "a name introduced earlier"),
    ("Tom", "Tom gave the ball to Lily. Lily gave the ball back to", " Tom", "who gave it first"),
    ("He", "Lily is a girl. She is three. Tom is a boy.", " He", "a pronoun that agrees with the character"),
]
FIGURE = ["she", "key", "Ben", "Lily"]  # the prompts the figure shows, the first one by default
SHARES = [0.5, 0.6, 0.7, 0.8, 0.9]  # the shares of the influence the figure's slider keeps
PAPER = 3  # index into SHARES of the paper's default, 0.8
EDGE_SHARE = 0.98
ABLATED = 50  # the most influential latents taken out of pico, a prompt
CONFIRM = 0.5  # a graph's account is confirmed when both rank correlations of the intervention test reach this
PROMOTES = 4.5  # a token is named as promoted by a latent from this many standard deviations up


def token_text(tok: Tokenizer, i: int) -> str:
    return tok.decode([int(i)])


def labels(model: GPT, tcs, tok: Tokenizer, stats: dict, tokens: np.ndarray, layer: int, latents: list[int]) -> list[dict]:
    """What each given latent of one layer is about: the tokens it fires on (by share of its summed activation on
    held-out stories), the tokens it promotes, how often it fires, and the held-out passage where it fires most
    (the token it fires on between brackets).

    The tokens it promotes are those its row of the dictionary raises most through the unembedding, the direct
    effect on the next token at its own position, among the tokens that occur at least once in 100,000 held-out
    tokens. Most latents promote none in particular: the largest of some 3,000 unrelated numbers stands about four
    standard deviations above their mean, so a token is named only from PROMOTES deviations up."""
    ids = torch.tensor(latents, dtype=torch.long)
    mass = stats["mass"][layer][ids]
    share = mass / mass.sum(-1, keepdim=True).clamp_min(1e-30)
    top = share.topk(3, dim=-1)
    common = stats["common"]
    with torch.no_grad():
        effect = (tcs.write(layer)[0][ids] @ model.unembed.weight.float().T)[:, common]
    deviations = (effect - effect.mean(-1, keepdim=True)) / effect.std(-1, keepdim=True)
    up, common_ids = deviations.topk(3, dim=-1), common.nonzero().flatten()
    out = []
    for row, i in enumerate(latents):
        fires = [[token_text(tok, t), round(float(s), 3)] for s, t in zip(top.values[row], top.indices[row]) if s > 0.02]
        example = ""
        if float(stats["best"][layer, i]) > 0:
            start = int(stats["starts"][int(stats["best_window"][layer, i])])
            at = int(stats["best_pos"][layer, i])
            before = tok.decode([int(t) for t in tokens[start + max(1, at - 6):start + at]])
            example = (before + "[" + token_text(tok, tokens[start + at]) + "]" + token_text(tok, tokens[start + at + 1])).replace("\n", " ")
        out.append({"fires": fires, "promotes": [token_text(tok, common_ids[t]) for z, t in zip(up.values[row], up.indices[row]) if z >= PROMOTES],
                    "density": float(stats["density"][layer, i]), "example": example})
    return out


def intervention(model: GPT, tcs, graph: ag.Graph, iota: np.ndarray, kept: np.ndarray, rng: np.random.Generator) -> dict:
    """Take each of the ABLATED most influential latents out of pico and compare.

        influence      rank correlation, over those latents, of the influence with the measured change of the
                       outputs (the absolute changes of the outputs' logits less the mean, weighted by their
                       probabilities): the statistic of Example 10.4.5
        output         rank correlation of the predicted with the measured change of the most probable output
        latents        the same over pairs (a latent taken out, a later latent of the pruned graph)
        points         the scatter: [predicted, measured, 1 for the top output | 0 for a later latent, the latent taken out]
    """
    is_latent, out = graph.kind == ag.LATENT, np.flatnonzero(graph.kind == ag.OUTPUT)
    order = np.argsort(-np.where(is_latent, iota, -1.0))[:min(ABLATED, int(is_latent.sum()))]
    measured = ag.ablate(model, tcs, graph, [int(s) for s in order])
    predicted = (ag.linear_effects(graph)[:, order] * graph.activation[order]).T
    w = graph.activation[out]
    change = (np.abs(measured[:, out]) * w).sum(1)
    pairs = [(a, t) for a, s in enumerate(order) for t in np.flatnonzero(kept & is_latent & (graph.layer > graph.layer[s]) & (graph.pos >= graph.pos[s]))]
    px, py = np.array([predicted[a, t] for a, t in pairs]), np.array([measured[a, t] for a, t in pairs])
    points = [[float(predicted[a, out[0]]), float(measured[a, out[0]]), 1, int(order[a])] for a in range(len(order))]
    if len(pairs):
        strongest = np.argsort(-np.abs(px))[:200]
        others = np.setdiff1d(np.arange(len(pairs)), strongest)
        shown = np.concatenate([strongest, rng.choice(others, size=min(200, len(others)), replace=False)])
        points += [[float(px[q]), float(py[q]), 0, int(order[pairs[q][0]])] for q in shown]
    result = {"ablated": int(len(order)), "influence": ag.spearman(iota[order], change), "output": ag.spearman(predicted[:, out[0]], measured[:, out[0]]),
              "latents": ag.spearman(px, py) if len(pairs) else float("nan"), "pairs": int(len(pairs)), "points": points,
              "sign": float(np.mean(np.sign(predicted[:, out[0]]) == np.sign(measured[:, out[0]])))}
    result["confirmed"] = bool(result["influence"] >= CONFIRM and result["output"] >= CONFIRM)
    return result


def likeliest_walk(graph: ag.Graph, At: np.ndarray, pruned: ag.Pruned) -> list[int]:
    """The most probable walk of the influence in a pruned graph: from the most probable output, at every node to
    the input with the largest share, down to an embedding or an error. Nodes from the output down."""
    node = int(np.flatnonzero(graph.kind == ag.OUTPUT)[0])
    walk = [node]
    while pruned.edges[node].any():
        node = int(np.argmax(np.where(pruned.edges[node], At[node], -1.0)))
        walk.append(node)
    return walk


def walk_report(model: GPT, tcs, tok: Tokenizer, stats: dict, val: np.ndarray, graph: ag.Graph, walk: list[int]) -> list[dict]:
    """The walk in words, from the token up (blocks counted from 1), and for each latent on it: its direct edge to
    the top output, and the change of the top output's value when it alone is taken out of pico, predicted by the
    edges and measured."""
    out = int(np.flatnonzero(graph.kind == ag.OUTPUT)[0])
    latents = [s for s in walk if graph.kind[s] == ag.LATENT]
    measured = ag.ablate(model, tcs, graph, latents) if latents else np.zeros((0, graph.n))
    effects = ag.linear_effects(graph)
    report = []
    for s in reversed(walk):
        kind, at = int(graph.kind[s]), token_text(tok, graph.tokens[int(graph.pos[s])])
        if kind == ag.LATENT:
            label = labels(model, tcs, tok, stats, val, int(graph.layer[s]), [int(graph.index[s])])[0]
            row = latents.index(s)
            report.append({"node": f"block {graph.layer[s] + 1}, latent {graph.index[s]}, on {at!r}", "fires on": label["fires"], "activation": float(graph.activation[s]),
                           "direct edge to the top output": float(graph.A[out, s]),
                           "change of the top output when taken out, predicted": float(effects[out, s] * graph.activation[s]),
                           "change of the top output when taken out, measured": float(measured[row, out])})
        else:
            report.append({"node": {ag.EMBEDDING: f"token {at!r}", ag.ERROR: f"block {graph.layer[s] + 1}, error on {at!r}",
                                    ag.OUTPUT: f"output {token_text(tok, graph.index[s])!r}"}[kind]})
    return report


def export_prompt(model: GPT, tcs, tok: Tokenizer, stats: dict, val: np.ndarray, key: str, text: str, wanted: str, depends: str, rng) -> tuple[dict, dict]:
    """One prompt: (what the figure draws, a line of the table of prompts tried)."""
    idx = torch.tensor(tok.encode(text, bos=True))
    graph = ag.build_graph(model, tcs, idx)
    kind, w = graph.kind, graph.weights()
    At = ag.normalize(graph.A)
    iota = ag.influence(At, w)
    full = ag.scores(At, kind, w)
    prunings = [ag.prune(graph, share, EDGE_SHARE) for share in SHARES]
    test = intervention(model, tcs, graph, iota, prunings[PAPER].nodes, rng)
    rec = ag.record(model, idx)
    best = rec.probs.topk(5)
    out_norm = torch.stack([m.norm(dim=-1) for m in rec.mlp_out]).numpy()  # (L, T)
    predicted_token = token_text(tok, best.indices[0])

    def count(p: ag.Pruned) -> dict:
        drawn = p.edges.any(0) | p.edges.any(1)
        return {"latents": int((p.nodes & (kind == ag.LATENT)).sum()), "errors": int((drawn & (kind == ag.ERROR)).sum()),
                "nodes": int((drawn | (kind == ag.EMBEDDING) | (kind == ag.OUTPUT)).sum()), "edges": int(p.edges.sum())}

    levels = [dict(share=share, **count(p), **p.scores) for share, p in zip(SHARES, prunings)]
    top = int(best.indices[0])
    cap = model.cfg.logit_softcap
    soft_cap = {"cap": cap, "top logit before the cap": float(rec.logits[top]), "after": float(cap * np.tanh(float(rec.logits[top]) / cap)) if cap > 0 else float(rec.logits[top]),
                "mean logit before the cap": float(rec.logits.mean()), "slope of the cap at the top logit": float(1 - np.tanh(float(rec.logits[top]) / cap) ** 2) if cap > 0 else 1.0}
    errors_by_block = [float(iota[(kind == ag.ERROR) & (graph.layer == l)].sum() / w.sum()) for l in range(model.cfg.n_layer)]
    line = {"key": key, "text": text, "wanted": wanted, "predicted": predicted_token, "p": float(best.values[0]), "right": predicted_token == wanted,
            "likeliest walk": walk_report(model, tcs, tok, stats, val, graph, likeliest_walk(graph, At, prunings[PAPER])),
            "share of walks ending at the errors of blocks 1 to 6": errors_by_block, "soft cap": soft_cap,
            "depends": depends, "tokens": len(idx), "nodes": graph.n, "edges": graph.edges(), "latents": int((kind == ag.LATENT).sum()),
            "replacement": full["replacement"], "completeness": full["completeness"],
            "mark": float(iota[(kind == ag.EMBEDDING) & (graph.pos == 0)].sum() / w.sum()),
            "pruned": levels[PAPER], "check": graph.check,
            "test": {k: v for k, v in test.items() if k != "points"}}

    # ---- the nodes and edges of every pruning (those of the loosest, as a rule), renumbered
    any_edge = np.any([p.edges for p in prunings], axis=0)
    drawn = any_edge.any(0) | any_edge.any(1) | (kind == ag.EMBEDDING) | (kind == ag.OUTPUT)
    keep = np.flatnonzero(drawn)
    number = -np.ones(graph.n, dtype=np.int64)
    number[keep] = np.arange(len(keep))
    for point in test["points"]:  # the latent taken out, by its number among the nodes the figure holds (-1: not among them)
        point[3] = int(number[point[3]])
    node_mask = np.zeros(graph.n, dtype=np.int64)
    for level, p in enumerate(prunings):
        node_mask |= ((p.edges.any(0) | p.edges.any(1) | (kind != ag.LATENT) & (kind != ag.ERROR)) & p.nodes).astype(np.int64) << level
    # a node: kind, layer, position, index, size, influence (the share of walks through it), the prunings that keep it. The
    # size is a latent's activation, an output's probability, and for an error its norm over the norm of the MLP's output
    size = np.where(kind == ag.ERROR, graph.activation / out_norm[np.clip(graph.layer, 0, model.cfg.n_layer - 1), graph.pos], graph.activation)
    nodes = [[int(kind[s]), int(graph.layer[s]), int(graph.pos[s]), int(graph.index[s]), float(size[s]), float(iota[s] / w.sum()),
              int(node_mask[s])] for s in keep]
    described: dict[int, dict] = {}
    for layer in range(model.cfg.n_layer):
        mine = [int(s) for s in keep if kind[s] == ag.LATENT and graph.layer[s] == layer]
        for s, label in zip(mine, labels(model, tcs, tok, stats, val, layer, [int(graph.index[s]) for s in mine])):
            described[int(number[s])] = label
    edge_mask = np.zeros(graph.A.shape, dtype=np.int64)
    for level, p in enumerate(prunings):
        edge_mask |= p.edges.astype(np.int64) << level
    t, s = np.nonzero(edge_mask)
    signed = np.sign(graph.A[t, s]) * At[t, s]
    edges = [[int(number[a]), int(number[b]), float(f"{v:.3g}"), int(m)] for a, b, v, m in zip(t, s, signed, edge_mask[t, s])]
    figure = {"key": key, "text": text, "depends": depends, "tokens": [token_text(tok, i) for i in idx.tolist()],
              "predictions": [[token_text(tok, i), float(p)] for p, i in zip(best.values, best.indices)],
              "outputs": [[token_text(tok, graph.index[o]), float(graph.activation[o]), float(graph.value[o])] for o in np.flatnonzero(kind == ag.OUTPUT)],
              "full": {"nodes": graph.n, "edges": graph.edges(), "latents": line["latents"], "errors": int((kind == ag.ERROR).sum()),
                       "mark": line["mark"], **full},
              "check": graph.check, "levels": levels, "nodes": nodes, "labels": described, "edges": [x for e in edges for x in e],
              "test": test}
    return figure, line


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", default=str(CKPT / "pico" / "model.pt"))
    ap.add_argument("--transcoders", default=str(WORK / "transcoders" / "pico" / "topk-n2048-k16.pt"))
    ap.add_argument("--out", default=str(DATA))
    ap.add_argument("--results", default=str(repo_dir() / "results" / "interp"))
    ap.add_argument("--windows", type=int, default=512, help="held-out stories read to describe the latents")
    ap.add_argument("--threads", type=int, default=8, help="CPU threads (a small model gains nothing from more)")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model, cfg = GPT.from_checkpoint(args.ckpt, device="cpu")
    tcs, info = load_transcoders(args.transcoders)
    tok = Tokenizer.load(info["tokenizer"])
    val = TokenData(f"{info['dataset']}/{info['tokenizer']}").val_tokens()
    stats = describe(model, tcs, val, tok.bos_id, n_windows=args.windows, device="cpu")
    stats = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in stats.items()}
    counts = np.bincount(np.asarray(val), minlength=cfg.vocab_size)
    stats["common"] = torch.from_numpy(counts >= len(val) / 100_000)

    figures, lines = [], []
    for key, text, wanted, depends in PROMPTS:
        figure, line = export_prompt(model, tcs, tok, stats, val, key, text, wanted, depends, rng)
        lines.append(line)
        if key in FIGURE:
            figures.append(figure)
        print(f"{key:5s} {line['predicted']!r:8s} p {line['p']:.2f}  nodes {line['nodes']:5d} edges {line['edges']:7d}  replacement {line['replacement']:.3f} "
              f"(pruned {line['pruned']['replacement']:.3f}, {line['pruned']['latents']} latents, {line['pruned']['edges']} edges)  mark {line['mark']:.2f}  "
              f"rank correlations: influence {line['test']['influence']:.2f}, output {line['test']['output']:.2f}, latents {line['test']['latents']:.2f}"
              f"  {'confirmed' if line['test']['confirmed'] else 'not confirmed'}  exact to {line['check']['edges_rel']:.1e}")
    figures.sort(key=lambda f: FIGURE.index(f["key"]))
    ev = info["eval"]
    transcoders = {"n": tcs.cfg.n, "k": tcs.cfg.k, "tokens": info["tokens"], "fvu": ev["fvu"], "l0": ev["l0"], "dead": ev["dead"],
                   "loss": ev["loss"], "loss_recovered": ev["loss_recovered"], "loss_recovered_mean": ev["loss_recovered_mean"],
                   "agreement": ev["agreement"], "heldout_tokens": ev["tokens"]}
    tried, confirmed = len(lines), sum(line["test"]["confirmed"] for line in lines)
    data = {"model": {"name": "pico", "layers": cfg.n_layer, "width": cfg.d_model, "checkpoint": info["checkpoint"]},
            "transcoders": transcoders, "shares": SHARES, "paper": PAPER, "edge_share": EDGE_SHARE, "ablated": ABLATED, "confirm": CONFIRM,
            "prompts": figures, "tried": lines, "confirmed": confirmed}
    size = write_json(Path(args.out) / "attribution-graph.json", data)

    first = lines[0]
    notes = {
        "width of pico's transcoders": [tcs.cfg.n, "latents of each of the six per-layer transcoders (gptpp/interp/transcoder.py)"],
        "their k": [tcs.cfg.k, "TopK: the k largest positive pre-activations of a token are kept"],
        "transcoder training tokens": [info["tokens"], "tokens of TinyStories training stories, each read from its start mark, batches of 4,096 tokens"],
        "pico's held-out loss, nats per token": [round(ev["loss"]["clean"], 3), f"mean next-token loss on {ev['tokens']:,} tokens of {ev['windows']:,} held-out stories read from their start mark (contexts of 256)"],
        "the same with all six MLPs replaced": [round(ev["loss"]["replaced"], 3), "the same tokens, every MLP's output replaced by its transcoder's, later layers reading the replaced stream; with every MLP output set to zero the loss is "
                                                + f"{ev['loss']['zero']:.2f}, set to its mean {ev['loss']['mean']:.2f}"],
        "share of positions with the same top token": [round(ev["agreement"], 3), "positions of those stories where the replacement model's most probable next token is pico's"],
        "prompt of the attribution-graph figure": [first["text"], "the figure's default prompt, after the start mark; " + first["depends"]],
        "pico's predicted token and its probability": [[first["predicted"], round(first["p"], 3)], "pico's most probable next token (float32 checkpoint) and its probability"],
        "nodes of the unpruned graph": [first["nodes"], f"{first['tokens']} embeddings, {first['latents']} active latents, {6 * first['tokens']} error nodes and the output tokens that reach 95% of the probability"],
        "edges of the unpruned graph": [first["edges"], "nonzero entries of the adjacency matrix"],
        "nodes after pruning": [first["pruned"]["nodes"], f"nodes with an edge after keeping the latents that carry 80% of the influence and the edges that carry 98%: {first['pruned']['latents']} latents, {first['pruned']['errors']} error nodes, the embeddings and the outputs"],
        "edges after pruning": [first["pruned"]["edges"], "edges kept at 80% and 98%"],
        "replacement score of the pruned graph": [round(first["pruned"]["replacement"], 3), "the share of walks from the outputs that end at token embeddings in the pruned graph, walks that reach a pruned edge counted with those that end at error nodes; "
                                                  + f"for the unpruned graph it is {first['replacement']:.3f}, and {first['mark']:.2f} of all walks end at the start mark's embedding"],
        "latents ablated in the intervention test": [first["test"]["ablated"], "the most influential latents of the unpruned graph, each taken out of pico's own forward pass alone: its activation times its dictionary row is subtracted from its MLP's output at its position"],
        "that rank correlation on pico": [round(first["test"]["influence"], 2), "Spearman correlation, over those latents, of the influence with the measured change of the outputs (absolute changes of their logits less the mean logit, before the soft cap, weighted by their probabilities); "
                                          + f"between the change of the top output that the edges predict and the measured one it is {first['test']['output']:.2f}, and over {first['test']['pairs']:,} pairs of an ablated latent and a later latent of the pruned graph {first['test']['latents']:.2f}"],
        "prompts for which graphs were built": [tried, "the prompts of PROMPTS in gptpp/export/attribution_graph.py; pico's top token is the one the text calls for in " + str(sum(line["right"] for line in lines))],
        "prompts whose graph an intervention confirmed": [confirmed, f"prompts where both rank correlations (influence against the change of the outputs, predicted against measured change of the top output) are at least {CONFIRM}; a rule fixed before the test was run"],
    }
    worst = min(lines, key=lambda line: min(line["test"]["influence"], line["test"]["output"]))
    results = {"placeholders": {k: v[0] for k, v in notes.items()}, "notes": {k: v[1] for k, v in notes.items()},
               "output_nodes": "logits before the soft cap, less their mean over the vocabulary; weights: the model's probabilities, after the cap. "
                               "The cap is a nonlinearity that is neither a pattern nor a denominator; the logit before it is linear in the final stream, so the "
                               "identity (10.12) is exact for it, the cap is increasing so the order of the tokens is unchanged, and the intervention measures "
                               "the same quantity. See 'soft cap' under each prompt for the sizes.",
               "a prompt on which the graph fails": {"text": worst["text"], "predicted": worst["predicted"], "rank correlations": worst["test"]},
               "prompts": lines, "transcoders": transcoders}
    Path(args.results).mkdir(parents=True, exist_ok=True)
    (Path(args.results) / "attribution-graph.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print(f"wrote {Path(args.out) / 'attribution-graph.json'} ({size / 1e6:.2f} MB) and {Path(args.results) / 'attribution-graph.json'}; "
          f"{confirmed} of {tried} graphs confirmed by intervention")


if __name__ == "__main__":
    main()
