"""The measured numbers of the post "The block" (part 5): one entry for each placeholder of the post.

    python -m gptpp.export.part5_numbers [--out results/interp/part-5.json]

CPU only, a second. It measures nothing: it reads what is already on disk and says where each number comes from.

    results/runs/<run>/            the summaries and logs of the ladder (ladder-<k>-<name>-s<seed>), of the swaps
                                   around the 2026 recipe (swap-*) and of the four warmup runs (warm-2019-*),
                                   through `gptpp.export.ladder_results.collect`
    assets/data/gpt/ (the site)    gradient-by-depth.json (gradients at initialization, written by
                                   `gptpp.export.gradient_by_depth`) and activation-zoo.json (what the hidden units
                                   do, written by `gptpp.export.activation_zoo`)
    results/interp/lens-patching-memories.json    the two neurons of gpt-pp (their entries are copied)
    assets/data/gpt/training-run-replayed.json    the loss of token frequencies alone on held-out text (part 8's unigram
                                   level), for scale beside the runs that do not train
    $GPTPP_WORK/logs/swap-norm-none.log           the run without normalization, which did not finish
    the post itself                the exact words of its placeholders `@@...@@` and of its RESULTS questions

It writes one JSON object: the placeholders' words as keys, each with a value, its unit, a text ready for the page
and a note on how it was computed; `results_lines`, the measured answers to the post's `<!-- RESULTS: ... -->`
questions, keyed by the line of the post and the question; and `_supplement`, the numbers behind those answers.
It stops if a placeholder of the post has no entry. Entries that need two seeds of one run are filled from both
when the second has finished and say so when it has not: run the script again when more seeds land.

Losses are held-out losses in nats per token at the end of training (the trainer's own evaluation on 1,048,576
held-out tokens). "Seed noise" is twice the largest difference seen between two seeds of one rung of the ladder.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from .. import repo_dir
from .common import DATA, SITE, WORK, write_json
from .ladder_results import collect, round_sig

POST = SITE / "_posts" / "gpt" / "2026-11-14-gpt-05.md"
ACTS = (("relu", "ReLU", "swap-act-relu"), ("gelu", "GELU", "swap-act-gelu"), ("silu", "SiLU", "swap-act-silu"),
        ("relu2", "squared ReLU", "swap-base"), ("swiglu", "SwiGLU", "swap-act-swiglu"), ("geglu", "GeGLU", "swap-act-geglu"))
NORMS = (("layer norm", "swap-norm-layernorm"), ("RMS norm with a gain", "swap-norm-rmsnorm-gain"),
         ("parameter-free RMS norm", "swap-base"), ("dynamic tanh", "swap-norm-dyt"), ("no normalization", "swap-norm-none"))
WARM = (("post-norm with warmup", "warm-2019-post"), ("post-norm without warmup", "warm-2019-post-none"),
        ("pre-norm with warmup", "warm-2019-base"), ("pre-norm without warmup", "warm-2019-none"))
TABLE = (  # the post's table: the 2026 recipe with one component swapped back
    ("ladder, learned positions: change in held-out loss", "swap-pos-learned", "a learned position table in place of rotary positions"),
    ("ladder, layer norm with gain and bias: change in held-out loss", "swap-norm-layernorm",
     "layer norm with its learned gain in place of the parameter-free RMS norm; the run has no bias in the norm, because the recipe has no biases anywhere (biases alone are the row below)"),
    ("ladder, GELU: change in held-out loss", "swap-act-gelu", "GELU in place of squared ReLU"),
    ("ladder, no query-key normalization: change in held-out loss", "swap-attn-no-qk-norm", "scores q.k/sqrt(d_head) in place of 12 cos(q, k)"),
    ("ladder, with biases: change in held-out loss", "swap-bias", "a bias in every linear layer"),
    ("ladder, tied embedding: change in held-out loss", "swap-tied", "one matrix for embedding and unembedding, and no norm on the embedding"),
    ("ladder, no soft-cap: change in held-out loss", "swap-no-softcap", "unbounded logits"),
    ("ladder, scaled 0.02 writers: change in held-out loss", "swap-init-gpt2", "GPT-2's initialization: 0.02, the writers scaled by 1/sqrt(2L)"),
)


def f3(v: float) -> str:
    return f"{v:.3f}"


def signed(v: float, digits: int = 3) -> str:
    s = f"{abs(v):.{digits}f}"
    return ("+" if v > 0 else "−" if v < 0 and float(s) != 0 else "") + s


def failed_run(name: str) -> dict | None:
    """What the queue's log says of a run that stopped on a loss that is not finite: the step, and the last finite loss."""
    log = WORK / "logs" / f"{name}.log"
    if not log.exists():
        return None
    last, bad = None, None
    for line in log.read_text(errors="replace").splitlines():
        m = re.match(r"step\s+(\d+)/\d+ \| tokens\s+([\d.]+)M \| loss (\S+)", line)
        if m and m.group(3) == "nan" and bad is None:
            bad = int(m.group(1))
        elif m and bad is None:
            last = (int(m.group(1)), float(m.group(3)))
    return {"not_finite_at_step": bad, "last_finite_step": last[0], "last_finite_loss": last[1]} if bad and last else None


def unigram_level(data: Path, vocab: int) -> float | None:
    """The held-out loss of token frequencies alone, in nats per token, from the levels of the replayed training run (part 8).

    That file gives its levels in bits per byte; its uniform level, log2(vocab) bits a token, fixes the bytes in a token."""
    path = data / "training-run-replayed.json"
    if not path.exists():
        return None
    levels = json.loads(path.read_text()).get("numbers", {})
    if "level_unigram" not in levels or "level_uniform" not in levels:
        return None
    return levels["level_unigram"] * math.log(vocab) / levels["level_uniform"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--out", default=str(repo_dir() / "results" / "interp" / "part-5.json"))
    args = ap.parse_args()
    data = Path(args.data)
    # six significant digits, as the three files the figures read hold them (ladder.json, activation-zoo.json,
    # gradient-by-depth.json): the page and its figures then print the same digits, rung by rung and swap by swap
    R = round_sig(collect(repo_dir() / "results" / "runs", ["ladder-", "swap-", "warm-"]))
    runs, steps, spread = R["runs"], R["ladder"]["steps"], R["ladder"]["spread"]
    grad = json.loads((data / "gradient-by-depth.json").read_text())
    zoo = {a["act"]: a for a in json.loads((data / "activation-zoo.json").read_text())["acts"]}
    memories = json.loads((repo_dir() / "results" / "interp" / "lens-patching-memories.json").read_text())
    trained, V = grad["training"]["runs"], grad["training"]["model"]["vocab_size"]
    rate, warmup = trained[0]["lr"], round(max(r["warmup_tokens"] for r in trained) / 1e6)
    unigram = unigram_level(data, V)
    scale = (f"for scale, token frequencies alone give {unigram:.2f} on held-out text (the unigram level of part 8's replayed run) and a uniform guess over the {V:,} tokens {math.log(V):.2f}"
             if unigram else f"for scale, a uniform guess over the {V:,} tokens gives {math.log(V):.2f}")

    def loss(run: str) -> float | None:
        return runs[run]["val_loss"] if run in runs else None

    def need(run: str) -> float:
        if run not in runs:
            raise SystemExit(f"{run} has not finished: its number cannot be written")
        return runs[run]["val_loss"]

    base = need("swap-base")
    M = round(runs["swap-base"]["tokens"] / 1e6)
    if not spread["rungs"]:
        raise SystemExit("no rung of the ladder has two seeds yet: the seed noise that every entry is judged against is not known")
    noise = 2 * spread["max_abs_diff"]
    pairs = {s["name"]: abs(s["val_loss"][0] - s["val_loss"][1]) for s in steps if len(s["val_loss"]) > 1 and None not in s["val_loss"]}
    seeds_text = (f"between {min(pairs.values()):.3f} and {max(pairs.values()):.3f} nats on the {len(pairs)} rungs of the ladder that have two seeds "
                  f"({spread['mean_abs_diff']:.3f} on average)") if pairs else "not known: no rung has two seeds yet"

    def within(d: float) -> bool:
        return noise is not None and abs(d) < noise

    def judged(d: float) -> str:
        return "inside seed noise" if within(d) else ("worse" if d > 0 else "better") + " by more than seed noise"

    out: dict = {
        "_post": "_posts/gpt/2026-11-14-gpt-05.md",
        "_written_by": "python -m gptpp.export.part5_numbers (from results/runs, assets/data/gpt/gradient-by-depth.json and activation-zoo.json, results/interp/lens-patching-memories.json)",
        "_note": "one entry for each placeholder @@...@@ of the post, keyed by its exact words: value, unit, a text for the page, and how it was computed. "
                 "Losses are held-out losses in nats per token at the end of training. One seed a run unless an entry says two. "
                 f"Seed noise: twice the largest difference between two seeds of one rung of the ladder (100M tokens), {noise:.3f} nats today; the swaps are single runs at 50M tokens, "
                 "where no pair of seeds was run, so the threshold is borrowed from the ladder. Numbers are rounded to six significant digits first, as in the files the figures read.",
    }

    # ---- Section 5.2: where the norm sits
    depths = grad["init"]["depths"]
    ratio = {init: {place: [c["proj"][-1] / c["proj"][0] for c in (grad["init"]["inits"][init]["layers"][place][str(d)] for d in depths)]
                    for place in ("post", "pre")} for init in ("gpt2", "glorot")}

    def three(v: list[float]) -> str:
        return f"{v[0]:.2f}, {v[1]:.2f} and {v[2]:.2f}"

    out["gradient norm of the MLP output matrix at initialization, last block divided by first block, for post-norm and for pre-norm at depths 12, 24 and 48"] = {
        "value": ratio, "unit": "ratio (last block over first block)",
        "text": f"{three(ratio['gpt2']['post'])} for post-norm and {three(ratio['gpt2']['pre'])} for pre-norm at 12, 24 and 48 blocks under GPT-2's initialization; "
                f"{three(ratio['glorot']['post'])} against {three(ratio['glorot']['pre'])} when every block writes at the scale of the stream (Glorot weights, the setting of the proposition)",
        "note": f"gptpp.export.gradient_by_depth: the 2019 recipe at width {grad['init']['model']['d_model']}, {grad['init']['seeds']} random initializations a cell, one batch of "
                f"{grad['init']['batch']['windows']} x {grad['init']['batch']['tokens']} held-out tokens; the Frobenius norm of the gradient of each block's second MLP matrix, averaged over the seeds, "
                "last block over first block. Post-norm does NOT rise toward the output; pre-norm falls toward it. Under GPT-2's initialization the first post-norm block as a whole "
                f"receives {grad['init']['inits']['gpt2']['layers']['post']['48']['all'][0] / grad['init']['inits']['gpt2']['layers']['post']['48']['all'][1]:.0f} times the gradient "
                "of the others (its attention reads the raw embedding), which this matrix alone does not show.",
    }
    warm = [loss(run) for _, run in WARM]
    if None in warm:
        raise SystemExit("a warmup run has not finished: " + ", ".join(run for (_, run), v in zip(WARM, warm) if v is None))
    out["held-out loss of the 8-layer ladder model in post-norm with warmup, post-norm without warmup, pre-norm with warmup and pre-norm without warmup"] = {
        "value": warm, "unit": "nats per token",
        "text": f"{f3(warm[0])} and {f3(warm[1])} for post-norm with and without warmup, {f3(warm[2])} and {f3(warm[3])} for pre-norm",
        "note": f"runs {', '.join(run for _, run in WARM)}: the 2019 recipe (AdamW, peak rate {rate}, cosine decay), {round(runs['warm-2019-base']['tokens'] / 1e6)}M tokens, one seed each; "
                f"the warmup is {warmup}M tokens. Both post-norm runs end at {warm[0]:.2f}; {scale}: at this rate post-norm "
                + ("does not get past token frequencies" if unigram and abs(warm[0] - unigram) < 0.05 else "does not train") +
                f", with or without warmup; pre-norm trains in both and is {warm[3] - warm[2]:.2f} nats behind without it.",
    }

    # ---- Section 5.2: which norm
    gone = failed_run("swap-norm-none")
    norm_values = [loss(run) for _, run in NORMS]
    out["held-out loss of the 8-layer ladder model with layer norm, with RMS norm with a gain, with parameter-free RMS norm, with dynamic tanh, and with no normalization"] = {
        "value": norm_values, "unit": "nats per token",
        "text": f"{f3(need('swap-norm-layernorm'))}, {f3(need('swap-norm-rmsnorm-gain'))}, {f3(base)} and {f3(need('swap-norm-dyt'))}; without normalization the run did not finish: "
                + (f"its loss was {gone['last_finite_loss']:.2f} at step {gone['last_finite_step']} and no longer finite at step {gone['not_finite_at_step']}" if gone and norm_values[4] is None
                   else f"{f3(norm_values[4])}" if norm_values[4] is not None else "its loss stopped being finite"),
        "note": f"the 2026 recipe with only the norm changed, {M}M tokens, one seed each (swap-norm-layernorm, swap-norm-rmsnorm-gain, swap-base, swap-norm-dyt, swap-norm-none; the last also "
                "drops the norm on the embedding). Layer norm here has its gain and no bias (the recipe has no biases). Against the parameter-free norm of the recipe: layer norm "
                f"{signed(need('swap-norm-layernorm') - base)} ({judged(need('swap-norm-layernorm') - base)}), RMS norm with a gain {signed(need('swap-norm-rmsnorm-gain') - base)} "
                f"({judged(need('swap-norm-rmsnorm-gain') - base)}), dynamic tanh {signed(need('swap-norm-dyt') - base)} ({judged(need('swap-norm-dyt') - base)}). "
                "The model without normalization diverges although its projections start at zero.",
    }

    # ---- Section 5.3: activations
    acts = {key: need(run) for key, _, run in ACTS}
    out["held-out loss of the 8-layer ladder model with ReLU, GELU, SiLU, squared ReLU, SwiGLU and GeGLU at matched parameters"] = {
        "value": [acts[key] for key, _, _ in ACTS], "unit": "nats per token",
        "text": ", ".join(f"{f3(acts[key])} ({name})" for key, name, _ in ACTS[:-1]) + f" and {f3(acts['geglu'])} (GeGLU)",
        "note": f"the 2026 recipe with only the activation changed, {M}M tokens, one seed each; the gated two have hidden width {zoo['swiglu'].get('d_mlp')} (two thirds of "
                f"{zoo['relu2'].get('d_mlp')}) so that every model has {runs['swap-base']['params']['total']:,} parameters. Against squared ReLU: "
                + ", ".join(f"{name} {signed(acts[key] - base)}" for key, name, _ in ACTS if key != "relu2") + ".",
    }
    own = [s for s in steps if s["name"] == "softcap"][0]
    both = len(own["val_loss"]) > 1 and None not in own["val_loss"]
    out["difference in held-out loss between two seeds of the 8-layer ladder model in the 2026 recipe"] = {
        "value": abs(own["val_loss"][0] - own["val_loss"][1]) if both else (max(pairs.values()) if pairs else None), "unit": "nats per token",
        "two_seeds_of_the_2026_recipe": both,
        "text": (f"{abs(own['val_loss'][0] - own['val_loss'][1]):.3f} nats ({seeds_text})" if both else seeds_text + "; the second seed of the 2026 recipe itself has not finished"),
        "note": "the last rung of the ladder (ladder-9-softcap, the 2026 recipe, 100M tokens) with seeds 0 and 1 when both have finished; until then the range over the rungs that have two seeds: "
                + ", ".join(f"{k} {v:.4f}" for k, v in pairs.items()) + ". The swaps are one seed each at 50M tokens; no pair of seeds was run at that budget.",
    }

    # ---- Section 5.4: two neurons, measured elsewhere
    for key, entry in memories["placeholders"].items():
        if entry.get("post") == "gpt-05":
            out[key] = {"value": {k: entry[k] for k in ("block", "neuron", "same_last", "follows", "commonest_trigger_token", "top_tokens") if k in entry}, "unit": "a neuron",
                        "text": entry["text"], "note": "copied from results/interp/lens-patching-memories.json: " + entry["how"]}

    # ---- Section 5.5: initializations
    scaled = need("swap-init-gpt2")
    out["held-out loss of the 8-layer ladder model with every matrix drawn at 0.02, with the writers scaled by 1/sqrt(2L), and with the writers at zero"] = {
        "value": [None, scaled, base], "unit": "nats per token",
        "text": f"{f3(scaled)} with the writers scaled by 1/√(2L) and {f3(base)} with the writers at zero; the first of the three, every matrix at 0.02 with no scaling, was not run",
        "note": f"only two of the three initializations were run: swap-init-gpt2 ({scaled:.4f}) and swap-base ({base:.4f}), the 2026 recipe, {M}M tokens, one seed each; they are {abs(scaled - base):.4f} apart, "
                f"{judged(scaled - base)}. No run draws every matrix at 0.02 without the scaling: the sentence of the post must name two initializations, not three. On the ladder, under AdamW and "
                f"at 100M tokens, going from the scaled writers to zero (rung 5 to rung 6) changed the loss by {signed(steps[6]['mean_val_loss'] - steps[5]['mean_val_loss'])}.",
    }

    # ---- Section 5.6: the table, one component swapped back
    for words, run, what in TABLE:
        d = need(run) - base
        out[words] = {"value": d, "unit": "nats per token", "text": signed(d),
                      "note": f"{run} less swap-base: the 2026 recipe with {what}; {M}M tokens, one seed each; {judged(d)} (seed noise {noise:.3f})."}

    # ---- the measured answers to the post's RESULTS questions
    post = POST.read_text().splitlines() if POST.exists() else []
    questions = [(i + 1, re.search(r"<!-- RESULTS: (.*?) -->", line).group(1)) for i, line in enumerate(post) if "<!-- RESULTS:" in line]
    g, G = grad["init"]["inits"]["gpt2"], grad["init"]["inits"]["glorot"]
    uneven = max(abs(G["layers"]["post"][str(d)]["all"][-1] / G["layers"]["post"][str(d)]["all"][0] - 1) for d in depths)

    def ends(init: dict, place: str, d: int) -> str:
        c = init["layers"][place][str(d)]
        return f"{c['all'][0]:.2f} to {c['all'][-1]:.2f}"

    lm = {init: {place: [grad["init"]["inits"][init]["last_mlp"][place][str(d)] for d in depths] for place in ("post", "pre")} for init in ("gpt2", "glorot")}
    order = sorted(ACTS, key=lambda a: acts[a[0]])
    second, first = [acts[k] for k in ("relu2", "swiglu", "geglu")], [acts[k] for k in ("relu", "gelu", "silu")]
    by = memories["more"]["neurons-as-memories"]["by_block"]
    chance = memories["more"]["neurons-as-memories"]["chance"]
    hi_same = max(range(len(by["block"])), key=lambda i: by["same_last"][i])
    hi_agree = max(range(len(by["block"])), key=lambda i: by["agreement"][i])
    swaps = {name: r["val_loss"] - base for name, r in runs.items() if r["family"] == "swap" and name != "swap-base"}
    costly = sorted(swaps.items(), key=lambda kv: -kv[1])
    quiet = [k for k, v in swaps.items() if within(v)]
    better = sorted(((k, v) for k, v in swaps.items() if v < 0 and not within(v)), key=lambda kv: kv[1])
    total = steps[9]["mean_val_loss"] - steps[0]["mean_val_loss"]
    opt = steps[7]["mean_val_loss"] - steps[6]["mean_val_loss"]
    rungs = ", ".join(f"{s['name']} {signed(s['mean_val_loss'] - steps[s['k'] - 1]['mean_val_loss'])}" for s in steps[1:])
    curve = {name: dict(zip(runs[name]["curve"]["tokens"], runs[name]["curve"]["val_loss"])) for name in ("swap-init-gpt2", "swap-base")}
    early = sorted(set(curve["swap-init-gpt2"]) & set(curve["swap-base"]))[1:3]
    recipe = (f"At 8 layers and {M}M tokens three parts of the 2026 block are not better than what they replaced: with a learned gain the norm is {abs(swaps['swap-norm-rmsnorm-gain']):.3f} better "
              f"(layer norm {abs(swaps['swap-norm-layernorm']):.3f}), with biases the model is {abs(swaps['swap-bias']):.3f} better, and GELU ties squared ReLU ({signed(swaps['swap-act-gelu'])}); one seed each, "
              f"against a seed spread of up to {spread['max_abs_diff']:.3f}. What is clearly worth something: rotary positions ({signed(swaps['swap-pos-learned'])} with a learned table), the optimizer "
              f"({signed(opt)} on the ladder), query-key normalization ({signed(swaps['swap-attn-no-qk-norm'])} without) and pre-norm ({signed(swaps['swap-norm-post'])} with post-norm and layer norm, "
              f"{signed(need('swap-norm-post') - need('swap-norm-layernorm'))} against the same layer norm before the sublayers). The norm on the embedding is at the edge: "
              f"{signed(swaps['swap-no-embed-norm'])} without it, {signed(swaps['swap-tied'])} when it goes together with the second vocabulary matrix (the tied run, "
              f"{(runs['swap-base']['params']['total'] - runs['swap-tied']['params']['total']) / 1e6:.1f}M parameters fewer).")
    two_answers = (f"The ladder and the swaps ask different questions and twice give different answers: squared ReLU gained {abs(steps[4]['mean_val_loss'] - steps[3]['mean_val_loss']):.3f} on the ladder "
                   f"(under AdamW, right after the norm lost its gain) and GELU put back into the finished recipe costs {signed(swaps['swap-act-gelu'])}; writers at zero gained "
                   f"{abs(steps[6]['mean_val_loss'] - steps[5]['mean_val_loss']):.3f} on the ladder and the scaled rule put back costs {signed(swaps['swap-init-gpt2'])}. "
                   f"Query-key normalization goes the other way: {signed(steps[5]['mean_val_loss'] - steps[4]['mean_val_loss'])} on the ladder, {signed(swaps['swap-attn-no-qk-norm'])} when taken out of the finished recipe.")
    answers_in_order = [
        # where the norm sits
        f"No. At initialization post-norm is level, not rising: under GPT-2's initialization every block but the first receives the same gradient ({g['layers']['post']['48']['all'][1]:.3f} at 48 blocks; "
        f"the first, which reads the raw embedding, {g['layers']['post']['48']['all'][0]:.2f}), and with Glorot weights the two ends differ by at most {100 * uneven:.0f}% and in no steady direction ("
        + ", ".join(f"{ends(G, 'post', d)} at {d} blocks" for d in depths) + "). "
        f"Pre-norm is the one with a slope, and it falls toward the output: {ends(g, 'pre', 48)} at 48 blocks under GPT-2's initialization, {ends(G, 'pre', 48)} with Glorot weights, as its stream grows "
        f"({G['layers']['pre']['48']['stream'][0]:.2f} to {G['layers']['pre']['48']['stream'][-1]:.2f}). The top-layer gradient with depth agrees with Proposition 5.2.5 in its own setting (Glorot weights): "
        f"from 12 to 48 blocks it changes by a factor {lm['glorot']['post'][2] / lm['glorot']['post'][0]:.2f} for post-norm and {lm['glorot']['pre'][2] / lm['glorot']['pre'][0]:.2f} for pre-norm "
        f"(1/sqrt(4) = 0.50); under GPT-2's initialization, whose 1/sqrt(2L) makes the pre-norm stream end at the same size at every depth ("
        + ", ".join(f"{g['layers']['pre'][str(d)]['stream'][-1]:.3f}" for d in depths) + f"), it does not change in either ({lm['gpt2']['post'][2] / lm['gpt2']['post'][0]:.2f}, "
        f"{lm['gpt2']['pre'][2] / lm['gpt2']['pre'][0]:.2f}). Training at 8 layers (the 2019 recipe, rate 0.0024, 50M tokens, one seed each): post-norm does not train with or without warmup "
        f"({f3(warm[0])}, {f3(warm[1])}; {scale}); pre-norm is not indifferent to warmup: {f3(warm[2])} with, {f3(warm[3])} without ({warm[3] - warm[2]:.2f} behind). "
        f"In the 2026 recipe (Muon, writers at zero, {M}M tokens) post-norm does train: {f3(need('swap-norm-post'))} against {f3(need('swap-norm-layernorm'))} for the same layer norm placed before the sublayers.",
        # which norm
        f"Layer norm ({f3(need('swap-norm-layernorm'))}) and RMS norm with a gain ({f3(need('swap-norm-rmsnorm-gain'))}) are within {abs(need('swap-norm-layernorm') - need('swap-norm-rmsnorm-gain')):.3f} of "
        f"each other; the parameter-free RMS norm of the recipe ({f3(base)}) is {abs(swaps['swap-norm-rmsnorm-gain']):.3f} behind them, more than seed noise ({noise:.3f}): at this size the learned gain helps and "
        f"centering does not matter. On the ladder the same step (rung 3, layer norm with gain to parameter-free RMS norm) cost {signed(steps[3]['mean_val_loss'] - steps[2]['mean_val_loss'])}. "
        f"Dynamic tanh lands at {f3(need('swap-norm-dyt'))}, {signed(swaps['swap-norm-dyt'])}. Without normalization the model does not train even with zero-initialized projections: "
        + (f"the loss was {gone['last_finite_loss']:.2f} at step {gone['last_finite_step']} and not finite at step {gone['not_finite_at_step']}." if gone else "the run did not finish.") + " One seed each.",
        # activations
        "Best to worst: " + ", ".join(f"{name} {f3(acts[key])}" for key, name, _ in order) + f". Seed noise is {noise:.3f}: only ReLU ({signed(acts['relu'] - base)} against squared ReLU) is "
        f"outside it; SwiGLU ({signed(acts['swiglu'] - base)}), GeGLU ({signed(acts['geglu'] - base)}), GELU ({signed(acts['gelu'] - base)}) and SiLU ({signed(acts['silu'] - base)}) are inside. The second-order "
        f"units average {sum(second) / 3:.3f} and the first-order ones {sum(first) / 3:.3f}: {sum(first) / 3 - sum(second) / 3:.3f} apart, at the edge of noise and mostly ReLU's doing; GELU ties squared ReLU. "
        "The 0.02 to 0.05 of the literature is not there at 8 layers and 50M tokens: close to a null result. What does differ is what the units do: "
        f"{100 * zoo['relu2']['zero']:.0f}% of squared ReLU's activations and {100 * zoo['relu']['zero']:.0f}% of ReLU's are exact zeros, GELU has {100 * zoo['gelu']['zero']:.0f}%; "
        f"the kurtosis of the activations is {zoo['relu2']['kurtosis']:.0f} for squared ReLU, {zoo['swiglu']['kurtosis']:.0f} and {zoo['geglu']['kurtosis']:.0f} for the two gated units and {zoo['gelu']['kurtosis']:.0f} for GELU; "
        f"dead units are rare in all six: none in {sum(a['dead'] == 0 for a in zoo.values())} of them and {100 * max(a['dead'] for a in zoo.values()):.2f}% of the units at most "
        f"({max(zoo.values(), key=lambda a: a['dead'])['act']}).",
        # neurons
        f"In part. The share of a neuron's 25 strongest triggers that end in its commonest last token is highest in block {by['block'][hi_same]} ({by['same_last'][hi_same]:.2f}) and lowest in block "
        f"{by['block'][-1]} ({by['same_last'][-1]:.2f}); random positions give {chance['same_last']:.2f}. Lower layers are the more shallow, as in the 2021 table, but by a modest margin. The top token of a "
        f"value is the token after its key's strongest trigger for {100 * by['agreement'][0]:.0f}% of the neurons of block 1, {100 * by['agreement'][hi_agree]:.0f}% in block {by['block'][hi_agree]} and "
        f"{100 * by['agreement'][-1]:.0f}% in block 12 (chance {100 * chance['agreement']:.3f}%): values continue their triggers in the upper half, not at the very top. Coefficients are non-negative "
        f"(squared ReLU). {memories['more']['neurons-as-memories']['sample']} neurons sampled a block, the {memories['more']['neurons-as-memories']['triggers']} strongest triggers of each.",
        # initializations
        f"Only two of the three were run. With the writers scaled by 1/sqrt(2L) the 2026 recipe ends at {scaled:.4f}, with the writers at zero at {base:.4f}: {abs(scaled - base):.4f} apart, {judged(scaled - base)}. "
        + "During training the two cross: " + ", ".join(f"{signed(curve['swap-init-gpt2'][t] - curve['swap-base'][t])} at {round(t / 1e6)}M tokens" for t in early) +
        f" (scaled less zero). No difference at the end and none that lasts early, at 8 layers. Plain 0.02 on every matrix was not run, so the sources' claim that the scaled rule clearly beats it is "
        f"untested here. On the ladder, under AdamW at 100M tokens, zero writers were {signed(steps[6]['mean_val_loss'] - steps[5]['mean_val_loss'])} against the scaled rule (rung 6 has "
        f"{sum(v is not None for v in steps[6]['val_loss'])} seed).",
        # the table
        "Swapped back into the 2026 recipe, the rows of the table that cost the most: " + ", ".join(f"{k.replace(': change in held-out loss', '').replace('ladder, ', '')} {signed(v)}"
                                                                                                    for k, v in sorted(((w, need(r) - base) for w, r, _ in TABLE), key=lambda kv: -kv[1])[:2]) +
        "; among the other swaps, worse by more than noise: " + ", ".join(f"{k.replace('swap-', '')} {signed(v)}" for k, v in costly if v > 0 and not within(v) and k not in {r for _, r, _ in TABLE}) +
        (", and without normalization the run diverged" if gone else "") + f". Inside seed noise ({noise:.3f}): " + ", ".join(f"{k.replace('swap-', '')} {signed(swaps[k])}" for k in sorted(quiet, key=lambda k: swaps[k])) +
        ". The other way, the old part better by more than noise: " + ", ".join(f"{k.replace('swap-', '')} {signed(v)}" for k, v in better) +
        f". The full 2019 recipe ends at {f3(steps[0]['mean_val_loss'])} and the full 2026 recipe at {f3(steps[9]['mean_val_loss'])} (the ladder, 100M tokens): a gap of {abs(total):.3f}, of which the optimizer "
        f"(rung 7) is {abs(opt):.3f} ({100 * opt / total:.0f}%) and the eight changes to the block {abs(total - opt):.3f} ({100 * (total - opt) / total:.0f}%); rotary positions alone are "
        f"{abs(steps[2]['mean_val_loss'] - steps[1]['mean_val_loss']):.3f}. Rung by rung: {rungs}. " + recipe + " " + two_answers,
    ]
    if len(questions) != len(answers_in_order):
        raise SystemExit(f"the post has {len(questions)} RESULTS questions and this script answers {len(answers_in_order)}")
    answers = {f"line {line}: {q[:110]}": a for (line, q), a in zip(questions, answers_in_order)}
    answers["what the swaps show about the recipe"] = recipe + " " + two_answers
    out["results_lines"] = answers
    out["_supplement"] = {
        "seed_noise": noise, "seed_to_seed": pairs, "ladder_mean_loss_by_rung": {s["name"]: s["mean_val_loss"] for s in steps}, "ladder_seeds_by_rung": {s["name"]: s["val_loss"] for s in steps},
        "swaps_less_the_recipe": dict(costly), "swap_base": base, "warmup_runs": dict(zip((w for w, _ in WARM), warm)), "run_without_normalization": gone,
        "token_frequencies_alone": unigram, "uniform_guess": math.log(V),
        "gradient_last_over_first_mlp_output_matrix": ratio, "gradient_last_block_mlp_output_matrix_by_depth": lm,
        "hidden_units": {k: {f: zoo[k].get(f) for f in ("zero", "negligible", "dead", "kurtosis", "pre_negative")} for k in zoo},
    }
    if POST.exists():   # every placeholder of the post has its entry, word for word
        wanted = re.findall(r"@@(.+?)@@", POST.read_text())
        missing = [w for w in wanted if w not in out]
        if missing:
            raise SystemExit("no entry for: " + "; ".join(missing))
        out["_placeholders_of_the_post"] = len(wanted)
    size = write_json(Path(args.out), out, compact=False)
    print(f"wrote {args.out} ({size / 1e3:.1f} kB): {sum(1 for k in out if not k.startswith('_') and k != 'results_lines')} placeholders, {len(answers)} results lines")
    for k, v in answers.items():
        print("\n" + k + "\n  " + v)


if __name__ == "__main__":
    main()
