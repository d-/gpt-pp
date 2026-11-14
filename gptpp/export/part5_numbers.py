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
    the post itself                the exact words of its placeholders `@@...@@`, while it still has them

It writes one JSON object: the placeholders' words as keys, each with a value, its unit, a text ready for the page
and a note on how it was computed; `results_lines`, the measured answers to the six questions the draft asked in its
`<!-- RESULTS: ... -->` comments (keyed by the post's line while the post still has the comments, by section once
the writer has taken them out); and `_supplement`, the numbers behind those answers. It stops if a placeholder
still in the post has no entry.

Losses are held-out losses in nats per token at the end of training (the trainer's own evaluation on 1,048,576
held-out tokens). A rung of the ladder is the mean of its two seeds; the swaps and the warmup runs are one seed each.

The seed yardstick. Every rung of the ladder was trained twice. The root mean square of the differences between the
two seeds of a rung estimates the standard deviation of the difference between two runs that differ only by seed.
A difference between two single runs under twice that is "inside seed noise"; three times it or more is "clear";
between the two it is "at the edge". The swaps are single runs at 50M tokens, where no pair of seeds was run: the
yardstick is borrowed from the ladder (100M tokens).
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
RUNGS = {"gpt2-2019": "the 2019 recipe", "no-biases": "no biases", "rotary": "rotary positions", "rmsnorm": "RMS norm without parameters",
         "relu2": "squared ReLU", "qk-norm": "QK normalization", "zero-init": "zero-initialized writers", "muon": "the training recipe of 2026",
         "untied": "untied tables with a normalized embedding", "softcap": "the soft cap"}
QUESTIONS = (  # the six questions of the draft, in the post's order: the section each stood in, and its words
    ("Section 5.2, where the norm sits", "whether the post-norm profile rises toward the output while the pre-norm profile is flat, how the top-layer gradient changes with depth, and what warmup does to each"),
    ("Section 5.2, which norm", "whether layer norm, RMS norm and parameter-free RMS norm are within seed noise of each other, where dynamic tanh lands, and whether the model without normalization trains"),
    ("Section 5.3, the activations", "the order of the six activations, which differences exceed the gap between seeds, and whether the second-order units lead by 0.02 to 0.05 nats"),
    ("Section 5.4, neurons as memories", "whether lower-layer triggers share a last token and upper-layer ones a meaning, and whether a value's top tokens continue its key's triggers"),
    ("Section 5.5, the initializations", "whether the initializations differ at the end or only early, at 8 layers"),
    ("Section 5.6, the table", "the swaps that cost the most and those within seed noise; the 2019 and the 2026 recipe, and the shares of the block and of the optimizer"),
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
    R = round_sig(collect(repo_dir() / "results" / "runs", ["ladder-", "swap-", "warm-", "split-"]))
    runs, steps, spread = R["runs"], R["ladder"]["steps"], R["ladder"]["spread"]
    grad = json.loads((data / "gradient-by-depth.json").read_text())
    zoo = {a["act"]: a for a in json.loads((data / "activation-zoo.json").read_text())["acts"]}
    memories = json.loads((repo_dir() / "results" / "interp" / "lens-patching-memories.json").read_text())
    small, V = grad["training"]["runs"], grad["training"]["model"]["vocab_size"]      # the two-by-two in steps of 8,192 tokens
    trained = grad["training"]["ladder_step"]["runs"]                                # the same four at the ladder's step
    rate, warmup = trained[0]["lr"], round(max(r["warmup_tokens"] for r in trained) / 1e6)
    step_small, step_ladder = grad["training"]["step_tokens"], grad["training"]["ladder_step"]["step_tokens"]
    warm_small = next(r for r in small if r["warmup_tokens"] > 0)
    warm_ladder = next(r for r in trained if r["warmup_tokens"] > 0)
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

    # ---- the seed yardstick
    if not spread["rungs"]:
        raise SystemExit("no rung of the ladder has two seeds yet: the seed noise that every entry is judged against is not known")
    rms = spread["rms_abs_diff"]
    noise, clear = 2 * rms, 3 * rms
    pairs = {s["name"]: abs(s["val_loss"][0] - s["val_loss"][1]) for s in steps if len(s["val_loss"]) > 1 and None not in s["val_loss"]}
    widest = sorted(pairs.items(), key=lambda kv: -kv[1])
    seeds_text = (f"the two seeds of a rung differ by {min(pairs.values()):.3f} to {max(pairs.values()):.3f} nats over the {len(pairs)} rungs of the ladder"
                  + (f": {widest[2][1]:.3f} or less on {len(pairs) - 2} of them, {widest[1][1]:.3f} on {RUNGS[widest[1][0]]} and {widest[0][1]:.3f} on {RUNGS[widest[0][0]]}" if len(pairs) > 3 else "")
                  + f"; root mean square {rms:.3f}")
    muon = [pairs[k] for k in ("muon", "untied", "softcap") if k in pairs]
    L = [s["mean_val_loss"] for s in steps]

    def by_seed(k: int) -> list[float]:
        """What change k did in each seed that ran both rungs."""
        return [b - a for a, b in zip(steps[k - 1]["val_loss"], steps[k]["val_loss"]) if a is not None and b is not None]

    def seeds_words(k: int) -> str:
        d = by_seed(k)
        return f"{signed(d[0])} in one seed, {signed(d[1])} in the other" if len(d) > 1 else "one seed"

    def judged(d: float) -> str:
        if abs(d) < noise:
            return "inside seed noise"
        return ("worse" if d > 0 else "better") + (" by more than seed noise" if abs(d) >= clear else ", at the edge of seed noise")

    def standing(d: float) -> str:
        if abs(d) < noise:
            return f"inside seed noise (under {noise:.3f} between two single runs)"
        if abs(d) < clear:
            return f"at the edge of seed noise (between {noise:.3f} and {clear:.3f} for two single runs): probably real, not settled by one seed"
        return f"clear of seed noise ({clear:.3f} or more between two single runs)"

    out: dict = {
        "_post": "_posts/gpt/2026-11-14-gpt-05.md",
        "_written_by": "python -m gptpp.export.part5_numbers (from results/runs, assets/data/gpt/gradient-by-depth.json and activation-zoo.json, results/interp/lens-patching-memories.json)",
        "_note": "one entry for each placeholder @@...@@ of the post, keyed by its exact words: value, unit, a text for the page, and how it was computed. "
                 "Losses are held-out losses in nats per token at the end of training. A rung of the ladder is the mean of its two seeds; the swaps and the warmup runs are one seed each. "
                 f"The seed yardstick: {seeds_text}. A difference between two single runs under {noise:.3f} is inside seed noise, {clear:.3f} or more is clear, between the two it is at the edge. "
                 "The swaps are single runs at 50M tokens, where no pair of seeds was run, so the yardstick is borrowed from the ladder (100M tokens). "
                 "Numbers are rounded to six significant digits first, as in the files the figures read.",
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
    by_step = [r["near_frequencies_at_step"] for r in trained if r["placement"] == "post" and r.get("near_frequencies_at_step")]
    fell = f", within {max(by_step)} steps" if by_step else ""
    out["held-out loss of the 8-layer ladder model in post-norm with warmup, post-norm without warmup, pre-norm with warmup and pre-norm without warmup"] = {
        "value": warm, "unit": "nats per token",
        "text": f"{f3(warm[0])} and {f3(warm[1])} for post-norm with and without warmup, {f3(warm[2])} and {f3(warm[3])} for pre-norm",
        "note": f"runs {', '.join(run for _, run in WARM)}: the 2019 recipe (AdamW, peak rate {rate}, cosine decay), {round(runs['warm-2019-base']['tokens'] / 1e6)}M tokens, one seed each; "
                f"the warmup is {warmup}M tokens, {warm_ladder['warmup_steps']} steps of {step_ladder:,} tokens. THIS IS THE LADDER'S STEP SIZE, at which both post-norm runs collapsed to "
                f"predicting token frequencies ({scale}){fell}: a collapse, with a warmup and without, not evidence about warmup and not evidence that post-norm cannot train. "
                f"The comparison the page reports is the two-by-two in steps of {step_small:,} tokens (the next entry). Pre-norm at this step is {warm[3] - warm[2]:.2f} nats behind without "
                "warmup, at the rate that was chosen with warmup.",
    }
    two = {(r["placement"], r["warmup_tokens"] > 0): need(r["run"]) for r in small}
    out["held-out loss of the 8-layer model in the 2019 recipe trained in steps of 8,192 tokens: pre-norm and post-norm, with a warmup of 8M tokens and without"] = {
        "value": {"pre": [two[("pre", True)], two[("pre", False)]], "post": [two[("post", True)], two[("post", False)]]}, "unit": "nats per token",
        "text": f"pre-norm {f3(two[('pre', True)])} with a warmup and {f3(two[('pre', False)])} without; post-norm {f3(two[('post', True)])} with a warmup and {f3(two[('post', False)])} without",
        "note": f"runs {', '.join(r['run'] for r in small)}: the 2019 recipe (AdamW, peak rate {warm_small['lr']}, cosine decay) in steps of {step_small:,} tokens, {M}M tokens, one seed each; "
                f"the warmup of {round(warm_small['warmup_tokens'] / 1e6)}M tokens is {warm_small['warmup_steps']} steps. With a warmup post-norm trains and is "
                f"{two[('pre', True)] - two[('post', True)]:.3f} ahead of pre-norm; without one it collapses to token frequencies ({scale}) where pre-norm loses "
                f"{two[('pre', False)] - two[('pre', True)]:.2f}. At the ladder's step of {step_ladder:,} tokens and rate {rate} (the same warmup: {warm_ladder['warmup_steps']} steps) post-norm collapsed "
                f"with a warmup too ({f3(warm[0])}, {f3(warm[1])}){fell}; pre-norm reached {f3(warm[2])} and {f3(warm[3])}.",
    }
    split = R["ladder"]["split"]
    if split:
        between = ", ".join("%s (%.4f)" % (name, runs[name]["val_loss"]) for name in split["runs"])
        out["the ladder's step from rung 6 to rung 7 in one seed, split: the rest of the training recipe, and Muon in place of AdamW"] = {
            "value": {k: split[k] for k in ("seed", "before", "with_adamw", "after", "rest_of_recipe", "rule")}, "unit": "nats per token",
            "text": f"in seed {split['seed']} the step is {abs(split['after'] - split['before']):.3f} nats ({f3(split['before'])} to {f3(split['after'])}); with AdamW in place of Muon under the 2026 "
                    f"training recipe the same model ends at {f3(split['with_adamw'])}: {abs(split['rest_of_recipe']):.3f} for the rest of the recipe and {abs(split['rule']):.3f} for Muon in place of AdamW",
            "note": f"runs {between}: the rung-6 model under the 2026 training recipe with AdamW on the hidden matrices in place of Muon (its own warmup of 8M tokens and clipping at norm one, "
                    "two rates; the better is taken), 100M tokens, one seed. Rung 7 changes the whole training recipe (Muon, the tables' rates, beta1, the decay, the clipping, the warmup, the "
                    f"schedule), never Muon alone. One seed: the two seeds of a rung trained with AdamW have differed by up to {max(pairs.values()):.3f}.",
        }

    # ---- Section 5.2: which norm
    gone = failed_run("swap-norm-none")
    norm_values = [loss(run) for _, run in NORMS]
    d_ln, d_gain, d_dyt = need("swap-norm-layernorm") - base, need("swap-norm-rmsnorm-gain") - base, need("swap-norm-dyt") - base
    out["held-out loss of the 8-layer ladder model with layer norm, with RMS norm with a gain, with parameter-free RMS norm, with dynamic tanh, and with no normalization"] = {
        "value": norm_values, "unit": "nats per token",
        "text": f"{f3(need('swap-norm-layernorm'))}, {f3(need('swap-norm-rmsnorm-gain'))}, {f3(base)} and {f3(need('swap-norm-dyt'))}; without normalization the run did not finish: "
                + (f"its loss was {gone['last_finite_loss']:.2f} at step {gone['last_finite_step']} and no longer finite at step {gone['not_finite_at_step']}" if gone and norm_values[4] is None
                   else f"{f3(norm_values[4])}" if norm_values[4] is not None else "its loss stopped being finite"),
        "note": f"the 2026 recipe with only the norm changed, {M}M tokens, one seed each (swap-norm-layernorm, swap-norm-rmsnorm-gain, swap-base, swap-norm-dyt, swap-norm-none; the last also "
                "drops the norm on the embedding). Layer norm here has its gain and no bias (the recipe has no biases). Against the parameter-free norm of the recipe: layer norm "
                f"{signed(d_ln)} ({judged(d_ln)}), RMS norm with a gain {signed(d_gain)} ({judged(d_gain)}), dynamic tanh {signed(d_dyt)} ({judged(d_dyt)}); inside seed noise is under "
                f"{noise:.3f}, clear is {clear:.3f} or more. The model without normalization diverges although its projections start at zero.",
    }

    # ---- Section 5.3: activations
    acts = {key: need(run) for key, _, run in ACTS}
    out["held-out loss of the 8-layer ladder model with ReLU, GELU, SiLU, squared ReLU, SwiGLU and GeGLU at matched parameters"] = {
        "value": [acts[key] for key, _, _ in ACTS], "unit": "nats per token",
        "text": ", ".join(f"{f3(acts[key])} ({name})" for key, name, _ in ACTS[:-1]) + f" and {f3(acts['geglu'])} (GeGLU)",
        "note": f"the 2026 recipe with only the activation changed, {M}M tokens, one seed each; the gated two have hidden width {zoo['swiglu'].get('d_mlp')} (two thirds of "
                f"{zoo['relu2'].get('d_mlp')}) so that every model has {runs['swap-base']['params']['total']:,} parameters. Against squared ReLU: "
                + ", ".join(f"{name} {signed(acts[key] - base)} ({judged(acts[key] - base)})" for key, name, _ in ACTS if key != "relu2") + f"; inside seed noise is under {noise:.3f}.",
    }
    own = [s for s in steps if s["name"] == "softcap"][0]
    both = len(own["val_loss"]) > 1 and None not in own["val_loss"]
    out["difference in held-out loss between two seeds of the 8-layer ladder model in the 2026 recipe"] = {
        "value": abs(own["val_loss"][0] - own["val_loss"][1]) if both else None, "unit": "nats per token",
        "two_seeds_of_the_2026_recipe": both,
        "text": (f"{abs(own['val_loss'][0] - own['val_loss'][1]):.3f} nats; over the whole ladder {seeds_text}" if both else seeds_text + "; the second seed of the 2026 recipe itself has not finished"),
        "note": "the last rung of the ladder (ladder-9-softcap, the 2026 recipe, 100M tokens), seeds 0 and 1. The pair of the recipe alone understates the noise: by rung, "
                + ", ".join(f"{k} {v:.4f}" for k, v in pairs.items()) + f". Use the yardstick of all the pairs: under {noise:.3f} between two single runs is noise, {clear:.3f} or more is clear. "
                "The swaps are one seed each at 50M tokens; no pair of seeds was run at that budget.",
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
                f"at 100M tokens, going from the scaled writers to zero (rung 5 to rung 6) changed the loss by {signed(L[6] - L[5])} ({seeds_words(6)}).",
    }

    # ---- Section 5.6: the table, one component swapped back
    for words, run, what in TABLE:
        d = need(run) - base
        out[words] = {"value": d, "unit": "nats per token", "text": signed(d),
                      "note": f"{run} less swap-base: the 2026 recipe with {what}; {M}M tokens, one seed each; {judged(d)} (inside under {noise:.3f}, clear from {clear:.3f})."}

    # ---- the measured answers to the six questions of the draft
    post = POST.read_text().splitlines() if POST.exists() else []
    asked = [(i + 1, re.search(r"<!-- RESULTS: (.*?) -->", line).group(1)) for i, line in enumerate(post) if re.search(r"<!-- RESULTS: (.*?) -->", line)]
    g, G = grad["init"]["inits"]["gpt2"], grad["init"]["inits"]["glorot"]
    uneven = max(abs(G["layers"]["post"][str(d)]["all"][-1] / G["layers"]["post"][str(d)]["all"][0] - 1) for d in depths)

    def ends(init: dict, place: str, d: int) -> str:
        c = init["layers"][place][str(d)]
        return f"{c['all'][0]:.2f} to {c['all'][-1]:.2f}"

    lm = {init: {place: [grad["init"]["inits"][init]["last_mlp"][place][str(d)] for d in depths] for place in ("post", "pre")} for init in ("gpt2", "glorot")}
    order = sorted(ACTS, key=lambda a: acts[a[0]])
    against = [(name, acts[key] - base) for key, name, _ in order if key != "relu2"]
    second, first = [acts[k] for k in ("relu2", "swiglu", "geglu")], [acts[k] for k in ("relu", "gelu", "silu")]
    mem = memories["more"]["neurons-as-memories"]
    by, chance = mem["by_block"], mem["chance"]
    hi_same = max(range(len(by["block"])), key=lambda i: by["same_last"][i])
    hi_agree = max(range(len(by["block"])), key=lambda i: by["agreement"][i])
    swaps = {name: r["val_loss"] - base for name, r in runs.items() if r["family"] == "swap" and name != "swap-base"}
    costly = sorted(swaps.items(), key=lambda kv: -kv[1])

    def listed(items) -> str:
        return ", ".join(f"{k.replace('swap-', '')} {signed(v)}" for k, v in items) or "none"

    total, opt = L[9] - L[0], L[7] - L[6]
    rungs = ", ".join(f"{RUNGS[s['name']]} {signed(L[s['k']] - L[s['k'] - 1])} ({seeds_words(s['k'])})" for s in steps[1:])
    split_words = (f" Rung 7 taken apart in seed {split['seed']}: of the {abs(split['after'] - split['before']):.3f} nats, {abs(split['rest_of_recipe']):.3f} is the rest of the training recipe and "
                   f"{abs(split['rule']):.3f} Muon in place of AdamW (the 2026 training recipe with AdamW at its better rate ends at {f3(split['with_adamw'])}).") if split else ""
    curve = {name: dict(zip(runs[name]["curve"]["tokens"], runs[name]["curve"]["val_loss"])) for name in ("swap-init-gpt2", "swap-base")}
    early = sorted(set(curve["swap-init-gpt2"]) & set(curve["swap-base"]))[1:3]
    simpler = [swaps["swap-norm-layernorm"], swaps["swap-norm-rmsnorm-gain"], swaps["swap-bias"]]
    recipe = (f"At 8 layers and {M}M tokens three simplifications of the 2026 block measure slightly worse than what they replaced: layer norm with its gain put back is "
              f"{abs(simpler[0]):.3f} better, RMS norm with a gain {abs(simpler[1]):.3f}, biases {abs(simpler[2]):.3f}; one seed each, {standing(min(simpler, key=abs))} "
              f"(the largest difference seen between the two seeds of a rung is {max(pairs.values()):.3f}; the three rungs trained with Muon differ by {max(muon):.4f} at most). "
              f"GELU ties squared ReLU ({signed(swaps['swap-act-gelu'])}). On the ladder, removing the biases is worth nothing ({signed(L[1] - L[0], 4)} on the means: {seeds_words(1)}), "
              f"and the parameter-free norm costs {signed(L[3] - L[2])} ({seeds_words(3)}). What is clear: rotary positions ({signed(swaps['swap-pos-learned'])} with a learned table), the training recipe of 2026 "
              f"({signed(opt)} on the ladder), query-key normalization ({signed(swaps['swap-attn-no-qk-norm'])} without) and pre-norm ({signed(swaps['swap-norm-post'])} with post-norm and layer norm, "
              f"{signed(need('swap-norm-post') - need('swap-norm-layernorm'))} against the same layer norm before the sublayers). The norm on the embedding: "
              f"{signed(swaps['swap-no-embed-norm'])} without it ({judged(swaps['swap-no-embed-norm'])}), {signed(swaps['swap-tied'])} when it goes together with the second vocabulary matrix (the tied run, "
              f"{(runs['swap-base']['params']['total'] - runs['swap-tied']['params']['total']) / 1e6:.1f}M parameters fewer).")
    two_answers = (f"The ladder and the swaps ask different questions and twice give different answers: squared ReLU gained {abs(L[4] - L[3]):.3f} on the ladder "
                   f"({seeds_words(4)}; under AdamW, right after the norm lost its gain) and GELU put back into the finished recipe costs {signed(swaps['swap-act-gelu'])}; writers at zero gained "
                   f"{abs(L[6] - L[5]):.3f} on the ladder ({seeds_words(6)}) and the scaled rule put back costs {signed(swaps['swap-init-gpt2'])}. "
                   f"Query-key normalization goes the other way: {signed(L[5] - L[4])} on the ladder ({seeds_words(5)}), {signed(swaps['swap-attn-no-qk-norm'])} when taken out of the finished recipe.")
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
        f"{lm['gpt2']['pre'][2] / lm['gpt2']['pre'][0]:.2f}). Training at 8 layers, the 2019 recipe, {M}M tokens, one seed each, in steps of {step_small:,} tokens at rate {warm_small['lr']} "
        f"(the warmup of {round(warm_small['warmup_tokens'] / 1e6)}M tokens is {warm_small['warmup_steps']} steps): pre-norm {f3(two[('pre', True)])} with a warmup and {f3(two[('pre', False)])} without; "
        f"post-norm {f3(two[('post', True)])} with a warmup and {f3(two[('post', False)])} without. With a warmup post-norm trains and is {two[('pre', True)] - two[('post', True)]:.3f} ahead; without one it "
        f"collapses to token frequencies ({scale}) where pre-norm loses {two[('pre', False)] - two[('pre', True)]:.2f}. At the ladder's step of {step_ladder:,} tokens and rate {rate}, where the same warmup is "
        f"{warm_ladder['warmup_steps']} steps, post-norm collapsed to token frequencies with a warmup too ({f3(warm[0])}, {f3(warm[1])}){fell} and pre-norm reached {f3(warm[2])} and {f3(warm[3])}: those two "
        f"post-norm runs are a collapse, not evidence about warmup, and nothing here says that post-norm cannot train. In the 2026 recipe (Muon, writers at zero, {M}M tokens) post-norm trains too: "
        f"{f3(need('swap-norm-post'))} against {f3(need('swap-norm-layernorm'))} for the same layer norm placed before the sublayers.",
        # which norm
        f"Layer norm ({f3(need('swap-norm-layernorm'))}) and RMS norm with a gain ({f3(need('swap-norm-rmsnorm-gain'))}) are within {abs(d_ln - d_gain):.3f} of each other: centering does not matter. "
        f"Both are better than the parameter-free RMS norm of the recipe ({f3(base)}), by {abs(d_ln):.3f} and {abs(d_gain):.3f}: {standing(min(d_ln, d_gain, key=abs))}. "
        f"On the ladder, where a rung is the mean of two seeds, the same step (rung 3, layer norm with its gain to parameter-free RMS norm) cost {signed(L[3] - L[2])} ({seeds_words(3)}): "
        f"the learned gain helps at this size. Dynamic tanh lands at {f3(need('swap-norm-dyt'))}, {signed(d_dyt)} ({judged(d_dyt)}). Without normalization the model does not train even with "
        "zero-initialized projections: "
        + (f"the loss was {gone['last_finite_loss']:.2f} at step {gone['last_finite_step']} and not finite at step {gone['not_finite_at_step']}." if gone else "the run did not finish.") + " One seed each for the swaps.",
        # activations
        "Lowest to highest loss: " + ", ".join(f"{name} {f3(acts[key])}" for key, name, _ in order) + f". One run each, so the order is not a ranking. Against squared ReLU, inside seed noise (under {noise:.3f}): "
        + (", ".join(f"{n} {signed(d)}" for n, d in against if abs(d) < noise) or "none") + "; at the edge: " + (", ".join(f"{n} {signed(d)}" for n, d in against if noise <= abs(d) < clear) or "none")
        + f"; clear ({clear:.3f} or more): " + (", ".join(f"{n} {signed(d)}" for n, d in against if abs(d) >= clear) or "none") + ". "
        f"The second-order units average {sum(second) / 3:.3f} and the first-order ones {sum(first) / 3:.3f}: {sum(first) / 3 - sum(second) / 3:.3f} apart, mostly ReLU's doing; GELU ties squared ReLU. "
        f"The 0.02 to 0.05 of the literature is not there at 8 layers and {M}M tokens: close to a null result. What does differ is what the units do: "
        f"{100 * zoo['relu2']['zero']:.0f}% of squared ReLU's activations and {100 * zoo['relu']['zero']:.0f}% of ReLU's are exact zeros, GELU has {100 * zoo['gelu']['zero']:.0f}%; "
        f"the kurtosis of the activations is {zoo['relu2']['kurtosis']:.0f} for squared ReLU, {zoo['swiglu']['kurtosis']:.0f} and {zoo['geglu']['kurtosis']:.0f} for the two gated units and "
        f"{zoo['gelu']['kurtosis']:.0f} for GELU; dead units are rare in all six: none in {sum(a['dead'] == 0 for a in zoo.values())} of them and "
        f"{100 * max(a['dead'] for a in zoo.values()):.2f}% of the units at most ({max(zoo.values(), key=lambda a: a['dead'])['act']}).",
        # neurons
        f"In part. The share of a neuron's {mem['triggers']} strongest triggers that end in its commonest last token is highest in block {by['block'][hi_same]} ({by['same_last'][hi_same]:.2f}) and lowest in block "
        f"{by['block'][-1]} ({by['same_last'][-1]:.2f}); random positions give {chance['same_last']:.2f}. Lower layers are the more shallow, as in the 2021 table, but by a modest margin. The top token of a "
        f"value is the token after its key's strongest trigger for {100 * by['agreement'][0]:.0f}% of the neurons of block 1, {100 * by['agreement'][hi_agree]:.0f}% in block {by['block'][hi_agree]} and "
        f"{100 * by['agreement'][-1]:.0f}% in block 12 (chance {100 * chance['agreement']:.3f}%): values continue their triggers in the upper half, not at the very top. Coefficients are non-negative "
        f"(squared ReLU). {mem['sample']} neurons sampled a block, the {mem['triggers']} strongest triggers of each.",
        # initializations
        f"Only two of the three were run. With the writers scaled by 1/sqrt(2L) the 2026 recipe ends at {scaled:.4f}, with the writers at zero at {base:.4f}: {abs(scaled - base):.4f} apart, {judged(scaled - base)}. "
        + "During training the two cross: " + ", ".join(f"{signed(curve['swap-init-gpt2'][t] - curve['swap-base'][t])} at {round(t / 1e6)}M tokens" for t in early) +
        f" (scaled less zero). No difference at the end and none that lasts early, at 8 layers. Plain 0.02 on every matrix was not run, so the sources' claim that the scaled rule clearly beats it is "
        f"untested here. On the ladder, under AdamW at 100M tokens, zero writers were {signed(L[6] - L[5])} against the scaled rule ({seeds_words(6)}): there the change is clear.",
        # the table
        f"The 2026 recipe with one old part put back ({M}M tokens, one seed each; inside seed noise under {noise:.3f}, clear from {clear:.3f}). The rows of the post's table that cost the most: "
        + ", ".join(f"{k.replace(': change in held-out loss', '').replace('ladder, ', '')} {signed(v)}" for k, v in sorted(((w, need(r) - base) for w, r, _ in TABLE), key=lambda kv: -kv[1])[:2])
        + ". Every swap run, by standing. Clear, the old part costs: " + listed((k, v) for k, v in costly if v >= clear) + (", and without normalization the run diverged" if gone else "")
        + ". At the edge: " + listed((k, v) for k, v in costly if noise <= abs(v) < clear) + ". Inside seed noise: " + listed((k, v) for k, v in costly if abs(v) < noise)
        + ". Clear, the old part is better: " + listed((k, v) for k, v in costly if v <= -clear)
        + f". The full 2019 recipe ends at {f3(L[0])} and the full 2026 recipe at {f3(L[9])} (the ladder, 100M tokens, means of two seeds): a gap of {abs(total):.3f}, of which the training recipe of 2026 "
        f"(rung 7: Muon on the hidden matrices, with its rates and its schedule) is {abs(opt):.3f} ({100 * opt / total:.0f}%) and the eight changes to the block {abs(total - opt):.3f} ({100 * (total - opt) / total:.0f}%); rotary positions alone are "
        f"{abs(L[2] - L[1]):.3f}. Rung by rung, the step of the means and of each seed: {rungs}. " + recipe + " " + two_answers + split_words,
    ]
    if asked and len(asked) != len(answers_in_order):
        raise SystemExit(f"the post has {len(asked)} RESULTS questions and this script answers {len(answers_in_order)}")
    if asked:   # the draft still has its questions: key by its lines
        answers = {f"line {line}: {q[:110]}": a for (line, q), a in zip(asked, answers_in_order)}
    else:       # the writer has taken them out: key by section
        answers = {f"{section}: {q}": a for (section, q), a in zip(QUESTIONS, answers_in_order)}
    answers["what the swaps show about the recipe"] = recipe + " " + two_answers + split_words
    out["results_lines"] = answers
    out["_supplement"] = {
        "seed_yardstick": {"rungs": len(pairs), "root_mean_square": rms, "inside_noise_under": noise, "clear_from": clear, "largest": max(pairs.values()), "smallest": min(pairs.values())},
        "seed_to_seed": pairs, "ladder_mean_loss_by_rung": {s["name"]: s["mean_val_loss"] for s in steps}, "ladder_seeds_by_rung": {s["name"]: s["val_loss"] for s in steps},
        "ladder_step_by_seed": {s["name"]: by_seed(s["k"]) for s in steps[1:]},
        "swaps_less_the_recipe": dict(costly), "swap_base": base, "warmup_runs": dict(zip((w for w, _ in WARM), warm)), "run_without_normalization": gone,
        "token_frequencies_alone": unigram, "uniform_guess": math.log(V), "rung_7_split": split,
        "two_by_two_in_small_steps": {"step_tokens": step_small, "rate": warm_small["lr"], "warmup_steps": warm_small["warmup_steps"],
                                      "pre": [two[("pre", True)], two[("pre", False)]], "post": [two[("post", True)], two[("post", False)]]},
        "same_four_at_the_ladders_step": {"step_tokens": step_ladder, "rate": rate, "warmup_steps": warm_ladder["warmup_steps"], "pre": [warm[2], warm[3]], "post": [warm[0], warm[1]]},
        "gradient_last_over_first_mlp_output_matrix": ratio, "gradient_last_block_mlp_output_matrix_by_depth": lm,
        "hidden_units": {k: {f: zoo[k].get(f) for f in ("zero", "negligible", "dead", "kurtosis", "pre_negative")} for k in zoo},
    }
    if POST.exists():   # every placeholder still in the post has its entry, word for word
        wanted = re.findall(r"@@(.+?)@@", POST.read_text())
        missing = [w for w in wanted if w not in out]
        if missing:
            raise SystemExit("no entry for: " + "; ".join(missing))
        out["_placeholders_still_in_the_post"] = len(wanted)
    size = write_json(Path(args.out), out, compact=False)
    print(f"wrote {args.out} ({size / 1e3:.1f} kB): {sum(1 for k in out if not k.startswith('_') and k != 'results_lines')} placeholders, {len(answers)} results lines")
    for k, v in answers.items():
        print("\n" + k + "\n  " + v)


if __name__ == "__main__":
    main()
