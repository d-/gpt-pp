"""The numbers that part 8 of the series ("What training learns") quotes from the lab, each keyed by the words of
its placeholder in the post, with a note on how it was computed.

    python -m gptpp.export.part_8_numbers            # writes results/interp/part-8.json; run again when a run ends

Nothing is trained or evaluated here: every value is read from what the lab has already written.

    Example 8.1.3   the export of fig-08-01 (`statistics-of-increasing-order.json`: pico's 21 log-spaced
                    checkpoints against the n-gram levels and predictors of TinyStories)
    Example 8.3.1   the modular-addition run (`$GPTPP_WORK/ckpt/grokking`: config.json and history.json)
    Example 8.6.2   the summaries of the runs `pico` and `gptpp-base` and the meta of their data; the entry for
                    gpt-pp stays empty until its run has written `summary.json`
    Section 8.2     (not placeholders: what the paragraph after fig-08-03 needs) the export of fig-08-03

A crossing of a level is located between the two checkpoints that bracket it, by a straight line in the loss
against the logarithm of the tokens seen; the bracket is given with it, since the run has no checkpoint between.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from .. import repo_dir, work_dir
from .common import DATA, write_json

POST = DATA.parent.parent.parent / "_posts" / "gpt" / "2026-11-14-gpt-08.md"


def crossing(tokens: list[int], loss: list[float], level: float) -> dict:
    """Where a falling loss first passes a level: the tokens seen, interpolated in log tokens between the two
    checkpoints that bracket it."""
    for i in range(1, len(loss)):
        if loss[i] < level <= loss[i - 1]:
            if tokens[i - 1] == 0:
                return {"tokens": tokens[i], "between": [tokens[i - 1], tokens[i]]}
            share = (loss[i - 1] - level) / (loss[i - 1] - loss[i])
            at = math.exp(math.log(tokens[i - 1]) + share * (math.log(tokens[i]) - math.log(tokens[i - 1])))
            return {"tokens": at, "between": [tokens[i - 1], tokens[i]]}
    raise ValueError("the loss never passes the level")


def first_at(steps: list[int], values: list[float], test) -> int:
    """The first logged step from which every later value passes `test`."""
    for i in range(len(values)):
        if all(test(v) for v in values[i:]):
            return steps[i]
    raise ValueError("never")


def example_8_1_3(site: Path) -> dict:
    export = json.loads((site / "statistics-of-increasing-order.json").read_text())
    run = next(r for r in export["runs"] if r["name"] == "pico")
    ln2, levels = math.log(2), {level["n"]: level for level in export["levels"]}
    out = {}
    for n, name in ((1, "unigram"), (2, "bigram")):
        level = levels[n]
        out[f"{name} level of TinyStories, bits per byte, as fig-08-01 draws it"] = {
            "value": round(level["bits_per_byte"], 3),
            "note": f"{export['levels_source']}; {level['bits_per_token']:.3f} bits per token = "
                    f"{level['bits_per_token'] * ln2:.3f} nats, divided by {export['bytes_per_token']:.4f} held-out bytes per token. "
                    f"The entropy of the maximum-likelihood {name} table of the whole training set is "
                    f"{export['predictors'][name + '_entropy']:.3f} bits per token "
                    f"({export['predictors'][name + '_entropy'] / export['bytes_per_token']:.3f} bits per byte)."}
        where = crossing(run["tokens"], run["loss"], level["bits_per_token"])
        out[f"training tokens at which pico's held-out loss crosses the {name} level"] = {
            "value": float(f"{where['tokens']:.2g}"), "between_checkpoints_at_tokens": where["between"],
            "note": f"pico's loss on 1,024 held-out windows at its 21 checkpoints is above the level at "
                    f"{where['between'][0]:,} tokens and below it at {where['between'][1]:,}; interpolated in log tokens "
                    f"the crossing is at {where['tokens']:,.0f} tokens (step {where['tokens'] / 65536:.1f} of 8,240)."}
        key = f"kl_{name}"
        i = min(range(len(run[key])), key=lambda j: run[key][j])
        out[f"minimum over checkpoints of pico's mean divergence from the {name} predictor, nats"] = {
            "value": round(run[key][i] * ln2, 3), "at_step": run["steps"][i], "at_tokens": run["tokens"][i],
            "note": f"{export['divergence']}. The smallest of the 21 checkpoints' means, {run[key][i]:.3f} bits = "
                    f"{run[key][i] * ln2:.3f} nats, is at step {run['steps'][i]} ({run['tokens'][i]:,} tokens), where the "
                    f"model's held-out loss is {run['loss'][i] * ln2:.3f} nats; at step 0 the divergence is {run[key][0] * ln2:.3f} nats "
                    f"and at the end {run[key][-1] * ln2:.3f}."}
    return out


def example_8_3_1(run: Path) -> dict:
    config, history = json.loads((run / "config.json").read_text()), json.loads((run / "history.json").read_text())
    steps, train, test = history["step"], history["train_acc"], history["test_acc"]
    full = first_at(steps, train, lambda v: v >= 1.0)
    passed = first_at(steps, test, lambda v: v > 0.99)
    i, j = steps.index(full), steps.index(passed)
    logged = f"measured at {len(steps)} log-spaced steps, so each epoch is known to the spacing of the log"
    return {
        "fraction of the 12,769 pairs used for training in our modular-addition run": {
            "value": config["fraction"], "note": f"{config['n_train']:,} of the {config['n_train'] + config['n_test']:,} pairs, chosen at random (data seed {config['data_seed']}); the other {config['n_test']:,} are the test set."},
        "weight decay of our modular-addition run": {
            "value": config["weight_decay"], "note": f"AdamW on every parameter, rate {config['lr']}, betas {config['betas']}, no schedule."},
        "number of epochs of our modular-addition run": {
            "value": config["steps"], "note": "every step takes the whole training set, so a step is an epoch."},
        "epoch at which training accuracy first reaches 100%": {
            "value": full, "note": f"the first logged step at which all {config['n_train']:,} training pairs are right, and they stay right; "
                                   f"the logged step before it is {steps[i - 1]} ({100 * train[i - 1]:.1f}%). {logged}. "
                                   f"(99% of the training pairs are right from step {first_at(steps, train, lambda v: v >= 0.99)}: the figure's readout quotes that step.)"},
        "test accuracy at that epoch": {
            "value": round(test[i], 4), "note": f"{100 * test[i]:.1f}% of the {config['n_test']:,} held-out pairs at step {full}."},
        "epoch at which test accuracy first exceeds 99%": {
            "value": passed, "note": f"{100 * test[j]:.1f}% at step {passed}; the logged step before it is {steps[j - 1]} ({100 * test[j - 1]:.1f}%). "
                                     f"{passed / full:.0f} times the epoch of 100% training accuracy."},
        "key frequencies of our model's embedding": {
            "value": sorted(config["key_frequencies"]),
            "note": f"the frequencies holding at least a tenth of the strongest one's power in the trained embedding; by falling power "
                    f"{config['key_frequencies']}. Together {100 * sum(history['embedding_power'][-1][k] for k in config['key_frequencies']) / sum(history['embedding_power'][-1]):.1f}% "
                    f"of the embedding's power at the end. Nanda et al. report 14, 35, 41, 42, 52 for their seed."},
    }


def megabytes(tokens: float, bytes_per_token: float, bits_per_byte: float) -> float:
    return tokens * bytes_per_token * bits_per_byte / 8 / 1e6


def example_8_6_2(work: Path) -> dict:
    out = {}
    pico = json.loads((work / "ckpt" / "pico" / "summary.json").read_text())
    stories = json.loads((work / "data" / "tinystories" / "bpe4k" / "meta.json").read_text())
    per_token = round(stories["bytes_per_token_val"], 4)
    out["pico's held-out bits per byte"] = {
        "value": round(pico["val_bpb"], 4),
        "note": f"the run's final evaluation on its fixed held-out windows (summary.json: {pico['val_loss']:.4f} nats per token, "
                f"divided by ln 2 and by {stories['bytes_per_token_val']:.4f} held-out bytes per token)."}
    out["megabytes of information in the TinyStories training text at pico's held-out rate"] = {
        "value": round(megabytes(stories["n_train_tokens"], per_token, pico["val_bpb"]), 1),
        "note": f"{stories['n_train_tokens']:,} tokens x {per_token} bytes x {pico['val_bpb']:.4f} bits per byte / 8 / 10^6."}
    base = work / "ckpt" / "gptpp-base"
    web = json.loads((work / "data" / "fineweb-edu" / "bpe32k" / "meta.json").read_text())
    keys = ("training tokens of gpt-pp's main run", "megabytes of information in gpt-pp's training text at gpt-pp's held-out rate")
    if (base / "summary.json").exists():
        summary = json.loads((base / "summary.json").read_text())
        out[keys[0]] = {"value": summary["tokens"], "note": f"summary.json of the run: {summary['steps']:,} steps on {summary['n_gpu']} cards, one pass."}
        out[keys[1]] = {
            "value": round(megabytes(summary["tokens"], web["bytes_per_token_val"], summary["val_bpb"]), 0),
            "note": f"{summary['tokens']:,} tokens x {web['bytes_per_token_val']:.4f} bytes x {summary['val_bpb']:.4f} bits per byte "
                    f"(the run's final held-out evaluation) / 8 / 10^6."}
    else:
        planned = json.loads((base / "config.json").read_text())["train_config"]["tokens"]
        for key in keys:
            out[key] = {"value": None, "note": f"PENDING: the run has not written summary.json yet (budget {planned:,.0f} tokens); run this command again when it has."}
    return out


def section_8_2(site: Path) -> dict:
    """What the paragraph after fig-08-03 states, from the figure's export of each run that has one."""
    out = {}
    for run in ("gptpp-base", "pico"):
        f = site / f"loss-against-position-{run}.json"
        if not f.exists():
            continue
        m = json.loads(f.read_text())
        t0, t1 = m["score_positions"]
        ratio = [b / a for a, b in zip(m["repeat_first"], m["repeat_second"])]
        apart = next((i for i in range(len(ratio)) if all(r < 0.5 for r in ratio[i:])), None)   # the copy costs under half, for good
        out[run] = {
            "score_positions": [t0, t1], "checkpoints": len(m["steps"]), "last_step": m["steps"][-1], "last_tokens": m["tokens"][-1],
            "score_on_text_at_the_last_checkpoint_bits": round(m["text_early"][-1] - m["text_late"][-1], 3),
            "score_on_text_at_the_last_checkpoint_nats": round((m["text_early"][-1] - m["text_late"][-1]) * math.log(2), 3),
            "loss_on_the_first_copy_bits": [round(m["repeat_first"][0], 2), round(m["repeat_first"][-1], 2)],
            "loss_on_the_second_copy_bits": [round(m["repeat_second"][0], 2), round(m["repeat_second"][-1], 2)],
            "tokens_from_which_the_second_copy_costs_less_than_half_the_first": None if apart is None else m["tokens"][apart],
            "step_from_which_the_second_copy_costs_less_than_half_the_first": None if apart is None else m["steps"][apart],
            "note": f"{m['windows']} windows of held-out text of {m['context']} tokens, each inside one document; the loss near "
                    f"position t is the mean over the positions within a quarter of t (a single position over {m['windows']} windows "
                    f"is too noisy to subtract). {m['sequences']} sequences of {m['half']} random tokens written twice after <|bos|>, "
                    f"uniform over the {m['pool']:,} tokens that occur in the training set; the second copy is scored without its "
                    f"first token. Bits per token; first and last checkpoint.",
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(repo_dir() / "results" / "interp" / "part-8.json"))
    args = ap.parse_args()
    work = work_dir()
    numbers = {}
    numbers.update(example_8_1_3(DATA))
    numbers.update(example_8_3_1(work / "ckpt" / "grokking"))
    numbers.update(example_8_6_2(work))
    out = {"post": "Keys and values, 8. What training learns", "placeholders": numbers, "section_8_2_after_fig_08_03": section_8_2(DATA)}
    if POST.exists():                                    # which placeholders of the post this file answers
        asked = re.findall(r"@@(.+?)@@", POST.read_text())
        out["placeholders_of_the_post_not_computed_here"] = [a for a in asked if a not in numbers]
        missing = [k for k in numbers if k not in asked]
        if missing:
            out["computed_here_but_not_asked_by_the_post"] = missing
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, out, compact=False)
    for key, entry in numbers.items():
        print(f"{entry['value']!s:>28}  {key}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
