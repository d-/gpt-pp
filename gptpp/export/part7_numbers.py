"""Collect the numbers that the post of part 7 ("Training gpt-pp") quotes into one file, each with a note on how it
was computed: `results/interp/part-7.json`.

    python -m gptpp.export.part7_numbers [--post <the post's Markdown file>] [--out results/interp/part-7.json]

Nothing is measured here. The script reads what the trainer, the evaluation, the measuring scripts and the figures'
exporters wrote, fills what exists, and lists what is still missing with the reason, so that it can be run after
every step of the queue:

    results/runs/<run>/summary.json            the trainer's own summary of a run (main run, race, grid, ladder)
    results/eval/<run>.json                    `gptpp.evaluate` and `gptpp.benchmarks` on a model of ours
    results/interp/throughput-waterfall.json   `gptpp.export.where_the_time_goes --measure`
    results/interp/part-7-measured.json        `gptpp.export.part7_measure` (windows of 512, writing speed, samples)
    <site>/assets/data/gpt/scaling-surface.json   `gptpp.export.scaling_surface` (the fit under each count)

The post marks every number it still needs as `@@words@@`. The file answers each occurrence, in the order of the
post, keyed by the placeholder's own words:

    placeholders[k]   {n (its place in the post, from 1), line, words, value (the text to put in its place, or null),
                       raw (the number behind it), note (how it was computed, from which file), missing (why there is
                       no value yet; absent when filled)}
    by_words          {words: value} for words that occur once, {words: [values in the order of the post]} for words
                      that occur several times (a table row repeats "BLiMP", "its bits per byte", "beta")
    numbers           the same numbers and their neighbors as structured data with notes, by subject: main_run,
                      waterfall, race, eval (bits per byte with standard errors on the four held-out sets, the nine
                      benchmarks and the centered mean, for gpt-pp, the 2026 race run and the best 2019 race run),
                      hosted_subset, scaling (the fit under each count, with its caveats), writing, samples, accounting
    missing           the words that have no value yet, with the reason
    filled, total     counts

Formats: tokens per second as whole numbers with thousands separators; hours to two decimals; losses in nats per
token and bits per byte to three decimals; bits per byte "value ± standard error"; accuracies in percent to one
decimal; exponents to four significant digits with a 95% percentile interval over resampled runs.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

from .. import repo_dir, work_dir
from .benchmark_table import SETS, TASKS, centered, on_subset
from .common import CKPT, SITE

POST = SITE / "_posts" / "gpt" / "2026-11-14-gpt-07.md"
OUT = repo_dir() / "results" / "interp" / "part-7.json"
RACE_2019 = ["race-2019-lr6e-4-100M", "race-2019-lr1.2e-3-100M", "race-2019-lr2.4e-3-100M", "race-2019-lr1.2e-3-w8M-100M", "race-2019-lr2.4e-3-w8M-100M",
             "race-2019-lr4.8e-3-100M"]   # three rates at the preset's warmup, two of them with a warmup of 8M tokens, and a fourth rate, queued when the best stood at the edge
RACE_2026 = "race-2026-100M"
SET_WORDS = {"WikiText-103": "wikitext103", "TinyStories": "tinystories", "FineWeb-EDU": "fineweb-edu", "fresh essays": "fresh", "the fresh essays": "fresh"}
TASK_WORDS = {"HellaSwag": "hellaswag", "BLiMP": "blimp", "ARC-Easy": "arc_easy", "ARC-Challenge": "arc_challenge", "LAMBADA": "lambada", "PIQA": "piqa",
              "WinoGrande": "winogrande", "SciQ": "sciq", "CommonsenseQA": "commonsense_qa"}
SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


class Missing(Exception):
    pass


def load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def summary(run: str) -> dict:
    for root in (repo_dir() / "results" / "runs", CKPT):
        s = load(root / run / "summary.json")
        if s and s.get("status") == "completed":
            return s
    raise Missing(f"the run {run} has no summary yet")


def scored(run: str) -> dict:
    record = load(repo_dir() / "results" / "eval" / f"{run}.json")
    if record is None:
        raise Missing(f"{run} has not been scored: results/eval/{run}.json is not there")
    return record


def race_2019_runs() -> list[str]:
    """The runs of the 2019 recipe in the race: the six named ones and any other `race-2019-*` that has finished."""
    found = sorted(p.parent.name for root in (repo_dir() / "results" / "runs", work_dir() / "ckpt") for p in root.glob("race-2019-*/summary.json"))
    return RACE_2019 + [r for r in dict.fromkeys(found) if r not in RACE_2019]


def best_2019() -> str:
    """The run of the 2019 recipe with the lowest final held-out loss, among those that have finished."""
    done = []
    for run in race_2019_runs():
        try:
            done.append((summary(run)["val_loss"], run))
        except Missing:
            pass
    if not done:
        raise Missing("no run of the 2019 recipe has finished: the best is not known yet")
    return min(done)[1]


def thousands(v: float) -> str:
    return f"{round(v):,}"


def sci(v: float, digits: int = 1) -> str:
    if v == 0:
        return "0"
    e = math.floor(math.log10(abs(v)))
    return f"{v / 10**e:.{digits}f} × 10{str(e).translate(SUP)}"


def words_of(n: float) -> str:
    """A count in the prose's words: 99 million, 3.3 billion."""
    for size, name in ((1e9, "billion"), (1e6, "million")):
        if n >= size:
            v = n / size
            return (f"{v:.0f}" if v >= 20 else f"{v:.1f}") + " " + name
    return thousands(n)


class Sources:
    """Everything the placeholders are answered from, loaded lazily; `ctx` carries what a table row is about."""

    def __init__(self) -> None:
        self.ctx: dict = {}
        self.cache: dict = {}

    def get(self, key: str, loader):
        if key not in self.cache:
            self.cache[key] = loader()
        return self.cache[key]

    def main(self) -> dict:
        return self.get("main", lambda: summary("gptpp-base"))

    def waterfall(self) -> dict:
        raw = load(repo_dir() / "results" / "interp" / "throughput-waterfall.json")
        if raw is None or not raw.get("configs"):
            raise Missing("the single-card configurations have not been timed: run gptpp.export.where_the_time_goes --measure on a free card")
        return raw

    def config(self, key: str) -> dict:
        c = next((q for q in self.waterfall()["configs"] if q["key"] == key), None)
        if c is None:
            raise Missing(f"the configuration {key} has not been timed")
        return c

    def measured(self, part: str) -> dict:
        m = load(repo_dir() / "results" / "interp" / "part-7-measured.json")
        if m is None or part not in m:
            raise Missing("the final model has not been measured: run gptpp.export.part7_measure on a card")
        if m["model"]["run"] != "gptpp-base" or m["model"]["checkpoint"] != "model.pt":
            raise Missing(f"part-7-measured.json was made on {m['model']['run']}/{m['model']['checkpoint']}, not on the final model")
        return m[part]

    def fit(self, count: str) -> dict:
        data = load(SITE / "assets" / "data" / "gpt" / "scaling-surface.json")
        if data is None:
            raise Missing("the scaling fit has not been exported: run gptpp.export.scaling_surface")
        if data.get("synthetic"):
            raise Missing("scaling-surface.json holds synthetic cells")
        f = data["fits"].get(count)
        if f is None:
            raise Missing(f"{len(data['cells'])} of {len(data['expected'])} cells of the grid have finished: no fit on fewer than {data['min_cells']} cells"
                          + (f" of {data['min_sizes']} sizes" if data.get("min_sizes") else ""))
        if data["missing"]:
            f = dict(f, _partial=f"{len(data['cells'])} of {len(data['expected'])} cells")
        return f


def bpb(run: str, set_key: str) -> tuple[str, float, str]:
    r = scored(run)["bits_per_byte"].get(set_key)
    if r is None:
        raise Missing(f"{run} was not scored on {set_key}")
    return (f"{r['bits_per_byte']:.3f} ± {r['bits_per_byte_se']:.3f}", r["bits_per_byte"],
            f"results/eval/{run}.json, bits_per_byte.{set_key}: bits over bytes of {r['n_docs']} documents, standard error over documents (ratio estimator)")


def accuracy(run: str, task: str) -> tuple[str, float, str]:
    r = scored(run).get("benchmarks", {}).get(task)
    if r is None or r.get("limit") is not None:
        raise Missing(f"{run} was not scored on {task}")
    return f"{100 * r['accuracy']:.1f}", r["accuracy"], f"results/eval/{run}.json, benchmarks.{task}.accuracy ({r['headline']} rule, {r['n']} items)"


def centered_mean(run: str) -> tuple[str, float, str]:
    record = scored(run)
    acc = {t: [r["accuracy"]] for t, r in record.get("benchmarks", {}).items() if t in TASKS and r.get("limit") is None}
    chance = {t: record["benchmarks"][t]["chance"] for t in acc}
    c = centered(acc, chance)
    if c is None:
        raise Missing(f"{run} was not scored on all nine benchmarks")
    return f"{100 * c:.1f}", c, f"results/eval/{run}.json: mean over the nine tasks of (accuracy - chance) / (1 - chance), on a scale of 100"


def interval(f: dict, key: str, digits: int) -> str:
    ci = f.get("ci")
    main = f"{f[key]:.4g}" if key in ("alpha", "beta") else f"{f[key]:.{digits}f}"
    return main + (f" ({ci[key]['lo95']:.{digits}f} to {ci[key]['hi95']:.{digits}f})" if ci else "")


def resolve(words: str, S: Sources) -> tuple[str, object, str]:
    """(the text for the placeholder, the number behind it, the note); raises Missing with the reason."""
    ctx = S.ctx
    # ---- the main run
    if words.startswith("wall-clock hours of the main run"):
        s = S.main()
        return f"{s['seconds'] / 3600:.2f}", s["seconds"] / 3600, "summary.json of gptpp-base: seconds / 3600 (the whole run, evaluations and checkpoints included)"
    if words.startswith("tokens per second of the main run on three cards"):
        s = S.main()
        return thousands(s["tokens_per_second"]), s["tokens_per_second"], "summary.json of gptpp-base: tokens_per_second (tokens a step over the mean step time, the first logging interval left out)"
    if words.startswith("largest difference between the three cards' parameter checksums"):
        s = S.main()
        return ("zero" if s["rank_spread"] == 0 else sci(s["rank_spread"])), s["rank_spread"], "summary.json of gptpp-base: rank_spread, the largest difference between the cards' sums of all parameters at the end"
    if words.startswith("final held-out bits per byte of the main run"):
        s = S.main()
        return f"{s['val_bpb']:.3f}", s["val_bpb"], "summary.json of gptpp-base: val_bpb, the trainer's monitor at the last step (1,024 windows of 1,024 tokens, bfloat16)"
    if words == "card-hours of the main run":
        s = S.main()
        return f"{s['n_gpu'] * s['seconds'] / 3600:.1f}", s["n_gpu"] * s["seconds"] / 3600, "summary.json of gptpp-base: n_gpu x seconds / 3600"
    # ---- the waterfall
    table = [("float32 products without TensorFloat-32", "float32"), ("float32 without TensorFloat-32", "float32"), ("the same with TensorFloat-32 products", "tf32"),
             ("the same with bfloat16 autocast", "bf16"), ("the same with the fused attention kernel", "fused"), ("the same compiled", "compiled")]
    for needle, key in table:
        if needle in words:
            c = S.config(key)
            return thousands(c["tokens_per_second"]), c["tokens_per_second"], (
                f"throughput-waterfall.json, configuration {key}: {c['steps']} optimizer steps of 48 x 1,024 tokens in {c['seconds']:.1f} s on one card, "
                f"each step read in {c['passes']} pass(es) of {c['micro_batch']} sequences (what fits 32 GB), after three unmeasured steps")
    # ---- the race
    m = re.match(r"final held-out loss of the 2019 recipe at ([\d.e-]+), 100M tokens", words)
    if m or words == "final held-out loss of the 2026 recipe, 100M tokens":
        run = f"race-2019-lr{m.group(1)}-100M" if m else RACE_2026
        ctx["race"] = None                              # a row's later cells must not answer for another run
        s = summary(run)
        ctx["race"] = run
        return f"{s['val_loss']:.3f}", s["val_loss"], f"summary.json of {run}: val_loss at the last step, nats per token on the trainer's 1,048,576 held-out tokens"
    if words == "its bits per byte":
        if not ctx.get("race"):
            raise Missing("the run of this row has no summary yet")
        s = summary(ctx["race"])
        return f"{s['val_bpb']:.3f}", s["val_bpb"], f"summary.json of {ctx['race']}: val_bpb = val_loss / ln 2 / 4.628 bytes per token"
    # ---- bits per byte of our three models
    m = re.match(r"bits per byte of gpt-pp on (.+), with its standard error$", words) or re.match(r"race model, (2026 recipe|best 2019 rate): (WikiText-103|TinyStories|FineWeb-EDU|fresh essays)$", words)
    if m and words.startswith("bits per byte of gpt-pp on"):
        ctx["model"], ctx["table"] = "gptpp-base", "bpb"
        return bpb("gptpp-base", SET_WORDS[m.group(1)])
    if m:
        ctx["model"], ctx["table"] = None, "bpb"        # until the row's model is known, its cells are missing too
        ctx["model"] = RACE_2026 if m.group(1) == "2026 recipe" else best_2019()
        return bpb(ctx["model"], SET_WORDS[m.group(2)])
    if words == "bits per byte of gpt-pp on held-out FineWeb-EDU":
        r = scored("gptpp-base")["bits_per_byte"]["fineweb-edu"]
        return f"{r['bits_per_byte']:.3f}", r["bits_per_byte"], "results/eval/gptpp-base.json, bits_per_byte.fineweb-edu (1,000 held-out documents, windows of 1,024, tf32)"
    if ((words.startswith("on ") and words[3:] in SET_WORDS) or words in SET_WORDS) and ctx.get("table") == "bpb":
        if not ctx.get("model"):
            raise Missing("the model of this row is not known yet (the race has not finished)")
        return bpb(ctx["model"], SET_WORDS[words[3:] if words.startswith("on ") else words])
    # ---- the benchmarks
    m = re.match(r"(gpt-pp|race model, 2026 recipe|race model, best 2019 rate): (HellaSwag)$", words)
    if m:
        ctx["model"], ctx["table"] = None, "acc"
        ctx["model"] = {"gpt-pp": "gptpp-base", "race model, 2026 recipe": RACE_2026}.get(m.group(1)) or best_2019()
        return accuracy(ctx["model"], "hellaswag")
    if (words in TASK_WORDS or words == "centered mean") and ctx.get("table") == "acc":
        if not ctx.get("model"):
            raise Missing("the model of this row is not known yet (the race has not finished)")
        return centered_mean(ctx["model"]) if words == "centered mean" else accuracy(ctx["model"], TASK_WORDS[words])
    if words == "centered mean of gpt-pp":
        return centered_mean("gptpp-base")
    m = re.match(r"gpt-pp: (ARC-Challenge|WinoGrande), percent$", words)
    if m:
        return accuracy("gptpp-base", TASK_WORDS[m.group(1)])
    # ---- the hosted models' subset
    m = re.match(r"(?:gpt-pp )?on the (\d+) fixed (HellaSwag items|ARC-Easy items|BLiMP pairs)$", words)
    if m:
        task = {"HellaSwag items": "hellaswag", "ARC-Easy items": "arc_easy", "BLiMP pairs": "blimp"}[m.group(2)]
        sub = on_subset(scored("gptpp-base"), {task: int(m.group(1))})
        if task not in sub:
            raise Missing(f"gpt-pp was not scored on {task}")
        a, se, n = sub[task]
        return f"{100 * a:.1f} ± {100 * se:.1f}", a, f"results/eval/gptpp-base.json: the per-item marks of {task} at the {n} items of the fixed subset the hosted models were asked; standard error sqrt(a (1 - a) / n)"
    # ---- the scaling fit
    m = re.match(r"alpha, (blocks counted|blocks and output layer counted|every parameter counted), with its interval$", words)
    if not m and words.endswith(", every parameter counted"):      # the third row names its count in every cell
        ctx["count"], words = "all", words[:-len(", every parameter counted")]
    if m:
        ctx["count"] = {"blocks counted": "blocks", "blocks and output layer counted": "output", "every parameter counted": "all"}[m.group(1)]   # (a count is always known: its cells report the fit's own state)
        f = S.fit(ctx["count"])
        return interval(f, "alpha", 2), f["alpha"], _fit_note(f, "alpha")
    if words in ("beta", "E", "a with its interval", "largest relative residual of the fit", "predicted held-out loss at the main run's N and D"):
        f = S.fit(ctx["count"])
        if words == "beta":
            return interval(f, "beta", 2), f["beta"], _fit_note(f, "beta")
        if words == "E":
            return interval(f, "E", 2), f["E"], _fit_note(f, "E")
        if words == "a with its interval":
            return interval(f, "a", 2), f["a"], _fit_note(f, "a = beta / (alpha + beta)")
        if words.startswith("largest"):
            return f"{100 * f['max_rel_residual']:.1f}%", f["max_rel_residual"], f"scaling-surface.json, fits.{ctx['count']}.max_rel_residual: the largest |fitted - measured| / measured over the cells ({f['max_rel_residual_run']})"
        return f"{f['main']['predicted']:.3f}", f["main"]["predicted"], (
            f"scaling-surface.json, fits.{ctx['count']}.main.predicted: the fitted law at N = {f['main']['N']:,.0f}, D = {f['main']['D']:,.0f}, nats per token")
    if words.startswith("held-out loss of gpt-pp on the trainer's held-out tokens in windows of 512"):
        v = S.measured("val")["w512"]
        return f"{v['val_loss']:.3f}", v["val_loss"], f"part-7-measured.json, val.w512: the trainer's monitor on the final model with windows of 512 ({v['windows']} windows, {v['tokens']:,} tokens, bfloat16)"
    frontier = {"compute-optimal parameters at the main run's budget, second count": "N_opt", "compute-optimal tokens at that budget": "D_opt", "their ratio": "ratio",
                "share of the reducible loss that the main run's allocation gives up under the fit": "excess"}
    if words in frontier:
        f = S.fit("output")
        fr = f.get("frontier")
        if not fr:
            raise Missing("the fitted law has no interior optimum (an exponent is at its bound)")
        key = frontier[words]
        text = {"N_opt": words_of(fr["N_opt"]), "D_opt": words_of(fr["D_opt"]), "ratio": f"{fr['ratio']:.0f}", "excess": f"{100 * fr['excess']:.1f}%"}[key]
        return text, fr[key], (f"scaling-surface.json, fits.output.frontier.{key}: the minimum of the fitted law on 6 N D = {fr['C']:.3g} (N: the blocks and the output layer)"
                               + ("; excess = reducible loss at the main run's (N, D) over that at the optimum, minus 1" if key == "excess" else ""))
    # ---- writing
    if words.startswith("tokens per second of gpt-pp writing 1,024 tokens") or words == "the same without the cache":
        w = S.measured("writing")
        if (w.get("other_mib") or 0) > 500:                 # a timing needs the card to itself
            raise Missing(f"the writing speed was timed while another process held {w['other_mib']} MiB of the card "
                          f"({w['cache']['tokens_per_second']:.0f} tokens a second with the cache, {w['no_cache']['tokens_per_second']:.0f} without: indicative only): time it again on an idle card")
        part = "no_cache" if words == "the same without the cache" else "cache"
        return f"{w[part]['tokens_per_second']:.0f}", w[part]["tokens_per_second"], (
            f"part-7-measured.json, writing.{part}: {w['new_tokens']} tokens after a short prompt, batch 1, bfloat16, the best of {w['repeats']} timings on one card"
            + ("; the whole context through the fused path at every step" if part == "no_cache" else "; gptpp.generate.generate"))
    samples = {'gpt-pp, greedy continuation of "The chemical symbol for gold is", 12 tokens': "gold",
               'gpt-pp, continuation of "def f(x):\\n    return ", 16 tokens at temperature 0.8': "code",
               'gpt-pp, continuation of "Write a poem about the sea.", 40 tokens at temperature 0.8': "poem",
               'gpt-pp, greedy continuation of "Question: What is 17 + 26?\\nAnswer:", 8 tokens': "sum"}
    if words in samples:
        s = S.measured("samples")[samples[words]]
        shown = s["text"].replace("\n", " ⏎ ")
        return f"“{shown}”", s["text"], (f"part-7-measured.json, samples.{samples[words]}: {s['settings']['new_tokens']} tokens after the prompt, {s['settings']['decoding']}"
                                         + (f" at temperature {s['settings']['temperature']}, seed 0" if s["settings"]["temperature"] else "") + "; a line break is shown as ⏎")
    if words.startswith('probability that gpt-pp gives the token "{"'):
        b = S.measured("samples")["brace"]
        return sci(b["probability"]), b["probability"], (f"part-7-measured.json, samples.brace: softmax of the float32 logits after the prompt; the token has rank {b['rank']} of 32,768, "
                                                         f"a uniform guess would give {sci(b['uniform'])}")
    # ---- the accounting
    if words.startswith("largest difference in final held-out loss between the two seeds of ladder rungs"):
        diffs, runs = [], repo_dir() / "results" / "runs"
        for rung in (0, 6, 7, 9):
            pair = {}
            for seed in (0, 1):
                found = sorted(runs.glob(f"ladder-{rung}-*-s{seed}/summary.json"))
                if found:
                    pair[seed] = json.loads(found[0].read_text())["val_loss"]
            if len(pair) < 2:
                raise Missing(f"rung {rung} of the ladder does not have two finished seeds yet")
            diffs.append((abs(pair[0] - pair[1]), rung))
        return f"{max(diffs)[0]:.4f}", max(diffs)[0], "summary.json of ladder-<rung>-*-s0 and -s1 for rungs 0, 6, 7, 9: the largest |val_loss(seed 0) - val_loss(seed 1)|; by rung: " + ", ".join(f"{r}: {d:.4f}" for d, r in sorted(diffs, key=lambda q: q[1]))
    if words == "card-hours of the four race runs":
        runs = RACE_2019[:3] + [RACE_2026]
        total = sum(summary(r)["seconds"] * summary(r).get("n_gpu", 1) for r in runs) / 3600
        return f"{total:.2f}", total, "summary.json of the four race runs of the post's table (2019 at 6e-4, 1.2e-3, 2.4e-3; 2026): the sum of seconds / 3600, one card each"
    if words == "card-hours of the grid":
        from .scaling_surface import expected

        total = sum(summary(r)["seconds"] * summary(r).get("n_gpu", 1) for r in expected()) / 3600
        return f"{total:.2f}", total, "summary.json of the 23 runs of the grid: the sum of seconds / 3600, one card each"
    raise Missing("no rule answers these words: the post asks for something this exporter does not know")


def _fit_note(f: dict, what: str) -> str:
    ci = f.get("ci")
    return (f"scaling-surface.json: {what} of L = E + A / N^alpha + B / D^beta, the sum of Huber losses (delta 1e-3) on ln L minimized from {f['starts']} starts"
            + (f"; 95% percentile interval over {ci['resamples']} resamples of the cells" if ci else "") + (f"; ON {f['_partial']} ONLY" if f.get("_partial") else ""))


def structured(S: Sources) -> dict:
    """The numbers by subject, with notes: what the placeholders say and their neighbors."""
    out: dict = {}

    def attempt(key: str, fn) -> None:
        try:
            out[key] = fn()
        except Missing as why:
            out[key] = {"missing": str(why)}

    def main_run() -> dict:
        s = S.main()
        return {"tokens": s["tokens"], "steps": s["steps"], "cards": s["n_gpu"], "seconds": s["seconds"], "hours": s["seconds"] / 3600, "card_hours": s["n_gpu"] * s["seconds"] / 3600,
                "tokens_per_second": s["tokens_per_second"], "final_val_loss": s["val_loss"], "final_val_bpb": s["val_bpb"], "best_val_loss": s["best_val_loss"],
                "best_val_bpb": s["best_val_bpb"], "best_step": s["best_step"], "rank_spread": s["rank_spread"], "peak_memory_gb": s.get("peak_memory_gb"), "params": s["params"],
                "note": "summary.json of gptpp-base, written by the trainer; val_* are the trainer's monitor: 1,024 windows of 1,024 held-out tokens, bfloat16; best over the evaluations every 50M tokens"}

    def waterfall() -> dict:
        raw = S.waterfall()
        by = {c["key"]: c for c in raw["configs"]}
        out = {"step": raw["step"], "machine": raw["machine"], "configs": [{k: c[k] for k in ("key", "what", "tokens_per_second", "passes", "micro_batch", "steps", "seconds", "peak_memory_gb")} for c in raw["configs"]],
               "split": raw.get("split"), "note": "throughput-waterfall.json: each configuration timed for at least 25 s after a first step and at least 12 s of further steps, unmeasured; "
                                                  "one optimizer step for every 48 x 1,024 tokens, in as few passes as fit the card"}
        if all(k in by for k in ("float32", "bf16", "fused", "compiled")):
            out["factors"] = {key: by[key]["tokens_per_second"] / by[before]["tokens_per_second"] for before, key in (("float32", "tf32"), ("tf32", "bf16"), ("bf16", "fused"), ("fused", "compiled")) if before in by and key in by}
            out["compiled_over_float32"] = by["compiled"]["tokens_per_second"] / by["float32"]["tokens_per_second"]
            # for Section 3.6: attention written out against the fused kernel, both with bfloat16 products, neither compiled
            out["fused_against_written_out"] = {"written_out_tokens_per_second": by["bf16"]["tokens_per_second"], "fused_tokens_per_second": by["fused"]["tokens_per_second"],
                                               "factor": by["fused"]["tokens_per_second"] / by["bf16"]["tokens_per_second"],
                                               "written_out_gb_per_sequence": by["bf16"]["peak_memory_gb"] / by["bf16"]["micro_batch"],
                                               "fused_gb_per_sequence": by["fused"]["peak_memory_gb"] / by["fused"]["micro_batch"],
                                               "note": "peak memory of the card over the sequences of one pass (1,024 tokens each): the whole model's, not the attention's alone"}
        return out

    def race() -> dict:
        runs = {}
        for run in race_2019_runs() + [RACE_2026]:
            try:
                s = summary(run)
                runs[run] = {k: s[k] for k in ("val_loss", "val_bpb", "best_val_loss", "best_val_bpb", "tokens", "steps", "seconds", "tokens_per_second")}
                runs[run]["n_gpu"] = s.get("n_gpu", 1)
                rate, warm = re.search(r"-lr([0-9.e-]+?)-(?:w|\d+M)", run), re.search(r"-w(\d+)M-", run)
                if rate:                                # the 2019 runs carry their rate and their warmup in their names
                    runs[run]["lr"], runs[run]["warmup_tokens"] = float(rate.group(1)), (int(warm.group(1)) * 10**6 if warm else 32 * 10**6)
            except Missing as why:
                runs[run] = {"missing": str(why)}
        done = {r: v for r, v in runs.items() if "val_loss" in v}
        info = {"runs": runs, "note": "summary.json of each race run: 100M tokens on one card, 2,035 steps of 49,152 tokens, one seed; val_* on the trainer's 1,048,576 held-out tokens"}
        done19 = [r for r in done if r != RACE_2026]
        info["missing_2019"] = [r for r in runs if r != RACE_2026 and r not in done]
        if done19:                                          # the best of the 2019 runs that exist; `missing_2019` names the ones that do not yet
            best = min(done19, key=lambda r: done[r]["val_loss"])
            info["best_2019"] = best
            info["best_2019_is_largest_rate"] = done[best].get("lr") is not None and done[best]["lr"] >= max(v.get("lr") or 0 for r, v in done.items() if r != RACE_2026)
            if RACE_2026 in done:
                info["gap_loss"] = done[best]["val_loss"] - done[RACE_2026]["val_loss"]
                info["gap_bpb"] = done[best]["val_bpb"] - done[RACE_2026]["val_bpb"]
        info["card_hours_all"] = sum(v["seconds"] * v.get("n_gpu", 1) for v in done.values()) / 3600
        info["card_hours_runs"] = len(done)
        return info

    def evaluation() -> dict:
        rows = {}
        for label, run in (("gpt-pp", lambda: "gptpp-base"), ("race-2026", lambda: RACE_2026), ("race-2019-best", best_2019)):
            try:
                name = run()
                record = scored(name)
                row = {"run": name, "bits_per_byte": {s: [record["bits_per_byte"][s]["bits_per_byte"], record["bits_per_byte"][s]["bits_per_byte_se"]] for s in SETS if s in record.get("bits_per_byte", {})},
                       "accuracy": {t: [r["accuracy"], r.get("stderr", r.get(r.get("headline", "") + "_se"))] for t, r in record.get("benchmarks", {}).items() if t in TASKS}}
                try:
                    row["centered_mean"] = centered_mean(name)[1]
                except Missing:
                    row["centered_mean"] = None
                row["hosted_subset"] = on_subset(record, {"hellaswag": 200, "arc_easy": 200, "blimp": 300})
                rows[label] = row
            except Missing as why:
                rows[label] = {"missing": str(why)}
        rows["note"] = ("results/eval/<run>.json: bits per byte by document (windows of 1,024, tf32) with the standard error over documents; accuracy by log-likelihood with the standard "
                        "error over items; hosted_subset: [accuracy, standard error, n] on the items the hosted models were asked")
        return rows

    def scaling() -> dict:
        data = load(SITE / "assets" / "data" / "gpt" / "scaling-surface.json")
        if data is None or data.get("synthetic"):
            raise Missing("the scaling fit has not been exported from real cells")
        keep = ("alpha", "beta", "E", "A", "B", "a", "objective", "grad_norm", "starts", "agree", "second_optimizer", "at_bound", "max_rel_residual", "max_rel_residual_run",
                "rms_rel_residual", "ci", "main", "frontier")
        return {"cells": len(data["cells"]), "expected": len(data["expected"]), "missing_cells": data["missing"], "fits": {k: ({q: f[q] for q in keep} if f else None) for k, f in data["fits"].items()},
                "caveats": data["caveats"], "main": data["main"],
                "note": "scaling-surface.json (gptpp.export.scaling_surface): the additive law fitted to the grid's final held-out losses under three counts of N; intervals from resampled cells"}

    attempt("main_run", main_run)
    attempt("waterfall", waterfall)
    attempt("race", race)
    attempt("eval", evaluation)
    attempt("scaling", scaling)
    attempt("writing", lambda: dict(S.measured("writing")))
    attempt("samples", lambda: dict(S.measured("samples")))
    attempt("windows_of_512", lambda: dict(S.measured("val")))
    return out


def placeholders(post: Path, previous: dict | None) -> list[dict]:
    """Every `@@words@@` of the post in order; when the post has been filled since, the list of the last run."""
    found = []
    if post.exists():
        for number, line in enumerate(post.read_text(encoding="utf-8").splitlines(), 1):
            for m in re.finditer(r"@@(.+?)@@", line):
                found.append({"line": number, "words": m.group(1)})
    old = [{"line": p["line"], "words": p["words"]} for p in (previous or {}).get("placeholders", [])]
    return found if found else old                      # the post as it stands; the last list only when the post has none left


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--post", default=str(POST))
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args(argv)
    out_path = Path(args.out)
    S = Sources()
    items, by_words, missing = [], {}, []
    for n, p in enumerate(placeholders(Path(args.post), load(out_path)), 1):
        entry = {"n": n, "line": p["line"], "words": p["words"], "placeholder": p["words"]}
        try:
            value, raw, note = resolve(p["words"], S)
            entry.update(value=value, raw=raw, note=note)
        except Missing as why:
            entry.update(value=None, raw=None, note=None, missing=str(why))
            missing.append({"n": n, "line": p["line"], "words": p["words"], "why": str(why)})
        except KeyError as key:  # a row's words before the words that say which row it is
            entry.update(value=None, raw=None, note=None, missing=f"no context for these words ({key})")
            missing.append({"n": n, "line": p["line"], "words": p["words"], "why": entry["missing"]})
        items.append(entry)
        by_words.setdefault(p["words"], []).append(entry["value"])
    result = {"generated_by": "gptpp.export.part7_numbers", "when": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), "post": str(args.post),
              "filled": len(items) - len(missing), "total": len(items), "placeholders": items,
              "by_words": {w: (v[0] if len(v) == 1 else v) for w, v in by_words.items()}, "numbers": structured(S), "missing": missing}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{result['filled']} of {result['total']} placeholders filled; wrote {out_path}")
    reasons: dict = {}
    for m in missing:
        reasons.setdefault(m["why"], []).append(m["n"])
    for why, ns in reasons.items():
        print(f"  missing ({len(ns)}): {why}")


if __name__ == "__main__":
    main()
