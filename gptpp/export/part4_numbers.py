"""The measured numbers of the post "Position" (part 4, Section 4.5): one entry for each placeholder of the post.

    python -m gptpp.export.part4_numbers [--out results/interp/part-4.json]

CPU only, a second. It reads what is already on disk: the export of `gptpp.export.position_schemes_measured` in the
site's data folder (the five position swaps of the ladder scored on held-out windows of 2,048 tokens, and every
head's share of attention by distance), each run's summary and log under results/runs, and the post itself for the
exact words of its placeholders `@@...@@`. It writes one JSON object: the placeholders' words as keys, each with a
value, its unit, a text ready for the page and a note on how it was computed; `results_lines`, the measured answers
to the post's four `<!-- RESULTS: ... -->` questions; and `_supplement`, the numbers behind those answers.

Positions are written as the post writes them, counted from 1: "positions 1 to 512" are the first 512 tokens of a
window. Losses are in nats per token, as in the post's tables (the figure draws bits per token: nats / ln 2).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from .. import repo_dir
from .common import DATA, SITE, write_json

EXPORT = "position-schemes-measured"
POST = SITE / "_posts" / "gpt" / "2026-11-14-gpt-04.md"
NAMES = {"learned": "learned table", "sinusoidal": "sinusoid", "rotary": "rotary", "alibi": "ALiBi", "none": "no positions"}
PLACEHOLDER = {  # the words the post uses for a scheme in its placeholders
    "learned": ("learned table", "a learned position table (run swap-pos-learned)"),
    "sinusoidal": ("sinusoid", "sinusoidal positions (run swap-pos-sinusoidal)"),
    "rotary": ("rotary", "rotary positions (run swap-base)"),
    "alibi": ("ALiBi", "ALiBi (run swap-pos-alibi)"),
    "none": ("no positions", "no position signal (run swap-pos-none)"),
}
LADDER = {"learned": "ladder-1-no-biases", "rotary": "ladder-2-rotary"}   # the two rungs of the 2019 recipe that differ in position alone
SEED_NOISE = 0.0006          # nats per token between two seeds of the recipe in a short run at this size, as the post quotes it
NOISE = 0.01                 # a difference under this is inside seed noise: two seeds of a ladder rung differ by 0.0025 to 0.018
SHORT, UNIFORM = 0.5, 0.10   # the two rules for a head: this share on offsets 1 to 3; this much attention placed otherwise than a uniform reader's


def load(data: Path) -> dict:
    meta = json.loads((data / f"{EXPORT}.json").read_text())
    entry = json.loads((data / f"{EXPORT}.entries.json").read_text())[f"{EXPORT}.attention"]
    raw = np.fromfile(data / entry["file"], dtype="<f2", count=entry["bytes"] // 2, offset=entry["offset"])
    return {"meta": meta, "att": raw.reshape(entry["shape"]).astype(np.float64)}


def range_mean(meta: dict, key: str, lo: int, hi: int) -> float | None:
    """Mean loss in nats per token over the 0-based positions lo .. hi - 1 (whole buckets), or None if the scheme has none there."""
    P, loss = np.array(meta["positions"]), np.array(meta["schemes"][key]["loss"])
    n = len(loss)
    sel = (P[:n] >= lo) & (P[1:n + 1] <= hi)
    width = np.diff(P)[:n][sel]
    return float((loss[sel] * width).sum() / width.sum()) if width.sum() == hi - lo else None


def half_offset(share: np.ndarray, edges: np.ndarray) -> tuple[float, int]:
    """The offset within which half of a head's attention falls, and the bucket that holds it. Inside a bucket of
    several offsets the place is taken linearly, so the answer is only as fine as the bucket."""
    c = np.cumsum(share)
    b = int(np.searchsorted(c, 0.5))
    before = c[b - 1] if b else 0.0
    width = edges[b + 1] - edges[b]
    return (float(edges[b] + (0.5 - before) / max(share[b], 1e-12) * (width - 1)) if width > 1 else float(edges[b])), b


def heads(ex: dict, key: str) -> dict:
    """What the 64 heads of a scheme do, read from inside the training length and, where the model runs, from past it."""
    meta, T = ex["meta"], ex["meta"]["train_length"]
    E, s = np.array(meta["distances"]), meta["order"].index(key)
    keys_in = np.array(meta["attention"]["keys"]["inside"])
    nb = int((keys_in > 0).sum())
    uniform = keys_in[:nb] / keys_in[:nb].sum()
    share = ex["att"][s, 0][:, :, :nb]
    share = share / share.sum(-1, keepdims=True)
    halves = np.array([[half_offset(share[l, h], E)[0] for h in range(share.shape[1])] for l in range(share.shape[0])])
    buckets = np.array([[half_offset(share[l, h], E)[1] for h in range(share.shape[1])] for l in range(share.shape[0])])
    short, previous = share[:, :, 1:4].sum(-1), share[:, :, 1]
    tv = 0.5 * np.abs(share - uniform).sum(-1)
    mb = int(np.median(buckets))
    out = {
        "heads": int(share.shape[0] * share.shape[1]),
        "median_half_offset": float(np.median(halves)), "median_bucket": [int(E[mb]), int(E[mb + 1]) - 1],
        "half_offset_quartiles": [float(np.percentile(halves, 25)), float(np.percentile(halves, 75))],
        "heads_on_offsets_1_to_3": int((short >= SHORT).sum()), "by_layer_on_offsets_1_to_3": [int(x) for x in (short >= SHORT).sum(1)],
        "heads_with_half_on_the_previous_token": int((previous >= 0.5).sum()), "largest_share_on_the_previous_token": float(previous.max()),
        "largest_share_on_one_offset_1_to_3": float(share[:, :, 1:4].max()),
        "heads_following_uniform": int((tv <= UNIFORM).sum()), "by_layer_following_uniform": [int(x) for x in (tv <= UNIFORM).sum(1)],
        "share_on_offsets_1_to_3": float(short.mean()), "share_on_the_8_nearest": float(share[:, :, :6].sum(-1).mean()),
        "kernel_over_uniform_at_384_to_511": float((share[:, :, nb - 1] / uniform[nb - 1]).mean()),
        "uniform_half_offset": half_offset(uniform, E)[0],
    }
    if key == "alibi":   # head h of every layer has slope 2^-(h + 1): a window of 2^(h + 1) tokens
        out["half_offset_by_head"] = [float(np.median(halves[:, h])) for h in range(share.shape[1])]
        out["window_by_head"] = [2 ** (h + 1) for h in range(share.shape[1])]
    if meta["schemes"][key]["longest"] > T:
        keys_out = np.array(meta["attention"]["keys"]["outside"])
        far_keys = E[:-1] >= T
        past = ex["att"][s, 1]
        past = past / past.sum(-1, keepdims=True)
        far = past[:, :, far_keys].sum(-1)
        out["past"] = {
            "share_512_or_more_back": float(far.mean()), "largest": float(far.max()), "heads_over_half": int((far >= 0.5).sum()),
            "share_on_offsets_1_to_3": float(past[:, :, 1:4].sum(-1).mean()), "share_on_the_8_nearest": float(past[:, :, :6].sum(-1).mean()),
            "kernel_over_uniform_at_256_to_511": float((past[:, :, 16:18].sum(-1) / (keys_out[16:18].sum() / keys_out.sum())).mean()),
            "kernel_over_uniform_past_511": float((far / (keys_out[far_keys].sum() / keys_out.sum())).mean()),
        }
    return out


def curve(meta: dict, key: str) -> dict:
    """How a scheme's loss runs along the window: by blocks of 256 positions, its lowest bucket, where it first stands above its own level."""
    T, L, s = meta["train_length"], meta["schemes"][key]["longest"], meta["schemes"][key]
    P, loss = np.array(meta["positions"]), np.array(s["loss"])
    n = len(loss)
    ref = range_mean(meta, key, T // 2, T)
    blocks = {f"{lo + 1} to {lo + 256}": round(range_mean(meta, key, lo, lo + 256), 4) for lo in range(0, L, 256)}
    coarse = P[:n] >= 64
    low = int(np.argmin(loss[coarse]))
    above = [int(P[b]) + 1 for b in range(n) if P[b] >= T and loss[b] > ref + 0.035]
    after = int(np.where(P[:n] == T)[0][0]) if L > T else None
    return {"at_257_to_512": ref, "by_256": blocks, "lowest_bucket": [int(P[:n][coarse][low]) + 1, int(P[1:n + 1][coarse][low])],
            "first_bucket_0.035_above_its_level": above[0] if above else None,
            "first_32_past_the_boundary": float(loss[after]) if after is not None else None, "last_32": float(loss[-1]),
            "standard_error_of_a_bucket": float(np.mean(s["se"][8:]))}


def training(runs: Path, run: str) -> list[list[float]]:
    log = runs / run / "log.jsonl"
    if not log.exists():
        return []
    return [[int(o["tokens"]), round(o["val_loss"], 4)] for o in map(json.loads, log.read_text().splitlines()) if o.get("kind") == "val"]


def summary(runs: Path, run: str) -> dict | None:
    f = runs / run / "summary.json"
    return json.loads(f.read_text()) if f.exists() else None


def f4(v: float) -> str:
    return f"{v:.4f}"


def f3(v: float) -> str:
    return f"{v:.3f}"


def signed(v: float, digits: int = 3) -> str:
    return ("+" if v >= 0 else "−") + f"{abs(v):.{digits}f}"


def pct(v: float) -> str:
    return (f"{100 * v:.0f}" if v >= 0.0095 else f"{100 * v:.1f}") + " %"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--out", default=str(repo_dir() / "results" / "interp" / "part-4.json"))
    args = ap.parse_args()
    runs = repo_dir() / "results" / "runs"
    ex = load(Path(args.data))
    meta, T, L = ex["meta"], ex["meta"]["train_length"], ex["meta"]["length"]
    order = [k for k in ("learned", "sinusoidal", "rotary", "alibi", "none") if k in meta["schemes"]]
    S = {k: meta["schemes"][k] for k in order}
    H = {k: heads(ex, k) for k in order}
    C = {k: curve(meta, k) for k in order}
    windows = meta["data"]["windows"]
    scored = f"{windows:,} held-out windows of {L:,} tokens ({meta['data']['tokens']:,} tokens), one forward pass a window, bf16 as in training"

    def variant(k: str) -> str:   # which run stands for a scheme, where there has been more than one
        e = meta.get("earlier", {}).get(k)
        if not e:
            return ""
        return (f". The run is {S[k]['run']}" + (" (the sinusoids added after the embedding's normalization, at unit amplitude)" if S[k].get("after_norm") else "")
                + f"; the first run, {e['run']}, with {e['what']}, ended at {f4(e['val_loss'])} and is not used: it measured a fading signal")

    out: dict = {
        "_post": "_posts/gpt/2026-11-14-gpt-04.md, Section 4.5 (Examples 4.5.1 to 4.5.3)",
        "_written_by": "python -m gptpp.export.part4_numbers (from assets/data/gpt/position-schemes-measured.* and results/runs)",
        "_note": "one entry for each placeholder @@...@@ of the post, keyed by its exact words: value, unit, a text for the page, and how it was computed. "
                 "Losses are in nats per token. One seed a run. Positions are counted from 1, as the post writes them.",
    }
    # Example 4.5.1: the five runs
    tokens = sorted({S[k]["tokens"] for k in order})
    out["number of training tokens of each position-swap run on the ladder"] = {
        "value": tokens[0] if len(tokens) == 1 else tokens, "unit": "tokens", "text": f"{tokens[0]:,}" if len(tokens) == 1 else ", ".join(f"{t:,}" for t in tokens),
        "note": "`tokens` of each run's summary.json (the five agree); the budget was 50e6, rounded up to whole steps of 65,536 tokens"}
    for k in order:
        out[f"held-out loss of the ladder model with {PLACEHOLDER[k][1]}"] = {
            "value": round(S[k]["val_loss"], 4), "unit": "nats per token", "text": f4(S[k]["val_loss"]),
            "note": f"final `val_loss` of {S[k]['run']}/summary.json: the trainer's own evaluation at the end of training, on its fixed set of 2,048 held-out "
                    f"windows of {T} tokens (1,048,576 tokens); seed {S[k]['seed']}; {S[k]['val_bpb']:.4f} bits per byte" + variant(k)}
    # the same switch on the ladder proper
    rungs = {}
    for name, stem in LADDER.items():
        seeds = {}
        for seed in range(4):
            sm = summary(runs, f"{stem}-s{seed}")
            if sm:
                seeds[f"seed {seed}"] = round(sm["val_loss"], 4)
        rungs[name] = seeds
    both = all(len(v) >= 2 for v in rungs.values())
    mean = {k: float(np.mean(list(v.values()))) for k, v in rungs.items() if v}
    spread = {k: float(max(v.values()) - min(v.values())) for k, v in rungs.items() if len(v) >= 2}
    ladder_key = ("held-out loss of the ladder rung with learned positions (the 2019 recipe without biases) and of the next rung, the same with rotary positions: "
                  "mean of two seeds each, and the difference between the two seeds of each")
    if both:
        text = (f"The rung with a learned table scores {f3(mean['learned'])} nats per token and the rung with rotary positions {f3(mean['rotary'])}, means of two seeds "
                f"that differ by {spread['learned']:.4f} and {spread['rotary']:.4f}: a step of {abs(mean['rotary'] - mean['learned']):.3f}.")
    else:
        text = (f"With seed 0 the rung with a learned table scores {f3(rungs['learned']['seed 0'])} nats per token and the rung with rotary positions "
                f"{f3(rungs['rotary']['seed 0'])}: a step of {abs(rungs['rotary']['seed 0'] - rungs['learned']['seed 0']):.3f}.")
    out[ladder_key] = {
        "value": {"learned": rungs["learned"], "rotary": rungs["rotary"], "mean": {k: round(v, 4) for k, v in mean.items()},
                  "difference_between_seeds": {k: round(v, 4) for k, v in spread.items()}, "step_learned_to_rotary": round(mean["rotary"] - mean["learned"], 4)},
        "unit": "nats per token", "text": text, "status": "two seeds" if both else "second seed pending",
        "note": "final `val_loss` of ladder-1-no-biases-s* and ladder-2-rotary-s* (the 2019 recipe without biases, and the same with pos=rope), 100,007,936 tokens "
                "each, evaluated as above; this script fills the means and the seeds' differences once the s1 runs are under results/runs"}
    # Example 4.5.2: three ranges of positions
    ranges = [("positions 1 to 512", 0, T), ("positions 513 to 1,024", T, 2 * T), ("positions 1,025 to 2,048", 2 * T, L)]
    for k in order:
        for words, lo, hi in ranges:
            v = range_mean(meta, k, lo, hi)
            if v is None:
                continue
            key = f"loss, {PLACEHOLDER[k][0]}, {words}" + (" of the 2,048-token windows" if k == "learned" else "")
            extra = ""
            if lo == 0:
                extra = (f"; these are other windows than the run's own evaluation ({f4(S[k]['val_loss'])}), and the range holds the first tokens of a window, which have "
                         f"little context: the level to hold later ranges against is positions 257 to 512, {f4(C[k]['at_257_to_512'])}")
            out[key] = {"value": round(v, 4), "unit": "nats per token", "text": f3(v),
                        "note": f"mean cross-entropy of {S[k]['run']} over {words} of {scored}; mean of the export's position buckets weighted by their widths{extra}" + variant(k)}
    out["number of seeds behind each entry of this table"] = {
        "value": 1, "unit": "seeds", "text": "One seed each.",
        "note": "each of the five swaps was trained once (seed 0); the standard error of a range mean over the windows is about 0.005 nats, and tells nothing of the spread between seeds"}
    # Example 4.5.3: where the heads look
    names = [("learned", "learned"), ("sinusoid", "sinusoidal"), ("rotary", "rotary"), ("ALiBi", "alibi"), ("none", "none")]
    med = {n: H[k]["median_half_offset"] for n, k in names if k in H}
    out["median over heads of the offset containing half of the mean attention mass: learned, sinusoid, rotary, ALiBi, none"] = {
        "value": {n: round(v, 1) for n, v in med.items()}, "bucket": {n: H[k]["median_bucket"] for n, k in names if k in H},
        "quartiles": {n: [round(x, 1) for x in H[k]["half_offset_quartiles"]] for n, k in names if k in H},
        "uniform_reader": round(H[order[0]]["uniform_half_offset"], 1), "unit": "tokens back (offset 0 is the token itself)",
        "text": ", ".join(f"about {v:.0f} tokens for {'the learned table' if n == 'learned' else 'the sinusoid' if n == 'sinusoid' else 'no positions' if n == 'none' else n}"
                          for n, v in med.items()) + f" (a head that read every token alike: {H[order[0]]['uniform_half_offset']:.0f})",
        "note": f"for each of the 64 heads, the mean share of attention at each offset over {meta['attention']['windows']} held-out windows, for readers at positions 257 to 512 "
                "(the second half of the training length: every reader has at least 257 keys, so a uniform head is level and not falling, unlike an average over all "
                "queries i > r); the offset within which half of the share falls; then the median over heads. COARSE: offsets 0 to 3 are exact, beyond them the export "
                "has two buckets an octave (4-5, 6-7, 8-11, 12-15, 16-23, 24-31, 32-47, ...) and the half-way point is taken linearly inside its bucket; the medians "
                "fall in the buckets given under `bucket`, so learned, rotary and ALiBi are not told apart by this statistic, and the quartiles show how wide the heads spread"}

    # the four RESULTS questions, answered from the numbers: every word that depends on a number is chosen by it
    val = {k: round(S[k]["val_loss"], 4) for k in order}   # as the table prints them, so that a difference in a sentence is the difference of two cells
    by_loss = sorted(order, key=val.get)
    tr = {k: training(runs, S[k]["run"]) for k in order}
    ref_tr = dict(map(tuple, tr.get("rotary", [])))
    gaps = [[t, round(v - ref_tr[t], 4)] for t, v in tr.get("none", []) if t > 0 and t in ref_tr]
    early = meta.get("earlier", {}).get("sinusoidal")
    answers = {}
    close = [f"{NAMES[x]} and {NAMES[y]} are {val[y] - val[x]:.4f} apart: inside seed noise, not a ranking on one seed" for x, y in zip(by_loss, by_loss[1:]) if val[y] - val[x] < NOISE]
    seeds = (f"the two seeds of the ladder rungs with a learned table and with rotary positions by {spread['learned']:.4f} and {spread['rotary']:.4f}" if both else
             "the second seeds of the two ladder rungs are pending")
    haviv = ["alibi", "sinusoidal", "learned", "none"]
    ours = [k for k in by_loss if k in haviv]
    line = (f"One seed each. The scale: two seeds of the recipe differed by {SEED_NOISE} nats in a short run, {seeds}; a difference under about {NOISE} is inside seed "
            f"noise. Order at the training length, best to worst: "
            + ", ".join(f"{NAMES[k]} {f4(val[k])}" for k in by_loss) + ". " + "".join(c[0].upper() + c[1:] + ". " for c in close)
            + "Against rotary: " + ", ".join(f"{NAMES[k]} {signed(val[k] - val['rotary'])}" for k in by_loss if k != "rotary") + "; the sinusoid is "
            + f"{abs(val['learned'] - val['sinusoidal']):.3f} {'ahead of' if val['sinusoidal'] < val['learned'] else 'behind'} the learned table. ")
    line += ("This is the order of Haviv et al. (ALiBi, sinusoid, learned, none). " if ours == haviv else
             "Haviv et al. have ALiBi, sinusoid, learned, none; here it is " + ", ".join(NAMES[k] for k in ours) + ". ")
    if gaps:
        line += ("The model with no positions is behind rotary from the first evaluation: by " + ", ".join(f"{g:.3f} after {t / 1e6:.0f}M tokens" for t, g in gaps)
                 + (": the gap narrows and does not close. " if gaps[-1][1] < gaps[0][1] and gaps[-1][1] > 0 else ". "))
    line += f"ALiBi under QK normalization is {signed(val['alibi'] - val['rotary'], 4)} against rotary at the training length" + (": inside seed noise, not hurt. " if val["alibi"] - val["rotary"] < NOISE else ". ")
    if early:
        line += (f"The sinusoid here has its sinusoids added after the embedding's normalization. The first sinusoid run ({early['run']}), with {early['what']}, ended at "
                 f"{f4(early['val_loss'])} ({signed(round(early['val_loss'], 4) - val['rotary'], 4)} against rotary, {signed(round(early['val_loss'], 4) - val['learned'], 4)} against the learned table): "
                 f"it measured a fading signal and is not in the table.")
    answers["line 316: which schemes are within seed noise at the training length; the order; the model with no positions during training; ALiBi under QK normalization"] = line.strip()
    step19 = (mean["rotary"] - mean["learned"]) if mean else None
    step26 = val["rotary"] - val["learned"]
    answers["line 320: the step from a learned table to rotary inside the 2019 recipe, against the seeds' spread; whether it agrees with the swap inside the 2026 recipe"] = (
        (f"Two seeds: " if both else "Seed 0 only (second seed pending, so no spread between seeds yet): ") +
        f"inside the 2019 recipe the step from a learned table ({f4(mean['learned'])}) to rotary ({f4(mean['rotary'])}) is {signed(step19, 4)} nats per token after 100M tokens"
        + (f", against {spread['learned']:.4f} and {spread['rotary']:.4f} between the seeds of each" if both else "") +
        f"; inside the 2026 recipe the same swap is {signed(step26, 4)} ({f4(val['learned'])} to {f4(val['rotary'])}) after 50M tokens: "
        + ("the same sign and the same size." if step19 * step26 > 0 and abs(step19 - step26) < 0.02 else "the same sign." if step19 * step26 > 0 else "the opposite sign."))

    RISEN = 0.035
    far = {k: range_mean(meta, k, 2 * T, L) for k in order if S[k]["longest"] > T}

    def past(k: str) -> str:
        c, ref = C[k], C[k]["at_257_to_512"]
        near_, far_ = range_mean(meta, k, T, 2 * T), far[k]
        d = lambda block: signed(c["by_256"][block] - ref)
        if far_ - ref <= RISEN and c["first_bucket_0.035_above_its_level"] is None:
            return (f"{NAMES[k]} holds: {f4(ref)} at 257 to 512, {f4(near_)} at 513 to 1,024, {f4(far_)} at 1,025 to 2,048 ({signed(far_ - ref)}); no 32 positions stand {RISEN} above "
                    f"its level, and its lowest are {c['lowest_bucket'][0]:,} to {c['lowest_bucket'][1]:,}")
        first = c["first_32_past_the_boundary"] - ref
        how = ("it rises at the boundary itself" if first > RISEN else
               f"it is level just past the boundary and first stands {RISEN} above its level from position {c['first_bucket_0.035_above_its_level']:,}")
        return (f"{NAMES[k]} does not hold ({f4(ref)} at 257 to 512, {f4(near_)} at 513 to 1,024, {f4(far_)} at 1,025 to 2,048): {how}; {signed(first)} in the first 32 positions, "
                f"{d('513 to 768')} over 513 to 768, {d('1025 to 1280')} at 1,025 to 1,280, {d('1793 to 2048')} at 1,793 to 2,048")
    ranked = sorted(far, key=far.get)
    turn = {k: C[k]["first_bucket_0.035_above_its_level"] for k in far}
    press = ("alibi" in far and far["alibi"] - C["alibi"]["at_257_to_512"] <= RISEN and turn.get("sinusoidal") is not None and turn.get("rotary") is not None
             and turn["sinusoidal"] < turn["rotary"])
    answers["line 343: for each scheme, whether the loss past position 512 falls, stays level or rises, and where it turns; the order at the far end"] = (
        "Each against its own loss at positions 257 to 512 (positions 1 to 512 include the context-poor start of a window). "
        + " ".join(past(k)[0].upper() + past(k)[1:] + "." for k in ranked)
        + " Order over 1,025 to 2,048, best to worst: " + ", ".join(f"{NAMES[k]} {f3(far[k])}" for k in ranked) + "; the learned table cannot run. "
        + ("This is the order of Press et al. (the sinusoid fails first, rotary later, ALiBi holds)" if press else "This is not the order of Press et al. (the sinusoid first, rotary later, ALiBi holds)")
        + (f"; the model with no positions ends {'worst' if ranked[-1] == 'none' else 'at ' + f3(far['none'])}. " if "none" in far else ". ")
        + f"One seed: no spread across seeds; the standard error over the {windows:,} windows is about {C['rotary']['standard_error_of_a_bucket']:.3f} for a bucket of 32 positions.")
    count = lambda field: ", ".join(f"{NAMES[k]} {H[k][field]}" for k in order)
    layers = lambda k: ", ".join(f"{n} in layer {l + 1}" for l, n in enumerate(H[k]["by_layer_following_uniform"]) if n)
    sharp = lambda k: H[k]["largest_share_on_the_previous_token"]
    a = H["alibi"]
    sharper = sharp("rotary") > sharp("learned") and sharp("rotary") > sharp("sinusoidal")
    answers["line 347: how many heads concentrate on offsets 1 to 3, how many follow the uniform reference; rotary against learned and sinusoid; no positions; ALiBi's windows; offsets beyond 512"] = (
        f"Rules, for readers at positions 257 to 512: a head concentrates on offsets 1 to 3 if at least {SHORT:.0%} of its attention falls there; it follows the uniform "
        f"reference if at most {UNIFORM:.0%} of its attention is placed otherwise than a reader of every token alike would place it (total variation over the distance buckets). "
        f"Of 64 heads, on offsets 1 to 3: {count('heads_on_offsets_1_to_3')}. Following uniform: {count('heads_following_uniform')}"
        + "".join(f" ({NAMES[k]}: {layers(k)})" for k in order if H[k]["heads_following_uniform"]) + ". "
        + "Heads with over half on the previous token, and the sharpest head's share there: "
        + ", ".join(f"{NAMES[k]} {H[k]['heads_with_half_on_the_previous_token']} ({sharp(k):.2f})" for k in order)
        + (": rotary has the sharper short-offset heads than the learned table and the sinusoid. " if sharper else ": rotary's are not sharper than both the learned table's and the sinusoid's. ")
        + (f"With no positions no head puts half of its attention on one fixed offset: the most on a single offset from 1 to 3 is {H['none']['largest_share_on_one_offset_1_to_3']:.2f}. "
           if H["none"]["largest_share_on_one_offset_1_to_3"] < 0.5 else
           f"With no positions a head puts {H['none']['largest_share_on_one_offset_1_to_3']:.2f} of its attention on one fixed offset. ")
        + "ALiBi's heads against their windows: half of the attention falls within " + ", ".join(f"{v:.0f}" if v >= 9.5 else f"{v:.1f}" for v in a["half_offset_by_head"])
        + " tokens for windows of " + ", ".join(str(w) for w in a["window_by_head"]) + " (median over the eight layers). Read from positions 1,537 to 2,048, the share of attention "
        "on tokens 512 or more back is " + ", ".join(f"{pct(H[k]['past']['share_512_or_more_back'])} for {NAMES[k]} ({H[k]['past']['heads_over_half']} heads over half)" for k in order if "past" in H[k])
        + "; the share on offsets 1 to 3 goes " + ", ".join(f"from {pct(H[k]['share_on_offsets_1_to_3'])} to {pct(H[k]['past']['share_on_offsets_1_to_3'])} for {NAMES[k]}" for k in order if "past" in H[k]) + ".")
    out["results_lines"] = answers
    out["_supplement"] = {
        "loss_along_the_window": C, "heads": H, "held_out_loss_during_training": tr, "no_positions_minus_rotary_during_training": gaps,
        "rules": {"concentrates_on_offsets_1_to_3": SHORT, "follows_uniform_total_variation": UNIFORM, "level_to_hold": "positions 257 to 512", "risen": 0.035},
        "export": {"windows": windows, "attention_windows": meta["attention"]["windows"], "same_recipe": meta.get("same_recipe"), "runs": {k: S[k]["run"] for k in order},
                   "earlier": meta.get("earlier", {})},
    }
    if POST.exists():   # every placeholder of the post has its entry, word for word
        wanted = re.findall(r"@@(.+?)@@", POST.read_text())
        missing = [w for w in wanted if w not in out]
        if missing:
            raise SystemExit("no entry for: " + "; ".join(missing))
        out["_placeholders_still_open_in_the_post"] = len(wanted)   # 0 once the post has taken its numbers from this file
    size = write_json(Path(args.out), out, compact=False)
    print(f"wrote {args.out} ({size / 1e3:.1f} kB): {sum(1 for k in out if not k.startswith('_') and k != 'results_lines')} placeholders, {len(answers)} results lines")
    for k, v in answers.items():
        print("\n" + k + "\n  " + v)


if __name__ == "__main__":
    main()
