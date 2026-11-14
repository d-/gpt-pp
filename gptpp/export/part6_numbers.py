"""The measured numbers of part 6 ("Optimization") that its four measured figures also show, for the prose:
`results/interp/part-6.json`.

    python -m gptpp.export.part6_numbers [--data <folder of the figures' exports>] [--out <file>]

The post's draft marks each number it still needs with a placeholder `@@words@@`. This writes, for every
placeholder that the exports of fig-06-03, fig-06-04, fig-06-05 and fig-06-07 can fill, the value, the text as
the prose would print it, how it was computed and whether it is final. It reads the same four files the figures
read (assets/data/gpt/learning-rate-schedules.json, optimizers-on-the-ladder.json, rare-tokens-under-sgd.json,
loss-landscape.json), so that figure and prose cannot disagree; run the four exporters first, and this again
whenever one of them has been run again.

    {"placeholders": [{"placeholder": "<the words between the marks>", "occurrence": 1, "value": ..., "text": "...",
                       "how": "...", "status": "final" | "stand-in ..." | "pending ..."}],
     "also": {...}}       # what the placeholders did not ask for and the prose may want

A value is null, and its status says which run is awaited, when the run has not finished. `occurrence` counts
the placeholders that share the same words, in the order of the post (the words "the same under Adam" occur
twice in Example 6.1.5).

One computation is not a lookup: the tokens after which a run is at a given held-out loss. The held-out loss is
measured every 10M tokens only. Between two evaluations it is read off the training loss, which is recorded
every ten steps on batches the model has not yet trained on: a three-record mean of the training loss, minus the
amount by which it stands above the held-out loss at each evaluation (interpolated between evaluations), so that
the curve passes through every evaluation; the answer is the first crossing of that curve, by linear
interpolation. The figure computes the same thing in the page (`heldOutCurve`, `tokensToReach` in
gpt-optimizers-on-the-ladder-figure.js), and its test compares the two.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .. import repo_dir
from .common import DATA


CHECK_LEVEL = 4.5  # a held-out loss every optimizer that trains at all passes: see `also.check`


def interp(xs: list[float], ys: list[float], x: float) -> float:
    if not xs:
        return 0.0
    if x <= xs[0]:
        return ys[0]
    for k in range(1, len(xs)):
        if x <= xs[k]:
            return ys[k - 1] + (ys[k] - ys[k - 1]) * (x - xs[k - 1]) / (xs[k] - xs[k - 1])
    return ys[-1]


def held_out_curve(run: dict, tps: int) -> tuple[list[float], list[float]]:
    """The held-out loss of a run between its evaluations, read off its training loss (see the module's text)."""
    steps, loss = run["train"]["steps"], run["train"]["loss"]
    x, previous = [], 0
    for s in steps:
        x.append((previous + s) / 2 * tps)
        previous = s
    n = len(x)
    smooth = [loss[k] if k in (0, n - 1) else (loss[k - 1] + loss[k] + loss[k + 1]) / 3 for k in range(n)]
    vx, vy = [s * tps for s in run["val"]["steps"]], run["val"]["loss"]
    ex, eo = [], []
    for xe, ye in zip(vx, vy):
        if xe > 0 and n:
            ex.append(xe)
            eo.append(interp(x, smooth, xe) - ye)
    points = [(xk, smooth[k] - (interp(ex, eo, xk) if ex else 0.0)) for k, xk in enumerate(x)] + list(zip(vx, vy))
    points.sort(key=lambda p: p[0])
    kept = [p for k, p in enumerate(points) if k == 0 or p[0] > points[k - 1][0] + 1e-6]
    return [p[0] for p in kept], [p[1] for p in kept]


def tokens_to_reach(curve: tuple[list[float], list[float]], level: float):
    xs, ys = curve
    for k, y in enumerate(ys):
        if not y <= level:
            continue
        if k == 0 or not ys[k - 1] > level:
            return xs[k]
        return xs[k - 1] + (ys[k - 1] - level) / (ys[k - 1] - y) * (xs[k] - xs[k - 1])
    return None


def millions(tokens) -> str:
    return "not reached" if tokens is None else f"{tokens / 1e6:.1f} million"


class Sheet:
    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.seen: dict[str, int] = {}

    def put(self, placeholder: str, value, text, how: str, status: str = "final") -> None:
        self.seen[placeholder] = self.seen.get(placeholder, 0) + 1
        self.rows.append({"placeholder": placeholder, "occurrence": self.seen[placeholder], "value": value,
                          "text": text, "how": how, "status": status})


def load(folder: Path, name: str):
    file = folder / f"{name}.json"
    return json.loads(file.read_text()) if file.exists() else None


def schedules(sheet: Sheet, data, also: dict) -> None:
    by = {r["key"]: r for r in data["runs"]} if data else {}

    def loss(key: str, run: str):
        r = by.get(key)
        if r is None:
            return None, f"pending: {run} has not finished", None
        status = f"stand-in: {r['run']}, the same configuration, until {r['stands_in_for']} finishes" if r.get("stand_in") else "final"
        if r["status"] != "completed":
            return None, f"{r['run']} stopped at {r.get('stopped_at_tokens')} tokens: {r.get('error')}", r
        return r["val_loss"], status, r

    how = "held-out loss in nats per token at the end of the run (summary.json of the run), 1,048,576 held-out tokens"
    for words, key, run in [
        ("held-out loss of the 2026 recipe on the ladder model at 50M tokens (run swap-base)", "trapezoid", "swap-base"),
        ("the same without warmup (run swap-no-warmup)", "no-warmup", "swap-no-warmup"),
        ("run swap-sched-cosine", "cosine", "swap-sched-cosine"),
        ("run swap-sched-linear", "linear", "swap-sched-linear"),
        ("run swap-sched-constant", "constant", "swap-sched-constant"),
    ]:
        value, status, _ = loss(key, run)
        sheet.put(words, value, None if value is None else f"{value:.3f}", how, status)
    parts, values, pending = [], {}, []
    for label, key, run in [("with warmup", "2019", "warm-2019-base"), ("without warmup", "2019-no-warmup", "warm-2019-none"),
                            ("normalization after each sublayer, with warmup", "2019-post", "warm-2019-post"),
                            ("normalization after each sublayer, without warmup", "2019-post-no-warmup", "warm-2019-post-none")]:
        value, status, r = loss(key, run)
        values[run] = value
        if value is None:
            pending.append(run if r is None else f"{run} (stopped)")
            parts.append(f"{label}: {'not finished' if r is None else 'stopped on a loss that is not finite'}")
        else:
            parts.append(f"{label}: {value:.3f}" + (f" ({r['run']})" if r.get("stand_in") else ""))
    sheet.put("held-out losses of runs warm-2019-none, warm-2019-post and warm-2019-post-none, and of the 2019 recipe with warmup at the same budget",
              values, "; ".join(parts), how + "; the 2019 recipe with warmup is warm-2019-base (core-probe-gpt2-lr24, the same "
              "configuration run earlier, stands in until it finishes)", "final" if not pending else "pending: " + ", ".join(pending))
    if data:
        also["schedule gaps (held-out loss at the end minus the run with its recipe's own schedule)"] = {
            r["key"]: round(r["val_loss"] - by[r["control"]]["val_loss"], 4) for r in data["runs"]
            if r["control"] and r["control"] in by and r["val_loss"] is not None and by[r["control"]]["val_loss"] is not None and not r.get("cut")}
        long = by.get("2019-long")
        if long:
            also["the 2019 recipe's 100M-token run at 50M tokens (ladder-0-gpt2-2019-s0; its cosine is twice as long)"] = long["val_loss"]


def optimizers(sheet: Sheet, data, also: dict) -> None:
    by = {o["key"]: o for o in data["optimizers"]} if data else {}
    tps = data["tokens_per_step"] if data else 0

    def best(key: str):
        o = by.get(key)
        return o["runs"][o["best"]] if o and o["best"] is not None else None

    def pending(key: str) -> str:
        o = by.get(key, {"runs": []})
        return f"pending: {3 - len(o['runs'])} of the three runs of {key} have not finished" if len(o["runs"]) < 3 else "final"

    adamw = best("adamw")
    level = adamw["val_loss"] if adamw else None
    how_loss = "held-out loss in nats per token at the end of the best of the three runs (lowest final held-out loss)"
    how_tokens = ("first crossing of the level (the final held-out loss of the best AdamW run) by the run's held-out curve, "
                  "which between the evaluations every 10M tokens follows the training loss (see the text of gptpp/export/part6_numbers.py)")
    names = {"sgd": "SGD", "adam": "Adam", "adamw": "AdamW", "lion": "Lion"}
    sgd, aw = best("sgd"), best("adamw")
    sheet.put("held-out loss, nats per token, of the best of the three SGD runs on the ladder model with the 2026 block, 50M tokens",
              sgd and sgd["val_loss"], sgd and f"{sgd['val_loss']:.3f}", how_loss, pending("sgd"))
    sheet.put("the same for the best of the three AdamW runs", aw and aw["val_loss"], aw and f"{aw['val_loss']:.3f}", how_loss, pending("adamw"))
    reached = {}
    for key, rate_words, loss_words, tokens_words in [
        ("sgd", "best of the three rates for SGD (runs opt-sgd-lr0.03, 0.1, 0.3)", "held-out loss of the best SGD run",
         'tokens at which the best SGD run reaches the final loss of the best AdamW run, or "not reached"'),
        ("adam", "best rate for Adam (runs opt-adam-lr*)", "held-out loss of the best Adam run", "tokens to AdamW's final loss, Adam"),
        ("adamw", "best rate for AdamW (runs opt-adamw-lr*)", "held-out loss of the best AdamW run", None),
        ("lion", "best rate for Lion (runs opt-lion-lr*)", "held-out loss of the best Lion run", "tokens to AdamW's final loss, Lion"),
    ]:
        run = best(key)
        sheet.put(rate_words, run and run["lr"], run and f"{run['lr']:g}", "the rate of the run with the lowest final held-out loss", pending(key))
        sheet.put(loss_words, run and run["val_loss"], run and f"{run['val_loss']:.3f}", how_loss, pending(key))
        if tokens_words:
            tokens = tokens_to_reach(held_out_curve(run, tps), level) if run and level is not None else None
            reached[key] = tokens
            ok = run is not None and level is not None
            sheet.put(tokens_words, tokens if ok else None, millions(tokens) if ok else None, how_tokens,
                      "final" if ok and pending(key) == "final" and pending("adamw") == "final" else "pending: " + (pending(key) if not run else pending("adamw")))
    recipe = best("muon")
    status = "pending: swap-base has not finished" if recipe is None else \
        f"stand-in: {recipe['run']}, the same configuration, until {recipe['stands_in_for']} finishes" if recipe.get("stand_in") else "final"
    sheet.put("held-out loss of the 2026 recipe at 50M tokens (run swap-base)", recipe and recipe["val_loss"],
              recipe and f"{recipe['val_loss']:.3f}", "held-out loss in nats per token at the end of the run", status)
    tokens = tokens_to_reach(held_out_curve(recipe, tps), level) if recipe and level is not None else None
    ok = recipe is not None and level is not None
    sheet.put("tokens at which the 2026 recipe reaches the final loss of the best AdamW run", tokens if ok else None,
              millions(tokens) if ok else None, how_tokens, status if ok and pending("adamw") == "final" else "pending: " + (pending("adamw") if recipe else "swap-base"))
    if data:
        also["rates of the 2026 recipe's Muon that have been run on this model at the present defaults"] = [r["lr"] for r in by["muon"]["runs"]]
        also["final held-out loss of every run of the sweep, by optimizer and rate"] = {
            o["key"]: {f"{r['lr']:g}": r["val_loss"] for r in o["runs"]} for o in data["optimizers"]}
        if ok and tokens:
            also["ratio of AdamW's 50M tokens to the tokens the 2026 recipe needs for AdamW's final loss"] = round(data["tokens"] / tokens, 3)
        # the same computation at a fixed level, for the figure's test to compare its own with (CHECK_LEVEL)
        also["check"] = {"level": CHECK_LEVEL, "tokens_to_level": {
            o["key"]: tokens_to_reach(held_out_curve(o["runs"][o["best"]], tps), CHECK_LEVEL) for o in data["optimizers"] if o["best"] is not None}}
        also["best rate at an end of its grid (the optimizer may be undertuned)"] = {
            o["key"]: o["best"] in (0, len(o["runs"]) - 1) for o in data["optimizers"] if o["best"] is not None and len(o["runs"]) > 1}


def rare(sheet: Sheet, data, also: dict) -> None:
    by = {o["key"]: o for o in data["optimizers"]} if data else {}

    def best(key: str):
        o = by.get(key)
        return o["runs"][o["best"]] if o and o["best"] is not None else None

    plain, clipped, adam, sign = best("sgdraw"), best("sgd"), best("adam"), best("sign")
    done = "final" if data and not data.get("missing") else "pending: no export" if not data else "pending: not scored yet: " + ", ".join(data["missing"])
    n_sgd = len(by.get("sgdraw", {"runs": []})["runs"]), len(by.get("sgd", {"runs": []})["runs"])
    n_adam = len(by.get("adam", {"runs": []})["runs"])
    sheet.put("the number of rates tried for SGD and for Adam in the runs freq-sgd-* and freq-adam-*",
              {"SGD, plain": n_sgd[0], "SGD, gradient clipped at norm one": n_sgd[1], "Adam": n_adam},
              f"{n_sgd[0]} rates for plain SGD, {n_sgd[1]} for SGD with the gradient clipped at norm one, {n_adam} for Adam" if data else None,
              "the runs freq-sgdraw-lr* (SGD with Nesterov momentum 0.9, the gradient as it is), freq-sgd-lr* (the same with the "
              "gradient norm clipped to 1) and freq-adam-lr*; every rate is half a decade from the next, and each grid was "
              "extended until its best rate had a neighbor on both sides", done)
    if data:
        b = data["buckets"]
        sheet.put("the number of frequency buckets of the export, and how they are cut (equal shares of the held-out tokens, as in Kunstner et al., or by rank)",
                  b["n"], f"{b['n']} buckets of about equal share of the training tokens (between {100 * min(b['mass']):.1f}% and {100 * max(b['mass']):.1f}% each): "
                  f"the vocabulary is sorted by frequency in the training set and cut where the running share passes each tenth, as in Kunstner et al.; "
                  f"the most frequent bucket is {b['tokens'][0]} tokens, the rarest {b['tokens'][-1]:,}",
                  "token counts over all training shards; a token belongs to the tenth that holds the middle of its own share; "
                  "the loss of a bucket is the mean held-out loss over the held-out tokens whose target is in it "
                  f"(between {min(b['held_out_tokens']):,} and {max(b['held_out_tokens']):,} of the {data['held_out_tokens']:,} held-out tokens a bucket)")
    else:
        sheet.put("the number of frequency buckets of the export, and how they are cut (equal shares of the held-out tokens, as in Kunstner et al., or by rank)",
                  None, None, "", "pending: no export")
    how = "mean held-out loss in nats of the tokens of the bucket at the last checkpoint of the run with the best rate"
    sgd = plain or clipped
    which = "plain SGD (no clipping)" if plain else "SGD with the gradient clipped"
    for words, run, index in [
        ("held-out loss of the most frequent bucket under SGD at its best rate", sgd, 0),
        ("the same under Adam", adam, 0),
        ("held-out loss of the rarest bucket under SGD", sgd, -1),
        ("the same under Adam", adam, -1),
    ]:
        value = run["bucket_loss"][-1][index] if run else None
        sheet.put(words, value, None if value is None else f"{value:.2f}",
                  how + (f"; SGD here is {which}, rate {sgd['lr']:g}" if run is sgd and run else f"; Adam at rate {adam['lr']:g}" if run else ""),
                  done if run else "pending: no finished run")
    if data:
        table = {}
        for o in data["optimizers"]:
            run = best(o["key"])
            if run:
                last = run["bucket_loss"][-1]
                table[o["label"]] = {"rates tried": [r["lr"] for r in o["runs"]], "final loss at each rate": [r["val_loss"] for r in o["runs"]],
                                     "best rate": run["lr"], "best rate is inside its grid": o["interior"], "all tokens": run["val_loss"],
                                     "loss by bucket, most frequent first": [round(v, 3) for v in last]}
        also["loss by frequency bucket at the end, every optimizer at its best rate"] = table
        weights = data["buckets"]["held_out_tokens"]
        for label, run in (("plain SGD (no clipping)", plain), ("SGD with the gradient clipped", clipped), ("sign descent with momentum", sign)):
            if run and adam:
                gap = [a - b2 for a, b2 in zip(run["bucket_loss"][-1], adam["bucket_loss"][-1])]
                total = sum(g * w for g, w in zip(gap, weights))
                rare = sum(g * w for g, w in list(zip(gap, weights))[len(gap) // 2:])
                also[f"{label} minus Adam at the end, each at its best rate"] = {
                    "by bucket, most frequent first": [round(g, 3) for g in gap],
                    "over all tokens": round(run["val_loss"] - adam["val_loss"], 4),
                    "share of that gap made up by the rarer half of the buckets (each bucket's gap weighted by its held-out tokens)":
                        round(rare / total, 4) if total > 0 else None}
        if clipped and plain:
            also["SGD with the gradient clipped minus plain SGD at the same rate, final held-out loss (where both were run)"] = {
                f"{c['lr']:g}": round(c["val_loss"] - p["val_loss"], 4) for c in by["sgd"]["runs"] for p in by["sgdraw"]["runs"]
                if c["lr"] == p["lr"] and c["val_loss"] is not None and p["val_loss"] is not None}


def landscape(sheet: Sheet, data, also: dict) -> None:
    if not data:
        for words in ("the two directions of the landscape plane, how they are chosen and how they are normalized", "grid size and extent of the plane",
                      "number of checkpoints on the path", "held-out loss at the first checkpoint of the projected path"):
            sheet.put(words, None, None, "", "pending: no export")
        return
    g, path = data["grid"], data["path"]
    sheet.put("the two directions of the landscape plane, how they are chosen and how they are normalized", None,
              "the unit vector from the trained weights to the initialization, and the first principal direction of what is left of the "
              "checkpoints' differences from the trained weights once their component along that vector is removed; distances are measured "
              "after each row of the embedding matrix is divided by its root mean square (the model normalizes each embedding vector, so its "
              f"length changes nothing, and it grows {data['embedding_rms']['trained'] / data['embedding_rms']['init']:.0f}-fold in training), "
              f"and both axes are in units of the distance L = {data['L']:.0f} between the two ends",
              "gptpp/export/loss_landscape.py: `plane` (the directions), `flatten` (the row rescaling)")
    sheet.put("grid size and extent of the plane", [g["nx"], g["ny"]],
              f"{g['nx']} by {g['ny']} points, from {g['a'][0]:+.2f} L to {g['a'][1]:+.2f} L along the first direction (the trained weights at 0, the "
              f"initialization at 1) and from {g['b'][0]:+.2f} L to {g['b'][1]:+.2f} L along the second, {g['h']:.3f} L apart, each loss on "
              f"{data['held_out_tokens']:,} held-out tokens", "the grid of gptpp/export/loss_landscape.py")
    sheet.put("number of checkpoints on the path", len(path), str(len(path)), "the initialization, 19 log-spaced steps and the end (the run's --save-log 20)")
    sheet.put("held-out loss at the first checkpoint of the projected path", path[0]["loss"], f"{path[0]['loss']:.3f}",
              f"the loss of the initialization, which lies on the plane, on the trainer's {data.get('path_held_out_tokens', data['held_out_tokens']):,} "
              f"held-out tokens (the uniform guess over {data['vocab_size']:,} tokens is {data['uniform_loss']:.3f})")
    also["landscape"] = {
        "loss at the last checkpoint": path[-1]["loss"], "loss the trainer logged at the end": data.get("logged_final_loss"),
        "share of the squared distances of the checkpoints from the trained weights that lies in the plane": data["share"]["displacement"],
        "the same along the first direction alone": data["share"]["displacement_e1"],
        "share of the squared moves from one checkpoint to the next that lies in the plane": data["share"]["steps"],
        "share of the path's length that lies in the plane": data["share"]["length"],
        "length of the path through the checkpoints, in units of L": data["share"]["path_length_over_L"],
        "the same shares for two free principal directions (Li et al., 2018)": data["pca"],
        "raw weights, without the row rescaling: share of the squared distance from initialization to trained weights that is in the embedding matrix": data["raw_embedding_share_of_distance"],
        "largest loss on the grid": g["max"], "lowest loss on the grid": g["min"],
        "checkpoint farthest from the plane": max(({"step": p["step"], "distance in L": p["off"]} for p in path), key=lambda q: q["distance in L"]),
        "largest difference between the plane under a checkpoint and the checkpoint's own loss":
            max(({"step": p["step"], "plane": p["plane_loss"], "own": p["loss"]} for p in path), key=lambda q: abs(q["plane"] - q["own"])),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--out", default=str(repo_dir() / "results" / "interp" / "part-6.json"))
    args = ap.parse_args()
    folder = Path(args.data)
    sheet, also = Sheet(), {}
    optimizers(sheet, load(folder, "optimizers-on-the-ladder"), also)
    rare(sheet, load(folder, "rare-tokens-under-sgd"), also)
    schedules(sheet, load(folder, "learning-rate-schedules"), also)
    landscape(sheet, load(folder, "loss-landscape"), also)
    out = {
        "post": "_posts/gpt/2026-11-14-gpt-06.md", "written_by": "python -m gptpp.export.part6_numbers",
        "note": "one entry per placeholder @@...@@ of the post that the four measured figures' exports can fill; `occurrence` counts "
                "placeholders with the same words in the order of the post; losses are held-out, in nats per token; one seed a run",
        "placeholders": sheet.rows, "also": also,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    for r in sheet.rows:
        r["status"] = r["status"].replace("pending: pending:", "pending:")
    done = sum(1 for r in sheet.rows if r["status"] == "final")
    print(f"{args.out}: {len(sheet.rows)} placeholders, {done} final")
    for r in sheet.rows:
        if r["status"] != "final":
            print(f"  {r['status'][:90]:<92} {r['placeholder'][:70]}")


if __name__ == "__main__":
    main()
