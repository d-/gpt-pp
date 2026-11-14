"""Fit the scaling law L(N, D) = E + A / N^alpha + B / D^beta to our own grid of runs, for the figure
`gpt-scaling-surface-3d` (fig-07-05): `assets/data/gpt/scaling-surface.json`.

    python -m gptpp.export.scaling_surface [--runs results/runs] [--out <dir>] [--bootstrap 500] [--starts full|fast]
    python -m gptpp.export.scaling_surface --synthetic [--seed 0] [--noise 0.004] --out <a scratch folder>
    python -m gptpp.export.scaling_surface --first --out <a scratch folder>      # the first grid, for the record

It takes the cells of the grid that have finished (six sizes by four budgets of FineWeb-EDU in the 2026 recipe at
context 512, one seed, the largest size not run at the largest budget: 23 cells), says how many of them it holds,
and can be run again when more land. With fewer than eight cells, four sizes or three budgets no fit is reported
(five constants on a handful of points determine nothing, and two sizes cannot give the three constants in N
whatever the number of cells). `--synthetic` writes cells generated from a known law on the grid's own sizes and
budgets, marked `synthetic: true`, for testing the fit and for developing the figure before the grid has run; never
leave such a file in the site's data folder.

The runs. The grid's commands start from the preset of the 512-wide model and set the width (`--preset gptpp-ladder
--set d_model=128 ...`). The recipe trains the embedding and the unembedding at 0.3 and 0.008 times sqrt(768 /
width), and until 14 November 2026 the trainer applied that rule to the preset's width only: the first grid,
`grid-w<width>-l<layers>-<tokens>M`, trained the two tables at the rates of width 512 at every width
(docs/REVIEW.md). The 19 runs that are not 512 wide were made again by the same commands with the rule applied,
`gridr-w<width>-l<layers>-<tokens>M`, and those are the cells of the fit, with `grid-w512-l8-<tokens>M` for the
preset's own width, which the rule never changed. A cell whose recorded rates are not the rule's is refused. The
first grid stays in results/runs as the record of what was run, and the export carries it as `first_grid`: the 19
losses that were replaced and the constants of the same fit on its 23 runs. `--first` fits the first grid in place
of the grid (the export as it stood before); like a synthetic file, it must not stand in the site's data folder.

The fit is that of Hoffmann et al. (2022, Appendix D.2) with the corrections of Besiroglu et al. (2024): minimize over
(a, b, e, alpha, beta) the SUM over cells of Huber_delta(logsumexp(a - alpha ln N, b - beta ln D, e) - ln L), delta =
1e-3, with L-BFGS from the grid of starts alpha, beta in {0, 0.5, ..., 2}, e in {-1, -0.5, ..., 1}, a, b in {0, 5, ...,
25} (4,500 starts), then polished; (A, B, E) = (e^a, e^b, e^e). It is done once for each way of counting N:

    blocks     the attention and MLP matrices of the blocks, 12 L d^2
    output     those and the unembedding: the parameters that cost arithmetic (the count of 6 N D in the post)
    all        every parameter, with the embedding

Losses are the trainer's final held-out loss in nats per token (1,048,576 fixed tokens, windows of 512).

`scaling-surface.json`, field by field:

    generated_by     "gptpp.export.scaling_surface"
    synthetic        true only for --synthetic (then `truth` holds the law the cells were drawn from)
    expected         the 23 run names of the grid (`gridr-*`, and `grid-w512-*`); `missing`: those without a summary yet
    cells[k]         {run, layers, width, blocks, output, all (parameters under the three counts), tokens, steps,
                      warmup_steps, val_loss, val_bpb, best_val_loss, seconds, tokens_per_second, lr_embedding,
                      lr_unembedding (the AdamW rates the run trained its two tables at, from its config.json)}
    width_rule       whether every cell trained its two tables at the recipe's rates for its width (false only under
                     --first: otherwise such a cell is refused)
    first_grid       the grid as it was first run; null while no run made again has finished, in a synthetic file and
                     under --first:
      runs                        how many of its 23 runs were found
      rates                       {lr_embedding, lr_unembedding, width}: the rates of the two tables in those runs, the
                                  same at every width, and the width whose rates they are by the rule; null if they differ
      replaced[k]                 {run, by, width, tokens, val_loss, moved}: a run of the first grid that was made
                                  again, the run that replaced it, and `moved` = the new loss minus the old
      fits.<count>                the same fit on the first grid's 23 runs, without intervals: {alpha, beta, E, A, B,
                                  a, max_rel_residual, predicted (at the main run), frontier}; null without enough runs
    counts           [{key, label}] in the order blocks, output, all
    enough           whether the cells can carry a fit: at least min_cells (8) of them, of at least min_sizes (4) sizes
                     and min_budgets (3) budgets
    fits.<count>     null unless `enough`; else
      alpha, beta, E, A, B        the constants (alpha, beta to four significant digits and beyond: unrounded)
      a                           beta / (alpha + beta): the exponent of the compute-optimal size, N_opt ~ C^a
      objective, grad_norm        the minimum of the sum of Huber losses, and the gradient's norm there
      starts, polished, agree     how many starts were tried (each for 300 iterations), how many of the best were then
                                  run to the end (48), and how many of those ended within 1e-6 (relative) of the minimum
      second_optimizer            the objective Nelder-Mead reaches from the solution (a cross-check: not lower)
      at_bound                    true if alpha or beta ended on its bound (0 or 5): then the exponent is not determined
      residuals[k]                ln(fitted) - ln(measured) for cell k; max_rel_residual and its run: the largest
                                  |fitted - measured| / measured
      ci                          percentile intervals over `bootstrap` resamples of the cells, each refitted from the
                                  best distinct solutions of the fit: {alpha, beta, E, a: {lo80, hi80, lo95, hi95}}; null
                                  if bootstrap is 0
      main                        the fitted law at the main run: {N, D, C (6 N D with N in this count), predicted,
                                   measured_512, measured_1024, error_512, error_1024 (relative; null until measured)}
      frontier                    the compute-optimal allocation of the fitted law at the main run's compute, with
                                  C = 6 N D in this count: {C, N_opt, D_opt, ratio (tokens per parameter), excess
                                  (reducible loss of the main run's allocation over that of the optimum, minus 1)}
    extras[k]        points that are not in the fit: the main run, the rungs of the ladder, the race's 2026 run:
                     {key, label, kind, layers, width, context, blocks, output, all, tokens, val_loss, final, note}
    caveats          the limits of the fit, as short sentences the figure and the post can print
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from .. import repo_dir
from ..config import preset, train_preset
from .common import CKPT, DATA, write_atomic

NAME = "scaling-surface"
DELTA = 1e-3
VOCAB = 32768
SIZES = [(128, 4), (192, 6), (256, 8), (384, 8), (512, 8), (768, 12)]
BUDGETS = [25, 50, 100, 200]
PRESET = "gptpp-ladder"  # the preset the grid's commands start from; its own width, 512, was never set on the command line
CELL = re.compile(r"gridr?-w(\d+)-l(\d+)-(\d+)M$")  # a run of the grid, or of the first grid: width, layers, budget
COUNTS = [("blocks", "the blocks"), ("output", "the blocks and the output layer"), ("all", "every parameter")]
WORKERS = max(1, min(32, (os.cpu_count() or 2) // 2))  # processes that share the starts of a fit and the resampled fits
MIN_CELLS, MIN_SIZES, MIN_BUDGETS = 8, 4, 3  # no fit on less: the law has five constants, three of them in N alone


def enough(cells: list[dict]) -> bool:
    """Whether the cells can carry a fit: at least eight, of at least four sizes and three budgets. Two sizes give the
    law's three constants in N (E, A, alpha) two levels to read them from, whatever the number of cells."""
    return (len(cells) >= MIN_CELLS and len({(c["width"], c["layers"]) for c in cells}) >= MIN_SIZES
            and len({round(c["tokens"] / 25e6) for c in cells}) >= MIN_BUDGETS)


def expected(first: bool = False) -> list[str]:
    """The 23 runs of the grid: `gridr-*`, made with the width rule applied, and `grid-w512-*` at the preset's own
    width. `first`: the 23 runs of the first grid, `grid-*` at every width (the module's docstring has the story)."""
    own = preset(PRESET).d_model
    return [f"{'grid' if first or w == own else 'gridr'}-w{w}-l{l}-{t}M" for t in BUDGETS for (w, l) in SIZES if not (w == 768 and t == 200)]


def rule_rates(width: int) -> tuple[float, float]:
    """The AdamW rates of the embedding and the unembedding in the 2026 recipe at a width: 0.3 and 0.008 times
    sqrt(768 / width), as the trainer rounds them."""
    cfg = train_preset(PRESET, d_model=width)
    return cfg.lr_embedding, cfg.lr_unembedding


def counts_of(params: dict) -> dict:
    blocks = params["attention"] + params["mlp"]
    return {"blocks": blocks, "output": blocks + params["unembedding"], "all": params["total"]}


def formula_counts(width: int, layers: int) -> dict:
    """The three counts of the 2026 recipe by formula: no biases, no norm parameters, rotary positions."""
    blocks = 12 * layers * width * width
    return {"blocks": blocks, "output": blocks + VOCAB * width, "all": blocks + 2 * VOCAB * width}


def read_run(runs: Path, run: str) -> dict | None:
    for root in (runs, CKPT):
        summary, config = root / run / "summary.json", root / run / "config.json"
        if summary.exists() and config.exists():
            s, c = json.loads(summary.read_text()), json.loads(config.read_text())
            if s.get("status") != "completed":
                return None
            m, t, d = c["model_config"], c["train_config"], c["derived"]
            out = {"run": run, "layers": m["n_layer"], "width": m["d_model"], "context": d["seq_len"], "tokens": int(s["tokens"]), "steps": int(s["steps"]),
                   "warmup_steps": int(round(t["warmup_tokens"] / d["tokens_per_step"])), "val_loss": s["val_loss"], "val_bpb": s["val_bpb"],
                   "best_val_loss": s["best_val_loss"], "seconds": s["seconds"], "tokens_per_second": s["tokens_per_second"], "n_gpu": s.get("n_gpu", 1)}
            out.update(counts_of(s["params"]))
            groups = d.get("groups", {})  # what the run's optimizers were built with: the rates as trained, not as asked for
            out.update(lr_embedding=groups.get("embedding", {}).get("lr"), lr_unembedding=groups.get("unembedding", {}).get("lr"))
            return out
    return None


def off_the_rule(cells: list[dict]) -> list[str]:
    """The cells whose two tables were not trained at the recipe's rates for their width, each with what it had."""
    return [f"{c['run']} (width {c['width']}: {c['lr_embedding']} and {c['lr_unembedding']}, the rule gives {' and '.join(map(str, rule_rates(c['width'])))})"
            for c in cells if (c["lr_embedding"], c["lr_unembedding"]) != rule_rates(c["width"])]


# ----------------------------------------------------------------------------------------------------------------
# the fit


def predict(theta: np.ndarray, n: np.ndarray, d: np.ndarray) -> np.ndarray:
    """ln L of the law at ln N = n, ln D = d; theta = (a, b, e, alpha, beta)."""
    a, b, e, alpha, beta = theta
    stack = np.stack([a - alpha * n, b - beta * d, np.full_like(n, e, dtype=np.float64)])
    m = stack.max(axis=0)
    return m + np.log(np.exp(stack - m).sum(axis=0))


def objective(theta: np.ndarray, n: np.ndarray, d: np.ndarray, y: np.ndarray, delta: float = DELTA) -> tuple[float, np.ndarray]:
    """The sum of Huber losses on the residuals of ln L, and its gradient."""
    a, b, e, alpha, beta = theta
    t = np.stack([a - alpha * n, b - beta * d, np.full_like(n, e, dtype=np.float64)])
    m = t.max(axis=0)
    lse = m + np.log(np.exp(t - m).sum(axis=0))
    r = lse - y
    small = np.abs(r) <= delta
    value = np.where(small, 0.5 * r * r, delta * (np.abs(r) - 0.5 * delta)).sum()
    psi = np.clip(r, -delta, delta)
    w = np.exp(t - lse)
    grad = np.array([(psi * w[0]).sum(), (psi * w[1]).sum(), (psi * w[2]).sum(), -(psi * w[0] * n).sum(), -(psi * w[1] * d).sum()])
    return float(value), grad


BOUNDS = [(None, None), (None, None), (None, None), (0.0, 5.0), (0.0, 5.0)]
SCALE = 1.0 / DELTA**2  # the optimizer sees the objective in units of delta^2, so that its tolerances mean something


def _run(theta0: np.ndarray, n: np.ndarray, d: np.ndarray, y: np.ndarray, maxiter: int = 2000) -> tuple[float, np.ndarray]:
    def f(theta):
        v, g = objective(theta, n, d, y)
        return SCALE * v, SCALE * g
    res = minimize(f, theta0, jac=True, method="L-BFGS-B", bounds=BOUNDS, options={"maxiter": maxiter, "ftol": 1e-15, "gtol": 1e-10, "maxcor": 20})
    return res.fun / SCALE, res.x


# Runs whose scatter is far above delta make the Huber loss nearly a sum of absolute values, on which L-BFGS creeps: a
# start can take its 2,000 iterations. So the starts are first run for SWEEP iterations each, in parallel, and only
# the best of them to the end; the resampled fits are divided among the same workers.
SWEEP, POLISH = 300, 48


def _sweep(job: tuple) -> list[tuple[float, np.ndarray]]:
    starts, n, d, y, maxiter = job
    return [_run(s, n, d, y, maxiter) for s in starts]


def _resampled(job: tuple) -> list[float]:
    pool, n, d, y = job
    value, theta = min((_run(s, n, d, y, SWEEP) for s in pool), key=lambda r: r[0])
    value2, theta2 = _run(theta, n, d, y)
    c = describe(theta2 if value2 < value else theta)
    return [c["alpha"], c["beta"], c["E"], c["a"] if c["a"] is not None else float("nan")]


def _map(function, jobs: list, workers: int) -> list:
    """`function` over `jobs`, in order, on `workers` processes (in this process when there is one worker or one job)."""
    if workers <= 1 or len(jobs) <= 1:
        return [function(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=multiprocessing.get_context("fork")) as executor:
        return list(executor.map(function, jobs))


def start_grid(kind: str) -> list[np.ndarray]:
    if kind == "full":  # the grid of Hoffmann et al., Appendix D.2
        return [np.array([a, b, e, al, be], dtype=np.float64) for al in (0, 0.5, 1, 1.5, 2) for be in (0, 0.5, 1, 1.5, 2)
                for e in (-1, -0.5, 0, 0.5, 1) for a in (0, 5, 10, 15, 20, 25) for b in (0, 5, 10, 15, 20, 25)]
    return [np.array([a, b, e, al, be], dtype=np.float64) for al in (0.1, 0.5, 1) for be in (0.1, 0.5, 1) for e in (0, 0.5, 1) for a in (0, 5, 10, 20) for b in (0, 5, 10, 20)]


def fit(N: np.ndarray, D: np.ndarray, L: np.ndarray, starts: list[np.ndarray], keep: int = 12, workers: int = WORKERS) -> dict:
    """The minimum over all starts, polished; with the solutions that tie it and a cross-check by a second optimizer.

    Every start is run for SWEEP iterations; the POLISH best of them are then run to the end, and `agree` counts
    how many of those end at the minimum."""
    n, d, y = np.log(N), np.log(D), np.log(L)
    size = max(1, -(-len(starts) // (4 * max(1, workers))))
    swept = [end for chunk in _map(_sweep, [(starts[i:i + size], n, d, y, SWEEP) for i in range(0, len(starts), size)], workers) for end in chunk]
    swept.sort(key=lambda r: r[0])
    ends = sorted((end for chunk in _map(_sweep, [([theta], n, d, y, 2000) for _, theta in swept[:POLISH]], workers) for end in chunk), key=lambda r: r[0])
    best_value, best = ends[0]
    for _ in range(3):  # polish: L-BFGS again from the solution, until it stops moving
        value, theta = _run(best, n, d, y)
        if value < best_value - 1e-18:
            best_value, best = value, theta
    agree = sum(1 for v, _ in ends if v <= best_value * (1 + 1e-6) + 1e-18)
    second = minimize(lambda th: SCALE * objective(th, n, d, y)[0], best, method="Nelder-Mead",
                      options={"xatol": 1e-10, "fatol": 1e-14, "maxiter": 20000, "maxfev": 40000})
    _, grad = objective(best, n, d, y)
    free = np.array([not ((k == 3 or k == 4) and (best[k] <= 1e-9 or best[k] >= 5 - 1e-9)) for k in range(5)])
    distinct = []
    for v, th in ends:  # a few distinct good solutions: the starts of the bootstrap
        if len(distinct) >= keep:
            break
        if all(np.abs(th - q).max() > 1e-3 for q in distinct):
            distinct.append(th)
    return {"theta": best, "objective": best_value, "grad_norm": float(np.linalg.norm(grad[free])), "starts": len(starts), "polished": len(ends), "agree": agree,
            "second_optimizer": float(second.fun / SCALE), "at_bound": bool((~free).any()), "pool": [best] + distinct}


def describe(theta: np.ndarray) -> dict:
    a, b, e, alpha, beta = (float(v) for v in theta)
    return {"alpha": alpha, "beta": beta, "E": math.exp(e), "A": math.exp(a), "B": math.exp(b), "a": beta / (alpha + beta) if alpha + beta > 0 else None,
            "ln_A": a, "ln_B": b, "ln_E": e}


def law(c: dict, N: float, D: float) -> float:
    return c["E"] + c["A"] * N ** -c["alpha"] + c["B"] * D ** -c["beta"]


def frontier(c: dict, C: float) -> dict | None:
    """The allocation that minimizes the law at 6 N D = C (Hoffmann et al., eq. 4)."""
    if not (c["alpha"] > 0 and c["beta"] > 0):
        return None
    G = (c["alpha"] * c["A"] / (c["beta"] * c["B"])) ** (1 / (c["alpha"] + c["beta"]))
    N = G * (C / 6) ** (c["beta"] / (c["alpha"] + c["beta"]))
    D = (C / 6) / N
    return {"C": C, "N_opt": N, "D_opt": D, "ratio": D / N, "reducible": law(c, N, D) - c["E"]}


def bootstrap(N: np.ndarray, D: np.ndarray, L: np.ndarray, pool: list[np.ndarray], resamples: int, seed: int = 0, workers: int = WORKERS) -> dict | None:
    if resamples <= 0:
        return None
    rng, n, d, y = np.random.default_rng(seed), np.log(N), np.log(D), np.log(L)
    picks = [rng.integers(0, len(N), len(N)) for _ in range(resamples)]
    rows = np.array(_map(_resampled, [(pool, n[pick], d[pick], y[pick]) for pick in picks], workers))
    out = {"resamples": resamples}
    for k, key in enumerate(("alpha", "beta", "E", "a")):
        col = rows[:, k][np.isfinite(rows[:, k])]
        lo95, lo80, med, hi80, hi95 = np.percentile(col, [2.5, 10, 50, 90, 97.5])
        out[key] = {"lo95": float(lo95), "lo80": float(lo80), "median": float(med), "hi80": float(hi80), "hi95": float(hi95)}
    return out


# ----------------------------------------------------------------------------------------------------------------
# the export


def measured_main() -> dict:
    """The main run's held-out loss: its own monitor (windows of 1,024) and the same tokens in windows of 512."""
    out = {"measured_1024": None, "measured_512": None, "final": False, "tokens": None}
    for root in (repo_dir() / "results" / "runs", CKPT):
        summary = root / "gptpp-base" / "summary.json"
        if summary.exists():
            s = json.loads(summary.read_text())
            out.update(measured_1024=s["val_loss"], final=True, tokens=int(s["tokens"]))
            break
    measured = repo_dir() / "results" / "interp" / "part-7-measured.json"
    if measured.exists():
        m = json.loads(measured.read_text())
        if m.get("model", {}).get("run") == "gptpp-base" and m.get("model", {}).get("checkpoint") == "model.pt" and "val" in m:
            out["measured_512"] = m["val"]["w512"]["val_loss"]
    return out


def extras(runs: Path) -> list[dict]:
    out = []
    main = read_run(runs, "gptpp-base")
    if main:
        out.append({**{k: main[k] for k in ("layers", "width", "context", "blocks", "output", "all", "tokens", "val_loss")}, "key": "gptpp-base", "label": "gpt-pp",
                    "kind": "main", "final": True, "note": "the main run: context 1,024, 147,456 tokens a step on three cards"})
    else:  # still running: its size is known, its loss is not final
        cfg = CKPT / "gptpp-base" / "config.json"
        if cfg.exists():
            c = json.loads(cfg.read_text())
            out.append({"key": "gptpp-base", "label": "gpt-pp", "kind": "main", "layers": c["model_config"]["n_layer"], "width": c["model_config"]["d_model"],
                        "context": c["derived"]["seq_len"], **counts_of(c["params"]), "tokens": int(c["derived"]["tokens"]), "val_loss": None, "final": False,
                        "note": "the main run, still training: no final loss yet"})
    rung = re.compile(r"ladder-(\d+)-(.+)-s(\d+)$")  # a rung and its seed; not a rung run again some other way (ladder-*-w32M)
    names = sorted(p.name for p in runs.glob("ladder-*") if rung.match(p.name) and (p / "summary.json").exists())
    for run in names + ["race-2026-100M", "gptpp-base-1B"]:
        r = read_run(runs, run)
        if not r:
            continue
        m = rung.match(run)
        out.append({**{k: r[k] for k in ("layers", "width", "context", "blocks", "output", "all", "tokens", "val_loss")}, "key": run,
                    "label": f"ladder, rung {m.group(1)}: {m.group(2)}" + (f", seed {m.group(3)}" if m.group(3) != "0" else "") if m else
                    "gpt-pp's recipe at its shape, its schedule laid out for 1.08B tokens" if run == "gptpp-base-1B" else "the 2026 recipe at gpt-pp's shape, 100M tokens",
                    "kind": "ladder" if m else "race", "final": True,
                    "note": ("a rung of the ladder: another recipe unless it is the last rung" if m else "context 1,024, 49,152 tokens a step")})
    return out


def synthetic_cells(seed: int, noise: float, truth: dict) -> list[dict]:
    rng, cells = np.random.default_rng(seed), []
    for run in expected():
        w, l, t = (int(v) for v in CELL.match(run).groups())
        c = formula_counts(w, l)
        steps = round(t * 1e6 / 65536)
        tokens = steps * 65536
        loss = law(truth, c[truth["count"]], tokens) * math.exp(noise * rng.standard_normal())
        rates = rule_rates(w)
        cells.append({"run": run, "layers": l, "width": w, "context": 512, **c, "tokens": tokens, "steps": steps, "warmup_steps": 31, "val_loss": loss,
                      "val_bpb": loss / math.log(2) / 4.628316806198068, "best_val_loss": loss, "seconds": None, "tokens_per_second": None, "n_gpu": 1,
                      "lr_embedding": rates[0], "lr_unembedding": rates[1]})
    return cells


def fit_cells(cells: list[dict], key: str, starts: list[np.ndarray], resamples: int, main_counts: dict, main_tokens: int, main_measured: dict) -> dict | None:
    """The law fitted to the cells with N counted as `key`, as the export holds it (`fits.<count>`); None on too few cells."""
    if not enough(cells):
        return None
    N, D, L = (np.array([c[k] for c in cells], dtype=np.float64) for k in (key, "tokens", "val_loss"))
    f = fit(N, D, L, starts)
    c = describe(f["theta"])
    resid = predict(f["theta"], np.log(N), np.log(D)) - np.log(L)
    rel = np.exp(resid) - 1
    worst = int(np.abs(rel).argmax())
    Nm, Dm = float(main_counts[key]), float(main_tokens)
    pred = law(c, Nm, Dm)
    C = 6 * Nm * Dm
    fr = frontier(c, C)
    if fr:
        fr["excess"] = (pred - c["E"]) / fr["reducible"] - 1
    return {**c, "objective": f["objective"], "grad_norm": f["grad_norm"], "starts": f["starts"], "polished": f["polished"], "agree": f["agree"], "second_optimizer": f["second_optimizer"],
            "at_bound": f["at_bound"], "residuals": resid.tolist(), "max_rel_residual": float(np.abs(rel).max()), "max_rel_residual_run": cells[worst]["run"],
            "rms_rel_residual": float(np.sqrt((rel**2).mean())), "ci": bootstrap(N, D, L, f["pool"], resamples),
            "main": {"N": Nm, "D": Dm, "C": C, "predicted": pred, "measured_512": main_measured["measured_512"], "measured_1024": main_measured["measured_1024"],
                     "error_512": pred / main_measured["measured_512"] - 1 if main_measured["measured_512"] else None,
                     "error_1024": pred / main_measured["measured_1024"] - 1 if main_measured["measured_1024"] else None},
            "frontier": fr}


def first_grid(runs: Path, cells: list[dict], fit_of) -> dict | None:
    """The grid as it was first run, beside the cells of the fit: the losses that were replaced and the constants of
    the same fit on its runs (`fit_of(cells, key)`). None when no cell of the fit replaces a run of the first grid."""
    old = [c for c in (read_run(runs, run) for run in expected(first=True)) if c]
    new = {CELL.match(c["run"]).groups(): c for c in cells}
    replaced = []
    for c in old:
        by = new.get(CELL.match(c["run"]).groups())
        if by is not None and by["run"] != c["run"]:
            replaced.append({"run": c["run"], "by": by["run"], "width": c["width"], "tokens": c["tokens"], "val_loss": c["val_loss"], "moved": by["val_loss"] - c["val_loss"]})
    if not replaced:
        return None
    rates, own = {(c["lr_embedding"], c["lr_unembedding"]) for c in old}, preset(PRESET).d_model
    one = rates.pop() if len(rates) == 1 else None  # one pair of rates at every width, and the width whose rates they are
    fits = {}
    for key, _ in COUNTS:
        f = fit_of(old, key)
        fits[key] = f and {**{k: f[k] for k in ("alpha", "beta", "E", "A", "B", "a", "max_rel_residual")}, "predicted": f["main"]["predicted"], "frontier": f["frontier"]}
    return {"runs": len(old), "rates": one and {"lr_embedding": one[0], "lr_unembedding": one[1], "width": own if one == rule_rates(own) else None},
            "replaced": replaced, "fits": fits}


def export(runs: Path, resamples: int, starts_kind: str, synthetic: dict | None = None, first: bool = False) -> dict:
    names = expected(first)
    if synthetic:
        cells, missing = synthetic_cells(synthetic["seed"], synthetic["noise"], synthetic["truth"]), []
        extra, main_measured = [], {"measured_1024": None, "measured_512": None, "final": False, "tokens": None}
        main_counts = formula_counts(768, 12)
        main_tokens = 2999992320
    else:
        cells = [c for c in (read_run(runs, run) for run in names) if c]
        missing = [run for run in names if run not in {c["run"] for c in cells}]
        if not first and off_the_rule(cells):
            raise ValueError("the fit takes runs whose two tables trained at the recipe's rates for their width, and these did not: " + "; ".join(off_the_rule(cells)))
        extra, main_measured = extras(runs), measured_main()
        main_extra = next((e for e in extra if e["kind"] == "main"), None)
        main_counts = {k: main_extra[k] for k in ("blocks", "output", "all")} if main_extra else formula_counts(768, 12)
        main_tokens = main_extra["tokens"] if main_extra else 2999992320
    starts = start_grid(starts_kind)
    fits = {key: fit_cells(cells, key, starts, resamples, main_counts, main_tokens, main_measured) for key, _ in COUNTS}
    before = None if synthetic or first else first_grid(runs, cells, lambda old, key: fit_cells(old, key, starts, 0, main_counts, main_tokens, main_measured))
    steps = [c["steps"] for c in cells]
    caveats = ["one seed a cell", "no learning rate was tuned for a size", "the budgets span a factor of eight, the block counts two decades",
               "E is weakly determined on this range and is not the entropy of the text",
               f"the shortest budget is {min(steps)} steps" if steps else "no cell has finished", "the additive form is itself in question",
               "the main run lies outside the grid: 15 times its longest budget, and a context of 1,024 where the grid has 512"]
    out = {"generated_by": "gptpp.export.scaling_surface", "synthetic": bool(synthetic), "expected": names, "missing": missing, "cells": cells,
           "counts": [{"key": k, "label": label} for k, label in COUNTS], "fits": fits, "extras": extra, "delta": DELTA, "min_cells": MIN_CELLS, "min_sizes": MIN_SIZES, "min_budgets": MIN_BUDGETS, "enough": enough(cells), "caveats": caveats,
           "main": {"blocks": main_counts["blocks"], "output": main_counts["output"], "all": main_counts["all"], "tokens": main_tokens,
                    "measured_1024": main_measured["measured_1024"], "measured_512": main_measured["measured_512"], "final": main_measured["final"]},
           "width_rule": not off_the_rule(cells),  # every cell's two tables trained at the recipe's rates for its width (false only under --first)
           "first_grid": before}
    if synthetic:
        out["truth"] = synthetic["truth"]
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", default=str(repo_dir() / "results" / "runs"))
    parser.add_argument("--out", default=str(DATA))
    parser.add_argument("--bootstrap", type=int, default=500, help="resamples of the cells for the intervals (0: none)")
    parser.add_argument("--starts", default="full", choices=["full", "fast"], help="full: the 4,500 starts of the 2022 study; fast: 432")
    parser.add_argument("--synthetic", action="store_true", help="write cells drawn from a known law instead of the runs (for tests and development)")
    parser.add_argument("--first", action="store_true", help="fit the first grid (grid-* at every width, the tables' rates of width 512) in place of the grid: for the record")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise", type=float, default=0.004, help="the relative noise of a synthetic cell")
    args = parser.parse_args(argv)
    synthetic = None
    if (args.synthetic or args.first) and Path(args.out).resolve() == DATA.resolve():
        raise SystemExit(f"--{'synthetic' if args.synthetic else 'first'} needs --out: such a file must not stand in the site's data folder")
    if args.synthetic:
        synthetic = {"seed": args.seed, "noise": args.noise, "truth": {"count": "blocks", "E": 2.2, "A": 110.0, "alpha": 0.32, "B": 4.5e3, "beta": 0.45}}
    data = export(Path(args.runs), args.bootstrap, args.starts, synthetic, first=args.first and not args.synthetic)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_digits(data), separators=(",", ":"), ensure_ascii=False)
    write_atomic(out / f"{NAME}.json", text.encode("utf-8"))
    size = len(text.encode("utf-8"))
    print(f"{len(data['cells'])} of {len(data['expected'])} cells" + (" (synthetic)" if data["synthetic"] else " (THE FIRST GRID)" if args.first else "") + f"; missing {len(data['missing'])}")
    for key, label in COUNTS:
        f = data["fits"][key]
        if f is None:
            print(f"{label}: no fit (fewer than {MIN_CELLS} cells, {MIN_SIZES} sizes or {MIN_BUDGETS} budgets)")
            continue
        ci = f["ci"]
        print(f"{label}: alpha {f['alpha']:.4f}" + (f" [{ci['alpha']['lo95']:.3f}, {ci['alpha']['hi95']:.3f}]" if ci else "") + f"  beta {f['beta']:.4f}"
              + (f" [{ci['beta']['lo95']:.3f}, {ci['beta']['hi95']:.3f}]" if ci else "") + f"  E {f['E']:.4f}  a {f['a']:.4f}  objective {f['objective']:.3e}  grad {f['grad_norm']:.1e}  "
              f"agree {f['agree']}/{f['polished']} of {f['starts']} starts  second {f['second_optimizer']:.3e}  worst residual {100 * f['max_rel_residual']:.2f}% ({f['max_rel_residual_run']})  "
              f"main run predicted {f['main']['predicted']:.4f}" + ("  AT A BOUND" if f["at_bound"] else ""))
    before = data["first_grid"]
    if before:
        moved, rates = [r["moved"] for r in before["replaced"]], before["rates"]
        print(f"the first grid, {before['runs']} runs" + (f" with the tables at {rates['lr_embedding']} and {rates['lr_unembedding']} at every width" if rates else "")
              + f": {len(moved)} of its losses are replaced, by runs that ended {min(moved):+.4f} to {max(moved):+.4f} nats from them")
        for key, label in COUNTS:
            f = before["fits"][key]
            if f:
                print(f"  {label}, on the first grid: alpha {f['alpha']:.4f}  beta {f['beta']:.4f}  E {f['E']:.4f}  a {f['a']:.4f}  worst residual {100 * f['max_rel_residual']:.2f}%  main run predicted {f['predicted']:.4f}")
    print(f"{NAME}.json {size} bytes")


def _digits(o):
    """Floats to nine significant digits: `common.write_json` would cut the exponents to five."""
    if isinstance(o, float):
        return float(f"{o:.9g}") if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _digits(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_digits(v) for v in o]
    if isinstance(o, np.generic):
        return _digits(o.item())
    return o


if __name__ == "__main__":
    main()
