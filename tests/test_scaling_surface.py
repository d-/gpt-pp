"""The fit of the scaling law (gptpp/export/scaling_surface.py) on cells drawn from a known law: it recovers the
constants without noise, stays near them with noise, its frontier is the minimum of the law at equal compute, and it
refuses to fit a handful of cells."""
import math

import numpy as np

from gptpp.export import scaling_surface as ss

TRUTH = {"count": "blocks", "E": 2.2, "A": 110.0, "alpha": 0.32, "B": 4.5e3, "beta": 0.45}


def cells(noise: float, seed: int = 0):
    made = ss.synthetic_cells(seed, noise, TRUTH)
    return (np.array([c[k] for c in made], dtype=np.float64) for k in ("blocks", "tokens", "val_loss"))


def test_the_grid_is_23_runs():
    names = ss.expected()
    assert len(names) == 23 and "grid-w768-l12-200M" not in names and "grid-w768-l12-100M" in names
    assert ss.formula_counts(128, 4) == {"blocks": 786432, "output": 4980736, "all": 9175040}
    assert ss.formula_counts(768, 12) == {"blocks": 84934656, "output": 110100480, "all": 135266304}


def test_the_fit_recovers_a_known_law_without_noise():
    N, D, L = cells(0.0)
    f = ss.fit(N, D, L, ss.start_grid("fast"))
    c = ss.describe(f["theta"])
    assert abs(c["alpha"] - TRUTH["alpha"]) < 2e-3 and abs(c["beta"] - TRUTH["beta"]) < 2e-3 and abs(c["E"] - TRUTH["E"]) < 2e-2
    assert f["objective"] < 1e-9 and not f["at_bound"]
    assert f["second_optimizer"] >= f["objective"] - 1e-12  # the second optimizer finds nothing lower


def test_the_fit_stays_near_the_law_with_noise_and_its_gradient_vanishes():
    N, D, L = cells(0.004, seed=1)
    f = ss.fit(N, D, L, ss.start_grid("fast"))
    c = ss.describe(f["theta"])
    assert abs(c["alpha"] - TRUTH["alpha"]) < 0.08 and abs(c["beta"] - TRUTH["beta"]) < 0.08
    assert f["grad_norm"] < 1e-6 and f["starts"] == 432 and f["polished"] == 48 and f["agree"] > 0.5 * f["polished"]
    one = ss.fit(N, D, L, ss.start_grid("fast"), workers=1)            # the same answer in one process
    assert np.allclose(one["theta"], f["theta"], rtol=0, atol=1e-9) and one["objective"] == f["objective"]
    value, grad = ss.objective(f["theta"], np.log(N), np.log(D), np.log(L))
    step = 1e-6
    for k in range(5):  # the analytic gradient is the numerical one
        up, down = f["theta"].copy(), f["theta"].copy()
        up[k] += step
        down[k] -= step
        numeric = (ss.objective(up, np.log(N), np.log(D), np.log(L))[0] - ss.objective(down, np.log(N), np.log(D), np.log(L))[0]) / (2 * step)
        assert abs(numeric - grad[k]) < 1e-6


def test_the_frontier_is_the_minimum_at_equal_compute():
    C = 2e18
    o = ss.frontier(TRUTH, C)
    assert abs(6 * o["N_opt"] * o["D_opt"] / C - 1) < 1e-12
    for r in (0.5, 0.9, 1.1, 2.0):
        assert ss.law(TRUTH, r * o["N_opt"], o["D_opt"] / r) > ss.law(TRUTH, o["N_opt"], o["D_opt"])
    assert abs(TRUTH["alpha"] * TRUTH["A"] * o["N_opt"] ** -TRUTH["alpha"] - TRUTH["beta"] * TRUTH["B"] * o["D_opt"] ** -TRUTH["beta"]) < 1e-9
    assert ss.frontier({**TRUTH, "alpha": 0.0}, C) is None


def test_no_fit_on_a_handful_of_cells(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "CKPT", tmp_path)  # no runs anywhere: nothing has finished
    data = ss.export(tmp_path, 0, "fast")
    assert data["cells"] == [] and len(data["missing"]) == 23 and all(f is None for f in data["fits"].values())
    assert data["synthetic"] is False and data["main"]["tokens"] > 0 and data["enough"] is False


def test_no_fit_on_two_sizes_however_many_cells():
    made = ss.synthetic_cells(0, 0.0, TRUTH)
    two = [c for c in made if c["width"] in (128, 192)]
    assert len(two) == 8 and not ss.enough(two)                        # eight cells, but only two levels of N
    four = [c for c in made if c["width"] in (128, 192, 256, 384)]
    assert ss.enough(four) and ss.enough(made)
    one_budget = [c for c in made if round(c["tokens"] / 25e6) in (1, 2)]
    assert len(one_budget) == 12 and not ss.enough(one_budget)         # twelve cells of six sizes, but two budgets


def test_the_bootstrap_brackets_the_truth():
    N, D, L = cells(0.004, seed=2)
    f = ss.fit(N, D, L, ss.start_grid("fast"))
    ci = ss.bootstrap(N, D, L, f["pool"], 60, seed=0)
    assert ci["alpha"]["lo95"] - 0.05 < TRUTH["alpha"] < ci["alpha"]["hi95"] + 0.05
    assert ci["beta"]["lo95"] - 0.05 < TRUTH["beta"] < ci["beta"]["hi95"] + 0.05
    assert ci["alpha"]["lo95"] <= ci["alpha"]["lo80"] <= ci["alpha"]["median"] <= ci["alpha"]["hi80"] <= ci["alpha"]["hi95"]
    assert math.isfinite(ci["E"]["median"])
