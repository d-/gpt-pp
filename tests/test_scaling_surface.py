"""The fit of the scaling law (gptpp/export/scaling_surface.py) on cells drawn from a known law: it recovers the
constants without noise, stays near them with noise, its frontier is the minimum of the law at equal compute, and it
refuses to fit a handful of cells. And the runs it reads: the grid made under the recipe's width rule, a cell that
was not trained at the rule's rates refused, the first grid kept beside it."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from gptpp.export import scaling_surface as ss

TRUTH = {"count": "blocks", "E": 2.2, "A": 110.0, "alpha": 0.32, "B": 4.5e3, "beta": 0.45}


def cells(noise: float, seed: int = 0):
    made = ss.synthetic_cells(seed, noise, TRUTH)
    return (np.array([c[k] for c in made], dtype=np.float64) for k in ("blocks", "tokens", "val_loss"))


def write_run(root: Path, run: str, loss: float, rates: tuple, shape: tuple = ()) -> None:
    """The two small files of a finished run, as the trainer leaves them, with what the exporter reads. A run of the
    grid has its width, layers and budget (in millions of tokens) in its name; any other is given them as `shape`."""
    width, layers, budget = shape or (int(v) for v in ss.CELL.match(run).groups())
    counts, steps = ss.formula_counts(width, layers), round(budget * 1e6 / 65536)
    folder = root / run
    folder.mkdir(parents=True)
    (folder / "summary.json").write_text(json.dumps({
        "run": run, "status": "completed", "tokens": steps * 65536, "steps": steps, "val_loss": loss, "val_bpb": loss / math.log(2) / 4.6283,
        "best_val_loss": loss, "seconds": 60.0, "tokens_per_second": 500000, "n_gpu": 1,
        "params": {"attention": counts["blocks"] // 3, "mlp": counts["blocks"] - counts["blocks"] // 3, "unembedding": counts["output"] - counts["blocks"],
                   "total": counts["all"]}}))
    (folder / "config.json").write_text(json.dumps({
        "model_config": {"n_layer": layers, "d_model": width}, "train_config": {"warmup_tokens": 2e6},
        "derived": {"seq_len": 512, "tokens_per_step": 65536,
                    "groups": {"embedding": {"lr": rates[0]}, "unembedding": {"lr": rates[1]}, "matrices": {"lr": 0.02}}}}))


def test_the_grid_is_23_runs():
    names = ss.expected()
    assert len(names) == 23 and "gridr-w768-l12-200M" not in names and "gridr-w768-l12-100M" in names
    # the preset's own width was never set on the command line: its four runs are the first grid's
    assert [n for n in names if not n.startswith("gridr-")] == [f"grid-w512-l8-{t}M" for t in (25, 50, 100, 200)]
    first = ss.expected(first=True)
    assert first == [n.replace("gridr-", "grid-") for n in names] and "grid-w768-l12-100M" in first
    assert all(ss.CELL.match(n) for n in names + first) and not ss.CELL.match("ladder-0-gpt2-2019-s0")
    assert ss.formula_counts(128, 4) == {"blocks": 786432, "output": 4980736, "all": 9175040}
    assert ss.formula_counts(768, 12) == {"blocks": 84934656, "output": 110100480, "all": 135266304}


def test_the_rates_of_the_width_rule():
    assert ss.rule_rates(768) == (0.3, 0.008) and ss.rule_rates(512) == (0.3674, 0.0098) and ss.rule_rates(128) == (0.7348, 0.0196)
    assert all(c["lr_embedding"] == ss.rule_rates(c["width"])[0] for c in ss.synthetic_cells(0, 0.0, TRUTH))


def test_the_grid_is_read_from_the_runs_under_the_rule_and_keeps_the_first_grid(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "CKPT", tmp_path / "none")
    # which runs are read is the subject here, not the quality of the fit (the tests below): a handful of starts
    monkeypatch.setattr(ss, "start_grid", lambda kind: [np.array([a, 8.0, 0.8, alpha, beta]) for a in (3.0, 5.0) for alpha in (0.2, 0.4) for beta in (0.3, 0.6)])
    def cell(run: str) -> tuple:                            # (width, layers, budget): the same for a run and the run that replaced it
        return ss.CELL.match(run).groups()

    law = {cell(c["run"]): c["val_loss"] for c in ss.synthetic_cells(0, 0.0, TRUTH)}
    wide = ss.rule_rates(512)
    for run in ss.expected(first=True):                     # the first grid: the rates of width 512 at every width, 0.01 nats worse off it
        write_run(tmp_path, run, law[cell(run)] + (0.0 if "-w512-" in run else 0.01), wide)
    data = ss.export(tmp_path, 0, "fast")                   # none of the runs made again has finished
    assert [c["run"] for c in data["cells"]] == [f"grid-w512-l8-{t}M" for t in (25, 50, 100, 200)] and len(data["missing"]) == 19
    assert data["enough"] is False and data["first_grid"] is None and data["width_rule"] is True
    for run in ss.expected():
        if run.startswith("gridr-"):
            write_run(tmp_path, run, law[cell(run)], ss.rule_rates(int(cell(run)[0])))
    data = ss.export(tmp_path, 0, "fast")
    assert [c["run"] for c in data["cells"]] == ss.expected() and data["missing"] == [] and data["width_rule"] is True
    assert all((c["lr_embedding"], c["lr_unembedding"]) == ss.rule_rates(c["width"]) for c in data["cells"])
    assert abs(data["fits"]["blocks"]["alpha"] - TRUTH["alpha"]) < 0.02 and data["fits"]["blocks"]["starts"] == 8
    assert all(data["fits"][k]["max_rel_residual_run"] in ss.expected() for k in ("blocks", "output", "all"))
    before = data["first_grid"]
    assert before["runs"] == 23 and before["rates"] == {"lr_embedding": wide[0], "lr_unembedding": wide[1], "width": 512}
    assert [r["run"] for r in before["replaced"]] == [n for n in ss.expected(first=True) if "-w512-" not in n]
    assert all(r["by"] == r["run"].replace("grid-", "gridr-") and r["moved"] == pytest.approx(-0.01) for r in before["replaced"])
    assert set(before["fits"]) == {"blocks", "output", "all"} and before["fits"]["blocks"]["alpha"] != data["fits"]["blocks"]["alpha"]
    assert set(before["fits"]["output"]) == {"alpha", "beta", "E", "A", "B", "a", "max_rel_residual", "predicted", "frontier"}
    # the first grid in place of the grid: the export as it stood, and it says that its cells are off the rule
    old = ss.export(tmp_path, 0, "fast", first=True)
    assert [c["run"] for c in old["cells"]] == ss.expected(first=True) and old["width_rule"] is False and old["first_grid"] is None
    assert old["fits"]["blocks"]["alpha"] == before["fits"]["blocks"]["alpha"] and old["fits"]["all"]["main"]["predicted"] == before["fits"]["all"]["predicted"]
    # a run under the new name that was not trained at the rule's rates is not a cell of the fit
    write_run(tmp_path / "again", "gridr-w256-l8-50M", law[cell("gridr-w256-l8-50M")], wide)
    (tmp_path / "gridr-w256-l8-50M" / "config.json").write_text((tmp_path / "again" / "gridr-w256-l8-50M" / "config.json").read_text())
    with pytest.raises(ValueError, match=r"gridr-w256-l8-50M \(width 256: 0.3674 and 0.0098, the rule gives 0.5196 and 0.01386\)"):
        ss.export(tmp_path, 0, "fast")


def test_the_ladder_among_the_extras_is_its_rungs_only(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "CKPT", tmp_path / "none")
    for run in ("ladder-0-gpt2-2019-s0", "ladder-0-gpt2-2019-s1", "ladder-0-gpt2-2019-w32M"):   # the last: a rung run again with another warmup
        write_run(tmp_path, run, 4.0, ss.rule_rates(512), shape=(512, 8, 100))
    extras = ss.extras(tmp_path)
    assert [(e["key"], e["kind"]) for e in extras] == [("ladder-0-gpt2-2019-s0", "ladder"), ("ladder-0-gpt2-2019-s1", "ladder")]
    assert [e["label"] for e in extras] == ["ladder, rung 0: gpt2-2019", "ladder, rung 0: gpt2-2019, seed 1"]


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
