"""The exporter of the throughput waterfall (gptpp/export/where_the_time_goes.py, fig-07-04): the table of
docs/DESIGN.md is read as written, the figure's file follows from a raw file by arithmetic that can be done by hand,
and the measuring half runs end to end on a model of two layers on the CPU (no card, no compilation)."""
import json

import pytest

from gptpp.export import where_the_time_goes as wt


def raw_file(tmp_path, rates=(20000.0, 26000.0, 39000.0, 78000.0, 93600.0), shared_at=None):
    configs = [{"key": c["key"], "what": c["label"], "passes": p, "micro_batch": 48 // p, "steps": 12, "seconds": 25.0,
                "tokens_per_second": r, "tokens_per_second_median": r, "step_seconds": [25.0 / 12] * 12, "warmup_seconds": 9.0,
                "peak_memory_gb": 20.0, "tried": [], "other_mib": 9000 if c["key"] == shared_at else 0}
               for c, r, p in zip(wt.CONFIGS, rates, (6, 6, 3, 1, 1))]
    raw = {"machine": {"device": "test", "shared": shared_at is not None, "other_mib": 0}, "step": {"sequences": 48, "seq_len": 1024, "tokens": 49152},
           "configs": configs, "split": [{"sequences": 32, "seq_len": 1024, "forward_backward_ms": 290.0, "head_ms": 56.0, "optimizer_ms": 36.0,
                                          "optimizer_adamw_ms": 6.5, "load_ms": 0.2, "other_mib": 0}]}
    path = tmp_path / "throughput-waterfall.json"
    path.write_text(json.dumps(raw))
    return path


def test_the_design_table_is_read_as_written():
    s = wt.design_split()
    assert s["sequences"] == 32 and s["seq_len"] == 1024 and s["source"] == "docs/DESIGN.md, Section 5.2"
    assert (s["forward_backward_ms"], s["head_ms"], s["optimizer_ms"], s["optimizer_adamw_ms"], s["load_ms"]) == (293.0, 57.0, 35.0, 6.0, 0.2)


def test_the_export_is_arithmetic_on_the_raw_file(tmp_path, monkeypatch):
    runs = {"core-tput-base-b48": 90000.0, "core-ddp-base": 171000.0, "gptpp-base": 261000.0}
    monkeypatch.setattr(wt, "run_rate", lambda run: {"run": run, "tokens_per_second": runs[run], "final": run != "gptpp-base", "seconds": 1.0,
                                                    "tokens": 1, "peak_memory_gb": None})
    d = wt.export(raw_file(tmp_path))
    by = {b["key"]: b for b in d["bars"]}
    assert [b["key"] for b in d["bars"]] == ["float32", "tf32", "bf16", "fused", "compiled", "cards-1", "cards-2", "cards-3"]
    assert d["measured"] is True and d["complete"] is False            # the main run's bar is not final
    assert by["float32"]["factor"] is None and by["tf32"]["factor"] == pytest.approx(1.3) and by["fused"]["factor"] == pytest.approx(2.0)
    assert by["cards-2"]["of_linear"] == pytest.approx(171000 / 180000) and by["cards-3"]["of_linear"] == pytest.approx(261000 / 270000)
    assert by["cards-3"]["main"] is True and by["cards-3"]["final"] is False and by["cards-1"]["of_linear"] is None
    n = d["numbers"]
    assert n["factor_single"] == pytest.approx(4.68) and n["largest_step"] == "fused" and n["largest_step_factor"] == pytest.approx(2.0)
    assert n["compiled_over_run"] == pytest.approx(1.04) and n["three_cards"] == 261000.0
    assert d["split"]["forward_backward_ms"] == 293.0 and d["split_measured"]["forward_backward_ms"] == 290.0
    assert [b["passes"] for b in d["bars"][:5]] == [6, 6, 3, 1, 1]


def test_without_a_raw_file_only_the_runs_are_drawn(tmp_path, monkeypatch):
    monkeypatch.setattr(wt, "run_rate", lambda run: {"run": run, "tokens_per_second": 9e4, "final": True, "seconds": 1.0, "tokens": 1, "peak_memory_gb": None})
    d = wt.export(tmp_path / "none.json")
    assert d["measured"] is False and d["complete"] is False and [b["key"] for b in d["bars"]] == ["cards-1", "cards-2", "cards-3"]
    assert "factor_single" not in d["numbers"] and d["split"]["optimizer_ms"] == 35.0 and d["split_measured"] is None


def test_the_measuring_half_runs_on_a_small_model_on_the_cpu(tmp_path):
    raw = tmp_path / "raw.json"
    tiny = {"n_layer": 2, "n_head": 2, "d_model": 32, "block_size": 32}
    try:
        wt.measure(0.05, None, raw, device="cpu", sequences=4, seq_len=32, shape=tiny, compile=False, split_at=(4, 2), split_seconds=0.05, warm_seconds=0.0)
    except FileNotFoundError as e:                                     # the training data are not on this machine
        pytest.skip(str(e))
    d = json.loads(raw.read_text())
    assert d["step"] == {"sequences": 4, "seq_len": 32, "tokens": 128} and d["machine"]["shared"] is False
    assert [c["key"] for c in d["configs"]] == [c["key"] for c in wt.CONFIGS]
    for c in d["configs"]:
        assert c["passes"] * c["micro_batch"] == 4 and c["steps"] >= 5 and len(c["step_seconds"]) == c["steps"] and c["warmup_steps"] == 3
        assert c["tokens_per_second"] == pytest.approx(c["steps"] * 128 / c["seconds"]) and c["tokens_per_second_median"] > 0 and c["other_mib"] == 0
    assert [s["sequences"] for s in d["split"]] == [4, 2]
    for s in d["split"]:
        # (on this toy the unembedding of 32,768 tokens is nearly the whole pass, so its share is not asserted)
        assert min(s["forward_backward_ms"], s["head_ms"], s["optimizer_ms"], s["optimizer_adamw_ms"], s["load_ms"]) > 0
    wt.measure(0.05, ["tf32", "split2"], raw, device="cpu", sequences=4, seq_len=32, shape=tiny, compile=False, split_at=(4, 2), split_seconds=0.05, warm_seconds=0.0)
    again = json.loads(raw.read_text())                                # one configuration measured again: the others are kept as they were
    assert again["configs"][0] == d["configs"][0] and again["configs"][1]["step_seconds"] != d["configs"][1]["step_seconds"]
    assert again["split"][0] == d["split"][0] and [s["sequences"] for s in again["split"]] == [4, 2]
