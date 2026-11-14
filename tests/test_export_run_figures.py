"""The exporters of the two run figures of part 7 (`gptpp.export.training_run_replayed`, fig-07-01, and
`gptpp.export.recipe_race`, fig-07-03): their arithmetic by hand, and each of them end to end on runs made up in a
temporary folder, with no model, no card and no tokenizer.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

from gptpp.export import recipe_race, training_run_replayed

REPO = Path(__file__).resolve().parents[1]
BPT = 4.628316806198068


def test_crossing_between_two_points_linear_in_log_tokens():
    at = training_run_replayed.crossing([0, 100, 1000, 10000], [4.0, 3.0, 2.0, 1.0], 2.5)
    assert at["tokens"] == pytest.approx(10 ** 2.5) and at["between"] == [100.0, 1000.0] and at["sure"] is False
    assert training_run_replayed.crossing([0, 100, 1000], [4.0, 3.0, 2.0], 3.5)["tokens"] == 50.0  # from 0: linear in tokens
    near = training_run_replayed.crossing([100, 200], [4.0, 2.0], 3.0)
    assert near["sure"] is True and near["tokens"] == pytest.approx(math.sqrt(100 * 200))
    assert training_run_replayed.crossing([0, 100], [4.0, 3.0], 1.0) is None
    assert training_run_replayed.crossing([7, 100], [4.0, 3.0], 5.0)["tokens"] == 7.0
    race = recipe_race.crossing([100, 1000, 10000], [3.0, 2.0, 1.0], 1.5)
    assert race["tokens"] == pytest.approx(10 ** 3.5) and race["between"] == [1000.0, 10000.0]


def test_interpolation_and_smoothing():
    assert training_run_replayed.interp(5, [0, 10], [0.0, 100.0]) == 50.0
    assert training_run_replayed.interp(-1, [0, 10], [3.0, 100.0]) == 3.0
    assert training_run_replayed.interp(11, [0, 10], [3.0, 100.0]) == 100.0
    assert training_run_replayed.sig(1.23456789) == 1.23457 and training_run_replayed.sig(None) is None
    # a record is averaged with those whose tokens lie within 15% of its own
    assert recipe_race.smooth([100, 110, 200, 400], [4.0, 2.0, 1.0, 7.0], 0.15) == [3.0, 3.0, 1.0, 7.0]
    assert recipe_race.interp_log(math.sqrt(1e5), [100, 1000], [4.0, 2.0]) == pytest.approx(3.0)


def _config(preset: str, run: str, steps: int, tps: int, lr: float, warmup: float, checkpoint_steps=()) -> dict:
    return {
        "run": run, "preset": preset,
        "model_config": {"vocab_size": 32768, "n_layer": 12, "d_model": 768, "n_head": 12, "block_size": 1024},
        "train_config": {"data": "fineweb-edu/bpe32k", "tokens": float(steps * tps), "device_batch": 48, "optimizer": "x",
                         "lr": lr, "lr_embedding": 0.3, "lr_unembedding": 0.008, "schedule": "trapezoid",
                         "warmup_tokens": warmup, "cooldown_frac": 0.65, "final_lr_frac": 0.05, "val_tokens": 1048576.0,
                         "seed": 0},
        "data_meta": {"dataset": "fineweb-edu", "tokenizer": "bpe32k", "bos_id": 32763, "bytes_per_token_val": BPT,
                      "bytes_per_token_train": 4.640156063025337},
        "params": {"embedding": 1, "position": 0, "unembedding": 1, "attention": 10, "mlp": 20, "norm": 0, "total": 32},
        "derived": {"seq_len": 1024, "n_gpu": 1, "tokens_per_step": tps, "steps": steps, "tokens": steps * tps,
                    "eval_every_steps": 20, "val_windows": 1024, "checkpoint_steps": list(checkpoint_steps)},
    }


def _run(root: Path, name: str, preset: str, lr: float, warmup: float, floor: float, amp: float, tok_s: int, steps: int = 400,
         tps: int = 49152, done: bool = True, checkpoint_steps=()) -> Path:
    """A run made up from a law: loss = floor + amp * (tokens / 1M) ** -0.3, logged every ten steps, read every 20."""
    d = root / name
    d.mkdir(parents=True)
    law = lambda s: floor + amp * (max(s, 1) * tps / 1e6) ** -0.3
    (d / "config.json").write_text(json.dumps(_config(preset, name, steps, tps, lr, warmup, checkpoint_steps)))
    lines = [{"kind": "val", "step": 0, "tokens": 0, "time": 0.0, "val_loss": math.log(32768), "val_bpb": math.log2(32768) / BPT}]
    for s in range(10, steps + 1, 10):
        lines.append({"kind": "train", "step": s, "tokens": s * tps, "time": s * tps / tok_s, "loss": law(s), "lr": lr,
                      "lr_mult": 1.0 if s * tps <= 0.35 * steps * tps else 0.5, "tok_s": tok_s})
        if s % 20 == 0:
            lines.append({"kind": "val", "step": s, "tokens": s * tps, "time": s * tps / tok_s, "val_loss": law(s) - 0.01,
                          "val_bpb": (law(s) - 0.01) / math.log(2) / BPT})
            lines.append({"kind": "sample", "step": s, "tokens": s * tps, "text": "x"})
    (d / "log.jsonl").write_text("\n".join(json.dumps(r) for r in lines) + "\n")
    if done:
        end = law(steps) - 0.01
        (d / "summary.json").write_text(json.dumps({
            "run": name, "status": "completed", "preset": preset, "val_loss": end, "val_bpb": end / math.log(2) / BPT,
            "best_val_loss": end, "best_val_bpb": end / math.log(2) / BPT, "best_step": steps, "train_loss": law(steps),
            "steps": steps, "tokens": steps * tps, "n_gpu": 1, "seconds": steps * tps / tok_s + 20, "train_seconds": steps * tps / tok_s,
            "tokens_per_second": tok_s, "peak_memory_gb": 1.0, "rank_spread": 0.0, "params": {}}))
    return d


def test_recipe_race_end_to_end(tmp_path, monkeypatch):
    root, out = tmp_path / "ckpt", tmp_path / "out"
    _run(root, "race-2019-lr6e-4-100M", "gpt2-2019-base", 6e-4, 32e6, 3.2, 2.6, 100000)
    _run(root, "race-2019-lr1.2e-3-100M", "gpt2-2019-base", 1.2e-3, 32e6, 3.2, 2.4, 100000)
    _run(root, "race-2019-lr1.2e-3-w8M-100M", "gpt2-2019-base", 1.2e-3, 8e6, 3.2, 2.2, 100000)
    _run(root, "race-2019-lr9e-4-extra", "gpt2-2019-base", 9e-4, 8e6, 3.2, 2.3, 100000)       # joins by its name
    _run(root, "race-2019-lr2.4e-3-100M", "gpt2-2019-base", 2.4e-3, 32e6, 3.2, 2.5, 100000, done=False)   # still running
    _run(root, "race-2026-100M", "gptpp-base", 0.02, 2e6, 3.2, 1.5, 90000)
    monkeypatch.setattr(sys, "argv", ["recipe_race", "--ckpt-root", str(root), "--out", str(out)])
    monkeypatch.setattr(recipe_race, "REPO", tmp_path)   # the lab's own copies of the real race under results/runs stay out of it
    recipe_race.main()
    D = json.loads((out / "recipe-race.json").read_text())
    assert D["stand_in"] is False and len(D["runs"]) == 5
    assert sorted(D["missing"]) == ["race-2019-lr2.4e-3-100M", "race-2019-lr2.4e-3-w8M-100M", "race-2019-lr4.8e-3-100M"]
    assert D["best_2019"] == "race-2019-lr1.2e-3-w8M-100M" and D["run_2026"] == "race-2026-100M"
    assert [r["recipe"] for r in D["runs"]] == ["2019"] * 4 + ["2026"]
    best = D["finals"][D["best_2019"]]
    assert best["warmup_tokens"] == 8e6 and best["lr"] == 1.2e-3 and best["n_gpu"] == 1
    assert D["numbers"]["best_2019_is_largest_rate"] is True and D["numbers"]["best_2019_warmup_tokens"] == 8e6
    assert len(D["levels"]) == 3 and D["levels"][-1]["tokens_2019"] == D["budget"]["tokens"]
    assert not any(level["bound"] for level in D["levels"])
    for level in D["levels"]:
        # the 2026 law equals the level where 3.2 + 1.5 x^-0.3 = level in nats of training text
        nats = level["bpb"] * math.log(2) * D["data"]["bytes_per_token_train"]
        exact = 1e6 * ((nats - 3.2) / 1.5) ** (-1 / 0.3)
        assert level["tokens_2026"] == pytest.approx(exact, rel=0.05)
        assert level["ratio"] == pytest.approx(level["tokens_2019"] / level["tokens_2026"], rel=1e-4)
        assert level["time_ratio"] == pytest.approx(level["ratio"] * 90000 / 100000, rel=1e-4)
    assert D["numbers"]["ratio_at_end"] == D["levels"][-1]["ratio"]
    # other names than the race: a stand-in, and it says so
    monkeypatch.setattr(sys, "argv", ["recipe_race", "--ckpt-root", str(root), "--out", str(out), "--runs2019",
                                      "race-2019-lr6e-4-100M", "--run2026", "race-2026-100M"])
    recipe_race.main()
    assert json.loads((out / "recipe-race.json").read_text())["stand_in"] is True


def test_recipe_race_with_no_run_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["recipe_race", "--ckpt-root", str(tmp_path / "none"), "--out", str(tmp_path)])
    monkeypatch.setattr(recipe_race, "REPO", tmp_path)   # no copy under results/runs either
    recipe_race.main()
    D = json.loads((tmp_path / "recipe-race.json").read_text())
    assert D["runs"] == [] and len(D["missing"]) == 7 and D["best_2019"] is None and D["levels"] == []


@pytest.mark.skipif(not (REPO / "results" / "data" / "fineweb-edu" / "bpe32k" / "stats.json").exists(), reason="needs the corpus statistics")
@pytest.mark.parametrize("done", [False, True])
def test_training_run_replayed_end_to_end(tmp_path, monkeypatch, done):
    root, out, tps = tmp_path / "ckpt", tmp_path / "out", 147456
    steps = [0, 1, 20, 100]
    d = _run(root, "main", "gptpp-base", 0.02, 2e6, 3.0, 2.0, 264000, steps=100, tps=tps, done=done, checkpoint_steps=steps + [200])
    for s in steps:
        (d / f"ckpt_{s}.pt").write_bytes(b"")                      # only their presence is read: the side file holds the rest
    side = {"samples": {str(s): {"texts": ["a", "b\nc", "d"], "sampling": training_run_replayed.SAMPLING,
                                 "openings": [o["text"] for o in training_run_replayed.OPENINGS]} for s in steps},
            "monitor": {"1": {"loss": 10.0, "bpb": 10.0 / math.log(2) / BPT}}, "docs": {"100": {
                "bits_per_byte": 0.9, "se": 0.01, "loss": 2.9, "n_docs": 1000, "n_tokens": 5, "n_bytes": 9, "sha256": "ab"}},
            "opening_tokens": [7, 5, 8]}
    side_file = tmp_path / "side.json"
    side_file.write_text(json.dumps(side))
    monkeypatch.setattr(sys, "argv", ["x", "--run", "main", "--ckpt-root", str(root), "--out", str(out), "--side", str(side_file)])
    training_run_replayed.main()
    D = json.loads((out / "training-run-replayed.json").read_text())
    assert D["complete"] is done and (D["summary"] is not None) is done
    assert [c["step"] for c in D["checkpoints"]] == steps and D["checkpoints_planned"] == 5
    assert D["checkpoints"][1]["monitor"]["loss"] == 10.0 and D["checkpoints"][3]["docs"]["bits_per_byte"] == 0.9
    assert D["held_out"]["step"] == [0, 1, 20, 40, 60, 80, 100] and D["held_out"]["source"][1] == "checkpoint"
    assert len(D["train"]["step"]) == 10 and D["val"]["step"] == [0, 20, 40, 60, 80, 100]
    levels = {lv["key"]: lv for lv in D["levels"]}
    assert levels["uniform"]["bits_per_byte"] == pytest.approx(15 / BPT, rel=1e-5)
    assert levels["4gram"]["bits_per_byte"] < levels["unigram"]["bits_per_byte"] < levels["uniform"]["bits_per_byte"]
    post = D["numbers"]["post"]
    assert post["first_reading"]["step"] == 20 and post["decay_start_tokens"] == round(0.35 * 100 * tps)
    assert (post["last_500M"] is not None) is done and (D["numbers"]["tokens"] is not None) is done
    if "gpt2" in levels:
        assert levels["gpt2"]["protocol"] == "documents" and levels["gpt2"]["monitor"] is None
