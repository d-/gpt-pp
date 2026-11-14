"""The data loaders, the trainer end to end on byte-level Shakespeare, two-card training, and the job queue.

The trainer tests use the first visible GPU when there is one; the distributed test needs two and is skipped
otherwise.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import REPO

from gptpp import train
from gptpp.config import preset_names
from gptpp.data import ByteTokenizer, TokenData, TrainLoader, ValLoader
from gptpp.model import GPT

# ----------------------------------------------------------------------------------------------------------------
# The dataset on disk and the loaders


def test_byte_dataset_obeys_the_contract(byte_data: Path) -> None:
    meta = json.loads((byte_data / "meta.json").read_text())
    for key in ("dataset", "tokenizer", "vocab_size", "bos_id", "n_train_tokens", "n_val_tokens", "train_shards",
                "val_bytes", "val_docs", "bytes_per_token_val", "source", "license"):
        assert key in meta, key
    assert (meta["dataset"], meta["tokenizer"], meta["vocab_size"], meta["bos_id"]) == ("shakespeare", "bytes", 257, 256)
    shards = [np.fromfile(byte_data / name, dtype="<u2") for name in meta["train_shards"]]
    val = np.fromfile(byte_data / "val.bin", dtype="<u2")
    assert len(shards) >= 3 and sum(map(len, shards)) == meta["n_train_tokens"] and len(val) == meta["n_val_tokens"]
    assert (byte_data / "val.bin").stat().st_size == 2 * len(val)  # two bytes a token
    assert shards[0][0] == 256 and val[0] == 256 and max(s.max() for s in shards) <= 256
    # the held-out documents as text: one line each, the same documents as val.bin, each begun by BOS
    docs = [json.loads(line)["text"] for line in (byte_data.parent / "val_docs.jsonl").read_text().splitlines()]
    assert len(docs) == meta["val_docs"] == int((val == 256).sum())
    assert sum(len(d.encode()) for d in docs) == meta["val_bytes"]
    assert meta["bytes_per_token_val"] == pytest.approx(meta["val_bytes"] / len(val))
    tokenizer = ByteTokenizer()
    assert tokenizer.decode(val.tolist()) == "".join(docs)
    assert tokenizer.encode("né", bos=True) == [256, 110, 195, 169] and tokenizer.decode([256, 110, 195, 169]) == "né"
    # training and held-out documents are disjoint
    train_text = tokenizer.decode(np.concatenate(shards).tolist())
    assert not any(d in train_text for d in docs if len(d) > 200)


def test_train_loader_reads_every_window_once(byte_data: Path) -> None:
    data, T, B = TokenData(byte_data), 64, 16
    loader = TrainLoader(data, T, B, seed=0)
    seen: list[Counter] = [Counter() for _ in data.train_files]
    batches = 0
    while True:
        x, y = loader.next_batch()
        if loader.epoch > 0:
            break
        assert x.shape == y.shape == (B, T) and x.dtype == torch.int64
        assert torch.equal(x[:, 1:], y[:, :-1])  # the targets are the inputs shifted by one
        seen[loader.shard_index].update(bytes(row) for row in x.numpy().astype(np.uint16))
        batches += 1
    expected_batches = 0
    for i, counter in enumerate(seen):
        tokens = data.shard(i)
        n_windows = (len(tokens) - 1) // T
        windows = Counter(bytes(tokens[w * T:(w + 1) * T]) for w in range(n_windows))
        assert sum(counter.values()) == (n_windows // B) * B  # all but the last partial batch of the shard
        assert not counter - windows  # without replacement: no window more often than it exists
        expected_batches += n_windows // B
    assert batches == expected_batches


def test_train_loader_is_deterministic_by_seed(byte_data: Path) -> None:
    data = TokenData(byte_data)
    a, b, c = (TrainLoader(data, 64, 8, seed=seed) for seed in (0, 0, 1))
    first = [a.next_batch() for _ in range(5)]
    assert all(torch.equal(x, x2) and torch.equal(y, y2) for (x, y), (x2, y2) in zip(first, [b.next_batch() for _ in range(5)]))
    assert not torch.equal(first[0][0], c.next_batch()[0])
    # the windows of a batch come from all over the shard, not from one place
    starts = a.order[:8] * 64
    assert starts.max() - starts.min() > 50 * 64


def test_ranks_read_disjoint_slices_of_the_same_step(byte_data: Path) -> None:
    """Two ranks with batch 8 see, step by step, exactly the windows one rank with batch 16 sees."""
    data = TokenData(byte_data)
    alone = TrainLoader(data, 64, 16, seed=3)
    ranks = [TrainLoader(data, 64, 8, rank=r, world=2, seed=3) for r in range(2)]
    for _ in range(400):  # far enough to cross into the second shard
        x, y = alone.next_batch()
        parts = [loader.next_batch() for loader in ranks]
        assert torch.equal(torch.cat([p[0] for p in parts]), x) and torch.equal(torch.cat([p[1] for p in parts]), y)
    assert alone.shard_index >= 1 and ranks[0].shard_index == ranks[1].shard_index == alone.shard_index


def test_train_loader_needs_a_batch_per_shard(byte_data: Path) -> None:
    with pytest.raises(ValueError, match="smaller batch"):
        TrainLoader(TokenData(byte_data), 1024, 4096)


def test_val_loader_is_fixed(byte_data: Path) -> None:
    data, T = TokenData(byte_data), 64
    tokens = torch.from_numpy(np.fromfile(data.val_file, dtype="<u2").astype(np.int64))
    n = 20000 // T
    reference = tokens[:n * T + 1]

    def windows(loaders: list[ValLoader]) -> tuple[torch.Tensor, torch.Tensor, int]:
        """All (input, target) rows that count, sorted, and the number of targets."""
        rows, total = [], 0
        for loader in loaders:
            for x, y, n_targets in loader:
                valid = (y >= 0).all(dim=1)
                assert int(valid.sum()) * T == n_targets and (y[~valid] == -1).all()
                rows += [(tuple(a.tolist()), tuple(b.tolist())) for a, b in zip(x[valid], y[valid])]
                total += n_targets
        return sorted(rows), total

    expected = sorted((tuple(reference[i * T:(i + 1) * T].tolist()), tuple(reference[i * T + 1:(i + 1) * T + 1].tolist()))
                      for i in range(n))
    for batch, world in ((8, 1), (32, 1), (7, 1), (8, 2), (5, 3)):
        loaders = [ValLoader(data, T, batch, rank=r, world=world, val_tokens=20000) for r in range(world)]
        rows, total = windows(loaders)
        assert rows == expected and total == n * T, (batch, world)
        assert all(x.shape == (batch, T) for loader in loaders for x, _, _ in loader)  # one shape: no retracing
        assert loaders[0].n_windows == n
    whole = ValLoader(data, T, 64, val_tokens=1e12)  # more than there is: everything
    assert whole.n_windows == (len(tokens) - 1) // T


# ----------------------------------------------------------------------------------------------------------------
# The trainer


def run_trainer(work: Path, name: str, preset: str, *flags: str, tokens: str = "6e4", sets: tuple[str, ...] = ()) -> tuple[dict, Path]:
    """Train in this process; returns the summary and the run directory."""
    argv = ["--preset", preset, "--data", str(work / "data" / "shakespeare" / "bytes"), "--run", name,
            "--tokens", tokens, "--out", str(work / "ckpt"), "--results", str(work / "results"), "--no-compile", *flags]
    summary = train.main(argv + (["--set", *sets] if sets else []))
    return summary, work / "ckpt" / name


def read_log(run_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (run_dir / "log.jsonl").read_text().splitlines()]


SHORT = ("device_batch=8", "warmup_tokens=1e4", "eval_tokens=3e4", "val_tokens=2e4", "log_every=5")


@pytest.mark.parametrize("preset", preset_names())
def test_every_preset_trains_and_its_loss_falls(work: Path, preset: str) -> None:
    """A few dozen steps of each preset, at its real size, on byte-level Shakespeare."""
    size = preset.rsplit("-", 1)[1]
    tokens = {"pico": "8e4", "ladder": "1.6e5", "base": "2.4e5"}[size]
    summary, run_dir = run_trainer(work, f"smoke-{preset}", preset, tokens=tokens, sets=SHORT)
    log = read_log(run_dir)
    vals = [r for r in log if r["kind"] == "val"]
    trains = [r for r in log if r["kind"] == "train"]

    # the loss starts near the entropy of a uniform guess over 257 tokens (exactly there with zero-initialized
    # projections; a little above with the GPT-2 initialization, whose first logits are not zero) and falls
    assert vals[0]["step"] == 0 and abs(vals[0]["val_loss"] - math.log(257)) < (0.01 if preset.startswith("gptpp") else 0.3)
    assert vals[-1]["val_loss"] < vals[0]["val_loss"] - 1.5
    assert trains[-1]["loss"] < trains[0]["loss"] - 1.0
    assert summary["val_loss"] == vals[-1]["val_loss"] and summary["best_val_loss"] == min(v["val_loss"] for v in vals)

    # the records of Contract 4
    config = json.loads((run_dir / "config.json").read_text())
    block = config["model_config"]["block_size"]
    steps = round(float(tokens) / (8 * block))
    assert summary["steps"] == steps == trains[-1]["step"] and summary["tokens"] == steps * 8 * block == trains[-1]["tokens"]
    for record in trains:
        assert {"step", "tokens", "time", "loss", "lr", "tok_s"} <= set(record) and record["tokens"] == record["step"] * 8 * block
    for record in vals:
        assert {"step", "tokens", "time", "val_loss", "val_bpb"} <= set(record)
        meta = config["data_meta"]
        assert record["val_bpb"] == pytest.approx(record["val_loss"] / math.log(2) / meta["bytes_per_token_val"])
    assert [r["time"] for r in log if "time" in r] == sorted(r["time"] for r in log if "time" in r)
    assert vals[-1]["step"] == steps and [v["step"] for v in vals[:2]] == [0, max(1, round(3e4 / (8 * block)))]
    assert config["model_config"]["vocab_size"] == 257  # taken from the data, not from the preset
    assert config["params"] == summary["params"] and config["data_meta"]["tokenizer"] == "bytes"
    assert {"train_config", "git_commit", "hardware", "derived"} <= set(config)
    for key in ("status", "val_loss", "val_bpb", "best_val_loss", "best_val_bpb", "tokens", "seconds",
                "tokens_per_second", "peak_memory_gb", "params", "steps", "n_gpu"):
        assert key in summary, key
    assert summary["status"] == "completed" and summary["tokens_per_second"] > 0
    assert json.loads((run_dir / "summary.json").read_text()) == summary

    # the final model loads and is the model that was evaluated last
    model, cfg = GPT.from_checkpoint(run_dir / "model.pt", device="cuda" if torch.cuda.is_available() else "cpu")
    assert cfg.to_dict() == config["model_config"] and model.num_params() == summary["params"]
    checkpoint = torch.load(run_dir / "model.pt", weights_only=True)
    assert checkpoint["step"] == steps and checkpoint["tokens"] == summary["tokens"]
    device = next(model.parameters()).device
    loader = ValLoader(TokenData(work / "data" / "shakespeare" / "bytes"), block, 8, val_tokens=2e4, device=device)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")
    assert train.evaluate(model, loader, autocast, world=1) == pytest.approx(summary["val_loss"], abs=0.02)

    # a copy of the small files went to the results directory
    copied = work / "results" / f"smoke-{preset}"
    assert sorted(p.name for p in copied.iterdir()) == ["config.json", "log.jsonl", "summary.json"]
    assert json.loads((copied / "summary.json").read_text()) == summary


def test_samples_checkpoints_and_schedule(work: Path) -> None:
    summary, run_dir = run_trainer(work, "smoke-checkpoints", "gptpp-pico", "--save-steps", "3,7", "--save-log", "4",
                                   "--seed", "5", tokens="4e4", sets=SHORT + ("sample_tokens=24", "schedule=cosine"))
    log = read_log(run_dir)
    steps = summary["steps"]
    assert steps == 20
    # checkpoints: the steps asked for, the initialization and four log-spaced steps up to the last
    saved = sorted(int(p.stem.split("_")[1]) for p in run_dir.glob("ckpt_*.pt"))
    assert saved == sorted({0, 3, 7} | {int(round(s)) for s in np.geomspace(1, steps, 4)}) and saved[-1] == steps
    for step in saved:
        checkpoint = torch.load(run_dir / f"ckpt_{step}.pt", weights_only=True)
        assert checkpoint["step"] == step and checkpoint["tokens"] == step * 8 * 256
    first, _ = GPT.from_checkpoint(run_dir / "ckpt_0.pt", device="cpu")
    last, _ = GPT.from_checkpoint(run_dir / f"ckpt_{steps}.pt", device="cpu")
    final, _ = GPT.from_checkpoint(run_dir, device="cpu")
    assert first.blocks[0].attn.o.weight.abs().max() == 0  # the initialization: projections at zero
    assert torch.equal(last.embed.weight, final.embed.weight) and not torch.equal(first.embed.weight, final.embed.weight)
    # a sample at each evaluation, decoded by the byte tokenizer named in the data meta
    samples = [r for r in log if r["kind"] == "sample"]
    assert [s["step"] for s in samples] == [v["step"] for v in log if v["kind"] == "val"]
    assert all(isinstance(s["text"], str) for s in samples)
    # the logged rate follows warmup then cosine, read at the middle of each step
    config = json.loads((run_dir / "config.json").read_text())
    assert config["train_config"]["seed"] == 5 and config["derived"]["checkpoint_steps"] == saved
    for record in (r for r in log if r["kind"] == "train"):
        tokens_mid = (record["step"] - 0.5) * 8 * 256
        if tokens_mid < 1e4:
            expected = tokens_mid / 1e4
        else:
            progress = (tokens_mid - 1e4) / (summary["tokens"] - 1e4)
            expected = 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress))
        assert record["lr_mult"] == pytest.approx(expected) and record["lr"] == pytest.approx(0.02 * expected)


def test_same_seed_same_run(work: Path) -> None:
    """Deterministic given the seed: the same data, the same initialization, nearly the same losses."""
    runs = [run_trainer(work, f"smoke-seed-{name}", "gptpp-pico", "--seed", seed, tokens="3e4", sets=SHORT)[1]
            for name, seed in (("a", "1"), ("b", "1"), ("c", "2"))]
    a, b, c = ([r for r in read_log(run) if r["kind"] in ("train", "val")] for run in runs)
    assert a[0]["val_loss"] == pytest.approx(b[0]["val_loss"], abs=1e-5) and a[0]["val_loss"] != c[0]["val_loss"]
    assert all(abs(x.get("loss", x.get("val_loss")) - y.get("loss", y.get("val_loss"))) < 2e-2 for x, y in zip(a, b))
    assert abs(a[1]["loss"] - c[1]["loss"]) > 1e-3


def test_a_completed_run_is_not_overwritten(work: Path) -> None:
    run_trainer(work, "smoke-once", "gptpp-pico", tokens="1e4", sets=SHORT)
    with pytest.raises(SystemExit, match="--overwrite"):
        run_trainer(work, "smoke-once", "gptpp-pico", tokens="1e4", sets=SHORT)
    summary, _ = run_trainer(work, "smoke-once", "gptpp-pico", "--overwrite", tokens="2e4", sets=SHORT)
    assert summary["steps"] == 10


def test_a_crash_leaves_a_readable_log_and_no_summary(work: Path) -> None:
    """A learning rate of 1e30 destroys the weights at the first step; the run must stop and say so."""
    with pytest.raises(FloatingPointError, match="training loss"):
        run_trainer(work, "smoke-crash", "gpt2-2019-pico", tokens="4e4",
                    sets=SHORT + ("lr=1e30", "warmup_tokens=0", "grad_clip=0", "log_every=1"))
    run_dir = work / "ckpt" / "smoke-crash"
    log = read_log(run_dir)  # every line parses
    assert log[0]["kind"] == "val" and log[-1]["kind"] == "error" and "FloatingPointError" in log[-1]["error"]
    assert any(r["kind"] == "train" for r in log)
    assert not (run_dir / "summary.json").exists() and not (run_dir / "model.pt").exists()
    assert not (work / "results" / "smoke-crash").exists()
    assert "NaN" not in (run_dir / "log.jsonl").read_text()  # strict JSON: a loss that is not finite is null


def test_gradient_accumulation_and_other_optimizers(work: Path) -> None:
    summary, run_dir = run_trainer(work, "smoke-accum", "gptpp-pico", tokens="6e4",
                                   sets=SHORT + ("grad_accum=2", "device_batch=4", "grad_clip=1.0"))
    config = json.loads((run_dir / "config.json").read_text())
    assert config["derived"]["tokens_per_step"] == 4 * 256 * 2 and summary["val_loss"] < math.log(257) - 1.0
    for optimizer, rate in (("sgd", "lr=0.5"), ("lion", "lr=1e-3"), ("adam", "lr=3e-3")):
        summary, _ = run_trainer(work, f"smoke-{optimizer}", "gpt2-2019-pico", tokens="6e4",
                                 sets=SHORT + (f"optimizer={optimizer}", rate))
        assert summary["val_loss"] < math.log(257) - 0.5, optimizer


def test_train_preset_trains_the_2019_model_the_2026_way(work: Path) -> None:
    """--train-preset takes the optimizer, rates and schedule from another preset: the rung of the ladder that
    changes only the optimizer. The 2019 model ties its embeddings, so its one vocabulary matrix must get the
    unembedding's small rate, not the embedding's."""
    summary, run_dir = run_trainer(work, "smoke-mixed", "gpt2-2019-pico", "--train-preset", "gptpp-pico",
                                   tokens="8e4", sets=SHORT)
    config = json.loads((run_dir / "config.json").read_text())
    assert (config["preset"], config["train_preset"]) == ("gpt2-2019-pico", "gptpp-pico")
    assert config["model_config"]["tie_embeddings"] and config["model_config"]["pos"] == "learned"
    assert config["train_config"]["optimizer"] == "muon+adamw"
    groups = config["derived"]["groups"]
    assert set(groups) == {"embedding", "matrices", "gains"}
    assert groups["matrices"] == {"optimizer": "Muon", "lr": 0.02, "weight_decay": 0.0}
    assert groups["embedding"]["optimizer"] == "AdamW"
    assert groups["embedding"]["lr"] == config["train_config"]["lr_unembedding"] < config["train_config"]["lr_embedding"]
    assert summary["val_loss"] < math.log(257) - 1.5


def test_checkpoint_steps() -> None:
    assert train.checkpoint_steps(0, "", 100) == set()
    assert train.checkpoint_steps(0, "10, 50,500", 100) == {10, 50}
    assert train.checkpoint_steps(5, "", 10000) == {0, 1, 10, 100, 1000, 10000}
    assert train.checkpoint_steps(3, "7", 100) == {0, 1, 7, 10, 100}


def test_results_copy_is_thinned(tmp_path: Path) -> None:
    run_dir, results = tmp_path / "run", tmp_path / "results"
    run_dir.mkdir()
    (run_dir / "config.json").write_text("{}")
    (run_dir / "summary.json").write_text("{}")
    records = []
    for step in range(1, 5001):
        records.append({"kind": "train", "step": step, "loss": 1.0})
        if step % 500 == 0:
            records.append({"kind": "val", "step": step, "val_loss": 1.0})
    (run_dir / "log.jsonl").write_text("\n".join(map(json.dumps, records)) + "\n")
    train.copy_results(run_dir, results)
    copied = [json.loads(line) for line in (results / "log.jsonl").read_text().splitlines()]
    kept = [r["step"] for r in copied if r["kind"] == "train"]
    assert len(kept) <= 2000 and kept[-1] == 5000 and kept[:3] == [3, 6, 9]
    assert [r["step"] for r in copied if r["kind"] == "val"] == list(range(500, 5001, 500))
    assert [r["step"] for r in copied] == sorted(r["step"] for r in copied)


# ----------------------------------------------------------------------------------------------------------------
# The command line, compiled; and two cards


def run_cli(work: Path, *args: str, launcher: tuple[str, ...] = (), timeout: int = 280) -> subprocess.CompletedProcess:
    command = [sys.executable, *launcher, "-m", "gptpp.train", "--data", "shakespeare/bytes", *args]
    env = dict(os.environ, GPTPP_WORK=str(work))  # the dataset and the run directory are found under the workspace
    return subprocess.run(command, cwd=REPO, env=env, capture_output=True, text=True, timeout=timeout)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_command_line_with_compilation(work: Path) -> None:
    """The documented command, from the repository root, with torch.compile on."""
    done = run_cli(work, "--preset", "gptpp-pico", "--run", "smoke-cli", "--tokens", "1.2e5", "--eval-tokens", "6e4",
                   "--results", str(work / "results"), "--set", "device_batch=16", "warmup_tokens=1e4", "val_tokens=2e4")
    assert done.returncode == 0, done.stderr[-2000:]
    run_dir = work / "ckpt" / "smoke-cli"
    summary = json.loads((run_dir / "summary.json").read_text())
    config = json.loads((run_dir / "config.json").read_text())
    assert config["train_config"]["compile"] is True and summary["steps"] == 29
    assert summary["val_loss"] < math.log(257) - 1.5 and summary["compile_seconds"] > 1
    lines = [line for line in done.stdout.splitlines() if line.startswith(("step", "eval", "done"))]
    assert len([line for line in lines if line.startswith("step")]) == 3  # one line per log interval
    assert lines[-1].startswith("done") and "tok/s" in lines[-1]
    # sampling from the run with the generation command line
    sampled = subprocess.run([sys.executable, "-m", "gptpp.generate", "--run", "smoke-cli", "--prompt", "Zq7# ",
                              "--max-new-tokens", "40", "--top-k", "20", "--num-samples", "2"],
                             cwd=REPO, env=dict(os.environ, GPTPP_WORK=str(work)), capture_output=True, text=True, timeout=120)
    assert sampled.returncode == 0, sampled.stderr[-2000:]
    assert sampled.stdout.count("Zq7# ") == 2  # two samples, each printed with its prompt


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_two_cards(work: Path) -> None:
    """Two ranks of batch 8 against one rank of batch 16: the same data, the same initialization, and so the
    same losses up to rounding; and the same held-out set."""
    sets = ("warmup_tokens=1e4", "eval_tokens=3e4", "val_tokens=2e4", "log_every=5", "sample_tokens=0")
    common = ("--preset", "gptpp-pico", "--tokens", "8e4", "--no-compile", "--results", "none")
    two = run_cli(work, *common, "--run", "smoke-ddp", "--set", "device_batch=8", *sets,
                  launcher=("-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2"))
    assert two.returncode == 0, two.stderr[-3000:]
    one = run_cli(work, *common, "--run", "smoke-ddp-reference", "--set", "device_batch=16", *sets)
    assert one.returncode == 0, one.stderr[-3000:]
    ddp, reference = work / "ckpt" / "smoke-ddp", work / "ckpt" / "smoke-ddp-reference"
    summary = json.loads((ddp / "summary.json").read_text())
    config = json.loads((ddp / "config.json").read_text())
    assert summary["n_gpu"] == 2 and config["derived"]["tokens_per_step"] == 2 * 8 * 256
    assert summary["steps"] == json.loads((reference / "summary.json").read_text())["steps"] == 20
    a, b = read_log(ddp), read_log(reference)
    assert [r["kind"] for r in a] == [r["kind"] for r in b]
    assert a[0]["val_loss"] == pytest.approx(b[0]["val_loss"], abs=1e-4)  # the same initialization on both ranks
    for x, y in zip(a, b):
        key = "val_loss" if x["kind"] == "val" else "loss"
        assert x["step"] == y["step"] and x[key] == pytest.approx(y[key], abs=5e-3), (x, y)
    assert summary["val_loss"] < math.log(257) - 1.5
    assert summary["rank_spread"] == 0.0  # both ranks ended with the same parameters, bit for bit
    assert two.stdout.count("done:") == 1  # only rank 0 reports


# ----------------------------------------------------------------------------------------------------------------
# The job queue


def run_queue(tmp_path: Path, jobs: list[dict], *flags: str, timeout: int = 60) -> tuple[subprocess.CompletedProcess, list[dict]]:
    jobs_file = tmp_path / "jobs.jsonl"
    jobs_file.write_text("".join(json.dumps(job) + "\n" for job in jobs))
    env = dict(os.environ, GPTPP_WORK=str(tmp_path / "work"))
    done = subprocess.run([sys.executable, str(REPO / "scripts" / "job_queue.py"), str(jobs_file), "--no-check-free",
                           "--poll", "0.1", *flags], env=env, capture_output=True, text=True, timeout=timeout)
    state = tmp_path / "work" / "logs" / "queue-jobs.state.jsonl"
    events = [json.loads(line) for line in state.read_text().splitlines()] if state.exists() else []
    return done, events


def test_queue_runs_one_job_per_card(tmp_path: Path) -> None:
    stamp = "date +%s.%N; echo card $CUDA_VISIBLE_DEVICES; sleep 0.6; date +%s.%N"
    jobs = [{"name": f"job{i}", "cmd": stamp, "summary": None} for i in range(5)]
    done, events = run_queue(tmp_path, jobs, "--gpus", "2,3")
    assert done.returncode == 0, done.stdout + done.stderr
    logs = {job["name"]: (tmp_path / "work" / "logs" / f"{job['name']}.log").read_text().split("\n") for job in jobs}
    spans = sorted((float(lines[0]), float(lines[2]), lines[1]) for lines in logs.values())
    assert {span[2] for span in spans} == {"card 2", "card 3"}
    for card in ("card 2", "card 3"):  # never two jobs on a card at once
        mine = [span for span in spans if span[2] == card]
        assert all(later[0] >= earlier[1] for earlier, later in zip(mine, mine[1:]))
    assert spans[1][0] < spans[0][1]  # and the two cards do work at the same time
    assert [e["event"] for e in events].count("started") == 5 and [e["event"] for e in events].count("finished") == 5
    assert [e["name"] for e in events if e["event"] == "started"] == [f"job{i}" for i in range(5)]  # in file order
    assert all({"time", "name", "event"} <= set(e) for e in events)


def test_queue_skips_done_jobs_and_survives_failures(tmp_path: Path) -> None:
    summary = tmp_path / "work" / "ckpt" / "trained" / "summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text("{}")
    marker = tmp_path / "ran"
    jobs = [
        {"name": "trained", "cmd": f"touch {marker}-trained"},  # its default summary exists: skipped
        {"name": "broken", "cmd": "echo about to fail; exit 7"},
        {"name": "after", "cmd": f"touch {marker}-after", "summary": str(tmp_path / "nowhere.json")},
        {"name": "greedy", "cmd": "true", "gpus": 3},  # more cards than the queue has
        {"name": "wide", "cmd": f"echo $CUDA_VISIBLE_DEVICES > {marker}-wide", "gpus": 2, "summary": None},
        {"name": "uses-py", "cmd": f"$PY -c \"import queue, sys; print(sys.executable)\" > {marker}-py", "summary": None},
    ]
    done, events = run_queue(tmp_path, jobs, "--gpus", "0,1")
    assert done.returncode == 1  # some job failed
    outcome = {e["name"]: e["event"] for e in events if e["event"] != "started"}
    assert outcome == {"trained": "skipped", "broken": "failed", "after": "finished", "greedy": "failed",
                       "wide": "finished", "uses-py": "finished"}
    assert not Path(f"{marker}-trained").exists() and Path(f"{marker}-after").exists()
    assert Path(f"{marker}-wide").read_text().strip() == "0,1"
    assert Path(f"{marker}-py").read_text().strip() == sys.executable
    assert next(e for e in events if e["name"] == "broken" and e["event"] == "failed")["returncode"] == 7
    assert "about to fail" in (tmp_path / "work" / "logs" / "broken.log").read_text()
    assert "1 failed" not in done.stdout and "2 failed (broken, greedy)" in done.stdout


def test_queue_picks_up_appended_jobs_and_bad_lines(tmp_path: Path) -> None:
    jobs_file = tmp_path / "jobs.jsonl"
    jobs_file.write_text(json.dumps({"name": "first", "cmd": "sleep 1.5", "summary": None}) + "\n"
                         + "# a comment\n\nnot json at all\n" + json.dumps({"name": "first", "cmd": "true"}) + "\n")
    env = dict(os.environ, GPTPP_WORK=str(tmp_path / "work"))
    queue = subprocess.Popen([sys.executable, str(REPO / "scripts" / "job_queue.py"), str(jobs_file), "--gpus", "0",
                              "--no-check-free", "--poll", "0.1"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    time.sleep(0.7)
    with open(jobs_file, "a") as f:  # appended while the first job runs
        f.write(json.dumps({"name": "second", "cmd": "echo second ran", "summary": None}) + "\n")
    output, _ = queue.communicate(timeout=60)
    assert queue.returncode == 0, output
    assert "second ran" in (tmp_path / "work" / "logs" / "second.log").read_text()
    assert "not JSON" in output and "second job named 'first'" in output
    events = [json.loads(line) for line in (tmp_path / "work" / "logs" / "queue-jobs.state.jsonl").read_text().splitlines()]
    assert [(e["name"], e["event"]) for e in events] == [("first", "started"), ("first", "finished"),
                                                         ("second", "started"), ("second", "finished")]


def test_queue_interrupt_stops_the_jobs(tmp_path: Path) -> None:
    import signal
    jobs_file = tmp_path / "jobs.jsonl"
    jobs_file.write_text(json.dumps({"name": "long", "cmd": "sleep 600", "summary": None}) + "\n"
                         + json.dumps({"name": "never", "cmd": "true", "summary": None}) + "\n")
    env = dict(os.environ, GPTPP_WORK=str(tmp_path / "work"))
    queue = subprocess.Popen([sys.executable, str(REPO / "scripts" / "job_queue.py"), str(jobs_file), "--gpus", "0",
                              "--no-check-free", "--poll", "0.1"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    time.sleep(1.0)
    queue.send_signal(signal.SIGTERM)
    output, _ = queue.communicate(timeout=60)
    events = [json.loads(line) for line in (tmp_path / "work" / "logs" / "queue-jobs.state.jsonl").read_text().splitlines()]
    assert [(e["name"], e["event"]) for e in events] == [("long", "started"), ("long", "failed")]
    assert events[-1]["reason"] == "terminated with the queue" and queue.returncode == 1
    pid = events[0]["pid"]
    assert not Path(f"/proc/{pid}").exists()  # the job is gone
