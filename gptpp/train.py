"""The trainer.

    python -m gptpp.train --preset gptpp-ladder --data fineweb-edu/bpe32k --run NAME --tokens 150e6 \
        [--set key=value ...] [--train-preset PRESET] [--out DIR] [--seed 0] [--eval-tokens 10e6] \
        [--save-log 12 | --save-steps 100,1000] [--no-compile] [--results DIR | --results none] [--overwrite]

    python -m torch.distributed.run --standalone --nproc_per_node=4 -m gptpp.train ...   # four cards (torchrun)

The budget is in tokens. One step sees device_batch * seq_len * n_gpu * grad_accum tokens; the number of steps
is the budget divided by that, rounded. The model runs in bfloat16 autocast with TF32 matrix products and is
compiled with torch.compile. The data is read once, in order, without replacement (data.py); with several cards
each rank reads its own slice of every step and DistributedDataParallel averages the gradients.

What a run writes in <out>/<run>/ (default $GPTPP_WORK/ckpt/<run>/):

    config.json    model and training configuration, data meta, parameter counts, git commit, hardware
    log.jsonl      one JSON object a line, flushed as it is written:
                     {"kind": "train", "step", "tokens", "time", "loss", "lr", "lr_mult", "tok_s"}   every
                         log_every steps; the loss is the mean over the steps since the previous record, lr the
                         rate of the hidden matrices
                     {"kind": "val", "step", "tokens", "time", "val_loss", "val_bpb"}    at each evaluation, on the
                         fixed held-out batches
                     {"kind": "sample", "step", "tokens", "text"}    at each evaluation, if the data's tokenizer
                         is found: sample_tokens tokens after BOS at temperature 0.8, always with the same seed
                     {"kind": "error", "step", "error"}              if the run died
    ckpt_<step>.pt the checkpoints asked for; model.pt at the end:
                     {"model": state_dict, "model_config": dict, "step", "tokens"}
    summary.json   written only when the run completes: final and best val_loss and val_bpb, tokens, seconds,
                   tokens per second, compile seconds, peak memory, parameter counts

`step` counts optimizer steps taken, `tokens` the training tokens seen, `time` the seconds since the loop began.
val_bpb is bits per byte: val_loss / ln 2 / bytes_per_token_val, with the bytes per token of the held-out text
from the data meta. A copy of config.json, summary.json and log.jsonl (thinned to 2,000 training records) goes
to <repo>/results/runs/<run>/. Resuming is not supported; a crash or an interruption leaves the log readable,
with a last record that says what happened, and no summary.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from . import repo_dir, work_dir
from .config import RECIPES, ModelConfig, TrainConfig, apply_overrides, preset, preset_names, train_preset
from .data import TokenData, TrainLoader, ValLoader
from .generate import generate, load_tokenizer
from .model import GPT
from .optim import build_optimizers, lr_multiplier, set_lr


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GPT.", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", required=True, choices=preset_names() + list(RECIPES),
                        help="<recipe>-<size>; a recipe alone means its base size")
    parser.add_argument("--train-preset", default=None, choices=preset_names() + list(RECIPES),
                        help="take the training recipe (optimizer, rates, schedule) from this preset instead: "
                             "--preset gpt2-2019-ladder --train-preset gptpp-ladder trains the 2019 model the 2026 way")
    parser.add_argument("--run", required=True, help="name of the run: its directory under --out")
    parser.add_argument("--data", default=None, help="<dataset>/<tokenizer> under $GPTPP_WORK/data, or a directory")
    parser.add_argument("--tokens", type=float, default=None, help="training budget in tokens, e.g. 150e6")
    parser.add_argument("--set", dest="overrides", nargs="*", action="extend", default=[], metavar="KEY=VALUE",
                        help="override any field of ModelConfig or TrainConfig, e.g. --set act=gelu lr=0.01")
    parser.add_argument("--out", default=None, help="where run directories live (default $GPTPP_WORK/ckpt)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eval-tokens", type=float, default=None, help="evaluate every this many training tokens")
    parser.add_argument("--save-log", type=int, default=0, metavar="N",
                        help="checkpoint at the initialization and at N log-spaced steps")
    parser.add_argument("--save-steps", default="", metavar="STEPS", help="checkpoint at these steps, e.g. 100,1000")
    parser.add_argument("--no-compile", action="store_true", help="do not torch.compile the model")
    parser.add_argument("--results", default=None,
                        help="directory that receives <run>/ with the small result files "
                             "(default <repo>/results/runs; 'none' to skip)")
    parser.add_argument("--overwrite", action="store_true", help="replace a completed run of the same name")
    return parser.parse_args(argv)


def build_configs(args: argparse.Namespace) -> tuple[ModelConfig, TrainConfig]:
    """The preset, then the dedicated flags, then the --set overrides."""
    flags = dict(data=args.data, tokens=args.tokens, seed=args.seed, eval_tokens=args.eval_tokens)
    train_cfg = train_preset(args.train_preset or args.preset)
    train_cfg = train_cfg.replace(**{k: v for k, v in flags.items() if v is not None})
    if args.no_compile:
        train_cfg = train_cfg.replace(compile=False)
    model_cfg, train_cfg = apply_overrides([preset(args.preset), train_cfg], args.overrides)
    return model_cfg, train_cfg


def setup_distributed() -> tuple[int, int, torch.device]:
    """(rank, number of ranks, device). torchrun starts one process per card and tells each its RANK, LOCAL_RANK
    and WORLD_SIZE; a process started directly is alone."""
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and "LOCAL_RANK" in os.environ:
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl")
        return dist.get_rank(), dist.get_world_size(), device
    return 0, 1, torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")


def checkpoint_steps(save_log: int, save_steps: str, n_steps: int) -> set[int]:
    """The steps after which a checkpoint is written (the final model is always saved as model.pt)."""
    steps = {int(s) for s in save_steps.split(",") if s.strip()}
    if save_log > 0:
        steps |= {0} | {int(round(s)) for s in np.geomspace(1, n_steps, save_log)}
    return {s for s in steps if 0 <= s <= n_steps}


def save_checkpoint(path: Path, model: GPT, step: int, tokens: int) -> None:
    """Written under a temporary name and renamed, so that a file with the final name is always whole."""
    temporary = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "model_config": model.cfg.to_dict(), "step": step, "tokens": tokens},
               temporary)
    os.replace(temporary, path)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: ValLoader, autocast: torch.autocast, world: int) -> float:
    """Mean cross-entropy in nats over the fixed held-out windows of all ranks."""
    model.eval()
    device = next(model.parameters()).device
    total = torch.zeros(2, dtype=torch.float64, device=device)  # sum of token losses, number of tokens
    for x, y, n_targets in loader:
        with autocast:
            loss = model(x, y)
        total[0] += loss.double() * n_targets
        total[1] += n_targets
    model.train()
    if world > 1:
        dist.all_reduce(total)
    return (total[0] / total[1]).item()


def rank_spread(model: GPT) -> float:
    """The largest difference between ranks of the sum of all parameters. Ranks exchange gradients, never
    parameters: they hold the same model only because every rank takes the same step, and this checks it."""
    checksum = torch.stack([p.detach().double().sum() for p in model.parameters()]).sum()
    low, high = checksum.clone(), checksum.clone()
    dist.all_reduce(low, op=dist.ReduceOp.MIN)
    dist.all_reduce(high, op=dist.ReduceOp.MAX)
    return (high - low).item()


class RunLog:
    """log.jsonl. Every record is flushed when written: whatever happens to the run, the log can be read."""

    def __init__(self, path: Path):
        self.file = open(path, "w", encoding="utf-8")

    def write(self, kind: str, **fields: object) -> None:
        clean = {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in fields.items()}
        self.file.write(json.dumps({"kind": kind, **clean}) + "\n")
        self.file.flush()

    def close(self) -> None:
        self.file.close()


def git_commit() -> Optional[str]:
    try:
        out = subprocess.run(["git", "-C", str(repo_dir()), "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def copy_results(run_dir: Path, results_dir: Path, max_train_records: int = 2000) -> None:
    """Copy the small files of a run into the repository: config.json, summary.json and log.jsonl, the latter
    thinned to at most `max_train_records` evenly spaced training records (all other records are kept)."""
    results_dir.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "summary.json"):
        shutil.copyfile(run_dir / name, results_dir / name)
    lines = (run_dir / "log.jsonl").read_text(encoding="utf-8").splitlines()
    is_train = [json.loads(line)["kind"] == "train" for line in lines]
    stride = max(1, math.ceil(sum(is_train) / max_train_records))
    kept, seen = [], 0
    for line, train in zip(lines, is_train):
        if train:
            seen += 1
            if seen % stride != 0 and seen != sum(is_train):
                continue
        kept.append(line)
    (results_dir / "log.jsonl").write_text("\n".join(kept) + "\n", encoding="utf-8")


def _interrupt(signum: int, frame: object) -> None:
    raise KeyboardInterrupt(f"signal {signum}")


def train(args: argparse.Namespace) -> Optional[dict]:
    """Run the training described by the arguments; rank 0 returns the summary."""
    rank, world, device = setup_distributed()
    master = rank == 0
    on_gpu = device.type == "cuda"

    def say(message: str) -> None:
        if master:
            print(message, flush=True)

    # ---- configuration, data, budget
    model_cfg, train_cfg = build_configs(args)
    data = TokenData(train_cfg.data)
    if model_cfg.vocab_size != data.vocab_size:
        say(f"vocab_size {model_cfg.vocab_size} -> {data.vocab_size}, the vocabulary of {train_cfg.data}")
        model_cfg = model_cfg.replace(vocab_size=data.vocab_size)
    seq_len = train_cfg.seq_len or model_cfg.block_size
    tokens_per_step = train_cfg.device_batch * seq_len * world * train_cfg.grad_accum
    n_steps = max(1, round(train_cfg.tokens / tokens_per_step))
    total_tokens = n_steps * tokens_per_step
    eval_every = max(1, round(train_cfg.eval_tokens / tokens_per_step))
    eval_steps = set(range(0, n_steps, eval_every)) | {n_steps}
    save_steps = checkpoint_steps(args.save_log, args.save_steps, n_steps)
    bytes_per_token = data.meta.get("bytes_per_token_val")

    run_dir = Path(args.out) / args.run if args.out else work_dir() / "ckpt" / args.run
    if master:
        if (run_dir / "summary.json").exists() and not args.overwrite:
            raise SystemExit(f"{run_dir} holds a completed run; pass --overwrite to replace it")
        run_dir.mkdir(parents=True, exist_ok=True)
        for pattern in ("summary.json", "model.pt", "ckpt_*.pt", "interrupted_*.pt"):  # of an earlier attempt
            for stale in run_dir.glob(pattern):
                stale.unlink()

    # ---- model and optimizers
    torch.backends.cuda.matmul.allow_tf32 = True  # TF32: float32 matrix products with a 10-bit mantissa
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(train_cfg.seed)  # the same initialization on every rank
    model = GPT(model_cfg).to(device)
    torch.manual_seed(train_cfg.seed + 1 + rank)  # different dropout masks on every rank
    optimizers = build_optimizers(model, train_cfg, tokens_per_step)
    compiled = torch.compile(model) if train_cfg.compile else model
    net = DistributedDataParallel(compiled, device_ids=[device.index]) if world > 1 else compiled
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=on_gpu)

    train_loader = TrainLoader(data, seq_len, train_cfg.device_batch, rank, world, train_cfg.seed, device)
    val_loader = ValLoader(data, seq_len, train_cfg.device_batch, rank, world, train_cfg.val_tokens, device)
    tokenizer = None
    if master and train_cfg.sample_tokens > 0 and data.meta.get("tokenizer"):
        tokenizer = load_tokenizer(data.meta["tokenizer"])
        if tokenizer is None:
            say(f"no samples will be logged: the tokenizer {data.meta['tokenizer']!r} could not be loaded")

    params = model.num_params()
    log: Optional[RunLog] = None
    if master:
        config = {
            "run": args.run, "preset": args.preset, "train_preset": args.train_preset or args.preset,
            "overrides": args.overrides, "argv": sys.argv,
            "model_config": model_cfg.to_dict(), "train_config": train_cfg.to_dict(), "data_meta": data.meta,
            "params": params,
            "derived": {"seq_len": seq_len, "n_gpu": world, "tokens_per_step": tokens_per_step, "steps": n_steps,
                        "tokens": total_tokens, "eval_every_steps": eval_every,
                        "val_windows": val_loader.n_windows, "checkpoint_steps": sorted(save_steps),
                        # what each parameter group is trained with: optimizer, peak rate, weight decay
                        "groups": {group["name"]: {"optimizer": type(optimizer).__name__, "lr": group["base_lr"],
                                                   "weight_decay": group["weight_decay"]}
                                   for optimizer in optimizers for group in optimizer.param_groups}},
            "git_commit": git_commit(),
            "hardware": {"gpu": torch.cuda.get_device_name(device) if on_gpu else "cpu", "n_gpu": world,
                         "torch": torch.__version__, "cuda": torch.version.cuda, "python": platform.python_version()},
        }
        (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        log = RunLog(run_dir / "log.jsonl")
        say(f"{args.run}: {args.preset}, {params['total'] / 1e6:.1f}M parameters, {train_cfg.optimizer}; "
            f"{n_steps} steps of {tokens_per_step:,} tokens ({world} x {train_cfg.device_batch} x {seq_len}"
            f"{' x ' + str(train_cfg.grad_accum) if train_cfg.grad_accum > 1 else ''}) = {total_tokens / 1e6:.1f}M "
            f"tokens of {train_cfg.data}; evaluation every {eval_every} steps on {val_loader.n_windows} windows")

    # ---- the loop
    previous_handler = signal.signal(signal.SIGTERM, _interrupt)  # a kill is handled like Ctrl-C
    if on_gpu:
        torch.cuda.reset_peak_memory_stats(device)
    step = 0
    start = time.perf_counter()
    interval_start = start  # start of the current logging interval, pushed forward past every evaluation
    interval_steps = 0
    loss_sum = torch.zeros((), device=device)
    train_seconds, first_interval_seconds, first_interval_steps = 0.0, 0.0, 0
    eval_durations: list[float] = []
    val_loss = best_val_loss = float("inf")
    best_step = 0
    train_loss = float("nan")
    try:
        for step in range(n_steps + 1):
            tokens = step * tokens_per_step

            if step in eval_steps or step in save_steps:
                if on_gpu:
                    torch.cuda.synchronize(device)
                pause = time.perf_counter()
                if step in eval_steps:
                    val_loss = evaluate(compiled, val_loader, autocast, world)
                    eval_durations.append(time.perf_counter() - pause)
                    if val_loss < best_val_loss:
                        best_val_loss, best_step = val_loss, step
                    if master:
                        val_bpb = val_loss / math.log(2) / bytes_per_token if bytes_per_token else None
                        log.write("val", step=step, tokens=tokens, time=round(pause - start, 3),
                                  val_loss=val_loss, val_bpb=val_bpb)
                        say(f"eval  step {step:>6} | tokens {tokens / 1e6:8.2f}M | val_loss {val_loss:.4f}"
                            + (f" | val_bpb {val_bpb:.4f}" if val_bpb else "") + f" | {eval_durations[-1]:.1f} s")
                        if tokenizer is not None:
                            try:
                                prompt = torch.tensor([[int(data.meta["bos_id"])]], device=device)
                                with autocast:
                                    sampled = generate(model, prompt, train_cfg.sample_tokens, temperature=0.8, seed=0)
                                text = tokenizer.decode(sampled[0].tolist())
                                log.write("sample", step=step, tokens=tokens, text=text)
                            except Exception as error:  # a sample is a convenience: it must never stop a run
                                say(f"sampling failed and is turned off: {error!r}")
                                tokenizer = None
                if master and step in save_steps:
                    save_checkpoint(run_dir / f"ckpt_{step}.pt", model, step, tokens)
                interval_start += time.perf_counter() - pause

            if step == n_steps:
                break

            # The rate of this step is the schedule read at the middle of the step.
            multiplier = lr_multiplier((step + 0.5) * tokens_per_step, total_tokens, train_cfg)
            set_lr(optimizers, multiplier)
            for micro_step in range(train_cfg.grad_accum):
                x, y = train_loader.next_batch()
                if world > 1:  # ranks exchange gradients only on the last backward pass of the step
                    net.require_backward_grad_sync = micro_step == train_cfg.grad_accum - 1
                with autocast:
                    loss = net(x, y)
                (loss / train_cfg.grad_accum).backward()
                loss_sum += loss.detach() / train_cfg.grad_accum
            if train_cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            for optimizer in optimizers:
                optimizer.step()
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            interval_steps += 1

            if (step + 1) % train_cfg.log_every == 0 or step + 1 == n_steps:
                if world > 1:
                    dist.all_reduce(loss_sum)
                train_loss = loss_sum.item() / (interval_steps * world)  # .item() waits for the card
                now = time.perf_counter()
                seconds, interval_start = now - interval_start, now
                train_seconds += seconds
                if first_interval_steps == 0:  # the first interval pays for compilation
                    first_interval_seconds, first_interval_steps = seconds, interval_steps
                tok_s = interval_steps * tokens_per_step / seconds
                if master:
                    done = (step + 1) * tokens_per_step
                    matrix_lr = next(g["lr"] for o in optimizers for g in o.param_groups if g["name"] == "matrices")
                    log.write("train", step=step + 1, tokens=done, time=round(now - start, 3), loss=train_loss,
                              lr=matrix_lr, lr_mult=multiplier, tok_s=round(tok_s))
                    remaining = (total_tokens - done) / tok_s / 60
                    say(f"step {step + 1:>6}/{n_steps} | tokens {done / 1e6:8.2f}M | loss {train_loss:.4f} | "
                        f"lr x{multiplier:.3f} | {tok_s / 1e3:6.1f}k tok/s | {remaining:5.1f} min left")
                if not math.isfinite(train_loss):
                    raise FloatingPointError(f"the training loss is {train_loss} at step {step + 1}")
                loss_sum.zero_()
                interval_steps = 0
    except BaseException as error:
        if master and log is not None:
            log.write("error", step=step, tokens=step * tokens_per_step,
                      error=f"{type(error).__name__}: {error}")
            log.close()
            if isinstance(error, KeyboardInterrupt) and step > 0:
                save_checkpoint(run_dir / f"interrupted_{step}.pt", model, step, step * tokens_per_step)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_handler)

    # ---- the end
    spread = rank_spread(model) if world > 1 else 0.0
    if not master:
        return None
    save_checkpoint(run_dir / "model.pt", model, n_steps, total_tokens)
    seconds = time.perf_counter() - start
    steady_steps = n_steps - first_interval_steps
    if steady_steps > 0:  # throughput and step time without the first interval
        step_seconds = (train_seconds - first_interval_seconds) / steady_steps
    else:
        step_seconds = train_seconds / n_steps
    eval_seconds = float(np.median(eval_durations[1:])) if len(eval_durations) > 1 else eval_durations[0]
    to_bpb = (lambda nats: nats / math.log(2) / bytes_per_token) if bytes_per_token else (lambda nats: None)
    summary = {
        "run": args.run, "status": "completed", "preset": args.preset,
        "val_loss": val_loss, "val_bpb": to_bpb(val_loss),
        "best_val_loss": best_val_loss, "best_val_bpb": to_bpb(best_val_loss), "best_step": best_step,
        "train_loss": train_loss, "steps": n_steps, "tokens": total_tokens, "n_gpu": world,
        "seconds": round(seconds, 1), "train_seconds": round(train_seconds, 1),
        "tokens_per_second": round(tokens_per_step / step_seconds),
        "step_seconds": round(step_seconds, 4), "eval_seconds": round(eval_seconds, 2),
        # What compilation cost: the first interval and the first evaluation, beyond their steady-state time.
        "compile_seconds": round(max(0.0, first_interval_seconds - first_interval_steps * step_seconds)
                                 + max(0.0, eval_durations[0] - eval_seconds), 1),
        "peak_memory_gb": round(torch.cuda.max_memory_allocated(device) / 2**30, 2) if on_gpu else None,
        "rank_spread": spread,  # zero when all ranks ended with the same parameters
        "params": params,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log.close()
    if args.results != "none":
        results_root = Path(args.results) if args.results else repo_dir() / "results" / "runs"
        copy_results(run_dir, results_root / args.run)
    say(f"done: val_loss {val_loss:.4f}" + (f" ({summary['val_bpb']:.4f} bits per byte)" if bytes_per_token else "")
        + f", {summary['tokens_per_second']:,} tok/s, {seconds / 60:.1f} min, "
        + (f"peak memory {summary['peak_memory_gb']} GB, " if on_gpu else "") + f"results in {run_dir}")
    return summary


def main(argv: Optional[list[str]] = None) -> Optional[dict]:
    try:
        return train(parse_args(argv))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
