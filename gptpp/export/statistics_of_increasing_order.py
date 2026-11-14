"""Export for fig-08-01 `gpt-statistics-of-increasing-order`: a model's checkpoints against the n-gram predictors.

    python -m gptpp.export.statistics_of_increasing_order --run $GPTPP_WORK/ckpt/pico --run $GPTPP_WORK/ckpt/pico-2019

For every `ckpt_<step>.pt` of each run, on the run's own data: the loss on held-out text; the divergence from the
unigram and from the bigram predictor of the training set to the model, averaged over the positions of that text;
the loss on sequences sampled from each predictor; and what the checkpoint writes after a fixed opening (and, once,
what each predictor writes after it). With them go the levels the figure draws as hairlines: the held-out cross-entropy of the interpolated Kneser-Ney n-gram
models of orders 1 to 4, read from the dataset's `stats.json` (they are not recomputed here).

Units: everything is written in bits per token. The levels are also given in bits per byte, as the dataset's
statistics give them; the two are related by the held-out bytes per token of the data's meta (`<|bos|>` counted),
the convention of the trainer's `val_bpb`: bits per byte = bits per token / bytes_per_token.

Writes `statistics-of-increasing-order.json` (a few kilobytes; no arrays, so no manifest entries).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import torch

from ..data import TokenData
from ..interp import ngram_order as ng
from ..model import GPT
from ..tokenizer import Tokenizer
from .common import DATA, write_json

NAME = "statistics-of-increasing-order"
OPENING = "Once upon a time"


def checkpoints(run: Path) -> list[tuple[int, Path]]:
    """The run's `ckpt_<step>.pt` files, by step."""
    found = [(int(m.group(1)), f) for f in run.glob("ckpt_*.pt") if (m := re.fullmatch(r"ckpt_(\d+)\.pt", f.name))]
    return sorted(found)


def export_run(run: Path, device: str, windows: int, sampled: int, new_tokens: int, seed: int,
               cache: dict) -> dict:
    config = json.loads((run / "config.json").read_text())
    data_name = config["train_config"]["data"]
    if data_name not in cache:                           # the tables of a corpus are counted once
        data = TokenData(data_name)
        started = time.time()
        predictors = ng.Predictors(*ng.count_tables(data)).to(device)
        print(f"{data_name}: unigram and bigram tables of {data.meta['n_train_tokens']:,} tokens in {time.time() - started:.0f} s")
        cache[data_name] = (data, predictors)
    data, predictors = cache[data_name]
    tok = Tokenizer.load(data.meta["tokenizer"])
    length = config["model_config"]["block_size"] + 1
    text = ng.held_out_windows(data, windows, length).to(device)
    from_unigram = ng.sample_unigram(predictors, sampled, length, seed)
    from_bigram = ng.sample_bigram(predictors, sampled, length, data.meta["bos_id"], seed)
    opening = torch.tensor([tok.encode(OPENING, bos=True)], device=device)
    bits = ng.nats_to_bits
    out: dict = {"name": run.name, "preset": config.get("preset"), "data": data_name, "context": length - 1,
                 "steps": [], "tokens": [],
                 "loss": [], "kl_unigram": [], "kl_bigram": [], "loss_unigram_samples": [], "loss_bigram_samples": [],
                 "samples": []}
    for step, path in checkpoints(run):
        model, _ = GPT.from_checkpoint(path, device)
        seen = torch.load(path, map_location="cpu", weights_only=True)["tokens"]
        row = ng.measure(model, text, predictors)
        out["steps"].append(step)
        out["tokens"].append(int(seen))
        out["loss"].append(bits(row["loss"]))
        out["kl_unigram"].append(bits(row["kl_unigram"]))
        out["kl_bigram"].append(bits(row["kl_bigram"]))
        out["loss_unigram_samples"].append(bits(ng.loss_on(model, from_unigram)))
        out["loss_bigram_samples"].append(bits(ng.loss_on(model, from_bigram)))
        written = model.generate(opening, new_tokens, temperature=1.0, seed=seed)[0, opening.size(1):].tolist()
        out["samples"].append([tok.token_bytes(i).decode("utf-8", errors="replace") for i in written])
        print(f"{run.name} step {step:6d}  loss {out['loss'][-1]:6.3f} bits  KL from unigram {out['kl_unigram'][-1]:6.3f}  "
              f"from bigram {out['kl_bigram'][-1]:6.3f}  on unigram samples {out['loss_unigram_samples'][-1]:6.3f}  "
              f"on bigram samples {out['loss_bigram_samples'][-1]:6.3f}  | {''.join(out['samples'][-1])[:60]!r}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", action="append", required=True, help="a run directory with ckpt_<step>.pt files (repeatable)")
    ap.add_argument("--out", default=str(DATA))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--windows", type=int, default=1024, help="held-out windows of the model's context")
    ap.add_argument("--sampled", type=int, default=1024, help="sequences sampled from each predictor")
    ap.add_argument("--new-tokens", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cache: dict = {}
    runs = [export_run(Path(r), args.device, args.windows, args.sampled, args.new_tokens, args.seed, cache) for r in args.run]
    data, predictors = cache[runs[0]["data"]]
    tok = Tokenizer.load(data.meta["tokenizer"])
    opening = tok.encode(OPENING, bos=True)
    stats = json.loads((data.dir / "stats.json").read_text())
    per_token = data.meta["bytes_per_token_val"]
    levels = [{"n": o["n"], "bits_per_token": o["held_out_bits_per_token"], "bits_per_byte": o["held_out_bits_per_byte"]}
              for o in stats["ngrams"]["orders"]]
    for level in levels:                                 # the two units of a level are the same number
        assert abs(level["bits_per_token"] / per_token - level["bits_per_byte"]) < 1e-6
    out = {
        "dataset": data.meta["dataset"], "tokenizer": data.meta["tokenizer"], "vocabulary": data.vocab_size,
        "bytes_per_token": per_token, "train_tokens": data.meta["n_train_tokens"],
        "levels": levels, "levels_note": stats["ngrams"]["smoothing"],
        "levels_source": f"the loss on the held-out stream of n-gram models of orders 1 to 4 counted on the first "
                         f"{stats['ngrams']['sample_tokens']:,} training tokens, with interpolated Kneser-Ney smoothing "
                         f"(the dataset's stats.json): upper estimates of the levels H_1 .. H_4",
        "divergence": f"KL(n-gram predictor || model) between next-token distributions, at every position of the "
                      f"held-out windows, averaged; the unigram and the bigram predictor are the maximum-likelihood "
                      f"tables of all {data.meta['n_train_tokens']:,} training tokens (their own entropies are under "
                      f"`predictors`)",
        "predictors": {"unigram_entropy": ng.nats_to_bits(predictors.unigram_entropy()),
                       "bigram_entropy": ng.nats_to_bits(predictors.bigram_entropy()),
                       "uniform": math.log2(data.vocab_size)},
        "eval": {"windows": args.windows, "length": runs[0]["context"], "sampled": args.sampled, "opening": OPENING, "new_tokens": args.new_tokens, "temperature": 1.0,
                 "seed": args.seed},
        "predictor_samples": {name: [tok.token_bytes(i).decode("utf-8", errors="replace")
                                      for i in ng.continue_with(predictors, order, opening[-1], args.new_tokens, args.seed)]
                              for name, order in (("unigram", 1), ("bigram", 2))},
        "units": "bits per token; bits per byte = bits per token / bytes_per_token",
        "runs": runs,
    }
    size = write_json(Path(args.out) / f"{NAME}.json", out)
    print(f"wrote {NAME}.json, {size:,} bytes")


if __name__ == "__main__":
    main()
