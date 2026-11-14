"""Where the normalization sits, measured: the gradient of every layer at initialization for post-norm and pre-norm
stacks of three depths under two initializations, and the first tokens of four short training runs (pre and post,
with and without warmup). Writes `assets/data/gpt/gradient-by-depth.json` for the figure `gpt-gradient-by-depth`.

    python -m gptpp.export.gradient_by_depth [--device cpu] [--threads 24] [--seeds 4] [--runs results/runs] [--out <dir>]
    python -m gptpp.export.gradient_by_depth --keep-init        # only the training runs again, when more have finished

Part one needs no training (about a minute on a CPU with 8 seeds and 24 threads). For each
initialization, each placement (post, pre) and each depth (12, 24, 48) it builds `--seeds` randomly initialized
models of the 2019 recipe (learned positions, LayerNorm with gain and bias, GELU, biases) at width 256 with four
heads, runs one batch of real held-out tokens (8 windows of 256 tokens of FineWeb-EDU) through each, backpropagates
the mean cross-entropy, and records for every block the norm of the gradient of its weights and the size of the
residual stream after it. The same batch is used for every model. The two initializations:

    gpt2     the 2019 recipe as the lab draws it (`init=gpt2`): every weight N(0, 0.02^2), the matrices that write
             into the stream scaled by 1/sqrt(2L), embedding tied to the unembedding
    glorot   the initialization of the 2017 Transformer, the setting of the theorem of Xiong et al. (2020,
             arXiv:2002.04745): every matrix of a block N(0, 2/(fan_in + fan_out)) (Glorot and Bengio, 2010), biases
             zero, token and position embeddings N(0, 1/2) each so that the stream enters with unit variance, and an
             unembedding of its own, N(0, 1/d), so that the first logits have unit variance. A branch then writes
             at the scale of the stream. (The lab's model has no such option: this exporter overwrites the weights.)

Part two copies the training loss of the four runs that differ only in placement and warmup (8 layers, 512 wide,
the 2019 recipe with AdamW at its tuned rate, 50M tokens): `warm-2019-base` (pre-norm, warmup), `warm-2019-none`
(pre-norm, no warmup), `warm-2019-post` (post-norm, warmup), `warm-2019-post-none` (post-norm, no warmup). Until
`warm-2019-base` has finished, the ladder's first rung, `ladder-0-gpt2-2019-s0`, stands in for it: the same model
and warmup, trained for 100M tokens, so its rate decays more slowly over the 50M that are drawn. A run that has
not finished is listed with "finished": false and no curve, and the exporter is run again when it lands.

The file, field by field:

    generated_by   "gptpp.export.gradient_by_depth"
    init           the measurement at initialization
      model          {d_model, n_head, vocab_size, block_size, recipe: "gpt2-2019"}
      batch          {windows, tokens (per window), source}: the one batch every model sees
      seeds          how many random initializations each cell averages
      depths         [12, 24, 48]
      inits          {"gpt2": {...}, "glorot": {...}}, each holding
        loss           {"post": {"12": mean loss at initialization, ...}, "pre": {...}} (ln 32768 = 10.397 is uniform)
        layers         {"post": {"12": {...}, "24": {...}, "48": {...}}, "pre": {...}}; each cell holds, per block
                       (index 0 is the block next to the embedding):
                         all      [mean over seeds of the gradient norm of all the block's parameters]
                         lo, hi   [the smallest and the largest of the seeds]
                         attn     [the same for the four attention matrices q, k, v, o together]
                         qk       [queries and keys alone: the matrices that decide where attention looks]
                         mlp      [the two MLP matrices together]
                         proj     [the MLP's second matrix alone, the one that writes into the stream]
                         stream   [root-mean-square of a coordinate of the residual stream after the block]
                       and the scalars  ratio (largest block / smallest block of `all`),
                                        last_over_first (last block / first block of `all`),
                                        embedding (gradient norm of the embedding matrices, mean over seeds)
        last_mlp       {"post": {"12": ..}, "pre": {..}}: gradient norm of the last block's second MLP matrix, the
                       quantity of the theorem, mean over seeds
    training       the four runs
      model          {n_layer, d_model, ...} of the runs
      runs[i]        {run, placement ("pre" | "post"), warmup_tokens, finished, lr (the peak learning rate),
                      tokens: [...], loss: [...] (the logged training loss, every tenth step), lr_mult: [...] (the
                      learning rate as a fraction of its peak at those steps), val_tokens: [...], val_loss: [...],
                      final_val_loss, final_val_bpb, loss_at: {"5e6": training loss at 5M tokens, "10e6", "20e6",
                      "50e6"}} (everything after `finished` only when the run has finished)
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import preset
from ..model import GPT
from .common import DATA, WORK, write_atomic

REPO = Path(__file__).resolve().parents[2]
DEPTHS = (12, 24, 48)
PLACEMENTS = ("post", "pre")
INITS = ("gpt2", "glorot")
# The four runs: for each, the names to look for in order (the first that has finished is taken) and its placement.
RUNS = ((("warm-2019-base", "ladder-0-gpt2-2019-s0"), "pre"), (("warm-2019-none",), "pre"),
        (("warm-2019-post",), "post"), (("warm-2019-post-none",), "post"))


def round_sig(o, digits: int = 6):   # six, as ladder.json: at five a loss of 7.69046 is stored as 7.6905 and printed to three decimals as 7.691
    if isinstance(o, float):
        return int(o) if o.is_integer() and abs(o) < 1e15 else float(f"{o:.{digits}g}")
    if isinstance(o, dict):
        return {k: round_sig(v, digits) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [round_sig(v, digits) for v in o]
    return o


def held_out_batch(windows: int, tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The first `windows` windows of the held-out tokens of FineWeb-EDU, with their targets (the next token)."""
    val = np.fromfile(WORK / "data" / "fineweb-edu" / "bpe32k" / "val.bin", dtype="<u2", count=windows * tokens + 1)
    idx = torch.from_numpy(val[:-1].astype(np.int64)).view(windows, tokens)
    targets = torch.from_numpy(val[1:].astype(np.int64)).view(windows, tokens)
    return idx, targets


def norm_of(params) -> float:
    return math.sqrt(sum(float(p.grad.double().pow(2).sum()) for p in params if p.grad is not None))


def glorot_(model: GPT) -> None:
    """Overwrite a model's weights with the initialization of the 2017 Transformer (see the top of this file)."""
    d = model.cfg.d_model
    for module in model.blocks.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    nn.init.normal_(model.embed.weight, std=math.sqrt(0.5))
    nn.init.normal_(model.pos_embed.weight, std=math.sqrt(0.5))
    nn.init.normal_(model.unembed.weight, std=1 / math.sqrt(d))


def measure(init: str, placement: str, depth: int, seed: int, idx: torch.Tensor, targets: torch.Tensor, width: int,
            heads: int, device: str) -> dict:
    cfg = preset("gpt2-2019-ladder").replace(d_model=width, n_head=heads, n_layer=depth, norm_placement=placement,
                                             tie_embeddings=init == "gpt2")
    torch.manual_seed(seed)
    model = GPT(cfg)
    if init == "glorot":
        glorot_(model)
    model = model.to(device)
    # One pass on the explicit path: the logits for the loss, and the stream after every block.
    logits, cache = model.run_with_cache(idx, names=["resid_post.*"])
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
    loss.backward()
    out = {"loss": loss.item(), "all": [], "attn": [], "qk": [], "mlp": [], "proj": []}
    for block in model.blocks:
        a = block.attn
        out["all"].append(norm_of(block.parameters()))
        out["attn"].append(norm_of([a.q.weight, a.k.weight, a.v.weight, a.o.weight]))
        out["qk"].append(norm_of([a.q.weight, a.k.weight]))
        out["mlp"].append(norm_of([block.mlp.fc.weight, block.mlp.proj.weight]))
        out["proj"].append(norm_of([block.mlp.proj.weight]))
    out["last_mlp"] = norm_of([model.blocks[-1].mlp.proj.weight])
    out["embedding"] = norm_of([model.embed.weight, model.pos_embed.weight])
    out["stream"] = [cache[f"resid_post.{l}"].detach().double().pow(2).mean().sqrt().item() for l in range(depth)]
    del model, logits, cache, loss
    if device != "cpu":
        torch.cuda.empty_cache()
    return out


def initialization(seeds: int, device: str, width: int = 256, heads: int = 4, windows: int = 8, tokens: int = 256) -> dict:
    idx, targets = held_out_batch(windows, tokens)
    idx, targets = idx.to(device), targets.to(device)
    inits = {}
    for init in INITS:
        layers: dict = {p: {} for p in PLACEMENTS}
        loss: dict = {p: {} for p in PLACEMENTS}
        last: dict = {p: {} for p in PLACEMENTS}
        for placement in PLACEMENTS:
            for depth in DEPTHS:
                runs = [measure(init, placement, depth, seed, idx, targets, width, heads, device) for seed in range(seeds)]
                stack = {k: np.array([r[k] for r in runs]) for k in ("all", "attn", "qk", "mlp", "proj", "stream")}
                mean = stack["all"].mean(0)
                cell = {
                    "all": mean.tolist(), "lo": stack["all"].min(0).tolist(), "hi": stack["all"].max(0).tolist(),
                    "attn": stack["attn"].mean(0).tolist(), "qk": stack["qk"].mean(0).tolist(),
                    "mlp": stack["mlp"].mean(0).tolist(), "proj": stack["proj"].mean(0).tolist(),
                    "stream": stack["stream"].mean(0).tolist(),
                    "ratio": float(mean.max() / mean.min()), "last_over_first": float(mean[-1] / mean[0]),
                    "embedding": float(np.mean([r["embedding"] for r in runs])),
                }
                layers[placement][str(depth)] = cell
                loss[placement][str(depth)] = float(np.mean([r["loss"] for r in runs]))
                last[placement][str(depth)] = float(np.mean([r["last_mlp"] for r in runs]))
                print(f"  {init:<6} {placement:<4} depth {depth:>2}: loss {loss[placement][str(depth)]:.3f}  block gradient first "
                      f"{mean[0]:.4f} last {mean[-1]:.4f} max/min {cell['ratio']:.2f}  last MLP matrix "
                      f"{last[placement][str(depth)]:.4f}  stream rms first {cell['stream'][0]:.3f} last {cell['stream'][-1]:.3f}",
                      flush=True)
        inits[init] = {"loss": loss, "layers": layers, "last_mlp": last}
    return {
        "model": {"d_model": width, "n_head": heads, "vocab_size": 32768, "block_size": 512, "recipe": "gpt2-2019"},
        "batch": {"windows": windows, "tokens": tokens, "source": "FineWeb-EDU, held-out"},
        "seeds": seeds, "depths": list(DEPTHS), "inits": inits,
    }


def training(runs_dir: Path) -> dict:
    out, model = [], None
    def finished(name: str) -> bool:
        summary = runs_dir / name / "summary.json"
        return summary.exists() and json.loads(summary.read_text()).get("status") == "completed"

    for names, placement in RUNS:
        run = next((name for name in names if finished(name)), names[0])
        folder = runs_dir / run
        rec = {"run": run, "placement": placement, "warmup_tokens": None, "finished": False}
        if finished(run):
            config, summary = json.loads((folder / "config.json").read_text()), json.loads((folder / "summary.json").read_text())
            assert config["model_config"]["norm_placement"] == placement, run
            model = model or {k: config["model_config"][k] for k in ("n_layer", "d_model", "n_head", "block_size", "vocab_size")}
            tokens, loss, lr_mult, val_tokens, val_loss = [], [], [], [], []
            for line in (folder / "log.jsonl").read_text().splitlines():
                r = json.loads(line)
                if r.get("kind") == "train":
                    tokens.append(int(r["tokens"])); loss.append(r["loss"]); lr_mult.append(r.get("lr_mult"))
                elif r.get("kind") == "val":
                    val_tokens.append(int(r["tokens"])); val_loss.append(r["val_loss"])

            def at(t: float):
                later = [v for tok, v in zip(tokens, loss) if tok >= t]
                return later[0] if later else None

            rec.update(finished=True, warmup_tokens=config["train_config"]["warmup_tokens"], lr=config["train_config"]["lr"],
                       tokens=tokens, loss=loss, lr_mult=lr_mult, val_tokens=val_tokens, val_loss=val_loss,
                       final_val_loss=summary["val_loss"], final_val_bpb=summary["val_bpb"],
                       loss_at={"5e6": at(5e6), "10e6": at(10e6), "20e6": at(20e6), "50e6": at(50e6)})
        out.append(rec)
        print(f"  {run:<24} {placement:<4} " + (f"warmup {rec['warmup_tokens'] / 1e6:.0f}M tokens, final validation loss "
              f"{rec['final_val_loss']:.4f}, {len(rec['tokens'])} points" if rec["finished"] else "not finished"))
    return {"model": model, "runs": out}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--threads", type=int, default=0, help="CPU threads to use (0: PyTorch's default)")
    ap.add_argument("--runs", default=str(REPO / "results" / "runs"))
    ap.add_argument("--out", default=str(DATA))
    ap.add_argument("--keep-init", action="store_true", help="reuse the measurement at initialization already in the file")
    args = ap.parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    out = Path(args.out)
    target = out / "gradient-by-depth.json"
    if args.keep_init and target.exists():
        init = json.loads(target.read_text())["init"]
        print("  initialization: kept from the existing file")
    else:
        init = initialization(args.seeds, args.device)
    result = {"generated_by": "gptpp.export.gradient_by_depth", "init": init, "training": training(Path(args.runs))}
    out.mkdir(parents=True, exist_ok=True)
    text = json.dumps(round_sig(result), separators=(",", ":"))
    write_atomic(target, text.encode("utf-8"))
    print(f"{target}: {len(text)} bytes")


if __name__ == "__main__":
    main()
