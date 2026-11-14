"""The six activations of the MLP, measured on the runs that differ only in the activation: the loss each reaches,
and what its hidden units do on held-out text. Writes `assets/data/gpt/activation-zoo.json` for the figure
`gpt-activation-zoo`.

    python -m gptpp.export.activation_zoo [--device cpu] [--threads 24] [--tokens 102400] [--out <dir>]
                                          [--run relu2=<run> ...]

The runs are the one-at-a-time swaps around the 2026 recipe (8 layers, 512 wide, 50M tokens each): `swap-base`
(squared ReLU, the recipe's own) and `swap-act-relu`, `-gelu`, `-silu`, `-swiglu`, `-geglu`. A run that has not
finished is listed with "finished": false and nothing measured; run the exporter again when it lands. `--run`
names another run for an activation (to try the exporter on a checkpoint that exists; the file records the name).

The loss is copied from the trainer's summary. The rest is measured here from the run's `model.pt`: the model
reads `--tokens` held-out tokens of FineWeb-EDU (windows of 512, the first of the held-out file) through
`run_with_cache`, and for every block the hidden units of its MLP (`mlp.l.post`: for a gated unit, the gate's
output times the linear value) and what enters the nonlinearity (`mlp.l.pre`: for a gated unit, the gate's input)
are summarized. About two minutes a run on a CPU.

The file, field by field:

    generated_by   "gptpp.export.activation_zoo"
    tokens         how many held-out tokens every model read
    threshold      0.01: an activation is "negligible" when its size is under this share of the root mean square
                   of its layer's activations (over all units and tokens), and a unit "fires" when it exceeds it
    model          {n_layer, d_model, n_head, block_size, vocab_size} of the runs
    acts[i]        one entry an activation, in the order relu, gelu, silu, geglu, swiglu, relu2:
      act            the name in the model's configuration
      run            the run it was measured on
      finished       whether that run has finished (everything below is absent otherwise)
      val_loss, val_bpb, best_val_loss     the run's validation loss at the end (nats a token, bits a byte) and its best
      train_tokens, seconds, tokens_per_second, params (total), d_mlp (hidden width of the MLP)
      measured       whether `model.pt` was there to measure (the fields below are absent otherwise)
      zero           the fraction of activations that are exactly zero (all layers pooled)
      negligible     the fraction that are negligible by the threshold, exact zeros included
      dead           the fraction of units that never fire on these tokens
      dead_exact     the fraction of units whose activation is exactly zero on every token
      kurtosis       the kurtosis of a layer's activations (fourth central moment over the squared variance; 3 for
                     a normal law), mean over the layers
      pre_negative   the fraction of inputs to the nonlinearity that are negative
      pre_hist       {lo, hi, counts}: the distribution of those inputs (all layers pooled) as fractions in equal
                     bins from lo to hi; what falls outside is in the first and the last bin
      by_layer       {zero, negligible, dead, dead_exact, kurtosis, rms, pre_negative, pre_rms}: the same per block
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..model import GPT
from .common import CKPT, DATA, WORK, write_atomic

REPO = Path(__file__).resolve().parents[2]
ACTS = ("relu", "gelu", "silu", "geglu", "swiglu", "relu2")
THRESHOLD = 0.01
PRE_LO, PRE_HI, PRE_BINS = -6.0, 6.0, 60
LOG_LO, LOG_HI, LOG_BINS = -12.0, 4.0, 320  # the sizes of activations, in decades, for the share under a threshold


def round_sig(o, digits: int = 5):
    if isinstance(o, float):
        return int(o) if o.is_integer() and abs(o) < 1e15 else float(f"{o:.{digits}g}")
    if isinstance(o, dict):
        return {k: round_sig(v, digits) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [round_sig(v, digits) for v in o]
    return o


def default_run(act: str) -> str:
    return "swap-base" if act == "relu2" else f"swap-act-{act}"


def held_out(tokens: int, window: int = 512) -> torch.Tensor:
    n = tokens // window
    val = np.fromfile(WORK / "data" / "fineweb-edu" / "bpe32k" / "val.bin", dtype="<u2", count=n * window)
    return torch.from_numpy(val.astype(np.int64)).view(n, window)


@torch.no_grad()
def measure(model: GPT, idx: torch.Tensor, device: str, batch: int = 8) -> dict:
    L = model.cfg.n_layer
    width = model.cfg.d_mlp
    count = 0
    sums = np.zeros((L, 4))                                   # sums of a, a^2, a^3, a^4 over a layer's activations
    zeros = np.zeros(L)
    logh = np.zeros((L, LOG_BINS))
    peak = torch.zeros(L, width, dtype=torch.float64)         # the largest size of each unit's activation
    alive = torch.zeros(L, width, dtype=torch.bool)           # whether a unit was ever not exactly zero
    pre_hist = np.zeros((L, PRE_BINS))
    pre_neg = np.zeros(L)
    pre_sq = np.zeros(L)
    for start in range(0, idx.size(0), batch):
        x = idx[start:start + batch].to(device)
        _, cache = model.run_with_cache(x, names=["mlp.*.pre", "mlp.*.post"])
        for l in range(L):
            a = cache[f"mlp.{l}.post"].reshape(-1, width).double().cpu()
            sums[l] += [a.sum().item(), a.pow(2).sum().item(), a.pow(3).sum().item(), a.pow(4).sum().item()]
            zeros[l] += (a == 0).sum().item()
            size = a.abs()
            logh[l] += torch.histc(torch.log10(size[size > 0]), bins=LOG_BINS, min=LOG_LO, max=LOG_HI).numpy()
            peak[l] = torch.maximum(peak[l], size.amax(0))
            alive[l] |= (a != 0).any(0)
            p = cache[f"mlp.{l}.pre"].reshape(-1).double().cpu()
            pre_hist[l] += torch.histc(p.clamp(PRE_LO, PRE_HI - 1e-9), bins=PRE_BINS, min=PRE_LO, max=PRE_HI).numpy()
            pre_neg[l] += (p < 0).sum().item()
            pre_sq[l] += p.pow(2).sum().item()
            if l == 0:
                count += a.numel()
        del cache
    n = float(count)
    n_pre = pre_hist.sum(1)
    mean = sums[:, 0] / n
    m2 = sums[:, 1] / n - mean ** 2
    m4 = sums[:, 3] / n - 4 * mean * sums[:, 2] / n + 6 * mean ** 2 * sums[:, 1] / n - 3 * mean ** 4
    rms = np.sqrt(sums[:, 1] / n)
    eps = THRESHOLD * rms
    edges = np.linspace(LOG_LO, LOG_HI, LOG_BINS + 1)
    negligible = np.zeros(L)
    for l in range(L):
        cut = np.log10(max(eps[l], 1e-300))
        below = np.clip((cut - edges[:-1]) / (edges[1] - edges[0]), 0, 1)      # the share of each bin under the threshold
        negligible[l] = (zeros[l] + float((logh[l] * below).sum())) / n
    dead = (peak <= torch.from_numpy(eps)[:, None]).double().mean(1).numpy()
    dead_exact = (~alive).double().mean(1).numpy()
    by_layer = {
        "zero": (zeros / n).tolist(), "negligible": negligible.tolist(), "dead": dead.tolist(), "dead_exact": dead_exact.tolist(),
        "kurtosis": (m4 / m2 ** 2).tolist(), "rms": rms.tolist(), "pre_negative": (pre_neg / n_pre).tolist(),
        "pre_rms": np.sqrt(pre_sq / n_pre).tolist(),
    }
    return {
        "measured": True, "zero": float(zeros.sum() / (n * L)), "negligible": float(negligible.mean()),
        "dead": float(dead.mean()), "dead_exact": float(dead_exact.mean()), "kurtosis": float(np.mean(m4 / m2 ** 2)),
        "pre_negative": float(pre_neg.sum() / n_pre.sum()),
        "pre_hist": {"lo": PRE_LO, "hi": PRE_HI, "counts": (pre_hist.sum(0) / n_pre.sum()).tolist()},
        "by_layer": by_layer,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads to use (0: PyTorch's default)")
    ap.add_argument("--tokens", type=int, default=102400)
    ap.add_argument("--runs", default=str(REPO / "results" / "runs"), help="the folder of finished runs' small copies")
    ap.add_argument("--ckpt", default=str(CKPT), help="the folder of the runs' checkpoints")
    ap.add_argument("--run", action="append", default=[], metavar="ACT=RUN", help="measure another run for an activation")
    ap.add_argument("--out", default=str(DATA))
    args = ap.parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    named = dict(item.split("=", 1) for item in args.run)
    idx = held_out(args.tokens)
    acts, model_shape = [], None
    for act in ACTS:
        run = named.get(act, default_run(act))
        rec: dict = {"act": act, "run": run, "finished": False}
        summary_file = next((p for p in (Path(args.runs) / run / "summary.json", Path(args.ckpt) / run / "summary.json") if p.exists()), None)
        summary = json.loads(summary_file.read_text()) if summary_file else {}
        if summary.get("status") == "completed":
            config = json.loads((summary_file.parent / "config.json").read_text())
            cfg = config["model_config"]
            if cfg["act"] != act:
                raise SystemExit(f"{run} was trained with act={cfg['act']}, not {act}")
            model_shape = model_shape or {k: cfg[k] for k in ("n_layer", "d_model", "n_head", "block_size", "vocab_size")}
            rec.update(finished=True, val_loss=summary["val_loss"], val_bpb=summary["val_bpb"], best_val_loss=summary["best_val_loss"],
                       train_tokens=summary["tokens"], seconds=summary["seconds"], tokens_per_second=summary.get("tokens_per_second"),
                       params=summary["params"]["total"], measured=False)
            weights = Path(args.ckpt) / run / "model.pt"
            if weights.exists():
                model, model_cfg = GPT.from_checkpoint(weights, device=args.device)
                rec["d_mlp"] = model_cfg.d_mlp
                rec.update(measure(model, idx, args.device))
                del model
        acts.append(rec)
        print(f"  {act:<7} {run:<24} " + (f"loss {rec['val_loss']:.4f}  " + (f"zero {rec['zero']:.3f}  negligible {rec['negligible']:.3f}  dead "
              f"{rec['dead']:.4f} (exactly {rec['dead_exact']:.4f})  kurtosis {rec['kurtosis']:.1f}  inputs below zero {rec['pre_negative']:.3f}"
              if rec.get("measured") else "no model.pt to measure") if rec["finished"] else "not finished"), flush=True)
    result = {"generated_by": "gptpp.export.activation_zoo", "tokens": int(idx.numel()), "text": "FineWeb-EDU, held-out",
              "threshold": THRESHOLD, "model": model_shape, "acts": acts}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    text = json.dumps(round_sig(result), separators=(",", ":"))
    write_atomic(out / "activation-zoo.json", text.encode("utf-8"))
    print(f"{out / 'activation-zoo.json'}: {len(text)} bytes")


if __name__ == "__main__":
    main()
