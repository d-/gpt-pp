"""Models that differ only in their position scheme, measured past the length they were trained on.

    python -m gptpp.export.position_schemes_measured [--ckpt RUN ...] [--length 2048] [--out DIR]

The runs (by default the five position swaps of the ladder: rotary, learned, sinusoidal, ALiBi, none) are the same
model trained on the same tokens; only `pos` differs. For each finished run this writes what the site's figure of
length generalization draws:

    loss by position     The held-out stream is cut into windows of `length` tokens (2,048: four times the 512 the
                         models were trained on) and each window is scored in one pass. A causal model's loss at
                         position t depends on tokens 0..t only, so one pass gives the loss at every context length
                         up to `length`. Mean cross-entropy in nats per bucket of positions, with its standard error
                         over windows. A learned position table has no vector past its last row: that model is
                         scored on the first 512 tokens of the same windows and its curve ends there.
    attention by         For every head of every layer, the share of a query's attention that falls in each bucket
    distance             of relative distance (0, 1, 2, 3, then two buckets an octave), averaged over queries in
                         the second half of the training length (positions 256 to 511: what the head learned) and,
                         for a model that accepts it, over the last quarter of the long context (1,536 to 2,047:
                         what the head does past the training length). Dividing a share by the number of keys in
                         its bucket gives the mean weight on one key at that distance: the head's positional
                         kernel, as measured.
    final loss           Each run's own final held-out loss, copied from its summary.json.

The JSON also says whether the runs differ in the position scheme alone (`same_recipe`: every other field of the
model and training configurations is equal), which is what the figure's caption claims of the five swaps. The
sinusoidal run is `swap-pos-sinusoidal-after-norm` (`pos_after_norm`: the fixed sinusoids added after the embedding's
normalization); the first sinusoidal run, which added them before it and so measured a signal that fades in training,
is not measured: its final loss alone is copied under `earlier`, with what it was.

A run that is not finished (no model.pt or no summary.json) is skipped and listed under "missing", so the exporter
can be run while the swaps are still training and run again when they are done. It writes
position-schemes-measured.json, .bin and .entries.json into the site's data folder; `python -m
gptpp.export.manifest` then merges the entries into the manifest.

GPU: one card. With the defaults (1,536 windows, one at a time) a scheme takes about a minute and a half on a
card it shares with a training run, and the process holds about 1 GB: the logits of one 2,048-token window over a
vocabulary of 32,768 are a quarter of a gigabyte, which is why windows are not batched. `--memory-gb` caps what the
process may allocate, so that on a shared card it fails rather than crowd out a training run.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..data import TokenData
from ..model import GPT
from .common import CKPT, DATA, Blob, write_json

NAME = "position-schemes-measured"
DEFAULT_RUNS = ["swap-base", "swap-pos-learned", "swap-pos-sinusoidal-after-norm", "swap-pos-alibi", "swap-pos-none"]
# A run that is not drawn but named once: the first sinusoidal run added the fixed sinusoids before the embedding's
# normalization, where they fade as training lengthens the token rows, so it measured a fading signal. Only its final
# loss is copied, with what it was.
EARLIER = {"sinusoidal": ("swap-pos-sinusoidal", "the sinusoids added before the embedding's normalization, where they fade as the token vectors grow")}
POSITION_FIELDS = ("pos", "pos_after_norm")   # what a position scheme is allowed to change in the model's configuration
SCHEME = {"rope": "rotary", "learned": "learned", "sinusoidal": "sinusoidal", "alibi": "alibi", "none": "none"}
ORDER = ["rotary", "learned", "sinusoidal", "alibi", "none"]


def position_buckets(length: int, fine: int = 32) -> np.ndarray:
    """Edges of the position buckets: 0, 1, 2, 4, ..., `fine`, then every `fine` positions up to `length`. The
    first tokens of a window have almost no context and their loss falls fast, so they get buckets of their own."""
    edges = [0, 1]
    while edges[-1] < fine:
        edges.append(edges[-1] * 2)
    edges += list(range(2 * fine, length + 1, fine))
    return np.array(edges, dtype=np.int64)


def distance_buckets(length: int) -> np.ndarray:
    """Edges of the buckets of relative distance: 0, 1, 2, 3, 4, then two buckets an octave (6, 8, 12, 16, ...)."""
    edges, power = [0, 1, 2, 3], 4
    while power < length:
        edges += [power, power * 3 // 2]
        power *= 2
    edges = [e for e in edges if e < length] + [length]
    return np.array(edges, dtype=np.int64)


def bucket_of(edges: np.ndarray, n: int) -> np.ndarray:
    """For 0..n-1, the index of the bucket [edges[b], edges[b + 1]) that holds it."""
    return np.searchsorted(edges, np.arange(n), side="right") - 1


def windows(tokens: np.ndarray, length: int, count: int) -> np.ndarray:
    """The first `count` consecutive windows of length + 1 tokens (inputs and the targets after them), (count, length + 1)."""
    available = (len(tokens) - 1) // length
    count = available if count <= 0 else min(count, available)
    starts = np.arange(count) * length
    return np.asarray(tokens[starts[:, None] + np.arange(length + 1)[None, :]]).astype(np.int64)


@torch.no_grad()
def loss_by_position(model: GPT, blocks: np.ndarray, edges: np.ndarray, batch: int, device: str) -> dict:
    """Mean cross-entropy per position bucket over the windows, and its standard error over windows."""
    length = blocks.shape[1] - 1
    which = torch.from_numpy(bucket_of(edges, length)).to(device)
    n_buckets = len(edges) - 1
    width = torch.zeros(n_buckets, dtype=torch.float64, device=device).index_add_(
        0, which, torch.ones(length, dtype=torch.float64, device=device))
    total = torch.zeros(n_buckets, dtype=torch.float64, device=device)
    squares = torch.zeros_like(total)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda"))
    for i in range(0, len(blocks), batch):
        block = torch.from_numpy(blocks[i:i + batch]).to(device)
        x, y = block[:, :-1], block[:, 1:]
        with autocast:
            logits = model(x)                                             # float32, (B, T, V)
        nll = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1), reduction="none").view(x.shape).double()
        del logits
        per_window = torch.zeros(len(x), n_buckets, dtype=torch.float64, device=device).index_add_(1, which, nll) / width
        total += per_window.sum(0)
        squares += (per_window ** 2).sum(0)
    n = len(blocks)
    mean = total / n
    se = ((squares / n - mean ** 2).clamp_min(0) / max(1, n - 1)).sqrt()
    return {"loss": mean.cpu().numpy(), "se": se.cpu().numpy(), "windows": n}


@torch.no_grad()
def attention_by_distance(model: GPT, blocks: np.ndarray, edges: np.ndarray, queries: dict[str, tuple[int, int]],
                          batch: int, device: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """For each named range of query positions: the mean share of attention per distance bucket,
    (n_layer, n_head, n_buckets), and the mean number of keys a query has in each bucket, (n_buckets,)."""
    length = blocks.shape[1] - 1
    cfg, n_buckets = model.cfg, len(edges) - 1
    position = torch.arange(length, device=device)
    by_distance = torch.from_numpy(bucket_of(edges, length)).to(device)
    index, keys, share = {}, {}, {}
    for name, (first, last) in queries.items():
        distance = position[first:last, None] - position[None, :]             # (queries, T); negative: a key after the query
        index[name] = torch.where(distance >= 0, by_distance[distance.clamp_min(0)], n_buckets)   # masked keys go to a spare bucket
        counts = torch.zeros(last - first, n_buckets + 1, device=device).scatter_add_(1, index[name], torch.ones_like(distance, dtype=torch.float32))
        keys[name] = counts[:, :n_buckets].mean(0).cpu().numpy()
        share[name] = torch.zeros(cfg.n_layer, cfg.n_head, n_buckets, dtype=torch.float64, device=device)

    def collect(layer: int):
        def hook(pattern: torch.Tensor) -> None:                               # (B, H, T, T), each row sums to one
            for name, (first, last) in queries.items():
                rows = pattern[:, :, first:last, :].float()
                target = index[name].expand(rows.shape)
                mass = torch.zeros(*rows.shape[:3], n_buckets + 1, device=device).scatter_add_(3, target, rows)
                share[name][layer] += mass[..., :n_buckets].double().sum(dim=(0, 2))
        return hook

    hooks = {f"attn.{layer}.pattern": collect(layer) for layer in range(cfg.n_layer)}
    for i in range(0, len(blocks), batch):
        model.run_with_hooks(torch.from_numpy(blocks[i:i + batch, :-1]).to(device), hooks)
    out = {}
    for name, (first, last) in queries.items():
        out[name] = (share[name] / (len(blocks) * (last - first))).cpu().numpy()
    return out, keys


def same_recipe(recipes: list[dict]) -> bool:
    """Do the runs agree on everything but the position scheme? Models: every field. Training: every field that all
    the runs record (a field added to the trainer since the first run cannot be compared)."""
    shared = set.intersection(*(set(r["train"]) for r in recipes))
    first = recipes[0]
    return all(r["model"] == first["model"] and all(r["train"][k] == first["train"][k] for k in shared) for r in recipes)


def resolve(run: str) -> Path:
    path = Path(run)
    return path if path.is_dir() else CKPT / run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", nargs="*", default=DEFAULT_RUNS, help="run directories, or run names under the checkpoint folder")
    ap.add_argument("--data", default="fineweb-edu/bpe32k")
    ap.add_argument("--length", type=int, default=2048, help="the longest context scored")
    ap.add_argument("--windows", type=int, default=1536, help="held-out windows scored for the loss (0: all that fit)")
    ap.add_argument("--attention-windows", type=int, default=64, help="held-out windows whose attention patterns are averaged")
    ap.add_argument("--batch", type=int, default=1, help="windows of the longest context scored at once")
    ap.add_argument("--memory-gb", type=float, default=2.0, help="cap on this process's memory on the card (0: none)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(DATA))
    args = ap.parse_args()
    device = args.device
    if device.startswith("cuda") and args.memory_gb > 0:
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.memory_gb * 2 ** 30 / total))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    data = TokenData(args.data)
    tokens = data.val_tokens()
    blocks = windows(tokens, args.length, args.windows)
    p_edges, d_edges = position_buckets(args.length), distance_buckets(args.length)

    schemes, missing, train_length, shape = {}, [], None, None
    attention, recipes = {}, []
    for run in args.ckpt:
        path = resolve(run)
        if not (path / "model.pt").exists() or not (path / "summary.json").exists():
            print(f"skipped {path.name}: not finished (no model.pt or no summary.json)")
            missing.append(path.name)
            continue
        t0 = time.time()
        model, cfg = GPT.from_checkpoint(path, device=device)
        summary = json.loads((path / "summary.json").read_text())
        config = json.loads((path / "config.json").read_text())
        key = SCHEME[cfg.pos]
        # What the run is apart from its position scheme: equal for all runs when they differ in that alone. The
        # model's side is read from the loaded configuration, so that a field added to ModelConfig since a run was
        # made counts at its default.
        recipes.append({"model": {k: v for k, v in cfg.to_dict().items() if k not in POSITION_FIELDS}, "train": config.get("train_config", {})})
        if key in schemes:
            raise SystemExit(f"{path.name} and {schemes[key]['run']} both use pos={cfg.pos}: give one run a scheme")
        if train_length is None:
            train_length, shape = cfg.block_size, (cfg.n_layer, cfg.n_head, cfg.d_model)
        elif (cfg.block_size, (cfg.n_layer, cfg.n_head, cfg.d_model)) != (train_length, shape):
            raise SystemExit(f"{path.name} is not the same size as the other runs")
        longest = cfg.block_size if cfg.pos == "learned" else args.length      # a learned table ends at block_size
        scored = blocks[:, :longest + 1]
        edges = p_edges[p_edges <= longest]
        res = loss_by_position(model, scored, edges, args.batch * (args.length // longest), device)
        queries = {"inside": (train_length // 2, train_length)}
        if longest >= args.length:
            queries["outside"] = (args.length - train_length, args.length)
        n_att = min(args.attention_windows, len(blocks))
        att, keys = attention_by_distance(model, scored[:n_att], d_edges[d_edges <= longest], queries, max(1, args.batch // 2), device)
        for name, a in att.items():
            sums = a.sum(-1)
            assert np.allclose(sums, 1.0, atol=2e-3), f"{path.name}: attention shares of {name} queries sum to {sums.min():.4f}..{sums.max():.4f}"
        attention[key] = (att, keys)
        widths = np.diff(edges)
        inside = edges[1:] <= train_length
        at_training_length = float((res["loss"][inside] * widths[inside]).sum() / widths[inside].sum())
        schemes[key] = {
            "run": path.name, "pos": cfg.pos, "longest": int(longest),
            "val_loss": summary["val_loss"], "val_bpb": summary.get("val_bpb"),
            "tokens": summary.get("tokens"), "params": summary.get("params", {}).get("total"),
            "seed": config.get("train_config", {}).get("seed"), "preset": summary.get("preset"),
            "after_norm": bool(getattr(cfg, "pos_after_norm", False)),   # sinusoids added after the embedding's normalization, at unit amplitude
            "loss": res["loss"], "se": res["se"], "windows": res["windows"],
        }
        peak = torch.cuda.max_memory_allocated() / 2 ** 30 if device.startswith("cuda") else 0.0
        print(f"{path.name}: pos={cfg.pos}, scored to {longest}; first {train_length} positions {at_training_length:.4f} nats "
              f"(its own evaluation: {summary['val_loss']:.4f}); last bucket {res['loss'][-1]:.4f}; "
              f"{time.time() - t0:.0f} s, peak {peak:.2f} GB")
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    if not schemes:
        raise SystemExit("no finished run among " + ", ".join(args.ckpt))

    # The heads' shares of attention per distance bucket, float16: [scheme, queries (inside, outside), layer, head, bucket].
    order = [k for k in ORDER if k in schemes]
    n_layer, n_head, _ = shape
    n_buckets = len(d_edges) - 1
    packed = np.zeros((len(order), 2, n_layer, n_head, n_buckets), dtype=np.float16)
    keys_out = {"inside": None, "outside": None}
    for s, key in enumerate(order):
        att, keys = attention[key]
        for q, name in enumerate(("inside", "outside")):
            if name in att:
                packed[s, q, :, :, :att[name].shape[-1]] = att[name]
                if keys_out[name] is None or len(keys[name]) > len(keys_out[name]):
                    keys_out[name] = keys[name]
    earlier = {}
    for key, (run, what) in EARLIER.items():
        f = resolve(run) / "summary.json"
        if key in schemes and f.exists() and schemes[key]["run"] != run:
            sm = json.loads(f.read_text())
            earlier[key] = {"run": run, "what": what, "val_loss": sm["val_loss"], "val_bpb": sm.get("val_bpb"), "tokens": sm.get("tokens")}
    blob = Blob(NAME, args.out)
    blob.put(f"{NAME}.attention", packed,
             "share of a query's attention per bucket of relative distance: [scheme, queries (inside the training length, "
             "past it), layer, head, distance bucket]; zeros where a scheme cannot run")
    size = blob.save()

    meta = data.meta
    out = {
        "name": NAME,
        "train_length": int(train_length), "length": int(args.length),
        "model": {"layers": n_layer, "heads": n_head, "width": shape[2]},
        "data": {"dataset": meta.get("dataset"), "tokenizer": meta.get("tokenizer"), "windows": int(len(blocks)),
                 "tokens": int(len(blocks) * args.length), "bytes_per_token": meta.get("bytes_per_token_val")},
        "positions": p_edges, "distances": d_edges,
        "attention": {"windows": int(min(args.attention_windows, len(blocks))),
                      "queries": {"inside": [train_length // 2, train_length], "outside": [args.length - train_length, args.length]},
                      "keys": {k: (v if v is not None else []) for k, v in keys_out.items()}},
        "order": order, "schemes": schemes, "missing": missing,
        "same_recipe": same_recipe(recipes),     # true when the runs differ in the position scheme and nothing else
        "earlier": earlier,
    }
    n_json = write_json(Path(args.out) / f"{NAME}.json", out)
    print(f"wrote {NAME}.json ({n_json / 1e3:.1f} kB) and {NAME}.bin ({size / 1e3:.1f} kB) to {args.out}: "
          f"{', '.join(order)}" + (f"; missing {', '.join(missing)}" if missing else ""))
    print("now run: python -m gptpp.export.manifest")


if __name__ == "__main__":
    main()
