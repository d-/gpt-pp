"""Export for fig-08-02 `gpt-induction-phase-change`: the two-layer attention-only model through training.

    python -m gptpp.export.induction_phase_change [--two circ-attn2-repeats] [--one circ-attn1-repeats]
                                                  [--stories circ-attn2] [--windows 4096] [--device cuda]
                                                  [--only two | one | stories] [--fresh two,one,stories]

Measuring one run's checkpoints takes a few minutes on a card: `--only two` measures that run, keeps the rows in
$GPTPP_WORK/cache/induction-phase-change/ and stops, so that each run can be one short job; a call without
`--only` measures what is not cached and writes the files.

For every checkpoint of each run, measured with gptpp/interp/circuits.py:

    text        the loss on held-out stories, nats per token (the first `--windows` windows of 256 tokens)
    icl         the in-context learning score on those windows: the loss on the 25th token of the context minus the
                loss on the 250th, positive when the later token is the easier one (Olsson et al., 2022, take the
                50th and the 500th of 512)
    repeated    the loss on the tokens of a random row given twice that the first copy predicts, and
    first       on the tokens of the first copy, which nothing predicts (256 rows of 16 random tokens)
    pm          the induction score of every head on those rows, [layer][head]: the attention a token of the
                second copy pays to the token that followed its first occurrence, averaged over the row
    pm25x4      the same score by the paper's protocol, 64 rows of 25 tokens given four times, where a token of
                a later copy has up to three earlier occurrences and the attention to all of them is added
    prev        the previous-token score of every head on 64 windows of the stories

and, for the first run, the attention pattern of every head on one row of 16 random word tokens given twice, at
every checkpoint.

The three runs are the model of the figure (`--two`), the one-layer model trained on the same data, which cannot
form the circuit (`--one`), and the two-layer model trained on the stories alone, which has little use for it
(`--stories`). Each run also carries `change`: when its best head's score and its loss on the repeated tokens make
their move (circuits.rise_interval). `opportunity` says how much the held-out stories repeat themselves in the way
an induction head can use (circuits.repeat_statistics), and what share of those tokens the last checkpoint of the
`--stories` and `--one` runs already predict.

Writes `induction-phase-change.json` and `induction-phase-change.bin` (the patterns, one byte a cell).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from ..interp import circuits as C
from .circuit_runs import ONE, TWO, TWO_STORIES, Run, held_out, is_word, pick_device, pool_for, token_names
from .common import DATA, WORK, Blob, write_json

NAME = "induction-phase-change"
SCORE = {"length": 16, "repeats": 2, "sequences": 256, "seed": 0}  # a random row given twice
PAPER = {"length": 25, "repeats": 4, "sequences": 64, "seed": 0}  # Olsson et al.'s evaluator: 25 tokens, 4 times
SHOWN = {"length": 16, "repeats": 2, "seed": 3}  # the row whose patterns are drawn
ICL = {"early": 25, "late": 250}  # Olsson et al.: the 50th and the 500th token of 512


@torch.no_grad()
def text_losses(model, windows: torch.Tensor, batch: int = 128) -> torch.Tensor:
    """Loss by position on the held-out windows, (B, T - 1), computed in batches on the model's device."""
    device = model.embed.weight.device
    return torch.cat([C.loss_by_position(model, windows[i:i + batch].to(device)).cpu()
                      for i in range(0, len(windows), batch)])


@torch.no_grad()
def measure(model, windows: torch.Tensor, sequences: torch.Tensor, paper: torch.Tensor, shown: torch.Tensor) -> dict:
    """Everything the figure draws of one checkpoint."""
    device = model.embed.weight.device
    losses = text_losses(model, windows)
    on_rows = C.attention_patterns(model, sequences.to(device))
    on_paper = C.attention_patterns(model, paper.to(device))
    on_text = C.attention_patterns(model, windows[:64].to(device))
    repeated = C.repeated_loss(C.loss_by_position(model, sequences.to(device)), SCORE["length"], SCORE["repeats"])
    return {
        "text": losses.double().mean().item(),
        "icl": C.in_context_learning_score(losses, ICL["early"], ICL["late"]),
        "repeated": repeated["repeated"], "first": repeated["first"],
        "pm": [C.prefix_matching_score(p, SCORE["length"], SCORE["repeats"]).tolist() for p in on_rows],
        "pm25x4": [C.prefix_matching_score(p, PAPER["length"], PAPER["repeats"]).tolist() for p in on_paper],
        "prev": [C.previous_token_score(p).tolist() for p in on_text],
        "patterns": torch.cat([p[0].cpu() for p in C.attention_patterns(model, shown.to(device))]),  # (layers * H, T, T)
    }


def export_run(run: Run, key: str, windows: torch.Tensor, sequences: torch.Tensor, paper: torch.Tensor,
               shown: torch.Tensor, fresh: bool) -> tuple[dict, list]:
    """The series of one run. Measuring a run's checkpoints takes minutes, so the rows are kept in the workspace
    ($GPTPP_WORK/cache/induction-phase-change/<run>.pt) and read from there unless `fresh` or the settings changed."""
    start = time.perf_counter()
    cache = WORK / "cache" / NAME / f"{run.name}.pt"
    settings = {"score": SCORE, "paper": PAPER, "shown": SHOWN, "icl": ICL, "windows": len(windows),
                "steps": [s for s, _ in run.checkpoints()], "version": 2}
    rows = None
    if cache.exists() and not fresh:
        saved = torch.load(cache, weights_only=False)
        rows = saved["rows"] if saved["settings"] == settings else None
    if rows is None:
        rows = []
        for step, path in run.checkpoints():
            row = measure(run.model(path), windows, sequences, paper, shown)
            row["step"], row["tokens"] = step, step * run.tokens_per_step
            rows.append(row)
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"settings": settings, "rows": rows}, cache)
    final = rows[-1]
    layer = max(range(run.cfg.n_layer), key=lambda l: max(final["pm"][l]))
    head = max(range(run.cfg.n_head), key=lambda h: final["pm"][layer][h])
    tokens = [row["tokens"] for row in rows]
    out = run.describe()
    out.update(
        key=key, steps=[row["step"] for row in rows], seen=tokens,
        text=[row["text"] for row in rows], icl=[row["icl"] for row in rows],
        repeated=[row["repeated"] for row in rows], first=[row["first"] for row in rows],
        pm=[row["pm"] for row in rows], pm25x4=[row["pm25x4"] for row in rows], prev=[row["prev"] for row in rows],
        best={"layer": layer, "head": head, "score": final["pm"][layer][head]},
        change={"score": C.rise_interval(tokens, [row["pm"][layer][head] for row in rows]),
                "repeated": C.rise_interval(tokens, [row["repeated"] for row in rows])},
    )
    print(f"{run.name}: {len(rows)} checkpoints in {time.perf_counter() - start:.0f} s; best head {layer}.{head} "
          f"score {final['pm'][layer][head]:.3f}; fall of the repeated loss {out['change']['repeated']}")
    return out, [row["patterns"] for row in rows]


@torch.no_grad()
def opportunity(windows: torch.Tensor, bos_id: int, runs: dict[str, Run]) -> dict:
    """How often the stories continue as they did the last time, and how well the finished models named already
    predict those tokens (top-1 accuracy and loss), with the same for all tokens."""
    out = dict(C.repeat_statistics(windows, exclude=[bos_id]), windows=len(windows))
    mask = torch.zeros(windows.shape[0], windows.shape[1] - 1, dtype=torch.bool)
    for b, row in enumerate(windows.tolist()):
        last: dict[int, int] = {}
        for i, token in enumerate(row[:-1]):
            if token in last and token != bos_id and row[last[token] + 1] == row[i + 1]:
                mask[b, i] = True
            last[token] = i
    for key, run in runs.items():
        model = run.model()
        device = model.embed.weight.device
        right, losses = [], []
        for i in range(0, len(windows), 128):
            idx = windows[i:i + 128].to(device)
            logits = model(idx)[:, :-1]
            right.append((logits.argmax(dim=-1) == idx[:, 1:]).cpu())
            losses.append(C.loss_by_position(model, idx).cpu())
        right, losses = torch.cat(right), torch.cat(losses)
        out[key] = {"accuracyOnThose": right[mask].double().mean().item(), "lossOnThose": losses[mask].double().mean().item(),
                    "accuracy": right.double().mean().item(), "loss": losses.double().mean().item()}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--two", default=TWO, help="the two-layer model of the figure")
    parser.add_argument("--one", default=ONE, help="the one-layer model trained on the same data")
    parser.add_argument("--stories", default=TWO_STORIES, help="the two-layer model trained on the stories alone")
    parser.add_argument("--windows", type=int, default=4096, help="held-out windows for the text loss and the score")
    parser.add_argument("--fresh", default="", help="runs to measure again although cached: any of two,one,stories")
    parser.add_argument("--only", default="", help="measure (and cache) these runs and stop: for one short job a run")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=str(DATA))
    args = parser.parse_args()
    device = pick_device(args.device)
    runs = {"two": Run(args.two, device), "one": Run(args.one, device), "stories": Run(args.stories, device)}
    pool, bos = pool_for(list(runs.values())), int(runs["two"].data.meta["bos_id"])
    windows = held_out(args.windows)
    names = token_names()
    sequences = C.repeated_random_tokens(SCORE["sequences"], SCORE["length"], SCORE["repeats"], pool, bos, SCORE["seed"])
    paper = C.repeated_random_tokens(PAPER["sequences"], PAPER["length"], PAPER["repeats"], pool, bos, PAPER["seed"])
    words = torch.tensor([t for t in pool.tolist() if is_word(names[t]) and names[t].startswith(" ") and len(names[t]) >= 4])
    shown = C.repeated_random_tokens(1, SHOWN["length"], SHOWN["repeats"], words, bos, SHOWN["seed"])

    exported, patterns = {}, None
    for key, run in runs.items():
        if args.only and key not in args.only.split(","):
            continue
        exported[key], run_patterns = export_run(run, key, windows, sequences, paper, shown, key in args.fresh.split(","))
        if key == "two":
            patterns = torch.stack(run_patterns)  # (checkpoints, heads of layer 0 then layer 1, T, T)
    if args.only:
        return
    blob = Blob(NAME, args.out)
    blob.put(f"{NAME}.pattern", np.rint(patterns.numpy() * 255).astype(np.uint8),
             "attention pattern of every head of the two-layer model on the shown row at every checkpoint: "
             "(checkpoints, heads of layer 0 then layer 1, destination, source)", lo=0.0, hi=1.0)
    size = blob.save()
    out = {
        "units": "nats per token", "score": SCORE, "paper": PAPER, "icl": dict(ICL, windows=args.windows, context=windows.size(1)),
        "shown": dict(SHOWN, tokens=shown[0].tolist(), names=[names[t] for t in shown[0].tolist()]),
        "pool": {"size": len(pool), "uniformLoss": float(np.log(len(pool)))},
        "runs": [exported[key] for key in ("two", "one", "stories")],
        "opportunity": opportunity(windows[:512], bos, {"stories": runs["stories"], "one": runs["one"]}),
    }
    json_size = write_json(Path(args.out) / f"{NAME}.json", out)
    print(f"wrote {NAME}.json ({json_size:,} bytes) and {NAME}.bin ({size:,} bytes) on {device}")


if __name__ == "__main__":
    main()
