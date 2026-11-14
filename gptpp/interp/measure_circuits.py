"""The numbers the text of the circuits part quotes about the attention-only models, measured and written to
results/interp/circuits.json.

    python -m gptpp.interp.measure_circuits [--one circ-attn1] [--two circ-attn2-repeats] [--two-stories circ-attn2]
                                            [--device cpu]

Each quoted number is an entry keyed by the words of its placeholder in the post, {"value", "model", "note"}; the
note says how it was computed. The models:

    --one           the one-layer model (trained on the stories)
    --two           the two-layer model of the induction figures (trained on the stories with one window in ten
                    of repeated random tokens: gptpp/interp/repeated_token_windows.py)
    --two-stories   the two-layer model trained on the stories alone, measured the same way: its numbers are
                    under "_stories_alone", with the same keys

The numbers of part 8's example of the phase change are read from the export of its figure
(induction-phase-change.json, written by gptpp/export/induction_phase_change.py), so that text and figure quote one
measurement. "_tables" holds every per-head score and both composition tables of each model, with layers and heads
counted from zero; the notes name heads as a reader sees them, counted from one.
Definitions are those of gptpp/interp/circuits.py, which are the post's: positivity (Definition 9.2.9),
composition less 1 / sqrt(d) (Definition 9.3.3 and Proposition 9.3.4; the rotary turn taken at offset 0), the
three head scores (Definition 9.3.8, with 25 tokens repeated 4 times).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from .. import repo_dir
from ..export.circuit_runs import ONE_STORIES, TWO, TWO_STORIES, Run, held_out, pool_for
from ..export.common import SITE
from . import circuits as C

POSITIVITY = 0.8  # a head is counted as copying above this (Example 9.2.10 of the post)
INDUCTION = 0.5  # a second-layer head is counted as an induction head above this prefix-matching score
SCORE = {"length": 25, "repeats": 4, "sequences": 64, "seed": 0}
ROW = {"length": 25, "repeats": 2, "sequences": 256, "seed": 1}  # "a row of random tokens given twice"

K1 = "held-out bits per byte of the one-layer attention-only model with every head's output set to zero (direct path only)"
K2 = "largest absolute difference between the logits of the one-layer attention-only model and the sum of its nine paths, over held-out text"
K3 = "number of the eight heads of the trained one-layer attention-only model with OV positivity above 0.8"
K4 = "number of the eight heads of the one-layer attention-only model with OV positivity above 0.8 at initialization"
K5 = "mean loss in nats per token of the two-layer attention-only model on the first copy of a repeated random sequence"
K6 = "mean loss in nats per token of the two-layer attention-only model on the second copy of a repeated random sequence"
K7 = "largest previous-token score among the first-layer heads of the two-layer attention-only model"
K8 = "largest K-composition score less 0.0625 over the 64 pairs of heads of the two-layer attention-only model"
K9 = "prefix-matching score of the second-layer head with the largest K-composition score"
K10 = "copying score of the same second-layer head"
K11 = "nats gained by the direct path over a uniform guess in the two-layer attention-only model"
K12 = "nats gained by the single-head terms over the direct path in the two-layer attention-only model"
K13 = "nats gained by the virtual-head terms over the first two orders in the two-layer attention-only model"
K14 = ("proportion of the induction heads' effect on the loss explained by the value, query and previous-token-key "
       "paths together in the two-layer attention-only model")


def name_of(layer: int, head: int) -> str:
    """A head as a reader sees it: layers and heads are counted from one in every note (from zero in the tables)."""
    return f"layer {layer + 1}, head {head + 1}"


def entry(value, model: str, note: str) -> dict:
    return {"value": value, "model": model, "note": note}


@torch.no_grad()
def mean_loss(model, windows: torch.Tensor, hooks=None, batch: int = 128) -> float:
    device = model.embed.weight.device
    total = sum(C.loss_by_position(model, windows[i:i + batch].to(device), hooks).double().sum().item()
                for i in range(0, len(windows), batch))
    return total / (windows.shape[0] * (windows.shape[1] - 1))


def positivities(model) -> list[list[float]]:
    return [[C.copying_statistic(C.ov_eigenvalues(model, layer, head)) for head in range(model.cfg.n_head)]
            for layer in range(model.cfg.n_layer)]


@torch.no_grad()
def one_layer(run: Run, windows: torch.Tensor) -> tuple[dict, dict]:
    model = run.model()
    device = model.embed.weight.device
    heads = [(0, h) for h in range(run.cfg.n_head)]
    bytes_per_token = run.data.meta["bytes_per_token_val"]
    direct = mean_loss(model, windows, C.ablation_hooks(heads, "zero"))
    whole = mean_loss(model, windows)
    worst = scale = 0.0
    for i in range(0, 256, 16):
        idx = windows[i:i + 16].to(device)
        logits = model(idx).double()
        worst = max(worst, (C.path_expansion(model, idx).logits().sum(dim=0) - logits).abs().max().item())
        scale = max(scale, logits.abs().max().item())
    trained = positivities(model)[0]
    initial = positivities(run.model(run.checkpoints()[0][1]))[0]
    to_bpb = lambda nats: nats / math.log(2) / bytes_per_token  # noqa: E731
    out = {
        K1: entry(to_bpb(direct), run.name, f"all 8 heads' outputs (attn.0.z) set to zero, mean loss {direct:.4f} nats on the first "
                  f"{windows.numel():,} held-out tokens in windows of 256, divided by ln 2 and by {bytes_per_token:.4f} bytes a token; "
                  f"the whole model on the same windows: {whole:.4f} nats, {to_bpb(whole):.4f} bits per byte"),
        K2: entry(worst, run.name, f"path_expansion on the first 256 held-out windows of 256 tokens (the model computes in float32, the "
                  f"paths in float64); the largest logit in absolute value there is {scale:.1f}"),
        K3: entry(sum(p > POSITIVITY for p in trained), run.name, "copying_statistic(ov_eigenvalues) of the final model, by head: "
                  + ", ".join(f"{p:.3f}" for p in trained)),
        K4: entry(sum(p > POSITIVITY for p in initial), run.name, "the same at the checkpoint before training (GPT-2 initialization, "
                  "every matrix random): " + ", ".join(f"{p:.3f}" for p in initial)),
    }
    table = {"positivity": trained, "positivityInitial": initial, "loss": whole, "lossDirectPathOnly": direct,
             "bitsPerByte": to_bpb(whole), "bitsPerByteDirectPathOnly": to_bpb(direct)}
    return out, table


@torch.no_grad()
def two_layer(run: Run, windows: torch.Tensor, pool: torch.Tensor) -> tuple[dict, dict]:
    model = run.model()
    device, cfg, bos = model.embed.weight.device, run.cfg, int(run.data.meta["bos_id"])
    name = run.name
    scores = C.head_scores(model, windows[:256], pool, bos, SCORE["length"], SCORE["repeats"], SCORE["sequences"], SCORE["seed"])
    rows = C.repeated_random_tokens(ROW["sequences"], ROW["length"], ROW["repeats"], pool, bos, ROW["seed"]).to(device)
    by_position = C.loss_by_position(model, rows)
    L = ROW["length"]
    halves = C.repeated_loss(by_position, L, ROW["repeats"])
    first_all, second_all = by_position[:, :L].double().mean().item(), by_position[:, L:2 * L].double().mean().item()
    second_first_token = by_position[:, L].double().mean().item()
    on_rows = C.attention_patterns(model, rows)
    previous_on_rows = C.previous_token_score(on_rows[0]).tolist()

    base = C.baseline(cfg.d_model)
    table = {k: (v - base) for k, v in C.composition_table(model, 0, 1).items()}
    permuted_base = C.composition_baseline(cfg.d_model, cfg.d_head, draws=100, cyclic=True)
    permuted = {k: (v - permuted_base) for k, v in C.composition_table(model, 0, 1, cyclic=True).items()}
    at = int(table["k"].argmax())
    g, h = at // cfg.n_head, at % cfg.n_head
    turned = {offset: C.composition(model, (0, g), (1, h), offset=offset)["k"] - base for offset in (1, 4, 16, 64)}

    losses = [0.0, 0.0, 0.0]
    for i in range(0, 512, 32):
        for n, value in enumerate(C.loss_by_order(model, windows[i:i + 32].to(device))):
            losses[n] += value / 16
    uniform = math.log(cfg.vocab_size)

    induction = [head for head in range(cfg.n_head) if scores["prefix_matching"][1][head] > INDUCTION] or [h]
    others = C.repeated_random_tokens(ROW["sequences"], L, ROW["repeats"], pool, bos, ROW["seed"] + 1000).to(device)
    explained = C.induction_paths_explained(model, rows, others, induction, g, L, ROW["repeats"])
    k_positivity = [C.copying_statistic(C.k_composition_eigenvalues(model, (0, g), (1, head))) for head in range(cfg.n_head)]
    prev_text = scores["previous_token"][0]
    top_prev = max(range(cfg.n_head), key=lambda a: prev_text[a])

    out = {
        K5: entry(halves["first"], name, f"{ROW['sequences']} rows of {L} random tokens given twice after the start token; mean "
                  f"over tokens 2 to {L} of the first copy (the first token included: {first_all:.3f}). A uniform guess among the "
                  f"{len(pool):,} tokens the rows are drawn from costs {math.log(len(pool)):.2f}"),
        K6: entry(halves["repeated"], name, f"the same rows; mean over tokens 2 to {L} of the second copy, the ones the first copy "
                  f"predicts. The first token of the second copy follows the row's last token for the first time and costs "
                  f"{second_first_token:.2f}; with it the mean over the whole second copy is {second_all:.3f}"),
        K7: entry(prev_text[top_prev], name, f"Definition 9.3.8 (i) on 256 held-out windows of stories: {name_of(0, top_prev)}. By head: "
                  + ", ".join(f"{p:.3f}" for p in prev_text) + ". On the repeated random rows the same score is, by head: "
                  + ", ".join(f"{p:.3f}" for p in previous_on_rows)
                  + f". The layer 1 head of the largest K-composition pair is head {g + 1}"),
        K8: entry(table["k"][g, h].item(), name, f"Definition 9.3.3 with W_QK = W_Q^T W_K (the rotary turn at offset 0), less "
                  f"1/sqrt({cfg.d_model}) = {base:.4f}: {name_of(0, g)} with {name_of(1, h)}. With the key turned "
                  "by an offset of 1, 4, 16, 64 positions the same pair scores "
                  + ", ".join(f"{turned[o]:.3f}" for o in (1, 4, 16, 64))),
        K9: entry(scores["prefix_matching"][1][h], name, f"Definition 9.3.8 (ii), {name_of(1, h)}, {SCORE['sequences']} sequences of "
                  f"{SCORE['length']} random tokens repeated {SCORE['repeats']} times. Heads 1 to 8 of layer 2 by score: "
                  + ", ".join(f"{p:.3f}" for p in scores["prefix_matching"][1])),
        K10: entry(scores["copying_score"][1][h], name, f"Definition 9.3.8 (iii), {name_of(1, h)}: the mean over 40 samples S of 25 random "
                   "tokens; a head that raises tokens at random scores 1/25 = 0.04. Heads 1 to 8 of layer 2 by score: "
                   + ", ".join(f"{p:.3f}" for p in scores["copying_score"][1])),
        K11: entry(uniform - losses[0], name, f"loss_by_order on the first 512 held-out windows of 256 tokens, every pattern frozen: "
                   f"ln {cfg.vocab_size} = {uniform:.3f} less the loss of the direct path alone, {losses[0]:.3f} nats"),
        K12: entry(losses[0] - losses[1], name, f"the direct path alone ({losses[0]:.3f}) less the direct path and the "
                   f"{2 * cfg.n_head} single-head terms ({losses[1]:.3f})"),
        K13: entry(losses[1] - losses[2], name, f"orders 0 and 1 ({losses[1]:.3f}) less all three orders, the model itself "
                   f"({losses[2]:.3f})"),
        K14: entry(explained["all"], name, "induction_paths_explained: path patching on the same rows, the other input another "
                   f"set of rows; induction heads (prefix matching above {INDUCTION}): "
                   + "layer 2, heads " + ", ".join(str(a + 1) for a in induction) + f"; previous-token head: {name_of(0, g)}. Loss on the repeated tokens: clean "
                   f"{explained['loss.clean']:.3f}, with the heads fed the other input along every path {explained['loss.none']:.3f}, "
                   f"with only the three paths real {explained['loss.all']:.3f}. Alone: value {explained['value']:.3f}, query "
                   f"{explained['query']:.3f}, key {explained['key']:.3f}"),
    }
    tables = {
        "scores": scores, "previousTokenOnRows": previous_on_rows, "positivity": positivities(model),
        "composition": {k: v.tolist() for k, v in table.items()}, "compositionBaseline": base,
        "compositionPermuted": {k: v.tolist() for k, v in permuted.items()}, "compositionPermutedBaseline": permuted_base,
        "largestK": {"first": g, "second": h}, "kCompositionPositivity": k_positivity, "inductionHeads": induction,
        "lossByOrder": losses, "pathsExplained": explained, "rowLoss": {"first": halves["first"], "repeated": halves["repeated"]},
    }
    return out, tables


P1 = "training tokens at which the two-layer model's loss on the repeated half has made a tenth of its total fall"
P2 = "training tokens at which it has made nine tenths of its total fall"
P3 = "loss on the repeated half at the start of the window, nats"
P4 = "loss on the repeated half at the end of the window, nats"
P5 = "highest induction score at the start of the window"
P6 = "highest induction score at the end of the window"
P7 = "final loss on the repeated half of the one-layer attention-only model, nats"


def phase_change(path: Path) -> dict:
    """The numbers of part 8's example of the phase change, read from the export of its figure
    (gptpp/export/induction_phase_change.py), so that the text and the figure quote one measurement."""
    export = json.loads(path.read_text())
    runs = {run["key"]: run for run in export["runs"]}
    two, one, stories = runs["two"], runs["one"], runs["stories"]
    window, seen, loss = two["change"]["repeated"], two["seen"], two["repeated"]
    k0, k1 = seen.index(window["start"]), seen.index(window["end"])
    top = [max(max(layer) for layer in row) for row in two["pm"]]
    half = next(k for k in range(1, len(seen)) if loss[k] <= (window["from"] + window["to"]) / 2)
    plateau = [f"{loss[k]:.2f}" for k in range(len(seen)) if 40e6 < seen[k] < 95e6]
    row = export["score"]
    how = (f"{row['sequences']} rows of {row['length']} random tokens given twice; the loss is the mean over tokens 2 to "
           f"{row['length']} of the second copy; checkpoints stand {(seen[k0] - seen[k0 - 1]) / 1e6:.1f}M tokens apart there")
    return {
        P1: entry(window["start"], two["run"], f"the first checkpoint at which the loss is below its value before training "
                  f"({window['from']:.3f}, a uniform guess over the vocabulary) by a tenth of the way to its final value "
                  f"({window['to']:.3f}); {how}. The loss first rises, to {max(loss):.2f}, and stands at {loss[k0 - 3]:.2f} just "
                  f"before the fall"),
        P2: entry(window["end"], two["run"], f"the first checkpoint at which it has made nine tenths of that way. The fall is a cliff "
                  f"and a tail: half of it is made by {seen[half]:,} tokens ({loss[half]:.2f} nats), then the loss stays between "
                  f"{min(plateau)} and {max(plateau)} from 40M to 95M tokens and comes down as the learning rate decays (the rate "
                  f"is flat for the first 35 % of the run)"),
        P3: entry(loss[k0], two["run"], f"at {seen[k0]:,} tokens"),
        P4: entry(loss[k1], two["run"], f"at {seen[k1]:,} tokens; at the end of training {loss[-1]:.3f}"),
        P5: entry(top[k0], two["run"], f"Definition 8.2.2 on the same rows, the largest over the 16 heads, at {seen[k0]:,} tokens; "
                  f"before the window every head is at {top[0]:.3f}, what a uniform pattern gives"),
        P6: entry(top[k1], two["run"], f"at {seen[k1]:,} tokens; at the end of training {top[-1]:.3f} ({name_of(two['best']['layer'], two['best']['head'])}). "
                  f"By the paper's protocol (25 tokens given four times, Definition 9.3.8) the same checkpoints give "
                  f"{max(max(l) for l in two['pm25x4'][k0]):.3f}, {max(max(l) for l in two['pm25x4'][k1]):.3f} and "
                  f"{max(max(l) for l in two['pm25x4'][-1]):.3f}"),
        P7: entry(one["repeated"][-1], one["run"], f"the one-layer model trained on the same text: its loss on the repeated half ends "
                  f"where a uniform guess over the vocabulary stands, and its best head's induction score never passes "
                  f"{math.ceil(1000 * max(max(max(l) for l in r) for r in one['pm'])) / 1000:.3f} (the largest value, rounded "
                  f"up). The two-layer model trained on the stories alone "
                  f"({stories['run']}) ends at {stories['repeated'][-1]:.2f} nats and a best score of "
                  f"{max(max(l) for l in stories['pm'][-1]):.3f}"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--one", default=ONE_STORIES)
    parser.add_argument("--two", default=TWO)
    parser.add_argument("--two-stories", default=TWO_STORIES)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=str(repo_dir() / "results" / "interp" / "circuits.json"))
    args = parser.parse_args()
    torch.set_num_threads(min(16, torch.get_num_threads()))
    one, two, stories = Run(args.one, args.device), Run(args.two, args.device), Run(args.two_stories, args.device)
    windows = held_out(4096)
    pool = pool_for([two])
    out: dict = {
        "_about": "Numbers quoted in the circuits part of the series about the attention-only models, keyed by the words "
                  "of their placeholders; written by `python -m gptpp.interp.measure_circuits`. One training run a model.",
        "_models": {"one-layer": one.describe(), "two-layer": two.describe(), "two-layer, stories alone": stories.describe()},
    }
    values, one_table = one_layer(one, windows)
    out.update(values)
    values, two_table = two_layer(two, windows, pool)
    out.update(values)
    alone, alone_table = two_layer(stories, windows, pool)
    out["_stories_alone"] = alone
    export = SITE / "assets" / "data" / "gpt" / "induction-phase-change.json"
    if export.exists():
        out.update(phase_change(export))
    out["_tables"] = {one.name: one_table, two.name: two_table, stories.name: alone_table}
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n")
    for key, item in out.items():
        if not key.startswith("_"):
            print(f"{item['value']!s:>22}  {key}")
    print("stories alone:")
    for key, item in alone.items():
        print(f"{item['value']!s:>22}  {key}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
