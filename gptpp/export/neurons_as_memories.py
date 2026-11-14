"""Export for fig-05-06 `gpt-neurons-as-memories`: neurons of the 12-layer model as key-value pairs.

    python -m gptpp.export.neurons_as_memories --ckpt $GPTPP_WORK/ckpt/gptpp-base            (the run directory: model.pt)
    python -m gptpp.export.neurons_as_memories --ckpt $GPTPP_WORK/ckpt/gptpp-base/ckpt_6000.pt --tokens 5e5

For every layer, a random sample of `--sample` neurons of its MLP (the same sample for every checkpoint: it
depends on `--seed` only). With `gptpp.interp.memories`:

    the key     one pass over the first `--tokens` tokens of the training set, read as the trainer reads them, in
                windows of the model's context: for each sampled neuron the `--triggers` positions with the largest
                memory coefficient (the hook `mlp.l.post`), each kept as the tokens before it, the trigger token
                and the tokens that follow
    the value   the neuron's output weights through the unembedding, W_U v: the `--top` tokens with the largest
                score (promoted when the neuron fires) and those with the smallest (suppressed)
    measures    per neuron: the share of its top triggers that end in one token (`same_last`: an automatic
                stand-in for the "shallow patterns" that annotators found in the keys of the lower layers);
                whether the value's top token is the token that follows the top trigger (`agrees`: the papers'
                agreement); the share of the top triggers followed by one of the value's top tokens (`follows`)
    by layer    the means of the three over the sampled neurons that fire at all, with standard errors, the same
                two value measures with the values dealt to the wrong neurons (a chance level measured on this
                model), and the level of `same_last` for positions drawn at random

Ten neurons a layer are exported in full, chosen by the measures and not by eye: five that span the range of
`same_last` (the neurons at ranks 0, 1/4, 1/2, 3/4 and 1 of the sample sorted by it) and five that span the range
of `follows`. The figure's default is the exported neuron with the largest product of the two.

Writes `neurons-as-memories.json`:

    model, text {tokens, window, shard}, sample, triggers, top
    vocab     the texts of the tokens mentioned; every token below is an index into it. An e-mail address or a
              telephone number in a snippet is replaced by the mark [...] (`left_out` counts them)
    chance    {same_last, agreement: 1 / vocabulary}
    layers    [{same_last, same_last_se, shallow (share of neurons with same_last >= 1/2), agreement, agreement_se,
              follows, follows_se, agreement_shuffled, follows_shuffled, neurons, alive,
              shown: [{neuron, same_last, last_token, agrees, follows, max, top: [[token, score]], bottom: [[token,
              score]], triggers: [[coefficient, [tokens before], trigger token, [tokens after]]]}]}]
    default   {layer, shown}
"""
from __future__ import annotations

import argparse
import math
import re
import time

import numpy as np
import torch

from ..data import TokenData
from ..interp import memories
from ..tokenizer import Tokenizer
from .common import DATA, write_json
from .interp_common import TOKENIZER, Texts, load_model, pick_device

NAME = "neurons-as-memories"


# What a snippet of web text must not carry onto a page: an e-mail address, a telephone number
PRIVATE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+|\(?\b\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b")
LEFT_OUT = "[…]"


def snippet(texts: Texts, before: np.ndarray, trigger: int, after: np.ndarray) -> tuple[list[int], int, list[int], int]:
    """A trigger with its context as indices into the table of texts. An e-mail address or a telephone number in it
    is replaced by one mark; the trigger token itself is never touched. Returns also the number of things left out."""
    ids = [int(t) for t in before] + [int(trigger)] + [int(t) for t in after]
    pieces = [texts.tokenizer.decode([t]) for t in ids]
    starts = np.cumsum([0] + [len(piece) for piece in pieces])
    out, at, left_out = [texts.index(t) for t in ids], len(before), 0
    for match in PRIVATE.finditer("".join(pieces)):
        hit = [i for i in range(len(ids)) if starts[i] < match.end() and starts[i + 1] > match.start() and i != at]
        for n, i in enumerate(hit):
            out[i] = texts.literal(LEFT_OUT if n == 0 else "")
        left_out += bool(hit)
    return out[:at], out[at], out[at + 1:], left_out


def spanning(order: list[int], taken: set[int], n: int) -> list[int]:
    """n members of `order` (indices sorted by a measure) at evenly spaced ranks from first to last, skipping those
    already taken (the next free rank is used instead)."""
    free = [i for i in order if i not in taken]
    if len(free) <= n:
        return free
    return [free[round(q * (len(free) - 1) / (n - 1))] for q in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True, help="model.pt, ckpt_<step>.pt, or the run directory")
    ap.add_argument("--out", default=str(DATA))
    ap.add_argument("--device", default=None)
    ap.add_argument("--data", default="fineweb-edu/bpe32k")
    ap.add_argument("--tokens", type=float, default=2e6, help="training tokens read")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--sample", type=int, default=200, help="neurons sampled in each layer")
    ap.add_argument("--triggers", type=int, default=25)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--bottom", type=int, default=12)
    ap.add_argument("--shown", type=int, default=10, help="neurons exported in full, a layer")
    ap.add_argument("--before", type=int, default=10, help="tokens kept before a trigger")
    ap.add_argument("--after", type=int, default=2, help="tokens kept after it")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--float32", action="store_true", help="no bfloat16 autocast on the card")
    args = ap.parse_args()
    t0 = time.time()
    device = pick_device(args.device)
    model, meta = load_model(args.ckpt, device)
    tokenizer, texts = Tokenizer.load(TOKENIZER), Texts(Tokenizer.load(TOKENIZER))
    data = TokenData(args.data)
    L, T, bos = meta["layers"], model.cfg.block_size, data.meta["bos_id"]
    n_windows = int(args.tokens) // T
    text = np.asarray(data.shard(0)[: n_windows * T + args.after + 1]).astype(np.int64)

    generator = torch.Generator().manual_seed(args.seed)
    neurons = {l: torch.randperm(meta["mlp"], generator=generator)[: args.sample].sort().values for l in range(L)}

    def windows():
        for i in range(0, n_windows, args.batch):
            j = min(n_windows, i + args.batch)
            yield torch.from_numpy(text[i * T: j * T].reshape(j - i, T))

    found = memories.top_triggers(model, windows(), neurons, k=args.triggers, autocast=not args.float32)
    scan = time.time() - t0

    rng = np.random.default_rng(args.seed)                          # same_last of positions drawn at random
    draws = text[rng.integers(0, n_windows * T, size=(2000, args.triggers))]
    chance = float(memories.same_last_token(torch.from_numpy(draws))[0].mean())

    layers, best, left_out = [], (-1.0, 0, 0), 0
    for l in range(L):
        where = (found[l]["window"] * T + found[l]["pos"]).numpy()             # (n, k) positions in the text
        coef = found[l]["coef"]
        trigger_tokens, next_tokens = torch.from_numpy(text[where]), torch.from_numpy(text[where + 1])
        ends = memories.promoted(memories.value_scores(model, l, neurons[l]), k=args.top)
        top_ids = ends["top_ids"].cpu()
        m = memories.neuron_measures(trigger_tokens, next_tokens, coef, top_ids)
        s = memories.layer_statistics(m)
        shuffled = memories.layer_statistics(memories.neuron_measures(trigger_tokens, next_tokens, coef, top_ids.roll(1, dims=0)))
        alive = m["alive"]
        n = max(1, s["alive"])
        se = lambda x: float(x[alive].float().std() / math.sqrt(n)) if n > 1 else 0.0
        row = {"same_last": s["same_last"], "same_last_se": se(m["same_last"]),
               "shallow": float((m["same_last"][alive] >= 0.5).float().mean()) if s["alive"] else float("nan"),
               "agreement": s["agreement"], "agreement_se": math.sqrt(max(s["agreement"] * (1 - s["agreement"]), 0) / n),
               "follows": s["follows"], "follows_se": se(m["follows"]),
               "agreement_shuffled": shuffled["agreement"], "follows_shuffled": shuffled["follows"],
               "neurons": s["neurons"], "alive": s["alive"], "shown": []}
        ids = [i for i in range(args.sample) if bool(alive[i])]
        by_same = sorted(ids, key=lambda i: (-float(m["same_last"][i]), -float(m["follows"][i]), i))
        chosen = spanning(by_same, set(), args.shown // 2)
        by_follows = sorted(ids, key=lambda i: (-float(m["follows"][i]), -float(m["same_last"][i]), i))
        chosen += spanning(by_follows, set(chosen), args.shown - len(chosen))
        chosen.sort(key=lambda i: (-float(m["same_last"][i]), -float(m["follows"][i]), i))
        for i in chosen:
            triggers = []
            for c, at in zip(coef[i].tolist(), where[i].tolist()):
                start = at - at % T                                  # the model read from the start of its window
                before = text[max(start, at - args.before): at]
                cut = np.flatnonzero(before == bos)
                before = before[cut[-1] + 1:] if len(cut) else before      # nothing of the document before
                after = text[at + 1: at + 1 + args.after]
                stop = np.flatnonzero(after == bos)
                after = after[: stop[0]] if len(stop) else after
                b_, t_, a_, n_ = snippet(texts, before, text[at], after)
                left_out += n_
                triggers.append([round(c, 3), b_, t_, a_])
            pair = lambda ids_, scores: [[texts.index(t), round(float(v), 3)] for t, v in zip(ids_[i].tolist(), scores[i].tolist())]
            shown = {"neuron": int(neurons[l][i]), "same_last": float(m["same_last"][i]), "last_token": texts.index(m["last_token"][i]),
                     "agrees": bool(m["agrees"][i]), "follows": float(m["follows"][i]), "max": float(coef[i, 0]),
                     "top": pair(ends["top_ids"], ends["top_scores"]),
                     "bottom": pair(ends["bottom_ids"][:, : args.bottom], ends["bottom_scores"][:, : args.bottom]),
                     "triggers": triggers}
            score = shown["same_last"] * shown["follows"]
            if score > best[0]:
                best = (score, l, len(row["shown"]))
            row["shown"].append(shown)
        layers.append(row)

    out = {"model": meta, "text": {"tokens": n_windows * T, "window": T, "shard": data.train_files[0].name}, "sample": args.sample,
           "triggers": args.triggers, "top": args.top, "vocab": texts.table,
           "chance": {"same_last": chance, "agreement": 1.0 / meta["vocabulary"]}, "left_out": left_out, "layers": layers,
           "default": {"layer": best[1], "shown": best[2]}}
    size = write_json(f"{args.out}/{NAME}.json", out)
    print(f"{NAME}.json: {size / 1e3:.0f} kB, {n_windows * T:,} tokens scanned in {scan:.0f} s, {time.time() - t0:.0f} s in all, "
          f"on {meta['checkpoint']} (step {meta['step']}); chance same_last {chance:.3f}; {left_out} addresses or numbers left out of the snippets")
    for l, row in enumerate(layers):
        print(f"  layer {l + 1:2d}: same_last {row['same_last']:.3f} ± {row['same_last_se']:.3f}, shallow {row['shallow']:.2f}, agreement "
              f"{row['agreement']:.3f} (shuffled {row['agreement_shuffled']:.3f}), follows {row['follows']:.3f} (shuffled "
              f"{row['follows_shuffled']:.3f}), alive {row['alive']}/{row['neurons']}")
    d = layers[best[1]]["shown"][best[2]]
    print(f"  default: layer {best[1] + 1}, neuron {d['neuron']}: same_last {d['same_last']:.2f}, follows {d['follows']:.2f}")


if __name__ == "__main__":
    main()
