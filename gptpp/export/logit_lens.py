"""Export for fig-09-05 `gpt-logit-lens`: the prediction at every layer and position of the 12-layer model.

    python -m gptpp.export.logit_lens --ckpt $GPTPP_WORK/ckpt/gptpp-base            (the run directory: model.pt)
    python -m gptpp.export.logit_lens --ckpt $GPTPP_WORK/ckpt/gptpp-base/ckpt_6000.pt

For each of a few short passages whose next token can be told from the passage (a fact, a repeated name, an open
bracket, a list of numbers, an agreement in number, a pronoun), the logit lens of `gptpp.interp.lens`: the model's
own readout (final normalization, unembedding, cap on the logits, softmax) applied to the stream after the
embedding and after each block. A passage is kept only if the model's top token after it is the expected one;
candidates are tried in the order written, at most `--per-kind` of a kind and `--passages` in all, and the first
kept is the figure's default. The model reads `<|bos|>` before each passage, as at the start of a document; that
position is not exported.

And, over held-out text (the first tokens of held-out documents), the mean by depth of the relative entropy from
the model's final distribution to the lens's, with the share of positions whose top token is already the final
one, the lens's bias and the size of the stream. The same two averages are measured through a tuned lens (an affine
map per depth, fitted here in `--tuned-steps` steps on other held-out text: `gptpp.interp.lens.fit_tuned_lens`),
which says how much of the logit lens's distance is the readout's basis and how much is computation still to come.

Writes `logit-lens.json`:

    model      what was read: run, checkpoint, step, tokens seen, shape
    vocab      the texts of the tokens mentioned; every token below is an index into it
    passages   [{key, kind, text, answer, tokens (T), next (T: the token that follows each position in the text,
               the expected answer at the last), final (T: the model's prediction), top (D x T x 5), p (D x T x 5),
               p_final, rank_final, kl_bits (D x T), settles, first_top (T), answer_p, mean_kl (D),
               settle_median, tuned {top (D x T: the tuned lens's top token), p_final, rank_final (D x T),
               settles (T)}}]     D = layers + 1 depths: 0 after the embedding, l after block l
    held_out   {windows, window, positions, kl_bits, kl_se, top1, p_final, bias_bits, settle_hist (D),
               settle_median, rms (D: root mean square of the stream's coordinates), tuned_kl_bits, tuned_kl_se,
               tuned_top1 (D), tuned {steps, windows, positions, lr, first_bits, last_bits (the fit's loss by depth)}}
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from ..data import TokenData
from ..interp import lens
from ..tokenizer import Tokenizer
from .common import DATA, write_json
from .interp_common import TOKENIZER, Texts, load_model, pick_device

NAME = "logit-lens"
TUNED_FROM = 5_000_000        # the tuned lens is fitted on held-out tokens from here on; the averages use the first 4M

# (key, kind, the passage, the token expected after it), in order of preference
CANDIDATES = [
    ("president", "a fact", "The first president of the United States was George", " Washington"),
    ("name", "a repeated name", "Dr. Miller teaches biology at the school. The students like Dr.", " Miller"),
    ("bracket", "an open bracket", "The World Health Organization (WHO", ")"),
    ("count", "a list of numbers", "1, 2, 3, 4, 5, ", "6"),
    ("keys", "agreement in number", "The keys to the cabinet", " are"),
    ("language", "a fact", "In France, the official language is", " French"),
    ("queen", "a pronoun", "The queen smiled at the crowd and raised", " her"),
    ("play", "a title", "Shakespeare wrote the play Romeo and", " Juliet"),
    # spares, used when the model misses one of the above
    ("house", "a fact", "The president of the United States lives in the White", " House"),
    ("breath", "a fact", "Humans breathe in oxygen and breathe out carbon", " dioxide"),
    ("mrs", "a repeated name", 'Mrs. Baker opened the door. "Come in," said Mrs.', " Baker"),
    ("nasa", "an open bracket", "The National Aeronautics and Space Administration (NASA", ")"),
    ("letters", "a list of letters", "A, B, C, D,", " E"),
    ("ordinals", "a list of ordinals", "first, second, third,", " fourth"),
    ("children", "agreement in number", "The children in the garden", " are"),
    ("tired", "a pronoun", "The queen was tired, so", " she"),
]


def passage(model, tokenizer: Tokenizer, texts: Texts, key: str, kind: str, text: str, answer: str, top: int, tuned: dict | None = None) -> dict | None:
    """The lens on one passage, or None if the model's top token after it is not the expected one. With a tuned
    lens, also what it makes of the passage: its top token, and the probability and rank it gives the final one."""
    device = model.embed.weight.device
    ids = tokenizer.encode(text, bos=True)
    want = tokenizer.encode(answer, bos=False)[0]                    # the first token of the answer
    t = lens.lens_table(model, torch.tensor(ids, device=device), top=top)
    if int(t["final_id"][-1]) != want:
        return None
    body = slice(1, None)                                           # without <|bos|>
    through = {}
    if tuned is not None:
        u = lens.tuned_table(model, torch.tensor(ids, device=device), tuned, top=1)
        through = {"tuned": {"top": texts.indices(u["top_ids"][:, body, 0]), "p_final": np.round(u["p_final"][:, body], 5),
                             "rank_final": u["rank_final"][:, body], "settles": u["settles"][body]}}
    return {**through,
        "key": key, "kind": kind, "text": text, "answer": texts.index(want),
        "tokens": texts.indices(ids[1:]), "next": texts.indices(ids[2:] + [want]), "final": texts.indices(t["final_id"][body]),
        "top": texts.indices(t["top_ids"][:, body]), "p": np.round(t["top_p"][:, body], 4),
        "p_final": np.round(t["p_final"][:, body], 5), "rank_final": t["rank_final"][:, body],
        "kl_bits": np.round(t["kl_bits"][:, body], 3), "settles": t["settles"][body], "first_top": t["first_top"][body],
        "answer_p": float(t["p_final"][-1, -1]), "mean_kl": np.round(t["kl_bits"][:, body].mean(1), 3),
        "settle_median": float(np.median(t["settles"][body])),
    }


def held_out_windows(data: TokenData, n: int, length: int) -> torch.Tensor:
    """The first `length` tokens (`<|bos|>` first) of the first n held-out documents that are long enough."""
    val, bos = data.val_tokens(), data.meta["bos_id"]
    head = np.asarray(val[: min(len(val), 4_000_000)])
    starts = np.flatnonzero(head == bos)
    ends = np.append(starts[1:], len(head))
    keep = starts[(ends - starts) >= length][:n]
    if len(keep) < n:
        raise ValueError(f"only {len(keep)} held-out documents of at least {length} tokens")
    return torch.tensor(np.stack([head[s:s + length].astype(np.int64) for s in keep]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True, help="model.pt, ckpt_<step>.pt, or the run directory")
    ap.add_argument("--out", default=str(DATA))
    ap.add_argument("--device", default=None)
    ap.add_argument("--data", default="fineweb-edu/bpe32k")
    ap.add_argument("--passages", type=int, default=8)
    ap.add_argument("--per-kind", type=int, default=2)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--windows", type=int, default=64, help="held-out documents for the averages")
    ap.add_argument("--window", type=int, default=128, help="tokens read of each, after <|bos|>")
    ap.add_argument("--tuned-steps", type=int, default=200, help="steps of the tuned lens's fit; 0: no tuned lens")
    ap.add_argument("--tuned-batch", type=int, default=8, help="windows a step")
    ap.add_argument("--tuned-lr", type=float, default=1e-3)
    args = ap.parse_args()
    t0 = time.time()
    model, meta = load_model(args.ckpt, pick_device(args.device))
    tokenizer, texts = Tokenizer.load(TOKENIZER), Texts(Tokenizer.load(TOKENIZER))

    data = TokenData(args.data)
    windows = held_out_windows(data, args.windows, args.window + 1)
    tuned, fit = None, {}
    if args.tuned_steps > 0:                               # fitted on held-out text that the averages do not use
        n, width = args.tuned_steps * args.tuned_batch, args.window + 1
        val = data.val_tokens()
        if TUNED_FROM + n * width > len(val):
            raise SystemExit(f"the held-out set is too short for {n} windows after token {TUNED_FROM}")
        train = torch.from_numpy(np.asarray(val[TUNED_FROM: TUNED_FROM + n * width]).astype(np.int64).reshape(n, width))
        tuned = lens.fit_tuned_lens(model, train, steps=args.tuned_steps, batch=args.tuned_batch, lr=args.tuned_lr)
        fit = {"steps": args.tuned_steps, "windows": n, "positions": n * args.window, "lr": args.tuned_lr,
               "first_bits": np.round(tuned["loss"][:5].mean(0).numpy(), 3), "last_bits": np.round(tuned["loss"][-5:].mean(0).numpy(), 3)}

    kept, by_kind, missed = [], {}, []
    for key, kind, text, answer in CANDIDATES:
        if len(kept) == args.passages or by_kind.get(kind, 0) == args.per_kind:
            continue
        p = passage(model, tokenizer, texts, key, kind, text, answer, args.top, tuned)
        if p is None:
            missed.append(key)
            continue
        kept.append(p)
        by_kind[kind] = by_kind.get(kind, 0) + 1
    if not kept:
        raise SystemExit("the model predicts the expected token after none of the passages")

    stats = lens.lens_statistics(model, windows, skip=1, batch=8)
    x = lens.streams(model, windows[:16].to(model.embed.weight.device))[:, :, 1:].float()
    held = {"windows": stats["windows"], "window": args.window, "positions": stats["positions"],
            "kl_bits": np.round(stats["kl_bits"], 4), "kl_se": np.round(stats["kl_se"], 4), "top1": np.round(stats["top1"], 4),
            "p_final": np.round(stats["p_final"], 5), "bias_bits": np.round(stats["bias_bits"], 4),
            "settle_hist": stats["settle_hist"].astype(int), "settle_median": stats["settle_median"],
            "rms": x.square().mean(-1).sqrt().mean(dim=(1, 2)).cpu().numpy()}
    if tuned is not None:
        t = lens.tuned_statistics(model, windows, tuned, skip=1, batch=8)
        held.update(tuned_kl_bits=np.round(t["kl_bits"], 4), tuned_kl_se=np.round(t["kl_se"], 4), tuned_top1=np.round(t["top1"], 4), tuned=fit)
    out = {"model": meta, "depths": meta["layers"] + 1, "top": args.top, "vocab": texts.table, "passages": kept, "held_out": held,
           "missed": missed}
    size = write_json(f"{args.out}/{NAME}.json", out)
    print(f"{NAME}.json: {size / 1e3:.0f} kB, {len(kept)} passages ({', '.join(p['key'] for p in kept)}), missed {missed}, "
          f"{held['positions']} held-out positions, {time.time() - t0:.0f} s on {meta['checkpoint']} (step {meta['step']})")
    for p in kept:
        print(f"  {p['key']:10s} {len(p['tokens']):2d} tokens, answer {texts.table[p['answer']]!r} at p = {p['answer_p']:.3f}, "
              f"median settling depth {p['settle_median']:g}, mean KL after layer 1: {p['mean_kl'][1]:.2f}, 11: {p['mean_kl'][-2]:.2f} bits")
    print("  held-out mean KL by depth:", " ".join(f"{v:.2f}" for v in held["kl_bits"]))
    if tuned is not None:
        print("  through the tuned lens:   ", " ".join(f"{v:.2f}" for v in held["tuned_kl_bits"]))


if __name__ == "__main__":
    main()
