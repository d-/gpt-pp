"""Export for fig-09-04 `gpt-activation-patching`: where the answer is restored in the 12-layer model.

    python -m gptpp.export.activation_patching --ckpt $GPTPP_WORK/ckpt/gptpp-base            (the run directory: model.pt)
    python -m gptpp.export.activation_patching --ckpt $GPTPP_WORK/ckpt/gptpp-base/ckpt_6000.pt

Pairs of prompts that differ in one token and in their answer: a clean prompt ("In France, the official language
is", answer " French") and a corrupted one ("In Spain, ...", " Spanish"). A pair is kept only if it is well formed
for this tokenizer (the same number of tokens, exactly one of them different and not the last, both answers one
token) and the model handles it: the clean answer is its top token after the clean prompt and the corrupted answer
after the corrupted one, each with a clear logit difference (at least `--margin`, 2 by default, over the other
prompt's answer). Candidates are tried in the order written, one pair of each kind, `--pairs` in all; the first
kept is the figure's default.

For each pair, with `gptpp.interp.patching` (the metric is the logit difference at the last position between the
two answers, before the model's cap on its logits):

    denoising   the clean value of one node written into the corrupted run, for the stream entering each layer
                (`resid_pre`), each layer's attention output and each layer's MLP output, at every position: the
                share of the logit difference restored, 0 as the corrupted run and 1 as the clean one
    noising     the corrupted value written into the clean run: the share that is left
    direct      the direct logit attribution of every head, of every MLP and of the embedding on the clean prompt,
                the scale of the final normalization frozen; they add up to the clean logit difference
    attention   every head's attention from the last position on the clean prompt, and each head's direct
                attribution split by the position it read

The model reads `<|bos|>` before each prompt; that position is the same in both runs and is not exported.

Writes `activation-patching.json`:

    model   what was read          vocab   the texts of the tokens mentioned (tokens below are indices into it)
    pairs   [{key, kind, clean, corrupt (the prompts' texts), tokens, tokens_corrupt (T), changed (the position that
            differs), answer, answer_corrupt, ld_clean, ld_corrupt, p_clean, p_corrupt (the probability of each
            prompt's own answer), denoise {resid_pre, attn_out, mlp_out: L x T}, noise {the same}, heads (L x H),
            mlp (L), embed, rho, attention (L x H x T), sources (L x H x T), attention_start, sources_start (L x H:
            the head's attention to `<|bos|>` and what it reads there)}]
    missed  the candidates not kept, each with the reason
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from ..interp import patching
from ..tokenizer import Tokenizer
from .common import DATA, write_json
from .interp_common import TOKENIZER, Texts, load_model, one_token, pick_device

NAME = "activation-patching"

# (key, kind, clean prompt, corrupted prompt, clean answer, corrupted answer), in order of preference within a kind
CANDIDATES = [
    ("language", "a fact", "In France, the official language is", "In Spain, the official language is", " French", " Spanish"),
    ("language-of", "a fact", "The official language of France is", "The official language of Spain is", " French", " Spanish"),
    ("language-asia", "a fact", "In Japan, the official language is", "In China, the official language is", " Japanese", " Chinese"),
    ("keys", "agreement in number", "The keys to the cabinet", "The key to the cabinet", " are", " is"),
    ("children", "agreement in number", "The children in the garden", "The child in the garden", " are", " is"),
    ("queen", "a pronoun", "The queen smiled at the crowd and raised", "The king smiled at the crowd and raised", " her", " his"),
    ("tired", "a pronoun", "The queen was tired, so", "The king was tired, so", " she", " he"),
    ("name", "a repeated name", "Dr. Miller teaches biology at the school. The students like Dr.",
     "Dr. Baker teaches biology at the school. The students like Dr.", " Miller", " Baker"),
    ("mrs", "a repeated name", 'Mrs. Baker opened the door. "Come in," said Mrs.', 'Mrs. Miller opened the door. "Come in," said Mrs.',
     " Baker", " Miller"),
    ("bracket", "an open bracket", "The World Health Organization (WHO", "The World Health Organization [WHO", ")", "]"),
    ("bracket-un", "an open bracket", "The United Nations (the UN", "The United Nations [the UN", ")", "]"),
    ("gift", "an indirect object", "When Mary and John went to the store, John gave a drink to",
     "When Mary and John went to the store, Mary gave a drink to", " Mary", " John"),
    ("ball", "an indirect object", "Mary and John went to the park. John gave the ball to",
     "Mary and John went to the park. Mary gave the ball to", " Mary", " John"),
    ("present", "an indirect object", "Alice and Bob are friends. Alice gave a present to", "Alice and Bob are friends. Bob gave a present to", " Bob", " Alice"),
    ("tense", "a tense", "She said that tomorrow she", "She said that yesterday she", " will", " was"),
    ("capital", "a capital", "The capital of Japan is", "The capital of Italy is", " Tokyo", " Rome"),
]


def measure(model, tokenizer: Tokenizer, texts: Texts, key: str, kind: str, clean: str, corrupt: str, a_text: str,
            b_text: str, margin: float = 2.0) -> tuple[dict | None, str]:
    """One pair measured, or (None, why it is not kept)."""
    device = model.embed.weight.device
    ci, ki = tokenizer.encode(clean, bos=True), tokenizer.encode(corrupt, bos=True)
    a, b = one_token(tokenizer, a_text), one_token(tokenizer, b_text)
    if a is None or b is None:
        return None, "an answer is not one token"
    differ = [i for i, (u, v) in enumerate(zip(ci, ki)) if u != v]
    if len(ci) != len(ki) or len(differ) != 1 or differ[0] == len(ci) - 1:
        return None, f"not one changed token before the last ({len(ci)} and {len(ki)} tokens, differing at {differ})"
    ct, kt = torch.tensor(ci, device=device), torch.tensor(ki, device=device)
    with torch.no_grad():
        pc, pk = model(ct.view(1, -1))[0, -1].float().softmax(-1), model(kt.view(1, -1))[0, -1].float().softmax(-1)
    if int(pc.argmax()) != a or int(pk.argmax()) != b:
        return None, (f"the model's top tokens are {tokenizer.decode([int(pc.argmax())])!r} and "
                      f"{tokenizer.decode([int(pk.argmax())])!r} (answers at p = {float(pc[a]):.3f} and {float(pk[b]):.3f})")
    ld = (float(patching.logit_difference(model, ct.view(1, -1), a, b)), float(patching.logit_difference(model, kt.view(1, -1), a, b)))
    if ld[0] < margin or ld[1] > -margin:
        return None, f"the logit difference is not clear: {ld[0]:+.2f} on the clean prompt, {ld[1]:+.2f} on the corrupted one (answers at p = {float(pc[a]):.3f} and {float(pk[b]):.3f})"
    e = patching.patching_effects(model, ct, kt, a, b)
    d = patching.direct_attribution(model, ct, a, b)
    attention, sources = patching.attention_from(model, ct), patching.head_sources(model, ct, a, b)
    body = slice(1, None)                                           # without <|bos|>
    cut = lambda maps: {site: np.round(m[:, body].numpy(), 4) for site, m in maps.items()}
    return {
        "key": key, "kind": kind, "clean": clean, "corrupt": corrupt,
        "tokens": texts.indices(ci[1:]), "tokens_corrupt": texts.indices(ki[1:]), "changed": differ[0] - 1,
        "answer": texts.index(a), "answer_corrupt": texts.index(b),
        "ld_clean": e["ld_clean"], "ld_corrupt": e["ld_corrupt"], "p_clean": float(pc[a]), "p_corrupt": float(pk[b]),
        "denoise": cut(e["denoise"]), "noise": cut(e["noise"]),
        "heads": np.round(d["heads"].numpy(), 4), "mlp": np.round(d["mlp"].numpy(), 4), "embed": d["embed"], "rho": d["rho"],
        "total": d["total"],
        "attention": np.round(attention[:, :, body].numpy(), 3), "attention_start": np.round(attention[:, :, 0].numpy(), 3),
        "sources": np.round(sources[:, :, body].numpy(), 3), "sources_start": np.round(sources[:, :, 0].numpy(), 3),
    }, ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True, help="model.pt, ckpt_<step>.pt, or the run directory")
    ap.add_argument("--out", default=str(DATA))
    ap.add_argument("--device", default=None)
    ap.add_argument("--pairs", type=int, default=6)
    ap.add_argument("--margin", type=float, default=2.0, help="the least logit difference, on each prompt, of a pair that is kept")
    args = ap.parse_args()
    t0 = time.time()
    model, meta = load_model(args.ckpt, pick_device(args.device))
    tokenizer, texts = Tokenizer.load(TOKENIZER), Texts(Tokenizer.load(TOKENIZER))
    kept, kinds, missed = [], set(), []
    for key, kind, clean, corrupt, a, b in CANDIDATES:
        if len(kept) == args.pairs or kind in kinds:
            continue
        pair, why = measure(model, tokenizer, texts, key, kind, clean, corrupt, a, b, args.margin)
        if pair is None:
            missed.append({"key": key, "kind": kind, "why": why})
            continue
        kept.append(pair)
        kinds.add(kind)
    if not kept:
        raise SystemExit("the model handles none of the pairs")
    out = {"model": meta, "sites": list(patching.SITES), "vocab": texts.table, "pairs": kept, "missed": missed}
    size = write_json(f"{args.out}/{NAME}.json", out)
    print(f"{NAME}.json: {size / 1e3:.0f} kB, {len(kept)} pairs, {time.time() - t0:.0f} s on {meta['checkpoint']} (step {meta['step']})")
    for p in kept:
        heads = np.asarray(p["heads"])
        l, h = np.unravel_index(np.abs(heads).argmax(), heads.shape)
        print(f"  {p['key']:10s} {len(p['tokens']):2d} tokens, changed at {p['changed']}, LD {p['ld_clean']:+.2f} clean, {p['ld_corrupt']:+.2f} "
              f"corrupted, p {p['p_clean']:.2f} and {p['p_corrupt']:.2f}; largest head {l + 1}.{h + 1}: {heads[l, h]:+.2f}; "
              f"attributions sum to {p['total']:+.3f}")
    for m in missed:
        print(f"  not kept: {m['key']} ({m['kind']}): {m['why']}")


if __name__ == "__main__":
    main()
