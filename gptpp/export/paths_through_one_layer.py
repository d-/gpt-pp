"""Export for fig-09-01 `gpt-paths-through-one-layer`: predictions of the one-layer attention-only model on held-out
stories, each split exactly into its nine paths, with the tables the paths are read from.

    python -m gptpp.export.paths_through_one_layer [--one circ-attn1] [--device cpu] [--examples 240]

The logits of a one-layer attention-only model without normalization are the direct path U E^T at the present token
plus, for each head, the sum over earlier positions of (attention to the position) x (the OV circuit's column of the
token there): gptpp/interp/circuits.py, path_expansion. For each of a few hundred frequent word tokens the export
holds one place in the held-out stories where it occurs with 15 tokens of its story before it, and there:

    outputs      the model's three first choices for the next token, with probability and logit
    terms        for each of them: the direct path's term and the eight heads' terms, (examples, 3, 9); a term is
                 measured from its mean over the vocabulary, so the nine add up to the logit measured the same way
    sources      each head's term split by source position, (examples, 3, heads, 16)
    attn, score  each head's attention over the 16 positions, and the scores before the softmax
    direct, corpus, zero   what follows the present token: the six largest entries of the direct path's column
                 (logits), of the corpus's bigram counts (probabilities), and of a zero-layer model, the two matrices
                 alone, fitted to the same counts (circuits.fit_zero_layer)

and, for every token that occurs in one of those contexts, each head's OV column cut to its four largest entries
(`ov`: what a look at that token writes). Lists of outputs are restricted to tokens seen at least 10,000 times in
training: unrestricted, the largest entries of a column fall on rare word pieces that other paths cancel.
`agreement` says how close the direct path and the zero-layer model are to the corpus's bigrams over the whole
vocabulary (circuits.bigram_agreement). Writes `paths-through-one-layer.json` and `.bin`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..interp import circuits as C
from ..interp import ngram_order as ng
from .circuit_runs import ONE_STORIES, Run, held_out, is_word, story_data, token_names
from .common import DATA, Blob, write_json

NAME = "paths-through-one-layer"
CONTEXT = 16  # the present token and the 15 before it
OUTPUTS = 3
LISTED = 6  # entries of a list of next tokens
WRITTEN = 4  # entries of an OV column
MIN_COUNT = 10_000  # an output is listed only if it was seen this often


def find_examples(windows: torch.Tensor, candidates: list[int], bos: int, limit: int) -> list[tuple[int, int, int]]:
    """(token, window, position) of the first held-out occurrence of each candidate with 15 tokens of its own
    story before it and a token after it."""
    first: dict[int, tuple[int, int]] = {}
    wanted = set(candidates)
    for b, row in enumerate(windows.tolist()):
        for p in range(CONTEXT - 1, len(row) - 1):
            token = row[p]
            if token in wanted and token not in first and bos not in row[p - CONTEXT + 1:p + 2]:
                first[token] = (b, p)
        if len(first) == len(wanted):
            break
    return [(t, *first[t]) for t in candidates if t in first][:limit]


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--one", default=ONE_STORIES)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--examples", type=int, default=240)
    parser.add_argument("--fit-steps", type=int, default=800)
    parser.add_argument("--out", default=str(DATA))
    args = parser.parse_args()
    torch.set_num_threads(min(16, torch.get_num_threads()))
    run = Run(args.one, args.device)
    model, cfg, bos = run.model(), run.cfg, int(run.data.meta["bos_id"])
    device, H, names = model.embed.weight.device, cfg.n_head, token_names()
    unigram, bigram = (torch.from_numpy(a) for a in ng.count_tables(story_data()))
    listed = torch.tensor([t for t in range(cfg.vocab_size) if unigram[t] >= MIN_COUNT and t != bos], device=device)
    order = unigram.argsort(descending=True).tolist()
    candidates = [t for t in order[:700] if is_word(names[t]) and names[t].startswith(" ") and t != bos]
    examples = find_examples(held_out(2048), candidates, bos, args.examples)
    windows = held_out(2048)

    unembed, embed = C.unembedding(model), C.embedding(model)
    direct_all = C.direct_path(model)  # (V, V) [output, source]
    zero_e, zero_u, zero_losses = C.fit_zero_layer(bigram, cfg.d_model, steps=args.fit_steps, device=device)
    zero_all = zero_u @ zero_e.T
    agreement = {"direct": C.bigram_agreement(direct_all, bigram), "zero": C.bigram_agreement(zero_all, bigram),
                 "zeroSteps": args.fit_steps, "zeroLoss": zero_losses[-1]}

    def top(values: torch.Tensor, k: int) -> tuple[list[int], list[float]]:
        """The k largest of `values` (V,) among the listed tokens: ids and values."""
        best = values[listed].topk(k)
        return listed[best.indices].tolist(), best.values.tolist()

    E = len(examples)
    terms = np.zeros((E, OUTPUTS, 1 + H), dtype=np.float32)
    sources = np.zeros((E, OUTPUTS, H, CONTEXT), dtype=np.float32)
    attn = np.zeros((E, H, CONTEXT), dtype=np.uint8)
    score = np.zeros((E, H, CONTEXT), dtype=np.float16)
    records, used, worst = [], set(), 0.0
    for e, (token, b, p) in enumerate(examples):
        idx = windows[b:b + 1, p - CONTEXT + 1:p + 1].to(device)
        logits, cache = model.run_with_cache(idx, names=["attn.0.pattern", "attn.0.scores"])
        last = logits[0, -1].double()
        probability = F.softmax(last, dim=-1)
        outputs = last.topk(OUTPUTS).indices.tolist()
        pattern = cache["attn.0.pattern"][0, :, -1].double()  # (H, T): the present token's attention
        columns = embed[idx[0]]  # (T, d)
        centered = unembed[outputs] - unembed.mean(dim=0, keepdim=True)  # (3, d): logits measured from their mean
        terms[e, :, 0] = (centered @ columns[-1]).cpu().numpy()
        for h in range(H):
            written = columns @ C.ov_matrix(model, 0, h).T  # (T, d): what a look at each position writes
            split = pattern[h][None, :] * (centered @ written.T)  # (3, T)
            sources[e, :, h] = split.cpu().numpy()
            terms[e, :, 1 + h] = split.sum(dim=1).cpu().numpy()
        whole = (last[outputs] - last.mean()).cpu().numpy()
        worst = max(worst, float(np.abs(terms[e].sum(axis=1) - whole).max()))
        attn[e] = np.rint(pattern.cpu().numpy() * 255)
        score[e] = cache["attn.0.scores"][0, :, -1].float().cpu().numpy()
        counts = bigram[token].double()
        d_ids, d_values = top(direct_all[:, token] - direct_all[:, token].mean(), LISTED)
        z_ids, z_values = top(F.softmax(zero_all[:, token], dim=0), LISTED)
        c_best = counts.topk(LISTED)
        records.append({
            "tokens": idx[0].tolist(), "next": int(windows[b, p + 1]), "outputs": outputs,
            "probability": probability[outputs].tolist(), "logit": whole.tolist(),
            "direct": [d_ids, d_values], "zero": [z_ids, z_values],
            "corpus": [c_best.indices.tolist(), (c_best.values / counts.sum()).tolist()], "count": int(unigram[token]),
        })
        used.update(idx[0].tolist())

    rows = sorted(used)
    ov_ids = np.zeros((len(rows), H, WRITTEN), dtype=np.float32)
    ov_values = np.zeros((len(rows), H, WRITTEN), dtype=np.float16)
    for h in range(H):
        table = C.ov_circuit(model, 0, h, sources=rows)  # (V, rows)
        table = table - table.mean(dim=0, keepdim=True)
        best = table[listed].topk(WRITTEN, dim=0)
        ov_ids[:, h] = listed[best.indices].T.cpu().numpy()
        ov_values[:, h] = best.values.T.cpu().numpy()

    # The example the figure opens on: the model's first choice is the token that came, it is sure of it, and one
    # head's term is the largest of the nine by the widest margin.
    def margin(e: int) -> float:
        record, t = records[e], terms[e, 0]
        if record["outputs"][0] != record["next"] or record["probability"][0] < 0.5:
            return -1.0
        ranked = np.sort(t[1:])[::-1]
        return float(ranked[0] - max(ranked[1], t[0]))
    featured = max(range(E), key=margin)

    blob = Blob(NAME, args.out)
    blob.put(f"{NAME}.terms", terms, "the direct path's term and each head's, for the three first choices: (examples, outputs, 1 + heads)")
    blob.put(f"{NAME}.sources", sources, "each head's term by source position: (examples, outputs, heads, positions)")
    blob.put(f"{NAME}.attn", attn, "the present token's attention: (examples, heads, positions)", lo=0.0, hi=1.0)
    blob.put(f"{NAME}.score", score, "its scores before the softmax: (examples, heads, positions)")
    blob.put(f"{NAME}.ovids", ov_ids, "token ids of the largest entries of each head's OV column: (context tokens, heads, entries)")
    blob.put(f"{NAME}.ov", ov_values, "those entries, in logits from the column's mean: (context tokens, heads, entries)")
    size = blob.save()
    mentioned = set(rows) | {int(t) for t in ov_ids.flatten()}
    for record in records:
        mentioned.update(record["outputs"] + [record["next"]] + record["direct"][0] + record["zero"][0] + record["corpus"][0])
    out = {
        "model": run.describe(), "context": CONTEXT, "minCount": MIN_COUNT, "listed": len(listed), "featured": featured,
        "examples": records, "ovRows": {str(t): i for i, t in enumerate(rows)}, "agreement": agreement,
        "largestError": worst, "names": {str(t): names[t] for t in sorted(mentioned)},
    }
    json_size = write_json(Path(args.out) / f"{NAME}.json", out)
    print(f"wrote {NAME}.json ({json_size:,} bytes) and {NAME}.bin ({size:,} bytes): {E} examples, {len(rows)} context tokens; "
          f"largest |sum of the nine terms - logit| {worst:.2e}; featured example {featured} "
          f"({names[records[featured]['tokens'][-1]]!r} -> {names[records[featured]['outputs'][0]]!r}); agreement {agreement}")


if __name__ == "__main__":
    main()
