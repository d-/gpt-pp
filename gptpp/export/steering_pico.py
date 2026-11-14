"""Export pico's dictionary for clamping in the page, and measure what a clamp does: what fig-10-05
(`gpt-steering-pico`) needs, and the numbers of the post's example of steering.

    python -m gptpp.export.steering_pico --run <pico's run dir> --layer 2 --dict topk-n4096-k16 [--out <dir>]

To clamp latent i at the value c is to replace the stream x after the dictionary's block by x + (c - f_i(x)) u_i
at every token after the document-start mark, and to let the model run on (gptpp.interp.sae.clamp). The page does
this with pico as it ships it (8-bit weights), so everything here is computed on that model, `page_model('pico')`,
not on the checkpoint: the two differ in the last digits, and a story sampled here must be the story the page
writes.

For each feature of the hand-made list `gptpp/export/curated/pico-L<layer>-steering.json` (`{ id, name, words,
note }`) and each strength (a multiple of the latent's largest activation on held-out text):

    theme       the share of `stories` sampled stories in which one of the feature's words appears, with the clamp
                and without it, from the same opening and on the same random draws
    differ      the mean share of positions at which the clamped story has another token than the unclamped one
    loss        the model's next-token loss on held-out stories when the latent is clamped everywhere, against
                its loss as trained: what the clamp costs on text that is not about the feature
    surprise    how unlikely the unclamped model finds the clamped stories (their mean negative log-likelihood,
                nats a token), against the stories it wrote itself: a measure of how far the clamp broke the text

Sampling is the page's: the uniform that draws the token at position i is a hash of the seed and i (the page's
`GPTModel.rng`), the probabilities are computed and the token picked as the page does, so that the recorded
story of the default view is the story a reader gets on pressing [write].

Files: `steering-pico.json` (the features with their measurements, the openings, the default view and its recorded
stories with the latent's reading along them) and `steering-pico.bin` (the dictionary's encoder as float16, its
biases and normalization, and the directions of the listed features as float32).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from gptpp.data import TokenData
from gptpp.export.browser_model import page_model
from gptpp.export.common import DATA, Blob, write_json
from gptpp.export.feature_browser import run_facts
from gptpp.generate import load_tokenizer
from gptpp.interp import sae as S
from gptpp.model import GPT, Hooks, KVCache

NAME = "steering-pico"
OPENINGS = ["Once upon a time,", "One day, a little girl named Lily", "Tom and his dog went to the"]
MULTIPLES = [0.25, 0.5, 1.0, 2.0]
TEMPERATURE, STEPS = 0.7, 72


# ----------------------------------------------------------------------------------------------------------------
# The page's sampler, to the bit


def _fmix(a: int) -> int:
    a &= 0xFFFFFFFF
    a ^= a >> 16
    a = (a * 0x85EBCA6B) & 0xFFFFFFFF
    a ^= a >> 13
    a = (a * 0xC2B2AE35) & 0xFFFFFFFF
    return (a ^ (a >> 16)) & 0xFFFFFFFF


def uniform(seed: int, position: int) -> float:
    """The uniform in [0, 1) that draws the token at `position` of a story with `seed`: `GPTModel.rng(seed).at(i)`."""
    base = _fmix((seed & 0xFFFFFFFF) + 0x9E3779B9)
    return _fmix((base + ((position + 1) * 0x9E3779B1 & 0xFFFFFFFF)) & 0xFFFFFFFF) / 4294967296


def page_sample(logits: torch.Tensor, temperature: float, u: Sequence[float]) -> torch.Tensor:
    """One token per row of logits (B, V) as the page draws it: probabilities exp((l - max) / T) / sum in double
    precision, kept as float32, and the first token, in the order of the ids, at which the running sum passes u."""
    z = (logits.double() - logits.double().max(-1, keepdim=True).values) / temperature
    e = z.exp()
    probs = (e.float().double() / e.sum(-1, keepdim=True)).float().double()
    passed = probs.cumsum(-1) > torch.tensor(u, dtype=torch.float64, device=logits.device)[:, None]
    last = (probs > 0).float().cumsum(-1).argmax(-1)  # if rounding leaves u above the whole sum: the last token with any probability
    return torch.where(passed.any(-1), passed.float().argmax(-1), last)


# ----------------------------------------------------------------------------------------------------------------
# Writing with a latent clamped


@torch.no_grad()
def write(model: GPT, sae: S.SparseAutoencoder, layer: int, prompt: Sequence[int], seeds: Sequence[int], steps: int,
          temperature: float, latent: int, value: Optional[float]) -> tuple[torch.Tensor, torch.Tensor]:
    """One story per seed from one prompt (which begins with the document-start mark), with `latent` clamped at
    `value` after block `layer` at every token but the mark (value None: nothing is clamped). Returns the tokens
    (B, P + steps) and the latent's reading of the stream before the clamp at every position (B, P + steps)."""
    device = model.embed.weight.device
    idx = torch.tensor([list(prompt)] * len(seeds), device=device)
    readings, kv, start = [], KVCache(model.cfg.n_layer), [True]

    def hook(x: torch.Tensor) -> Optional[torch.Tensor]:
        y, f = S.clamp(sae, x, latent, 0.0 if value is None else value)
        if start[0]:  # the first call holds the prompt: its first token is the mark, which is never clamped
            y[:, 0], f[:, 0], start[0] = x[:, 0], 0.0, False
        readings.append(f)
        return None if value is None else y

    hooks = Hooks({f"resid_post.{layer}": hook})
    logits = model.forward_explicit(idx, hooks, kv)[:, -1]
    for step in range(steps):
        position = idx.size(1)
        token = page_sample(logits, temperature, [uniform(s, position) for s in seeds])[:, None]
        idx = torch.cat((idx, token), dim=1)
        logits = model.forward_explicit(token, hooks, kv)[:, -1]
    return idx, torch.cat(readings, dim=1)


def story_ends(tokens: torch.Tensor, n_prompt: int, bos: int) -> list[int]:
    """Where each story stops: at the first document-start mark it wrote, or at its last token."""
    ends = []
    for row in tokens.tolist():
        tail = row[n_prompt:]
        ends.append(n_prompt + tail.index(bos) if bos in tail else len(row))
    return ends


def has_word(text: str, words: Sequence[str]) -> bool:
    return any(re.search(rf"\b{re.escape(w)}", text, flags=re.IGNORECASE) for w in words)


@torch.no_grad()
def surprise(model: GPT, tokens: torch.Tensor, n_prompt: int, ends: Sequence[int]) -> float:
    """The mean negative log-likelihood (nats a token) that the model as trained gives to what was written."""
    logits = model(tokens[:, :-1])
    nll = F.cross_entropy(logits.transpose(1, 2), tokens[:, 1:], reduction="none")  # (B, T - 1): row t predicts t + 1
    mask = torch.zeros_like(nll, dtype=torch.bool)
    for b, end in enumerate(ends):
        mask[b, n_prompt - 1:end - 1] = True
    return float(nll[mask].mean())


@torch.no_grad()
def heldout_loss(model: GPT, sae: S.SparseAutoencoder, tokens: np.ndarray, harvest: S.Harvest, windows: int,
                 latent: int, value: Optional[float], batch: int = 64) -> float:
    """The model's loss on held-out windows with the latent clamped at every counted token (value None: as trained)."""
    total, count = 0.0, 0
    device = str(model.embed.weight.device)
    for _, idx in S.window_batches(tokens, harvest.context, batch, windows, device):
        x = S.activations(model, idx, harvest)
        replacement = x if value is None else S.clamp(sae, x, latent, value)[0]
        total += float(S.spliced_loss(model, idx, x, replacement, harvest))
        count += int(harvest.keep(idx)[:, :-1].sum())
    return total / count


def measure(model: GPT, sae: S.SparseAutoencoder, tok, layer: int, feature: dict, peak: float, opening: str,
            seeds: Sequence[int], heldout: np.ndarray, harvest: S.Harvest, windows: int, plain: dict, say=print) -> list[dict]:
    """A feature at every strength: the theme, the tokens changed, the price in loss, the surprise of the text."""
    prompt = tok.encode(opening, bos=True)
    rows = []
    for m in MULTIPLES:
        value = m * peak
        tokens, _ = write(model, sae, layer, prompt, seeds, STEPS, TEMPERATURE, feature["id"], value)
        ends = story_ends(tokens, len(prompt), tok.bos_id)
        texts = [tok.decode(row[len(prompt):end]) for row, end in zip(tokens.tolist(), ends)]
        differ = [float((tokens[b, len(prompt):min(e, pe)] != plain["tokens"][b, len(prompt):min(e, pe)]).float().mean()) if min(e, pe) > len(prompt) else 0.0
                  for b, (e, pe) in enumerate(zip(ends, plain["ends"]))]
        rows.append({"multiple": m, "value": value,
                     "theme": float(np.mean([has_word(t, feature["words"]) for t in texts])),
                     "theme_plain": float(np.mean([has_word(t, feature["words"]) for t in plain["texts"]])),
                     "differ": float(np.mean(differ)),
                     "loss": heldout_loss(model, sae, heldout, harvest, windows, feature["id"], value),
                     "surprise": surprise(model, tokens, len(prompt), ends),
                     "sample": texts[0][:240]})
        r = rows[-1]
        say(f"  #{feature['id']} {feature['name']!r} x{m:g} (c = {value:.1f}): theme {r['theme']:.2f} (plain {r['theme_plain']:.2f})  "
            f"differ {r['differ']:.2f}  loss {r['loss']:.3f}  surprise {r['surprise']:.2f} | {texts[0][:150]!r}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="pico's run directory (for its dictionaries and its data)")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--dict", required=True)
    ap.add_argument("--features", default=None, help="the hand-made list (default: gptpp/export/curated/pico-L<layer>-steering.json)")
    ap.add_argument("--stories", type=int, default=300, help="stories sampled per condition")
    ap.add_argument("--windows", type=int, default=1024, help="held-out windows for the loss under a clamp")
    ap.add_argument("--default", default="0:1:0:4", help="the default view: feature index : index of the strength : opening : seed")
    ap.add_argument("--skip", type=int, default=16)
    ap.add_argument("--out", default=str(DATA))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    run_dir = Path(args.run) if Path(args.run).is_dir() else S.work_dir() / "ckpt" / args.run
    folder = S.dict_dir(run_dir.name, args.layer)
    record = json.loads((folder / f"{args.dict}.json").read_text())
    meta = json.loads((run_dir / "config.json").read_text())["data_meta"]
    data, tok = TokenData(f"{meta['dataset']}/{meta['tokenizer']}"), load_tokenizer(meta["tokenizer"])
    model, cfg = page_model("pico", DATA, args.device)  # the weights the page loads, from the site's data folder
    sae = S.load(folder / f"{args.dict}.pt", args.device)
    harvest = S.Harvest(args.layer, cfg.block_size, args.skip, int(data.meta["bos_id"]))
    heldout = data.val_tokens()
    path = Path(args.features) if args.features else Path(__file__).parent / "curated" / f"pico-L{args.layer}-steering.json"
    features = json.loads(path.read_text())
    codes = json.loads((folder / f"{args.dict}.json").read_text())["eval"]
    from gptpp.interp.dashboard import Codes
    peaks = Codes.load(folder / f"{args.dict}.codes.npz")
    peak = {f["id"]: float(peaks.of(f["id"])[1].max()) for f in features}

    # the dictionary on the page's weights: it was trained on the checkpoint's activations
    on_page = S.evaluate(model, [sae], heldout, harvest, n_windows=args.windows, n_splice_windows=min(512, args.windows),
                         model_batch=64, device=args.device)[0]
    print(f"the dictionary on the page's weights: fvu {on_page['fvu']:.4f} (on the checkpoint {codes['fvu']:.4f}), "
          f"loss {on_page['loss_clean']:.4f} -> {on_page['loss_spliced']:.4f}", flush=True)
    loss_clean = heldout_loss(model, sae, heldout, harvest, args.windows, features[0]["id"], None)
    seeds = list(range(args.stories))
    plain = {}
    for o, opening in enumerate(OPENINGS[:1]):
        prompt = tok.encode(opening, bos=True)
        tokens, _ = write(model, sae, args.layer, prompt, seeds, STEPS, TEMPERATURE, features[0]["id"], None)
        ends = story_ends(tokens, len(prompt), tok.bos_id)
        plain[o] = {"tokens": tokens, "ends": ends, "texts": [tok.decode(r[len(prompt):e]) for r, e in zip(tokens.tolist(), ends)],
                    "surprise": surprise(model, tokens, len(prompt), ends)}
    print(f"as trained: loss {loss_clean:.4f}, surprise of its own stories {plain[0]['surprise']:.3f}; {args.stories} stories, "
          f"{args.windows} windows", flush=True)
    out_features = []
    for f in features:
        rows = measure(model, sae, tok, args.layer, f, peak[f["id"]], OPENINGS[0], seeds, heldout, harvest, args.windows, plain[0],
                       say=lambda s: print(s, flush=True))
        out_features.append({"id": f["id"], "name": f["name"], "note": f.get("note", ""), "words": f["words"], "peak": peak[f["id"]],
                             "strengths": [{k: v for k, v in r.items() if k != "sample"} for r in rows]})

    # the default view, recorded: the two stories and the latent's reading along each
    fi, mi, oi, seed = (int(v) for v in args.default.split(":"))
    f0, value = features[fi], MULTIPLES[mi] * peak[features[fi]["id"]]
    prompt = tok.encode(OPENINGS[oi], bos=True)
    recorded = {}
    for key, v in (("plain", None), ("clamped", value)):
        tokens, reading = write(model, sae, args.layer, prompt, [seed], STEPS, TEMPERATURE, f0["id"], v)
        end = story_ends(tokens, len(prompt), tok.bos_id)[0]
        recorded[key] = {"tokens": tokens[0, :end].tolist(), "text": [tok.decode([t]) for t in tokens[0, :end].tolist()],
                         "reading": [round(float(r), 3) for r in reading[0, :end]]}
    blob = Blob(NAME, Path(args.out))
    blob.put(f"{NAME}.encoder", sae.encoder.detach().cpu().numpy().astype(np.float16), "the dictionary's encoder, a row a latent")
    blob.put(f"{NAME}.enc_bias", sae.enc_bias.detach().cpu().numpy().astype(np.float32), "the encoder's bias, a number a latent")
    blob.put(f"{NAME}.bias", sae.bias.detach().cpu().numpy().astype(np.float32), "the dictionary's bias, subtracted before encoding (normalized units)")
    blob.put(f"{NAME}.mean", sae.mean.cpu().numpy().astype(np.float32), "the mean of the stream, subtracted first (the stream's units)")
    blob.put(f"{NAME}.rows", sae.dictionary.detach()[[f["id"] for f in features]].cpu().numpy().astype(np.float32),
             "the unit directions of the listed features, in the order of the list")
    size = blob.save()
    doc = {
        "model": run_facts(run_dir, args.layer, record), "weights": "pico", "depth": args.layer + 1,
        "dict": {"name": args.dict, "n": sae.cfg.n, "k": sae.cfg.k, "d": sae.cfg.d, "scale": float(sae.scale),
                 "fvu": codes["fvu"], "fvu_page": on_page["fvu"], "loss_page": on_page["loss_clean"], "loss_spliced_page": on_page["loss_spliced"]},
        "sampling": {"temperature": TEMPERATURE, "steps": STEPS}, "multiples": MULTIPLES,
        "openings": [{"text": o, "tokens": tok.encode(o, bos=True)} for o in OPENINGS],
        "stories": args.stories, "windows": args.windows, "measured_opening": 0, "loss_clean": loss_clean, "surprise_plain": plain[0]["surprise"],
        "features": out_features, "default": {"feature": fi, "strength": mi, "opening": oi, "seed": seed}, "recorded": recorded,
    }
    js = write_json(Path(args.out) / f"{NAME}.json", doc)
    print(f"{NAME}: {len(features)} features, {len(MULTIPLES)} strengths; json {js / 1e3:.0f} kB, bin {size / 1e6:.2f} MB")
    print("recorded, as trained:", repr(tok.decode(recorded["plain"]["tokens"][1:])))
    print("recorded, clamped:   ", repr(tok.decode(recorded["clamped"]["tokens"][1:])))


if __name__ == "__main__":
    main()
