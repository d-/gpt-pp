"""Sampling from the model: temperature, top-k, top-p, min-p; decoding with the KV cache.

    out = model.generate(idx, max_new_tokens=200, temperature=0.8, top_k=50, seed=0)       # (B, T + 200) token ids

    python -m gptpp.generate --run <name or directory> --prompt "Once upon a time" --max-new-tokens 200 \
        [--temperature 0.8] [--top-k 50] [--top-p 0.95] [--min-p 0.05] [--seed 0] [--num-samples 3]

The model gives a distribution over the next token; sampling decides how much of its tail to trust.

    temperature   divides the logits, as in a Boltzmann distribution: below 1 sharpens the distribution,
                  above 1 flattens it, 0 takes the most likely token.
    top-k         keeps the k most likely tokens (Fan, Lewis and Dauphin, 2018, arXiv:1805.04833).
    top-p         keeps the smallest set of most likely tokens whose probability reaches p, the "nucleus"
                  (Holtzman et al., 2019, arXiv:1904.09751): the set adapts to how sure the model is.
    min-p         keeps the tokens whose probability is at least min_p times that of the most likely one
                  (Nguyen et al., 2024, arXiv:2407.01082): a cut-off relative to the peak.

The three truncations are computed on the same temperature-scaled distribution and a token must pass all of
those that are set; what is kept is renormalized.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor

from . import work_dir
from .model import GPT, KVCache


def truncate(probs: Tensor, top_k: Optional[int] = None, top_p: Optional[float] = None,
             min_p: Optional[float] = None) -> Tensor:
    """Zero the tail of the distributions probs (B, V) by the rules that are set, and renormalize. The most
    likely token always survives."""
    keep = torch.ones_like(probs, dtype=torch.bool)
    if top_k is not None and top_k < probs.size(-1):
        kth_largest = probs.topk(top_k, dim=-1).values[:, -1:]
        keep &= probs >= kth_largest
    if top_p is not None and top_p < 1.0:
        ordered, order = probs.sort(dim=-1, descending=True)
        mass_before = ordered.cumsum(dim=-1) - ordered  # probability of the tokens more likely than each one
        keep &= torch.zeros_like(keep).scatter(-1, order, mass_before < top_p)
    if min_p is not None and min_p > 0.0:
        keep &= probs >= min_p * probs.max(dim=-1, keepdim=True).values
    probs = probs * keep
    return probs / probs.sum(dim=-1, keepdim=True)


def sample(logits: Tensor, temperature: float = 1.0, top_k: Optional[int] = None, top_p: Optional[float] = None,
           min_p: Optional[float] = None, generator: Optional[torch.Generator] = None) -> Tensor:
    """One token id per row of logits (B, V), returned as (B, 1)."""
    if temperature == 0.0:
        return logits.argmax(dim=-1, keepdim=True)
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    return torch.multinomial(truncate(probs, top_k, top_p, min_p), num_samples=1, generator=generator)


@torch.no_grad()
def generate(model: GPT, idx: Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None,
             top_p: Optional[float] = None, min_p: Optional[float] = None, seed: Optional[int] = None) -> Tensor:
    """Continue idx (B, T) by max_new_tokens tokens; returns (B, T + max_new_tokens).

    The prompt is run once and fills the KV cache; after that each step feeds one token and reuses the keys and
    values of all the earlier ones. The context never exceeds the length the model was trained at: when the
    cache is full, it is rebuilt from the most recent half of the window, so that every token is predicted from
    at least block_size / 2 tokens and all cached keys were computed in the window they are used in.
    """
    was_training = model.training
    model.eval()
    generator = None
    if seed is not None:
        generator = torch.Generator(device=idx.device).manual_seed(seed)
    window = model.cfg.block_size
    kv = KVCache(model.cfg.n_layer)
    logits = model.forward_explicit(idx[:, -window:], kv=kv)[:, -1]
    for step in range(max_new_tokens):
        token = sample(logits, temperature, top_k, top_p, min_p, generator)
        idx = torch.cat((idx, token), dim=1)
        if step == max_new_tokens - 1:
            break
        if kv.length < window:
            logits = model.forward_explicit(token, kv=kv)[:, -1]
        else:
            kv = KVCache(model.cfg.n_layer)
            logits = model.forward_explicit(idx[:, -max(1, window // 2):], kv=kv)[:, -1]
    model.train(was_training)
    return idx


# ----------------------------------------------------------------------------------------------------------------
# Tokenizers by name, and the command line


def load_tokenizer(name: str):
    """A tokenizer by name: `bytes` is the byte-level one of the test fixture; any other name is looked up by
    the production tokenizer under $GPTPP_WORK/data/tokenizers. Returns None if it cannot be loaded."""
    if name == "bytes":
        from .data import ByteTokenizer

        return ByteTokenizer()
    try:
        from .tokenizer import Tokenizer

        return Tokenizer.load(name)
    except (ImportError, OSError, ValueError, KeyError):
        return None


def run_tokenizer_name(run_dir: Path) -> Optional[str]:
    """The name of the tokenizer a run was trained with, from the data meta in its config.json."""
    config = run_dir / "config.json"
    if not config.exists():
        return None
    return json.loads(config.read_text()).get("data_meta", {}).get("tokenizer")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Sample text from a trained run.")
    parser.add_argument("--run", required=True, help="run name under $GPTPP_WORK/ckpt, or a run directory")
    parser.add_argument("--ckpt", default="model.pt", help="checkpoint file inside the run (default model.pt)")
    parser.add_argument("--prompt", default="", help="text to continue; empty starts from the BOS token")
    parser.add_argument("--ids", default=None, help="prompt as comma-separated token ids, instead of --prompt")
    parser.add_argument("--tokenizer", default=None, help="tokenizer name (default: the one the run trained with)")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--min-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    run_dir = Path(args.run) if Path(args.run).is_dir() else work_dir() / "ckpt" / args.run
    model, cfg = GPT.from_checkpoint(run_dir / args.ckpt, device=args.device)
    name = args.tokenizer or run_tokenizer_name(run_dir)
    tokenizer = load_tokenizer(name) if name else None

    if args.ids is not None:
        prompt = [int(i) for i in args.ids.split(",")]
    elif tokenizer is not None:
        prompt = tokenizer.encode(args.prompt, bos=True)
    else:
        raise SystemExit(f"no tokenizer could be loaded ({name!r}): pass --tokenizer, or the prompt as --ids")

    idx = torch.tensor([prompt] * args.num_samples, dtype=torch.long, device=args.device)
    use_bf16 = torch.device(args.device).type == "cuda"
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
        out = generate(model, idx, args.max_new_tokens, args.temperature, args.top_k, args.top_p, args.min_p, args.seed)
    for row in out.tolist():
        print("-" * 80)
        print(tokenizer.decode(row) if tokenizer is not None else ",".join(map(str, row)))


if __name__ == "__main__":
    main()
