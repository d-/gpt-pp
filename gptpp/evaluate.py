"""Bits per byte of any language model on held-out text.

A language model gives a text a probability, and minus the base-two logarithm of that probability is the number
of bits an arithmetic coder driven by the model needs to write the text down (Shannon, "A mathematical theory of
communication", 1948). Divided by the length of the text in UTF-8 bytes this is *bits per byte*: a number that
does not depend on how the model cuts text into tokens, so models with different vocabularies can be compared on
the same documents. A model that knows nothing about bytes pays 8 bits for each; English prose compresses to
about one.

The protocol, the same for every model (our checkpoints and Hugging Face models go through one code path):

1. Each document is encoded by the model's own tokenizer, with no special tokens added.
2. The tokens are cut into consecutive windows. A window is the model's beginning-of-text token followed by at
   most ``window - 1`` tokens of the document, where ``window = min(the model's context, 1024)``.
3. The model scores every token after the first: ``-log p(token | beginning-of-text, earlier tokens of the window)``.
   Every token of the document is scored exactly once, so the total is the length of a real code for the document.
4. ``bits per byte = (sum of the scores in bits over all documents) / (sum of the documents' UTF-8 bytes)``.

What is excluded: nothing of the text. The beginning-of-text token is given, not scored; no end-of-text token is
appended or scored, because models differ in whether they were trained to predict one. Windows do not overlap,
so the first tokens of every window are predicted from little context; `by_position` in the result shows what
that costs. Documents with no text are dropped. The log-probabilities are computed in float32 and summed in
float64; `load_model` says how the model itself computes.

    python -m gptpp.evaluate --model hf:gpt2 --docs wikitext103,tinystories --out results/eval/gpt2.json
    python -m gptpp.evaluate --model $GPTPP_WORK/ckpt/<run> --docs fineweb-edu --out results/eval/<run>.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from gptpp import work_dir

MAX_WINDOW = 1024  # tokens in a window, the beginning-of-text token included
LN2 = math.log(2.0)

# ---------------------------------------------------------------------------------------------------------------
# Models: the three things the evaluation needs from one (a tokenizer, a beginning-of-text token, logits).
# ---------------------------------------------------------------------------------------------------------------


class LanguageModel:
    """A causal language model as the evaluation sees it.

    Subclasses set `name`, `context` (tokens the model can attend over), `bos_id`, `vocab_size`, `device` and
    `info` (what is recorded about the model), and implement `encode`, `decode` and `logits`.
    """

    name: str
    context: int
    bos_id: int
    vocab_size: int
    device: torch.device
    info: dict
    autocast: bool = False  # run the trunk in bfloat16; the scores are always computed in float32

    @property
    def window(self) -> int:
        """Tokens in an evaluation window, the beginning-of-text token included."""
        return min(self.context, MAX_WINDOW)

    def encode(self, texts: Sequence[str]) -> list[list[int]]:
        """Token ids of each text, without special tokens."""
        raise NotImplementedError

    def decode(self, ids: Sequence[int]) -> str:
        raise NotImplementedError

    def logits(self, idx: torch.Tensor) -> torch.Tensor:
        """(B, T) token ids -> (B, T, vocab) next-token logits. Position t sees tokens 0..t only."""
        raise NotImplementedError


# Short names people use for the 2019 checkpoints -> where they live on the Hub today.
HF_ALIASES = {
    "gpt2": "openai-community/gpt2",
    "gpt2-medium": "openai-community/gpt2-medium",
    "gpt2-large": "openai-community/gpt2-large",
    "gpt2-xl": "openai-community/gpt2-xl",
    "distilgpt2": "distilbert/distilgpt2",
}


class HFModel(LanguageModel):
    """A causal language model from the Hugging Face Hub: ``hf:<repo>`` or ``hf:<repo>@<revision>``."""

    def __init__(self, repo: str, device: torch.device | str = "cuda", context: int | None = None):
        from huggingface_hub import hf_hub_download
        from transformers import AutoModelForCausalLM, AutoTokenizer

        repo, _, revision = repo.partition("@")
        hub_id = HF_ALIASES.get(repo, repo)
        # the commit that is actually loaded: the cache keeps each one in a folder named after it
        commit = Path(hf_hub_download(hub_id, "config.json", revision=revision or None)).parent.name
        self.tokenizer = AutoTokenizer.from_pretrained(hub_id, revision=revision or None)
        self.tokenizer.model_max_length = int(1e30)  # we cut the windows ourselves
        model = AutoModelForCausalLM.from_pretrained(hub_id, revision=revision or None, dtype=torch.float32)
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        cfg = model.config

        # The beginning-of-text token. Models without one (Qwen) separated their training documents with the
        # end-of-text token, which then plays the same part: "a new document starts here".
        tok = self.tokenizer
        if tok.bos_token_id is not None:
            self.bos_id, bos_source = tok.bos_token_id, "bos_token"
        elif tok.eos_token_id is not None:
            self.bos_id, bos_source = tok.eos_token_id, "eos_token (the tokenizer defines no bos_token)"
        else:
            raise ValueError(f"{hub_id}: the tokenizer has neither a beginning- nor an end-of-text token")

        positions = getattr(cfg, "n_positions", None) or getattr(cfg, "max_position_embeddings", None)
        self.context = int(context or positions or MAX_WINDOW)
        self.vocab_size = int(model.get_input_embeddings().weight.shape[0])
        self.name = f"hf:{hub_id}"

        emb_in, emb_out = model.get_input_embeddings().weight, model.get_output_embeddings().weight
        n_params = sum(p.numel() for p in model.parameters())  # tied matrices are one Parameter, counted once
        tied = emb_in is emb_out
        self.info = {
            "name": self.name,
            "kind": "hf",
            "hub_id": hub_id,
            "revision": commit,
            "architecture": cfg.model_type,
            "n_layer": _first_attr(cfg, "n_layer", "num_hidden_layers", "num_layers"),
            "d_model": _first_attr(cfg, "n_embd", "hidden_size"),
            "n_head": _first_attr(cfg, "n_head", "num_attention_heads", "num_heads"),
            "n_params": n_params,
            "n_params_non_embedding": n_params - emb_in.numel() - (0 if tied else emb_out.numel()),
            "tied_embeddings": tied,
            "vocab_size": self.vocab_size,
            "context": self.context,
            "context_positions_in_config": positions,
            "bos_id": self.bos_id,
            "bos_token": tok.convert_ids_to_tokens(self.bos_id),
            "bos_source": bos_source,
        }

    def encode(self, texts: Sequence[str]) -> list[list[int]]:
        return self.tokenizer(list(texts), add_special_tokens=False)["input_ids"]

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def logits(self, idx: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=idx, use_cache=False).logits  # no key-value cache: nothing is generated


def _first_attr(cfg: object, *names: str) -> int | None:
    """Configurations name the same thing differently (n_layer, num_hidden_layers, ...): the first one present."""
    for name in names:
        if getattr(cfg, name, None) is not None:
            return int(getattr(cfg, name))
    return None


class OurModel(LanguageModel):
    """A checkpoint written by ``gptpp.train``: a run directory (its ``model.pt``) or a checkpoint file."""

    def __init__(self, path: str | Path, device: torch.device | str = "cuda", tokenizer: str | None = None):
        from gptpp.model import GPT
        from gptpp.tokenizer import Tokenizer

        path = Path(path)
        ckpt = path / "model.pt" if path.is_dir() else path
        run_dir = ckpt.parent
        self.model, cfg = GPT.from_checkpoint(str(ckpt), device=str(device))
        self.model.eval()
        self.device = torch.device(device)

        run_config = json.loads((run_dir / "config.json").read_text()) if (run_dir / "config.json").exists() else {}
        tok_name = tokenizer or _find_tokenizer_name(run_config) or _TOKENIZER_BY_VOCAB.get(cfg.vocab_size)
        if tok_name is None:
            raise ValueError(f"{path}: cannot tell which tokenizer the run used; pass --tokenizer")
        self.tokenizer = Tokenizer.load(tok_name)
        if self.tokenizer.vocab_size != cfg.vocab_size:
            raise ValueError(f"tokenizer {tok_name} has {self.tokenizer.vocab_size} tokens, the model {cfg.vocab_size}")

        self.bos_id = int(self.tokenizer.bos_id)
        self.context = int(cfg.block_size)
        self.vocab_size = int(cfg.vocab_size)
        self.name = run_dir.name if ckpt.name == "model.pt" else f"{run_dir.name}/{ckpt.stem}"
        params = self.model.num_params()
        self.info = {
            "name": self.name,
            "kind": "gptpp",
            "checkpoint": f"{run_dir.name}/{ckpt.name}",
            "tokenizer": tok_name,
            "model_config": cfg.to_dict() if hasattr(cfg, "to_dict") else dict(vars(cfg)),
            "n_layer": int(cfg.n_layer),
            "d_model": int(cfg.d_model),
            "n_head": int(cfg.n_head),
            "n_params": int(params["total"]),
            "n_params_by_part": {k: int(v) for k, v in params.items()},
            "vocab_size": self.vocab_size,
            "context": self.context,
            "bos_id": self.bos_id,
            "bos_token": "<|bos|>",
            "bos_source": "bos_token",
        }

    def encode(self, texts: Sequence[str]) -> list[list[int]]:
        return self.tokenizer.encode_batch(list(texts), bos=False)

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(ids))

    def logits(self, idx: torch.Tensor) -> torch.Tensor:
        return self.model(idx)


_TOKENIZER_BY_VOCAB = {32768: "bpe32k", 4096: "bpe4k"}


def _find_tokenizer_name(obj: object) -> str | None:
    """The first string stored under a key named "tokenizer" anywhere in a run's config.json."""
    if isinstance(obj, dict):
        if isinstance(obj.get("tokenizer"), str):
            return obj["tokenizer"]
        obj = list(obj.values())
    if isinstance(obj, list):
        for value in obj:
            found = _find_tokenizer_name(value)
            if found:
                return found
    return None


def load_model(spec: str, device: str | None = None, precision: str = "tf32", tokenizer: str | None = None,
               context: int | None = None) -> LanguageModel:
    """``hf:<repo>[@revision]`` for a Hub model, anything else is a run directory or checkpoint of ours.

    `precision` is how the trunk computes: ``float32`` (exact), ``tf32`` (the default: float32 weights and
    activations with TensorFloat-32 matrix products, which round the factors to ten bits of mantissa and are
    about twice as fast on the GPU) or ``bf16`` (bfloat16 autocast, the way our models train). The
    log-probabilities are computed in float32 in every case. On WikiText-103, GPT-2 small scores 1.0719 bits per
    byte in float32, 1.0720 in tf32 and 1.0751 in bf16: tf32 is exact to the digits we report, bf16 is not for a
    model that was not trained in it.
    """
    if precision not in ("float32", "tf32", "bf16"):
        raise ValueError(f"precision must be float32, tf32 or bf16, not {precision!r}")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = precision != "float32"
    torch.backends.cudnn.allow_tf32 = precision != "float32"
    if spec.startswith("hf:"):
        model: LanguageModel = HFModel(spec[3:], device=device, context=context)
    else:
        model = OurModel(spec, device=device, tokenizer=tokenizer)
        if context:
            model.context = int(context)
            model.info["context"] = int(context)
    model.autocast = precision == "bf16"
    model.info["precision"] = precision
    model.info["window"] = model.window
    return model


# ---------------------------------------------------------------------------------------------------------------
# The scoring primitive, shared with benchmarks.py.
# ---------------------------------------------------------------------------------------------------------------


@dataclass
class Scores:
    """What `score_sequences` returns, one entry a sequence (in the order given)."""

    nll: np.ndarray  # float64, sum over the scored tokens of -log p(token | earlier tokens), in nats
    n_scored: np.ndarray  # int64, how many tokens were scored
    greedy: np.ndarray | None  # bool, every scored token was the model's most probable next token (on request)
    by_position: tuple[np.ndarray, np.ndarray]  # (sum of nll, number of tokens) at each position of the sequence


def batch_token_budget(model: LanguageModel, longest: int) -> int:
    """Tokens in a batch: bounded so that the (tokens, vocab) float32 logits stay near 2.4 GB."""
    return max(longest, min(65536, int(0.6e9 / model.vocab_size)))


@torch.no_grad()
def score_sequences(model: LanguageModel, sequences: Sequence[Sequence[int]], starts: Sequence[int] | int = 1,
                    batch_tokens: int | None = None, want_greedy: bool = False,
                    progress: str | None = None) -> Scores:
    """Score token sequences under the model.

    Sequence i is scored from position ``starts[i]`` (at least 1: position 0 is the given first token) to its
    end: the sum of ``-log p(x_t | x_0 .. x_{t-1})``. Sequences are sorted by length and padded on the right; in
    a causal model the padding lies in the future of every real token and cannot change its prediction, and the
    padded positions are left out of the sums.
    """
    n = len(sequences)
    starts = [starts] * n if isinstance(starts, int) else list(starts)
    lengths = np.array([len(s) for s in sequences], dtype=np.int64)
    if n and (lengths.max() > model.context or min(starts) < 1):
        raise ValueError("a sequence is longer than the model's context, or a start is below 1")
    longest = int(lengths.max()) if n else 1
    budget = batch_tokens or batch_token_budget(model, longest)

    nll = np.zeros(n, dtype=np.float64)
    n_scored = np.zeros(n, dtype=np.int64)
    greedy = np.zeros(n, dtype=bool) if want_greedy else None
    pos_sum = torch.zeros(longest, dtype=torch.float64, device=model.device)
    pos_cnt = torch.zeros(longest, dtype=torch.float64, device=model.device)

    order = np.argsort(-lengths, kind="stable")
    done, t0, last_report = 0, time.time(), time.time()
    while done < n:
        width = int(lengths[order[done]])  # the longest sequence of this batch
        rows = order[done:done + max(1, budget // max(width, 1))]
        idx = torch.full((len(rows), width), model.bos_id, dtype=torch.long)
        for r, i in enumerate(rows):
            idx[r, :lengths[i]] = torch.as_tensor(sequences[i], dtype=torch.long)
        idx = idx.to(model.device)
        length = torch.as_tensor(lengths[rows], device=model.device).unsqueeze(1)
        start = torch.as_tensor([starts[i] for i in rows], device=model.device).unsqueeze(1)

        with torch.autocast(model.device.type, dtype=torch.bfloat16, enabled=model.autocast):
            logits = model.logits(idx)  # (B, width, vocab); the logits at place t predict the token at t + 1

        predicted = torch.arange(1, width + 1, device=model.device).unsqueeze(0)  # position of the predicted token
        scored = (predicted >= start) & (predicted < length)
        next_token = torch.roll(idx, shifts=-1, dims=1)
        targets = torch.where(scored, next_token, -100)  # -100: cross_entropy returns 0 there
        token_nll = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                                                      targets.reshape(-1), reduction="none")
        token_nll = token_nll.view(len(rows), width).double()

        nll[rows] = token_nll.sum(dim=1).cpu().numpy()
        n_scored[rows] = scored.sum(dim=1).cpu().numpy()
        if want_greedy:
            greedy[rows] = ((logits.argmax(dim=-1) == next_token) | ~scored).all(dim=1).cpu().numpy()
        pos_sum[1:width] += token_nll[:, :width - 1].sum(dim=0)
        pos_cnt[1:width] += scored[:, :width - 1].sum(dim=0)

        done += len(rows)
        if progress and time.time() - last_report > 30:
            last_report = time.time()
            print(f"    {progress}: {done}/{n} sequences, {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    return Scores(nll, n_scored, greedy, (pos_sum.cpu().numpy(), pos_cnt.cpu().numpy()))


def padding_error(model: LanguageModel, lengths: Sequence[int] = (6, 19, 40, 64), seed: int = 0) -> float:
    """The largest change of a sequence's score, in nats per token, between scoring it alone and in a padded batch.

    The protocol pads on the right and relies on causality: nothing after a token can change its prediction.
    This measures it for a given model on a few random sequences, in exact float32 (reduced precision is switched
    off for the moment: its rounding depends on the shape of the batch and would hide what is measured). The
    answer should be float32 rounding, about 1e-5 on sequences that cost 10 nats a token.
    """
    rng = np.random.default_rng(seed)
    lengths = [min(n, model.context) for n in lengths]
    sequences = [[model.bos_id] + rng.integers(0, model.vocab_size, n - 1).tolist() for n in lengths]
    reduced = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, model.autocast
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = model.autocast = False
    try:
        together = score_sequences(model, sequences)
        alone = np.array([score_sequences(model, [s]).nll[0] for s in sequences])
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, model.autocast = reduced
    return float(np.max(np.abs(together.nll - alone) / together.n_scored))


# ---------------------------------------------------------------------------------------------------------------
# Held-out documents.
# ---------------------------------------------------------------------------------------------------------------

# name -> where the documents come from. A Hub file is pinned to a commit. The local files are written by
# scripts/prepare_fineweb.py and scripts/prepare_tinystories.py (the documents held out of our own training
# sets, in a shuffled order) and by scripts/make_fresh_corpus.py.
HELD_OUT: dict[str, dict] = {
    "wikitext103": {
        "what": "WikiText-103, test split, raw text as distributed (Merity et al. 2016): 60 Wikipedia articles",
        "hub": ("Salesforce/wikitext", "b08601e04326c79dfdd32d625aee71d232d685c3",
                "wikitext-103-raw-v1/test-00000-of-00001.parquet"),
        "max_docs": None,
    },
    "tinystories": {
        "what": "TinyStories (Eldan and Li 2023), the validation stories of its second, GPT-4-written version",
        "local": "data/tinystories/val_docs.jsonl",
        "max_docs": 2000,
    },
    "fineweb-edu": {
        "what": "FineWeb-EDU (Penedo et al. 2024), the documents held out of our training set",
        "local": "data/fineweb-edu/val_docs.jsonl",
        "max_docs": 1000,
    },
    "fresh": {
        "what": "prose published in October and November 2026, after every released model's training data",
        "local": "data/fresh/val_docs.jsonl",
        "max_docs": None,
    },
}


def hub_parquet(repo: str, revision: str, filename: str) -> list[dict]:
    """The rows of one parquet file of a Hub dataset, pinned to a commit."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo, filename, repo_type="dataset", revision=revision)
    return pq.read_table(path).to_pylist()


def wikitext_articles(rows: Sequence[str]) -> list[str]:
    """Join the rows of a WikiText split (one line of the source file each) back into its articles.

    An article begins at a top-level heading, a line `` = Title = `` between two empty rows. (A formula such as
    `` = 1 henry and capacitance C = `` also has that shape; it is not between empty rows.)
    """
    def is_title(i: int) -> bool:
        line = rows[i]
        return (line.startswith(" = ") and not line.startswith(" = = ") and line.endswith(" = \n")
                and (i == 0 or rows[i - 1] == "") and (i + 1 == len(rows) or rows[i + 1] == ""))

    titles = [i for i in range(len(rows)) if is_title(i)]
    return ["".join(rows[a:b]) for a, b in zip(titles, titles[1:] + [len(rows)])]


def load_docs(name: str, max_docs: int | None = None) -> tuple[list[str], dict]:
    """The documents of a held-out set (a name in `HELD_OUT`, or the path of a JSONL file with a "text" field).

    Returns the texts and a record of where they came from, with a fingerprint of the texts: two results are
    comparable only when their fingerprints are equal.
    """
    if name in HELD_OUT:
        spec = HELD_OUT[name]
        local_path = work_dir() / spec["local"] if "local" in spec else None
    else:
        spec = {"what": "a local file", "max_docs": None}
        local_path = Path(name)
    if local_path is not None and local_path.exists():
        with open(local_path, encoding="utf-8") as f:
            texts = [json.loads(line)["text"] for line in f if line.strip()]
        inside = local_path.resolve().is_relative_to(work_dir().resolve())
        source = {"file": f"$GPTPP_WORK/{local_path.resolve().relative_to(work_dir().resolve())}" if inside
                  else local_path.name}
    elif "hub" in spec:
        repo, revision, filename = spec["hub"]
        rows = [row["text"] for row in hub_parquet(repo, revision, filename)]
        texts = wikitext_articles(rows) if name == "wikitext103" else rows
        source = {"hub_id": repo, "revision": revision, "file": filename}
    else:
        raise FileNotFoundError(f"held-out set {name!r}: {local_path} does not exist")

    # Unpaired surrogates cannot be written as UTF-8; replace them so that every text has a byte length.
    texts = [t.encode("utf-8", errors="replace").decode("utf-8") for t in texts]
    texts = [t for t in texts if t.strip()]
    n_available = len(texts)
    limit = max_docs if max_docs is not None else spec.get("max_docs")
    if limit is not None:
        texts = texts[:limit]  # the first documents: the files are already in a fixed shuffled order
    digest = hashlib.sha256()
    for t in texts:
        digest.update(t.encode("utf-8"))
        digest.update(b"\x00")
    source.update({"name": name, "what": spec["what"], "n_docs": len(texts), "n_docs_available": n_available,
                   "sha256": digest.hexdigest()[:16]})
    return texts, source


# ---------------------------------------------------------------------------------------------------------------
# Bits per byte.
# ---------------------------------------------------------------------------------------------------------------


def ratio_standard_error(numerator: np.ndarray, denominator: np.ndarray) -> float | None:
    """Standard error of ``sum(numerator) / sum(denominator)`` when the documents are the sampled units.

    The ratio estimator's linearization (Cochran, "Sampling techniques", 1977, section 2.11): the residuals
    ``numerator_d - ratio * denominator_d`` of the documents carry the sampling noise.
    """
    n = len(numerator)
    if n < 2:
        return None
    ratio = numerator.sum() / denominator.sum()
    residual = numerator - ratio * denominator
    return float(math.sqrt(n / (n - 1) * float((residual ** 2).sum())) / denominator.sum())


def bits_per_byte(model: LanguageModel, docs: Sequence[str], batch_tokens: int | None = None,
                  progress: str | None = None) -> dict:
    """Bits per byte of `model` on `docs` under the protocol at the top of this file."""
    t0 = time.time()
    doc_tokens = model.encode(docs)
    step = model.window - 1  # tokens of the document in a window; the first place holds the beginning-of-text token
    windows: list[list[int]] = []
    owner: list[int] = []
    for d, ids in enumerate(doc_tokens):
        for i in range(0, len(ids), step):
            windows.append([model.bos_id] + ids[i:i + step])
            owner.append(d)

    scores = score_sequences(model, windows, starts=1, batch_tokens=batch_tokens, progress=progress)

    n_docs = len(docs)
    doc_bits = np.bincount(owner, weights=scores.nll, minlength=n_docs) / LN2
    doc_bytes = np.array([len(t.encode("utf-8")) for t in docs], dtype=np.float64)
    n_tokens = int(scores.n_scored.sum())
    n_bytes = int(doc_bytes.sum())
    assert n_tokens == sum(len(ids) for ids in doc_tokens), "every token of every document is scored exactly once"

    total_bits = float(doc_bits.sum())
    loss = total_bits * LN2 / n_tokens
    seconds = time.time() - t0
    return {
        "bits_per_byte": total_bits / n_bytes,
        "bits_per_byte_se": ratio_standard_error(doc_bits, doc_bytes),
        "loss": loss,  # nats per token
        "perplexity": math.exp(loss),
        "tokens_per_byte": n_tokens / n_bytes,
        "bytes_per_token": n_bytes / n_tokens,
        "n_docs": n_docs,
        "n_bytes": n_bytes,
        "n_tokens": n_tokens,
        "n_windows": len(windows),
        "window": model.window,
        "total_bits": total_bits,
        # documents the tokenizer does not reproduce exactly (normalization, unknown characters): 0 for a
        # byte-level tokenizer, and the bits are then a code for the documents' exact bytes
        "n_docs_not_reproduced": sum(model.decode(ids) != text for ids, text in zip(doc_tokens, docs)),
        "by_position": _by_position(*scores.by_position),
        "seconds": round(seconds, 2),
        "tokens_per_second": round(n_tokens / seconds),
        # one entry a document, for paired comparisons between models and for figures
        "doc_bits": [round(float(b), 1) for b in doc_bits],
        "doc_bytes": [int(b) for b in doc_bytes],
    }


def _by_position(nll_sum: np.ndarray, count: np.ndarray) -> list[dict]:
    """Mean bits per token as a function of the position in the window, in octaves (1, 2-3, 4-7, ...)."""
    out, lo = [], 1
    while lo < len(nll_sum):
        hi = min(2 * lo, len(nll_sum))
        tokens = float(count[lo:hi].sum())
        if tokens:
            out.append({"from": lo, "to": hi - 1, "n_tokens": int(tokens),
                        "bits_per_token": float(nll_sum[lo:hi].sum()) / tokens / LN2})
        lo = hi
    return out


# ---------------------------------------------------------------------------------------------------------------
# Results on disk: one JSON a model, filled in by this module and by benchmarks.py.
# ---------------------------------------------------------------------------------------------------------------


def environment() -> dict:
    import transformers

    return {
        "date": time.strftime("%Y-%m-%d"),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def dump_json(obj: object, level: int = 0) -> str:
    """JSON with one key a line, and lists of numbers (or of flat records) on one line each."""
    def flat(x: object) -> bool:
        return not isinstance(x, (dict, list)) or all(not isinstance(v, (dict, list)) for v in
                                                      (x.values() if isinstance(x, dict) else x))

    pad, end = " " * (level + 1), " " * level
    if isinstance(obj, dict) and obj:
        return "{\n" + ",\n".join(f"{pad}{json.dumps(k)}: {dump_json(v, level + 1)}" for k, v in obj.items()) + f"\n{end}}}"
    if isinstance(obj, list) and obj and not all(not isinstance(v, (dict, list)) for v in obj):
        return "[\n" + ",\n".join(pad + (json.dumps(v) if flat(v) else dump_json(v, level + 1)) for v in obj) + f"\n{end}]"
    return json.dumps(obj)


def merge_result(path: str | Path, model_info: dict, section: str, key: str, result: dict) -> dict:
    """Write ``result`` under ``[section][key]`` of the model's JSON file, keeping what the file already holds."""
    path = Path(path)
    record: dict = {}
    if path.exists():
        record = json.loads(path.read_text())
        if record.get("model", {}).get("name") != model_info["name"]:
            raise ValueError(f"{path} holds results of {record.get('model', {}).get('name')}, not {model_info['name']}")
    record["model"] = {**record.get("model", {}), **model_info}
    record.setdefault(section, {})[key] = {**result, "precision": model_info.get("precision"),
                                           "environment": environment()}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(dump_json(record) + "\n")
    tmp.replace(path)
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="a run directory or checkpoint of ours, or hf:<repo>[@revision]")
    ap.add_argument("--docs", required=True, help=f"comma-separated: {', '.join(HELD_OUT)}, or the path of a JSONL file")
    ap.add_argument("--max-docs", type=int, default=None, help="the first N documents (default: the set's own)")
    ap.add_argument("--out", default=None, help="JSON file to write or to add to (results/eval/<name>.json)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--precision", default="tf32", choices=["float32", "tf32", "bf16"])
    ap.add_argument("--tokenizer", default=None, help="for our checkpoints, when the run's config does not name it")
    ap.add_argument("--context", type=int, default=None, help="override the model's context length")
    ap.add_argument("--batch-tokens", type=int, default=None)
    args = ap.parse_args()

    model = load_model(args.model, device=args.device, precision=args.precision, tokenizer=args.tokenizer,
                       context=args.context)
    for name in args.docs.split(","):
        docs, source = load_docs(name, args.max_docs)
        result = bits_per_byte(model, docs, batch_tokens=args.batch_tokens, progress=f"{model.name} {name}")
        result["docs"] = source
        print(f"{model.name:40s} {name:12s} {result['bits_per_byte']:.4f} bits/byte "
              f"(se {result['bits_per_byte_se'] or 0:.4f})"
              f"  loss {result['loss']:.4f}  {result['bytes_per_token']:.3f} bytes/token  {result['n_tokens']} tokens"
              f"  {result['seconds']:.0f}s", flush=True)
        if args.out:
            merge_result(args.out, model.info, "bits_per_byte", Path(name).stem if name not in HELD_OUT else name, result)


if __name__ == "__main__":
    main()
