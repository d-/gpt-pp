"""Prepare TinyStories: download, shuffle, tokenize, write the token shards.

    python scripts/train_tokenizer.py --name bpe4k --vocab-size 4096 \
        --input $GPTPP_WORK/data/raw/tinystories/TinyStoriesV2-GPT4-train.txt --separator "<|endoftext|>"
    python scripts/prepare_tinystories.py --tokenizer bpe4k

(Run this script once with `--download-only` to fetch the files the tokenizer is trained on.)

TinyStories (Eldan and Li, "TinyStories: how small can language models be and still speak coherent
English?", 2023) is a set of short stories in the vocabulary of a young child, written by GPT-3.5
and GPT-4. We use the second version, written by GPT-4 alone: two text files in which the stories
are separated by `<|endoftext|>`.

What is written, under $GPTPP_WORK/data/tinystories/ (Contract 1 of the lab):

    <tokenizer>/train_0000.bin ...   uint16 little-endian token ids, about 100M tokens a shard
    <tokenizer>/val.bin              the validation split
    <tokenizer>/meta.json            counts, the shard list, bytes per token, source and license
    val_docs.jsonl                   the validation stories as text, in the order of val.bin

Every story is stripped of surrounding whitespace and begins with `<|bos|>`. Training stories whose
text also occurs in the validation split are dropped, so the two sets are disjoint; each set is then
put in an order drawn from a seeded generator. A shard holds whole stories.
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing
import os
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gptpp import work_dir  # noqa: E402
from gptpp.tokenizer import Tokenizer  # noqa: E402

REPO = "roneneldan/TinyStories"
REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
FILES = {"train": "TinyStoriesV2-GPT4-train.txt", "val": "TinyStoriesV2-GPT4-valid.txt"}
SEPARATOR = "<|endoftext|>"
LICENSE = "CDLA-Sharing-1.0"
WORK = work_dir()  # $GPTPP_WORK


def download(raw: Path) -> None:
    """Fetch the two text files (and the dataset card) at the pinned revision."""
    from huggingface_hub import hf_hub_download

    for name in [*FILES.values(), "README.md"]:
        if not (raw / name).exists():
            hf_hub_download(REPO, name, repo_type="dataset", revision=REVISION, local_dir=raw)


def read_stories(path: Path) -> list[str]:
    stories = (story.strip() for story in path.read_text(encoding="utf-8").split(SEPARATOR))
    return [story for story in stories if story]


# -- tokenizing in worker processes ----------------------------------------------------------------

_tok: Tokenizer | None = None


def _init_worker(tokenizer: str) -> None:
    global _tok
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # one core for each worker process
    _tok = Tokenizer.load(tokenizer)


def _encode(texts: list[str]) -> tuple[np.ndarray, int, int]:
    """A batch of documents as one array of ids, each document begun by <|bos|>; documents; bytes."""
    ids = _tok.encode_batch(texts, bos=True)
    tokens = np.fromiter(itertools.chain.from_iterable(ids), dtype="<u2", count=sum(map(len, ids)))
    return tokens, len(texts), sum(len(text.encode("utf-8")) for text in texts)


def tokenize(docs: list[str], tokenizer: str, workers: int, batch: int = 2000) -> Iterator[tuple[np.ndarray, int, int]]:
    """Encode the documents in order, a batch at a time, on `workers` processes."""
    batches = (docs[i : i + batch] for i in range(0, len(docs), batch))
    with multiprocessing.get_context("fork").Pool(workers, _init_worker, (tokenizer,)) as pool:
        yield from pool.imap(_encode, batches)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokenizer", default="bpe4k")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard-tokens", type=float, default=100e6, help="a shard is closed once it holds this many")
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 64))
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()

    start = time.time()
    raw = WORK / "data" / "raw" / "tinystories"
    out = WORK / "data" / "tinystories" / args.tokenizer
    download(raw)
    if args.download_only:
        return
    out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.load(args.tokenizer)

    # Read, make the two sets disjoint, shuffle.
    train, val = read_stories(raw / FILES["train"]), read_stories(raw / FILES["val"])
    n_train_read, held_out = len(train), set(val)
    train = [story for story in train if story not in held_out]
    rng = np.random.default_rng(args.seed)
    train = [train[i] for i in rng.permutation(len(train))]
    val = [val[i] for i in rng.permutation(len(val))]
    print(f"{n_train_read:,} training stories, {n_train_read - len(train):,} dropped because they are also in "
          f"validation; {len(val):,} validation stories ({time.time() - start:.0f} s)", flush=True)

    # Training shards: whole batches of stories until a shard has `shard_tokens` tokens.
    shards: list[str] = []
    shard_tokens: list[int] = []
    pending: list[np.ndarray] = []
    train_docs = train_bytes = 0

    def close_shard() -> None:
        tokens = np.concatenate(pending)
        name = f"train_{len(shards):04d}.bin"
        tokens.tofile(out / name)
        shards.append(name)
        shard_tokens.append(len(tokens))
        pending.clear()
        print(f"  {name}: {len(tokens):,} tokens ({time.time() - start:.0f} s)", flush=True)

    for tokens, n_docs, n_bytes in tokenize(train, args.tokenizer, args.workers):
        pending.append(tokens)
        train_docs += n_docs
        train_bytes += n_bytes
        if sum(map(len, pending)) >= args.shard_tokens:
            close_shard()
    if pending:
        close_shard()
    for stale in sorted(out.glob("train_*.bin"))[len(shards) :]:  # shards of an earlier, longer run
        stale.unlink()

    # Validation: one file of tokens, and the same stories as text for models with other tokenizers.
    parts = list(tokenize(val, args.tokenizer, args.workers))
    val_tokens = np.concatenate([tokens for tokens, _, _ in parts])
    val_bytes = sum(n_bytes for _, _, n_bytes in parts)
    val_tokens.tofile(out / "val.bin")
    with open(out.parent / "val_docs.jsonl", "w", encoding="utf-8") as f:
        for story in val:
            f.write(json.dumps({"text": story}, ensure_ascii=False) + "\n")

    n_train = sum(shard_tokens)
    meta = {
        "dataset": "tinystories",
        "tokenizer": args.tokenizer,
        "vocab_size": tok.vocab_size,
        "bos_id": tok.bos_id,
        "dtype": "uint16 little-endian",
        "n_train_tokens": n_train,
        "n_val_tokens": len(val_tokens),
        "train_shards": shards,
        "train_shard_tokens": shard_tokens,
        "n_train_docs": train_docs,
        "train_bytes": train_bytes,
        "bytes_per_token_train": train_bytes / n_train,
        "val_file": "val.bin",
        "val_bytes": val_bytes,
        "val_docs": len(val),
        "val_docs_file": "../val_docs.jsonl",
        "bytes_per_token_val": val_bytes / len(val_tokens),
        "bytes_per_text_token_val": val_bytes / (len(val_tokens) - len(val)),
        "note": "token counts include one <|bos|> a document; bytes are the UTF-8 bytes of the text; "
        "bytes_per_token_val = val_bytes / n_val_tokens, bytes_per_text_token_val leaves the <|bos|> tokens out",
        "seed": args.seed,
        "n_train_docs_dropped_also_in_val": n_train_read - len(train),
        "source": f"https://huggingface.co/datasets/{REPO} at revision {REVISION}: {FILES['train']}, {FILES['val']}",
        "license": LICENSE,
        "seconds": round(time.time() - start, 1),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"tinystories/{args.tokenizer}: {n_train:,} training tokens in {len(shards)} shards, "
          f"{len(val_tokens):,} validation tokens, {meta['bytes_per_token_val']:.3f} bytes per token, "
          f"{time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
