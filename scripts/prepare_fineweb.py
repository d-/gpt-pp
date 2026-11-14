"""Prepare FineWeb-EDU: download, hold out, train the tokenizer, tokenize, shuffle, write the shards.

    python scripts/prepare_fineweb.py --tokenizer bpe32k          # every stage, in order

FineWeb-EDU (Penedo et al., "The FineWeb datasets: decanting the web for the finest text data at
scale", 2024) is the part of the FineWeb crawl that a classifier scored as educational. We use the
subset `sample/10BT` of the release: 14 Parquet files, 9.67 million documents, about ten billion
GPT-2 tokens, a random sample of the whole.

The stages (each skips itself if its output is already there):

  download   the 14 files at a pinned revision, into $GPTPP_WORK/data/raw/fineweb-edu/
  scan       one pass over every document: characters, UTF-8 bytes, a 64-bit hash of the text and
             one of the URL, and the GPT-2 token count that the dataset ships; kept in docs.npz
  plan       one permutation of the documents from a seeded generator fixes everything else:
               - the validation set is the shortest prefix of the permutation with at least
                 `--val-tokens` tokens (whole documents);
               - the training set is the rest, in the order of the permutation, minus any document
                 that has the text or the URL of a validation document (the crawl was deduplicated
                 one dump at a time, so a page fetched in two dumps is in the dataset twice);
               - the tokenizer is trained on the last documents of the permutation, as many as it
                 takes to reach `--tokenizer-chars` characters (so on training documents only).
  tokenizer  writes those documents to tokenizer_text.parquet and runs scripts/train_tokenizer.py
  tokenize   every row group on its own core: `<|bos|>` + ids for each document, saved as arrays
  assemble   loads the arrays, walks the permutation, writes the shards (whole documents, a shard is
             closed once it holds `--shard-tokens`), val.bin, val_docs.jsonl and meta.json

Output, under $GPTPP_WORK/data/fineweb-edu/ (Contract 1 of the lab):

    <tokenizer>/train_0000.bin ...   uint16 little-endian token ids, about 100M tokens a shard
    <tokenizer>/val.bin              the held-out documents
    <tokenizer>/meta.json            counts, the shard list, bytes per token, source and license
    val_docs.jsonl                   the held-out documents as text (with id and url), in the order of val.bin
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gptpp import work_dir  # noqa: E402
from gptpp.tokenizer import Tokenizer, tokenizer_dir  # noqa: E402

REPO = "HuggingFaceFW/fineweb-edu"
REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
FILES = [f"sample/10BT/{i:03d}_00000.parquet" for i in range(14)]
LICENSE = "ODC-By 1.0"
RAW = work_dir() / "data" / "raw" / "fineweb-edu"  # work_dir() is $GPTPP_WORK
DATA = work_dir() / "data" / "fineweb-edu"

Task = tuple[int, int]  # (file, row group): a thousand documents


def log(message: str, start: float) -> None:
    print(f"[{time.time() - start:6.0f} s] {message}", flush=True)


def download() -> None:
    from huggingface_hub import hf_hub_download

    for name in [*FILES, "README.md"]:
        if not (RAW / name).exists():
            hf_hub_download(REPO, name, repo_type="dataset", revision=REVISION, local_dir=RAW)


def row_groups() -> tuple[list[Task], np.ndarray]:
    """Every (file, row group) in order, and the index of the first document of each."""
    tasks: list[Task] = []
    rows: list[int] = []
    for f, name in enumerate(FILES):
        metadata = pq.ParquetFile(RAW / name).metadata
        for g in range(metadata.num_row_groups):
            tasks.append((f, g))
            rows.append(metadata.row_group(g).num_rows)
    return tasks, np.concatenate([[0], np.cumsum(rows)])


# -- worker processes: each reads row groups on its own ----------------------------------------------

_files: dict[int, pq.ParquetFile] = {}
_tok: Tokenizer | None = None
_scratch: Path | None = None


def _init_worker(tokenizer: str | None = None, scratch: Path | None = None) -> None:
    global _tok, _scratch
    pa.set_cpu_count(1)  # one core for each worker process
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _tok = Tokenizer.load(tokenizer) if tokenizer else None
    _scratch = scratch


def _read(task: Task, columns: list[str]) -> pa.Table:
    f, g = task
    if f not in _files:
        _files[f] = pq.ParquetFile(RAW / FILES[f])
    return _files[f].read_row_group(g, columns=columns, use_threads=False)


def _hash(data: bytes) -> int:
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little")


def _scan(task: Task) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Of each document of a row group: characters, bytes, a hash of the text, a hash of the URL, GPT-2 tokens."""
    table = _read(task, ["text", "url", "token_count"])
    texts = table.column("text").to_pylist()
    encoded = [text.encode("utf-8") for text in texts]
    chars = np.array([len(text) for text in texts], dtype=np.uint32)
    n_bytes = np.array([len(e) for e in encoded], dtype=np.uint32)
    hashes = np.array([_hash(e) for e in encoded], dtype=np.uint64)
    urls = np.array([_hash(url.encode("utf-8")) for url in table.column("url").to_pylist()], dtype=np.uint64)
    return chars, n_bytes, hashes, urls, table.column("token_count").to_numpy().astype(np.uint32)


def _texts(job: tuple[Task, list[int]]) -> list[str]:
    """The texts of the chosen rows of a row group."""
    task, rows = job
    if not rows:
        return []
    texts = _read(task, ["text"]).column("text").to_pylist()
    return [texts[r] for r in rows]


def _tokenize(job: tuple[int, Task, list[int]]) -> tuple[int, int, list[dict]]:
    """Encode a row group; save ids and document lengths; return the full records of the chosen rows."""
    t, task, keep = job
    table = _read(task, ["text", "id", "url"])
    texts = table.column("text").to_pylist()
    ids = _tok.encode_batch(texts, bos=True)
    lengths = np.fromiter(map(len, ids), dtype=np.uint32, count=len(ids))
    tokens = np.fromiter(itertools.chain.from_iterable(ids), dtype="<u2", count=int(lengths.sum()))
    np.save(_scratch / f"tokens_{t:05d}.npy", tokens)
    np.save(_scratch / f"lengths_{t:05d}.npy", lengths)
    ids_of, urls_of = table.column("id"), table.column("url")
    kept = [{"row": r, "text": texts[r], "id": ids_of[r].as_py(), "url": urls_of[r].as_py()} for r in keep]
    return t, len(tokens), kept


# -- the assembly: the parent holds every token; forked workers gather and write one shard each ------

_tokens: np.ndarray  # every token of the corpus, in the order of the files
_starts: np.ndarray  # document -> index of its first token
_lengths: np.ndarray  # document -> its number of tokens, <|bos|> included


def _gather(docs: np.ndarray) -> np.ndarray:
    """The tokens of the given documents, concatenated in that order."""
    starts, lengths = _starts[docs].tolist(), _lengths[docs].tolist()
    out = np.empty(sum(lengths), dtype="<u2")
    position = 0
    for start, n in zip(starts, lengths):
        out[position : position + n] = _tokens[start : start + n]
        position += n
    return out


def _write_shard(job: tuple[Path, np.ndarray]) -> int:
    path, docs = job
    tokens = _gather(docs)
    tokens.tofile(path)
    return len(tokens)


def main() -> None:
    global _tokens, _starts, _lengths
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokenizer", default="bpe32k")
    parser.add_argument("--vocab-size", type=int, default=32768, help="used if the tokenizer has to be trained")
    parser.add_argument("--tokenizer-chars", type=float, default=2e9, help="characters of text the tokenizer is trained on")
    parser.add_argument("--val-tokens", type=float, default=10e6, help="hold out whole documents up to this many tokens")
    parser.add_argument("--val-candidates", type=int, default=30000, help="documents whose text is kept while tokenizing")
    parser.add_argument("--shard-tokens", type=float, default=100e6, help="a shard is closed once it holds this many")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 4))
    parser.add_argument("--keep-scratch", action="store_true", help="keep the per-row-group token arrays")
    args = parser.parse_args()

    start = time.time()
    out = DATA / args.tokenizer
    scratch = out / "_scratch"
    out.mkdir(parents=True, exist_ok=True)
    pool_context = multiprocessing.get_context("fork")

    download()
    tasks, first = row_groups()
    n_docs = int(first[-1])
    log(f"{len(FILES)} files, {len(tasks):,} row groups, {n_docs:,} documents", start)

    # scan
    if not (DATA / "docs.npz").exists():
        with pool_context.Pool(args.workers, _init_worker) as pool:
            parts = pool.map(_scan, tasks, chunksize=4)
        np.savez(
            DATA / "docs.npz",
            chars=np.concatenate([p[0] for p in parts]),
            bytes=np.concatenate([p[1] for p in parts]),
            hash=np.concatenate([p[2] for p in parts]),
            url_hash=np.concatenate([p[3] for p in parts]),
            gpt2_tokens=np.concatenate([p[4] for p in parts]),
        )
    docs = np.load(DATA / "docs.npz")
    chars, n_bytes, hashes, urls = docs["chars"], docs["bytes"], docs["hash"], docs["url_hash"]
    assert len(chars) == n_docs
    log(f"scanned: {int(chars.sum()):,} characters, {int(n_bytes.sum()):,} bytes; "
        f"{n_docs - len(np.unique(hashes)):,} documents repeat the text of another, "
        f"{n_docs - len(np.unique(urls)):,} the URL of another", start)

    # plan
    order = np.random.default_rng(args.seed).permutation(n_docs)
    from_the_end = np.cumsum(chars[order[::-1]].astype(np.int64))
    n_tokenizer_docs = int(np.searchsorted(from_the_end, args.tokenizer_chars)) + 1
    tokenizer_docs = np.sort(order[n_docs - n_tokenizer_docs :])
    candidates = order[: args.val_candidates]
    assert n_docs - n_tokenizer_docs > args.val_candidates, "the tokenizer text would reach into the validation documents"

    def rows_by_task(chosen: np.ndarray) -> list[list[int]]:
        """Split sorted document indices into the rows of each row group."""
        cuts = np.searchsorted(chosen, first)
        return [(chosen[cuts[t] : cuts[t + 1]] - first[t]).tolist() for t in range(len(tasks))]

    # tokenizer
    if not (tokenizer_dir(args.tokenizer) / "meta.json").exists():
        text_file = DATA / "tokenizer_text.parquet"
        schema = pa.schema([("text", pa.string())])
        writer = pq.ParquetWriter(text_file, schema, compression="zstd")
        with pool_context.Pool(args.workers, _init_worker) as pool, writer:
            for texts in pool.imap(_texts, zip(tasks, rows_by_task(tokenizer_docs)), chunksize=4):
                writer.write_table(pa.table({"text": texts}, schema=schema))
        log(f"tokenizer text: {n_tokenizer_docs:,} documents, {int(from_the_end[n_tokenizer_docs - 1]):,} characters", start)
        trained_on = (f"FineWeb-EDU sample-10BT: the last {n_tokenizer_docs:,} documents of the shuffled order "
                      f"(seed {args.seed}), all of them training documents")
        subprocess.run(
            [sys.executable, "-I", str(Path(__file__).with_name("train_tokenizer.py")), "--name", args.tokenizer,
             "--vocab-size", str(args.vocab_size), "--input", str(text_file), "--trained-on", trained_on],
            check=True,
        )
    tok = Tokenizer.load(args.tokenizer)
    log(f"tokenizer {args.tokenizer}: {tok.vocab_size:,} tokens", start)

    # tokenize
    kept_file = scratch / "kept.jsonl"
    if not kept_file.exists():
        scratch.mkdir(exist_ok=True)
        jobs = [(t, task, rows) for t, (task, rows) in enumerate(zip(tasks, rows_by_task(np.sort(candidates))))]
        done = total = 0
        kept: dict[int, dict] = {}
        with pool_context.Pool(args.workers, _init_worker, (args.tokenizer, scratch)) as pool:
            for t, n_tokens, records in pool.imap_unordered(_tokenize, jobs):
                done, total = done + 1, total + n_tokens
                for record in records:
                    kept[int(first[t]) + record.pop("row")] = record
                if done % 500 == 0:
                    log(f"tokenized {done:,} of {len(jobs):,} row groups, {total:,} tokens", start)
        with open(kept_file.with_suffix(".tmp"), "w", encoding="utf-8") as f:
            for doc in candidates.tolist():
                f.write(json.dumps({"doc": doc, **kept[doc]}, ensure_ascii=False) + "\n")
        kept_file.with_suffix(".tmp").rename(kept_file)
        log(f"tokenized: {total:,} tokens", start)

    # assemble
    _lengths = np.concatenate([np.load(scratch / f"lengths_{t:05d}.npy") for t in range(len(tasks))]).astype(np.int64)
    _starts = np.concatenate([[0], np.cumsum(_lengths)])
    _tokens = np.empty(int(_starts[-1]), dtype="<u2")
    for t in range(len(tasks)):
        _tokens[_starts[first[t]] : _starts[first[t + 1]]] = np.load(scratch / f"tokens_{t:05d}.npy")
    assert len(_lengths) == n_docs and int(_tokens.max()) < tok.vocab_size
    log(f"loaded {len(_tokens):,} tokens of {n_docs:,} documents", start)

    n_val_docs = int(np.searchsorted(np.cumsum(_lengths[order]), args.val_tokens)) + 1
    assert n_val_docs <= args.val_candidates, "raise --val-candidates"
    val_docs, rest = order[:n_val_docs], order[n_val_docs:]
    same_text = np.isin(hashes[rest], hashes[val_docs])
    same_url = np.isin(urls[rest], urls[val_docs])
    train_docs = rest[~(same_text | same_url)]

    ends = np.cumsum(_lengths[train_docs])
    bounds = [0]
    while bounds[-1] < len(train_docs):
        base = ends[bounds[-1] - 1] if bounds[-1] else 0
        bounds.append(min(int(np.searchsorted(ends, base + args.shard_tokens)) + 1, len(train_docs)))
    shards = [f"train_{s:04d}.bin" for s in range(len(bounds) - 1)]
    for stale in out.glob("train_*.bin"):
        stale.unlink()
    with pool_context.Pool(min(args.workers, 32)) as pool:
        jobs = [(out / name, train_docs[a:b]) for name, a, b in zip(shards, bounds, bounds[1:])]
        shard_tokens = pool.map(_write_shard, jobs)
    log(f"wrote {len(shards)} shards, {sum(shard_tokens):,} training tokens", start)

    val_tokens = _gather(val_docs)
    val_tokens.tofile(out / "val.bin")
    kept = {}
    with open(kept_file, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            kept[record.pop("doc")] = record
    with open(DATA / "val_docs.jsonl", "w", encoding="utf-8") as f:
        for doc in val_docs.tolist():
            f.write(json.dumps(kept[doc], ensure_ascii=False) + "\n")

    n_train, val_bytes = sum(shard_tokens), int(n_bytes[val_docs].sum())
    assert n_train == int(_lengths[train_docs].sum())
    assert all(os.path.getsize(out / name) == 2 * n for name, n in zip(shards, shard_tokens))
    meta = {
        "dataset": "fineweb-edu",
        "tokenizer": args.tokenizer,
        "vocab_size": tok.vocab_size,
        "bos_id": tok.bos_id,
        "dtype": "uint16 little-endian",
        "n_train_tokens": n_train,
        "n_val_tokens": len(val_tokens),
        "train_shards": shards,
        "train_shard_tokens": shard_tokens,
        "n_train_docs": len(train_docs),
        "train_bytes": int(n_bytes[train_docs].sum()),
        "bytes_per_token_train": int(n_bytes[train_docs].sum()) / n_train,
        "val_file": "val.bin",
        "val_bytes": val_bytes,
        "val_docs": n_val_docs,
        "val_docs_file": "../val_docs.jsonl",
        "bytes_per_token_val": val_bytes / len(val_tokens),
        "bytes_per_text_token_val": val_bytes / (len(val_tokens) - n_val_docs),
        "note": "token counts include one <|bos|> a document; bytes are the UTF-8 bytes of the text; "
        "bytes_per_token_val = val_bytes / n_val_tokens, bytes_per_text_token_val leaves the <|bos|> tokens out",
        "seed": args.seed,
        "n_docs_in_source": n_docs,
        "n_train_docs_dropped": int(len(rest) - len(train_docs)),
        "n_train_docs_dropped_same_text_as_val": int(same_text.sum()),
        "n_train_docs_dropped_same_url_as_val": int(same_url.sum()),
        "n_docs_repeating_text_of_another": int(n_docs - len(np.unique(hashes))),
        "n_docs_repeating_url_of_another": int(n_docs - len(np.unique(urls))),
        "gpt2_tokens_in_source": int(docs["gpt2_tokens"].astype(np.int64).sum()),
        "tokenizer_trained_on_docs": n_tokenizer_docs,
        "source": f"https://huggingface.co/datasets/{REPO} at revision {REVISION}: sample/10BT, {len(FILES)} Parquet files",
        "license": LICENSE,
        "seconds": round(time.time() - start, 1),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    if not args.keep_scratch:
        shutil.rmtree(scratch)
    log(f"fineweb-edu/{args.tokenizer}: {n_train:,} training tokens in {len(shards)} shards, {len(val_tokens):,} "
        f"validation tokens in {n_val_docs:,} documents, {meta['bytes_per_token_val']:.3f} bytes per token", start)


if __name__ == "__main__":
    main()
