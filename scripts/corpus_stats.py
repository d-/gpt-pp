"""Measure a tokenized corpus: counts, Zipf's law, document lengths, entropy, n-gram models.

    python scripts/corpus_stats.py                           # every dataset/tokenizer that is prepared
    python scripts/corpus_stats.py --data tinystories/bpe4k

Writes $GPTPP_WORK/data/<dataset>/<tokenizer>/stats.json and a copy in results/data/ of the repository.

What is measured, all of it on the token shards written by the prepare_* scripts:

- tokens, documents, bytes, bytes per token, for the training set and the held-out set;
- the count of every token in the training set, sorted: the rank-frequency table of a Zipf plot,
  the 200 most frequent tokens with their text, the unigram entropy in bits per token and per byte;
- the distribution of document lengths in tokens, and of token lengths in bytes;
- n-gram language models for n = 1..4, counted on the first `--ngram-tokens` tokens of the training
  stream (the documents are shuffled, so a prefix is a random sample) and scored on the held-out
  stream in bits per byte, the unit in which the neural models are scored. The smoothing is
  interpolated Kneser-Ney with one discount per level (Kneser and Ney, 1995; Chen and Goodman,
  1999), described at `ngram_statistics` below;
- general-purpose compressors (gzip, bzip2, xz, zstd) on the held-out text, in the same unit;
- checks of the contract over every shard: ids inside the vocabulary, no chat token, one <|bos|> per
  document, and no held-out document among the training documents; and the SHA-256 of every file.
"""

from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import lzma
import multiprocessing
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gptpp import bpe, repo_dir, work_dir  # noqa: E402
from gptpp.tokenizer import Tokenizer  # noqa: E402

WORK = work_dir()  # $GPTPP_WORK
RESULTS = repo_dir() / "results" / "data"


# ----------------------------------------------------------------------------------------------------
# One pass over the shards, and the checks on the held-out set (all in worker processes)
# ----------------------------------------------------------------------------------------------------

_tok: Tokenizer
_val: np.ndarray  # the held-out tokens
_held_out: set[bytes]  # a hash of the tokens of each held-out document
_texts: list[str]  # the held-out documents as text


def _hash(tokens: np.ndarray) -> bytes:
    return hashlib.blake2b(tokens.tobytes(), digest_size=8).digest()


def documents(tokens: np.ndarray, bos: int) -> tuple[np.ndarray, np.ndarray]:
    """Where each document of a stream starts and ends (a document begins with <|bos|>)."""
    starts = np.flatnonzero(tokens == bos)
    return starts, np.append(starts[1:], len(tokens))


def _shard_pass(path: Path) -> tuple[np.ndarray, np.ndarray, int, str]:
    """Of one shard: token counts, document lengths (without <|bos|>), held-out documents found, SHA-256."""
    tokens = np.fromfile(path, dtype="<u2")
    assert tokens[0] == _tok.bos_id, f"{path.name} does not begin with a document"
    counts = np.bincount(tokens, minlength=_tok.vocab_size)
    starts, ends = documents(tokens, _tok.bos_id)
    found = sum(_hash(tokens[a:b]) in _held_out for a, b in zip(starts.tolist(), ends.tolist()))
    return counts, (ends - starts - 1).astype(np.uint32), found, hashlib.sha256(tokens).hexdigest()


def _compress(name: str) -> tuple[str, int]:
    """The size of the held-out text under a general-purpose compressor at its strongest setting."""
    text = "\n".join(_texts).encode("utf-8")
    if name == "gzip":
        return f"gzip -9 (zlib {zlib.ZLIB_RUNTIME_VERSION})", len(zlib.compress(text, 9))
    if name == "bzip2":
        return "bzip2 -9", len(bz2.compress(text, 9))
    if name == "xz":
        return "xz -9e (LZMA2)", len(lzma.compress(text, preset=9 | lzma.PRESET_EXTREME))
    import zstandard

    level = int(name.removeprefix("zstd"))
    return f"zstd -{level} (zstandard {zstandard.__version__})", len(zstandard.ZstdCompressor(level=level).compress(text))


def _reference_differs(part: int, parts: int = 8) -> int:
    """How many held-out documents (every `parts`-th, from `part`) the pure-Python reference encodes differently."""
    reference = bpe.BPE(_tok.merges, _tok.pattern, list(_tok.special))
    starts, ends = documents(_val, _tok.bos_id)
    return sum(reference.encode(_texts[i], bos=True) != _val[starts[i] : ends[i]].tolist() for i in range(part, len(_texts), parts))


def length_statistics(lengths: np.ndarray) -> dict:
    """Summary of document lengths in tokens: moments, percentiles, a histogram with bins of ratio 2^(1/4)."""
    lengths = lengths.astype(np.int64)
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
    powers = np.floor(2 ** (np.arange(0, 4 * np.log2(lengths.max() + 1) + 2) / 4)).astype(np.int64)
    edges = np.unique(np.concatenate(([0], powers)))  # 0, 1, 2, 3, 4, 5, 6, 8, 9, 11, 13, 16, ... past the maximum
    histogram, _ = np.histogram(lengths, bins=edges)
    contexts = [128, 256, 512, 1024, 2048, 4096, 8192]
    return {
        "note": "tokens of text in a document, the <|bos|> token not counted",
        "mean": float(lengths.mean()),
        "std": float(lengths.std()),
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "percentiles": {str(q): float(np.percentile(lengths, q)) for q in percentiles},
        "histogram": {
            "note": "counts[i] documents have edges[i] <= length < edges[i+1]; the edges grow by a factor 2^(1/4)",
            "edges": edges.tolist(),
            "counts": histogram.tolist(),
        },
        "fraction_of_documents_longer_than": {str(c): float((lengths > c).mean()) for c in contexts},
        "fraction_of_tokens_in_documents_longer_than": {str(c): float(lengths[lengths > c].sum() / lengths.sum()) for c in contexts},
    }


# ----------------------------------------------------------------------------------------------------
# n-gram models
# ----------------------------------------------------------------------------------------------------


def ngram_keys(tokens: np.ndarray, n: int, bits: int) -> np.ndarray:
    """Every n-gram of a stream as one integer: its tokens side by side, `bits` bits each."""
    count = len(tokens) - n + 1
    keys = tokens[:count].astype(np.uint64)
    for k in range(1, n):
        keys = (keys << np.uint64(bits)) | tokens[k : k + count]
    return keys


def table(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The distinct values of `keys`, sorted, and the number of times each occurs."""
    keys = np.sort(keys)
    first = np.flatnonzero(np.concatenate(([True], keys[1:] != keys[:-1])))
    return keys[first], np.diff(first, append=len(keys))


def lookup(keys: np.ndarray, values: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """values[i] where keys[i] == query, and 0 where the query is not among the (sorted) keys."""
    index = np.minimum(np.searchsorted(keys, queries), len(keys) - 1)
    return np.where(keys[index] == queries, values[index], 0)


class Level:
    """One level of an interpolated n-gram model: counts of k-grams and of their contexts of k - 1 tokens."""

    def __init__(self, grams: np.ndarray, counts: np.ndarray, bits: int) -> None:
        self.grams, self.counts, self.bits = grams, counts, bits
        ones, twos = int((counts == 1).sum()), int((counts == 2).sum())
        self.discount = ones / (ones + 2 * twos) if ones and twos else 0.5  # Ney, Essen and Kneser (1994)
        contexts = grams >> np.uint64(bits)  # sorted, because the k-grams are
        first = np.flatnonzero(np.concatenate(([True], contexts[1:] != contexts[:-1])))
        self.contexts = contexts[first]
        self.totals = np.add.reduceat(counts, first)  # how often each context was continued
        self.types = np.diff(first, append=len(grams))  # by how many different tokens

    def probability(self, grams: np.ndarray, lower: np.ndarray) -> np.ndarray:
        """P(last token | the k - 1 tokens before it), given the estimate `lower` of the level below.

        Where the context was seen: (max(count - D, 0) + D * types * lower) / total. The discount D
        taken from every seen continuation is what pays for the lower-order estimate, so the
        probabilities of a context sum to one. Where the context was never seen: `lower`.
        """
        count = lookup(self.grams, self.counts, grams)
        contexts = grams >> np.uint64(self.bits)
        total = lookup(self.contexts, self.totals, contexts)
        types = lookup(self.contexts, self.types, contexts)
        seen = total > 0
        p = lower.copy()
        p[seen] = (np.maximum(count[seen] - self.discount, 0) + self.discount * types[seen] * lower[seen]) / total[seen]
        return p

    def conditional_entropy(self) -> float:
        """Plug-in entropy of the last token given the context, in bits: the model's loss on its own data."""
        totals = np.repeat(self.totals, self.types)
        return float(-(self.counts * np.log2(self.counts / totals)).sum() / self.counts.sum())


def ngram_statistics(sample: np.ndarray, held_out: np.ndarray, vocab_size: int, bytes_per_token: float, orders: int = 4) -> dict:
    """n-gram models for n = 1..orders, counted on `sample`, scored on `held_out`.

    The model of order n is interpolated Kneser-Ney. Its top level uses the counts of n-grams in the
    sample. Each level below uses, for a k-gram, the number of different tokens that precede it in
    the sample (the "continuation count": how many contexts the k-gram completes, not how often it
    occurs). The level under the unigrams is the uniform distribution over the vocabulary, so every
    token has a probability above zero. One discount per level, D = n1 / (n1 + 2 n2), with n1 and n2
    the numbers of k-grams of that level counted once and twice. Nothing is tuned on the held-out set.

    A model of order n is scored on every held-out token from the fourth on (the same tokens for every
    n); bits per byte are bits per token divided by the bytes per token of the held-out set.
    """
    bits = int(np.ceil(np.log2(vocab_size)))
    assert bits * orders <= 64
    raw: dict[int, Level] = {}
    continuation: dict[int, Level] = {}
    for n in range(1, orders + 1):
        grams, counts = table(ngram_keys(sample, n, bits))
        raw[n] = Level(grams, counts, bits)
        if n > 1:  # the (n-1)-grams, each counted by the number of different tokens seen before it
            suffixes = grams & np.uint64((1 << (bits * (n - 1))) - 1)
            continuation[n - 1] = Level(*table(suffixes), bits)

    start = orders - 1  # score the same positions for every order
    scored = {k: ngram_keys(held_out[start - k + 1 :], k, bits) for k in range(1, orders + 1)}
    result = []
    for n in range(1, orders + 1):
        p = np.full(len(held_out) - start, 1 / vocab_size)
        for k in range(1, n + 1):
            p = (raw[k] if k == n else continuation[k]).probability(scored[k], p)
        bits_per_token = float(-np.log2(p).mean())
        result.append({
            "n": n,
            "distinct_ngrams_in_sample": len(raw[n].grams),
            "ngrams_seen_once_in_sample": int((raw[n].counts == 1).sum()),
            "discount": raw[n].discount,
            "fraction_of_held_out_ngrams_not_in_sample": float((lookup(raw[n].grams, raw[n].counts, scored[n]) == 0).mean()),
            "sample_bits_per_token": raw[n].conditional_entropy(),
            "held_out_bits_per_token": bits_per_token,
            "held_out_perplexity": float(2**bits_per_token),
            "held_out_bits_per_byte": bits_per_token / bytes_per_token,
        })
    return {
        "smoothing": "interpolated Kneser-Ney, one discount per level D = n1/(n1 + 2 n2), uniform distribution below the unigrams",
        "note": "sample_bits_per_token is the entropy of the n-th token given the n-1 before it under the counts of the "
        "sample itself (no smoothing; it falls towards zero as n grows because most long n-grams occur once); "
        "held_out_* is the cross-entropy of the smoothed model on the held-out stream, <|bos|> tokens included",
        "sample_tokens": len(sample),
        "held_out_tokens_scored": len(held_out) - start,
        "bytes_per_token_held_out": bytes_per_token,
        "orders": result,
    }


# ----------------------------------------------------------------------------------------------------
# The whole report for one dataset
# ----------------------------------------------------------------------------------------------------


def corpus_statistics(dataset: str, tokenizer: str, ngram_tokens: int, workers: int) -> dict:
    global _tok, _val, _held_out, _texts
    start = time.time()
    folder = WORK / "data" / dataset / tokenizer
    meta = json.loads((folder / "meta.json").read_text())
    _tok = tok = Tokenizer.load(tokenizer)
    _val = val = np.fromfile(folder / "val.bin", dtype="<u2")
    val_starts, val_ends = documents(val, tok.bos_id)
    _held_out = {_hash(val[a:b]) for a, b in zip(val_starts.tolist(), val_ends.tolist())}
    with open(folder.parent / "val_docs.jsonl", encoding="utf-8") as f:
        _texts = [json.loads(line)["text"] for line in f]
    compressors = ["gzip", "bzip2", "xz"]
    try:
        import zstandard  # noqa: F401

        compressors += ["zstd19", "zstd22"]
    except ImportError:
        pass

    pool = multiprocessing.get_context("fork").Pool(workers)
    compressed = pool.map_async(_compress, compressors, chunksize=1)  # these two run beside everything below
    differs = pool.map_async(_reference_differs, range(8), chunksize=1)
    parts = pool.map(_shard_pass, [folder / name for name in meta["train_shards"]])
    assert all(len(p[0]) == tok.vocab_size for p in parts), "a shard holds an id outside the vocabulary"
    counts = np.sum([p[0] for p in parts], axis=0)
    lengths = np.concatenate([p[1] for p in parts])
    n_tokens, n_docs = int(counts.sum()), len(lengths)
    print(f"{dataset}/{tokenizer}: {n_tokens:,} tokens, {n_docs:,} documents counted ({time.time() - start:.0f} s)", flush=True)

    # Zipf: the counts sorted from the most frequent token down.
    order = np.argsort(-counts, kind="stable")
    ranked = counts[order]
    p = ranked[ranked > 0] / n_tokens
    entropy = float(-(p * np.log2(p)).sum())
    fit_ranks = np.arange(10, 1001)
    slope, intercept = np.polyfit(np.log10(fit_ranks), np.log10(ranked[fit_ranks - 1]), 1)
    token_lengths = np.array([len(tok.token_bytes(i)) for i in range(tok.vocab_size - len(tok.special))])
    text_counts = counts[: len(token_lengths)]
    bytes_per_token_train = meta["train_bytes"] / n_tokens

    first_shards = meta["train_shards"][: -(-ngram_tokens // 10**8) + 1]  # enough shards of 100M tokens, and one more
    sample = np.concatenate([np.fromfile(folder / name, dtype="<u2") for name in first_shards])[:ngram_tokens]
    ngrams = ngram_statistics(sample, val, tok.vocab_size, meta["bytes_per_token_val"])
    print(f"{dataset}/{tokenizer}: n-gram models done ({time.time() - start:.0f} s)", flush=True)
    compressed, differs = compressed.get(), sum(differs.get())
    pool.close()
    pool.join()

    return {
        "dataset": dataset,
        "tokenizer": tokenizer,
        "vocab_size": tok.vocab_size,
        "source": meta["source"],
        "license": meta["license"],
        "tokens": {"train": n_tokens, "val": len(val), "note": "one <|bos|> a document included"},
        "documents": {"train": n_docs, "val": len(val_starts)},
        "bytes": {"train": meta["train_bytes"], "val": meta["val_bytes"], "note": "UTF-8 bytes of the text"},
        "bytes_per_token": {
            "train": bytes_per_token_train,
            "val": meta["val_bytes"] / len(val),
            "train_text_tokens_only": meta["train_bytes"] / (n_tokens - n_docs),
            "val_text_tokens_only": meta["val_bytes"] / (len(val) - len(val_starts)),
            "note": "bytes / tokens; the first two count the <|bos|> tokens, as the trainer's bits per byte does",
        },
        "unigram": {
            "entropy_bits_per_token": entropy,
            "entropy_bits_per_byte": entropy / bytes_per_token_train,
            "tokens_never_seen": int((counts == 0).sum()),
            "ids_never_seen": np.flatnonzero(counts == 0).tolist(),
            "note": "entropy of the distribution of tokens in the training set, <|bos|> included",
        },
        "rank_frequency": {
            "note": "counts[r-1] is the number of occurrences in the training set of the token of rank r, ids[r-1] its id",
            "counts": ranked.tolist(),
            "ids": order.tolist(),
            "zipf_fit": {
                "note": "least squares of log10(count) on log10(rank) over ranks 10..1000: count ~ rank^(-exponent)",
                "exponent": float(-slope),
                "log10_count_at_rank_1": float(intercept),
            },
        },
        "top_tokens": [
            {
                "rank": r + 1,
                "id": int(i),
                "token": tok.token_bytes(int(i)).decode("utf-8", errors="replace"),
                "count": int(counts[i]),
                "frequency": float(counts[i] / n_tokens),
            }
            for r, i in enumerate(order[:200])
        ],
        "document_length_tokens": {"train": length_statistics(lengths), "val": length_statistics(val_ends - val_starts - 1)},
        "token_length_bytes": {
            "note": "by_type[k]: tokens of the vocabulary that are k bytes long; "
            "by_occurrence[k]: their share of the training text tokens",
            "by_type": np.bincount(token_lengths).tolist(),
            "by_occurrence": (np.bincount(token_lengths, weights=text_counts) / text_counts.sum()).tolist(),
            "mean_by_type": float(token_lengths.mean()),
        },
        "ngrams": ngrams,
        "compressors": {
            "note": "the held-out documents joined by line breaks, compressed as one stream: 8 * compressed bytes / val bytes",
            "results": [
                {"name": name, "compressed_bytes": size, "bits_per_byte": 8 * size / meta["val_bytes"]}
                for name, size in compressed
            ],
        },
        "checks": {
            "token_count_matches_meta": n_tokens == meta["n_train_tokens"],
            "ids_inside_vocabulary": True,  # asserted above for every shard
            "chat_tokens_in_training_data": int(sum(counts[i] for name, i in tok.special.items() if name != "<|bos|>")),
            "bos_tokens_equal_documents": int(counts[tok.bos_id]) == n_docs == meta["n_train_docs"],
            "held_out_documents_found_in_training": int(sum(p[2] for p in parts)),
            "held_out_documents_the_reference_encodes_differently": int(differs),
            "note": "every training shard read in full; a held-out document is 'found' if a training document has exactly its "
            "tokens; the last check encodes every held-out document with the pure-Python gptpp.bpe and compares with val.bin",
        },
        "sha256": {"val.bin": hashlib.sha256(val).hexdigest(), **{name: p[3] for name, p in zip(meta["train_shards"], parts)}},
        "seconds": round(time.time() - start, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", nargs="*", help="dataset/tokenizer pairs; default: every prepared one")
    parser.add_argument("--ngram-tokens", type=float, default=200e6, help="tokens of training data the n-grams are counted on")
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 32))
    args = parser.parse_args()

    prepared = [p.parent for p in (WORK / "data").glob("*/*/meta.json") if p.parent.parent.name != "tokenizers"]
    pairs = args.data or sorted(f"{folder.parent.name}/{folder.name}" for folder in prepared)
    for pair in pairs:
        dataset, tokenizer = pair.split("/")
        stats = corpus_statistics(dataset, tokenizer, int(args.ngram_tokens), args.workers)
        text = json.dumps(stats, indent=1, ensure_ascii=False) + "\n"
        (WORK / "data" / dataset / tokenizer / "stats.json").write_text(text, encoding="utf-8")
        (RESULTS / dataset / tokenizer).mkdir(parents=True, exist_ok=True)
        (RESULTS / dataset / tokenizer / "stats.json").write_text(text, encoding="utf-8")
        print(f"{pair}: {stats['tokens']['train']:,} tokens, {stats['documents']['train']:,} documents, "
              f"{stats['bytes_per_token']['train_text_tokens_only']:.3f} bytes per text token, "
              f"unigram entropy {stats['unigram']['entropy_bits_per_token']:.3f} bits per token")
        for order in stats["ngrams"]["orders"]:
            print(f"  {order['n']}-gram: {order['held_out_bits_per_token']:.3f} bits per token, "
                  f"{order['held_out_bits_per_byte']:.3f} bits per byte on the held-out set")
        for result in stats["compressors"]["results"]:
            print(f"  {result['name']}: {result['bits_per_byte']:.3f} bits per byte on the held-out text")
        print(f"  checks: {stats['checks']}")


if __name__ == "__main__":
    main()
