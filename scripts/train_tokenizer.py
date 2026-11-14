"""Train a byte-pair tokenizer and save it under $GPTPP_WORK/data/tokenizers/<name>/.

    python scripts/train_tokenizer.py --name bpe4k --vocab-size 4096 \
        --input $GPTPP_WORK/data/raw/tinystories/TinyStoriesV2-GPT4-train.txt --separator "<|endoftext|>" \
        --trained-on "TinyStories V2 (GPT-4), training split"

    python scripts/train_tokenizer.py --name bpe32k --vocab-size 32768 \
        --input $GPTPP_WORK/data/fineweb-edu/tokenizer_text.parquet \
        --trained-on "FineWeb-EDU sample-10BT, 2 billion characters of training documents"

An input is a text file (documents separated by `--separator`, each stripped of surrounding
whitespace), a Parquet file with a `text` column, or a JSON-lines file of {"text": ...}. The merges
are learned by `gptpp.tokenizer.Tokenizer.train`; the result is `tokenizer.json`, `merges.txt` and
`meta.json`, the last with the SHA-256 of every input so that the training text is on record.

With `--verify` nothing is trained: the saved tokenizer is loaded, the merges are learned again from
the same input by the pure-Python reference `gptpp.bpe.train_fast` (one core; minutes to an hour),
and the two lists are compared. The outcome is written to `reference_check.json` beside the tokenizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Iterator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gptpp import bpe  # noqa: E402
from gptpp.bpe import GPT4_PATTERN  # noqa: E402
from gptpp.tokenizer import Tokenizer, tokenizer_dir  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 24):
            digest.update(block)
    return digest.hexdigest()


def documents(path: Path, separator: str | None, column: str) -> Iterator[str]:
    """The documents of one input file."""
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        table = pq.ParquetFile(path)
        for group in range(table.num_row_groups):
            yield from table.read_row_group(group, columns=[column]).column(column).to_pylist()
    elif path.suffix == ".jsonl":
        with open(path, encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)[column]
    else:
        text = path.read_text(encoding="utf-8")
        if separator is None:
            yield text
        else:
            for document in text.split(separator):
                document = document.strip()
                if document:
                    yield document


def limited(texts: Iterator[str], max_chars: float | None) -> Iterator[str]:
    """Stop after the document that brings the total to `max_chars` characters."""
    total = 0
    for text in texts:
        yield text
        total += len(text)
        if max_chars is not None and total >= max_chars:
            return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="tokenizer name, e.g. bpe32k")
    parser.add_argument("--vocab-size", type=int, required=True, help="bytes + merges + special tokens")
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--separator", default=None, help="document separator inside text files")
    parser.add_argument("--column", default="text", help="text column of Parquet and JSON-lines inputs")
    parser.add_argument("--max-chars", type=float, default=None, help="use about this many characters")
    parser.add_argument("--pattern", default=GPT4_PATTERN, help="the split pattern")
    parser.add_argument("--trained-on", default="", help="a sentence for meta.json saying what the text is")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--verify", action="store_true", help="compare the saved tokenizer with the reference trainer")
    args = parser.parse_args()

    start = time.time()
    texts = limited((doc for path in args.input for doc in documents(path, args.separator, args.column)), args.max_chars)
    if args.verify:
        tok = Tokenizer.load(args.name)
        merges = bpe.train_fast(texts, tok.vocab_size, tok.pattern, list(tok.special))
        differ = [k for k, (ours, theirs) in enumerate(zip(merges, tok.merges)) if ours != theirs]
        check = {
            "tokenizer": args.name,
            "reference": "gptpp.bpe.train_fast, pure Python, on the same input",
            "inputs": [path.name for path in args.input],
            "n_merges_production": len(tok.merges),
            "n_merges_reference": len(merges),
            "identical": merges == tok.merges,
            "first_difference": differ[0] if differ else None,
            "seconds": round(time.time() - start, 1),
        }
        (tokenizer_dir(args.name) / "reference_check.json").write_text(json.dumps(check, indent=2) + "\n")
        print(json.dumps(check, indent=2))
        return
    tok = Tokenizer.train(
        texts, args.vocab_size, name=args.name, pattern=args.pattern, trained_on=args.trained_on, workers=args.workers
    )
    tok.info["sources"] = [{"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)} for path in args.input]
    if args.separator is not None:
        tok.info["separator"] = args.separator
    folder = tok.save()

    print(f"{args.name}: {tok.vocab_size} tokens = 256 bytes + {len(tok.merges)} merges + {len(tok.special)} special")
    print(f"trained on {tok.info['n_docs']:,} documents, {tok.info['n_chars']:,} characters in {time.time() - start:.0f} s")
    print(f"saved to {folder}")
    print("first merges:", " ".join(repr(tok.token_bytes(256 + k).decode("utf-8", "replace")) for k in range(30)))


if __name__ == "__main__":
    main()
