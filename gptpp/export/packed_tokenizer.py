"""A tokenizer for the page with its merges packed as bytes: gpt-pp's 32,768 tokens in 130 kB instead of 350.

    python -m gptpp.export.packed_tokenizer [--tokenizer bpe32k] [--out <dir>]

`gptpp.export.browser_model --tokenizer` writes a tokenizer's merges as JSON numbers, which is right for pico's
3,835 merges (33 kB) and heavy for 32,507. This exporter writes, into the site's data folder:

    <name>.json           what the page's tokenizer needs besides the merges: the size of the vocabulary, the special
                          tokens, the split pattern as JavaScript reads it, and `packed`, the manifest key of the merges
    <name>.bin            the merges, four bytes each: two little-endian uint16 ids; pair k makes token 256 + k
    <name>.entries.json   the manifest entry of the merges (merged into manifest.json by gptpp.export.manifest)
    <name>-test.json      strings with the ids that the two reference encoders give them (gptpp/bpe.py with Python's
                          `regex`, gptpp/tokenizer.py with the Rust library); no page loads it, the site's tests do

The test strings are the edge cases of `browser_model` (apostrophes, digits, every kind of whitespace, scripts,
emoji), the sentences that the figures of part 2 offer as presets, and a few hundred random cuts of held-out
documents of the tokenizer's own corpus. A string on which the two reference encoders disagree is left out and
counted.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from .browser_model import EDGE_STRINGS, FORMAT, JS_FLAGS, JS_PATTERN
from .common import DATA, WORK, Blob, write_json

# the sentences the figures offer (assets/js/gpt/gpt-three-tokenizers-figure.js keeps the same list; its test
# compares) and the tokens that the figures of the embedding space name
FIGURE_STRINGS = [
    "The transformer's residual stream has 768 dimensions; it's 2026.",
    "Once upon a time, there was a little girl named Lily.",
    "The speed of light is 299792458 meters per second.",
    "Pneumonoultramicroscopicsilicovolcanoconiosis is a lung disease.",
    "def f(x):\n    return {'a': x}",
    "注意力就是你所需要的一切。",
    " March", " seven", "7", " France", " walked", "ing", " king", " queen", " Paris", " 7", "768", "2026", "strawberry",
    " strawberry", " SolidGoldMagikarp", "It was the best of times, it was the worst of times,",
]
HELD_OUT = {"bpe32k": "fineweb-edu", "bpe4k": "tinystories"}


def test_strings(name: str, n_strings: int = 240, seed: int = 0) -> dict:
    from .. import bpe
    from ..tokenizer import Tokenizer

    production = Tokenizer.load(name)
    reference = bpe.BPE(production.merges, production.pattern, list(production.special))
    rng = random.Random(seed)
    texts = list(EDGE_STRINGS) + FIGURE_STRINGS
    docs_file = WORK / "data" / HELD_OUT.get(name, "fineweb-edu") / "val_docs.jsonl"
    if docs_file.exists():
        docs = []
        with open(docs_file, encoding="utf-8") as f:
            for line in f:
                docs.append(json.loads(line)["text"])
                if len(docs) >= 2000:
                    break
        for _ in range(n_strings):
            doc = rng.choice(docs)
            length = rng.choice([12, 30, 60, 120, 240])
            start = rng.randrange(0, max(1, len(doc) - length))
            texts.append(doc[start:start + length])
    strings, dropped = [], 0
    for text in texts:
        ids = reference.encode(text)
        if ids != production.encode(text):
            dropped += 1
            continue
        strings.append({"text": text, "ids": ids})
    decode = []
    for _ in range(40):
        ids = [rng.randrange(0, production.vocab_size) for _ in range(rng.randrange(1, 12))]
        decode.append({"ids": ids, "text": production.decode(ids)})
    return {"name": name, "reference": "gptpp/bpe.py (Python regex) and gptpp/tokenizer.py (tokenizers), which agree on every string kept",
            "dropped": dropped, "strings": strings, "decode": decode}


def export(name: str, out: Path) -> dict:
    from .. import bpe
    from ..tokenizer import Tokenizer

    tokenizer = Tokenizer.load(name)
    if tokenizer.pattern != bpe.GPT4_PATTERN:
        raise SystemExit(f"tokenizer {name} splits with another pattern than the one JS_PATTERN was written for")
    if tokenizer.vocab_size > 65536:
        raise SystemExit("ids do not fit in 16 bits")
    pairs = np.asarray(tokenizer.merges, dtype="<u2")
    blob = Blob(name, out)
    key = f"{name}.merges"
    blob.put(key, pairs.view(np.uint8).reshape(len(pairs), 4),
             "the merges in the order learned: a pair of little-endian uint16 token ids a row; pair k makes token 256 + k")
    size = blob.save()
    meta = {
        "name": name, "format": FORMAT, "vocabulary": tokenizer.vocab_size, "bos": tokenizer.bos_id, "special": tokenizer.special,
        "ids": "0..255 the bytes; then the merges in the order learned (merge k makes token 256 + k); then the special tokens",
        "pattern": {"python": tokenizer.pattern, "js": JS_PATTERN, "flags": JS_FLAGS},
        "trainedOn": tokenizer.info.get("trained_on", ""),
        "packed": {"key": key, "file": f"{name}.bin", "merges": int(len(pairs)), "bytes": size,
                   "form": "pairs of little-endian uint16 ids"},
    }
    json_size = write_json(out / f"{name}.json", meta)
    tests = test_strings(name)
    test_size = write_json(out / f"{name}-test.json", tests)
    return {"name": name, "merges": int(len(pairs)), "bytes": size, "jsonBytes": json_size, "testBytes": test_size,
            "strings": len(tests["strings"]), "dropped": tests["dropped"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tokenizer", default="bpe32k")
    ap.add_argument("--out", default=str(DATA))
    args = ap.parse_args()
    print(json.dumps(export(args.tokenizer, Path(args.out))))


if __name__ == "__main__":
    main()
