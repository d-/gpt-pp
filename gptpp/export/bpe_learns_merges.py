"""Export for fig-02-01 `gpt-bpe-learns-merges`: the twenty-odd lines of Python that the figure prints beside its
merge loop, run for real.

    python -m gptpp.export.bpe_learns_merges [--out <dir>]

The figure learns byte-pair merges in the page, on a text the reader chooses, and shows the algorithm as code with
the current line lit. This script holds that code as text (`CODE`, and the two lines that change when the pre-split
is switched off), executes it on the figure's two example texts, with the pre-split and without, checks its merges
against the lab's reference trainer `gptpp.bpe.train`, and writes

    bpe-learns-merges-test.json    {code: {on, off}, texts, merges: {<text>: {on, off}}, reference}

No page loads the file. The site's test of the figure compares the code the page prints with `code`, the page's
example texts with `texts`, and the merges the page's own trainer learns with `merges`: the printed program, the
lab's trainer and the page then agree merge for merge.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import regex

from .. import bpe
from .common import DATA, WORK, write_json

NAME = "bpe-learns-merges-test"

CODE = '''def train(text, n_merges):
    words = Counter(tuple(piece.encode())      # pre-split: a piece is
        for piece in PATTERN.findall(text))    # a word with its space
    for new_id in range(256, 256 + n_merges):
        counts = Counter()
        for word, n in words.items():          # count the pairs
            for pair in zip(word, word[1:]):   # of neighbors
                counts[pair] += n
        if not counts:
            return                             # every piece is one token
        pair = min(counts, key=lambda p: (-counts[p], p))  # most frequent
        words = {merge(word, pair, new_id): n  # fuse it everywhere
                 for word, n in words.items()}
        yield pair

def merge(word, pair, new_id):
    out, i = [], 0
    while i < len(word):
        if word[i:i + 2] == pair:
            out.append(new_id); i += 2
        else:
            out.append(word[i]); i += 1
    return tuple(out)'''.split("\n")

# lines 2 and 3 (from 1) when the pre-split is off: the whole text is one piece
OFF = {1: "    words = Counter([tuple(text.encode())])    # no pre-split: the text",
       2: "                                               # is one piece"}

TEXTS = {
    # C. Dickens, A Tale of Two Cities (1859), the first ten clauses of its first sentence; Project Gutenberg eBook 98
    "dickens": ("It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of "
                "foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, "
                "it was the season of Darkness, it was the spring of hope, it was the winter of despair,"),
    # the opening of a held-out story of TinyStories (the third of the validation documents)
    "story": ("Once upon a time, there was a fierce cat named Tom. Tom had a special style. He liked to wear a big, red "
              "hat. All the other cats thought he looked cool.\nOne day, Tom saw a ball in the yard. He wanted to play "
              "with it. So, he ran up to the ball and struck it with his paw. The ball went up, up, up into the sky. "
              "Tom was very happy."),
}


def program(split: bool):
    """The printed code as a function `train`, with the pre-split or without."""
    lines = [OFF.get(i, line) if not split else line for i, line in enumerate(CODE)]
    scope = {"Counter": Counter, "PATTERN": regex.compile(bpe.GPT4_PATTERN)}
    exec("\n".join(lines), scope)  # noqa: S102 - the text above, nothing else
    return scope["train"], lines


def export(out: Path) -> dict:
    merges, checked = {}, 0
    for key, text in TEXTS.items():
        merges[key] = {}
        for split in (True, False):
            train, _ = program(split)
            got = [list(pair) for pair in train(text, 100_000)]
            pattern = bpe.GPT4_PATTERN if split else r"[\s\S]+"
            want = bpe.train(text, 256 + len(got) + 1 + len(bpe.SPECIAL_TOKENS), pattern)
            if [tuple(p) for p in got] != [tuple(p) for p in want]:
                raise SystemExit(f"the printed code and gptpp.bpe.train disagree on {key!r}, pre-split {split}")
            merges[key]["on" if split else "off"] = got
            checked += len(got)
    story_file = WORK / "data" / "tinystories" / "val_docs.jsonl"
    if story_file.exists():
        with open(story_file, encoding="utf-8") as f:
            docs = [json.loads(next(f))["text"] for _ in range(3)]
        if not docs[2].startswith(TEXTS["story"]):
            raise SystemExit("the story is not the opening of the third held-out story")
    res = {"name": NAME, "code": {"on": program(True)[1], "off": program(False)[1]}, "texts": TEXTS, "merges": merges,
           "reference": "the code, executed by Python, learns the merges of gptpp.bpe.train on both texts, with the pre-split and without"}
    size = write_json(out / f"{NAME}.json", res)
    return {"bytes": size, "merges": {k: {s: len(v) for s, v in m.items()} for k, m in merges.items()}, "checked": checked}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(DATA))
    print(json.dumps(export(Path(ap.parse_args().out))))


if __name__ == "__main__":
    main()
