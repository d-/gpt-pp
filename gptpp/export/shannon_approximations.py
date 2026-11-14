"""Export for fig-01-01 `gpt-shannon-approximations`: the book that Shannon's procedure is run on in the page, and
what each of its orders is worth as a predictor.

    python -m gptpp.export.shannon_approximations [--chars 300000] [--held-out 1400] [--out <dir>]

The book is the first held-out TinyStories stories that a table of characters and a table of words can both read
back exactly (ASCII, words parted by one space or one line break), joined by line breaks, about 300 KB. The page
writes text from it at each order: a next symbol is whatever follows, somewhere in the book, the last n - 1 symbols
written.

Each order is then scored as a predictor on other held-out stories (the next ones in the file whose characters
all occur in the book), in bits per character of those stories:

    characters, order 0   every symbol of the book's alphabet equally likely: log2 of its size
    characters, order n   the table of n-grams of the book, interpolated Kneser-Ney (counted_models.py): the
                          maximum-likelihood table of the page gives unseen sequences probability zero, so a score
                          needs the back-off to shorter windows
    words, order n        the same over words (a run of characters between blanks; a line break is a word of its
                          own). A word the book has once or not at all is the word <unknown> to the table, and is
                          then spelled out by the character table of order 6: its letters and the blank after it
    pico                  the network the page runs (the shipped weights), under the evaluation's protocol: each
                          story from `<|bos|>`, in windows of the model's context

Every model is charged for exactly the characters of the stories (the line break that joins two stories is context,
not text). Then, in Node, the figure's own module writes the line that pico continues the opening with, so that the
page shows before any weight is loaded what its model writes with the same seed.

Writes `shannon-approximations.json`.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

from ..tokenizer import Tokenizer
from .browser_model import page_model
from .common import DATA, SITE, write_atomic
from .counted_models import KneserNey, plain, story_texts, window_bits

NAME = "shannon-approximations"
OPENING = "Once upon a time,"
CHAR_ORDERS, WORD_ORDERS = 6, 3

PICO_JS = r"""
const [site, out] = process.argv.slice(1), path = require('path'), fs = require('fs');   // after `node -e`, the arguments begin at 1
const M = require(path.join(site, 'assets/js/gpt/gpt-transformer-forward-pass.js')), T = require(path.join(site, 'assets/js/gpt/gpt-byte-pair-tokenizer.js'));
const F = require(path.join(site, 'assets/js/gpt/gpt-shannon-approximations-figure.js')), D = F['gpt-shannon-approximations.defaults'];
const file = path.join(out, 'shannon-approximations.json'), data = JSON.parse(fs.readFileSync(file, 'utf8'));
data.pico = F['gpt-shannon-approximations.record'](M.fromDisk(D.model, out), T.fromDisk(D.tokenizer, out), data);
fs.writeFileSync(file, JSON.stringify(data));
console.log(JSON.stringify({ seed: data.pico.seed, tokens: data.pico.tokens.length, text: data.pico.texts.join('') }));
"""


def word_symbols(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    """A text as words and line breaks, with where each stands: a word is a run of characters that are neither a
    blank nor a line break; every line break is a symbol of its own."""
    symbols, spans = [], []
    for m in re.finditer(r"\n|[^ \n]+", text):
        symbols.append(m.group())
        spans.append((m.start(), m.end()))
    return symbols, spans


def export(out: Path, chars: int, held_out: int, record: bool = True) -> dict:
    stories = [s for s in story_texts(4 * held_out + 4000) if plain(s)]
    book, size = [], 0
    while size < chars:
        size += len(stories[len(book)]) + 1
        book.append(stories[len(book)])
    text = "\n".join(book)
    alphabet = sorted(set(text))
    known = set(alphabet)
    rest = [s for s in stories[len(book):] if set(s) <= known]
    lead, scored = rest[0], rest[1:held_out + 1]            # one story of context, then the stories that are scored
    n_chars = sum(len(s) for s in scored)

    # ---- characters
    code = {ch: k for k, ch in enumerate(alphabet)}
    def encode(s: str) -> np.ndarray:
        return np.frombuffer(bytes(code[ch] for ch in s), dtype=np.uint8)
    stream_text = lead + "\n" + "\n".join(scored) + "\n"
    stream = encode(stream_text)
    start = len(lead) + 1                                    # the first character of the first scored story
    joins = np.zeros(len(stream), dtype=bool)                # the line breaks that join two stories: context only
    at = start
    for s in scored:
        at += len(s)
        joins[at] = True
        at += 1
    text_mask = ~joins[start:]
    assert int(text_mask.sum()) == n_chars
    letters = KneserNey(encode(text + "\n"), len(alphabet), CHAR_ORDERS)
    bits = {"c0": float(np.log2(len(alphabet)))}
    log_p = {}
    for n in range(1, CHAR_ORDERS + 1):
        log_p[n] = np.log2(letters.probabilities(stream, n, start))
        bits[f"c{n}"] = float(-log_p[n][text_mask].sum() / n_chars)

    # ---- words
    book_words, _ = word_symbols(text + "\n")
    counts = collections.Counter(book_words)
    vocabulary = ["<unknown>"] + sorted(w for w, c in counts.items() if c >= 2)
    word_id = {w: k for k, w in enumerate(vocabulary)}
    words = KneserNey(np.array([word_id.get(w, 0) for w in book_words], dtype=np.uint32), len(vocabulary), WORD_ORDERS)
    symbols, spans = word_symbols(stream_text)
    ids = np.array([word_id.get(w, 0) for w in symbols], dtype=np.uint32)
    first = next(k for k, (a, _) in enumerate(spans) if a >= start)
    span_a = np.array([a for a, _ in spans[first:]])
    span_b = np.array([b for _, b in spans[first:]])
    scored_symbol = ~joins[span_a]                           # every symbol but the joining line breaks
    spell = np.concatenate(([0.0], np.cumsum(-log_p[CHAR_ORDERS])))     # bits of the characters start .. start + i - 1
    unknown = (ids[first:] == 0) & scored_symbol
    spelled = np.where(unknown, spell[np.minimum(span_b - start + 1, len(spell) - 1)] - spell[span_a - start], 0.0)   # its letters and the blank after
    for n in range(1, WORD_ORDERS + 1):
        p = words.probabilities(ids, n, first)
        bits[f"w{n}"] = float((-np.log2(p[scored_symbol]).sum() + spelled.sum()) / n_chars)

    # ---- pico, as the page runs it
    tok = Tokenizer.load("bpe4k")
    model, cfg = page_model("pico", out)
    spent = window_bits(model, tok.encode_batch(scored), tok.bos_id, cfg.block_size)
    bits["pico"] = float(spent.sum() / n_chars)

    data = {
        "name": NAME, "opening": OPENING, "alphabet": "".join(alphabet), "text": text,
        "book": {"stories": len(book), "chars": len(text), "words": len(book_words), "wordTypes": len(counts),
                 "source": "the first held-out TinyStories stories in plain ASCII, joined by line breaks"},
        "heldOut": {"stories": len(scored), "chars": n_chars,
                    "source": "the next held-out stories whose characters all occur in the book"},
        "bits": {k: round(v, 4) for k, v in bits.items()},
        "words": {"known": len(vocabulary) - 1, "unknownShare": round(float(unknown.sum() / scored_symbol.sum()), 4),
                  "spelledBy": f"c{CHAR_ORDERS}", "bitsToSpell": round(float(spelled.sum() / max(1, unknown.sum())), 2)},
        "scoring": "tables: interpolated Kneser-Ney counted on the book; an unknown word is spelled by the character table of order 6; "
                   "pico: the shipped weights, each story from its start token, windows of 256 tokens",
    }
    previous = out / f"{NAME}.json"
    if previous.exists():                                    # keep pico's line until Node writes it again
        old = json.loads(previous.read_text())
        if "pico" in old and old.get("opening") == OPENING:
            data["pico"] = old["pico"]
    write_atomic(out / f"{NAME}.json", json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    report = {"bits": data["bits"], "book": data["book"], "heldOut": data["heldOut"], "words": data["words"]}
    if record:
        report["pico"] = record_pico(out)
    return report


def record_pico(out: Path) -> str | None:
    """The line that pico writes after the opening, by the figure's own module in Node (`record`), added to the file.
    None when Node or the module is not there."""
    node = shutil.which("node") or str(Path.home() / ".local" / "bin" / "node")
    figure = SITE / "assets" / "js" / "gpt" / "gpt-shannon-approximations-figure.js"
    if not Path(node).exists() or not figure.exists():
        return None
    done = subprocess.run([node, "-e", PICO_JS, str(SITE), str(out)], capture_output=True, text=True, timeout=600)
    if done.returncode != 0:
        raise SystemExit(f"recording pico's line failed: {done.stderr.strip()[-2000:]}")
    return done.stdout.strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chars", type=int, default=300_000, help="the size of the book in characters")
    ap.add_argument("--held-out", type=int, default=1400, help="the number of held-out stories that are scored")
    ap.add_argument("--out", default=str(DATA))
    ap.add_argument("--no-record", action="store_true", help="do not ask Node for pico's line")
    args = ap.parse_args()
    print(json.dumps(export(Path(args.out), args.chars, args.held_out, not args.no_record), indent=1))


if __name__ == "__main__":
    main()
