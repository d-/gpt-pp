"""Build the "fresh" held-out set: prose published after every released model's training data.

A released model may have read WikiText, TinyStories or the web pages of FineWeb during its training; nobody can
have read what was written afterwards. This script takes essays published on the web in October and November
2026 (Jekyll posts, Markdown with HTML and Liquid mixed in) and keeps only their paragraphs of plain prose:

- dropped whole: the front matter, HTML blocks (figures, scripts, styles, tables, raw paragraphs), Liquid tags,
  code blocks, display math, headings, lists, tables, block quotes (they quote older texts, which a model may
  know), footnote definitions, and any paragraph that still contains math, code or markup after cleaning;
- removed from inside a paragraph: sidenotes (citations and asides), footnote marks, emphasis marks and link
  targets (the link's words stay).

One document an essay, one JSON object a line: {"text", "id", "date", "n_paragraphs"}. The text is used locally
for scoring and is never sent to a hosted model.

    python scripts/make_fresh_corpus.py --posts '/path/to/_posts/2026-1*.md' --out $GPTPP_WORK/data/fresh/val_docs.jsonl
"""

from __future__ import annotations

import argparse
import glob
import html
import json
import os
import re
from pathlib import Path

WORK = Path(os.environ.get("GPTPP_WORK", "/home/ubuntu/d-work/gpt"))
EXCLUDED_SERIES = ("diffusion", "gpt")  # the two series of posts that are not essays
MIN_CHARS = 200  # a paragraph shorter than this is a caption, an epigraph's attribution or a transition
MIN_PARAGRAPHS = 5  # an essay with fewer paragraphs of prose is a page of something else

# HTML elements that hold a whole block: everything from the opening to the matching closing tag is dropped.
BLOCK_TAGS = ("style", "script", "figure", "pre", "div", "table", "details", "svg", "p", "ul", "ol", "blockquote",
              "h1", "h2", "h3", "h4", "h5", "h6", "aside", "section", "nav", "canvas", "iframe", "math", "dl", "form")
# Inline elements whose words are prose: the tags go, the words stay.
INLINE_TAGS = ("em", "i", "strong", "b", "mark", "cite", "abbr", "span", "a", "q", "u", "small", "dfn")


def split_front_matter(text: str) -> tuple[dict, str]:
    """The YAML front matter of a Jekyll post (top-level ``key: value`` lines only) and the body after it."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end < 0:
        return {}, text
    meta = {}
    for line in text[3:end].splitlines():
        m = re.match(r"^([A-Za-z_]+):\s*(.*)$", line)
        if m:
            meta[m.group(1)] = m.group(2).strip().strip('"')
    return meta, text[text.index("\n", end + 1) + 1:]


def drop_balanced(text: str, opening: re.Pattern, tag: str) -> str:
    """Remove every element that `opening` starts, through the closing tag that matches it (nesting counted)."""
    step = re.compile(rf"<{tag}\b[^>]*>|</{tag}\s*>", re.IGNORECASE)
    out, pos = [], 0
    while True:
        m = opening.search(text, pos)
        if not m:
            return "".join(out) + text[pos:]
        out.append(text[pos:m.start()])
        depth, pos = 1, m.end()
        while depth:
            n = step.search(text, pos)
            if not n:  # never closed: drop to the end rather than keep markup
                return "".join(out)
            depth += -1 if n.group(0).startswith("</") else 1
            pos = n.end()


def drop_blocks(body: str) -> str:
    """Remove everything that is not running text, leaving the paragraphs and blank lines."""
    body = re.sub(r"<!--.*?-->", "", body, flags=re.DOTALL)
    # Liquid: paired blocks first (their content is code or raw markup), then single tags and outputs.
    body = re.sub(r"\{%-?\s*(highlight|raw|comment|capture)\b.*?%\}.*?\{%-?\s*end\1\s*-?%\}", "\n\n", body, flags=re.DOTALL)
    body = re.sub(r"\{%.*?%\}", "\n\n", body, flags=re.DOTALL)
    # Fenced code and display math: a fence line to the next fence line.
    body = re.sub(r"^(```|~~~).*?^\1[^\n]*$", "\n", body, flags=re.DOTALL | re.MULTILINE)
    body = re.sub(r"^\s*\$\$.*?\$\$\s*$", "\n", body, flags=re.DOTALL | re.MULTILINE)
    for tag in BLOCK_TAGS:
        body = drop_balanced(body, re.compile(rf"^[ \t]*<{tag}\b[^>]*>", re.IGNORECASE | re.MULTILINE), tag)
    return body


def clean_inline(paragraph: str) -> str:
    """A paragraph's words: sidenotes and marks removed, links and emphasis unwrapped, one line."""
    text = re.sub(r'<span class="sidenote-number">\s*</span>', "", paragraph)
    text = drop_balanced(text, re.compile(r'<span class="(?:sidenote|marginnote)[^"]*"[^>]*>'), "span")
    text = re.sub(r"\[\^[^\]]+\]", "", text)  # footnote marks
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(rf"</?(?:{'|'.join(INLINE_TAGS)})\b[^>]*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\{:[^}]*\}", "", text)  # kramdown attribute lists
    text = re.sub(r"(?<!!)\[([^\]]*)\]\([^)\s]*(?:\s+\"[^\"]*\")?\)", r"\1", text)  # [words](target) -> words
    text = re.sub(r"(?<!!)\[([^\]]*)\]\[[^\]]*\]", r"\1", text)  # [words][reference] -> words
    text = re.sub(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1", r"\2", text)  # strong
    text = re.sub(r"(?<![\w*])\*(?=\S)(.+?)(?<=\S)\*(?![\w*])", r"\1", text)  # emphasis with asterisks
    text = re.sub(r"(?<![\w_])_(?=\S)(.+?)(?<=\S)_(?![\w_])", r"\1", text)  # emphasis with underscores
    text = re.sub(r"\\([\\`*_{}\[\]()#+\-.!$|~<>])", r"\1", text)  # Markdown's backslash escapes
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


# A block that is not a paragraph: heading, list item, block quote, table row, attribute list, link or footnote
# definition, horizontal rule, the underline of a heading.
BLOCK_START = re.compile(r"\s*(?:#{1,6}\s|[-*+]\s|\d+[.)]\s|>|\||\{:|\[\^?[^\]]+\]:|(?:[-*_]\s*){3,}$|=+\s*$)")
# What must not be left in a paragraph of plain prose: tags, code, math, tables, unresolved links or emphasis.
LEFTOVER = re.compile(r"[<>`|{}\\*_\[\]]|\$\$")


def why_not_prose(raw: str, cleaned: str) -> str | None:
    """None when the paragraph is plain prose, else the reason it is dropped."""
    if any(BLOCK_START.match(line) for line in raw.splitlines()):
        return "a heading, list, quotation, table or definition"
    if LEFTOVER.search(cleaned):
        return "markup, code or math left after cleaning"
    if len(cleaned) < MIN_CHARS:
        return f"shorter than {MIN_CHARS} characters"
    if sum(ch.isalpha() or ch.isspace() for ch in cleaned) / len(cleaned) < 0.9:
        return "less than nine tenths letters and spaces"
    if cleaned[-1] not in ".?!\u201d\"\u2019')\u2026":
        return "does not end as a sentence does"
    return None


def prose_paragraphs(markdown: str, dropped: dict[str, int] | None = None) -> list[str]:
    """The paragraphs of plain prose of one post, cleaned. `dropped` counts the paragraphs left out, by reason."""
    _, body = split_front_matter(markdown)
    paragraphs = []
    for raw in re.split(r"\n[ \t]*\n", drop_blocks(body)):
        raw = raw.strip("\n")
        if not raw.strip():
            continue
        cleaned = clean_inline(raw)
        reason = why_not_prose(raw, cleaned)
        if reason is None:
            paragraphs.append(cleaned)
        elif dropped is not None:
            dropped[reason] = dropped.get(reason, 0) + 1
    return paragraphs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--posts", required=True, help="glob of the essays (Jekyll posts), quoted")
    ap.add_argument("--out", default=str(WORK / "data/fresh/val_docs.jsonl"))
    args = ap.parse_args()

    docs, dropped = [], {}
    for path in sorted(glob.glob(args.posts)):
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        meta, _ = split_front_matter(text)
        permalink = meta.get("permalink", "").strip("/")
        in_series = any(part in EXCLUDED_SERIES for part in p.parts) or any(
            permalink == s or re.match(rf"{s}-\d", permalink) for s in EXCLUDED_SERIES)
        date = re.match(r"\d{4}-\d{2}-\d{2}", p.name)
        if in_series or not date:
            print(f"skipped  {p.name}  (series post or undated)")
            continue
        paragraphs = prose_paragraphs(text, dropped)
        if len(paragraphs) < MIN_PARAGRAPHS:
            print(f"skipped  {p.name}  ({len(paragraphs)} paragraphs of prose)")
            continue
        doc = {"text": "\n\n".join(paragraphs), "id": p.stem, "date": date.group(0), "n_paragraphs": len(paragraphs)}
        docs.append(doc)
        print(f"kept     {p.name}  {len(paragraphs)} paragraphs, {len(doc['text'].encode('utf-8'))} bytes")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    total = sum(len(d["text"].encode("utf-8")) for d in docs)
    print(f"{len(docs)} documents, {sum(d['n_paragraphs'] for d in docs)} paragraphs, {total} bytes -> {out}")
    for reason, count in sorted(dropped.items(), key=lambda item: -item[1]):
        print(f"  dropped {count:4d} blocks: {reason}")


if __name__ == "__main__":
    main()
