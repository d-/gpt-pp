"""The sources of the series as one file, for the site's figure `gpt-reading-map` (part 11): every work the research
dossiers fetched, with its title, authors, year, venue or arXiv number, the theme it is filed under, the Sections of
the series that use it, and the "builds on" links that a dossier's timeline states.

    python -m gptpp.export.reading_map [--dossiers <folder of the dossiers>] [--out <dir>] [--show]

Nothing is measured here, and nothing is added from memory: the script reads the Markdown of the research dossiers
(`$GPTPP_WORK/research/*.md`), section 3 ("Sources") for the works and section 2 ("Timeline") for the links.

Works. A dossier lists its sources as a table or as a numbered or bulleted list, one work an entry, with a note of
whether and how it was fetched. Every entry is read into authors, title, venue, year and an identifier (an arXiv
number, else a URL); entries the dossier did not fetch ("no", "N", "search result only") are left out, as are
entries without a title or a year (repositories at a commit, configuration files). The same work listed by several
dossiers (same arXiv number, or same title) is one work: it keeps every Section that uses it.

Themes. A work is filed under the theme of the dossier that lists it (the table `DOSSIERS` below: ten themes, in the
order of Sections 1 to 10). A few groups inside a dossier belong to another theme (`GROUPS`), and the survey of
2026 (`frontier-2026`), which spans every theme, is filed work by work by the words of its title (`FRONTIER`). A
work listed by several dossiers takes the theme most of them give it (a tie goes to the dossier of the lowest
Section), except for the works whose titles `HOME` names.

Links. Each dossier's timeline is a table: year, work, what it introduced, the problem it answered, what later
replaced or refined it. A row is tied to the works of the same dossier whose first author it names in the year it
gives; its last cell names later works the same way ("Hoffmann 2022", "Li et al. 2018"), or by arXiv number. Every
(later work, row's work) pair found is a link "the later work builds on the earlier one". A name is looked up
among the works of the same dossier only; one that matches no fetched work, or more than one, makes no link.

The file (`reading-map.json`), field by field:

    generated_by   "gptpp.export.reading_map"
    themes         the ten themes in order: [{key, label, section (the Section of the series it belongs to)}]
    sections       {"1": "Predicting the next token", ...}: the titles of the Sections that works point to
    works          one entry a work, by year then title:
      k              a stable key: "arXiv:2104.09864", or a slug of first author, year and title
      t, a           title; authors (at most three names, then "et al.")
      f              the first author's surname (a team's or a company's whole name): how a list names the work
      y              year (of the venue the dossier gives; of the arXiv number otherwise)
      v              venue as the dossier writes it, "" if none
      x              "arXiv:NNNN.NNNNN" or a URL or ""
      th             index into `themes`
      s              the Sections that use it: for each dossier that lists the work, the first Section the dossier
                     feeds (the theme's own Section first)
      n              in how many dossiers it is listed
    links          [[later, earlier], ...]: indices into `works`; the later work builds on the earlier one
    counts         {entries (read), not_fetched, unusable (no title or year), works, links, by_theme: [..]}
"""
from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

from .common import DATA, WORK, write_json

NAME = "reading-map"

THEMES = [("language", "language modeling", 1), ("tokens", "tokens", 2), ("attention", "attention", 3),
          ("position", "position", 4), ("block", "the block", 5), ("optimization", "optimization", 6),
          ("training", "training", 7), ("dynamics", "dynamics and theory", 8), ("circuits", "circuits", 9),
          ("features", "features", 10)]
SECTIONS = {1: "Predicting the next token", 2: "Tokens and embeddings", 3: "Attention", 4: "Position", 5: "The block",
            6: "Optimization", 7: "Training gpt-pp", 8: "What training learns", 9: "Circuits", 10: "Features",
            11: "Frontier, reading and recipe"}

# dossier -> (theme, the Sections it feeds: its works are counted as used by the first)
DOSSIERS = {
    "lm-history": ("language", [1]),
    "scaling-laws": ("training", [7, 1, 8]),
    "tokenization-embeddings": ("tokens", [2]),
    "high-dim-probability": ("tokens", [2, 3, 5, 6, 10]),
    "attention-evolution": ("attention", [3]),
    "flash-attention": ("attention", [3, 7]),
    "position-encoding": ("position", [4]),
    "normalization-residual-init": ("block", [5]),
    "activations-mlp": ("block", [5, 9]),
    "optimizers": ("optimization", [6]),
    "speedrun-and-muon": ("optimization", [6, 5, 7]),
    "nanogpt-vs-gpt2": ("training", [7, 5, 11]),
    "nanochat": ("training", [7, 5, 6, 11]),
    "training-frontier": ("training", [7, 11]),
    "learning-dynamics-theory": ("dynamics", [8]),
    "circuits-framework": ("circuits", [9]),
    "superposition-monosemanticity": ("features", [10]),
    "circuit-tracing-frontier": ("features", [10, 11]),
    "frontier-2026": (None, [11]),
}
# groups (sub-headings of a dossier's source list) that belong to another theme than their dossier's
GROUPS = {
    ("activations-mlp", "Transcoders and decomposition"): "features",
    ("circuits-framework", "Adjacent, for part 9's last section (another dossier's topic)"): "features",
    ("high-dim-probability", "Transformers: dynamics"): "attention",
    ("high-dim-probability", "Transformers: expressivity"): "attention",
    ("high-dim-probability", "Transformers: probability-flavored"): [
        (r"Superposition|Feature Representation", "features"), (r"Optimizer|Duality|Adam |Self-Regularization", "optimization"),
        (r"Hopfield|Softmax is not|Scalable-Softmax|nGPT|Attention", "attention"), (r"", "tokens")],
    ("speedrun-and-muon", "Architecture pieces named by the speedrun"): "block",
    ("normalization-residual-init", "Interpretability and the residual stream"): "circuits",
}
# the survey of 2026 spans every theme: its works are filed by the words of their titles, first match wins
FRONTIER = [
    (r"Interpretable|Circuit Extraction", "circuits"),
    (r"Biology of a Large Language Model", "features"),
    (r"memorize\?", "dynamics"),
    (r"Diffusion|Energy-Based", "language"),
    (r"Dynamic Chunking|Conditional Memory via Scalable Lookup", "tokens"),
    (r"Hyper-Connections|Attention Residuals|without Normalization|Looped|Recurrent Depth|Recursi|Hierarchical Reasoning"
     r"|Latent Thoughts|Continuous Latent|Superposition|Tiny Recursive|ARC Prize|HRM", "block"),
    (r"Attention|Delta|Mamba|IndexCache|Memoriz|Titans|ATLAS|Test-Time Training|Nested Learning|All Connected|Hybrid", "attention"),
    (r"", "training"),          # the model reports and cards: recipes
]
# works several dossiers list, whose home is not what the vote of their dossiers gives: by the words of the title
HOME = [
    (r"^Attention Is All You Need$|^GQA\b", "attention"),
    (r"^Improving Language Understanding by Generative Pre-Training$|^Language Models are (Unsupervised Multitask|Few-Shot) Learners$", "language"),
    (r"^Adam: A Method|^Decoupled Weight Decay|^Accurate, large minibatch SGD|^Modular Duality|^Heavy-Tailed Class Imbalance", "optimization"),
    (r"^Layer Normalization$|^Root mean square layer normalization$|^Transformer Feed-Forward Layers", "block"),
    (r"^RoFormer", "position"), (r"^Toy Models of Superposition$", "features"),
    (r"^In-context Learning and Induction Heads$", "circuits"), (r"^Exact solutions to the nonlinear dynamics", "dynamics"),
    (r"^Conditional Memory via Scalable Lookup", "tokens"),
    (r"Technical Report$|Herd of Models|Model Card$|^Gemma \d|^DeepSeek-V\d|^Kimi K\d|^GLM-\d|^SmolLM\d|^PaLM:", "training"),
]

YEAR = r"(?:1[89]\d\d|20[0-2]\d)"
ITEM = r"^ ?(?:\d{1,3}\.|-|\*|[A-Z]\d{1,2}\.)\s+(?=\S)"          # the start of a list entry (a continuation line is indented)
SPLIT = r"(?<=[^\s.]{2})\.\s+|(?<=[?!])\s+"                     # the end of a sentence, not of an initial
ARXIV = r"(?<![\d.])(\d{4}\.\d{4,5})(?:v\d+)?(?![\d])"
STOP = {"The", "This", "That", "These", "With", "From", "And", "For", "Not", "But", "Its", "Their", "When", "Where", "How", "Why",
        "What", "All", "Any", "Both", "New", "Old", "Used", "Same", "More", "Most", "Later", "Then", "Also", "See", "First",
        "Second", "Third", "Open", "Deep", "Large", "Small", "Section", "Table", "Figure", "Part", "Adam", "Muon", "Lion",
        "Transformer", "Attention", "Google", "Meta", "Anthropic", "May", "June", "July", "March", "April", "August", "January",
        "February", "September", "October", "November", "December"}


# ------------------------------------------------------------------------------------------------ reading entries
def _section(lines: list[str], number: int) -> list[str]:
    start = end = None
    for i, line in enumerate(lines):
        if re.match(rf"^## {number}\.", line):
            start = i
        elif start is not None and re.match(r"^## \d", line):
            end = i
            break
    return [] if start is None else lines[start + 1:end]


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _is_fetched(note: str) -> bool:
    f = re.sub(r"[*\"'`]", "", note).strip().lower()
    if not f:
        return False
    return not (re.match(r"(no|n|not|search|none)\b", f) or f.startswith("cited"))


def raw_entries(name: str, lines: list[str]) -> list[dict]:
    """The entries of a dossier's source list: {group, text (the citation), fetched (the note), cols (a table row's
    named cells) or None}."""
    out, group, i, sec = [], "", 0, _section(lines, 3)
    while i < len(sec):
        line = sec[i]
        s = line.strip()
        if s.startswith("|") and i + 1 < len(sec) and re.match(r"^\|\s*:?-", sec[i + 1].strip()):
            header = [h.lower() for h in _cells(s)]
            i += 2
            while i < len(sec) and sec[i].strip().startswith("|"):
                cells = _cells(sec[i])
                if len(cells) == len(header):
                    out.append({"group": group, "cols": dict(zip(header, cells)), "text": "", "fetched": ""})
                i += 1
            continue
        m = re.match(r"^(?:\*\*(.+?)\*\*:?\s*$|#{3,4}\s+(.+)$)", s)
        if m:
            group = (m.group(1) or m.group(2)).strip()
            i += 1
            continue
        if re.match(ITEM, line):
            text = re.sub(ITEM, "", line).strip()
            i += 1
            while i < len(sec) and sec[i].strip() and not re.match(ITEM, sec[i]) and not sec[i].strip().startswith(("|", "**", "#")) \
                    and (sec[i].startswith(" ") or not re.match(r"^[A-Z][a-z]+( [a-z,]+){0,5}$", sec[i].strip())):
                text += " " + sec[i].strip()
                i += 1
            out.append({"group": group, "cols": None, "text": text, "fetched": ""})
            continue
        if s and len(s) < 70 and not s.endswith(".") and not s.startswith(("\"", "(", "`")) and re.match(r"^[A-Z0-9]", s) and ":" not in s[:-1] and "=" not in s:
            group = s.rstrip(":")                                   # a plain sub-heading ("Classical optimizers")
        i += 1
    return out


def _first_year(text: str) -> int | None:
    clean = re.sub(r"arXiv[:\s]*\d{4}\.\d{4,5}(?:v\d+)?", " ", text)
    clean = re.sub(r"https?://\S+", " ", clean)
    clean = re.sub(ARXIV, " ", clean)
    m = re.search(rf"(?<![\d.,/-])({YEAR})(?![\d/]|\.\d|-\d\d\d)", clean)
    return int(m.group(1)) if m else None


def _identifier(text: str) -> str:
    m = re.search(r"arXiv[:\s]*" + ARXIV, text) or re.search(r"arxiv\.org/(?:abs|html|pdf)/" + ARXIV, text)
    if m:
        return "arXiv:" + m.group(1)
    m = re.search(r"https?://[^\s)>\]]+", text)
    return m.group(0).rstrip(".,;") if m else ""


def _clean(s: str) -> str:
    s = re.sub(r"[*`]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.strip(" .,;:")


def parse_citation(text: str) -> dict:
    """authors, title, venue, year, identifier out of one citation string."""
    ident = _identifier(text)
    body = text
    # the title: the first quoted or italic phrase that is not an aside in parentheses
    marks = [m for m in re.finditer(r"[\"“]([^\"”]{3,})[\"”]|(?<!\*)\*([^*]{3,})\*(?!\*)", body) if not body[:m.start()].rstrip().endswith("(")]
    if marks:
        m = marks[0]
        authors, title, rest = body[:m.start()], m.group(1) or m.group(2), body[m.end():]
    else:
        if True:
            parts = re.split(SPLIT, body)
            authors, title_parts, k = parts[0], [], 1
            if len(parts) > 1 and not re.match(r"arXiv|https?://|Fetched", parts[1]):
                title_parts, k = [parts[1]], 2                    # the sentence after the authors is the title
            while k < len(parts) and not re.search(rf"\b{YEAR}\b|arXiv|https?://|Fetched", parts[k]):
                title_parts.append(parts[k])
                k += 1
            title, rest = ". ".join(title_parts), ". ".join(parts[k:])
            if re.search(r"https?://|`", authors + title):         # a repository or a page, not a work with a title
                authors, title = "", ""
    venue = re.split(SPLIT + r"|https?://|arXiv[:\s]*\d|Fetched|Full text|PDF:|\bdoi:", rest.strip(" .,;"))[0] if rest else ""
    year = _first_year(rest) or _first_year(text)
    if year is None and ident.startswith("arXiv:"):
        yy = int(ident[6:8])
        year = 2000 + yy if yy < 90 else 1900 + yy
    return {"authors": _clean(authors), "title": _clean(title), "venue": _clean(venue.replace("†", "")), "year": year, "id": ident}


def read_entry(e: dict) -> dict | None:
    """One raw entry as a work, or None when it has no title or year."""
    if e["cols"] is not None:
        c = e["cols"]
        note = next((c[k] for k in c if k.startswith("fetched")), "")
        if "title" in c and any(k.startswith("authors") for k in c):
            venue_cell = next((c[k] for k in c if k.startswith("venue")), "")
            id_cell = next((c[k] for k in c if "id" in k.split() or k.startswith("id") or "url" in k or "arxiv" in k), "")
            ident = _identifier(id_cell) or ("arXiv:" + re.search(ARXIV, id_cell).group(1) if re.search(ARXIV, id_cell) else "")
            w = {"authors": _clean(next(c[k] for k in c if k.startswith("authors"))), "title": _clean(c["title"]),
                 "venue": _clean(re.sub(r"\((?:venue )?from memory\)|\(memory\)", "", venue_cell)), "year": _first_year(venue_cell), "id": ident}
            if w["year"] is None and ident.startswith("arXiv:"):
                yy = int(ident[6:8])
                w["year"] = 2000 + yy if yy < 90 else 1900 + yy
        else:
            key = next((k for k in c if k.startswith("work") or k.startswith("authors, title")), None)
            if key is None:
                return None
            extra = " ".join(c[k] for k in c if k in ("id", "where"))
            w = parse_citation(c[key] + (". " + extra if extra else ""))
    else:
        text = e["text"]
        m = re.search(r"Fetched:\s*(.*)$", text)
        if m:
            note = m.group(1)
        else:                                                     # "ar5iv: yes.", "Abstract page fetched ...: yes"
            note = "yes" if re.search(r":\s*yes\b", text) else ""
        w = parse_citation(text[:m.start()] if m else text)
    w["fetched"] = _is_fetched(note)
    w["note"] = note
    w["venue"] = clean_venue(w["venue"])
    return w


def clean_venue(v: str) -> str:
    """The venue as a reader can use it: nothing the dossier marks as unverified, no aside, no year, not "arXiv"
    (the identifier says that)."""
    if not v or re.search(r"memory|not verified|not confirmed|unverified|search|\?", v):
        return ""
    v = v.split(";")[0].split("(")[0]
    v = re.sub(rf"[,\s]*\b(?:\d{{1,2}} )?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)?[a-z]*\.? ?{YEAR}\s*$", "", v).strip(" ,;.")
    if len(v) > 60 or len(v) < 2 or re.match(r"(arxiv|yes|no|full|abstract|preprint)\b", v, re.I):
        return ""
    return v


def _names(authors: str) -> str:
    """The authors without asides: "OpenAI (Agarwal, Ahmad et al.)" -> "OpenAI"."""
    return re.sub(r"\s+", " ", re.sub(r"\s*\([^)]*(?:\)|$)", "", authors)).strip(" ,;")


def surname(authors: str) -> str:
    first = re.split(r",| and | & |;", _names(authors))[0].strip()
    words = [x for x in re.split(r"\s+", first) if x and not re.match(r"^[A-Z]\.?$", x) and x.lower() not in ("et", "al.", "al")]
    return words[-1].strip(".,") if words else ""


def first_author(authors: str, title: str) -> str:
    """How a work is named in a list: the first author's surname (with its particle), the whole name of a team or a
    company, or the first word of the title when the dossier names no author."""
    first = re.split(r",| and | & |;", _names(authors))[0].strip()
    first = re.sub(r"\s+et al\.?$", "", first)
    if not first:
        return title.split(":")[0].split(" ")[0]
    if re.search(r"\b(team|labs?|ai|research)$", first, re.I) or first.endswith("-AI") or " " not in first:
        return first
    words = first.split(" ")
    return (words[-2] + " " if len(words) > 1 and words[-2].lower() in ("von", "van", "de", "del", "der", "le", "la") else "") + words[-1]


def short_authors(authors: str) -> str:
    a = re.sub(r"\bet al$", "et al.", _names(authors))
    names = [x.strip() for x in re.split(r",\s*|\s+and\s+", a) if x.strip()]
    if len(names) > 3 and not re.search(r"et al|others", " ".join(names[:3])):
        return ", ".join(names[:3]) + " et al."
    return a


def title_key(title: str) -> str:
    return re.sub(r"[^a-z0-9]", "", title.lower())[:48]


# ------------------------------------------------------------------------------------------------ the works
def collect(folder: Path) -> tuple[list[dict], dict, dict]:
    stats = {"entries": 0, "not_fetched": 0, "unusable": 0}
    per = defaultdict(lambda: Counter())
    works, by_id, by_title, texts = [], {}, {}, {}
    order = sorted(DOSSIERS, key=lambda d: (DOSSIERS[d][1][0], d))
    for name in order:
        path = folder / f"{name}.md"
        lines = path.read_text(encoding="utf-8").split("\n")
        texts[name] = lines
        for e in raw_entries(name, lines):
            if e["cols"] is not None and ("organization" in e["cols"] or "item" in e["cols"]):
                continue                                           # the survey's table of configurations and cards
            if re.search(r"not fetched", e["group"], re.I):
                continue
            stats["entries"] += 1
            per[name]["entries"] += 1
            w = read_entry(e)
            if w is None or not w["title"] or not w["year"] or len(w["title"]) < 3 or not w["authors"] \
                    or re.search(r"config\.json|README|^same site$|\.py\b", w["title"]):
                stats["unusable"] += 1
                per[name]["unusable"] += 1
                continue
            if not w["fetched"]:
                stats["not_fetched"] += 1
                per[name]["not_fetched"] += 1
                continue
            theme = GROUPS.get((name, e["group"]), DOSSIERS[name][0])
            if theme is None or isinstance(theme, list):
                theme = next(t for pattern, t in (theme or FRONTIER) if re.search(pattern, w["title"]))
            # the Section that uses the entry: the dossier's first, or the theme's own for a group filed elsewhere
            section = next(s for k, _, s in THEMES if k == theme) if (name, e["group"]) in GROUPS else DOSSIERS[name][1][0]
            tk = title_key(w["title"])
            hit = by_id.get(w["id"]) if w["id"].startswith("arXiv:") else None
            if hit is None:
                hit = by_title.get(tk)
            if hit is None:
                hit = {"title": w["title"], "authors": w["authors"], "venue": w["venue"], "year": w["year"], "id": w["id"],
                       "themes": [], "sections": [], "dossiers": [], "surname": surname(w["authors"])}
                works.append(hit)
            else:
                if not hit["id"].startswith("arXiv:") and w["id"]:
                    hit["id"] = w["id"]
                if not hit["venue"] and w["venue"]:
                    hit["venue"] = w["venue"]
            if name not in hit["dossiers"]:
                hit["dossiers"].append(name)
                hit["themes"].append(theme)
                if section not in hit["sections"]:
                    hit["sections"].append(section)
            if hit["id"].startswith("arXiv:"):
                by_id[hit["id"]] = hit
            by_title.setdefault(tk, hit)
            per[name]["kept"] += 1
    keys = [k for k, _, _ in THEMES]
    for w in works:
        home = next((t for pattern, t in HOME if re.search(pattern, w["title"], re.I)), None)
        if home:
            w["theme"] = home
        else:
            votes = Counter(w["themes"])
            best = max(votes.values())
            w["theme"] = next(t for t in w["themes"] if votes[t] == best)      # dossiers were read in Section order
        own = THEMES[keys.index(w["theme"])][2]
        w["sections"] = ([own] if own in w["sections"] else []) + [s for s in w["sections"] if s != own]
    return works, texts, {"stats": stats, "per": per}


# ------------------------------------------------------------------------------------------------ the links
def find_links(works: list[dict], texts: dict) -> list[tuple[int, int, str]]:
    by_id = {w["id"]: i for i, w in enumerate(works) if w["id"].startswith("arXiv:")}
    names = defaultdict(list)                                   # first author's surname -> indices
    for i, w in enumerate(works):
        if w["surname"]:
            names[w["surname"]].append(i)

    NAME_RE = r"[A-Z][A-Za-zÀ-ÿ'’\-]+"
    GROUP_RE = rf"({NAME_RE})(?:(?:,| and| &) {NAME_RE})*(?:,? et al\.?)?,?\s*\(?$"   # "Lyu and Li (", "Jacot et al. "

    def resolve(word: str, year: int, dossier: str, slack: int) -> int | None:
        """The one work of this dossier whose first author is `word`, in `year` (or, failing that, within `slack`)."""
        local = [i for i in names.get(word, []) if dossier in works[i]["dossiers"]]
        exact = [i for i in local if works[i]["year"] == year]
        pick = exact or [i for i in local if abs(works[i]["year"] - year) <= slack]
        return pick[0] if len(pick) == 1 else None

    def by_number(cell: str) -> list[int]:
        return [by_id["arXiv:" + m.group(1)] for m in re.finditer(ARXIV, cell) if "arXiv:" + m.group(1) in by_id]

    def anchors_of(cell: str, dossier: str, years: list[int]) -> list[int]:
        """The row's own works: in each part of the cell, the first name that is a first author of this dossier."""
        found = by_number(cell)
        for part in re.sub(ARXIV, " ", cell).split(";"):
            for word in [x for x in re.findall(NAME_RE, part) if x not in STOP]:
                hits = [resolve(word, y, dossier, 0) for y in years]
                hits = [i for i in hits if i is not None] or [i for i in [resolve(word, y, dossier, 1) for y in years] if i is not None]
                if hits:
                    found.append(hits[0])
                    break
        return list(dict.fromkeys(found))

    def mentions(cell: str, dossier: str) -> list[int]:
        """Later works a cell names: "Hoffmann 2022", "Lyu and Li (2019)" (the first author counts), an arXiv number."""
        found = by_number(cell)
        clean = re.sub(ARXIV, " ", cell)
        for m in re.finditer(rf"(?<![\d.])({YEAR})(?![\d])", clean):
            window = clean[max(0, m.start() - 60):m.start()]
            window = window[max(window.rfind(";"), window.rfind(":"), window.rfind(" by ")) + 1:]
            year, g = int(m.group(1)), re.search(GROUP_RE, window)
            if g and g.group(1) not in STOP:
                i = resolve(g.group(1), year, dossier, 1)
            else:                                                 # "He init for ReLU (2015)": the nearest name, the exact year
                i = next((r for r in (resolve(x, year, dossier, 0) for x in reversed(re.findall(NAME_RE, window)) if x not in STOP) if r is not None), None)
            if i is not None:
                found.append(i)
        return list(dict.fromkeys(found))

    links = {}
    for dossier, lines in texts.items():
        sec = _section(lines, 2)
        k = 0
        while k < len(sec) - 1:
            if sec[k].strip().startswith("|") and re.match(r"^\|\s*:?-", sec[k + 1].strip()) and _cells(sec[k])[0].lower() == "year":
                k += 2
                while k < len(sec) and sec[k].strip().startswith("|"):
                    cells = _cells(sec[k])
                    k += 1
                    if len(cells) < 5:
                        continue
                    years = [int(y) for y in re.findall(rf"(?<![\d.])({YEAR})(?![\d])", cells[0])]
                    if not years:
                        continue
                    span = list(range(min(years), max(years) + 1)) if max(years) - min(years) <= 6 else years
                    anchors = anchors_of(cells[1], dossier, span)
                    later = [b for b in mentions(cells[-1], dossier) if b not in anchors]
                    for a in anchors:
                        for b in later:
                            if works[b]["year"] >= works[a]["year"]:
                                links.setdefault((b, a), dossier)
            else:
                k += 1
    return [(b, a, d) for (b, a), d in links.items()]


def slug(w: dict) -> str:
    if w["id"].startswith("arXiv:"):
        return w["id"]
    return re.sub(r"[^a-z0-9]+", "-", f"{w['surname']} {w['year']} {w['title']}".lower()).strip("-")[:60]


def build(folder: Path) -> tuple[dict, dict]:
    works, texts, info = collect(folder)
    links = find_links(works, texts)
    order = sorted(range(len(works)), key=lambda i: (works[i]["year"], works[i]["title"].lower()))
    new = {old: k for k, old in enumerate(order)}
    keys = [k for k, _, _ in THEMES]
    out, seen = [], set()
    for i in order:
        w = works[i]
        key = slug(w)
        while key in seen:
            key += "-b"
        seen.add(key)
        out.append({"k": key, "t": w["title"], "a": short_authors(w["authors"]), "f": first_author(w["authors"], w["title"]), "y": w["year"], "v": w["venue"], "x": w["id"],
                    "th": keys.index(w["theme"]), "s": w["sections"], "n": len(w["dossiers"])})
    pairs = sorted([new[b], new[a]] for b, a, _ in links)
    by_theme = [sum(1 for w in out if w["th"] == t) for t in range(len(THEMES))]
    counts = dict(info["stats"], works=len(out), links=len(pairs), by_theme=by_theme)
    data = {"generated_by": "gptpp.export.reading_map",
            "themes": [{"key": k, "label": lab, "section": s} for k, lab, s in THEMES],
            "sections": {str(k): v for k, v in SECTIONS.items()}, "works": out, "links": pairs, "counts": counts}
    info.update(works=works, links=links, new=new)
    return data, info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dossiers", default=str(WORK / "research"))
    ap.add_argument("--out", default=str(DATA))
    ap.add_argument("--show", default="", help="print for inspection: stats | works | links | multi | <dossier name>")
    args = ap.parse_args()
    data, info = build(Path(args.dossiers))
    size = write_json(Path(args.out) / f"{NAME}.json", data)
    c = data["counts"]
    print(f"{NAME}.json: {c['works']} works from {c['entries']} entries ({c['not_fetched']} not fetched, {c['unusable']} without "
          f"title or year), {c['links']} links, {size:,} bytes")
    print("  by theme: " + ", ".join(f"{t['label']} {n}" for t, n in zip(data["themes"], c["by_theme"])))
    if args.show == "stats":
        for name, s in sorted(info["per"].items()):
            print(f"  {name:32} entries {s['entries']:3}  kept {s['kept']:3}  not fetched {s['not_fetched']:3}  unusable {s['unusable']:3}")
    elif args.show == "works":
        for w in data["works"]:
            print(f"{w['y']} [{data['themes'][w['th']]['key'][:5]:5}] {w['a'][:34]:34} | {w['t'][:70]:70} | {w['v'][:24]:24} | {w['x'][:28]} | {w['s']}")
    elif args.show == "links":
        works = info["works"]
        for b, a, d in sorted(info["links"], key=lambda q: (q[2], works[q[1]]["year"])):
            print(f"{d[:14]:14} {works[a]['surname'][:12]:12} {works[a]['year']} {works[a]['title'][:38]:38} <- {works[b]['surname'][:12]:12} {works[b]['year']} {works[b]['title'][:44]}")
    elif args.show == "multi":
        for w in info["works"]:
            if len(set(w["themes"])) > 1:
                print(f"{w['year']} {w['title'][:60]:60} {w['id'][:18]:18} -> {w['theme']:12} of {w['themes']}")
    elif args.show:
        for w in info["works"]:
            if args.show in w["dossiers"]:
                print(f"{w['year']} | {w['authors'][:40]:40} | {w['title'][:64]:64} | {w['venue'][:30]:30} | {w['id'][:30]}")


if __name__ == "__main__":
    main()
