"""Export for fig-01-02 `gpt-guessing-game`: Shannon's guessing game of 1951 with tokens for letters, and two
machines as the other subjects.

    python -m gptpp.export.guessing_game [--stories 1000] [--out <dir>]

For each sentence the page offers (Shannon's own, and the openings of three held-out TinyStories stories) and for
each of its tokens: the guess number of the token under two predictors, that is, one plus the number of tokens the
predictor holds likelier; the surprisal of the token, minus log2 of the probability it gave it; and its eight first
guesses with their probabilities.

    the table   the bigram table the page ships (`token_pair_table.py`): it sees the one token before
    pico        the network the page runs (the shipped weights): it sees the sentence so far, from `<|bos|>`

The first token of a sentence is guessed after `<|bos|>` by both. Then the same two subjects play on held-out
stories at large (the first `--stories` of them, each from its start token, windows of the model's context), and
the frequencies q_i of their guess numbers go into Shannon's two bounds (1951, equation 17),

    sum_i i (q_i - q_{i+1}) log2 i   <=   entropy   <=   - sum_i q_i log2 q_i,

next to the loss each predictor really has on those tokens and to the price of its guess numbers under the game's
own charge, log2 r(r + 1) bits for the guess number r (the code length under the prior 1 / (r (r + 1))).

Shannon's sentence is example (9) of "Prediction and Entropy of Printed English" (Bell System Technical Journal 30,
1951, p. 56), set in sentence case with its two full stops; `SUBJECT` is the row of guess numbers printed under it
there, one for each of its 102 letters and spaces (as read from the scan in the research dossier).

Writes `guessing-game.json`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..tokenizer import Tokenizer
from .browser_model import page_model
from .common import DATA, write_atomic
from .counted_models import PairTable, story_texts

NAME = "guessing-game"
TOP = 8
SENTENCES = [
    ("lucy", "a story", "Once upon a time, there was a little girl named Lucy. She had a black cat named Shadow.", 13),
    ("garden", "another", "Tom and Mia are playing in the garden. They have a big pot and some seeds.", 4),
    ("pastry", "a third", "One day, a little bird named Tim was flying in the sky. He saw a pastry on a window sill.", 6),
    ("shannon", "Shannon's, 1951", "There is no reverse on a motorcycle. A friend of mine found this out rather dramatically the other day.", 3),
]
SUBJECT = [1, 1, 1, 5, 1, 1, 2, 1, 1, 2, 1, 1, 15, 1, 17, 1, 1, 1, 2, 1, 3, 2, 1, 2, 2, 7, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 3, 1,
           8, 6, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 6, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1,
           4, 1, 1, 1, 1, 1, 1, 11, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


def guesses(q: np.ndarray, token: int) -> tuple[int, float, list[int], list[float]]:
    """The guess number of `token` under the distribution q (ties: the smaller id is guessed first), its
    probability, and the first TOP guesses with theirs."""
    order = np.lexsort((np.arange(len(q)), -q))[:TOP]
    rank = int((q > q[token]).sum() + (q[:token] == q[token]).sum()) + 1
    return rank, float(q[token]), [int(i) for i in order], [float(q[i]) for i in order]


def bounds(ranks: np.ndarray, vocab: int) -> tuple[float, float]:
    """Shannon's lower and upper bound, in bits per token, from the guess numbers of many tokens."""
    q = np.bincount(ranks, minlength=vocab + 2)[1:vocab + 2] / len(ranks)        # q[i - 1]: the share of guess number i; q[vocab] = 0
    i = np.arange(1, vocab + 1)
    lower = float((i * (q[:-1] - q[1:]) * np.log2(i)).sum())
    upper = float(-(q[q > 0] * np.log2(q[q > 0])).sum())
    return lower, upper


@torch.no_grad()
def play(model, table: PairTable, documents: list[list[int]], bos: int, window: int) -> dict:
    """Both subjects on whole documents: the guess number and the probability of every token."""
    rank_table = np.empty((table.vocab, table.vocab), dtype=np.int16)        # rank_table[v, w]: the guess number of w after v
    prob_table = np.empty((table.vocab, table.vocab), dtype=np.float32)
    for v in range(table.vocab):
        q = table.distribution(v)
        order = np.lexsort((np.arange(table.vocab), -q))
        rank_table[v, order] = np.arange(1, table.vocab + 1)
        prob_table[v] = q
    out = {"table": {"rank": [], "p": []}, "pico": {"rank": [], "p": []}}
    pieces = []
    for tokens in documents:
        for start in range(0, len(tokens), window - 1):
            pieces.append([bos] + list(tokens[start:start + window - 1]))
    pieces.sort(key=len, reverse=True)
    done = 0
    while done < len(pieces):
        width = len(pieces[done])
        rows = pieces[done:done + max(1, (1 << 12) // width)]
        idx = torch.full((len(rows), width), bos, dtype=torch.long)
        for r, piece in enumerate(rows):
            idx[r, :len(piece)] = torch.as_tensor(piece)
        p = torch.softmax(model(idx)[:, :-1].float(), dim=-1)
        target = idx[:, 1:]
        p_true = p.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        rank = (p > p_true.unsqueeze(-1)).sum(-1) + 1
        for r, piece in enumerate(rows):
            n = len(piece) - 1
            out["pico"]["rank"].append(rank[r, :n].numpy()); out["pico"]["p"].append(p_true[r, :n].double().numpy())
            a, b = np.array(piece[:-1]), np.array(piece[1:])
            out["table"]["rank"].append(rank_table[a, b].astype(np.int64)); out["table"]["p"].append(prob_table[a, b].astype(np.float64))
        done += len(rows)
    return {k: {"rank": np.concatenate(v["rank"]), "p": np.concatenate(v["p"])} for k, v in out.items()}


def export(out: Path, n_stories: int) -> dict:
    torch.set_num_threads(min(16, torch.get_num_threads()))
    tok = Tokenizer.load("bpe4k")
    model, cfg = page_model("pico", out)
    table = PairTable.load(out=out)
    stories = story_texts()
    text_of = lambda i: "(the end)" if i == tok.bos_id else tok.decode([i])

    sentences = []
    for key, label, text, given in SENTENCES:
        ids = tok.encode(text)
        seq = [tok.bos_id] + ids
        with torch.no_grad():
            p = torch.softmax(model(torch.tensor([seq]))[0, :-1].double(), dim=-1).numpy()
        players = {"table": {"rank": [], "bits": [], "top": [], "topP": []}, "pico": {"rank": [], "bits": [], "top": [], "topP": []}}
        for k, t in enumerate(ids):
            for name, q in (("table", table.distribution(seq[k])), ("pico", p[k])):
                rank, prob, top, top_p = guesses(q, t)
                players[name]["rank"].append(rank); players[name]["bits"].append(round(float(-np.log2(prob)), 4))
                players[name]["top"].append([text_of(i) for i in top]); players[name]["topP"].append([round(v, 5) for v in top_p])
        players["table"]["topIds"] = [[int(i) for i in guesses(table.distribution(seq[k]), t)[2]] for k, t in enumerate(ids)]
        where = next((n for n, s in enumerate(stories) if s.startswith(text)), None)
        if key != "shannon" and where is None:
            raise SystemExit(f"the sentence {text!r} does not open a held-out story")
        item = {"key": key, "label": label, "text": text, "given": given, "tokens": ids, "texts": [tok.decode([i]) for i in ids],
                "source": "Shannon 1951, example (9), in sentence case" if key == "shannon" else f"the opening of held-out story {where + 1}", **players}
        if key == "shannon":
            item["subject"] = {"numbers": SUBJECT, "letters": len(SUBJECT), "first": int(sum(1 for r in SUBJECT if r == 1))}
        sentences.append(item)

    docs = [s for s in stories[:n_stories] if s]
    game = play(model, table, tok.encode_batch(docs), tok.bos_id, cfg.block_size)
    n_tokens, n_chars = len(game["pico"]["rank"]), sum(len(s) for s in docs)
    per_char = n_tokens / n_chars
    held = {"stories": len(docs), "tokens": n_tokens, "characters": n_chars}
    for name in ("table", "pico"):
        r, p = game[name]["rank"], game[name]["p"]
        lower, upper = bounds(r, table.vocab)
        held[name] = {"first": float((r == 1).mean()), "within5": float((r <= 5).mean()),
                      "lower": lower * per_char, "upper": upper * per_char, "loss": float(-np.log2(p).sum() / n_chars),
                      "charge": float(np.log2(r * (r + 1.0)).sum() / n_chars)}
    data = {"name": NAME, "vocabulary": table.vocab, "top": TOP, "charge": "log2 r(r + 1) bits for the guess number r",
            "sentences": sentences, "heldOut": held}
    write_atomic(out / f"{NAME}.json", json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    return {"heldOut": held, "sentences": [{"key": s["key"], "tokens": len(s["tokens"]), "table": s["table"]["rank"], "pico": s["pico"]["rank"]} for s in sentences]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stories", type=int, default=1000, help="held-out stories the two subjects are measured on")
    ap.add_argument("--out", default=str(DATA))
    args = ap.parse_args()
    print(json.dumps(export(Path(args.out), args.stories)))


if __name__ == "__main__":
    main()
