"""Tests of the from-scratch byte-pair encoding (`gptpp/bpe.py`). Pure Python, no data needed."""

from __future__ import annotations

import random

import regex

from gptpp import bpe

WORDS = (
    "the of and to a in is that it was for on are as with his they at be this from have or by one had not but "
    "what all were when we there can an your which their said if do will each about how up out them then she many "
    "some so these would other into has more her two like him see time could no make than first been its who now "
    "people my made over did down only way find use may water long little very after words called just where most "
    "know model token attention gradient layer learning language probability entropy matrix vector"
).split()
RARE = ["naïve", "café", "Zürich", "Gödel", "über", "注意力", "模型", "трансформер", "λ-calculus", "🙂", "👩‍🔬", "résumé"]

# Strings that a tokenizer must take and give back unchanged.
TRICKY = [
    "",
    " ",
    "   leading and trailing   ",
    "Hello world! It's 2026, isn't it? We'LL see; they'VE said so.",
    "line one\nline two\r\nline three\n\n\nparagraph\t tab  \n   indented",
    "3.14159 and 1,000,000 and 0xDEADBEEF and 12345678901234567890",
    "naïve café, Zürich; Gödel's proof — “quoted” ‘single’ … ellipsis",
    "注意力就是你所需要的一切。 自己注意力",
    "Трансформер и внимание; Ελληνικά; עברית; العربية; हिन्दी",
    "🙂 👩‍🔬 🇨🇭 👍🏽 e\u0301 (combining) ﬁ (ligature) \u00a0 (no-break space) \u200b (zero width)",
    "\x00\x01\x02 control \x7f \x85 \u2028 \u2029 \ufeff",
    "<|bos|> <|endoftext|> <|user_start|> are plain text here",
    "x" * 500 + " " * 300 + "\n" * 50 + "9" * 100,
    "\u017f \u017ft '\u017ftuff \u212a's it'\u017f (the long s and the Kelvin sign fold to s and k)",
]


def make_corpus(n_docs: int = 200, seed: int = 0) -> list[str]:
    """A small English-like corpus with a Zipfian vocabulary, numbers, contractions and some Unicode."""
    rng = random.Random(seed)
    weights = [1 / (rank + 1) for rank in range(len(WORDS))]
    docs = []
    for _ in range(n_docs):
        sentences = []
        for _ in range(rng.randint(3, 9)):
            words = rng.choices(WORDS, weights, k=rng.randint(4, 16))
            if rng.random() < 0.3:
                words.insert(rng.randrange(len(words)), rng.choice(RARE))
            if rng.random() < 0.3:
                words.insert(rng.randrange(len(words)), str(rng.choice([7, 42, 1994, 2017, 65536, 3.5, "1,024"])))
            if rng.random() < 0.3:
                words.insert(rng.randrange(len(words)), rng.choice(["don't", "it's", "we'll", "they've", "I'm"]))
            sentence = " ".join(words)
            sentences.append(sentence[0].upper() + sentence[1:] + rng.choice([".", ".", ".", "?", "!", ";", ":\n"]))
        docs.append(rng.choice([" ", " ", "\n", "\n\n"]).join(sentences))
    return docs


CORPUS = make_corpus()


def random_unicode(rng: random.Random, length: int) -> str:
    """Characters from all over Unicode (no surrogates, which are not text)."""
    planes = [(0x00, 0x7F), (0x80, 0x7FF), (0x800, 0xD7FF), (0xE000, 0xFFFF), (0x10000, 0x10FFFF)]
    return "".join(chr(rng.randint(*rng.choice(planes))) for _ in range(length))


def test_split_pattern() -> None:
    split = regex.compile(bpe.GPT4_PATTERN)
    assert split.findall("Hello world! It's 2026.") == ["Hello", " world", "!", " It", "'s", " ", "202", "6", "."]
    assert split.findall("a  b") == ["a", " ", " b"]  # the last space goes with the word
    assert split.findall("end.\n\nNext") == ["end", ".\n\n", "Next"]  # line breaks stay with the punctuation
    assert split.findall("x \n y") == ["x", " \n", " y"]
    assert split.findall("we'LL don’t") == ["we", "'LL", " don", "’t"]  # only the ASCII apostrophe contracts
    rng = random.Random(1)
    for text in TRICKY + CORPUS[:20] + [random_unicode(rng, 200) for _ in range(50)]:
        assert "".join(split.findall(text)) == text  # the pieces tile the text: nothing is dropped


def test_training_by_hand() -> None:
    # The textbook example. Pairs of "aaabdaaabac": aa 4, ab 2, others 1, so aa merges first (256).
    # Then ab and (256, a) both occur twice; the tie goes to the smaller ids, (97, 98) = ab (257).
    # Then (256, 257) = "aaab" occurs twice (258).
    merges = bpe.train("aaabdaaabac", vocab_size=256 + 3, special_tokens=[])
    assert merges == [(97, 97), (97, 98), (256, 257)]
    assert bpe.vocabulary(merges, special_tokens=[])[256:] == [b"aa", b"ab", b"aaab"]
    assert bpe.encode("aaabdaaabac", merges) == [258, 100, 258, 97, 99]
    assert bpe.decode([258, 100, 258, 97, 99], merges) == "aaabdaaabac"


def test_merge_scans_from_the_left() -> None:
    assert bpe.merge([1, 1, 1, 1, 1], (1, 1), 9) == [9, 9, 1]
    assert bpe.merge([1, 2, 1, 2, 3], (1, 2), 9) == [9, 9, 3]
    assert bpe.merge([], (1, 2), 9) == []


def test_merges_never_cross_a_piece_boundary() -> None:
    # Every piece of this text is one byte, so no pair exists inside a piece and nothing can be learned.
    assert regex.findall(bpe.GPT4_PATTERN, "1.2.1.2.") == ["1", ".", "2", ".", "1", ".", "2", "."]
    assert bpe.train("1.2.1.2.1.2.1.2.", vocab_size=300, special_tokens=[]) == []
    # A word takes the single character before it, whatever it is: here ".a" is a piece, and is learned.
    assert bpe.train("a.a.a.a.a.a.a.", vocab_size=300, special_tokens=[]) == [(46, 97)]
    # " the" is learned inside the piece; "e t" never is, however often "the the" occurs.
    merges = bpe.train("the the the the the the", vocab_size=262, special_tokens=[])
    vocab = bpe.vocabulary(merges, special_tokens=[])
    assert b" the" in vocab and not any(b"e " in token for token in vocab[256:])


def test_vocabulary_layout_and_early_stop() -> None:
    merges = bpe.train(CORPUS, vocab_size=400)
    assert len(merges) == 400 - 256 - len(bpe.SPECIAL_TOKENS)
    vocab = bpe.vocabulary(merges)
    assert len(vocab) == 400
    assert vocab[:256] == [bytes([b]) for b in range(256)]
    assert vocab[-5:] == [name.encode() for name in bpe.SPECIAL_TOKENS]
    assert all(vocab[256 + k] == vocab[a] + vocab[b] and a < 256 + k and b < 256 + k for k, (a, b) in enumerate(merges))
    assert len(set(vocab)) == len(vocab)  # no two tokens spell the same bytes
    # A corpus too small for the vocabulary asked: training stops when every word is one token.
    assert len(bpe.train("ab ab abc", vocab_size=1000)) == 3  # ab, " ab", " abc"... and nothing more


def test_fast_training_learns_the_same_merges() -> None:
    for vocab_size in (300, 700, 2000):
        assert bpe.train_fast(CORPUS, vocab_size) == bpe.train(CORPUS, vocab_size)
    assert bpe.train_fast(TRICKY, 600) == bpe.train(TRICKY, 600)  # many ties, long runs of one byte


def test_round_trip() -> None:
    tok = bpe.BPE.train(CORPUS + TRICKY, vocab_size=1000)
    rng = random.Random(2)
    for text in TRICKY + CORPUS[:50] + [random_unicode(rng, rng.randint(1, 300)) for _ in range(200)]:
        ids = tok.encode(text)
        assert tok.decode(ids) == text
        assert ids == bpe.encode(text, tok.merges)  # the class and the function agree
        assert all(0 <= i < tok.vocab_size - len(tok.special) for i in ids)
    assert bpe.decode(tok.encode(TRICKY[3]), tok.merges) == TRICKY[3]


def test_encoding_applies_the_earliest_merge_first() -> None:
    merges = [(97, 98), (98, 99), (256, 99)]  # ab, bc, abc
    assert bpe.encode("abc", merges) == [258]  # ab first, then ab+c; bc is never formed
    assert bpe.encode("bc", merges) == [257]
    merges = [(98, 99), (97, 98), (97, 256)]  # bc, ab, a+bc
    assert bpe.encode("abc", merges) == [258]  # bc first, then a+bc
    assert bpe.encode("abx", merges) == [257, 120]


def test_special_tokens() -> None:
    tok = bpe.BPE.train(CORPUS, vocab_size=500)
    assert list(tok.special) == bpe.SPECIAL_TOKENS
    assert list(tok.special.values()) == [495, 496, 497, 498, 499]
    assert tok.bos_id == 495 and tok.token_bytes(495) == b"<|bos|>"
    assert tok.encode("hello", bos=True) == [495] + tok.encode("hello")
    assert tok.decode(tok.encode("hello", bos=True)) == "<|bos|>hello"
    # Text that spells a special token is text: no special id comes out of `encode`.
    assert all(i < 495 for i in tok.encode("<|bos|><|user_start|>hi<|assistant_end|>"))


def test_decode_tolerates_a_character_cut_in_two() -> None:
    tok = bpe.BPE([])
    ids = tok.encode("é")
    assert ids == [0xC3, 0xA9]
    assert tok.decode(ids[:1]) == "\ufffd"


def test_merges_file(tmp_path) -> None:
    merges = bpe.train(CORPUS, vocab_size=400)
    path = str(tmp_path / "merges.txt")
    bpe.save_merges(merges, path)
    assert bpe.load_merges(path) == merges
