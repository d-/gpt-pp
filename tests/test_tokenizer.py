"""Tests of the production tokenizer (`gptpp/tokenizer.py`) and of the data it wrote.

The first half needs no data: the reference and the production tokenizer are trained on a small
corpus and must agree merge for merge and id for id. The second half checks the trained tokenizers
and the token shards under $GPTPP_WORK/data, and is skipped where they have not been prepared.
"""

from __future__ import annotations

import json
import os
import random
import unicodedata

import numpy as np
import pytest
import regex
import tokenizers
from tokenizers import Regex, pre_tokenizers

from gptpp import bpe, work_dir
from gptpp.tokenizer import Tokenizer, byte_alphabet, tokenizer_dir, train_merges
from test_bpe import CORPUS, TRICKY, random_unicode

DATA = work_dir() / "data"  # $GPTPP_WORK/data
CODE_POINTS = [cp for cp in range(0x110000) if not 0xD800 <= cp <= 0xDFFF]  # every character there can be


# ----------------------------------------------------------------------------------------------------
# The reference and the production tokenizer are the same tokenizer
# ----------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    return Tokenizer.train(CORPUS + TRICKY, vocab_size=1000, name="test", workers=2)


def test_reference_and_production_learn_the_same_merges() -> None:
    for vocab_size in (300, 700, 2000):
        merges, counted = train_merges(CORPUS, vocab_size, workers=2)
        assert merges == bpe.train(CORPUS, vocab_size)  # the plain loop, recounting every round
        assert merges == bpe.train_fast(CORPUS, vocab_size)
        assert counted == {"docs": len(CORPUS), "chars": sum(map(len, CORPUS))}
    # Strings with long runs of one byte, control characters and many equally frequent pairs.
    assert train_merges(TRICKY, 600, workers=2)[0] == bpe.train(TRICKY, 600)
    # Fewer pairs than merges asked for: both stop at the same point.
    assert train_merges("ab ab abc", 1000, workers=1)[0] == bpe.train("ab ab abc", 1000)


def test_reference_and_production_encode_identically(tok: Tokenizer, engines_disagree: set[int]) -> None:
    reference = bpe.BPE(tok.merges)
    rng = random.Random(3)
    alike = [cp for cp in CODE_POINTS if cp not in engines_disagree]
    texts = TRICKY + CORPUS + ["".join(chr(rng.choice(alike)) for _ in range(rng.randint(1, 300))) for _ in range(300)]
    for text in texts:
        assert tok.encode(text) == reference.encode(text)
        assert tok.encode(text, bos=True) == reference.encode(text, bos=True)
    assert tok.encode(TRICKY[3]) == bpe.encode(TRICKY[3], tok.merges)  # and the plain function
    assert [tok.token_bytes(i) for i in range(tok.vocab_size)] == reference.vocab
    assert (tok.vocab_size, tok.special, tok.bos_id) == (reference.vocab_size, reference.special, reference.bos_id)


def test_round_trip(tok: Tokenizer) -> None:
    rng = random.Random(4)
    for text in TRICKY + CORPUS[:50] + [random_unicode(rng, rng.randint(1, 300)) for _ in range(300)]:
        assert tok.decode(tok.encode(text)) == text


@pytest.fixture(scope="module")
def engine_classes() -> dict[str, tuple[set[int], set[int]]]:
    """The letters, the digits and the whitespace characters according to each regular expression engine.

    The pattern asks three things of a character: is it a letter (\\p{L}), a digit (\\p{N}), whitespace
    (\\s)? `bpe.py` asks Python's `regex` module, `tokenizers` asks Oniguruma, and each carries its own
    copy of the Unicode tables. Returns {pattern: (code points for Python, code points for Oniguruma)}.
    """
    text = "".join(map(chr, CODE_POINTS))
    classes = {}
    for pattern in (r"\p{L}", r"\p{N}", r"\s"):
        python = set(map(ord, regex.findall(pattern, text)))
        unmatched = pre_tokenizers.Split(Regex(pattern), behavior="removed").pre_tokenize_str(text)
        missed = {CODE_POINTS[i] for _, (a, b) in unmatched for i in range(a, b)}
        classes[pattern] = (python, set(CODE_POINTS) - missed)
    return classes


@pytest.fixture(scope="module")
def engines_disagree(engine_classes: dict[str, tuple[set[int], set[int]]]) -> set[int]:
    """The code points that the two engines do not put in the same class."""
    return set().union(*(python ^ oniguruma for python, oniguruma in engine_classes.values()))


def test_the_two_regex_engines_differ_only_in_their_unicode_tables(
    engine_classes: dict[str, tuple[set[int], set[int]]], engines_disagree: set[int]
) -> None:
    # 1. Put every code point of planes 0 to 3 and 14 (the others hold no letter, digit or space) in
    #    six contexts and cut the text with both engines: wherever the cuts differ, the character is
    #    one that the engines classify differently.
    python = regex.compile(bpe.GPT4_PATTERN)
    library = pre_tokenizers.Split(Regex(bpe.GPT4_PATTERN), behavior="isolated")
    code_points = [cp for cp in CODE_POINTS if cp < 0x40000 or 0xE0000 <= cp < 0xE1000]
    assert engines_disagree <= set(code_points)
    for template in ("a{}b", " {} ", "'{}", "1{}2", ".{}.", "{}"):
        width = len(template.format("x")) + 1  # the probes are joined by line breaks
        text = "\n".join(template.format(chr(cp)) for cp in code_points)
        cuts_python = {match.end() for match in python.finditer(text)}
        cuts_library = {end for _, (_, end) in library.pre_tokenize_str(text)}
        differ = {code_points[(position - 1) // width] for position in cuts_python ^ cuts_library}
        assert differ <= engines_disagree, template
    # 2. Both engines know the same 25 whitespace characters.
    python_space, oniguruma_space = engine_classes[r"\s"]
    assert python_space == oniguruma_space and len(python_space) == 25
    # 3. Every character that Python's own tables know (Unicode 15.1 for Python 3.13) is classified
    #    alike: the engines differ only on characters added to Unicode since (measured with regex
    #    2026.11.9 and tokenizers 0.23.3: 17,480 letters and digits that the regex module already knows).
    if tuple(map(int, unicodedata.unidata_version.split("."))) <= (16, 0, 0):
        assert all(unicodedata.category(chr(cp)) == "Cn" for cp in engines_disagree)


# ----------------------------------------------------------------------------------------------------
# The contract
# ----------------------------------------------------------------------------------------------------


def test_contract_api(tok: Tokenizer) -> None:
    assert tok.vocab_size == 1000 and tok.bos_id == 995
    assert tok.special == {name: 995 + k for k, name in enumerate(bpe.SPECIAL_TOKENS)}
    ids = tok.encode("Hello world", bos=True)
    assert isinstance(ids, list) and all(isinstance(i, int) for i in ids) and ids[0] == tok.bos_id
    assert tok.encode("Hello world") == ids[1:]
    assert tok.decode(ids) == "<|bos|>Hello world"
    assert b"".join(tok.token_bytes(i) for i in ids[1:]) == b"Hello world"
    assert [tok.token_bytes(b) for b in range(256)] == [bytes([b]) for b in range(256)]
    texts = CORPUS[:40] + TRICKY
    expected = [tok.encode(text, bos=True) for text in texts]
    assert tok.encode_batch(texts, bos=True) == expected
    assert tok.encode_batch(texts, bos=True, threads=1) == expected
    assert tok.encode_batch(texts, bos=True, threads=4) == expected
    assert tok.encode_batch(texts) == [ids[1:] for ids in expected]
    assert tok.encode_batch([]) == []


def test_text_is_never_read_as_special_tokens(tok: Tokenizer) -> None:
    text = "<|bos|><|user_start|>hello<|user_end|><|assistant_start|><|assistant_end|>"
    ids = tok.encode(text)
    assert all(i < tok.bos_id for i in ids) and tok.decode(ids) == text
    turn = [tok.special["<|user_start|>"], *tok.encode("hi"), tok.special["<|user_end|>"]]
    assert tok.decode(turn) == "<|user_start|>hi<|user_end|>"


def test_byte_alphabet_is_the_library_alphabet() -> None:
    alphabet = byte_alphabet()
    assert len(set(alphabet)) == 256 and set(alphabet) == set(pre_tokenizers.ByteLevel.alphabet())
    byte_level = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    seen: set[int] = set()
    bmp = [*range(0x800), *range(0x800, 0xD800, 0x400), *range(0xE000, 0x10000, 0x400), 0xFFFF]
    samples = [*bmp, *range(0x10000, 0x110000, 0x8000), 0x10FFFF]
    for cp in samples:
        encoded = chr(cp).encode("utf-8")
        seen.update(encoded)
        assert "".join(piece for piece, _ in byte_level.pre_tokenize_str(chr(cp))) == "".join(alphabet[b] for b in encoded)
    assert len(seen) == 243  # every byte that UTF-8 can produce (0xC0, 0xC1 and 0xF5..0xFF never occur)


def test_save_and_load(tok: Tokenizer, tmp_path, monkeypatch) -> None:
    folder = tok.save(tmp_path / "tokenizers" / "mine")
    assert {p.name for p in folder.iterdir()} == {"tokenizer.json", "merges.txt", "meta.json"}
    meta = json.loads((folder / "meta.json").read_text())
    assert meta["vocab_size"] == 1000 and meta["special_tokens"]["<|bos|>"] == 995 and meta["pattern"] == bpe.GPT4_PATTERN
    assert meta["merges_file"] == "merges.txt" and meta["n_chars"] == sum(map(len, CORPUS + TRICKY))
    again = Tokenizer.load(folder)
    assert again.merges == tok.merges and again.special == tok.special and again.info["n_chars"] == tok.info["n_chars"]
    assert all(again.encode(text, bos=True) == tok.encode(text, bos=True) for text in TRICKY)
    # By name, under $GPTPP_WORK/data/tokenizers.
    monkeypatch.setenv("GPTPP_WORK", str(tmp_path / "work"))
    assert tok.save("named") == tmp_path / "work" / "data" / "tokenizers" / "named" == tokenizer_dir("named")
    assert Tokenizer.load("named").merges == tok.merges
    # The file alone, read by the library with none of our code, gives the same ids and the same text.
    plain = tokenizers.Tokenizer.from_file(str(folder / "tokenizer.json"))
    for text in TRICKY + CORPUS[:20]:
        assert plain.encode(text).ids == tok.encode(text)
        assert plain.decode(plain.encode(text).ids) == text
    assert plain.get_vocab_size() == 1000 and plain.token_to_id("<|bos|>") == 995


# ----------------------------------------------------------------------------------------------------
# The trained tokenizers and the shards on disk (skipped where they do not exist)
# ----------------------------------------------------------------------------------------------------

PREPARED = [("tinystories", "bpe4k", 4096), ("fineweb-edu", "bpe32k", 32768)]


def held_out_texts(dataset: str, limit: int) -> list[str]:
    with open(DATA / dataset / "val_docs.jsonl", encoding="utf-8") as f:
        return [json.loads(line)["text"] for line, _ in zip(f, range(limit))]


@pytest.mark.parametrize("dataset, name, vocab_size", PREPARED)
def test_trained_tokenizer(dataset: str, name: str, vocab_size: int, engines_disagree: set[int]) -> None:
    if not (DATA / "tokenizers" / name / "meta.json").exists() or not (DATA / dataset / "val_docs.jsonl").exists():
        pytest.skip(f"{name} or {dataset} is not prepared")
    tok = Tokenizer.load(name)
    assert tok.vocab_size == vocab_size and len(tok.merges) == vocab_size - 256 - 5
    assert list(tok.special.values()) == list(range(vocab_size - 5, vocab_size)) and tok.bos_id == vocab_size - 5
    meta = json.loads((DATA / "tokenizers" / name / "meta.json").read_text())
    assert meta["pattern"] == bpe.GPT4_PATTERN and meta["special_tokens"] == tok.special and meta["n_chars"] > 1e9
    # The pure-Python reference, given the merges, encodes real held-out documents to the same ids.
    reference = bpe.BPE(tok.merges)
    texts = held_out_texts(dataset, 300)
    for text, ids in zip(texts, tok.encode_batch(texts, bos=True)):
        assert tok.decode(ids[1:]) == text
        if not any(ord(c) in engines_disagree for c in text):
            assert reference.encode(text, bos=True) == ids


def test_reference_trainer_on_real_text() -> None:
    if not (DATA / "tinystories" / "val_docs.jsonl").exists():
        pytest.skip("tinystories is not prepared")
    stories = held_out_texts("tinystories", 2000)
    assert train_merges(stories, 4096, workers=4)[0] == bpe.train_fast(stories, 4096)


@pytest.mark.parametrize("dataset, name, vocab_size", PREPARED)
def test_shards_obey_the_contract(dataset: str, name: str, vocab_size: int) -> None:
    folder = DATA / dataset / name
    if not (folder / "meta.json").exists():
        pytest.skip(f"{dataset}/{name} is not prepared")
    meta = json.loads((folder / "meta.json").read_text())
    tok = Tokenizer.load(name)
    for key in ("dataset", "tokenizer", "vocab_size", "bos_id", "n_train_tokens", "n_val_tokens", "train_shards",
                "val_bytes", "val_docs", "bytes_per_token_val", "source", "license"):
        assert key in meta, key
    assert (meta["dataset"], meta["tokenizer"], meta["vocab_size"], meta["bos_id"]) == (dataset, name, vocab_size, tok.bos_id)

    # Training shards: uint16, about 100M tokens each, whole documents, ids inside the vocabulary.
    sizes = [os.path.getsize(folder / shard) // 2 for shard in meta["train_shards"]]
    assert sorted(p.name for p in folder.glob("train_*.bin")) == meta["train_shards"]
    assert sum(sizes) == meta["n_train_tokens"] and sizes == meta["train_shard_tokens"]
    assert all(100e6 <= size < 101e6 for size in sizes[:-1]) and 0 < sizes[-1] < 101e6
    rng = np.random.default_rng(0)
    for shard, size in zip(meta["train_shards"], sizes):
        tokens = np.memmap(folder / shard, dtype="<u2", mode="r")
        assert tokens[0] == tok.bos_id  # a shard begins with a document
        start = int(rng.integers(0, size - 200_000))
        window = np.asarray(tokens[start : start + 200_000])
        assert window.max() <= tok.bos_id  # no id outside the vocabulary, no chat token
        assert 1 <= (window == tok.bos_id).sum()  # documents are much shorter than the window
    if dataset == "fineweb-edu":
        assert meta["n_train_tokens"] >= 6e9 and 9.9e6 <= meta["n_val_tokens"] <= 10.2e6

    # Validation: tokens, counts and bytes agree with each other and with the text file.
    val = np.fromfile(folder / "val.bin", dtype="<u2")
    assert len(val) == meta["n_val_tokens"] and val[0] == tok.bos_id and val.max() == tok.bos_id
    starts = np.flatnonzero(val == tok.bos_id)
    assert len(starts) == meta["val_docs"]
    texts = held_out_texts(dataset, 10**9)
    assert len(texts) == meta["val_docs"] and sum(len(text.encode("utf-8")) for text in texts) == meta["val_bytes"]
    assert meta["bytes_per_token_val"] == pytest.approx(meta["val_bytes"] / meta["n_val_tokens"])
    ends = [*starts[1:], len(val)]
    for i in [*range(100), *rng.integers(0, len(texts), 400).tolist(), len(texts) - 1]:
        assert tok.decode(val[starts[i] + 1 : ends[i]].tolist()) == texts[i]

    # Held-out documents are not in the training data: look for each of them in the first shard.
    held_out = {val[a:b].tobytes() for a, b in zip(starts, ends)}
    tokens = np.fromfile(folder / meta["train_shards"][0], dtype="<u2")
    cuts = np.flatnonzero(tokens == tok.bos_id)
    assert not any(tokens[a:b].tobytes() in held_out for a, b in zip(cuts, [*cuts[1:], len(tokens)]))
