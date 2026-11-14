"""Tests of the evaluation.

First the arithmetic, on models whose score is known in closed form (a uniform model costs log2(vocab) bits a
token; a model that ignores context costs the same however the text is cut into windows). Then the item formats
of the benchmarks. Then the two real paths: a checkpoint of ours, and GPT-2 small from the Hugging Face Hub on
WikiText-103, whose score is set beside the published one.

    python -m pytest -q tests/test_evaluate.py          # uses the GPU if CUDA_VISIBLE_DEVICES offers one
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from gptpp import benchmarks, evaluate
from gptpp.evaluate import LanguageModel, bits_per_byte, score_sequences

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEXTS = ["Hello, world. " * 9, "Zwölf Boxkämpfer jagen Viktor quer über den großen Sylter Deich.", "x",
         "The cat sat on the mat, and the mat said nothing."]


class Bytes(LanguageModel):
    """Bytes as tokens, and a next-token distribution that ignores the context (uniform unless `table` is given)."""

    name, bos_id, vocab_size, device = "bytes", 256, 257, torch.device("cpu")

    def __init__(self, table: torch.Tensor | None = None, context: int = 64):
        self.table = torch.zeros(self.vocab_size) if table is None else table
        self.context = context
        self.info = {"name": self.name}

    def encode(self, texts):
        return [list(t.encode("utf-8")) for t in texts]

    def decode(self, ids):
        return bytes(ids).decode("utf-8", errors="replace")

    def logits(self, idx):
        return self.table.expand(*idx.shape, -1)


class Pairs(Bytes):
    """Two bytes a token (a last odd byte is its own token), every next token equally likely."""

    vocab_size, bos_id = 65536 + 256 + 1, 65536 + 256

    def encode(self, texts):
        out = []
        for raw in (t.encode("utf-8") for t in texts):
            ids = [raw[i] * 256 + raw[i + 1] for i in range(0, len(raw) - 1, 2)]
            out.append(ids + ([65536 + raw[-1]] if len(raw) % 2 else []))
        return out

    def decode(self, ids):
        return b"".join(bytes([i - 65536]) if i >= 65536 else bytes(divmod(i, 256)) for i in ids).decode("utf-8")


class Running(Bytes):
    """A causal toy with context: the logits at a place are a fixed projection of the mean of the embeddings so
    far. Like a transformer, nothing after a place can change its prediction."""

    def __init__(self, context: int = 64):
        super().__init__(context=context)
        g = torch.Generator().manual_seed(0)
        self.embed = torch.randn(self.vocab_size, 16, generator=g)
        self.unembed = torch.randn(16, self.vocab_size, generator=g)

    def logits(self, idx):
        e = self.embed[idx]
        return (e.cumsum(dim=1) / torch.arange(1, idx.shape[1] + 1).view(1, -1, 1)) @ self.unembed


# -- bits per byte ------------------------------------------------------------------------------------------------


def test_uniform_model_costs_log2_vocab_bits_a_token():
    """Bits per byte of a uniform model is log2(vocab) / (bytes per token): 8.0056 for bytes plus one special."""
    r = bits_per_byte(Bytes(), TEXTS)
    assert r["bytes_per_token"] == 1.0 and r["n_docs_not_reproduced"] == 0
    assert r["bits_per_byte"] == pytest.approx(math.log2(257), rel=1e-6)

    r = bits_per_byte(Pairs(), TEXTS)
    assert 1.9 < r["bytes_per_token"] <= 2.0
    assert r["bits_per_byte"] == pytest.approx(math.log2(Pairs.vocab_size) / r["bytes_per_token"], rel=1e-6)
    assert r["bits_per_byte"] == pytest.approx(r["loss"] / math.log(2) * r["tokens_per_byte"], rel=1e-12)


def test_every_token_is_scored_once_whatever_the_window():
    """For a model that ignores context the total is the sum of the tokens' own costs: no token is missed or
    counted twice at a window boundary, and the beginning-of-text token is never charged."""
    table = torch.log_softmax(torch.randn(257, generator=torch.Generator().manual_seed(1)), dim=0)
    direct = -sum(table[b].item() for t in TEXTS for b in t.encode("utf-8")) / math.log(2)
    for context in (2, 5, 16, 64):
        r = bits_per_byte(Bytes(table, context=context), TEXTS)
        assert r["total_bits"] == pytest.approx(direct, rel=1e-6)
        assert r["n_tokens"] == sum(len(t.encode("utf-8")) for t in TEXTS)
        assert r["n_windows"] == sum(math.ceil(len(t.encode("utf-8")) / (context - 1)) for t in TEXTS)
        assert sum(r["doc_bits"]) == pytest.approx(direct, abs=0.2)  # rounded to a tenth of a bit a document
        assert sum(p["n_tokens"] for p in r["by_position"]) == r["n_tokens"]


def test_scores_equal_a_direct_computation():
    model = Running()
    rng = np.random.default_rng(0)
    seqs = [[model.bos_id] + rng.integers(0, 256, n).tolist() for n in (3, 12, 40, 63)]
    starts = [1, 5, 33, 60]
    scores = score_sequences(model, seqs, starts, want_greedy=True, batch_tokens=64)  # several batches
    for seq, start, nll, n, greedy in zip(seqs, starts, scores.nll, scores.n_scored, scores.greedy):
        logp = torch.log_softmax(model.logits(torch.tensor([seq]))[0].double(), dim=-1)
        places = range(start, len(seq))
        assert nll == pytest.approx(-sum(logp[t - 1, seq[t]].item() for t in places), rel=1e-5)
        assert n == len(seq) - start
        assert greedy == all(int(logp[t - 1].argmax()) == seq[t] for t in places)


def test_padding_does_not_change_a_score():
    assert evaluate.padding_error(Running()) < 1e-5


def test_standard_error_of_a_ratio():
    bits, nbytes = np.array([10.0, 20.0, 40.0]), np.array([10.0, 20.0, 40.0])
    assert evaluate.ratio_standard_error(bits, nbytes) == 0.0  # every document at the same rate
    assert evaluate.ratio_standard_error(bits[:1], nbytes[:1]) is None
    bits = np.array([12.0, 18.0, 44.0])  # residuals from the ratio 74/70: by hand
    r = bits.sum() / nbytes.sum()
    by_hand = math.sqrt(3 / 2 * sum((b - r * n) ** 2 for b, n in zip(bits, nbytes))) / nbytes.sum()
    assert evaluate.ratio_standard_error(bits, nbytes) == pytest.approx(by_hand, rel=1e-12)


def test_wikitext_articles_begin_at_titles_only():
    rows = ["", " = First = \n", "", " Text one . \n", "", " = = Section = = \n", "", " More text , where k \n",
            " = 1 henry and capacitance C = \n", " 1 farad . \n", "", " = Second = \n", "", " Text two . \n"]
    articles = evaluate.wikitext_articles(rows)
    assert len(articles) == 2 and articles[0].startswith(" = First = \n") and articles[1].startswith(" = Second = \n")
    assert "".join(articles) == "".join(rows)


def test_documents_are_fingerprinted(tmp_path):
    path = tmp_path / "docs.jsonl"
    path.write_text("".join(json.dumps({"text": t}) + "\n" for t in ["one", "", "two", "three"]), encoding="utf-8")
    texts, source = evaluate.load_docs(str(path))
    assert texts == ["one", "two", "three"] and source["n_docs"] == 3
    first_two, source_two = evaluate.load_docs(str(path), max_docs=2)
    assert first_two == ["one", "two"] and source_two["sha256"] != source["sha256"]
    assert evaluate.load_docs(str(path))[1]["sha256"] == source["sha256"]


def test_results_are_merged_into_one_file(tmp_path):
    out = tmp_path / "m.json"
    info = {"name": "bytes", "precision": "float32"}
    evaluate.merge_result(out, info, "bits_per_byte", "a", {"bits_per_byte": 8.0, "doc_bits": [1.0, 2.5]})
    record = evaluate.merge_result(out, info, "benchmarks", "t", {"accuracy": 0.5, "by_position": [{"from": 1}]})
    assert json.loads(out.read_text()) == record
    assert record["bits_per_byte"]["a"]["bits_per_byte"] == 8.0 and record["benchmarks"]["t"]["accuracy"] == 0.5
    with pytest.raises(ValueError):
        evaluate.merge_result(out, {"name": "another model"}, "benchmarks", "t", {})


# -- benchmarks ---------------------------------------------------------------------------------------------------


def toy_task(monkeypatch, rows):
    """Register a task whose rows are {"given", "options", "gold"}, without the Hub."""
    monkeypatch.setattr(benchmarks, "hub_parquet", lambda repo, revision, filename: rows)
    build = lambda rows: [benchmarks.Item([(r["given"], o) for o in r["options"]], r["gold"], r.get("group", ""))
                          for r in rows]
    monkeypatch.setitem(benchmarks.TASKS, "toy", benchmarks.Task("toy", "a toy", "none", "none", ["f"], build, "toy"))


def test_a_model_that_cannot_tell_scores_chance(monkeypatch):
    """Tied options share the credit: a uniform model gets exactly the chance level, not the share of items whose
    right answer happens to be listed first."""
    rows = [{"given": "Q:", "options": [" aa", " bb", " cc", " dd"], "gold": g} for g in (0, 1, 3, 3, 3)]
    rows += [{"given": "Q:", "options": [" yes", " no!"], "gold": 1}]
    toy_task(monkeypatch, rows)
    r = benchmarks.run_task(Bytes(), "toy")
    assert r["chance"] == pytest.approx((5 * 0.25 + 0.5) / 6)
    assert r["acc_per_token"] == pytest.approx(r["chance"]) and r["acc_norm"] == pytest.approx(r["chance"])
    assert r["accuracy"] == r["acc_norm"] and r["n"] == 6 and r["correct"] == "000000"


def test_the_most_probable_option_wins(monkeypatch):
    """A model that likes the letter e: it picks the option with e's, per byte; in total it prefers short options."""
    table = torch.full((257,), -5.0)
    table[ord("e")] = -0.1
    rows = [{"given": "", "options": ["eee", "xyz"], "gold": 0, "group": "a/x"},
            {"given": "Which?", "options": ["xyz", "eeee"], "gold": 1, "group": "a/y"},
            {"given": "Which?", "options": ["eeexxxxxx", "xx"], "gold": 0, "group": "b/z"}]
    toy_task(monkeypatch, rows)
    r = benchmarks.run_task(Bytes(table), "toy")
    assert r["acc_norm"] == 1.0 and r["acc"] == pytest.approx(2 / 3)
    assert r["by_family"]["a"]["accuracy"] == 1.0 and set(r["by_group"]) == {"a/x", "a/y", "b/z"}
    assert r["n_sequences"] == 6 and r["correct"] == "111"


def test_a_long_context_is_cut_from_the_left(monkeypatch):
    toy_task(monkeypatch, [{"given": "c" * 100, "options": ["ee", "xy"], "gold": 0}])
    table = torch.full((257,), -5.0)
    table[ord("e")] = -0.1
    r = benchmarks.run_task(Bytes(table, context=16), "toy")
    assert r["n_cut_to_window"] == 2 and r["accuracy"] == 1.0 and r["n_tokens"] == 2 * 16


def test_the_fixed_subset_never_changes():
    """The hosted models are asked these items; a change here would silently change what is compared."""
    assert benchmarks.fixed_subset("hellaswag", 10042, 5) == [5092, 7875, 8098, 8850, 9905]
    subset = benchmarks.fixed_subset("blimp", 67000, 300)
    assert len(set(subset)) == 300 and subset == sorted(subset)
    assert set(benchmarks.fixed_subset("blimp", 67000, 100)) <= set(subset)


def test_item_formats():
    row = {"activity_label": "Baking", "ctx_a": "[header] How to bake [title] Mix.", "ctx_b": "then you",
           "endings": ["stir [step] well.", "sleep."], "label": "0", "source_id": "wikihow~1"}
    item = benchmarks.TASKS["hellaswag"].build([row])[0]
    assert item.pairs[0] == ("Baking: How to bake. Mix. Then you", " stir well.") and item.gold == 0
    assert item.group == "wikihow"

    row = {"sentence": "Ann thanked Sue because _ had helped.", "option1": "Ann", "option2": "Sue", "answer": "2"}
    item = benchmarks.TASKS["winogrande"].build([row])[0]
    assert item.pairs == [("Ann thanked Sue because Ann", " had helped."), ("Ann thanked Sue because Sue", " had helped.")]
    assert item.gold == 1

    item = benchmarks.TASKS["lambada"].build([{"text": "He opened the door and saw the garden"}])[0]
    assert item.pairs == [("He opened the door and saw the", " garden")]

    row = {"question": "Why?", "choices": {"text": ["a", "b", "c"], "label": ["1", "2", "3"]}, "answerKey": "3"}
    item = benchmarks.TASKS["arc_easy"].build([row])[0]
    assert item.pairs[2] == ("Question: Why?\nAnswer:", " c") and item.gold == 2

    row = {"sentence_good": "The cats sleep.", "sentence_bad": "The cats sleeps.", "linguistics_term": "agreement",
           "UID": "plural"}
    item = benchmarks.TASKS["blimp"].build([row])[0]
    assert item.pairs == [("", "The cats sleep."), ("", "The cats sleeps.")] and item.group == "agreement/plural"


# -- our checkpoints ----------------------------------------------------------------------------------------------


def our_checkpoint(tmp_path, uniform: bool):
    """A run directory as the trainer writes it, holding an untrained pico model."""
    config = pytest.importorskip("gptpp.config")
    model_module = pytest.importorskip("gptpp.model")
    tokenizer = pytest.importorskip("gptpp.tokenizer")
    try:
        tokenizer.Tokenizer.load("bpe4k")
    except (OSError, KeyError):
        pytest.skip("the bpe4k tokenizer is not on this machine")
    torch.manual_seed(0)
    cfg = config.preset("gptpp-pico")
    model = model_module.GPT(cfg)
    with torch.no_grad():
        for p in model.parameters():
            if p.abs().sum() == 0:
                p.normal_(std=0.05)  # projections that start at zero: make the untrained model depend on its input
        if uniform:
            for p in model.unembed.parameters():
                p.zero_()
    run = tmp_path / "pico-test"
    run.mkdir()
    torch.save({"model": model.state_dict(), "model_config": cfg.to_dict(), "step": 0, "tokens": 0}, run / "model.pt")
    (run / "config.json").write_text(json.dumps({"data": {"tokenizer": "bpe4k", "dataset": "tinystories"}}))
    return run, model, cfg


def test_our_uniform_model_costs_log2_vocab_bits_a_token(tmp_path):
    """The spec's identity on our own path: checkpoint, tokenizer, windows of the model's context."""
    run, _, cfg = our_checkpoint(tmp_path, uniform=True)
    model = evaluate.load_model(str(run), device="cpu", precision="float32")
    assert model.info["tokenizer"] == "bpe4k" and model.window == cfg.block_size == 256
    r = bits_per_byte(model, TEXTS + ["Once upon a time there was a little girl named Lily. " * 40])
    assert r["bits_per_byte"] == pytest.approx(math.log2(cfg.vocab_size) / r["bytes_per_token"], rel=1e-6)
    assert r["n_docs_not_reproduced"] == 0 and r["n_windows"] > len(TEXTS) + 1


def test_our_model_scores_equal_its_own_loss(tmp_path):
    run, reference, cfg = our_checkpoint(tmp_path, uniform=False)
    model = evaluate.load_model(str(run), device="cpu", precision="float32")
    ids = [model.bos_id] + model.encode(["Once upon a time there was a little girl named Lily."])[0]
    x = torch.tensor([ids])
    with torch.no_grad():
        loss = reference.eval()(x[:, :-1], x[:, 1:]).item()  # the trainer's loss: mean nats a token
    scores = score_sequences(model, [ids, ids[:5]])
    assert scores.nll[0] / scores.n_scored[0] == pytest.approx(loss, rel=1e-5)
    assert evaluate.padding_error(model) < 1e-4


# -- GPT-2 small from the Hub -------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gpt2():
    try:
        return evaluate.load_model("hf:gpt2", device=DEVICE)
    except Exception as error:  # no network and nothing in the cache
        pytest.skip(f"GPT-2 small cannot be loaded: {type(error).__name__}")


def test_gpt2_scores_equal_the_library_loss(gpt2):
    """One window scored by us and by transformers' own loss (mean over the tokens after the first)."""
    ids = [gpt2.bos_id] + gpt2.encode(["It was the best of times, it was the worst of times, it was the age of wisdom."])[0]
    x = torch.tensor([ids], device=gpt2.device)
    with torch.no_grad():
        library = gpt2.model(input_ids=x, labels=x).loss.item()
    scores = score_sequences(gpt2, [ids, ids[:9]])
    assert scores.nll[0] / scores.n_scored[0] == pytest.approx(library, rel=1e-3)
    assert gpt2.info["n_params"] == 124_439_808 and gpt2.window == 1024 and gpt2.bos_id == 50256


def test_gpt2_small_on_wikitext103_is_plausible(gpt2):
    """GPT-2 small on the WikiText-103 test set, beside the published number.

    Published (Radford et al. 2019, table 3; also the model card): perplexity 37.50 a word, zero-shot, on text
    put through de-tokenizers that the paper says are worth 2.5 to 5 perplexity. We score the raw text as
    distributed, in windows that do not overlap, so we must land somewhat above: a word perplexity between 37.5
    and 55 (we measure 49.2), which is 1.07 bits per byte. The unit of the published number is the benchmark's
    token count, 245,569 = 241,211 words + 4,358 line ends (Merity et al. 2016, table 1).
    """
    try:
        docs, source = evaluate.load_docs("wikitext103", None if DEVICE == "cuda" else 6)
    except Exception as error:
        pytest.skip(f"WikiText-103 cannot be loaded: {type(error).__name__}")
    r = bits_per_byte(gpt2, docs)
    assert r["n_docs_not_reproduced"] == 0 and 4.3 < r["bytes_per_token"] < 4.8
    assert r["bits_per_byte"] == pytest.approx(r["loss"] / math.log(2) / r["bytes_per_token"], rel=1e-12)
    assert 0.95 < r["bits_per_byte"] < 1.20
    if len(docs) == 60:  # the whole test set
        text = "".join(docs)
        units = len(text.split()) + 4358
        assert source["sha256"] == "8be9b692dca47dc9" and units == 245_569
        word_perplexity = math.exp(r["total_bits"] * math.log(2) / units)
        assert 37.5 < word_perplexity < 55
        assert r["bits_per_byte"] == pytest.approx(1.072, abs=0.004)


# -- the scripts: hosted models, the grader, the fresh corpus (no network) ----------------------------------------


def script(name: str):
    """Import scripts/<name>.py as a module."""
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_reading_a_hosted_model_answer():
    """A reply counts only when it comes down to one letter; a long explanation must end with its letter."""
    read = script("eval_openrouter").read_letter
    explanation = '"Axes" is plural (of axe or axis), so it requires the plural demonstrative "these."'
    for reply, n_options, answer in [
        ("B", 4, "B"), ("B.", 4, "B"), ("(C)", 4, "C"), ("**D**", 4, "D"), ("Answer: A", 4, "A"),
        ("The answer is C.", 4, "C"), ("First I thought the answer is A, but the answer is C.", 4, "C"),
        ("B. pulls the barbell forward.", 4, "B"), ("B. A man walks away.", 4, "B"), ("I would say D", 4, "D"),
        ("Sentence B is acceptable.", 2, "B"), ("E", 5, "E"), (explanation + "\n\n**B**", 2, "B"),
        ("Sentence A violates the island constraint, while B does not.\n\nB", 2, "B"),
        ("E", 4, None), ("", 4, None), (None, 4, None), ("Neither.", 2, None), (" b", 4, None), ("A, B", 2, None),
        ("A and B", 2, None), ("A plausible continuation is B", 4, None), (explanation, 2, None),
        ("A spring scale is needed to measure the pull required to move the block, and a meter stick the distance.", 4, None),
    ]:
        assert read(reply, n_options) == answer, reply


def test_hosted_models_are_asked_with_the_least_thinking():
    settings = script("eval_openrouter").request_settings
    optional = {"supported_parameters": ["temperature", "reasoning"],
                "reasoning": {"mandatory": False, "supported_efforts": ["max", "high", "low", "none"]}}
    assert settings(optional, 64) == {"max_tokens": 64, "usage": {"include": True}, "temperature": 0,
                                      "reasoning": {"effort": "none"}}
    mandatory = {"supported_parameters": ["reasoning"],
                 "reasoning": {"mandatory": True, "supported_efforts": ["xhigh", "low", "medium"]}}
    assert settings(mandatory, 64)["reasoning"] == {"effort": "low"} and "temperature" not in settings(mandatory, 64)
    assert "reasoning" not in settings({"supported_parameters": ["max_tokens"]}, 64)


def test_hosted_questions_come_from_the_fixed_subset(monkeypatch):
    hosted = script("eval_openrouter")
    pairs = [benchmarks.Item([("", f"Good sentence {i}."), ("", f"Bad sentence {i}.")], 0) for i in range(40)]
    monkeypatch.setattr(hosted, "_items", lambda task: pairs)
    asked = hosted.questions("blimp", 12)
    assert [q["index"] for q in asked] == benchmarks.fixed_subset("blimp", 40, 12)
    assert {q["gold"] for q in asked} == {"A", "B"}  # the order shown is decided by a hash of the item's number
    for q in asked:
        right = f"{q['gold']}. Good sentence {q['index']}."
        assert right in q["prompt"] and q["prompt"].endswith("Answer with the letter only.")


def test_reading_a_grader_reply():
    read = script("judge_stories").read_grades
    assert read("Fine work.\nGrammar: 8/10, Creativity: 7/10, Consistency: 7/10, Plot: 6/10, Age group: E (10-12)") == {
        "grammar": 8.0, "creativity": 7.0, "consistency": 7.0, "plot": 6.0, "age_group": "E"}
    assert read("Grammar: 8/10\nCreativity: 7/10\nConsistency: 7.5/10\nPlot: 6/10\nAge group: (C)")["consistency"] == 7.5
    assert read("Grammar: 11/10, Creativity: 7/10, Consistency: 7/10, Plot: 6/10, Age group: E") is None
    assert read("The story is good.") is None and read("") is None


def test_story_grades_are_averaged_over_prompts():
    summarize = script("judge_stories").summarize
    row = lambda prompt, grammar: {"model": "m", "prompt_id": prompt, "grammar": grammar, "creativity": 5.0,
                                   "consistency": 5.0, "plot": 5.0, "age_group": "C"}
    rows = [row(0, 10.0), row(0, 8.0), row(1, 4.0), {**row(2, None), "age_group": None}]
    entry = summarize(rows)["m"]
    assert entry["n_completions"] == 4 and entry["n_graded"] == 3 and entry["n_prompts"] == 2
    assert entry["grammar"]["mean"] == pytest.approx(6.5)  # the mean of the two prompts' means, 9 and 4
    assert entry["grammar"]["stderr"] == pytest.approx(2.5) and entry["creativity"]["stderr"] == 0.0


def test_fresh_corpus_keeps_plain_prose_only():
    fresh = script("make_fresh_corpus")
    prose = ("The first paragraph of the essay is long enough to be kept: it says something plain about a subject, "
             "then says a little more, and it ends with a full stop after some two hundred characters of running text.")
    noted = prose.replace("subject,", 'subject,<span class="sidenote-number"></span> <span class="sidenote">A note '
                          "with <em>emphasis</em> and a [link](https://example.org).</span>")
    marked = prose.replace("plain", "*plain*").replace("a little more", "[a little more](/elsewhere)")
    post = "\n\n".join([
        '---\ntitle: "An essay"\npermalink: /an-essay\n---',
        "<style>\np { color: red; }\n</style>",
        "## A heading",
        noted,
        "> A block quotation from an older book, which a model may have read, so it is left out of the corpus.",
        marked,
        '<figure class="fullwidth">\n<div id="fig"></div>\n'
        "<figcaption>A caption that is long but is not prose.</figcaption>\n</figure>",
        "```python\nprint('code')\n```",
        "$$\nx = y + z\n$$",
        prose.replace("subject", "variable $$x$$"),
        "- a list item that is long enough to count as a paragraph by its length alone, but is still a list item.",
        "{% include frag/thing.html %}",
        "Short line.",
        prose.replace("full stop", "colon").rstrip(".") + ":",
    ])
    dropped: dict = {}
    assert fresh.prose_paragraphs(post, dropped) == [prose, prose]
    assert sum(dropped.values()) >= 5
    meta, body = fresh.split_front_matter(post)
    assert meta == {"title": "An essay", "permalink": "/an-essay"} and body.lstrip().startswith("<style>")


def test_the_published_wikitext103_perplexity_lies_between_raw_and_detokenized_text(gpt2):
    """The published 37.50 was measured on de-tokenized text, ours on the raw text. Under our own protocol GPT-2
    small scores 49.2 a word on the raw test set and 36.9 once the text is de-tokenized by our own rules
    (scripts/eval_released_models.py): the published number lies between the two, 1.5% from the second."""
    if DEVICE != "cuda":
        pytest.skip("two passes over the whole test set: on a GPU only")
    try:
        docs, _ = evaluate.load_docs("wikitext103")
    except Exception as error:
        pytest.skip(f"WikiText-103 cannot be loaded: {type(error).__name__}")
    detokenize = script("eval_released_models").detokenize_wikitext
    assert detokenize(" He scored 1 @,@ 200 points ( a record ) in the mid @-@ 1990s , and did n't stop . \n") == \
        " He scored 1,200 points (a record) in the mid-1990s, and didn't stop. \n"
    word_perplexity = lambda texts: math.exp(bits_per_byte(gpt2, texts)["total_bits"] * math.log(2) / 245_569)
    raw, detokenized = word_perplexity(docs), word_perplexity([detokenize(d) for d in docs])
    assert detokenized < 37.50 < raw
    assert 35 < detokenized and raw < 55  # we measure 36.9 and 49.2
