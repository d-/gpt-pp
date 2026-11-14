"""The chat stage on a tiny model: the format and its mask, the rows, the masked loss, the loop, and generation
that stops at the end token. Everything runs on the CPU in a few seconds.

    cd gpt-pp && $PY -m pytest -q tests/test_chat.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gptpp import chat
from gptpp.chat import ASSISTANT_END, ASSISTANT_START, BOS, USER_END, USER_START
from gptpp.config import ModelConfig
from gptpp.model import GPT
from gptpp.tokenizer import Tokenizer


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    """The 256 bytes and the five special tokens: a vocabulary of 261 with no merges."""
    return Tokenizer([], name="bytes-chat")


CONVERSATION = [
    {"role": "user", "content": "What is 2 + 2?"},
    {"role": "assistant", "content": "It is 4."},
    {"role": "user", "content": "Et en français, s'il vous plaît ?"},
    {"role": "assistant", "content": "Cela fait quatre. ✓"},
]


# ----------------------------------------------------------------------------------------------------------------
# The format


def test_the_mask_is_the_assistants_text_and_its_end_token(tok: Tokenizer) -> None:
    ids, mask = chat.render(tok, CONVERSATION)
    s = tok.special
    assert len(ids) == len(mask) and set(mask) == {0, 1}
    assert ids[0] == s[BOS] and ids.count(s[BOS]) == 1
    # walk the sequence: inside an assistant's turn, after its start marker, every token carries the loss,
    # the end marker included; everything else carries none
    inside, expected = False, []
    for i in ids:
        if i == s[ASSISTANT_START]:
            inside = True
            expected.append(0)
        else:
            expected.append(1 if inside else 0)
            if i == s[ASSISTANT_END]:
                inside = False
    assert mask == expected
    answers = [m["content"] for m in CONVERSATION if m["role"] == "assistant"]
    assert sum(mask) == sum(len(tok.encode(a)) + 1 for a in answers)
    assert [mask[k] for k, i in enumerate(ids) if i == s[ASSISTANT_END]] == [1, 1]
    for marker in (BOS, USER_START, USER_END, ASSISTANT_START):
        assert all(mask[k] == 0 for k, i in enumerate(ids) if i == s[marker]), marker
    # the order of the markers is the format
    markers = [name for i in ids for name, j in s.items() if i == j]
    assert markers == [BOS] + [USER_START, USER_END, ASSISTANT_START, ASSISTANT_END] * 2


def test_the_format_round_trips(tok: Tokenizer) -> None:
    ids, _ = chat.render(tok, CONVERSATION)
    assert chat.parse(tok, ids) == CONVERSATION
    # a prompt is the conversation up to the user's message, then the start of the assistant's turn
    prompt = chat.prompt_ids(tok, CONVERSATION[:3])
    assert prompt == ids[:len(prompt)] and prompt[-1] == tok.special[ASSISTANT_START]
    assert chat.parse(tok, prompt) == CONVERSATION[:3]
    # an answer cut short is returned as far as it goes
    # less the end marker and the four bytes of " ✓"
    assert chat.parse(tok, ids[:-5])[-1] == {"role": "assistant", "content": "Cela fait quatre."}
    # special tokens written as text stay text
    odd = [{"role": "user", "content": "say <|assistant_end|>"}, {"role": "assistant", "content": "<|assistant_end|>"}]
    odd_ids, odd_mask = chat.render(tok, odd)
    assert odd_ids.count(tok.special[ASSISTANT_END]) == 1 and chat.parse(tok, odd_ids) == odd
    # a system message goes in front of the first user message
    with_system = [{"role": "system", "content": "Be brief."}, *CONVERSATION[:2]]
    assert chat.parse(tok, chat.render(tok, with_system)[0])[0]["content"] == "Be brief.\n\nWhat is 2 + 2?"


def test_malformed_conversations_are_refused(tok: Tokenizer) -> None:
    with pytest.raises(ValueError):
        chat.render(tok, [{"role": "assistant", "content": "hello"}])
    with pytest.raises(ValueError):
        chat.render(tok, [CONVERSATION[0], CONVERSATION[2]])
    with pytest.raises(ValueError):
        chat.prompt_ids(tok, CONVERSATION[:2])  # ends with the assistant: nothing to answer
    s = tok.special
    with pytest.raises(ValueError):
        chat.parse(tok, [s[USER_START], 65, s[USER_END]])  # no <|bos|>
    with pytest.raises(ValueError):
        chat.parse(tok, [s[BOS], s[USER_START], 65, s[ASSISTANT_END]])  # a turn closed by the wrong marker


# ----------------------------------------------------------------------------------------------------------------
# Rows


def test_pack_places_every_conversation_whole_and_once() -> None:
    rng = np.random.default_rng(0)
    lengths = rng.integers(5, 60, size=500)
    order = rng.permutation(500)
    rows = chat.pack(lengths, 64, order)
    assert sorted(c for row in rows for c in row) == list(range(500))
    assert all(sum(lengths[c] for c in row) <= 64 for row in rows)
    filled = lengths.sum() / (64 * len(rows))
    assert filled > 0.93  # best fit leaves little padding
    assert len(rows) < len(chat.pack(lengths, 64, order, buffer=1))  # than taking them as they come
    assert chat.pack(lengths, 64, order) == rows  # deterministic
    # a subset in a given order; a conversation longer than a row gets a row to itself
    assert sorted(c for row in chat.pack(lengths, 64, [3, 1, 4]) for c in row) == [1, 3, 4]
    assert chat.pack([100, 10], 64) == [[0], [1]]


def write_chat_data(tok: Tokenizer, root: Path, train: list[list[dict]], val: list[list[dict]]) -> Path:
    """A chat dataset in the on-disk format, under <root>/data/toy/tok, with the tokenizer saved beside it."""
    out = root / "data" / "toy" / "tok"
    out.mkdir(parents=True, exist_ok=True)
    tok_dir = tok.save(root / "data" / "tokenizers" / "bytes-chat")
    meta = {"dataset": "toy", "tokenizer": str(tok_dir), "vocab_size": tok.vocab_size, "bos_id": tok.bos_id,
            "train_shards": ["train_0000.bin"], "val_file": "val.bin"}
    for name, split, conversations in (("train_0000", "train", train), ("val", "val", val)):
        rendered = [chat.render(tok, c) for c in conversations]
        np.array([i for ids, _ in rendered for i in ids], dtype="<u2").tofile(out / f"{name}.bin")
        np.array([m for _, mask in rendered for m in mask], dtype=np.uint8).tofile(out / f"{name}.mask.bin")
        counts = {"conversations": len(conversations), "exchanges": sum(len(c) // 2 for c in conversations),
                  "tokens": sum(len(ids) for ids, _ in rendered), "loss_tokens": sum(sum(mask) for _, mask in rendered)}
        meta[f"n_{split}_tokens"] = counts["tokens"]
        meta[split] = {**counts, "by_source": {"toy": counts}, "by_category": {}}
    # the fields the real preparation writes and the measurements copy (scripts/prepare_chat.py)
    meta.update(source="made for the test", license="Apache 2.0 (a test fixture)", format="as gptpp.chat.render",
                val_docs_file="../val_conversations.jsonl",
                selection={"rule": "all", "max_answer_tokens": 2, "context": 32, "dropped_categories": [],
                           "dropped_qualities": [], "repeats": {"max_repeats": 0}})
    (out / "meta.json").write_text(json.dumps(meta))
    return out


LETTERS = "abcdefgh"


def toy_conversation(rng: np.random.Generator) -> list[dict]:
    """One or two exchanges of a task a tiny model learns at once: the user writes a letter and a question mark,
    the assistant the same letter in upper case and a full stop."""
    out = []
    for letter in rng.choice(list(LETTERS), size=rng.integers(1, 3)):
        out += [{"role": "user", "content": f"{letter}?"}, {"role": "assistant", "content": f"{letter.upper()}."}]
    return out


@pytest.fixture(scope="module")
def toy(tok: Tokenizer, tmp_path_factory: pytest.TempPathFactory) -> Path:
    rng = np.random.default_rng(0)
    root = tmp_path_factory.mktemp("chat")
    return write_chat_data(tok, root, [toy_conversation(rng) for _ in range(3000)],
                           [toy_conversation(rng) for _ in range(100)])


def test_rows_hold_whole_conversations_and_targets_are_masked(tok: Tokenizer, toy: Path) -> None:
    data = chat.ChatData(toy)
    tokens, mask = data.shard(0), data.shard_mask(0)
    starts, lengths = chat.conversations(tokens, data.bos_id)
    assert len(starts) == 3000 and len(tokens) == len(mask) == lengths.sum()

    T, B = 32, 8
    loader = chat.ChatLoader(data, T, B, seed=0)
    assert loader.rows_per_pass() == len(loader.ids) // B * B
    s = tok.special
    seen = 0
    for _ in range(loader.rows_per_pass() // B):
        x, y = loader.next_batch()
        assert x.shape == y.shape == (B, T) and (x[:, 0] == s[BOS]).all()
        counted = y >= 0
        assert (y[counted] != s[BOS]).all() and (y[~counted] == -1).all()
        for row_x, row_y in zip(x.tolist(), y.tolist()):
            row = row_x + [row_y[-1] if row_y[-1] >= 0 else s[BOS]]
            # every conversation of the row parses: none is cut; padding is <|bos|> and parses to nothing
            cuts = [k for k, i in enumerate(row) if i == s[BOS]] + [len(row)]
            inside = False
            for a, b in zip(cuts, cuts[1:]):
                messages = chat.parse(tok, row[a:b])
                assert len(messages) % 2 == 0
                seen += len(messages) > 0
            # a target counts exactly when the next token is the assistant's text or its end marker
            for k in range(T):
                nxt = row[k + 1]
                if row[k] == s[ASSISTANT_START]:
                    inside = True
                assert (row_y[k] == nxt) == inside and (row_y[k] == -1) == (not inside), (k, row)
                if nxt == s[ASSISTANT_END]:
                    inside = False
    assert seen >= 3000 - 2 * B * 4  # all conversations but those of the dropped last batch

    # the held-out rows do not depend on the batch size
    a = chat.val_batches(data, T, 4)
    b = chat.val_batches(data, T, 16)
    assert torch.equal(torch.cat([x for x, _ in a]), torch.cat([x for x, _ in b]))
    assert sum(int((y >= 0).sum()) for _, y in a) == int(data.val_mask().sum())


def tiny_model(tok: Tokenizer, seed: int = 0) -> GPT:
    torch.manual_seed(seed)
    return GPT(ModelConfig().replace(vocab_size=tok.vocab_size, n_layer=2, n_head=2, d_model=64, block_size=32))


def test_evaluate_is_the_models_masked_loss(tok: Tokenizer, toy: Path) -> None:
    model = tiny_model(tok)
    with torch.no_grad():
        model.unembed.weight.normal_(std=0.5)  # so that the loss is not exactly log V
    batches = chat.val_batches(chat.ChatData(toy), 32, 8)
    result = chat.evaluate(model, batches, tok.special[ASSISTANT_END])
    with torch.no_grad():
        sums = sum(model(x, y).item() * int((y >= 0).sum()) for x, y in batches)
    n = sum(int((y >= 0).sum()) for _, y in batches)
    assert result["tokens"] == n and result["loss"] == pytest.approx(sums / n, rel=1e-5)
    n_end = result["end_tokens"]
    assert n_end == sum(int((y == tok.special[ASSISTANT_END]).sum()) for _, y in batches) > 0
    assert result["loss"] == pytest.approx((result["loss_text"] * (n - n_end) + result["loss_end"] * n_end) / n)


# ----------------------------------------------------------------------------------------------------------------
# Generation


def scripted(model: GPT, script: list[int]):
    """Make `model` write `script` whatever it reads: its logits are replaced by ones that put all the mass on
    the next token of the script. The KV cache is advanced as the real pass would."""
    calls = {"n": 0, "fed": []}

    def forward_explicit(idx, hook=None, kv=None):
        kv.length += idx.size(1)
        calls["fed"].append(idx[0].tolist())
        logits = torch.full((1, idx.size(1), model.cfg.vocab_size), -30.0)
        logits[0, -1, script[min(calls["n"], len(script) - 1)]] = 30.0
        calls["n"] += 1
        return logits

    model.forward_explicit = forward_explicit
    return calls


def test_generation_stops_at_the_end_token(tok: Tokenizer) -> None:
    s = tok.special
    stop = {i: name for name, i in s.items()}
    prompt = chat.prompt_ids(tok, CONVERSATION[:1])
    hi = list(b"Hi!")

    # the model writes "Hi!" and the end token: three tokens come back, the end token is not among them, and
    # nothing is asked of the model after it
    model = tiny_model(tok)
    calls = scripted(model, hi + [s[ASSISTANT_END]] + list(b" more"))
    seen: list[int] = []
    written, stopped = chat.reply(model, prompt, stop, max_new_tokens=20, on_token=seen.append)
    assert written == hi == seen and stopped == ASSISTANT_END
    assert calls["fed"] == [prompt] + [[t] for t in hi]  # the prompt once, then one token at a time (KV cache)

    # through `answer`: the text and the reason
    model = tiny_model(tok)
    scripted(model, hi + [s[ASSISTANT_END]])
    assert chat.answer(model, tok, CONVERSATION[:1], max_new_tokens=20) == ("Hi!", {"tokens": 3, "stopped": "end"})

    # a model that never writes the end token is cut at the budget, and says so
    model = tiny_model(tok)
    scripted(model, list(b"a"))
    written, stopped = chat.reply(model, prompt, stop, max_new_tokens=5)
    assert written == list(b"aaaaa") and stopped == "length"

    # another marker also ends the turn, and is named
    model = tiny_model(tok)
    scripted(model, hi + [s[USER_START]])
    assert chat.reply(model, prompt, stop, max_new_tokens=20) == (hi, USER_START)

    # the context is the limit: a prompt of block_size - 2 tokens leaves room for two
    model = tiny_model(tok)
    scripted(model, list(b"a"))
    long_prompt = prompt + list(b"x") * (model.cfg.block_size - 2 - len(prompt))
    assert chat.reply(model, long_prompt, stop, max_new_tokens=20) == (list(b"aa"), "length")
    with pytest.raises(ValueError):
        chat.reply(model, long_prompt + list(b"xx"), stop)

    # sampling: the same seed gives the same answer, on the real model
    model = tiny_model(tok)
    with torch.no_grad():
        model.unembed.weight.normal_(std=0.5)
    first = chat.reply(model, prompt, stop, max_new_tokens=8, temperature=1.0, seed=3)
    assert first == chat.reply(model, prompt, stop, max_new_tokens=8, temperature=1.0, seed=3)
    assert first != chat.reply(model, prompt, stop, max_new_tokens=8, temperature=1.0, seed=4)


# ----------------------------------------------------------------------------------------------------------------
# The stage, end to end


@pytest.fixture(scope="module")
def toy_run(tok: Tokenizer, toy: Path, tmp_path_factory: pytest.TempPathFactory):
    """The stage run once on the toy task, on the CPU: (directory of the base, directory of the run, summary,
    the base's embedding before the stage)."""
    out = tmp_path_factory.mktemp("runs")
    threads = torch.get_num_threads()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch.cuda, "is_available", lambda: False)  # the CPU is enough, and deterministic
        torch.set_num_threads(1)
        try:
            base = out / "base"
            base.mkdir()
            model = tiny_model(tok, seed=1)
            torch.save({"model": model.state_dict(), "model_config": model.cfg.to_dict(), "step": 0, "tokens": 0},
                       base / "model.pt")
            summary = chat.main(["train", "--base", str(base), "--run", "toy-chat", "--data", str(toy),
                                 "--epochs", "1", "--batch", "16", "--lr-scale", "1", "--warmup-tokens", "2000",
                                 "--eval-tokens", "20000", "--eval-batch", "32", "--plain-data", "none",
                                 "--sample-prompt", "c?", "--no-compile", "--out", str(out),
                                 "--results", str(out / "results")])
        finally:
            torch.set_num_threads(threads)
    return base, out / "toy-chat", summary, model.embed.weight.detach().clone()


def test_the_stage_teaches_a_tiny_model_to_answer_and_to_stop(tok: Tokenizer, toy: Path, toy_run) -> None:
    base, run, summary, before_embedding = toy_run
    assert summary["status"] == "completed" and summary["before"]["val_loss"] > 4.0  # about log 261 at the start
    assert summary["val_loss"] < 0.05 and summary["val_loss_end"] < 0.05 and summary["val_loss_text"] < 0.05
    assert summary["loss_tokens"] < summary["tokens"] * 0.5  # only the assistant's side carries the loss
    for name in ("config.json", "log.jsonl", "summary.json", "model.pt"):
        assert (run / name).exists(), name
    for name in ("config.json", "log.jsonl", "summary.json"):
        assert (run.parent / "results" / "toy-chat" / name).exists(), name
    records = [json.loads(line) for line in (run / "log.jsonl").read_text().splitlines()]
    kinds = {r["kind"] for r in records}
    assert kinds == {"train", "val", "sample"}
    vals = [r for r in records if r["kind"] == "val"]
    assert vals[0]["step"] == 0 and vals[-1]["step"] == summary["steps"] and vals[-1]["val_loss"] == summary["val_loss"]
    assert [r for r in records if r["kind"] == "sample"][-1]["text"] == "C."
    config = json.loads((run / "config.json").read_text())
    assert config["base"]["checkpoint"] == "base/model.pt" and config["derived"]["passes"] == pytest.approx(1, abs=0.05)
    assert set(config["derived"]["groups"]) == {"embedding", "unembedding", "matrices"}

    # the tuned model answers every letter, stops by itself, and carries a conversation on
    tuned, loaded = chat.load_run(str(run), device="cpu")
    assert not torch.equal(tuned.embed.weight, before_embedding)
    for letter in LETTERS:
        assert chat.answer(tuned, loaded, [{"role": "user", "content": f"{letter}?"}], max_new_tokens=10) == (
            f"{letter.upper()}.", {"tokens": 2, "stopped": "end"})
    two = [{"role": "user", "content": "a?"}, {"role": "assistant", "content": "A."}, {"role": "user", "content": "h?"}]
    assert chat.answer(tuned, loaded, two, max_new_tokens=10, temperature=0.5, seed=0)[0] == "H."

    # the base, asked the same way, has no idea of the format: it does not stop
    base_model, _ = chat.load_run(str(base), device="cpu",
                                  tokenizer=str(toy.parent.parent / "tokenizers" / "bytes-chat"))
    asked = [{"role": "user", "content": "c?"}]
    assert chat.answer(base_model, loaded, asked, max_new_tokens=10)[1]["stopped"] != "end"

    # a completed run is not overwritten by accident
    with pytest.raises(SystemExit):
        chat.main(["train", "--base", str(base), "--run", "toy-chat", "--data", str(toy), "--out", str(run.parent),
                   "--plain-data", "none", "--no-compile", "--results", "none"])


def test_the_measurements_the_tally_and_the_export(tok: Tokenizer, toy: Path, toy_run, tmp_path: Path, byte_data: Path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    from gptpp import chat_eval
    from gptpp.export import chat_stage

    base, run, summary, _ = toy_run
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    # twenty questions of the eight kinds that fit the tiny model's context of 32 tokens
    kinds = [kind for kind, _ in chat_eval.QUESTIONS]
    short = [(kind, f"{LETTERS[i % len(LETTERS)]}?") for i, kind in enumerate(kinds)]
    monkeypatch.setattr(chat_eval, "QUESTIONS", short)
    (toy.parent / "val_conversations.jsonl").write_text("".join(
        json.dumps({"source": "even" if i % 2 == 0 else "odd", "messages": chat.parse(tok, ids)}) + "\n"
        for i, ids in enumerate(_val_conversations(tok, toy))))
    out, tally_file = tmp_path / "chat-stage.json", tmp_path / "chat-tally.json"
    common = ["--out", str(out), "--tally-file", str(tally_file)]
    # the run's small copies stand in for a probe: they are looked for under <repository>/results/runs
    import shutil
    shutil.copytree(run.parent / "results" / "toy-chat", tmp_path / "results" / "runs" / "toy-chat")
    monkeypatch.setattr(chat_eval, "repo_dir", lambda: tmp_path)
    # plain text for the tiny model: byte-level Shakespeare (the same 256 bytes and <|bos|> = 256)
    settings = ["--base", str(base), "--data", str(toy), "--skip-plain-text", "--plain-data", str(byte_data),
                "--plain-windows", "8", "--batch", "8",
                "--base-tokens", "6", "--example", "3", "--example-exchanges", "1"]
    # the base model's half can be computed first (while the stage runs) and taken from its file afterwards
    chat_eval.main([*settings, "--base-only", *common])
    cache_file = tmp_path / "chat-stage.base.json"
    cache = json.loads(cache_file.read_text())
    assert not out.exists() and len(cache["result"]["questions"]) == 20
    honest = cache["result"]["questions"][0]["greedy"]["text"]
    cache["result"]["questions"][0]["greedy"]["text"] = "from the file"  # a mark, to see where the text comes from
    cache_file.write_text(json.dumps(cache))
    full = ["--run", str(run), "--chat-tokens", "8", "--probes", "toy-chat"]
    chat_eval.main([*settings, *full, "--reuse-base", *common])
    m = json.loads(out.read_text())
    assert m["questions"][0]["base"]["greedy"]["text"] == "from the file"
    other = [*settings[:-6], "--base-tokens", "5", "--example", "3", "--example-exchanges", "1"]
    chat_eval.main([*other, *full, "--reuse-base", *common])  # another setting: the file does not apply
    again = json.loads(out.read_text())["questions"][0]["base"]["greedy"]
    assert again["tokens"] == 5 and again["text"] != "from the file"
    chat_eval.main([*settings, *full, *common])  # without reuse: computed again
    m = json.loads(out.read_text())
    assert m["questions"][0]["base"]["greedy"]["text"] == honest
    probe = m["probes"]["runs"][0]
    assert (probe["run"], probe["lr_scale"], probe["val_loss"]) == ("toy-chat", 1.0, summary["val_loss"])
    assert probe["before"]["val_loss"] > 4 and probe["schedule"]["schedule"] == "linear"
    assert m["held_out"]["before"]["loss"] > 4 > 0.05 > m["held_out"]["after"]["loss"]
    assert m["held_out"]["after"]["loss"] == pytest.approx(summary["val_loss"], rel=1e-4)
    assert set(m["held_out"]["after"]["by_source"]) == {"even", "odd"}
    by_source = m["held_out"]["after"]["by_source"]
    assert sum(part["tokens"] for part in by_source.values()) == m["held_out"]["after"]["tokens"]
    # by whose token it is: the two roles that carry the loss add up to the held-out loss; the stage teaches
    # them, and leaves alone the markers it never asks for
    plain = m["pretraining_windows"]  # the stage never saw plain text: it has made the tiny model worse at it
    assert plain["windows"] == 8 and plain["after"] > plain["before"] > 4
    assert "fineweb" not in plain["data"] and len(cache["result"]["pretraining_windows"]) == 1
    for side in ("before", "after"):
        roles, whole = m["roles"][side], m["held_out"][side]
        carried = roles["assistant_text"]["tokens"] + roles["assistant_end"]["tokens"]
        assert carried == whole["tokens"] and roles["assistant_end"]["tokens"] == whole["end_tokens"]
        assert roles["assistant_end"]["loss"] == pytest.approx(whole["loss_end"], rel=1e-4)
        assert roles["assistant_text"]["loss"] == pytest.approx(whole["loss_text"], rel=1e-4)
        assert roles["markers"]["tokens"] == 3 * whole["end_tokens"]  # three markers given an exchange
        assert roles["user_text"]["tokens"] == 2 * whole["end_tokens"]  # the user writes a letter and a question mark
    assert m["roles"]["after"]["assistant_end"]["loss"] < 0.05 < 2 < m["roles"]["after"]["markers"]["loss"]
    assert len(m["questions"]) == 20 and len({q["id"] for q in m["questions"]}) == 20
    assert len({q["kind"] for q in m["questions"]}) == 8
    assert all(q["base"]["greedy"]["tokens"] <= 6 and q["chat"]["sampled"]["tokens"] <= 8 for q in m["questions"])
    ex = m["example"]
    assert len(ex["tokens"]) == len(ex["mask"]) == len(ex["loss_before"]) == len(ex["loss_after"]) == len(ex["ids"])
    assert ex["loss_before"][0] is None and sum(ex["special"]) == 5
    assert ex["mask"] == chat.render(tok, ex["messages"])[1]
    carried = [after for after, mask in zip(ex["loss_after"], ex["mask"]) if mask]
    assert max(carried) < 0.05 < min(before for before, mask in zip(ex["loss_before"], ex["mask"]) if mask)
    assert "tally" not in m

    # the tally: a template with the hash of every answer, verdicts by a reader, then into the file; a tally of
    # other answers is refused
    chat_eval.main(["--tally-template", *common])
    with pytest.raises(SystemExit):
        chat_eval.main(["--tally", *common])  # no verdict yet
    tally = json.loads(tally_file.read_text())
    tally["judge"] = "the test"
    for k, entry in enumerate(tally["verdicts"].values()):
        entry["greedy"]["verdict"], entry["sampled"]["verdict"] = ("right", "wrong", "off")[k % 3], "off"
    tally_file.write_text(json.dumps(tally))
    chat_eval.main(["--tally", *common])
    m = json.loads(out.read_text())
    assert m["tally"]["greedy"] == {"right": 7, "wrong": 7, "off": 6}
    assert m["tally"]["sampled"] == {"right": 0, "wrong": 0, "off": 20}
    assert m["questions"][1]["chat"]["greedy"]["verdict"] == "wrong"
    assert sum(sum(c.values()) for c in m["tally"]["by_kind"].values()) == 20
    chat_eval.main(["--tally-template", *common])  # made again: the verdicts given to the same answers are kept
    assert json.loads(tally_file.read_text())["verdicts"]["fact-1"]["greedy"]["verdict"] in ("right", "wrong", "off")
    tally["verdicts"]["fact-1"]["greedy"]["sha"] = "0" * 10
    tally_file.write_text(json.dumps(tally))
    with pytest.raises(SystemExit):
        chat_eval.main(["--tally", *common])

    # another full run measured beside the first (here the same run under its own name): no questions, no tally
    chat_eval.main([*settings, "--add-run", str(run), *common])
    chat_eval.main([*settings, "--add-run", str(run), *common])  # again: it replaces its entry
    m = json.loads(out.read_text())
    (other,) = m["other_runs"]["runs"]
    assert other["run"] == "toy-chat" and other["lr_scale"] == 1.0 and "questions" not in other and "tally" in m
    assert other["held_out"]["loss"] == pytest.approx(m["held_out"]["after"]["loss"], rel=1e-6)
    assert other["roles"]["assistant_end"] == m["roles"]["after"]["assistant_end"]
    assert other["pretraining_windows"]["after"] == pytest.approx(m["pretraining_windows"]["after"], rel=1e-6)
    assert other["monitor"]["val_loss"] == summary["val_loss"] and "plain_text" not in other

    # the export for the figure: the measurements and the run's curve in one small file
    data = chat_stage.export(out, str(run), tmp_path / "site")
    written = json.loads((tmp_path / "site" / "chat-stage.json").read_text())
    assert written["stage"]["steps"] == summary["steps"] and written["curve"]["val"]["tokens"][-1] == summary["tokens"]
    assert len(written["curve"]["val"]["loss"]) == len(written["curve"]["sample"]["text"]) >= 3
    assert written["curve"]["train"]["tokens"][0] < written["curve"]["train"]["tokens"][-1] <= summary["tokens"]
    assert written["tally"]["greedy"] == {"right": 7, "wrong": 7, "off": 6} and written["plain_text"] is None
    assert [q["chat"]["greedy"]["verdict"] for q in written["questions"]][:3] == ["right", "wrong", "off"]
    assert written["example"]["mask"] == ex["mask"] and data["questions"][0]["text"] == "a?"
    assert written["roles"]["after"]["assistant_end"]["tokens"] == m["held_out"]["after"]["end_tokens"]
    first = data["questions"][0]["chat"]["greedy"]
    assert first["text"] == "A." and first["stopped"] == "end"


def _val_conversations(tok: Tokenizer, toy: Path) -> list[list[int]]:
    data = chat.ChatData(toy)
    tokens = data.val_tokens()
    starts, lengths = chat.conversations(tokens, data.bos_id)
    return [tokens[s:s + n].tolist() for s, n in zip(starts, lengths)]


# ----------------------------------------------------------------------------------------------------------------
# The selection of conversations (scripts/prepare_chat.py)


def test_the_selection_keeps_whole_exchanges_that_are_plain_short_and_fit(tok: Tokenizer, tmp_path: Path) -> None:
    import importlib.util

    from conftest import REPO

    spec = importlib.util.spec_from_file_location("prepare_chat", REPO / "scripts" / "prepare_chat.py")
    prepare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)
    prepare._init_worker(str(tok.save(tmp_path / "bytes-chat")), 10, 40)  # answers of at most 10 tokens, 40 in all

    def conversation(*contents: str, first: str = "user") -> dict:
        roles = ("user", "assistant") if first == "user" else ("assistant", "user")
        return {"source": "test", "category": "test",
                "messages": [{"role": roles[i % 2], "content": c} for i, c in enumerate(contents)]}

    batch = [
        conversation("Hi", "Hello!", "How are you?", "Fine."),          # 0: both exchanges
        conversation("Hi", "Hello!", "And?", "An answer too long."),    # 1: the second answer has 19 tokens
        conversation("Hi {", "Hello!", "How are you?", "Fine."),        # 2: a brace in the first message
        conversation("Hi", "Hello!", "Code?", "```x```"),               # 3: a code fence in the second answer
        conversation("Hi", "Hello!", "A long question, is it not?", "Yes."),  # 4: the second exchange does not fit
        conversation("Hello!", "Hi", first="assistant"),                # 5: the assistant speaks first
        conversation("  Hi \n", " Hello!  ", "x"),                      # 6: stripped; a last message without an answer
        conversation("Hi", "   "),                                      # 7: an empty answer
        conversation("Tag?", "<b>bold</b>"),                            # 8: markup
        conversation("Slash?", "a\\b"),                                 # 9: a backslash
    ]
    kept = {index: (ids.tolist(), mask.tolist(), exchanges, n_bytes)
            for index, ids, mask, exchanges, n_bytes in prepare._select(batch)}
    assert sorted(kept) == [0, 1, 3, 4, 6]
    assert [kept[i][2] for i in (0, 1, 3, 4, 6)] == [2, 1, 1, 1, 1]
    for i, (ids, mask, exchanges, n_bytes) in kept.items():
        messages = [{"role": m["role"], "content": m["content"].strip()} for m in batch[i]["messages"][:2 * exchanges]]
        assert (ids, mask) == chat.render(tok, messages) and len(ids) <= 40  # rendered as the stage renders, whole
        assert n_bytes == sum(len(m["content"].encode()) for m in messages)
        assert max(len(tok.encode(m["content"])) for m in messages if m["role"] == "assistant") <= 10
