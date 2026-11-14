"""Tests of gptpp/interp/memories.py on a tiny random model: the top triggers found in one pass are those of a
brute-force search, the value scores are the unembedding of the output weights, the measures are their definitions.

    cd gpt-pp && $PY -m pytest -q tests/test_interp_memories.py
"""

from __future__ import annotations

import torch

from gptpp.config import ModelConfig
from gptpp.interp import memories as M
from gptpp.model import GPT


def tiny(**changes: object) -> GPT:
    base = dict(vocab_size=50, n_layer=3, n_head=4, d_model=32, block_size=24, init="gpt2")
    torch.manual_seed(2)
    return GPT(ModelConfig().replace(**{**base, **changes})).eval()


def test_top_triggers_are_the_brute_force_ones():
    model = tiny()
    text = torch.randint(0, 50, (7, 12))
    neurons = {0: torch.tensor([0, 5, 100]), 2: torch.tensor([7, 8, 9, 127])}
    found = M.top_triggers(model, iter([text[:3], text[3:5], text[5:]]), neurons, k=6)
    _, cache = model.run_with_cache(text, names=["mlp.*.post"])
    for layer, cols in neurons.items():
        acts = cache[f"mlp.{layer}.post"][..., cols].reshape(7 * 12, -1)             # (positions, n)
        vals, where = acts.topk(6, dim=0)
        assert torch.allclose(found[layer]["coef"], vals.T, atol=1e-6)
        got = found[layer]["window"] * 12 + found[layer]["pos"]
        for n in range(len(cols)):                         # the same positions wherever the coefficients are distinct
            distinct = vals[:, n].unique().numel() == 6
            assert not distinct or got[n].tolist() == where[:, n].tolist()
        assert (found[layer]["coef"][:, :-1] >= found[layer]["coef"][:, 1:]).all() and (found[layer]["coef"] >= 0).all()


def test_values_through_the_unembedding():
    model = tiny()
    cols = torch.tensor([3, 17])
    scores = M.value_scores(model, 1, cols)
    assert scores.shape == (2, 50)
    by_hand = model.unembed.weight @ model.blocks[1].mlp.proj.weight[:, 17]
    assert torch.allclose(scores[1], by_hand, atol=1e-6)
    ends = M.promoted(scores, k=4)
    assert ends["top_ids"][1].tolist() == by_hand.argsort(descending=True)[:4].tolist()
    assert ends["bottom_ids"][1].tolist() == by_hand.argsort()[:4].tolist()
    assert torch.allclose(ends["bottom_scores"][1], by_hand.sort().values[:4], atol=1e-6)
    # a neuron that fires adds its value to the stream: the MLP's output is the coefficients times the values
    idx = torch.randint(0, 50, (1, 8))
    _, cache = model.run_with_cache(idx, names=["mlp.1.post", "mlp_out.1"])
    assert torch.allclose(cache["mlp.1.post"] @ M.value_vectors(model, 1), cache["mlp_out.1"], atol=1e-5)


def test_measures_by_hand():
    triggers = torch.tensor([[4, 4, 4, 9], [1, 2, 3, 3], [5, 6, 7, 8]])
    nexts = torch.tensor([[10, 11, 10, 12], [20, 21, 22, 23], [30, -1, 31, 32]])
    coef = torch.tensor([[3.0, 2.0, 1.0, 0.5], [3.0, 2.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    top_ids = torch.tensor([[10, 12], [21, 20], [40, 41]])
    m = M.neuron_measures(triggers, nexts, coef, top_ids)
    assert m["same_last"].tolist() == [0.75, 0.5, 0.25] and m["last_token"].tolist()[:2] == [4, 3]
    assert m["agrees"].tolist() == [True, False, False]    # the top token against the token after the top trigger
    assert m["follows"].tolist() == [0.75, 0.5, 0.0]
    assert m["alive"].tolist() == [True, False, True]
    s = M.layer_statistics(m)
    assert s == {"same_last": 0.5, "agreement": 0.5, "follows": 0.375, "neurons": 3, "alive": 2}


def test_a_snippet_leaves_out_addresses_and_numbers():
    """The exporter's snippets: an e-mail address or a telephone number never reaches the page; the trigger does."""
    import numpy as np

    from gptpp.export.interp_common import Texts
    from gptpp.export.neurons_as_memories import LEFT_OUT, snippet

    class Bytes:                                           # a tokenizer of single characters
        def decode(self, ids):
            return "".join(chr(i) for i in ids)

    texts = Texts(Bytes())
    encode = lambda s: np.array([ord(ch) for ch in s])
    before, trigger, after, n = snippet(texts, encode("write to ann@example.org or call 555-123-4567 "), ord("n"), encode("ow"))
    shown = "".join(texts.table[i] for i in before) + "|" + texts.table[trigger] + "|" + "".join(texts.table[i] for i in after)
    assert shown == f"write to {LEFT_OUT} or call {LEFT_OUT} |n|ow" and n == 2
    before, trigger, after, n = snippet(texts, encode("in 1999-2000 and 2001, 12,345,678,901 "), ord("x"), encode("y"))
    assert "".join(texts.table[i] for i in before) == "in 1999-2000 and 2001, 12,345,678,901 " and n == 0
    before, trigger, after, n = snippet(texts, encode("a@b.c"), ord("o"), encode("m"))          # the trigger itself is kept
    assert texts.table[trigger] == "o" and n == 1
