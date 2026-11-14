"""Tests of gptpp/interp/patching.py on tiny random models: the direct attributions add up to the logit difference,
patching a whole layer restores the clean run, the first and last cells of the stream's map are what they must be.

    cd gpt-pp && $PY -m pytest -q tests/test_interp_patching.py
"""

from __future__ import annotations

import pytest
import torch

from gptpp.config import ModelConfig
from gptpp.interp import patching as P
from gptpp.model import GPT

VARIANTS = [{}, dict(logit_softcap=0.0), dict(norm="rmsnorm"), dict(norm="layernorm", bias=True), dict(norm="none", embed_norm=False),
            dict(n_kv_head=2), dict(pos="learned", norm="layernorm", embed_norm=False, act="gelu", qk_norm=False, bias=True,
                                    tie_embeddings=True, logit_softcap=0.0)]


def tiny(**changes: object) -> GPT:
    base = dict(vocab_size=50, n_layer=3, n_head=4, d_model=32, block_size=24, init="gpt2")
    torch.manual_seed(1)
    model = GPT(ModelConfig().replace(**{**base, **changes})).eval()
    with torch.no_grad():                                  # larger weights: patterns far from uniform, effects far from zero
        for p in model.blocks.parameters():
            p.mul_(4.0)
    return model


def prompts() -> tuple[torch.Tensor, torch.Tensor]:
    clean = torch.tensor([3, 14, 15, 9, 26, 5, 35, 8])
    corrupt = clean.clone()
    corrupt[2] = 40
    return clean, corrupt


@pytest.mark.parametrize("changes", VARIANTS)
def test_attributions_add_up_to_the_logit_difference(changes):
    model = tiny(**changes)
    clean, _ = prompts()
    d = P.direct_attribution(model, clean, 7, 11)
    logits = model.unembed(model.run_with_cache(clean.view(1, -1), names=["final_norm"])[1]["final_norm"])[0, -1]
    assert d["heads"].shape == (3, 4) and d["mlp"].shape == (3,)
    assert abs(d["ld"] - float((logits[7] - logits[11]).detach())) < 1e-4            # before the cap
    assert abs(d["total"] - d["ld"]) < 2e-4, (d["total"], d["ld"])
    assert abs(float(P.logit_difference(model, clean.view(1, -1), 7, 11)) - d["ld"]) < 1e-5
    sources = P.head_sources(model, clean, 7, 11)
    assert torch.allclose(sources.sum(-1), d["heads"], atol=2e-4)
    pattern = P.attention_from(model, clean)
    assert pattern.shape == (3, 4, 8) and torch.allclose(pattern.sum(-1), torch.ones(3, 4), atol=1e-5)


def test_attribution_refuses_what_is_not_linear():
    with pytest.raises(NotImplementedError):
        P.direct_attribution(tiny(norm="dyt"), prompts()[0], 7, 11)
    with pytest.raises(NotImplementedError):
        P.direct_attribution(tiny(norm_placement="post"), prompts()[0], 7, 11)


def test_the_map_by_hand_and_at_its_ends():
    model = tiny()
    clean, corrupt = prompts()
    e = P.patching_effects(model, clean, corrupt, 7, 11, batch=5)
    assert abs(e["ld_clean"] - e["ld_corrupt"]) > 1e-3
    stream = e["denoise"]["resid_pre"]
    assert stream.shape == (3, 8)
    # the stream entering the first layer is the embedding: restoring the changed token there restores everything
    assert abs(float(stream[0, 2]) - 1) < 1e-4 and stream[0, [0, 1, 3, 4, 5, 6, 7]].abs().max() < 1e-4
    assert stream[:, :2].abs().max() < 1e-4                # positions before the change are the same in both runs
    noise = e["noise"]["resid_pre"]
    assert abs(float(noise[0, 2])) < 1e-4 and (noise[0, [0, 1, 3, 4, 5, 6, 7]] - 1).abs().max() < 1e-4
    # one cell by hand, with the model's own hooks
    _, cache = model.run_with_cache(clean.view(1, -1), names=["mlp_out.1"])

    def restore(x):
        x = x.clone()
        x[:, 6] = cache["mlp_out.1"][:, 6]
        return x

    ld = float(P.logit_difference(model, corrupt.view(1, -1), 7, 11, {"mlp_out.1": restore}))
    by_hand = (ld - e["ld_corrupt"]) / (e["ld_clean"] - e["ld_corrupt"])
    assert abs(float(e["denoise"]["mlp_out"][1, 6]) - by_hand) < 1e-4
    # restoring every position of one layer's stream restores the clean run
    _, cache = model.run_with_cache(clean.view(1, -1), names=["resid_pre.2"])
    whole = float(P.logit_difference(model, corrupt.view(1, -1), 7, 11, {"resid_pre.2": lambda x: cache["resid_pre.2"]}))
    assert abs(whole - e["ld_clean"]) < 1e-4
    with pytest.raises(ValueError):
        P.patch_map(model, clean, corrupt[:-1], 7, 11)
