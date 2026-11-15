"""scripts/export_hf.py: a gpt-pp model written in the format of transformers computes the same logits."""

import importlib.util
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")

from gptpp.config import preset  # noqa: E402
from gptpp.hf.modeling_gptpp import GptPPForCausalLM  # noqa: E402
from gptpp.model import GPT  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("export_hf", REPO / "scripts" / "export_hf.py")
export_hf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export_hf)


def tiny(**overrides) -> GPT:
    cfg = preset("gptpp-pico").replace(n_layer=2, d_model=64, n_head=4, vocab_size=300, block_size=64, **overrides)
    torch.manual_seed(0)
    model = GPT(cfg)
    with torch.no_grad():  # the recipe starts the writers at zero: give every weight a value
        for p in model.parameters():
            p.normal_(0.0, 0.3)
    return model.eval()


@pytest.mark.parametrize("overrides", [{}, {"n_kv_head": 2}])
def test_the_exported_model_computes_gptpps_logits(tmp_path, overrides):
    model = tiny(**overrides)
    export_hf.write_model(model, tmp_path)
    hf = GptPPForCausalLM.from_pretrained(tmp_path, dtype=torch.float32).eval()
    idx = torch.randint(0, 300, (2, 50), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        ours, theirs = model(idx), hf(idx).logits
    assert (ours - theirs).abs().max().item() < 1e-4
    assert hf.config.qk_gain == pytest.approx(model.cfg.qk_scale / model.cfg.d_head ** 0.5)


def test_negating_the_second_halves_twice_gives_the_weights_back():
    w = torch.randn(4 * 16, 64)
    once = export_hf.negate_second_halves(w, 4, 16)
    assert torch.equal(once.view(4, 16, 64)[:, :8], w.view(4, 16, 64)[:, :8])
    assert torch.equal(export_hf.negate_second_halves(once, 4, 16), w)


@pytest.mark.parametrize("overrides", [{"pos": "learned"}, {"norm": "layernorm"}, {"act": "swiglu"},
                                       {"tie_embeddings": True}, {"qk_scale_learned": True}, {"norm_placement": "post"}])
def test_other_recipes_are_refused(overrides):
    assert export_hf.refusal(preset("gptpp-pico").replace(**overrides)) is not None
    assert export_hf.refusal(preset("gptpp-pico")) is None
