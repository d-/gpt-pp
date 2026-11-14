"""The width rule of the 2026 recipe when the width is not the preset's own.

The recipe trains the embedding and the unembedding with AdamW at 0.3 and 0.008 times sqrt(768 / d_model)
(docs/DESIGN.md, Section 2.9). The trainer applies the rule to the width of the model that will be trained: the
preset's own, a width set on the command line (`--preset gptpp-ladder --set d_model=128 ...`), or the model of
another preset under --train-preset. A rate given on the command line wins.

Until 14 November 2026 `train_preset` applied the rule to the width of the preset's *size* only, and a width set
on the command line kept the rates of the preset's own width: 0.3674 and 0.0098 on gptpp-ladder, where the rule
gives 0.7348 and 0.0196 at width 128. The review of the experiments found it (docs/REVIEW.md) and the first test
below stood here as a strict xfail until the trainer was fixed. The scaling grid had been launched that way
(experiments/make_jobs.py), so every size of `grid-*`, from width 128 to width 768, trained at the rates of width
512: see `derived.groups` in results/runs/grid-*/config.json. The runs `gridr-*` are the same commands run again
under the rule. The last test keeps the first grid's record readable as it was written.

    cd gpt-pp && $PY -m pytest -q tests/test_width_rule_under_overrides.py
"""

from __future__ import annotations

import json
import math
import warnings
from typing import Optional

import pytest
from conftest import REPO

from gptpp import train
from gptpp.config import ModelConfig, TrainConfig, train_preset


def configs(*overrides: str, preset: str = "gptpp-ladder", trained_as: Optional[str] = None) -> TrainConfig:
    """The training configuration the trainer builds for `--preset <preset> [--train-preset <trained_as>] --set
    <overrides>`."""
    flags = ["--train-preset", trained_as] if trained_as else []
    args = train.parse_args(["--preset", preset, *flags, "--run", "unused", "--set", *overrides])
    return train.build_configs(args)[1]


GRID = [(128, 4), (192, 6), (256, 8), (384, 8), (768, 12)]  # the widths of the scaling grid other than the preset's


@pytest.mark.parametrize("width, heads", GRID)
def test_the_rates_follow_a_width_set_on_the_command_line(width: int, heads: int) -> None:
    cfg = configs(f"d_model={width}", f"n_head={heads}")
    scale = math.sqrt(768 / width)
    assert cfg.lr_embedding == pytest.approx(0.3 * scale, abs=1e-4)
    assert cfg.lr_unembedding == pytest.approx(0.008 * scale, abs=1e-5)


def test_the_preset_width_itself_follows_the_rule() -> None:
    cfg = configs()
    assert cfg.lr_embedding == pytest.approx(0.3 * math.sqrt(768 / 512), abs=1e-4)
    assert cfg.lr_unembedding == pytest.approx(0.008 * math.sqrt(768 / 512), abs=1e-5)


def test_a_rate_given_on_the_command_line_wins() -> None:
    cfg = configs("d_model=128", "n_head=4", "lr_embedding=0.5", "lr_unembedding=0.01")
    assert (cfg.lr_embedding, cfg.lr_unembedding) == (0.5, 0.01)
    # one rate given (with the prefix or without): the other still follows the width
    cfg = configs("d_model=128", "n_head=4", "train.lr_embedding=0.5")
    assert (cfg.lr_embedding, cfg.lr_unembedding) == (0.5, 0.0196)
    # a rate is given even when it is the value the preset's own width would have had, or none (= lr)
    cfg = configs("d_model=128", "n_head=4", "lr_embedding=0.3674", "lr_unembedding=none")
    assert (cfg.lr_embedding, cfg.lr_unembedding) == (0.3674, None)


def test_the_width_changes_the_two_rates_and_nothing_else() -> None:
    narrow, own = configs("d_model=128", "n_head=4", "n_layer=4"), configs()
    assert {name for name, value in narrow.to_dict().items() if value != getattr(own, name)} == \
        {"lr_embedding", "lr_unembedding"}
    assert (narrow.lr_embedding, narrow.lr_unembedding) == (0.7348, 0.0196)
    # the same rates as the preset whose own width is 128
    pico = train_preset("gptpp-pico")
    assert (narrow.lr_embedding, narrow.lr_unembedding) == (pico.lr_embedding, pico.lr_unembedding)
    # the dedicated flags and the other overrides are applied as before (the widest command of the grid)
    args = train.parse_args(["--preset", "gptpp-ladder", "--run", "unused", "--tokens", "25e6", "--seed", "3",
                             "--eval-tokens", "12.5e6", "--no-compile",
                             "--set", "d_model=768", "n_layer=12", "n_head=12", "device_batch=64", "grad_accum=2"])
    model_cfg, cfg = train.build_configs(args)
    assert (model_cfg.d_model, model_cfg.n_layer, model_cfg.n_head, model_cfg.block_size) == (768, 12, 12, 512)
    assert (cfg.tokens, cfg.seed, cfg.eval_tokens, cfg.compile, cfg.device_batch, cfg.grad_accum) == \
        (25e6, 3, 12.5e6, False, 64, 2)
    assert (cfg.lr_embedding, cfg.lr_unembedding, cfg.lr) == (0.3, 0.008, 0.02)


def test_the_rule_goes_by_the_model_that_is_trained_under_another_training_preset() -> None:
    # the ladder's seventh rung: the 2019 model trained the 2026 way, both at the ladder's width
    cfg = configs(preset="gpt2-2019-ladder", trained_as="gptpp-ladder")
    assert (cfg.optimizer, cfg.lr_embedding, cfg.lr_unembedding) == ("muon+adamw", 0.3674, 0.0098)
    # a model of another width than the training preset's size: the model's width counts
    cfg = configs(preset="gpt2-2019-base", trained_as="gptpp-ladder")
    assert (cfg.lr_embedding, cfg.lr_unembedding, cfg.device_batch) == (0.3, 0.008, 128)
    cfg = configs("d_model=256", preset="gpt2-2019-ladder", trained_as="gptpp-ladder")
    assert (cfg.lr_embedding, cfg.lr_unembedding) == (0.5196, 0.01386)


def test_the_2019_recipe_has_no_width_rule() -> None:
    narrow, own = configs("d_model=128", "n_head=4", preset="gpt2-2019-ladder"), configs(preset="gpt2-2019-ladder")
    assert narrow == own and (own.lr, own.lr_embedding, own.lr_unembedding) == (2.4e-3, None, None)
    assert configs(preset="gptpp-ladder", trained_as="gpt2-2019-ladder") == own


def test_the_first_grid_keeps_its_record() -> None:
    """The runs made before the fix say what they were trained with: their configurations load as they were
    written, with the rates of width 512 at every width."""
    paths = sorted((REPO / "results" / "runs").glob("grid-w*/config.json"))
    if not paths:
        pytest.skip("this checkout has no results/runs")
    assert len(paths) == 23
    for path in paths:
        config = json.loads(path.read_text())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # a model field added since then is absent from the record
            model_cfg = ModelConfig.from_dict(config["model_config"])
        cfg = TrainConfig.from_dict(config["train_config"])
        assert cfg.to_dict() == config["train_config"], path
        assert model_cfg.d_model == int(path.parent.name.split("-")[1][1:])
        assert (cfg.lr_embedding, cfg.lr_unembedding) == (0.3674, 0.0098), path
        groups = config["derived"]["groups"]
        assert (groups["embedding"]["lr"], groups["unembedding"]["lr"]) == (0.3674, 0.0098), path
