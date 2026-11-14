"""A defect found by the review of the experiments (14 November 2026), demonstrated here and not yet fixed.

The 2026 recipe trains the embedding and the unembedding with AdamW at 0.3 and 0.008 times sqrt(768 / d_model)
(docs/DESIGN.md, Section 2.9). `train_preset` applies that rule to the width of the preset's *size*. A width
changed on the command line, `--preset gptpp-ladder --set d_model=128 ...`, keeps the rates of the preset's own
width: 0.3674 and 0.0098, where the rule gives 0.7348 and 0.0196.

That is how the scaling grid was launched (experiments/make_jobs.py, whose comment says "Rates of the vocabulary
matrices scale with width by the preset's rule"), so every size of the grid, from width 128 to width 768, was
trained at the rates of width 512: see `derived.groups` in results/runs/grid-*/config.json.

The first test states what the trainer should do and is marked xfail(strict=True): the suite stays green while
the defect stands, and the test turns into a failure (an unexpected pass) the day the rule follows the overridden
width; remove the mark then. The other two say what a fix must keep.

    cd gpt-pp && $PY -m pytest -q tests/test_width_rule_under_overrides.py
"""

from __future__ import annotations

import math

import pytest

from gptpp import train
from gptpp.config import TrainConfig


def configs(*overrides: str, preset: str = "gptpp-ladder") -> TrainConfig:
    """The training configuration the trainer builds for `--preset <preset> --set <overrides>`."""
    args = train.parse_args(["--preset", preset, "--run", "unused", "--set", *overrides])
    return train.build_configs(args)[1]


GRID = [(128, 4), (192, 6), (256, 8), (384, 8), (768, 12)]  # the widths of the scaling grid other than the preset's


@pytest.mark.xfail(strict=True, reason="--set d_model=... leaves the embedding and unembedding rates of the "
                                       "preset's own width (the scaling grid ran this way)")
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
