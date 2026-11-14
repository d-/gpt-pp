"""A defect found by the review of the experiments (14 November 2026), demonstrated here; the switch
`pos_after_norm` is the fix, and the run `swap-pos-sinusoidal-after-norm` the measurement made with it.

With `pos=sinusoidal` the fixed sinusoids (amplitude `embed_std`, which is 1 in the 2026 recipe) are added to
the token vector BEFORE the embedding norm. The token table is trained at the rate of a lookup table (0.37 at
width 512) without decay, and its rows grow, while the sinusoids cannot. In the finished run
`swap-pos-sinusoidal` the token vectors of running text have a root-mean-square of 5.9 and the sinusoids one of
0.71: the position's share of the energy that enters the norm fell from a third at initialization to 1.4%.
That run measures a position signal that fades during training, not sinusoidal positions.

With the sinusoids added after the norm instead (a scratch patch of the review, the same run for 10M tokens),
the held-out loss was 5.233 where the lab's run had 5.432, learned positions 5.427, no positions 5.553 and
rotary 5.118.

The second test states the defect of the default arrangement; the third, that `pos_after_norm` removes it.

    cd gpt-pp && $PY -m pytest -q tests/test_sinusoid_share_under_embed_norm.py
"""

from __future__ import annotations

import pytest
import torch

from gptpp.config import preset
from gptpp.model import GPT


def position_share(model: GPT, idx: torch.Tensor) -> float:
    """The share of the embedded vectors' energy that depends on the position: for one token placed at every
    position, the variance of its vector over the positions divided by the mean square of the vector."""
    with torch.no_grad():
        x = model.embed_tokens(idx)  # (1, T, d)
    return float(x.var(dim=1, unbiased=False).sum() / x.pow(2).sum(-1).mean())


def sinusoidal_model() -> tuple[GPT, torch.Tensor]:
    torch.manual_seed(0)
    model = GPT(preset("gptpp-pico").replace(pos="sinusoidal"))
    return model, torch.full((1, 128), 7)  # token 7 at 128 positions


def test_at_initialization_the_position_is_a_visible_part_of_the_embedding() -> None:
    model, idx = sinusoidal_model()
    assert position_share(model, idx) > 0.1


def grown(model: GPT, idx: torch.Tensor) -> float:
    with torch.no_grad():
        model.embed.weight.mul_(6.0)  # what 50M tokens of training did to the rows in use (measured: 5.9)
    return position_share(model, idx)


def test_before_the_norm_the_position_fades_when_the_token_table_grows() -> None:
    """The arrangement of the run `swap-pos-sinusoidal` (the default, kept so that the run can be reproduced)."""
    model, idx = sinusoidal_model()
    before = position_share(model, idx)
    assert grown(model, idx) < 0.2 * before


def test_after_the_norm_the_position_does_not_fade() -> None:
    """`pos_after_norm=true`, the arrangement of the run `swap-pos-sinusoidal-after-norm`."""
    torch.manual_seed(0)
    model = GPT(preset("gptpp-pico").replace(pos="sinusoidal", pos_after_norm=True))
    idx = torch.full((1, 128), 7)
    before = position_share(model, idx)
    assert before > 0.1  # sinusoids of rms 0.707 on a token vector of rms 1; the slow ones barely vary over 128 positions
    assert grown(model, idx) > 0.95 * before
