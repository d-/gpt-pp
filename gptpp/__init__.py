"""gpt-pp: a small modern GPT written from scratch, with every component switchable between its historical variants.

The package is import-light on purpose: `import gptpp` loads nothing heavy, so the tokenizer code can be used
without PyTorch. The pieces are imported by name:

    from gptpp.config import ModelConfig, TrainConfig, preset     # what the model is and how it is trained
    from gptpp.model import GPT                                   # the model, its hooks, cached decoding
    from gptpp.optim import build_optimizers, lr_multiplier       # AdamW groups, Muon, SGD, Adam, Lion; schedules
    from gptpp.data import TokenData, TrainLoader, ValLoader      # token shards on disk
    python -m gptpp.train ...                                     # the trainer
    python -m gptpp.generate ...                                  # sampling
"""

from __future__ import annotations

import os
from pathlib import Path

__version__ = "0.1.0"

DEFAULT_WORK = "/home/ubuntu/d-work/gpt"


def work_dir() -> Path:
    """The workspace that holds data, checkpoints and logs (never committed): $GPTPP_WORK, or the lab's default."""
    return Path(os.environ.get("GPTPP_WORK", DEFAULT_WORK))


def repo_dir() -> Path:
    """The root of the repository this package was imported from (the parent of `gptpp/`)."""
    return Path(__file__).resolve().parent.parent
