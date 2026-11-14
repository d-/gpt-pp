"""What the exporters of the circuits figures share: the runs of the attention-only models, their checkpoints, the
data they are measured on, and the names of tokens as a page prints them.

The circuit models (docs of gptpp/interp/circuits.py and gptpp/interp/repeated_token_windows.py):

    circ-attn1-repeats, circ-attn2-repeats   one and two layers, attention only, 256 wide, 8 heads, trained on
                                             TinyStories with one window in ten of repeated random tokens
    circ-attn1, circ-attn2                   the same models trained on TinyStories alone

Every exporter takes its runs as arguments (a run directory or a name under $GPTPP_WORK/ckpt), so the same code
runs on any other attention-only model of the lab.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional, Sequence, Union

import torch
from torch import Tensor

from ..config import ModelConfig
from ..data import TokenData
from ..interp import ngram_order as ng
from ..interp.repeated_token_windows import load_pool
from ..model import GPT
from ..tokenizer import Tokenizer
from .common import CKPT

TWO, ONE = "circ-attn2-repeats", "circ-attn1-repeats"  # the models of the figures
TWO_STORIES, ONE_STORIES = "circ-attn2", "circ-attn1"  # the same models trained on the stories alone


def run_dir(run: Union[str, Path]) -> Path:
    """A run directory: the path given, or the run of that name under $GPTPP_WORK/ckpt."""
    path = Path(run)
    return path if (path / "config.json").exists() else CKPT / str(run)


def pick_device(device: Optional[str] = None, need_gb: float = 3.0) -> str:
    """`device` if given; else the first visible card if it has `need_gb` free, else the CPU."""
    if device:
        return device
    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info(0)
        if free / 2**30 >= need_gb:
            return "cuda"
    return "cpu"


class Run:
    """One training run of a circuit model: its configuration, its data, its checkpoints."""

    def __init__(self, run: Union[str, Path], device: str = "cpu"):
        self.dir, self.device = run_dir(run), device
        self.config = json.loads((self.dir / "config.json").read_text())
        self.name = self.config["run"]
        self.cfg = ModelConfig.from_dict(self.config["model_config"])
        self.data = TokenData(self.config["train_config"]["data"])
        self.tokens_per_step = int(self.config["derived"]["tokens_per_step"])
        self.total_tokens = int(self.config["derived"]["tokens"])
        self.repeats = self.data.meta.get("repeats")  # None for a model trained on the corpus alone

    def checkpoints(self) -> list[tuple[int, Path]]:
        """(step, file) of every checkpoint, by step; the final model stands for its step if no checkpoint does."""
        found = {int(re.fullmatch(r"ckpt_(\d+)\.pt", p.name).group(1)): p for p in self.dir.glob("ckpt_*.pt")}
        last = int(self.config["derived"]["steps"])
        found.setdefault(last, self.dir / "model.pt")
        return sorted(found.items())

    def model(self, path: Optional[Path] = None) -> GPT:
        """The model of a checkpoint (the final one by default), in evaluation mode on the run's device."""
        return GPT.from_checkpoint(path or self.dir / "model.pt", device=self.device)[0]

    def describe(self) -> dict:
        """What a figure says of the model: its shape, its data, its budget, its held-out loss."""
        summary = json.loads((self.dir / "summary.json").read_text())
        cfg = self.cfg
        return {
            "run": self.name, "layers": cfg.n_layer, "heads": cfg.n_head, "width": cfg.d_model, "headWidth": cfg.d_head,
            "context": cfg.block_size, "vocabulary": cfg.vocab_size, "parameters": self.config["params"]["total"],
            "ropeBase": cfg.rope_base, "position": cfg.pos, "norm": cfg.norm, "init": cfg.init,
            "data": self.config["data_meta"]["dataset"],
            "repeatFraction": self.repeats["fraction"] if self.repeats else 0.0,
            "repeatBlock": [self.repeats["min_block"], self.repeats["max_block"]] if self.repeats else None,
            "tokens": self.total_tokens, "steps": int(self.config["derived"]["steps"]),
            "tokensPerStep": self.tokens_per_step, "heldOutLoss": summary["val_loss"],
            "optimizer": self.config["train_config"]["optimizer"], "seed": self.config["train_config"]["seed"],
        }


def story_data() -> TokenData:
    """TinyStories itself: the held-out text every circuit model is measured on, whatever it was trained on."""
    return TokenData("tinystories/bpe4k")


def pool_for(runs: Sequence[Run]) -> Tensor:
    """The pool random sequences are drawn from: the one the mixed training data was made with, if any run was
    trained on such data; else the same rule on TinyStories' counts."""
    for run in runs:
        if run.repeats:
            return load_pool(run.data)
    return load_pool(story_data())


def held_out(n: int, length: int = 256) -> Tensor:
    """The first n windows of the held-out stories, (n, length)."""
    return ng.held_out_windows(story_data(), n, length)


def token_names(tokenizer: Optional[Tokenizer] = None) -> list[str]:
    """Every token of the vocabulary as text: what `decode` gives for it alone. A token that is part of a
    multi-byte character decodes to the replacement character; the special tokens keep their names."""
    tokenizer = tokenizer or Tokenizer.load("bpe4k")
    return [tokenizer.decode([i]) for i in range(tokenizer.vocab_size)]


def is_word(name: str) -> bool:
    """A token a reader can read alone: a space and a whole lower- or upper-case word, or punctuation."""
    return bool(re.fullmatch(r" ?[A-Za-z][a-z']*|[.,!?;:\"']+|\.\n|<\|bos\|>", name))
