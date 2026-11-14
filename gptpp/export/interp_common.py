"""What the exporters of the lens, patching and memory figures share: loading a checkpoint with its place in the
run, the tokenizer's text of a token, a table of the token texts an export uses, and prompts as token ids."""
from __future__ import annotations

import re
from pathlib import Path

import torch

from ..config import ModelConfig
from ..model import GPT
from ..tokenizer import Tokenizer

TOKENIZER = "bpe32k"


def pick_device(name: str | None) -> str:
    return name or ("cuda" if torch.cuda.is_available() else "cpu")


def load_model(ckpt: str | Path, device: str) -> tuple[GPT, dict]:
    """The model of a checkpoint (`model.pt`, `ckpt_<step>.pt`, or the run directory that holds `model.pt`), in
    evaluation mode on `device`, and what the figure says about it: the file, the step and the tokens seen."""
    path = Path(ckpt)
    if path.is_dir():
        path = path / "model.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    cfg = ModelConfig.from_dict(checkpoint["model_config"])
    model = GPT(cfg)
    state = {re.sub(r"^(module\.|_orig_mod\.)+", "", key): value for key, value in checkpoint["model"].items()}
    model.load_state_dict(state)
    meta = {"name": "gpt-pp", "run": path.parent.name, "checkpoint": path.name, "final": path.name == "model.pt",
            "step": int(checkpoint.get("step", -1)), "tokens": float(checkpoint.get("tokens", -1)),
            "layers": cfg.n_layer, "heads": cfg.n_head, "width": cfg.d_model, "mlp": cfg.d_mlp,
            "vocabulary": cfg.vocab_size, "activation": cfg.act, "norm": cfg.norm, "softcap": cfg.logit_softcap}
    return model.to(device).eval(), meta


class Texts:
    """The texts of the tokens an export mentions, each stored once: `index(token id)` is its place in `table`."""

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer, self.table, self.where = tokenizer, [], {}

    def index(self, token: int) -> int:
        token = int(token)
        if token not in self.where:
            self.where[token] = len(self.table)
            self.table.append(self.tokenizer.decode([token]))
        return self.where[token]

    def literal(self, text: str) -> int:
        """The place of a text that is not a token (a mark that stands for something left out)."""
        key = ("literal", text)
        if key not in self.where:
            self.where[key] = len(self.table)
            self.table.append(text)
        return self.where[key]

    def indices(self, tokens) -> list:
        """The same nesting as `tokens` (a tensor, an array or nested lists of ids), as indices into the table."""
        if hasattr(tokens, "tolist"):
            tokens = tokens.tolist()
        if isinstance(tokens, (list, tuple)):
            return [self.indices(t) for t in tokens]
        return self.index(tokens)


def one_token(tokenizer: Tokenizer, text: str) -> int | None:
    """The id of `text` if it is exactly one token, else None."""
    ids = tokenizer.encode(text, bos=False)
    return ids[0] if len(ids) == 1 else None
