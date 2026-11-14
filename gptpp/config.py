"""Configuration: what the model is (ModelConfig) and how it is trained (TrainConfig).

Both are frozen dataclasses of plain values, so they round-trip through JSON and a checkpoint can rebuild its
model. A preset is named "<recipe>-<size>":

    recipes   gpt2-2019   the published GPT-2 design: learned positions, pre-LayerNorm, GELU, biases, tied
                          embeddings, GPT-2 initialization, AdamW with warmup and cosine decay
              gptpp       the 2026 recipe: rotary positions, parameter-free RMSNorm (also on the embedding),
                          squared-ReLU MLP, QK normalization, no biases, untied embeddings, zero-initialized
                          residual projections, logit soft-cap, Muon for the hidden matrices and AdamW for the rest
    sizes     pico        6 layers, 128 wide, 4 heads, context 256, vocabulary 4,096
              ladder      8 layers, 512 wide, 8 heads, context 512, vocabulary 32,768
              base        12 layers, 768 wide, 12 heads, context 1,024, vocabulary 32,768

    model_cfg = preset("gptpp-ladder")             # a recipe alone ("gptpp") means its base size
    train_cfg = train_preset("gptpp-ladder")
    model_cfg, train_cfg = apply_overrides([model_cfg, train_cfg], ["act=gelu", "lr=0.01"])

docs/DESIGN.md gives the source and the reason of every default.
"""

from __future__ import annotations

import dataclasses
import json
import types
import typing
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, TypeVar

POSITIONS = ("learned", "sinusoidal", "rope", "alibi", "none")
NORMS = ("layernorm", "rmsnorm", "rmsnorm_fixed", "dyt", "none")
PLACEMENTS = ("pre", "post")
ACTIVATIONS = ("relu", "gelu", "silu", "relu2", "swiglu", "geglu")
GATED = ("swiglu", "geglu")
INITS = ("gpt2", "zero_proj")
ATTENTIONS = ("softmax",)
OPTIMIZERS = ("adamw", "muon+adamw", "sgd", "adam", "lion",
              # the recipe with one line changed: another rule for the hidden matrices, the recipe's AdamW for the rest
              "adamw+adamw", "adam+adamw", "lion+adamw", "sgd+adamw")
SCHEDULES = ("cosine", "linear", "constant", "trapezoid")

C = TypeVar("C", bound="_Config")


class _Config:
    """JSON round trip, type checking and overrides shared by the two configurations."""

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)  # type: ignore[call-overload]

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls: type[C], values: dict) -> C:
        """Build from a dictionary (a checkpoint's `model_config`, a run's config.json).

        An unknown key is an error (a typo must not pass silently). A missing key takes today's default with a
        warning, so that a checkpoint written before a field existed still loads and says what was assumed.
        """
        hints = typing.get_type_hints(cls)
        unknown = sorted(set(values) - set(hints))
        if unknown:
            raise ValueError(f"{cls.__name__}: unknown field(s) {unknown}")
        missing = sorted(set(hints) - set(values))
        if missing:
            warnings.warn(f"{cls.__name__}: field(s) {missing} absent, using the current defaults", stacklevel=2)
        cfg = cls(**{name: _check_type(cls.__name__, name, value, hints[name]) for name, value in values.items()})
        cfg.validate()
        return cfg

    @classmethod
    def from_json(cls: type[C], text: str) -> C:
        return cls.from_dict(json.loads(text))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json() + "\n")

    @classmethod
    def load(cls: type[C], path: str | Path) -> C:
        return cls.from_json(Path(path).read_text())

    def replace(self: C, **changes: object) -> C:
        """A copy with some fields changed, type-checked and validated."""
        hints = typing.get_type_hints(type(self))
        for name in changes:
            if name not in hints:
                raise ValueError(f"{type(self).__name__} has no field {name!r}")
        checked = {name: _check_type(type(self).__name__, name, value, hints[name]) for name, value in changes.items()}
        cfg = dataclasses.replace(self, **checked)  # type: ignore[type-var]
        cfg.validate()
        return cfg

    def validate(self) -> None:
        raise NotImplementedError


@dataclass(frozen=True)
class ModelConfig(_Config):
    """The architecture. The defaults are the 2026 recipe at the base size."""

    # ---- size
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 12
    d_model: int = 768
    block_size: int = 1024  # the context length the model is trained at
    n_kv_head: Optional[int] = None  # key/value heads; fewer than n_head = grouped-query attention, 1 = multi-query

    # ---- position
    pos: str = "rope"  # learned (GPT-2) | sinusoidal (2017) | rope (2021) | alibi (2021) | none
    rope_base: float = 10000.0  # rotary wavelengths run from 2 pi to almost 2 pi rope_base positions

    # ---- normalization
    norm: str = "rmsnorm_fixed"  # layernorm (2016) | rmsnorm (2019, learned gain) | rmsnorm_fixed | dyt (2025) | none
    norm_placement: str = "pre"  # pre (GPT-2): inside the residual branch | post (2017): after the addition
    embed_norm: bool = True  # normalize the embedding before the first block
    dyt_alpha: float = 0.5  # initial slope of dynamic tanh

    # ---- the MLP
    act: str = "relu2"  # relu | gelu | silu | relu2 | swiglu | geglu (gated: hidden width scaled by 2/3)
    mlp_mult: float = 4.0  # hidden width over d_model
    use_mlp: bool = True  # False: an attention-only model, for circuit analysis

    # ---- attention
    attn_kind: str = "softmax"
    qk_norm: bool = True  # scores are cosines of queries and keys times a scale
    qk_scale: float = 12.0  # that scale: the scores are qk_scale * cos(q, k)
    qk_scale_learned: bool = False  # each head also learns a multiplier of the scale, initialized at one

    # ---- the rest
    bias: bool = False  # biases in the linear layers and in LayerNorm (GPT-2: True)
    tie_embeddings: bool = False  # the unembedding is the transpose of the embedding (GPT-2: True)
    logit_softcap: float = 15.0  # logits = cap * tanh(logits / cap); 0 = off
    init: str = "zero_proj"  # gpt2 (std 0.02, residual projections scaled by 1/sqrt(2L)) | zero_proj
    dropout: float = 0.0

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_head

    @property
    def kv_heads(self) -> int:
        return self.n_head if self.n_kv_head is None else self.n_kv_head

    @property
    def d_mlp(self) -> int:
        """Hidden width of the MLP. A gated unit has three matrices where a plain one has two, so its width is
        scaled by 2/3 and the parameter count is unchanged (Shazeer, 2020)."""
        width = self.mlp_mult * self.d_model
        return round(width * 2 / 3) if self.act in GATED else round(width)

    def validate(self) -> None:
        def need(condition: bool, message: str) -> None:
            if not condition:
                raise ValueError(f"ModelConfig: {message}")

        for name in ("vocab_size", "n_layer", "n_head", "d_model", "block_size"):
            need(getattr(self, name) >= 1, f"{name} must be positive, not {getattr(self, name)}")
        need(self.d_model % self.n_head == 0, f"d_model {self.d_model} is not divisible by n_head {self.n_head}")
        need(self.kv_heads >= 1 and self.n_head % self.kv_heads == 0,
             f"n_kv_head {self.n_kv_head} must divide n_head {self.n_head}")
        need(self.pos in POSITIONS, f"pos must be one of {POSITIONS}, not {self.pos!r}")
        need(self.pos != "rope" or self.d_head % 2 == 0, f"rotary positions need an even d_head, not {self.d_head}")
        need(self.pos != "sinusoidal" or self.d_model % 2 == 0, "sinusoidal positions need an even d_model")
        need(self.rope_base > 1, f"rope_base must exceed 1, not {self.rope_base}")
        need(self.norm in NORMS, f"norm must be one of {NORMS}, not {self.norm!r}")
        need(self.norm_placement in PLACEMENTS, f"norm_placement must be one of {PLACEMENTS}")
        need(self.dyt_alpha > 0, "dyt_alpha must be positive")
        need(self.act in ACTIVATIONS, f"act must be one of {ACTIVATIONS}, not {self.act!r}")
        need(self.mlp_mult > 0 and self.d_mlp >= 1, f"mlp_mult {self.mlp_mult} gives no hidden units")
        need(self.attn_kind in ATTENTIONS, f"attn_kind must be one of {ATTENTIONS}, not {self.attn_kind!r}")
        need(self.qk_scale > 0, "qk_scale must be positive")
        need(self.logit_softcap >= 0, "logit_softcap must be 0 (off) or positive")
        need(self.init in INITS, f"init must be one of {INITS}, not {self.init!r}")
        need(0 <= self.dropout < 1, f"dropout must be in [0, 1), not {self.dropout}")


@dataclass(frozen=True)
class TrainConfig(_Config):
    """The training run. The defaults are the 2026 recipe at the base size on one card.

    Everything about time is counted in tokens, not steps, so a schedule means the same thing whatever the batch
    size or the number of cards. One step sees device_batch * seq_len * n_gpu * grad_accum tokens.
    """

    # ---- data and budget
    data: str = "fineweb-edu/bpe32k"  # <dataset>/<tokenizer> under $GPTPP_WORK/data, or a directory
    tokens: float = 1e9  # training budget
    seq_len: int = 0  # tokens per sequence; 0 = the model's block_size
    device_batch: int = 64  # sequences per card per forward pass
    grad_accum: int = 1  # forward passes per optimizer step

    # ---- optimizer
    optimizer: str = "muon+adamw"  # adamw | muon+adamw | sgd | adam | lion | <adamw, adam, lion or sgd>+adamw
    lr: float = 0.02  # hidden matrices (with muon+adamw: Muon's rate, in spectral norm per step)
    lr_embedding: Optional[float] = 0.3  # None = lr
    lr_unembedding: Optional[float] = 0.008  # None = lr
    lr_gains: Optional[float] = 0.01  # gains and biases; None = lr
    lr_ref_batch: float = 0.0  # if positive, every rate is multiplied by sqrt(tokens per step / lr_ref_batch)
    weight_decay: float = 0.0  # hidden matrices; gains and biases are never decayed
    wd_embedding: float = 0.0
    wd_unembedding: float = 0.0
    beta1: float = 0.8  # Adam family and Lion
    beta2: float = 0.95
    eps: float = 1e-10
    momentum: float = 0.95  # Muon and SGD
    nesterov: bool = True  # Muon and SGD
    ns_steps: int = 5  # Newton-Schulz iterations of Muon
    grad_clip: float = 0.0  # clip the global gradient norm to this; 0 = off

    # ---- learning-rate schedule (in tokens)
    schedule: str = "trapezoid"  # after warmup: cosine | linear | constant | trapezoid (flat, then linear to the end)
    warmup_tokens: float = 2e6  # linear warmup from zero
    cooldown_frac: float = 0.65  # trapezoid: the final fraction of the budget spent decaying
    final_lr_frac: float = 0.05  # the rate at the end, as a fraction of the peak

    # ---- evaluation and logging
    eval_tokens: float = 10e6  # evaluate on the fixed held-out batches every this many training tokens
    val_tokens: float = 1048576.0  # size of the fixed held-out set
    log_every: int = 10  # steps between training-loss records
    sample_tokens: int = 64  # length of the sample logged at each evaluation (if a tokenizer is found); 0 = off
    seed: int = 0
    compile: bool = True  # torch.compile the model

    def validate(self) -> None:
        def need(condition: bool, message: str) -> None:
            if not condition:
                raise ValueError(f"TrainConfig: {message}")

        need(self.tokens > 0, "tokens must be positive")
        need(self.seq_len >= 0, "seq_len must be 0 (the model's block_size) or positive")
        need(self.device_batch >= 1 and self.grad_accum >= 1, "device_batch and grad_accum must be at least 1")
        need(self.optimizer in OPTIMIZERS, f"optimizer must be one of {OPTIMIZERS}, not {self.optimizer!r}")
        for name in ("lr", "lr_embedding", "lr_unembedding", "lr_gains"):
            value = getattr(self, name)
            need(value is None or value > 0, f"{name} must be positive, not {value}")
        for name in ("lr_ref_batch", "weight_decay", "wd_embedding", "wd_unembedding", "grad_clip", "warmup_tokens"):
            need(getattr(self, name) >= 0, f"{name} must not be negative")
        need(0 <= self.beta1 < 1 and 0 <= self.beta2 < 1, "beta1 and beta2 must be in [0, 1)")
        need(self.eps > 0, "eps must be positive")
        need(0 <= self.momentum < 1, "momentum must be in [0, 1)")
        need(self.ns_steps >= 1, "ns_steps must be at least 1")
        need(self.schedule in SCHEDULES, f"schedule must be one of {SCHEDULES}, not {self.schedule!r}")
        need(0 < self.cooldown_frac <= 1, "cooldown_frac must be in (0, 1]")
        need(0 <= self.final_lr_frac <= 1, "final_lr_frac must be in [0, 1]")
        need(self.eval_tokens > 0 and self.val_tokens > 0, "eval_tokens and val_tokens must be positive")
        need(self.log_every >= 1, "log_every must be at least 1")
        need(self.sample_tokens >= 0, "sample_tokens must not be negative")


# ----------------------------------------------------------------------------------------------------------------
# Presets

SIZES: dict[str, dict] = {
    "pico": dict(n_layer=6, d_model=128, n_head=4, block_size=256, vocab_size=4096),
    "ladder": dict(n_layer=8, d_model=512, n_head=8, block_size=512, vocab_size=32768),
    "base": dict(n_layer=12, d_model=768, n_head=12, block_size=1024, vocab_size=32768),
}

# The 2026 recipe is the set of defaults of ModelConfig; the 2019 recipe is written as its differences from it.
RECIPES: dict[str, dict] = {
    "gptpp": dict(),
    "gpt2-2019": dict(pos="learned", norm="layernorm", norm_placement="pre", embed_norm=False, act="gelu",
                      qk_norm=False, bias=True, tie_embeddings=True, logit_softcap=0.0, init="gpt2"),
}

# One card sees 65,536 tokens a step at every size.
_BATCH: dict[str, dict] = {
    "pico": dict(device_batch=256, tokens=100e6, eval_tokens=5e6),
    "ladder": dict(device_batch=128, tokens=150e6, eval_tokens=10e6),
    "base": dict(device_batch=64, tokens=1e9, eval_tokens=20e6),
}


def _split(name: str) -> tuple[str, str]:
    """"gptpp-ladder" -> ("gptpp", "ladder"); a recipe alone means the base size."""
    if name in RECIPES:
        return name, "base"
    recipe, _, size = name.rpartition("-")
    if recipe not in RECIPES or size not in SIZES:
        raise ValueError(f"unknown preset {name!r}; choose from {preset_names()}")
    return recipe, size


def preset_names() -> list[str]:
    return [f"{recipe}-{size}" for recipe in RECIPES for size in SIZES]


def preset(name: str) -> ModelConfig:
    """The model of a preset, e.g. preset("gptpp-base") or preset("gpt2-2019-ladder")."""
    recipe, size = _split(name)
    return ModelConfig().replace(**SIZES[size], **RECIPES[recipe])


def train_preset(name: str) -> TrainConfig:
    """The training recipe that goes with a preset's model."""
    recipe, size = _split(name)
    width = SIZES[size]["d_model"]
    if recipe == "gptpp":
        # Muon on the hidden matrices at the usual 0.02. The AdamW rates of the embedding and the unembedding
        # are nanochat's values at width 768 (0.3 and 0.008) scaled by sqrt(768 / width), as nanochat scales them.
        scale = (768 / width) ** 0.5
        recipe_values = dict(lr_embedding=round(0.3 * scale, 4), lr_unembedding=round(0.008 * scale, 5))
    else:
        # AdamW on everything: one rate, weight decay 0.1 on every matrix (embeddings included, as in GPT and
        # nanoGPT) and none on gains and biases, betas (0.9, 0.95), warmup then cosine decay to a tenth, clipping.
        # The rate at the base size is nanoGPT's for GPT-2 small. At the two smaller sizes it is the best of a
        # sweep on short runs (docs/DESIGN.md, Section 5.3): 6e-4 there would handicap the 2019 recipe.
        rate = {"pico": 6e-3, "ladder": 2.4e-3, "base": 6e-4}[size]
        warmup = {"pico": 4e6, "ladder": 8e6, "base": 32e6}[size]
        recipe_values = dict(optimizer="adamw", lr=rate, lr_embedding=None, lr_unembedding=None, lr_gains=None,
                             weight_decay=0.1, wd_embedding=0.1, wd_unembedding=0.1, beta1=0.9, beta2=0.95,
                             eps=1e-8, grad_clip=1.0, schedule="cosine", warmup_tokens=warmup, final_lr_frac=0.1)
    return TrainConfig().replace(**_BATCH[size], **recipe_values)


# ----------------------------------------------------------------------------------------------------------------
# Type checking and "--set key=value" overrides


def _unwrap_optional(kind: object) -> tuple[object, bool]:
    """Optional[int] -> (int, True); int -> (int, False)."""
    if typing.get_origin(kind) in (typing.Union, types.UnionType):
        inner = [arg for arg in typing.get_args(kind) if arg is not type(None)]
        return inner[0], True
    return kind, False


def _check_type(owner: str, name: str, value: object, kind: object) -> object:
    """Return `value` as the declared type of the field, or raise. An int is accepted where a float is declared
    (JSON does not tell them apart); a bool is never accepted as a number."""
    kind, optional = _unwrap_optional(kind)
    if value is None:
        if optional:
            return None
        raise ValueError(f"{owner}.{name} must be {kind.__name__}, not None")  # type: ignore[attr-defined]
    if kind is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if kind is int and isinstance(value, float) and value.is_integer():
        return int(value)
    if not isinstance(value, kind) or (kind is not bool and isinstance(value, bool)):  # type: ignore[arg-type]
        raise ValueError(f"{owner}.{name} must be {kind.__name__}, not {value!r}")  # type: ignore[attr-defined]
    return value


def _parse(owner: str, name: str, text: str, kind: object) -> object:
    """Parse the text of a command-line value as the declared type of the field."""
    kind, optional = _unwrap_optional(kind)
    if optional and text.lower() in ("none", "null"):
        return None
    try:
        if kind is bool:
            lowered = text.lower()
            if lowered in ("true", "1", "yes", "on"):
                return True
            if lowered in ("false", "0", "no", "off"):
                return False
            raise ValueError
        if kind is int:
            return int(text)
        if kind is float:
            return float(text)
    except ValueError:
        raise ValueError(f"{owner}.{name} expects {kind.__name__}, got {text!r}") from None  # type: ignore[attr-defined]
    return text


def apply_overrides(configs: Sequence[_Config], overrides: Sequence[str]) -> list:
    """Apply "key=value" strings to the configuration that owns each key, with type checking.

    A key is a field name ("act=gelu", "lr=0.01"), optionally prefixed by "model." or "train.". The values are
    parsed as the declared type of the field: int, float, bool (true/false), str, or "none" where a field is
    optional. An unknown key, a value of the wrong type or an invalid combination raises ValueError.
    """
    prefixes = {"model": ModelConfig, "train": TrainConfig}
    changes: list[dict] = [{} for _ in configs]
    for item in overrides:
        key, equals, text = item.partition("=")
        if not equals or not key:
            raise ValueError(f"override {item!r} is not of the form key=value")
        prefix, dot, field = key.rpartition(".")
        if dot and prefix not in prefixes:
            raise ValueError(f"override {item!r}: the prefix must be 'model.' or 'train.'")
        owners = [i for i, cfg in enumerate(configs)
                  if field in typing.get_type_hints(type(cfg)) and (not dot or isinstance(cfg, prefixes[prefix]))]
        if not owners:
            raise ValueError(f"override {item!r}: no such field")
        if len(owners) > 1:
            raise ValueError(f"override {item!r}: ambiguous, prefix it with 'model.' or 'train.'")
        owner = type(configs[owners[0]])
        changes[owners[0]][field] = _parse(owner.__name__, field, text, typing.get_type_hints(owner)[field])
    return [cfg.replace(**change) for cfg, change in zip(configs, changes)]
