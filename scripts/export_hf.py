"""Write a trained run of gpt-pp in the format of Hugging Face transformers, and check it against gpt-pp.

    python scripts/export_hf.py --run gptpp-base --out DIR [--chat] [--no-check]

The 2026 recipe of gpt-pp is the architecture that transformers calls NanoChat (`NanoChatForCausalLM`) except in
two points, and the export deals with each:

1. Rotary positions. gpt-pp turns each pair (i, i + d_head/2) of a query or a key by +angle (gptpp/model.py,
   `rotate`); NanoChat turns it by -angle. Negating the second half of every head's rows of W_Q and W_K makes the
   two the same function: the turn by -angle of a vector whose second half is negated is the turn by +angle of the
   vector, with its second half negated, and negating the same coordinates of a query and a key leaves their dot
   product unchanged. The QK normalization, which has no gain per coordinate, is unchanged by the negation, and so
   is its order with the turn (a turn keeps lengths).
2. The query gain. gpt-pp multiplies each normalized query by qk_gain = qk_scale / sqrt(d_head), so that a score
   is qk_scale times the cosine of query and key (gptpp/model.py, `CausalSelfAttention.position`). NanoChat has no
   gain, and no change of the weights can stand for one, because it acts after a normalization. The exported model
   therefore carries a few lines of code of its own (gptpp/hf/), a subclass of NanoChat that multiplies after the
   normalization, and is loaded with `trust_remote_code=True`.

Everything else is the same layer for layer: the embedding and its normalization, pre-norm blocks with RMS norms
without gains (eps 1e-6), squared ReLU, the untied unembedding, the soft cap. Other configurations of gpt-pp
(learned or sinusoidal positions, ALiBi, LayerNorm, norms with gains, biases, post-norm, gated MLPs, tied tables,
learned QK gains, attention without the MLP) are refused.

What is written in DIR:

    model.safetensors          every weight, in float32, as trained (W_Q and W_K with the negation of point 1)
    config.json                model_type "gptpp", with `auto_map` to the two files below
    configuration_gptpp.py     copies of gptpp/hf/
    modeling_gptpp.py
    generation_config.json     the special tokens that end a text, sampling at temperature 0.8 as in the series
    tokenizer.json             gpt-pp's tokenizer, with its special tokens registered and <|bos|> put before a text
    tokenizer_config.json      the special tokens; with --chat, the chat template of gptpp/chat.py

The check (skip it with --no-check) loads DIR as a user would, with `AutoModelForCausalLM` and `AutoTokenizer`,
and compares with gpt-pp in float32 on the CPU: the logits of a few texts, the tokens of a greedy continuation, the
tokenizer's ids, and with --chat the ids of a conversation rendered by the template against gptpp/chat.py.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parent.parent
sys.path[:] = [str(REPO)] + [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from gptpp import work_dir  # noqa: E402
from gptpp.config import ModelConfig  # noqa: E402
from gptpp.model import GPT  # noqa: E402

HF_CODE = REPO / "gptpp" / "hf"
SPECIAL = ("<|bos|>", "<|user_start|>", "<|user_end|>", "<|assistant_start|>", "<|assistant_end|>")

# gptpp/chat.py as a template: <|bos|>, then the turns, the user first; a leading system message is joined to the
# first user message by a blank line; with add_generation_prompt, <|assistant_start|> after the last user message.
CHAT_TEMPLATE = (
    "{{- bos_token -}}"
    "{%- set ns = namespace(system=none, offset=0) -%}"
    "{%- if messages and messages[0]['role'] == 'system' -%}"
    "{%- set ns.system = messages[0]['content'] -%}{%- set ns.offset = 1 -%}"
    "{%- endif -%}"
    "{%- for message in messages[ns.offset:] -%}"
    "{%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}"
    "{{- raise_exception('the turns alternate, the user first') -}}"
    "{%- endif -%}"
    "{%- set content = message['content'] -%}"
    "{%- if loop.first and ns.system is not none -%}{%- set content = ns.system + '\\n\\n' + content -%}{%- endif -%}"
    "{%- if message['role'] == 'user' -%}{{- '<|user_start|>' + content + '<|user_end|>' -}}"
    "{%- else -%}{{- '<|assistant_start|>' + content + '<|assistant_end|>' -}}{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- '<|assistant_start|>' -}}{%- endif -%}"
)


def refusal(cfg: ModelConfig) -> Optional[str]:
    """Why a configuration cannot be written as a NanoChat model with a query gain, or None if it can."""
    wanted = {"pos": "rope", "norm": "rmsnorm_fixed", "norm_placement": "pre", "act": "relu2", "qk_norm": True,
              "qk_scale_learned": False, "embed_norm": True, "tie_embeddings": False, "bias": False, "use_mlp": True}
    for key, value in wanted.items():
        if getattr(cfg, key) != value:
            return f"{key}={getattr(cfg, key)!r}: the export knows the 2026 recipe only ({key}={value!r})"
    if not cfg.logit_softcap > 0:
        return "logit_softcap=0: the export expects the recipe's soft cap"
    if cfg.n_head * cfg.d_head != cfg.d_model or cfg.d_head % 2:
        return "the heads must split d_model evenly into heads of even width"
    return None


def negate_second_halves(weight: torch.Tensor, heads: int, d_head: int) -> torch.Tensor:
    """W (heads * d_head, d) with rows d_head/2 .. d_head-1 of every head negated (point 1 of the docstring)."""
    w = weight.detach().float().clone().view(heads, d_head, -1)
    w[:, d_head // 2:] *= -1
    return w.view(heads * d_head, -1).contiguous()


def hf_state_dict(model: GPT) -> dict[str, torch.Tensor]:
    """gpt-pp's parameters under NanoChat's names, in float32."""
    cfg = model.cfg
    sd = {"model.embed_tokens.weight": model.embed.weight, "lm_head.weight": model.unembed.weight}
    for i, block in enumerate(model.blocks):
        attn, mlp, p = block.attn, block.mlp, f"model.layers.{i}."
        sd[p + "self_attn.q_proj.weight"] = negate_second_halves(attn.q.weight, cfg.n_head, cfg.d_head)
        sd[p + "self_attn.k_proj.weight"] = negate_second_halves(attn.k.weight, cfg.kv_heads, cfg.d_head)
        sd[p + "self_attn.v_proj.weight"] = attn.v.weight
        sd[p + "self_attn.o_proj.weight"] = attn.o.weight
        sd[p + "mlp.fc1.weight"] = mlp.fc.weight
        sd[p + "mlp.fc2.weight"] = mlp.proj.weight
    return {k: v.detach().float().contiguous().clone() for k, v in sd.items()}


def hf_config(cfg: ModelConfig, special: Optional[dict[str, int]] = None, chat: bool = False):
    """The GptPPConfig of a gpt-pp configuration (gptpp/hf/configuration_gptpp.py)."""
    sys.path.insert(0, str(HF_CODE.parent.parent))
    from gptpp.hf.configuration_gptpp import GptPPConfig

    bos = special["<|bos|>"] if special else 0
    eos = special["<|assistant_end|>"] if (special and chat) else bos
    config = GptPPConfig(
        vocab_size=cfg.vocab_size, hidden_size=cfg.d_model, intermediate_size=cfg.d_mlp,
        num_hidden_layers=cfg.n_layer, num_attention_heads=cfg.n_head, num_key_value_heads=cfg.kv_heads,
        max_position_embeddings=cfg.block_size, hidden_act="relu2", rms_norm_eps=1e-6,
        rope_parameters={"rope_type": "default", "rope_theta": float(cfg.rope_base)},
        final_logit_softcapping=float(cfg.logit_softcap), attention_bias=False, tie_word_embeddings=False,
        bos_token_id=bos, eos_token_id=eos, pad_token_id=bos,
        qk_gain=float(cfg.qk_scale) / cfg.d_head ** 0.5,
    )
    config.architectures = ["GptPPForCausalLM"]
    config.auto_map = {"AutoConfig": "configuration_gptpp.GptPPConfig",
                       "AutoModel": "modeling_gptpp.GptPPModel",
                       "AutoModelForCausalLM": "modeling_gptpp.GptPPForCausalLM"}
    config.dtype = "float32"
    return config


def write_model(model: GPT, out: Path, special: Optional[dict[str, int]] = None, chat: bool = False) -> None:
    """The weights, the configuration and the model code (no tokenizer)."""
    why = refusal(model.cfg)
    if why:
        raise ValueError(why)
    out.mkdir(parents=True, exist_ok=True)
    save_file(hf_state_dict(model), str(out / "model.safetensors"), metadata={"format": "pt"})
    hf_config(model.cfg, special, chat).save_pretrained(out)
    for name in ("configuration_gptpp.py", "modeling_gptpp.py"):
        shutil.copyfile(HF_CODE / name, out / name)


def write_tokenizer(tokenizer_dir: Path, out: Path, block_size: int, chat: bool) -> dict[str, int]:
    """gpt-pp's tokenizer.json with its special tokens registered (so that the chat template's markers are read
    as the markers) and <|bos|> put before every text, as gpt-pp's documents begin; returns the special ids."""
    from tokenizers import AddedToken, Tokenizer
    from tokenizers.processors import TemplateProcessing
    from transformers import GenerationConfig, PreTrainedTokenizerFast

    meta = json.loads((tokenizer_dir / "meta.json").read_text())
    special = meta["special_tokens"]
    tk = Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))
    tk.add_special_tokens([AddedToken(t, special=True, normalized=False) for t in SPECIAL])
    for t in SPECIAL:
        assert tk.token_to_id(t) == special[t], f"{t} moved: {tk.token_to_id(t)} against {special[t]}"
    bos = special["<|bos|>"]
    tk.post_processor = TemplateProcessing(single="<|bos|> $A", pair="<|bos|> $A <|bos|> $B",
                                           special_tokens=[("<|bos|>", bos)])
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tk, bos_token="<|bos|>", eos_token="<|assistant_end|>" if chat else "<|bos|>",
        pad_token="<|bos|>", model_max_length=block_size, clean_up_tokenization_spaces=False)
    fast.add_special_tokens({"additional_special_tokens": list(SPECIAL[1:])})
    if chat:
        fast.chat_template = CHAT_TEMPLATE
    fast.save_pretrained(out)
    # A text ends where the model writes <|bos|> (a new document); an answer ends at any special token, as
    # gptpp/chat.py's `answer` stops.
    eos = [special[t] for t in ("<|assistant_end|>", "<|bos|>", "<|user_start|>", "<|user_end|>", "<|assistant_start|>")] if chat else [bos]
    GenerationConfig(bos_token_id=bos, eos_token_id=eos, pad_token_id=bos, do_sample=True, temperature=0.8,
                     max_length=block_size).save_pretrained(out)
    return special


TEXTS = [
    "The mitochondrion is the organelle in which most of a cell's energy is produced. It has two membranes,",
    "In 1905 Einstein published four papers. The first explained the photoelectric effect by assuming that light",
    "Once upon a time, there was a little girl named Lucy. She had a black cat named Shadow.",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n",
    "Les élèves ont étudié la photosynthèse, et 3,14159 est une approximation de π. 日本語も少し。",
]


def check(model: GPT, out: Path, tokenizer_dir: Optional[Path], chat: bool) -> dict:
    """Load `out` as a user would and compare it with gpt-pp; returns the measurements, raises if they disagree."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from gptpp.generate import generate
    from gptpp.tokenizer import Tokenizer

    torch.backends.cuda.matmul.allow_tf32 = False
    hf = AutoModelForCausalLM.from_pretrained(out, trust_remote_code=True, dtype=torch.float32).eval()
    tok = AutoTokenizer.from_pretrained(out, trust_remote_code=True)
    ours = Tokenizer.load(tokenizer_dir)
    report: dict = {"logits_max_abs_diff": 0.0, "texts": 0}
    with torch.no_grad():
        for text in TEXTS:
            ids = tok(text)["input_ids"]
            want = [ours.bos_id] + list(ours.encode(text))
            assert ids == want, f"the tokenizers differ on {text[:30]!r}: {ids[:12]} against {want[:12]}"
            idx = torch.tensor([ids[: model.cfg.block_size]])
            a, b = model(idx), hf(idx).logits
            report["logits_max_abs_diff"] = max(report["logits_max_abs_diff"], (a - b).abs().max().item())
            report["texts"] += 1
        idx = torch.tensor([tok(TEXTS[0])["input_ids"]])
        g1 = generate(model, idx, 48, temperature=0.0)[0, idx.size(1):].tolist()
        g2 = hf.generate(idx, attention_mask=torch.ones_like(idx), max_new_tokens=48, do_sample=False,
                         eos_token_id=None, use_cache=True)[0, idx.size(1):].tolist()
        report["greedy_tokens_equal"] = sum(1 for _ in __import__("itertools").takewhile(lambda p: p[0] == p[1], zip(g1, g2)))
        report["greedy_tokens"] = len(g1)
        report["greedy_text"] = tok.decode(g2)
    if chat:
        from gptpp.chat import prompt_ids

        convo = [{"role": "system", "content": "Answer briefly."}, {"role": "user", "content": "What is a prime number?"},
                 {"role": "assistant", "content": "A number with exactly two divisors."}, {"role": "user", "content": "Name one."}]
        a = tok.apply_chat_template(convo, add_generation_prompt=True, tokenize=True)
        a = a["input_ids"] if isinstance(a, dict) or hasattr(a, "keys") else a
        b = prompt_ids(ours, convo)
        assert list(a) == list(b), f"the chat template and gptpp/chat.py differ: {list(a)[:16]} against {b[:16]}"
        report["chat_template_equals_gptpp_chat"] = True
    if report["logits_max_abs_diff"] > 1e-3:
        raise AssertionError(f"logits differ by {report['logits_max_abs_diff']:.2e}")
    return report


def main(argv: Optional[list[str]] = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True, help="run name under $GPTPP_WORK/ckpt, or a run directory")
    ap.add_argument("--ckpt", default="model.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chat", action="store_true", help="the run is a chat stage: chat template, stop at its markers")
    ap.add_argument("--no-check", action="store_true")
    a = ap.parse_args(argv)
    run = Path(a.run) if Path(a.run).is_dir() else work_dir() / "ckpt" / a.run
    model, cfg = GPT.from_checkpoint(run / a.ckpt, device="cpu")
    model.float().eval()
    name = json.loads((run / "config.json").read_text())["data_meta"]["tokenizer"]
    tokenizer_dir = work_dir() / "data" / "tokenizers" / name
    out = Path(a.out)
    special = write_tokenizer(tokenizer_dir, out, cfg.block_size, a.chat)
    write_model(model, out, special, a.chat)
    report = {"run": str(run), "tokenizer": name, "out": str(out)}
    if not a.no_check:
        report.update(check(model, out, tokenizer_dir, a.chat))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
