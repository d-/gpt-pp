"""The model that runs in the reader's browser: a checkpoint, its tokenizer and a test vector, written as the
files the site's JavaScript transformer reads.

    python -m gptpp.export.browser_model --ckpt <run dir> --name pico       # the weights, the configuration, a test vector
    python -m gptpp.export.browser_model --tokenizer bpe4k                  # the tokenizer for the page, with test strings
    python -m gptpp.export.browser_model --fixtures <dir>                   # tiny random models of other configurations,
                                                                            # for the tests of the JavaScript model
    python -m gptpp.export.browser_model --stories                          # pico-stories.json, for the opening figure

A model named <name> becomes four files in the site's data folder (--out, default assets/data/gpt of the site):

    <name>.json        the lab's ModelConfig, the sizes, the form of the weights and why it was chosen, and the
                       layout of <name>.bin: one entry a tensor, {name, shape, dtype, offset, count, scales}
    <name>.bin         the weights. Matrices as 8-bit integers with one float32 scale a row (form `int8-row`) or a
                       tensor (`int8-tensor`), or everything as float16 or float32; vectors (gains, biases) as float32
    <name>-test.json   a token sequence and the layout of <name>-test.bin; no page loads these two files
    <name>-test.bin    float32, computed by PyTorch on the weights exactly as the page decodes them: the logits at
                       every position, and a sample of the activations the page's model can tap

The form is decided by measurement. The held-out loss (the trainer's fixed batches, float32 arithmetic) is taken for
the float32 weights and for each smaller form; the smallest form whose loss is within 0.005 nats of float32 is
shipped. `page_model(name)` gives back the model with the shipped weights, so that any other exporter can compute
on exactly the network the page runs.

Any ModelConfig is exported: the layout names every parameter, and the JavaScript model is driven by the
configuration. The fixtures are small random models that together take every value of every field of the
configuration, with a test vector each; the JavaScript tests run on them.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from ..config import ModelConfig
from ..model import GPT
from .common import DATA, SITE, WORK, Blob, write_atomic, write_json

FORMAT = 1
FORMS = ("int8-tensor", "int8-row", "float16", "float32")  # from the smallest file to the largest
TOLERANCE = 0.005  # nats: the loss a smaller form may cost
TEST_TOKENS = 16  # the test sequence: every logit at every position is stored
BITS = 8

# The split pattern of the tokenizer (gptpp/bpe.py, GPT4_PATTERN) as JavaScript reads it, from docs/DATA.md,
# "A port to JavaScript". The Python pattern uses a modifier group, (?i:...), that JavaScript engines refuse,
# and \s, which in JavaScript is another set of characters (it has U+FEFF and lacks U+0085). With the `u` flag
# this spelling cuts like the Python one.
JS_WHITESPACE = r"\t\n\v\f\r \x85\xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000"
JS_PATTERN = (
    r"'(?:[sdmtSDMT\u017f]|[lL][lL]|[vV][eE]|[rR][eE])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^WS\p{L}\p{N}]+[\r\n]*"
    r"|[WS]*[\r\n]|[WS]+(?![^WS])|[WS]"
).replace("WS", JS_WHITESPACE)
JS_FLAGS = "gu"


# ----------------------------------------------------------------------------------------------------------------
# The weights: which tensors, in which form


def tensors(model: GPT) -> dict[str, np.ndarray]:
    """The parameters to ship, by their names in the state dict, as float32. A tied unembedding is left out: it is
    the embedding. The arrays are copies: on the CPU `.numpy()` is a view of the parameter, and loading another
    form into the model would otherwise overwrite the weights every later form is derived from."""
    out: dict[str, np.ndarray] = {}
    for name, value in model.state_dict().items():
        if name == "unembed.weight" and model.cfg.tie_embeddings:
            continue
        out[name] = value.detach().float().cpu().numpy().copy()
    return out


def quantize(w: np.ndarray, per_row: bool, bits: int = BITS) -> tuple[np.ndarray, np.ndarray]:
    """A matrix as signed integers of `bits` bits and float32 scales, one a row or one for the whole matrix:
    w is about q * scale, with the largest magnitude of the row (or of the matrix) at the largest integer."""
    top = 2 ** (bits - 1) - 1
    peak = np.abs(w).max(axis=1) if per_row else np.abs(w).max().reshape(1)
    scale = (peak / top).astype(np.float32)
    scale[scale == 0] = 1.0
    q = np.clip(np.rint(w / scale.reshape(-1, 1)), -top, top).astype(np.int8)
    return q, scale


def dequantize(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """What the page computes with: the float32 product of an integer and its scale."""
    return q.astype(np.float32) * scale.reshape(-1, 1)


def shipped(arrays: dict[str, np.ndarray], form: str, bits: int = BITS) -> dict[str, np.ndarray]:
    """The tensors as the page decodes them from a file of the given form (float32 again, with the form's error)."""
    if form not in FORMS:
        raise ValueError(f"form must be one of {FORMS}, not {form!r}")
    out = {}
    for name, w in arrays.items():
        if form == "float32":
            out[name] = w
        elif form == "float16":
            out[name] = w.astype(np.float16).astype(np.float32)
        elif w.ndim == 2:
            out[name] = dequantize(*quantize(w, per_row=form == "int8-row", bits=bits))
        else:
            out[name] = w
    return out


def form_bytes(arrays: dict[str, np.ndarray], form: str) -> int:
    """The size of the weights file in a form (without padding)."""
    total = 0
    for w in arrays.values():
        if form == "float32":
            total += 4 * w.size
        elif form == "float16":
            total += 2 * w.size
        elif w.ndim == 2:
            total += w.size + 4 * (w.shape[0] if form == "int8-row" else 1)
        else:
            total += 4 * w.size
    return total


def load_arrays(model: GPT, arrays: dict[str, np.ndarray]) -> GPT:
    """Put tensors (as `tensors` names them) into a model."""
    state = model.state_dict()
    for name, w in arrays.items():
        state[name] = torch.from_numpy(np.ascontiguousarray(w)).to(state[name].device)
    if model.cfg.tie_embeddings:
        state["unembed.weight"] = state["embed.weight"]
    model.load_state_dict(state)
    return model


def pack(name: str, arrays: dict[str, np.ndarray], form: str, out: Path, entries: bool = True) -> tuple[list, int]:
    """Write <name>.bin and return its layout and its size. With `entries`, also <name>.entries.json for the
    site's manifest: one entry for the 8-bit block and one for the floats, which together tile the file."""
    blob = Blob(name, out)
    layout = []
    if form in ("float32", "float16"):
        dtype, key = (np.float32, "f32") if form == "float32" else (np.float16, "f16")
        flat, at = [], 0
        for tensor, w in arrays.items():
            layout.append({"name": tensor, "shape": list(w.shape), "dtype": key, "offset": at * np.dtype(dtype).itemsize,
                           "count": int(w.size)})
            flat.append(w.astype(dtype).ravel())
            at += w.size
        entry = blob.put(f"{name}.{form}", np.concatenate(flat), f"the weights of {name}, {form}; layout in {name}.json")
        for item in layout:
            item["offset"] += entry["offset"]
    else:
        ints, floats, n_int, n_float = [], [], 0, 0
        for tensor, w in arrays.items():
            item = {"name": tensor, "shape": list(w.shape), "count": int(w.size)}
            if w.ndim == 2:
                q, scale = quantize(w, per_row=form == "int8-row")
                item.update(dtype="i8", offset=n_int, scales={"offset": 4 * n_float, "count": int(scale.size)})
                ints.append(q.ravel())
                floats.append(scale)
                n_int += q.size
                n_float += scale.size
            else:
                item.update(dtype="f32", offset=4 * n_float)
                floats.append(w.astype(np.float32).ravel())
                n_float += w.size
            layout.append(item)
        first = blob.put(f"{name}.int8", np.concatenate(ints).view(np.uint8),
                         f"the matrices of {name} as signed 8-bit integers; layout and scales in {name}.json")
        second = blob.put(f"{name}.float32", np.concatenate(floats).astype(np.float32),
                          f"the scales of the 8-bit matrices of {name}, and its vectors; layout in {name}.json")
        for item in layout:
            if item["dtype"] == "i8":
                item["offset"] += first["offset"]
                item["scales"]["offset"] += second["offset"]
            else:
                item["offset"] += second["offset"]
    if entries:
        size = blob.save()
    else:
        out.mkdir(parents=True, exist_ok=True)
        write_atomic(out / f"{name}.bin", bytes(blob.buf))
        size = len(blob.buf)
    return layout, size


def unpack(meta: dict, raw: bytes) -> dict[str, np.ndarray]:
    """The tensors of a weights file, decoded as the page decodes them."""
    out = {}
    for item in meta["layout"]:
        count, shape = item["count"], item["shape"]
        if item["dtype"] == "i8":
            q = np.frombuffer(raw, dtype=np.int8, count=count, offset=item["offset"]).reshape(shape)
            scale = np.frombuffer(raw, dtype="<f4", count=item["scales"]["count"], offset=item["scales"]["offset"])
            out[item["name"]] = dequantize(q, scale)
        else:
            dtype = "<f4" if item["dtype"] == "f32" else "<f2"
            out[item["name"]] = np.frombuffer(raw, dtype=dtype, count=count, offset=item["offset"]).astype(np.float32).reshape(shape)
    return out


def page_model(name: str, out: Path | str = DATA, device: str = "cpu") -> tuple[GPT, ModelConfig]:
    """The model exactly as the page runs it: the configuration of <name>.json with the weights decoded from
    <name>.bin. Exporters of figures that are compared with the in-page model compute on this one."""
    out = Path(out)
    meta = json.loads((out / f"{name}.json").read_text())
    cfg = ModelConfig.from_dict(meta["config"])
    model = load_arrays(GPT(cfg), unpack(meta, (out / f"{name}.bin").read_bytes()))
    return model.to(device).eval(), cfg


# ----------------------------------------------------------------------------------------------------------------
# The held-out loss of each form


@torch.no_grad()
def heldout_loss(model: GPT, batches: Sequence[tuple[torch.Tensor, torch.Tensor, int]]) -> float:
    """Mean cross-entropy in nats over the trainer's fixed held-out windows, in float32 arithmetic."""
    total, count = 0.0, 0
    for x, y, n_targets in batches:
        total += model(x, y).double().item() * n_targets
        count += n_targets
    return total / count


def measure_forms(model: GPT, arrays: dict[str, np.ndarray], batches, forms: Sequence[str] = FORMS) -> dict:
    """The held-out loss and the file size of every form; the model is left holding the float32 weights."""
    losses, sizes = {}, {}
    for form in forms:
        load_arrays(model, shipped(arrays, form))
        losses[form] = heldout_loss(model, batches)
        sizes[form] = form_bytes(arrays, form)
    load_arrays(model, arrays)
    return {"loss": losses, "bytes": sizes}


def choose_form(measured: dict, tolerance: float = TOLERANCE) -> str:
    """The smallest form whose held-out loss is within `tolerance` nats of the float32 weights."""
    reference = measured["loss"]["float32"]
    passing = [form for form in measured["loss"] if measured["loss"][form] - reference <= tolerance]
    return min(passing, key=lambda form: measured["bytes"][form])


# ----------------------------------------------------------------------------------------------------------------
# The test vector


@torch.no_grad()
def test_vector(model: GPT, tokens: Sequence[int], long_tokens: Optional[Sequence[int]] = None) -> tuple[dict, dict]:
    """What PyTorch computes on a token sequence: the logits at every position and a sample of the activations
    that the page's model can tap. Returns (description, {name: float32 array}).

    The sample: every attention pattern; the residual stream after the embedding and after every block; for one
    layer the attention's write, the MLP's write and its hidden activations; for one head of that layer its
    queries and keys as the scores use them (normalized and rotated), its values, scores, mixed values and what
    it writes into the stream; the logit lens (final normalization and unembedding of the stream) at two depths
    for the last position; and, for a sequence as long as the context, the logits of its last position.
    """
    cfg = model.cfg
    L, H = cfg.n_layer, cfg.n_head
    idx = torch.tensor([list(tokens)], dtype=torch.long)
    T = idx.size(1)
    logits, cache = model.run_with_cache(idx, names=lambda name: True)
    layer, head = L // 2, min(1, H - 1)
    kv_head = head // (H // cfg.kv_heads)
    attn = model.blocks[layer].attn
    q, k = attn.position(cache[f"attn.{layer}.q"], cache[f"attn.{layer}.k"], model.rotation(0, T))
    resid = [cache["embed"][0]] + [cache[f"resid_post.{l}"][0] for l in range(L)]
    depths = sorted({0, L // 2})
    arrays = {
        "logits": logits[0],
        "patterns": torch.stack([cache[f"attn.{l}.pattern"][0] for l in range(L)]),
        "resid": torch.stack(resid),
        "attnOut": cache[f"attn_out.{layer}"][0],
        "head.q": q[0, head],
        "head.k": k[0, kv_head],
        "head.v": cache[f"attn.{layer}.v"][0, kv_head],
        "head.scores": cache[f"attn.{layer}.scores"][0, head],
        "head.z": cache[f"attn.{layer}.z"][0, head],
        "head.result": cache[f"attn.{layer}.result"][0, head],
        "lens": torch.stack([model.unembed_logits(model.final_norm(resid[depth][-1])) for depth in depths]),
    }
    if cfg.use_mlp:
        arrays["mlpOut"] = cache[f"mlp_out.{layer}"][0]
        arrays["neurons"] = cache[f"mlp.{layer}.post"][0]
    describe = {"tokens": [int(t) for t in tokens], "layer": layer, "head": head, "lensDepths": depths, "lensPosition": T - 1}
    if long_tokens is not None:
        arrays["longLogits"] = model(torch.tensor([list(long_tokens)], dtype=torch.long))[0, -1]
        describe["long"] = [int(t) for t in long_tokens]
    return describe, {name: value.float().cpu().numpy() for name, value in arrays.items()}


def write_test(name: str, describe: dict, arrays: dict[str, np.ndarray], out: Path, entries: bool = True) -> int:
    """<name>-test.bin (float32) and <name>-test.json (the tokens and where each array lies)."""
    listing, flat, at = [], [], 0
    for key, value in arrays.items():
        listing.append({"name": key, "shape": list(value.shape), "offset": 4 * at, "count": int(value.size)})
        flat.append(value.astype(np.float32).ravel())
        at += value.size
    blob = Blob(f"{name}-test", out)
    blob.put(f"{name}-test.float32", np.concatenate(flat),
             f"test vector of {name} computed by PyTorch (float32); layout in {name}-test.json; no page loads it")
    if entries:
        size = blob.save()
    else:
        write_atomic(out / f"{name}-test.bin", bytes(blob.buf))
        size = len(blob.buf)
    write_json(out / f"{name}-test.json", {"name": name, "format": FORMAT, "torch": torch.__version__, **describe,
                                             "bytes": size, "arrays": listing})
    return size


# ----------------------------------------------------------------------------------------------------------------
# One model


def describe_model(name: str, cfg: ModelConfig, model: GPT, form: str, size: int, layout: list) -> dict:
    return {
        "name": name, "format": FORMAT, "config": cfg.to_dict(),
        "layers": cfg.n_layer, "heads": cfg.n_head, "width": cfg.d_model, "headWidth": cfg.d_head,
        "mlpWidth": cfg.d_mlp if cfg.use_mlp else 0, "vocabulary": cfg.vocab_size, "context": cfg.block_size,
        "params": model.num_params()["total"], "form": form, "bytes": size, "test": f"{name}-test", "layout": layout,
    }


def export_model(model: GPT, name: str, out: Path, form: str, tokens: Sequence[int],
                 long_tokens: Optional[Sequence[int]] = None, extra: Optional[dict] = None, entries: bool = True) -> dict:
    """Write the four files of a model in a given form. The model is left holding the weights as shipped."""
    out = Path(out)
    arrays = tensors(model)
    layout, size = pack(name, arrays, form, out, entries)
    meta = describe_model(name, model.cfg, model, form, size, layout)
    meta.update(extra or {})
    write_json(out / f"{name}.json", meta)
    read_back = unpack(meta, (out / f"{name}.bin").read_bytes())
    as_shipped = shipped(arrays, form)
    for tensor in arrays:
        if not np.array_equal(read_back[tensor], as_shipped[tensor]):
            raise AssertionError(f"{name}.bin: {tensor} does not decode to the weights that were measured")
    load_arrays(model, read_back)
    describe, vectors = test_vector(model, tokens, long_tokens)
    test_size = write_test(name, describe, vectors, out, entries)
    return {**meta, "testBytes": test_size}


def export_run(ckpt: str, name: str, out: Path, device: str, form: Optional[str] = None,
               tolerance: float = TOLERANCE) -> dict:
    """A run of the trainer: measure the forms on its held-out batches, ship the smallest that passes."""
    from ..data import TokenData, ValLoader

    path = Path(ckpt)
    run_dir = path if path.is_dir() else path.parent
    model, cfg = GPT.from_checkpoint(path, device="cpu")
    run = json.loads((run_dir / "config.json").read_text()) if (run_dir / "config.json").exists() else {}
    summary = json.loads((run_dir / "summary.json").read_text()) if (run_dir / "summary.json").exists() else {}
    train = run.get("train_config", {})
    extra: dict = {"run": run.get("run", run_dir.name), "preset": run.get("preset", "")}
    tokenizer = run.get("data_meta", {}).get("tokenizer")
    if tokenizer:
        extra.update(tokenizer=tokenizer, bos=run["data_meta"].get("bos_id"))
    if summary:
        extra["trained"] = {"data": train.get("data"), "tokens": summary.get("tokens"), "steps": summary.get("steps"),
                            "valLoss": summary.get("val_loss"), "valBitsPerByte": summary.get("val_bpb")}

    data = TokenData(train["data"]) if train.get("data") else None
    arrays = tensors(model)
    if data is not None:
        seq_len = train.get("seq_len") or cfg.block_size
        loader = ValLoader(data, seq_len, 256, 0, 1, train.get("val_tokens", 1048576.0), device)
        model.to(device)
        measured = measure_forms(model, arrays, list(loader))
        model.to("cpu")
        chosen = form or choose_form(measured, tolerance)
        extra["quantization"] = {
            "tolerance": tolerance, "windows": loader.n_windows, "seqLen": seq_len, "tokens": loader.n_windows * seq_len,
            "arithmetic": "float32", "loss": {form: round(loss, 6) for form, loss in measured["loss"].items()},
            "cost": {form: loss - measured["loss"]["float32"] for form, loss in measured["loss"].items()},
            "bytes": measured["bytes"], "chosen": chosen,
            "rule": "the smallest form whose held-out loss is within the tolerance of the float32 weights"
                    if form is None else "the form was given on the command line",
        }
        held_out = np.asarray(data.val_tokens()[: max(cfg.block_size, TEST_TOKENS)]).astype(np.int64)
        tokens, long_tokens = held_out[:TEST_TOKENS].tolist(), held_out[: cfg.block_size].tolist()
    else:
        if form is None:
            raise SystemExit(f"{run_dir} names no dataset: the forms cannot be measured; pass --form")
        chosen = form
        generator = torch.Generator().manual_seed(0)
        tokens = torch.randint(0, cfg.vocab_size, (TEST_TOKENS,), generator=generator).tolist()
        long_tokens = torch.randint(0, cfg.vocab_size, (cfg.block_size,), generator=generator).tolist()

    reference = model(torch.tensor([tokens]))[0]
    meta = export_model(model, name, out, chosen, tokens, long_tokens, extra)
    drift = (model(torch.tensor([tokens]))[0] - reference).abs().max().item()
    return {**meta, "logitDriftFromFloat32": drift}


# ----------------------------------------------------------------------------------------------------------------
# The tokenizer for the page


EDGE_STRINGS = [
    "", " ", "  ", "a", " a", "a ", "Hello, world!", "Once upon a time, there was a little girl named Lily.",
    "I'm sure it's fine; they'll say we've done what you're told, he'd know, I don't.",
    "I'M SURE IT'S FINE; THEY'LL SAY WE'VE DONE WHAT YOU'RE TOLD, HE'D KNOW.", "it'S they'Ll we'Ve you'rE he'D i'M don'T",
    "'s 'd 'm 't 'll 've 're", "rock 'n' roll", "the dogs' bones and James' hat", "it\u2019s a curly apostrophe",
    "it'\u017f the long s", "'tis the season", "''quoted''", "a 'quoted' word", "don''t",
    "1 22 333 4444 55555 666666 7777777", "3.14159 and 2,718,281.828", "in 2026, 12:30pm, 50%", "x2 y10 z100a",
    "\u0661\u0662\u0663\u0664 \u0969\u096a \u2460\u2461 \u00bd \u2167", "1st 2nd 3rd 4th", "v1.2.3-beta+build.5",
    "Wait... what?! No!!! Yes?? (maybe) [ok] {fine} <tag> a/b\\c", "a--b---c \u2014 d \u2013 e", "$5.00 #tag @home &co *bold* _it_ ~x^y",
    "\"She said, 'hello.'\"", "a.b,c;d:e!f?g", "...", "!!!\n", "?!\n\nNext", "end.\n", "end.\r\n", "end.\r\nNext line.\r\n",
    "tabs\tbetween\twords", "\tleading tab", "trailing tab\t", "two  spaces", "three   spaces", "many          spaces",
    " leading space", "trailing space ", "trailing spaces   ", "line one\nline two", "line one\n\nline three",
    "line one\n \n line three", "space then newline \nnext", "newline then space\n next", "\n", "\n\n\n", " \n ", "\r", "\r\r\n\n",
    "a\u00a0b no-break space", "a\u0085b next line", "a\u2028b line separator", "a\u2029b", "a\u3000b ideographic space",
    "a\u2009b thin space", "a\u200bb zero width space", "a\ufeffb byte order mark", "a\u001cb\u001db\u001eb\u001f file separators",
    "a\u000bb\u000cc vertical tab and form feed", "a\u1680b\u205fc\u202fd", "\u00a0\u00a0 x", "x \u00a0 \u2003 y", "x\u2028\u2028y",
    "caf\u00e9 na\u00efve r\u00e9sum\u00e9 \u00fcber stra\u00dfe", "cafe\u0301 with a combining accent", "\u00c5ngstr\u00f6m \u00d8rsted \u0141\u00f3d\u017a",
    "\u041f\u0440\u0438\u0432\u0435\u0442, \u043c\u0438\u0440!", "\u4f60\u597d\uff0c\u4e16\u754c\uff01", "\u3053\u3093\u306b\u3061\u306f\u4e16\u754c", "\uc548\ub155\ud558\uc138\uc694",
    "\u0645\u0631\u062d\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645", "\u05e9\u05dc\u05d5\u05dd \u05e2\u05d5\u05dc\u05dd", "\u0928\u092e\u0938\u094d\u0924\u0947 \u0926\u0941\u0928\u093f\u092f\u093e",
    "\u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35\u0e0a\u0e32\u0e27\u0e42\u0e25\u0e01", "\u03b1\u03b2\u03b3 \u0394x \u2264 \u03c0r\u00b2",
    "\U0001f600 a smile", "family \U0001f468\u200d\U0001f469\u200d\U0001f467 joined", "flag \U0001f1eb\U0001f1f7", "thumbs \U0001f44d\U0001f3fd up",
    "math \U0001d54f \U0001d4b3 letters", "\u2603\ufe0f snowman", "a\u0301\u0302\u0303 stacked marks",
    "<|bos|>", "<|bos|>Once upon a time", "text <|user_start|> more <|assistant_end|>", "<|endoftext|>",
    "camelCaseWord snake_case_word kebab-case-word SCREAMING", "http://example.com/a?b=c&d=e#f", "user@example.com",
    "def f(x):\n    return x + 1\n", "for (i = 0; i < n; i++) { s += a[i]; }", "x = y ** 2 // 3 % 4",
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", "abababababababababab", "zzz zzz zzz zzz", "the the the the the",
    "Supercalifragilisticexpialidocious", "antidisestablishmentarianism pneumonoultramicroscopicsilicovolcanoconiosis",
    "Mr. and Mrs. O'Brien-Smith went to St. John's.", "\"Stop!\" cried Tom. \"Why?\" asked Lily.",
    "One day, a little dog named Max found a big red ball in the park.", "The sun was shining and the birds were singing.",
]


def tokenizer_tests(name: str, n_strings: int = 220, seed: int = 0) -> dict:
    """Strings with the ids of the reference encoders, id sequences with their text, and small texts with the
    merges the reference trainer learns on them. A string on which the two reference encoders (gptpp/bpe.py with
    Python's `regex`, gptpp/tokenizer.py with the Rust library) disagree is left out and counted."""
    from .. import bpe
    from ..tokenizer import Tokenizer, tokenizer_dir

    production = Tokenizer.load(name)
    reference = bpe.BPE(production.merges, production.pattern, list(production.special))
    rng = random.Random(seed)
    texts = list(EDGE_STRINGS)
    docs_file = WORK / "data" / "tinystories" / "val_docs.jsonl"
    if docs_file.exists():
        docs = []
        with open(docs_file, encoding="utf-8") as f:
            for line in f:
                docs.append(json.loads(line)["text"])
                if len(docs) >= 2000:
                    break
        for _ in range(n_strings):
            doc = rng.choice(docs)
            length = rng.choice([12, 30, 60, 120, 240])
            start = rng.randrange(0, max(1, len(doc) - length))
            texts.append(doc[start:start + length])
    strings, dropped = [], 0
    for text in texts:
        ids = reference.encode(text)
        if ids != production.encode(text):
            dropped += 1
            continue
        strings.append({"text": text, "ids": ids})
    decode = []
    for _ in range(40):
        ids = [rng.randrange(0, production.vocab_size) for _ in range(rng.randrange(1, 12))]
        decode.append({"ids": ids, "text": production.decode(ids)})
    for ids in ([226], [226, 130], [226, 130, 172], [240, 159], [255, 254], [195], [195, 169, 195], [production.bos_id, 72, 105]):
        decode.append({"ids": ids, "text": production.decode(ids)})
    special = len(production.special)
    train = []
    for text, count in (
        ("low low low low low lower lower newest newest newest newest newest newest widest widest widest", 12),
        ("Once upon a time, there was a little girl named Lily. She loved to play with her toys. One day, Lily "
         "found a big red ball. She played with the ball all day. Lily was very happy.", 40),
        ("the cat sat on the mat. the cat ate the rat. \u00e9t\u00e9 \u00e9t\u00e9 na\u00efve na\u00efve 123 123 1234 !!! !!!\n\nthe end", 30),
        ("aaaa bbbb aaaa bbbb abab abab", 8),
    ):
        words = bpe.count_words(text, production.pattern)
        steps = []
        for new_id in range(256, 256 + count):
            counts = bpe.count_pairs(words)
            if not counts:
                break
            pair = min(counts, key=lambda p: (-counts[p], p))
            steps.append({"pair": list(pair), "count": counts[pair]})
            words = {tuple(bpe.merge(word, pair, new_id)): n for word, n in words.items()}
        merges = bpe.train(text, 256 + special + count, production.pattern, list(production.special))
        assert [tuple(step["pair"]) for step in steps] == merges, "the recorded steps are not the reference trainer's merges"
        train.append({"text": text, "count": count, "steps": steps})
    return {"name": name, "reference": "gptpp/bpe.py (Python regex) and gptpp/tokenizer.py (tokenizers), which agree on every string kept",
            "dropped": dropped, "strings": strings, "decode": decode, "train": train,
            "folder": str(tokenizer_dir(name))}


def export_tokenizer(name: str, out: Path) -> dict:
    """<name>.json (what the page needs: the merges, the special tokens, the split pattern as JavaScript reads it)
    and <name>-test.json (strings with their ids from the reference encoders; no page loads it)."""
    from .. import bpe
    from ..tokenizer import Tokenizer

    out = Path(out)
    tokenizer = Tokenizer.load(name)
    if tokenizer.pattern != bpe.GPT4_PATTERN:
        raise SystemExit(f"tokenizer {name} splits with another pattern than the one JS_PATTERN was written for")
    meta = {
        "name": name, "format": FORMAT, "vocabulary": tokenizer.vocab_size, "bos": tokenizer.bos_id, "special": tokenizer.special,
        "ids": "0..255 the bytes; then the merges in the order learned (merge k makes token 256 + k); then the special tokens",
        "pattern": {"python": tokenizer.pattern, "js": JS_PATTERN, "flags": JS_FLAGS},
        "trainedOn": tokenizer.info.get("trained_on", ""),
        "merges": [i for pair in tokenizer.merges for i in pair],
    }
    size = write_json(out / f"{name}.json", meta)
    tests = tokenizer_tests(name)
    folder = tests.pop("folder")
    test_size = write_json(out / f"{name}-test.json", tests)
    return {"name": name, "bytes": size, "testBytes": test_size, "strings": len(tests["strings"]), "dropped": tests["dropped"],
            "decode": len(tests["decode"]), "train": len(tests["train"]), "from": folder}


# ----------------------------------------------------------------------------------------------------------------
# Fixtures: tiny random models that together take every value of every field of the configuration


_TINY = dict(vocab_size=64, d_model=16, block_size=20)
FIXTURES: dict[str, dict] = {
    # the circuit models: attention only, no normalization anywhere, plain dot-product scores, no cap on the logits
    "tiny-attention-only": dict(_TINY, n_layer=2, n_head=8, pos="rope", norm="none", embed_norm=False, use_mlp=False,
                                qk_norm=False, logit_softcap=0.0, tie_embeddings=False, bias=False),
    "tiny-rmsnorm-relu-tied": dict(_TINY, n_layer=2, n_head=4, pos="none", norm="rmsnorm", embed_norm=True, act="relu",
                                   qk_norm=True, logit_softcap=15.0, tie_embeddings=True, bias=False),
    "tiny-layernorm-silu-learned": dict(_TINY, n_layer=3, n_head=2, pos="learned", norm="layernorm", embed_norm=False,
                                        act="silu", qk_norm=False, logit_softcap=0.0, tie_embeddings=False, bias=True),
    "tiny-gelu-rope-tied": dict(_TINY, n_layer=2, n_head=4, pos="rope", norm="rmsnorm_fixed", embed_norm=True, act="gelu",
                                mlp_mult=2.0, qk_norm=True, qk_scale=6.0, logit_softcap=0.0, tie_embeddings=True, bias=False),
    "tiny-relu2-one-layer": dict(_TINY, n_layer=1, n_head=1, pos="rope", rope_base=100.0, norm="rmsnorm_fixed", embed_norm=False,
                                 act="relu2", qk_norm=True, logit_softcap=15.0, tie_embeddings=False, bias=False),
    # the rest of the lab's switches: post-norm, sinusoidal and ALiBi positions, gated MLPs, grouped keys and
    # values, a learned score scale, dynamic tanh
    "tiny-postnorm-swiglu-sinusoidal": dict(_TINY, n_layer=2, n_head=4, n_kv_head=2, pos="sinusoidal", norm="layernorm",
                                            norm_placement="post", embed_norm=False, act="swiglu", qk_norm=True,
                                            qk_scale_learned=True, logit_softcap=0.0, tie_embeddings=False, bias=True,
                                            init="gpt2"),
    "tiny-dyt-geglu-alibi": dict(_TINY, n_layer=2, n_head=4, n_kv_head=1, pos="alibi", norm="dyt", embed_norm=True,
                                 act="geglu", qk_norm=False, logit_softcap=15.0, tie_embeddings=True, bias=True),
}


def random_model(cfg: ModelConfig, seed: int) -> GPT:
    """A model whose every parameter is random and far from its initial value, so that no path of the
    computation is switched off (the lab's own initialization starts each block as the identity)."""
    generator = torch.Generator().manual_seed(seed)
    model = GPT(cfg).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            leaf = name.rsplit(".", 1)[-1]
            if p.ndim == 2 and name.startswith(("embed", "pos_embed", "unembed")):
                p.copy_(torch.randn(p.shape, generator=generator) * 0.8)
            elif p.ndim == 2:
                p.copy_(torch.randn(p.shape, generator=generator) * (1.2 / p.shape[1] ** 0.5))
            elif leaf in ("gain", "qk_factor"):
                p.copy_(1.0 + 0.3 * torch.randn(p.shape, generator=generator))
            elif leaf == "alpha":
                p.copy_(torch.full(p.shape, 0.7))
            else:
                p.copy_(0.3 * torch.randn(p.shape, generator=generator))
    return model


def export_fixtures(out: Path) -> list[dict]:
    """The fixtures, in float32 (the weights of the test are then PyTorch's own), with their test vectors."""
    out = Path(out)
    done = []
    for k, (name, fields) in enumerate(FIXTURES.items()):
        cfg = ModelConfig().replace(**fields)
        model = random_model(cfg, seed=100 + k)
        generator = torch.Generator().manual_seed(200 + k)
        tokens = torch.randint(0, cfg.vocab_size, (12,), generator=generator).tolist()
        long_tokens = torch.randint(0, cfg.vocab_size, (cfg.block_size,), generator=generator).tolist()
        meta = export_model(model, name, out, "float32", tokens, long_tokens, {"fixture": True}, entries=False)
        done.append({"name": name, "bytes": meta["bytes"], "testBytes": meta["testBytes"], "params": meta["params"]})
    return done


# ----------------------------------------------------------------------------------------------------------------
# The stories of the series' opening figure, written by the page's own model


STORIES_JS = r"""
const [site, out] = process.argv.slice(1), path = require('path'), fs = require('fs');   // after `node -e`, the arguments begin at 1
const M = require(path.join(site, 'assets/js/gpt/gpt-transformer-forward-pass.js')), T = require(path.join(site, 'assets/js/gpt/gpt-byte-pair-tokenizer.js'));
const F = require(path.join(site, 'assets/js/gpt/gpt-pico-writes-a-story-figure.js')), name = F['gpt-pico-writes-a-story.defaults'].model;
const D = F['gpt-pico-writes-a-story.record'](M.fromDisk(name, out), T.fromDisk(F['gpt-pico-writes-a-story.defaults'].tokenizer, out));
fs.writeFileSync(path.join(out, 'pico-stories.json'), JSON.stringify(D));
console.log(JSON.stringify(D.stories.map((s) => ({ opening: s.opening, seed: s.seed, tokens: s.tokens.length, at: s.at }))));
"""


def export_stories(out: Path) -> Optional[str]:
    """pico-stories.json: what the figure `gpt-pico-writes-a-story` shows before the weights are loaded. The stories
    are written in Node by the JavaScript model itself (the figure module's `record`), so that the page continues
    them with the same numbers. Returns Node's report, or None when Node or the figure module is not there."""
    import shutil

    node = shutil.which("node") or str(Path.home() / ".local" / "bin" / "node")
    figure = SITE / "assets" / "js" / "gpt" / "gpt-pico-writes-a-story-figure.js"
    if not Path(node).exists() or not figure.exists():
        return None
    done = subprocess.run([node, "-e", STORIES_JS, str(SITE), str(out)], capture_output=True, text=True, timeout=600)
    if done.returncode != 0:
        raise SystemExit(f"recording the stories failed: {done.stderr.strip()[-2000:]}")
    return done.stdout.strip()


# ----------------------------------------------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Export a model, its tokenizer and a test vector for the browser.")
    parser.add_argument("--ckpt", help="a run directory of the trainer (with model.pt), or a checkpoint file")
    parser.add_argument("--name", help="the name of the files: <name>.json, <name>.bin, <name>-test.json, <name>-test.bin")
    parser.add_argument("--form", choices=FORMS, default=None, help="ship this form instead of the smallest that passes")
    parser.add_argument("--tolerance", type=float, default=TOLERANCE, help="nats of held-out loss a smaller form may cost")
    parser.add_argument("--tokenizer", default=None, help="export this tokenizer (default: the one of the run, if not there yet)")
    parser.add_argument("--fixtures", default=None, help="write the tiny random models of the JavaScript tests into this folder")
    parser.add_argument("--stories", action="store_true", help="record the stories of the opening figure (done by itself when --name is pico)")
    parser.add_argument("--out", default=str(DATA), help="the site's data folder")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="where the held-out loss is measured")
    args = parser.parse_args(argv)
    out = Path(args.out)
    if not (args.ckpt or args.tokenizer or args.fixtures or args.stories):
        parser.error("nothing to do: give --ckpt and --name, --tokenizer, --fixtures or --stories")

    if args.fixtures:
        for item in export_fixtures(Path(args.fixtures)):
            print(f"fixture {item['name']}: {item['params']} parameters, {item['bytes']} bytes, test {item['testBytes']} bytes")
    tokenizer = args.tokenizer
    if args.ckpt:
        if not args.name:
            parser.error("--ckpt needs --name")
        meta = export_run(args.ckpt, args.name, out, args.device, args.form, args.tolerance)
        q = meta.get("quantization")
        if q:
            print(f"{args.name}: held-out loss over {q['tokens']} tokens, float32 arithmetic")
            for form in FORMS[::-1]:
                print(f"  {form:<12} {q['loss'][form]:.5f} nats  ({q['loss'][form] - q['loss']['float32']:+.5f})  {q['bytes'][form] / 1e6:.3f} MB")
        print(f"{args.name}: shipped as {meta['form']}, {meta['bytes']} bytes, {meta['params']} parameters; "
              f"test vector {meta['testBytes']} bytes; largest change of a logit on the test sequence against the "
              f"float32 weights {meta['logitDriftFromFloat32']:.4f}")
        if tokenizer is None and meta.get("tokenizer") and not (out / f"{meta['tokenizer']}.json").exists():
            tokenizer = meta["tokenizer"]
    if tokenizer:
        done = export_tokenizer(tokenizer, out)
        print(f"tokenizer {done['name']}: {done['bytes']} bytes; test file {done['testBytes']} bytes with {done['strings']} strings "
              f"({done['dropped']} left out: the reference encoders disagree on them), {done['decode']} id sequences, "
              f"{done['train']} training texts")
    if args.stories or (args.ckpt and args.name == "pico"):
        report = export_stories(out)
        print(f"stories for the opening figure: {report}" if report else "stories for the opening figure: not recorded (Node or the figure module is not here)")
    if args.ckpt or tokenizer:
        merged = subprocess.run([sys.executable, "-m", "gptpp.export.manifest", "--out", str(out)], capture_output=True, text=True)
        print(merged.stdout.strip().splitlines()[-1] if merged.stdout.strip() else merged.stderr.strip())


if __name__ == "__main__":
    main()
