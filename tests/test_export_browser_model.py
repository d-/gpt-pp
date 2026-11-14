"""The exporter of the in-page model (gptpp/export/browser_model.py).

    cd gpt-pp && $PY -m pytest -q tests/test_export_browser_model.py

What is checked: the 8-bit forms and their error; that a weights file decodes to exactly the weights that were
measured, in every form; that the form shipped is the smallest within the tolerance; that the test vector is what
the decoded model computes; the fixtures (tiny random models that take every value of every field of the
configuration); and, when Node and the site are there, that the JavaScript model computes PyTorch's logits on the
fixtures and that the JavaScript spelling of the split pattern cuts like the Python pattern.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from gptpp import bpe
from gptpp.config import ACTIVATIONS, NORMS, PLACEMENTS, POSITIONS, ModelConfig
from gptpp.export import browser_model as bm
from gptpp.export.common import SITE, WORK
from gptpp.model import GPT

NODE = shutil.which("node") or (str(Path.home() / ".local/bin/node") if (Path.home() / ".local/bin/node").exists() else None)
JS_MODEL = SITE / "assets" / "js" / "gpt" / "gpt-transformer-forward-pass.js"


@pytest.fixture(scope="module", autouse=True)
def single_thread_no_gradients():
    """Tiny tensors: one thread is faster than waking a pool; nothing here differentiates."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.no_grad():
        yield
    torch.set_num_threads(threads)


def tiny(**fields) -> GPT:
    cfg = ModelConfig().replace(**{**dict(vocab_size=40, d_model=16, n_head=4, n_layer=2, block_size=12), **fields})
    return bm.random_model(cfg, seed=1)


def test_quantize_puts_the_largest_magnitude_at_127_and_errs_by_half_a_step():
    rng = np.random.default_rng(0)
    w = rng.normal(size=(6, 50)).astype(np.float32) * np.array([[0.01], [1], [30], [0.5], [2], [1]], dtype=np.float32)
    w[3] = 0.0
    for per_row in (True, False):
        q, scale = bm.quantize(w, per_row)
        assert q.dtype == np.int8 and np.abs(q).max() == 127 and scale.dtype == np.float32
        assert scale.shape == ((6,) if per_row else (1,))
        error = np.abs(bm.dequantize(q, scale) - w)
        assert (error <= scale.reshape(-1, 1) / 2 * 1.0001).all()
    q, scale = bm.quantize(w, per_row=True)
    assert (np.abs(q).max(axis=1)[[0, 1, 2, 4, 5]] == 127).all(), "every row uses its full range"
    assert scale[3] == 1.0 and not q[3].any(), "a row of zeros keeps a scale of one"
    # one scale for the matrix leaves the small rows with few levels: the reason a scale a row is tried
    coarse = np.abs(bm.dequantize(*bm.quantize(w, False)) - w)[0].max()
    fine = np.abs(bm.dequantize(*bm.quantize(w, True)) - w)[0].max()
    assert fine < coarse / 50


def test_tensors_are_copies_and_a_tied_unembedding_is_left_out():
    model = tiny(tie_embeddings=True)
    arrays = bm.tensors(model)
    assert "unembed.weight" not in arrays and "embed.weight" in arrays
    before = {name: w.copy() for name, w in arrays.items()}
    bm.load_arrays(model, bm.shipped(arrays, "int8-tensor"))
    assert all(np.array_equal(arrays[name], before[name]) for name in arrays), "loading another form must not touch the float32 weights"
    assert model.unembed.weight is model.embed.weight
    assert set(bm.tensors(tiny(tie_embeddings=False))) >= {"embed.weight", "unembed.weight"}


@pytest.mark.parametrize("form", bm.FORMS)
def test_a_weights_file_decodes_to_the_weights_that_were_measured(form, tmp_path):
    model = tiny(norm="layernorm", bias=True, pos="learned")
    arrays = bm.tensors(model)
    layout, size = bm.pack("m", arrays, form, tmp_path)
    raw = (tmp_path / "m.bin").read_bytes()
    assert size == len(raw) and 0 <= len(raw) - bm.form_bytes(arrays, form) < 4
    decoded = bm.unpack({"layout": layout}, raw)
    expected = bm.shipped(arrays, form)
    assert list(decoded) == list(arrays)
    for name in arrays:
        assert decoded[name].dtype == np.float32 and np.array_equal(decoded[name], expected[name]), name
    # the entries of the site's manifest tile the file
    entries = json.loads((tmp_path / "m.entries.json").read_text())
    covered = sorted((e["offset"], e["offset"] + e["bytes"]) for e in entries.values())
    assert covered[0][0] == 0 and covered[-1][1] == len(raw) and all(key.startswith("m.") for key in entries)
    for (_, end), (start, _) in zip(covered, covered[1:]):
        assert 0 <= start - end < 4
    kinds = {item["dtype"] for item in layout}
    assert kinds == ({"f32"} if form == "float32" else {"f16"} if form == "float16" else {"i8", "f32"})
    if form.startswith("int8"):
        for item in layout:
            if item["dtype"] == "i8":
                assert item["scales"]["count"] == (item["shape"][0] if form == "int8-row" else 1)
            else:
                assert len(item["shape"]) == 1, "vectors stay float32"


def test_the_smallest_form_within_the_tolerance_is_chosen():
    sizes = {"int8-tensor": 100, "int8-row": 103, "float16": 200, "float32": 400}
    def chosen(costs):
        return bm.choose_form({"loss": {form: 2.0 + cost for form, cost in costs.items()}, "bytes": sizes})
    assert chosen({"int8-tensor": 0.004, "int8-row": 0.001, "float16": 0.0, "float32": 0.0}) == "int8-tensor"
    assert chosen({"int8-tensor": 0.0077, "int8-row": 0.0021, "float16": 0.0, "float32": 0.0}) == "int8-row"
    assert chosen({"int8-tensor": 0.02, "int8-row": 0.006, "float16": 0.0001, "float32": 0.0}) == "float16"
    assert chosen({"int8-tensor": 0.02, "int8-row": 0.006, "float16": 0.0051, "float32": 0.0}) == "float32"
    assert chosen({"int8-tensor": -0.001, "int8-row": 0.0, "float16": 0.0, "float32": 0.0}) == "int8-tensor"


def test_the_forms_are_measured_on_the_same_weights_and_the_model_is_left_as_it_was():
    model = tiny()
    arrays = bm.tensors(model)
    generator = torch.Generator().manual_seed(0)
    x = torch.randint(0, 40, (8, 12), generator=generator)
    batches = [(x[:, :-1], x[:, 1:], 8 * 11)]
    measured = bm.measure_forms(model, arrays, batches)
    assert set(measured["loss"]) == set(bm.FORMS)
    assert measured["loss"]["float32"] == pytest.approx(bm.heldout_loss(model, batches), abs=0, rel=0)
    assert abs(measured["loss"]["float16"] - measured["loss"]["float32"]) < 1e-3
    assert measured["bytes"]["int8-tensor"] < measured["bytes"]["int8-row"] < measured["bytes"]["float16"] < measured["bytes"]["float32"]
    after = bm.tensors(model)
    assert all(np.array_equal(after[name], arrays[name]) for name in arrays)
    # measured again in another order, each form's loss is the same: no form is derived from another
    again = bm.measure_forms(model, arrays, batches, forms=bm.FORMS[::-1])
    assert again["loss"] == pytest.approx(measured["loss"], abs=1e-7)


def test_an_exported_model_is_read_back_and_computes_its_own_test_vector(tmp_path):
    model = tiny(act="gelu", norm="rmsnorm")
    reference = model(torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]))[0].clone()
    meta = bm.export_model(model, "m", tmp_path, "int8-row", [1, 2, 3, 4, 5, 6, 7, 8], list(range(12)), {"run": "a test"})
    assert {p.name for p in tmp_path.iterdir()} == {"m.json", "m.bin", "m.entries.json", "m-test.json", "m-test.bin", "m-test.entries.json"}
    stored = json.loads((tmp_path / "m.json").read_text())
    assert stored["form"] == "int8-row" and stored["run"] == "a test" and stored["params"] == meta["params"]
    assert [stored[k] for k in ("layers", "heads", "width", "headWidth", "mlpWidth", "vocabulary", "context")] == [2, 4, 16, 4, 64, 40, 12]
    assert ModelConfig.from_dict(stored["config"]) == model.cfg
    page, cfg = bm.page_model("m", tmp_path)
    test = json.loads((tmp_path / "m-test.json").read_text())
    vectors = np.frombuffer((tmp_path / "m-test.bin").read_bytes(), dtype="<f4")
    arrays = {a["name"]: vectors[a["offset"] // 4: a["offset"] // 4 + a["count"]].reshape(a["shape"]) for a in test["arrays"]}
    logits, cache = page.run_with_cache(torch.tensor([test["tokens"]]))
    assert np.allclose(arrays["logits"], logits[0].numpy(), atol=1e-6)
    assert arrays["patterns"].shape == (2, 4, 8, 8) and arrays["resid"].shape == (3, 8, 16) and arrays["neurons"].shape == (8, 64)
    assert np.allclose(arrays["patterns"][1], cache["attn.1.pattern"][0].numpy(), atol=1e-6)
    assert np.allclose(arrays["resid"][0], cache["embed"][0].numpy(), atol=1e-6)
    assert np.allclose(arrays["resid"][2], cache["resid_post.1"][0].numpy(), atol=1e-6)
    layer, head = test["layer"], test["head"]
    # the queries and keys of the test vector are the ones whose product is the score
    scores = arrays["head.q"] @ arrays["head.k"].T / np.sqrt(cfg.d_head)
    seen = np.isfinite(arrays["head.scores"])
    assert np.allclose(scores[seen], arrays["head.scores"][seen], atol=1e-4) and (seen == np.tril(np.ones((8, 8), dtype=bool))).all()
    assert np.allclose(arrays["head.result"], cache[f"attn.{layer}.z"][0, head].numpy() @ page.blocks[layer].attn.o.weight.detach().numpy()[:, 4 * head: 4 * head + 4].T, atol=1e-5)
    assert np.allclose(arrays["lens"][-1], page.unembed_logits(page.final_norm(cache[f"resid_post.{test['lensDepths'][-1] - 1}"][0, -1])).detach().numpy(), atol=1e-5)
    assert np.allclose(arrays["longLogits"], page(torch.tensor([test["long"]]))[0, -1].detach().numpy(), atol=1e-6)
    # the weights as shipped differ from the float32 weights, a little
    drift = (logits[0] - reference).abs().max().item()
    assert 0 < drift < 0.5


def test_the_fixtures_take_every_value_of_every_switch(tmp_path):
    configs = {name: ModelConfig().replace(**fields) for name, fields in bm.FIXTURES.items()}
    def values(field):
        return {getattr(cfg, field) for cfg in configs.values()}
    assert values("pos") == set(POSITIONS) and values("norm") == set(NORMS) and values("norm_placement") == set(PLACEMENTS)
    assert values("act") | {"relu2", "gelu"} == set(ACTIVATIONS), "relu2 and gelu are also pico and pico-2019"
    for field in ("use_mlp", "qk_norm", "qk_scale_learned", "embed_norm", "bias", "tie_embeddings"):
        assert values(field) == {True, False}, field
    assert values("logit_softcap") == {0.0, 15.0} and values("n_kv_head") == {None, 1, 2}
    circuit = configs["tiny-attention-only"]
    assert not circuit.use_mlp and circuit.norm == "none" and not circuit.embed_norm and not circuit.qk_norm and circuit.logit_softcap == 0
    done = bm.export_fixtures(tmp_path)
    assert [item["name"] for item in done] == list(bm.FIXTURES)
    assert not list(tmp_path.glob("*.entries.json")), "fixtures are not data of the site: no manifest entries"
    for name, cfg in configs.items():
        page, read = bm.page_model(name, tmp_path)
        assert read == cfg
        test = json.loads((tmp_path / f"{name}-test.json").read_text())
        vectors = np.frombuffer((tmp_path / f"{name}-test.bin").read_bytes(), dtype="<f4")
        first = test["arrays"][0]
        assert first["name"] == "logits" and first["shape"] == [12, cfg.vocab_size]
        logits = page(torch.tensor([test["tokens"]]))[0].detach().numpy()
        assert np.allclose(vectors[: first["count"]].reshape(first["shape"]), logits, atol=1e-5), name
        assert np.abs(logits).max() > 1 and np.isfinite(logits).all(), "a random model that says something"
    # every parameter of a random model is far from its initial value: no path of the computation is switched off
    model = bm.random_model(configs["tiny-layernorm-silu-learned"], seed=3)
    assert all(p.abs().mean() > 0.05 for p in model.parameters())


JS_COMPARE = r"""
const M = require(process.argv[2]), fs = require('fs'), path = require('path'), dir = process.argv[3], out = {};
for (const f of fs.readdirSync(dir).filter((f) => /\.json$/.test(f) && !/-test\.json$/.test(f))) {
  const name = f.replace(/\.json$/, ''), model = M.fromDisk(name, dir), t = JSON.parse(fs.readFileSync(path.join(dir, name + '-test.json'), 'utf8'));
  const b = fs.readFileSync(path.join(dir, name + '-test.bin')), ref = new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
  const logits = model.forward(t.tokens).logits, kv = model.newCache();
  let worst = 0, scale = 1, cached = 0;
  for (let i = 0; i < logits.length; i++) { worst = Math.max(worst, Math.abs(logits[i] - ref[i])); scale = Math.max(scale, Math.abs(ref[i])); }
  t.tokens.forEach((tok, i) => { const o = model.step(kv, tok).logits; for (let v = 0; v < o.length; v++) cached = Math.max(cached, Math.abs(o[v] - logits[i * o.length + v])); });
  out[name] = { worst, scale, cached };
}
console.log(JSON.stringify(out));
"""


@pytest.mark.skipif(NODE is None or not JS_MODEL.exists(), reason="Node or the site's JavaScript model is not here")
def test_the_javascript_model_computes_the_logits_of_pytorch_on_every_fixture(tmp_path):
    bm.export_fixtures(tmp_path)
    script = tmp_path / "compare.js"
    script.write_text(JS_COMPARE)
    result = subprocess.run([NODE, str(script), str(JS_MODEL), str(tmp_path)], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert set(report) == set(bm.FIXTURES)
    for name, item in report.items():
        assert item["worst"] <= 2e-5 * item["scale"], f"{name}: logits differ by {item['worst']:.2e}"
        assert item["cached"] == 0, f"{name}: the cache changes the logits"


JS_SPLIT = r"""
const strings = JSON.parse(require('fs').readFileSync(process.argv[2], 'utf8')), re = () => new RegExp(process.argv[3], process.argv[4]);
console.log(JSON.stringify(strings.map((s) => s.match(re()) || [])));
"""


@pytest.mark.skipif(NODE is None, reason="Node is not here")
def test_the_javascript_spelling_of_the_split_pattern_cuts_like_the_python_pattern(tmp_path):
    import regex

    rng = np.random.default_rng(0)
    alphabet = list(" \t\n\r'sStTdDmMlLvVeErR.,!?-ab12") + [chr(c) for c in (0x85, 0xA0, 0x2028, 0x3000, 0xFEFF, 0x17F, 0xE9, 0x4E2D, 0x661, 0x1F600, 0x1C, 0x200B)]
    strings = list(bm.EDGE_STRINGS) + ["".join(rng.choice(alphabet, size=rng.integers(1, 24))) for _ in range(3000)]
    (tmp_path / "strings.json").write_text(json.dumps(strings), encoding="utf-8")
    script = tmp_path / "split.js"
    script.write_text(JS_SPLIT)
    result = subprocess.run([NODE, str(script), str(tmp_path / "strings.json"), bm.JS_PATTERN, bm.JS_FLAGS], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    cuts = json.loads(result.stdout)
    split = regex.compile(bpe.GPT4_PATTERN)
    for text, cut in zip(strings, cuts):
        assert cut == split.findall(text), repr(text)
        assert "".join(cut) == text
    assert all(ord(ch) < 127 for ch in bm.JS_PATTERN), "the pattern is written with escapes, no invisible character in it"


@pytest.mark.skipif(not (WORK / "data" / "tokenizers" / "bpe4k" / "meta.json").exists(), reason="the tokenizer bpe4k is not in the workspace")
def test_the_tokenizer_for_the_page(tmp_path):
    done = bm.export_tokenizer("bpe4k", tmp_path)
    meta = json.loads((tmp_path / "bpe4k.json").read_text())
    tests = json.loads((tmp_path / "bpe4k-test.json").read_text())
    assert meta["vocabulary"] == 4096 and meta["bos"] == 4091 and len(meta["merges"]) == 2 * 3835 and len(meta["special"]) == 5
    assert meta["pattern"] == {"python": bpe.GPT4_PATTERN, "js": bm.JS_PATTERN, "flags": "gu"}
    merges = list(zip(meta["merges"][::2], meta["merges"][1::2]))
    reference = bpe.BPE(merges)
    assert done["strings"] == len(tests["strings"]) >= 300 and tests["dropped"] == 0
    for item in tests["strings"][::7]:
        assert reference.encode(item["text"]) == item["ids"] and reference.decode(item["ids"]) == item["text"]
    assert any(ord(ch) > 0xFFFF for item in tests["strings"] for ch in item["text"]), "characters beyond the basic plane are among the strings"
    for item in tests["decode"]:
        assert reference.decode(item["ids"]) == item["text"]
    for item in tests["train"]:
        learned = bpe.train(item["text"], 256 + 5 + item["count"])
        assert [tuple(step["pair"]) for step in item["steps"]] == learned
        assert all(a["count"] >= b["count"] for a, b in zip(item["steps"], item["steps"][1:])), "the counts of successive merges never rise"
    assert "strings" not in meta and "folder" not in tests, "the page's file carries no test strings"
