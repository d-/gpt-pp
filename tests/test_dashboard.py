"""Tests of gptpp/interp/dashboard.py and of the exporters that read a dictionary (feature_browser,
sparsity_against_fidelity, steering_pico's sampler), on a small run on the CPU.

    cd gpt-pp && $PY -m pytest -q tests/test_dashboard.py

What is asserted: the record of a dictionary on held-out text holds exactly its nonzero activations, sorted by
latent; a dashboard's density, peak, top examples, samples by level, snippets and neighbors are what they say; the
exported browser file describes itself; the family export keeps one model when another is written; and the
Python copy of the page's sampler draws the page's numbers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

from gptpp.interp import dashboard as B  # noqa: E402
from gptpp.interp import sae as S  # noqa: E402
from test_sae import BOS, write_run  # noqa: E402


class Letters:
    """A tokenizer of the toy run: token i is a letter, the mark is a bar."""

    bos_id = BOS

    def decode(self, ids) -> str:
        return "".join("|" if i == BOS else chr(97 + i % 26) for i in ids)


@pytest.fixture(scope="module")
def board(tmp_path_factory):
    root = tmp_path_factory.mktemp("work")
    monkey = pytest.MonkeyPatch()
    monkey.setenv("GPTPP_WORK", str(root))
    torch.set_num_threads(1)
    write_run(root)
    S.main(["--run", "toy-run", "--layer", "1", "--widths", "64", "--ks", "4", "--tokens", "30000", "--batch", "256", "--skip", "16",
            "--buffer", "4000", "--dead-after", "5000", "--splice-windows", "32", "--device", "cpu"])
    model, data, source = S.load_run("toy-run", None, "cpu")
    folder = S.dict_dir("toy-run", 1)
    sae = S.load(folder / "topk-n64-k4.pt", "cpu")
    harvest = S.Harvest(1, 32, 16, BOS)
    tokens = data.val_tokens()
    codes = B.cached_codes(folder, "topk-n64-k4", model, sae, tokens, harvest, 8, "cpu")
    yield B.Dashboards(model, sae, codes, tokens, harvest, Letters()), folder, root
    monkey.undo()


def test_the_record_holds_every_nonzero_activation_sorted_by_latent(board):
    D, folder, _ = board
    codes, sae, harvest = D.codes, D.sae, D.harvest
    windows = (len(D.tokens) - 1) // 32
    idx = torch.from_numpy(np.asarray(D.tokens[: windows * 32]).astype(np.int64).reshape(windows, 32))
    keep = harvest.keep(idx)
    assert codes.counted == int(keep.sum())
    with torch.no_grad():
        z = sae.encode(sae.normalize(S.activations(D.model, idx, harvest)[keep]))
    assert len(codes.act) == int((z > 0).sum()) and codes.start[-1] == len(codes.act)
    position = torch.arange(idx.numel()).view_as(idx)[keep]
    for i in (0, 7, 63):
        tok, act = codes.of(i)
        mine = z[:, i] > 0
        assert np.array_equal(tok, position[mine].numpy()) and np.allclose(act, z[mine, i].numpy(), atol=1e-3)
        assert np.all(np.diff(tok) > 0), "within a latent the tokens are in the order of the stream"
    assert (folder / "topk-n64-k4.codes.npz").exists()
    again = B.Codes.load(folder / "topk-n64-k4.codes.npz")
    assert np.array_equal(again.tok, codes.tok) and again.counted == codes.counted


def test_a_dashboard_is_what_it_says(board):
    D, _, _ = board
    i = int(np.argmax(D.count))
    tok, act = D.codes.of(i)
    assert D.density[i] == pytest.approx(len(tok) / D.codes.counted) and D.peak[i] == pytest.approx(float(act.max()))
    assert sum(D.histogram(i)) == len(tok) and len(D.histogram(i)) == B.BINS
    top = D.top(i, 5)
    assert [a for _, a in top] == sorted((a for _, a in top), reverse=True) and top[0][1] == pytest.approx(D.peak[i])
    assert all(abs(t1 - t2) > B.LEFT + B.RIGHT for k, (t1, _) in enumerate(top) for t2, _ in top[:k]), "no two top examples share a snippet"
    levels = D.samples(i, 3)
    assert len(levels) == B.LEVELS
    for q, level in enumerate(levels):
        for _, a in level:
            lo, hi = (B.LEVELS - q - 1) / B.LEVELS, (B.LEVELS - q) / B.LEVELS
            assert lo * D.peak[i] - 1e-6 <= a <= hi * D.peak[i] + 1e-6, "a sample lies in its level"
    assert D.samples(i, 3) == D.samples(i, 3), "the draw is seeded"
    t = top[0][0]
    ids, values, at = D.snippet(i, t)
    window = (t // 32) * 32
    assert int(D.tokens[t]) == int(ids[at]) and values[at] == pytest.approx(top[0][1], rel=1e-3)
    assert t - at >= window and t - at + len(ids) <= window + 32, "a snippet stays inside its window"
    assert BOS not in ids.tolist(), "and inside its document"
    assert len(ids) <= B.LEFT + B.RIGHT + 1 and B.LEFT <= D.harvest.skip
    fires = D.fires_on(i, 4)
    assert fires == sorted(fires, key=lambda f: -f[1]) and sum(s for _, s in fires) <= 1 + 1e-9
    near = D.neighbors(i, 5)
    assert len(near) == 5 and i not in [j for j, _ in near] and near[0][1] >= near[-1][1]
    up, down = D.logits(i, 6)
    assert up[0][1] >= up[-1][1] and down[0][1] <= down[-1][1] and up[0][1] > down[0][1]
    assert "density" in D.text(i) and "[[" in D.line(i, t)
    # removal: the logits as they are minus the logits with the latent's term taken out, at its strongest firing
    effect = D.removal(i, examples=1)
    with torch.no_grad():
        idx = torch.from_numpy(np.asarray(D.tokens[window:window + 32]).astype(np.int64))[None]
        x = S.residual_after(D.model, idx, D.harvest.layer)
        gone = x.clone()
        gone[0, t - window] -= top[0][1] * D.sae.directions()[i]
        by_hand = (S.logits_from(D.model, x, 1) - S.logits_from(D.model, gone, 1))[0, t - window]
    assert np.allclose(effect, by_hand.numpy(), atol=1e-4)
    held_up, held_down = D.held(i, 5, examples=8)
    assert len(held_up) == 5 and held_up[0][1] >= held_up[-1][1] >= held_down[-1][1] >= held_down[0][1]
    p = D.profile()
    assert p["purity"][i] == pytest.approx(fires[0][1]) and p["token"][i] == fires[0][0] and 0 <= p["early"][i] <= 1 and p["burst"][i] >= 1


def test_the_browser_export_describes_itself(board, tmp_path):
    D, folder, _ = board
    from gptpp.export import feature_browser as FB

    record = json.loads((folder / "topk-n64-k4.json").read_text())
    alive = [int(i) for i in np.flatnonzero(D.count > 0)[:3]]
    curated = [{"id": alive[0], "name": "one", "kind": "token", "note": "a note"}, {"id": alive[1], "name": "two", "kind": "role"}]
    res = FB.export(D, record, {"label": "toy", "layers": 3, "width": 32, "after": 2, "dataset": "toy", "checkpoint": "ckpt_7.pt"}, curated, 20, 0, "feature-browser-toy", tmp_path)
    doc = json.loads((tmp_path / "feature-browser-toy.json").read_text())
    entries = json.loads((tmp_path / "feature-browser-toy.entries.json").read_text())
    rows, length = res["rows"], B.LEFT + B.RIGHT + 1
    assert entries["feature-browser-toy.tok"]["shape"] == [rows, length, 2] and entries["feature-browser-toy.act"]["shape"] == [rows, length]
    assert entries["feature-browser-toy.auto"]["shape"] == [64, 3] and (tmp_path / "feature-browser-toy.bin").stat().st_size == res["bin"]
    assert len(doc["lists"]["curated"]) == 2 and len(doc["lists"]["random"]) == 20 and len(doc["features"]) == 22
    assert [doc["features"][k]["name"] for k in doc["lists"]["curated"]] == ["one", "two"] and all("name" not in doc["features"][k] for k in doc["lists"]["random"])
    assert [f["id"] for f in doc["features"][2:]] == sorted(f["id"] for f in doc["features"][2:]), "the random list in the order of its numbers"
    f = doc["features"][0]
    assert f["top"][1] >= 1 and len(f["levels"]) == B.LEVELS and len(f["hist"]) == B.BINS and len(f["up"]) == 10
    assert all(0 <= v < len(doc["vocab"]) for v, _ in f["fires"] + f["up"] + f["down"] + f["held_up"] + f["held_down"])
    assert doc["removal_examples"] == FB.REMOVED and len(f["held_up"]) == 10 and f["held_up"][0][1] >= f["held_up"][-1][1] >= f["held_down"][0][1]
    again = FB.export(D, record, doc["model"], curated, 20, 0, "feature-browser-toy", tmp_path)
    assert again["rows"] == rows and json.loads((tmp_path / "feature-browser-toy.json").read_text())["lists"] == doc["lists"], "the same seed draws the same latents"
    index = FB.update_index(tmp_path / "feature-browser.json", "pico", "feature-browser-pico")
    assert index["default"] == "pico"
    index = FB.update_index(tmp_path / "feature-browser.json", "gpt-pp", "feature-browser-gptpp")
    assert index["order"] == ["gpt-pp", "pico"] and index["default"] == "gpt-pp", "gpt-pp is the default view once it is exported"


def test_the_family_export_keeps_the_other_model(board, tmp_path, monkeypatch):
    _, folder, root = board
    from gptpp.export import sparsity_against_fidelity as E

    run_dir = root / "ckpt" / "toy-run"
    config = json.loads((run_dir / "config.json").read_text())
    config.update(preset="gptpp-pico", model_config={"n_layer": 3, "d_model": 32})
    (run_dir / "config.json").write_text(json.dumps(config))
    entry = E.family(run_dir, 1, "topk-n64-k4")
    assert entry["label"] == "pico" and entry["after"] == 2 and entry["used"] == "topk-n64-k4" and len(entry["dicts"]) == 1
    assert set(E.KEEP) <= set(entry["dicts"][0]) and entry["loss_zero"] > entry["loss_clean"]
    with pytest.raises(KeyError):
        E.family(run_dir, 1, "topk-n64-k5")
    (tmp_path / "sparsity-against-fidelity.json").write_text(json.dumps({"models": {"gpt-pp": {"label": "gpt-pp"}}}))
    monkeypatch.setattr(sys, "argv", ["x", "--run", str(run_dir), "--layer", "1", "--used", "topk-n64-k4", "--out", str(tmp_path)])
    E.main()
    doc = json.loads((tmp_path / "sparsity-against-fidelity.json").read_text())
    assert doc["order"] == ["gpt-pp", "pico"] and doc["models"]["gpt-pp"] == {"label": "gpt-pp"} and doc["models"]["pico"]["used"] == "topk-n64-k4"


def test_the_python_copy_of_the_page_sampler_draws_the_page_numbers():
    from gptpp.export.steering_pico import page_sample, uniform

    # GPTModel.rng(seed).at(i), printed by the page's module in Node
    expected = {(3, 0): 0.792522791307, (3, 1): 0.023072841112, (3, 77): 0.200904787751, (0, 5): 0.844124919502, (299, 12): 0.338563265512}
    for (seed, i), u in expected.items():
        assert uniform(seed, i) == pytest.approx(u, abs=1e-12)
    logits = torch.tensor([[1.5, -0.25, 0.75, 3.0, 2.0]] * 5)
    assert page_sample(logits, 0.7, [0.01, 0.2, 0.5, 0.9, 0.999999]).tolist() == [0, 3, 3, 4, 4], "GPTModel.pick on the same distribution"
    assert page_sample(torch.tensor([[0.0, 50.0, 0.0]]), 0.7, [0.999999999]).tolist() == [1]
