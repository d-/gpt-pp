"""Shared fixtures of the core tests: the import path, a byte-level Shakespeare dataset, the configuration grid.

    cd gpt-pp && $PY -m pytest -q tests/test_model.py tests/test_optim.py tests/test_train_smoke.py

The model and optimizer tests run on the CPU. The trainer tests use the first visible GPU if there is one (set
CUDA_VISIBLE_DEVICES to choose), and the distributed test needs two; without a GPU they run on the CPU (slowly at
the larger sizes) and the tests that need one are skipped.
"""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gptpp import work_dir  # noqa: E402
from gptpp.config import ModelConfig  # noqa: E402
from gptpp.data import write_byte_dataset  # noqa: E402

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
SHAKESPEARE_PLACES = (
    os.environ.get("GPTPP_SHAKESPEARE", ""),
    "/home/ubuntu/nanoGPT/data/shakespeare_char/input.txt",
    str(work_dir() / "data" / "fixtures" / "tinyshakespeare.txt"),
)


@pytest.fixture(autouse=True)
def exact_float32_products():
    """The trainer, the evaluator and the chat stage switch TF32 on (float32 matrix products with a 10-bit mantissa)
    for the rest of the process. A test that runs one of them in this process would then loosen every later test
    that compares two float32 results on a GPU: start each test with TF32 off."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    yield


def shakespeare_file() -> Path:
    """The tiny Shakespeare text (1.1 MB): $GPTPP_SHAKESPEARE, nanoGPT's copy, or the workspace's cached copy,
    which is downloaded the first time."""
    for place in SHAKESPEARE_PLACES:
        if place and Path(place).is_file():
            return Path(place)
    cache = Path(SHAKESPEARE_PLACES[-1])
    cache.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(SHAKESPEARE_URL, timeout=60) as response:
            text = response.read()
    except OSError as error:
        pytest.fail(f"the Shakespeare text is not at {SHAKESPEARE_PLACES[1]} and could not be downloaded "
                    f"({error}); set GPTPP_SHAKESPEARE to a text file")
    cache.write_bytes(text)
    return cache


CORE_TESTS = ("test_model", "test_optim", "test_train_smoke")


@pytest.fixture(scope="module", autouse=True)
def single_thread(request: pytest.FixtureRequest):
    """The core tests run tiny tensors on the CPU, for which waking a thread pool on a many-core machine costs
    far more than the arithmetic: one thread for those modules, the previous setting for all others. On a
    machine without a free GPU the trainer tests run on the CPU, where GPTPP_TEST_THREADS=16 helps them."""
    if request.module.__name__ not in CORE_TESTS:
        yield
        return
    threads = torch.get_num_threads()
    torch.set_num_threads(int(os.environ.get("GPTPP_TEST_THREADS", "1")))
    yield
    torch.set_num_threads(threads)


@pytest.fixture(scope="session")
def work(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A throwaway workspace laid out like $GPTPP_WORK, holding the dataset shakespeare/bytes."""
    root = tmp_path_factory.mktemp("work")
    write_byte_dataset(shakespeare_file(), root / "data" / "shakespeare" / "bytes")
    return root


@pytest.fixture(scope="session")
def byte_data(work: Path) -> Path:
    """Byte-level Shakespeare in the on-disk format of the lab: 256 byte tokens and BOS."""
    return work / "data" / "shakespeare" / "bytes"


def tiny(**changes: object) -> ModelConfig:
    """A model small enough for exact tests on the CPU: 2 layers, 32 wide, 4 heads, context 16, 50 tokens."""
    base = dict(vocab_size=50, n_layer=2, n_head=4, d_model=32, block_size=16)
    return ModelConfig().replace(**{**base, **changes})


# Every switch of the model, one at a time from the 2026 defaults, then the 2019 recipe and some combinations.
VARIANTS: dict[str, dict] = {
    "default": {},
    "pos=learned": dict(pos="learned"),
    "pos=sinusoidal": dict(pos="sinusoidal"),
    "pos=alibi": dict(pos="alibi"),
    "pos=none": dict(pos="none"),
    "norm=layernorm": dict(norm="layernorm"),
    "norm=layernorm+bias": dict(norm="layernorm", bias=True),
    "norm=rmsnorm": dict(norm="rmsnorm"),
    "norm=dyt": dict(norm="dyt"),
    "norm=none": dict(norm="none"),
    "post-norm": dict(norm_placement="post", norm="layernorm"),
    "post-norm,rmsnorm_fixed": dict(norm_placement="post"),
    "no embed_norm": dict(embed_norm=False),
    "act=relu": dict(act="relu"),
    "act=gelu": dict(act="gelu"),
    "act=silu": dict(act="silu"),
    "act=swiglu": dict(act="swiglu"),
    "act=geglu": dict(act="geglu"),
    "attention-only": dict(use_mlp=False),
    "no qk_norm": dict(qk_norm=False),
    "qk_scale learned": dict(qk_scale_learned=True, qk_scale=9.0),
    "gqa": dict(n_kv_head=2),
    "multi-query": dict(n_kv_head=1),
    "bias": dict(bias=True),
    "tied": dict(tie_embeddings=True),
    "no softcap": dict(logit_softcap=0.0),
    "init=gpt2": dict(init="gpt2"),
    "gpt2-2019": dict(pos="learned", norm="layernorm", embed_norm=False, act="gelu", qk_norm=False, bias=True,
                      tie_embeddings=True, logit_softcap=0.0, init="gpt2"),
    "alibi,gqa,no qk_norm": dict(pos="alibi", n_kv_head=2, qk_norm=False),
    "alibi,post-norm,swiglu": dict(pos="alibi", norm_placement="post", norm="rmsnorm", act="swiglu"),
    "sinusoidal,dyt,tied": dict(pos="sinusoidal", norm="dyt", tie_embeddings=True, bias=True),
}


@pytest.fixture(params=list(VARIANTS), ids=list(VARIANTS))
def variant(request: pytest.FixtureRequest) -> ModelConfig:
    """One configuration of the grid. Initialized with the GPT-2 scheme unless the variant says otherwise, so
    that every block contributes to the output (with zero-initialized projections the blocks are silent at
    initialization and a wrong block would go unnoticed)."""
    changes = VARIANTS[request.param]
    return tiny(**{"init": "gpt2", **changes})
