"""Bits per byte and benchmark accuracies of released models: the yardsticks our own models are set beside.

For each model, one JSON file under ``results/eval/``, filled in by ``gptpp.evaluate`` (bits per byte on the
held-out sets) and ``gptpp.benchmarks`` (the log-likelihood benchmarks) under the same protocol as our own
checkpoints. A run adds what is missing: a held-out set or a task already in a model's file, computed on the same
documents with the same precision, is not computed again.

    python scripts/eval_released_models.py                         # every model, every set that exists, every task
    python scripts/eval_released_models.py --models gpt2,gpt2-xl --docs fineweb-edu --tasks none
    python scripts/eval_released_models.py --table                 # the tables, from every JSON in results/eval/
    python scripts/eval_released_models.py --compare gpt2-medium gpt2    # one model against another, paired

Three checks have their own switch: `--padding-check` (padding does not change a score), `--tinystories-overlap`
(which held-out stories a model trained on the first version of TinyStories has read) and `--wikitext-detokenized`
(GPT-2 on WikiText-103 as its paper measured it, beside the published numbers).

What is recorded about each model's training data is what its model card on the Hugging Face Hub says (read on
2026-11-14); `seen` says, for each held-out set, whether the model can have read it during training.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# Python puts a script's own directory first on the module path, and this directory holds a file named queue.py,
# which would then be imported in place of the standard library's `queue` (by torch, by the thread pool). Put the
# repository first and this directory last, before anything else is imported.
sys.path[:] = [str(REPO)] + [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent] + [
    str(Path(__file__).resolve().parent)]

import torch  # noqa: E402

from gptpp import benchmarks, evaluate, work_dir  # noqa: E402

# The released models, in the order they are run: the four sizes of GPT-2 first.
MODELS: list[dict] = [
    {
        "id": "gpt2", "hub_id": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
        "label": "GPT-2 small", "year": 2019,
        "paper": "Radford et al., Language models are unsupervised multitask learners, 2019",
        "training_data": "WebText: 40 GB of text from the web pages linked from Reddit posts with at least 3 karma; "
                         "all Wikipedia pages removed; never released. Sequences of 1,024 tokens; the amount of "
                         "training was not disclosed.",
        "seen": {"wikitext103": "no (Wikipedia was removed from WebText)", "tinystories": "no (written in 2023)",
                 "fineweb-edu": "possible (web pages of 2017 and earlier)", "fresh": "no"},
    },
    {
        "id": "gpt2-medium", "hub_id": "openai-community/gpt2-medium",
        "revision": "6dcaa7a952f72f9298047fd5137cd6e4f05f41da", "label": "GPT-2 medium", "year": 2019,
        "paper": "Radford et al., Language models are unsupervised multitask learners, 2019",
        "training_data": "WebText, as GPT-2 small.",
        "seen": {"wikitext103": "no (Wikipedia was removed from WebText)", "tinystories": "no (written in 2023)",
                 "fineweb-edu": "possible (web pages of 2017 and earlier)", "fresh": "no"},
    },
    {
        "id": "gpt2-large", "hub_id": "openai-community/gpt2-large",
        "revision": "32b71b12589c2f8d625668d2335a01cac3249519", "label": "GPT-2 large", "year": 2019,
        "paper": "Radford et al., Language models are unsupervised multitask learners, 2019",
        "training_data": "WebText, as GPT-2 small.",
        "seen": {"wikitext103": "no (Wikipedia was removed from WebText)", "tinystories": "no (written in 2023)",
                 "fineweb-edu": "possible (web pages of 2017 and earlier)", "fresh": "no"},
    },
    {
        "id": "gpt2-xl", "hub_id": "openai-community/gpt2-xl", "revision": "15ea56dee5df4983c59b2538573817e1667135e2",
        "label": "GPT-2 XL", "year": 2019,
        "paper": "Radford et al., Language models are unsupervised multitask learners, 2019",
        "training_data": "WebText, as GPT-2 small.",
        "seen": {"wikitext103": "no (Wikipedia was removed from WebText)", "tinystories": "no (written in 2023)",
                 "fineweb-edu": "possible (web pages of 2017 and earlier)", "fresh": "no"},
    },
    {
        "id": "distilgpt2", "hub_id": "distilbert/distilgpt2", "revision": "2290a62682d06624634c1f46a6ad5be0f47f38aa",
        "label": "DistilGPT2", "year": 2019,
        "paper": "Sanh et al., DistilBERT, a distilled version of BERT, 2019, arXiv:1910.01108",
        "training_data": "OpenWebText, an open reproduction of WebText, with GPT-2 small as the teacher (knowledge "
                         "distillation).",
        "seen": {"wikitext103": "not stated", "tinystories": "no (written in 2023)",
                 "fineweb-edu": "possible (web pages)", "fresh": "no"},
    },
    {
        "id": "pythia-160m", "hub_id": "EleutherAI/pythia-160m", "revision": "50f5173d932e8e61f858120bcb800b97af589f46",
        "label": "Pythia 160M", "year": 2023,
        "paper": "Biderman et al., Pythia: a suite for analyzing large language models across training and scaling, "
                 "2023, arXiv:2304.01373",
        "training_data": "The Pile (825 GiB of English text from 22 sources), not deduplicated: 143,000 steps of "
                         "2,097,152 tokens, about 300 billion tokens, just under one epoch.",
        "seen": {"wikitext103": "possible (the Pile's 22 sources are listed in Gao et al. 2020, arXiv:2101.00027)",
                 "tinystories": "no (the Pile is of 2020)", "fineweb-edu": "possible (web pages)", "fresh": "no"},
    },
    {
        "id": "smollm2-135m", "hub_id": "HuggingFaceTB/SmolLM2-135M",
        "revision": "93efa2f097d58c2a74874c7e644dbc9b0cee75a2", "label": "SmolLM2 135M", "year": 2025,
        "paper": "Allal et al., SmolLM2: when smol goes big, 2025, arXiv:2502.02737",
        "training_data": "2 trillion tokens of FineWeb-Edu, DCLM, The Stack and further filtered datasets.",
        "seen": {"wikitext103": "possible", "tinystories": "possible",
                 "fineweb-edu": "yes (FineWeb-Edu is in its training data; our held-out documents are held out of "
                                "our training set, not of its)", "fresh": "no"},
    },
    {
        "id": "tinystories-33m", "hub_id": "roneneldan/TinyStories-33M",
        "revision": "2ad0a164221b7c4d21cac7c46aec74f6f98dbfc8", "label": "TinyStories 33M", "year": 2023,
        "paper": "Eldan and Li, TinyStories: how small can language models be and still speak coherent English?, "
                 "2023, arXiv:2305.07759",
        "training_data": "TinyStories (training split) only, with a context of 512 tokens.",
        # its configuration allows 2,048 positions, but it was trained on 512 (the model card's hyperparameters)
        "context": 512,
        "seen": {"wikitext103": "no", "tinystories": "the validation split is held out; the training split is its "
                 "only data", "fineweb-edu": "no", "fresh": "no"},
    },
    {
        "id": "smollm2-360m", "hub_id": "HuggingFaceTB/SmolLM2-360M",
        "revision": "f8027fd0eaeea54caa13c31d31b9fdc459c38b49", "label": "SmolLM2 360M", "year": 2025,
        "paper": "Allal et al., SmolLM2: when smol goes big, 2025, arXiv:2502.02737",
        "training_data": "4 trillion tokens of FineWeb-Edu, DCLM, The Stack and further filtered datasets.",
        "seen": {"wikitext103": "possible", "tinystories": "possible",
                 "fineweb-edu": "yes (FineWeb-Edu is in its training data; our held-out documents are held out of "
                                "our training set, not of its)", "fresh": "no"},
    },
    {
        "id": "qwen3-0.6b-base", "hub_id": "Qwen/Qwen3-0.6B-Base", "revision": "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
        "label": "Qwen3 0.6B base", "year": 2025,
        "paper": "Yang et al. (Qwen Team), Qwen3 technical report, 2025, arXiv:2505.09388",
        "training_data": "The Qwen3 family is pre-trained on 36 trillion tokens in 119 languages (web, code, books, "
                         "synthetic data); what the 0.6B model itself saw is not stated.",
        "seen": {"wikitext103": "possible", "tinystories": "possible", "fineweb-edu": "possible", "fresh": "no"},
    },
    {
        "id": "gemma-3-270m", "hub_id": "google/gemma-3-270m", "revision": "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1",
        "label": "Gemma 3 270M", "year": 2025,
        "paper": "Gemma 3 model card (the repository is gated: its license must be accepted on the Hub)",
        "training_data": "not read: the model card is behind the license gate.",
        "seen": {"wikitext103": "possible", "tinystories": "possible", "fineweb-edu": "possible", "fresh": "no"},
    },
]
BY_ID = {m["id"]: m for m in MODELS}
RESULTS = REPO / "results" / "eval"


def available_docs() -> list[str]:
    """The held-out sets that exist on this machine now (the local ones are written by other scripts)."""
    names = []
    for name, spec in evaluate.HELD_OUT.items():
        if "hub" in spec or (work_dir() / spec["local"]).exists():
            names.append(name)
    return names


def run_model(entry: dict, doc_sets: dict[str, tuple[list[str], dict]], tasks: list[str],
              args: argparse.Namespace) -> float:
    """Evaluate one released model on what its JSON file lacks. Returns the seconds spent with the model loaded."""
    out = Path(args.out_dir) / f"{entry['id']}.json"
    record = json.loads(out.read_text()) if out.exists() else {}
    docs = list(doc_sets)

    def fresh_enough(result: dict | None) -> bool:
        return result is not None and result.get("precision") == args.precision and not args.force
    todo_docs = [name for name in docs
                 if not (fresh_enough(record.get("bits_per_byte", {}).get(name))
                         and record["bits_per_byte"][name]["docs"]["sha256"] == doc_sets[name][1]["sha256"])]
    todo_tasks = [name for name in tasks
                  if not (fresh_enough(record.get("benchmarks", {}).get(name))
                          and record["benchmarks"][name].get("limit") == args.limit)]
    if not todo_docs and not todo_tasks:
        print(f"{entry['id']}: nothing to do", flush=True)
        return 0.0

    t0 = time.time()
    model = evaluate.load_model(f"hf:{entry['hub_id']}@{entry['revision']}", device=args.device,
                                precision=args.precision, context=entry.get("context"))
    model.info.update({key: entry[key] for key in ("label", "year", "paper", "training_data", "seen")})
    model.info["padding_error_nats_per_token"] = evaluate.padding_error(model)
    print(f"{entry['id']}: {model.info['n_params'] / 1e6:.1f}M parameters, window {model.window}, "
          f"padding error {model.info['padding_error_nats_per_token']:.1e}, loaded in {time.time() - t0:.0f}s", flush=True)

    for name in todo_docs:
        texts, source = doc_sets[name]
        result = evaluate.bits_per_byte(model, texts, progress=f"{entry['id']} {name}")
        result["docs"] = source
        evaluate.merge_result(out, model.info, "bits_per_byte", name, result)
        print(f"  {name:12s} {result['bits_per_byte']:.4f} bits/byte (se {result['bits_per_byte_se'] or 0:.4f})  "
              f"loss {result['loss']:.4f}  {result['bytes_per_token']:.3f} bytes/token  {result['seconds']:.0f}s",
              flush=True)
    for name in todo_tasks:
        result = benchmarks.run_task(model, name, limit=args.limit)
        evaluate.merge_result(out, model.info, "benchmarks", name, result)
        print(f"  {name:15s} {100 * result['accuracy']:6.2f}% (se {100 * result['stderr']:.2f}, chance "
              f"{100 * result['chance']:.0f}%, n {result['n']})  {result['seconds']:.0f}s", flush=True)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return time.time() - t0


def check_padding(entry: dict, args: argparse.Namespace) -> None:
    """Measure again, and record, how much padding changes a model's scores (see `evaluate.padding_error`)."""
    out = Path(args.out_dir) / f"{entry['id']}.json"
    if not out.exists():
        return
    model = evaluate.load_model(f"hf:{entry['hub_id']}@{entry['revision']}", device=args.device, precision="float32",
                                context=entry.get("context"))
    record = json.loads(out.read_text())
    record["model"]["padding_error_nats_per_token"] = evaluate.padding_error(model)
    out.write_text(evaluate.dump_json(record) + "\n")
    print(f"{entry['id']}: padding error {record['model']['padding_error_nats_per_token']:.1e} nats a token", flush=True)


def tinystories_overlap(args: argparse.Namespace) -> None:
    """Which of our TinyStories held-out stories a model trained on the dataset's first version has read.

    Our stories are the validation split of the second version of TinyStories; TinyStories-33M was trained on the
    training split of the first. The two versions share stories, so some of what is held out for us is training
    data for it. This looks every held-out story up in the first version's training file (1.9 GB, fetched from
    the Hub) and records the ones found, so that scores can be compared on the stories it has not seen.
    """
    import re

    from huggingface_hub import hf_hub_download

    def normal(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    texts, source = evaluate.load_docs("tinystories", args.max_docs)
    path = hf_hub_download("roneneldan/TinyStories", "TinyStories-train.txt", repo_type="dataset",
                           revision="f54c09fd23315a6f9c86f9dc80f725de7d8f9c64")
    with open(path, encoding="utf-8") as f:
        train = {normal(story) for story in f.read().split("<|endoftext|>")}
    seen = [i for i, text in enumerate(texts) if normal(text) in train]
    record = {"docs": source, "first_version_training_file": "roneneldan/TinyStories TinyStories-train.txt",
              "n_training_stories_distinct": len(train), "n_held_out": len(texts), "n_also_in_training_file": len(seen),
              "indices_also_in_training_file": seen}
    out = Path(args.out_dir) / "tinystories_overlap.json"
    out.write_text(evaluate.dump_json(record) + "\n")
    print(f"{len(seen)} of {len(texts)} held-out stories are in the first version's training file -> {out}")


# ---------------------------------------------------------------------------------------------------------------
# Tables.
# ---------------------------------------------------------------------------------------------------------------


def load_results(directory: Path = RESULTS) -> list[dict]:
    """Every model's record in the directory: the released models in the order above, then the others (ours)."""
    records = []
    for path in sorted(directory.glob("*.json")):
        record = json.loads(path.read_text())
        if "model" in record and ("bits_per_byte" in record or "benchmarks" in record):
            record["file"] = path.stem
            records.append(record)
    order = {m["id"]: i for i, m in enumerate(MODELS)}
    return sorted(records, key=lambda r: (order.get(r["file"], len(order)), r["file"]))


def centered_mean(record: dict, tasks: list[str]) -> float | None:
    """Mean over the tasks of (accuracy - chance) / (1 - chance): 0 is guessing and 1 is perfect on every task."""
    done = record.get("benchmarks", {})
    if any(t not in done or done[t].get("limit") is not None for t in tasks):
        return None
    return sum((done[t]["accuracy"] - done[t]["chance"]) / (1 - done[t]["chance"]) for t in tasks) / len(tasks)


def tables(records: list[dict]) -> str:
    """The two tables in Markdown: bits per byte by held-out set, accuracy by benchmark. Our models in bold."""
    def name(r: dict) -> str:
        label = r["model"].get("label", r["model"]["name"])
        return f"**{label}**" if r["model"].get("kind") == "gptpp" else label

    def params(r: dict) -> str:
        return f"{r['model']['n_params'] / 1e6:.0f}M"

    sets = [s for s in evaluate.HELD_OUT if any(s in r.get("bits_per_byte", {}) for r in records)]
    lines = ["| model | parameters | window | " + " | ".join(sets) + " |", "|:--|--:|--:|" + "--:|" * len(sets)]
    fingerprints: dict[str, set] = {s: set() for s in sets}
    for r in records:
        cells = []
        for s in sets:
            b = r.get("bits_per_byte", {}).get(s)
            cells.append(f"{b['bits_per_byte']:.3f} ± {b['bits_per_byte_se']:.3f}" if b else "")
            if b:
                fingerprints[s].add(b["docs"]["sha256"])
        lines.append(f"| {name(r)} | {params(r)} | {r['model']['window']} | " + " | ".join(cells) + " |")
    mixed = [s for s, f in fingerprints.items() if len(f) > 1]
    if mixed:
        lines.append(f"\nWARNING: not every model saw the same documents of: {', '.join(mixed)}.")

    tasks = [t for t in benchmarks.TASKS if any(t in r.get("benchmarks", {}) for r in records)]
    lines += ["", "| model | " + " | ".join(tasks) + " | centered mean |", "|:--|" + "--:|" * (len(tasks) + 1)]
    chance = {t: next(r["benchmarks"][t]["chance"] for r in records if t in r.get("benchmarks", {})) for t in tasks}
    lines.append("| *chance* | " + " | ".join(f"*{100 * chance[t]:.1f}*" for t in tasks) + " | *0.0* |")
    for r in records:
        cells = []
        for t in tasks:
            b = r.get("benchmarks", {}).get(t)
            cells.append(f"{100 * b['accuracy']:.1f} ± {100 * b['stderr']:.1f}" + ("*" if b.get("limit") else "")
                         if b else "")
        mean = centered_mean(r, tasks)
        lines.append(f"| {name(r)} | " + " | ".join(cells) + f" | {'' if mean is None else f'{100 * mean:.1f}'} |")
    return "\n".join(lines)


def summary(records: list[dict]) -> dict:
    """Everything the tables show, as one small JSON (for the figures of the site)."""
    sets = [s for s in evaluate.HELD_OUT if any(s in r.get("bits_per_byte", {}) for r in records)]
    tasks = [t for t in benchmarks.TASKS if any(t in r.get("benchmarks", {}) for r in records)]
    first = lambda section, key: next(r[section][key] for r in records if key in r.get(section, {}))
    return {
        "held_out": {s: {key: first("bits_per_byte", s)[key] if key in ("n_docs", "n_bytes") else
                         first("bits_per_byte", s)["docs"][key] for key in ("what", "sha256", "n_docs", "n_bytes")}
                     for s in sets},
        "tasks": {t: {key: first("benchmarks", t)[key] for key in ("what", "n", "chance", "headline")} for t in tasks},
        "models": [{
            "file": r["file"], "label": r["model"].get("label", r["model"]["name"]), "kind": r["model"]["kind"],
            "year": r["model"].get("year"), "n_params": r["model"]["n_params"],
            "n_params_non_embedding": r["model"].get("n_params_non_embedding"),
            "n_layer": r["model"].get("n_layer"), "d_model": r["model"].get("d_model"), "window": r["model"]["window"],
            "bits_per_byte": {s: [b["bits_per_byte"], b["bits_per_byte_se"], b["bytes_per_token"]]
                              for s in sets if (b := r.get("bits_per_byte", {}).get(s))},
            "accuracy": {t: [b["accuracy"], b["stderr"]] for t in tasks
                         if (b := r.get("benchmarks", {}).get(t)) and b.get("limit") is None},
            "centered_mean": centered_mean(r, tasks),
        } for r in records],
    }


def compare(a: dict, b: dict) -> str:
    """Model a against model b on the same documents and the same items: differences with paired standard errors.

    The two models read the same documents, so a hard document is hard for both and cancels in the difference;
    the standard error of the difference is far smaller than the two separate ones suggest.
    """
    import numpy as np

    lines = [f"{a['model'].get('label', a['file'])} minus {b['model'].get('label', b['file'])}", ""]
    for name, ra in a.get("bits_per_byte", {}).items():
        rb = b.get("bits_per_byte", {}).get(name)
        if rb is None or rb["docs"]["sha256"] != ra["docs"]["sha256"]:
            continue
        nbytes = np.array(ra["doc_bytes"], dtype=np.float64)
        diff = np.array(ra["doc_bits"]) - np.array(rb["doc_bits"])
        se = evaluate.ratio_standard_error(diff, nbytes)
        better = float(np.mean(diff < 0))
        lines.append(f"  {name:14s} {diff.sum() / nbytes.sum():+.4f} bits/byte (se {se:.4f}); the first is better on "
                     f"{100 * better:.0f}% of the {len(diff)} documents")
    for name, ra in a.get("benchmarks", {}).items():
        rb = b.get("benchmarks", {}).get(name)
        if rb is None or len(ra["correct"]) != len(rb["correct"]) or ra.get("limit") != rb.get("limit"):
            continue
        diff = np.array([int(x) - int(y) for x, y in zip(ra["correct"], rb["correct"])], dtype=np.float64)
        lines.append(f"  {name:14s} {100 * diff.mean():+.2f} points (se {100 * diff.std(ddof=1) / np.sqrt(len(diff)):.2f}); "
                     f"only the first right on {int((diff > 0).sum())} items, only the second on {int((diff < 0).sum())}")
    return "\n".join(lines)


# Published zero-shot numbers for the four sizes of GPT-2, to set ours beside.
# WikiText-103 perplexity a word and LAMBADA perplexity of the last word: Radford et al. 2019, table 3 (computed on
# de-tokenized text, which the paper says is worth 2.5 to 5 perplexity). HellaSwag: the two figures of EleutherAI's
# lm-evaluation-harness as quoted in the header of hellaswag.py in github.com/karpathy/build-nanogpt.
PUBLISHED = {
    "gpt2": {"wikitext103_word_ppl": 37.50, "lambada_ppl": 35.13, "hellaswag_acc": 28.92, "hellaswag_acc_norm": 31.14},
    "gpt2-medium": {"wikitext103_word_ppl": 26.37, "lambada_ppl": 15.60},
    "gpt2-large": {"wikitext103_word_ppl": 22.05, "lambada_ppl": 10.87},
    "gpt2-xl": {"wikitext103_word_ppl": 17.48, "lambada_ppl": 8.63, "hellaswag_acc": 40.04, "hellaswag_acc_norm": 50.89},
}
WIKITEXT103_TEST_TOKENS = 245_569  # the benchmark's own count: words and line ends (Merity et al. 2016, table 1)


def detokenize_wikitext(text: str) -> str:
    """Undo the tokenization WikiText is distributed in (our own rules): the marked hyphens and number separators,
    the space before punctuation, the spaces inside brackets and quotation marks, the split contractions."""
    import re

    text = text.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    text = re.sub(r" ([.,;:!?%])(?=\s|$)", r"\1", text)
    text = re.sub(r"\( (.*?) \)", r"(\1)", text)
    text = re.sub(r"\[ (.*?) \]", r"[\1]", text)
    text = re.sub(r'" (.*?) "', r'"\1"', text)
    text = re.sub(r" (n't|'s|'m|'re|'ve|'ll|'d)\b", r"\1", text)
    text = text.replace("s ' ", "s' ").replace("$ ", "$").replace("# ", "#").replace(" °", "°")
    return re.sub(r"(\d) : (\d)", r"\1:\2", text)


def wikitext_detokenized(args: argparse.Namespace) -> None:
    """GPT-2 on WikiText-103 the way its paper measured it: on de-tokenized text.

    Our held-out set is the raw text, as distributed; the published perplexities were computed after
    "invertible de-tokenizers" had removed the tokenization artifacts. This scores the four sizes on the test set
    de-tokenized by `detokenize_wikitext`, under our protocol otherwise, and records the perplexity per unit of
    the benchmark's own count beside the published one.
    """
    import math

    texts = [detokenize_wikitext(d) for d in evaluate.load_docs("wikitext103")[0]]
    record = {"what": "WikiText-103 test, de-tokenized by our own rules; perplexity per benchmark token",
              "units": WIKITEXT103_TEST_TOKENS, "precision": args.precision, "models": {}}
    for name, pub in PUBLISHED.items():
        entry = BY_ID[name]
        model = evaluate.load_model(f"hf:{entry['hub_id']}@{entry['revision']}", device=args.device,
                                    precision=args.precision)
        result = evaluate.bits_per_byte(model, texts)
        record["models"][name] = {
            "word_perplexity": math.exp(result["total_bits"] * math.log(2) / WIKITEXT103_TEST_TOKENS),
            "published": pub["wikitext103_word_ppl"], "bits_per_byte": result["bits_per_byte"]}
        print(f"{name}: {record['models'][name]['word_perplexity']:.2f} a word, published {pub['wikitext103_word_ppl']}",
              flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    (Path(args.out_dir) / "wikitext103_detokenized.json").write_text(evaluate.dump_json(record) + "\n")


def published_checks(records: list[dict], directory: Path = RESULTS) -> str:
    """Ours beside the published numbers, for the models that have them."""
    import math

    path = directory / "wikitext103_detokenized.json"
    detokenized = json.loads(path.read_text())["models"] if path.exists() else {}
    lines = ["| model | WikiText-103 word perplexity: published, ours on raw text, ours on de-tokenized text "
             "| LAMBADA last-word perplexity: published, ours | HellaSwag acc: harness, ours "
             "| HellaSwag acc_norm: harness, ours |", "|:--|--:|--:|--:|--:|"]
    for r in records:
        pub = PUBLISHED.get(r["file"])
        bpb, bench = r.get("bits_per_byte", {}), r.get("benchmarks", {})
        if not pub or "wikitext103" not in bpb or "lambada" not in bench or "hellaswag" not in bench:
            continue
        word_ppl = math.exp(bpb["wikitext103"]["total_bits"] * math.log(2) / WIKITEXT103_TEST_TOKENS)
        hs = bench["hellaswag"]
        cell = lambda key, ours: f"{pub[key]:.2f}, {ours:.2f}" if key in pub else f", {ours:.2f}"
        clean = f", {detokenized[r['file']]['word_perplexity']:.2f}" if r["file"] in detokenized else ""
        lines.append(f"| {r['model'].get('label', r['file'])} | {cell('wikitext103_word_ppl', word_ppl)}{clean} | "
                     f"{cell('lambada_ppl', bench['lambada']['word_perplexity'])} | "
                     f"{cell('hellaswag_acc', 100 * hs['acc'])} | {cell('hellaswag_acc_norm', 100 * hs['acc_norm'])} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="all", help=f"comma-separated ids, or all: {', '.join(BY_ID)}")
    ap.add_argument("--docs", default="available", help="comma-separated held-out sets, available, or none")
    ap.add_argument("--tasks", default="all", help="comma-separated benchmarks, all, or none")
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="a fixed subset of N items a task (for tests)")
    ap.add_argument("--precision", default="tf32", choices=["float32", "tf32", "bf16"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--force", action="store_true", help="compute again what the files already hold")
    ap.add_argument("--budget-minutes", type=float, default=None, help="start no further model after this long")
    ap.add_argument("--out-dir", default=str(RESULTS), help="where the JSON files are (default: results/eval)")
    ap.add_argument("--table", action="store_true", help="print the tables of everything in the directory and exit")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"),
                    help="two result files (names without .json): A minus B with paired standard errors")
    ap.add_argument("--wikitext-detokenized", action="store_true",
                    help="only score the four GPT-2 sizes on de-tokenized WikiText-103, beside the published numbers")
    ap.add_argument("--tinystories-overlap", action="store_true",
                    help="only record which TinyStories held-out stories are in the first version's training file")
    ap.add_argument("--padding-check", action="store_true",
                    help="only measure again how much padding changes each model's scores (cheap; --device cpu will do)")
    args = ap.parse_args()

    if args.table:
        records = load_results(Path(args.out_dir))
        print(tables(records) + "\n\n" + published_checks(records, Path(args.out_dir)))
        (Path(args.out_dir) / "summary.json").write_text(evaluate.dump_json(summary(records)) + "\n")
        return

    if args.compare:
        by_file = {r["file"]: r for r in load_results(Path(args.out_dir))}
        print(compare(by_file[args.compare[0]], by_file[args.compare[1]]))
        return
    if args.wikitext_detokenized:
        wikitext_detokenized(args)
        return
    if args.tinystories_overlap:
        tinystories_overlap(args)
        return
    entries = MODELS if args.models == "all" else [BY_ID[m] for m in args.models.split(",")]
    if args.padding_check:
        for entry in entries:
            check_padding(entry, args)
        return
    docs = {"available": available_docs(), "none": []}.get(args.docs, args.docs.split(","))
    tasks = {"all": list(benchmarks.TASKS), "none": []}.get(args.tasks, args.tasks.split(","))
    doc_sets = {name: evaluate.load_docs(name, args.max_docs) for name in docs}  # once: every model, same documents
    for name, (texts, source) in doc_sets.items():
        print(f"held-out set {name}: {len(texts)} documents, {sum(len(t.encode('utf-8')) for t in texts)} bytes, "
              f"fingerprint {source['sha256']}", flush=True)
    print(f"benchmarks: {tasks}\nprecision: {args.precision}", flush=True)
    if "HF_TOKEN" not in os.environ:
        print("HF_TOKEN is not set: gated repositories will fail", flush=True)

    spent, failed = 0.0, []
    for entry in entries:
        if args.budget_minutes is not None and spent > 60 * args.budget_minutes:
            print(f"budget of {args.budget_minutes} minutes used: {entry['id']} and later models are left", flush=True)
            break
        try:
            seconds = run_model(entry, doc_sets, tasks, args)
        except Exception as error:  # one model that cannot be loaded (a gated repository) must not stop the rest
            failed.append(entry["id"])
            print(f"{entry['id']}: FAILED: {type(error).__name__}: {str(error).splitlines()[0][:300]}", flush=True)
            traceback.print_exc(limit=3)
            continue
        spent += seconds
        print(f"{entry['id']}: {seconds:.0f}s; {spent / 60:.1f} minutes in all", flush=True)
    print(f"done in {spent / 60:.1f} minutes; failed: {failed or 'none'}", flush=True)


if __name__ == "__main__":
    main()
