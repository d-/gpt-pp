"""Benchmarks a small language model can take by log-likelihood.

A model that only predicts the next token can still answer a multiple-choice question: write each option after
the question and ask which one the model finds most probable. Nothing is generated and nothing is parsed, so a
model of two million parameters and a model of two billion take exactly the same test, and our checkpoints and
Hugging Face models go through one code path (`gptpp.evaluate.score_sequences`).

Every item is a list of (given text, text to score) pairs and the index of the right one. The sequence scored
is ``[beginning-of-text] + tokens(given) + tokens(text to score)``, the two encoded separately, cut from the left
if it exceeds the model's window; the score is the sum of the log-probabilities of the second part. Three ways
of choosing are recorded:

- ``acc``: the option with the highest total log-probability;
- ``acc_norm``: the highest log-probability per UTF-8 byte of the option (a long option is not punished for its
  length, and the normalization does not depend on the tokenizer);
- ``acc_per_token``: the highest log-probability per token.

Each task names one of them as its headline, `accuracy`. Options that tie share the credit, so a model that cannot
tell the options apart scores exactly the chance level. The standard error is that of a mean over the items.

The item formats of HellaSwag, ARC, PIQA, SciQ, WinoGrande and LAMBADA are the ones in common use (those of
EleutherAI's lm-evaluation-harness, re-implemented here), so the numbers can be set beside published ones.

    python -m gptpp.benchmarks --model hf:gpt2 --tasks all --out results/eval/gpt2.json
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from gptpp.evaluate import LanguageModel, hub_parquet, load_model, merge_result, score_sequences


@dataclass
class Item:
    """One question: the candidate (given text, text to score) pairs and which of them is right."""

    pairs: list[tuple[str, str]]
    gold: int
    group: str = ""  # a subdivision of the task, reported separately ("phenomenon/paradigm" in BLiMP)


@dataclass
class Task:
    name: str
    what: str  # one line for tables and documentation
    hub_id: str
    revision: str  # the commit the files are read from
    files: list[str]
    build: Callable[[list[dict]], list[Item]]
    form: str  # the template, in words
    headline: str = "acc_norm"  # acc | acc_norm | acc_per_token | greedy
    cite: str = ""

    def items(self) -> list[Item]:
        rows = [row for f in self.files for row in hub_parquet(self.hub_id, self.revision, f)]
        return self.build(rows)


# ---------------------------------------------------------------------------------------------------------------
# The tasks.
# ---------------------------------------------------------------------------------------------------------------


def _hellaswag(rows: list[dict]) -> list[Item]:
    def clean(text: str) -> str:
        # The WikiHow half of HellaSwag carries markup: "[title]" starts a step, other brackets are tags.
        text = text.strip().replace(" [title]", ". ")
        text = re.sub(r"\[.*?\]", "", text)
        return text.replace("  ", " ")

    items = []
    for row in rows:
        given = clean(f"{row['activity_label']}: {row['ctx_a']} {row['ctx_b'].capitalize()}")
        source = row["source_id"].split("~")[0]  # activitynet (video captions) or wikihow
        items.append(Item([(given, " " + clean(e)) for e in row["endings"]], int(row["label"]), source))
    return items


def _blimp(rows: list[dict]) -> list[Item]:
    # No given text: the two sentences are scored whole. The acceptable one comes first.
    return [Item([("", row["sentence_good"]), ("", row["sentence_bad"])], 0, f"{row['linguistics_term']}/{row['UID']}")
            for row in rows]


def _arc(rows: list[dict]) -> list[Item]:
    return [Item([(f"Question: {row['question']}\nAnswer:", " " + text) for text in row["choices"]["text"]],
                 row["choices"]["label"].index(row["answerKey"])) for row in rows]


def _lambada(rows: list[dict]) -> list[Item]:
    # The passage without its last word, and the last word.
    items = []
    for row in rows:
        given, last = row["text"].rsplit(" ", 1)
        items.append(Item([(given, " " + last)], 0))
    return items


def _piqa(rows: list[dict]) -> list[Item]:
    return [Item([(f"Question: {row['goal']}\nAnswer:", " " + row[key]) for key in ("sol1", "sol2")], int(row["label"]))
            for row in rows]


def _winogrande(rows: list[dict]) -> list[Item]:
    # The blank is filled with each option and the *rest of the sentence* is scored: the two candidates then
    # differ only in what they are conditioned on (Trinh and Le 2018, "partial scoring").
    items = []
    for row in rows:
        before, after = row["sentence"].split("_", 1)
        rest = " " + after.strip()
        items.append(Item([(before + row["option1"], rest), (before + row["option2"], rest)], int(row["answer"]) - 1))
    return items


def _sciq(rows: list[dict]) -> list[Item]:
    items = []
    for row in rows:
        given = f"{row['support'].lstrip()}\nQuestion: {row['question']}\nAnswer:".lstrip()
        options = [row["distractor1"], row["distractor2"], row["distractor3"], row["correct_answer"]]
        items.append(Item([(given, " " + o) for o in options], 3))
    return items


def _commonsense_qa(rows: list[dict]) -> list[Item]:
    return [Item([(f"Question: {row['question']}\nAnswer:", " " + text) for text in row["choices"]["text"]],
                 row["choices"]["label"].index(row["answerKey"])) for row in rows]


_BLIMP_PARADIGMS = """adjunct_island anaphor_gender_agreement anaphor_number_agreement animate_subject_passive
animate_subject_trans causative complex_NP_island coordinate_structure_constraint_complex_left_branch
coordinate_structure_constraint_object_extraction determiner_noun_agreement_1 determiner_noun_agreement_2
determiner_noun_agreement_irregular_1 determiner_noun_agreement_irregular_2 determiner_noun_agreement_with_adj_2
determiner_noun_agreement_with_adj_irregular_1 determiner_noun_agreement_with_adj_irregular_2
determiner_noun_agreement_with_adjective_1 distractor_agreement_relational_noun distractor_agreement_relative_clause
drop_argument ellipsis_n_bar_1 ellipsis_n_bar_2 existential_there_object_raising existential_there_quantifiers_1
existential_there_quantifiers_2 existential_there_subject_raising expletive_it_object_raising inchoative intransitive
irregular_past_participle_adjectives irregular_past_participle_verbs irregular_plural_subject_verb_agreement_1
irregular_plural_subject_verb_agreement_2 left_branch_island_echo_question left_branch_island_simple_question
matrix_question_npi_licensor_present npi_present_1 npi_present_2 only_npi_licensor_present only_npi_scope passive_1
passive_2 principle_A_c_command principle_A_case_1 principle_A_case_2 principle_A_domain_1 principle_A_domain_2
principle_A_domain_3 principle_A_reconstruction regular_plural_subject_verb_agreement_1
regular_plural_subject_verb_agreement_2 sentential_negation_npi_licensor_present sentential_negation_npi_scope
sentential_subject_island superlative_quantifiers_1 superlative_quantifiers_2 tough_vs_raising_1 tough_vs_raising_2
transitive wh_island wh_questions_object_gap wh_questions_subject_gap wh_questions_subject_gap_long_distance
wh_vs_that_no_gap wh_vs_that_no_gap_long_distance wh_vs_that_with_gap wh_vs_that_with_gap_long_distance""".split()

_ARC_REVISION = "210d026faf9955653af8916fad021475a3f00453"

TASKS: dict[str, Task] = {t.name: t for t in [
    Task("hellaswag", "HellaSwag, validation: which of four endings continues a described scene",
         "Rowan/hellaswag", "218ec52e09a7e7462a5400043bb9a69a41d06b76", ["data/validation-00000-of-00001.parquet"],
         _hellaswag, '"<activity>: <context>" then " <ending>"',
         cite="Zellers et al., HellaSwag: can a machine really finish your sentence?, ACL 2019, arXiv:1905.07830"),
    Task("blimp", "BLiMP, all 67 paradigms: which sentence of a minimal pair is grammatical",
         "nyu-mll/blimp", "877fba0801ffb7cbd8c39c1ff314a46f053f6036",
         [f"{p}/train-00000-of-00001.parquet" for p in _BLIMP_PARADIGMS],
         _blimp, "each sentence scored whole, from the beginning-of-text token", headline="acc",
         cite="Warstadt et al., BLiMP: the benchmark of linguistic minimal pairs for English, TACL 2020, arXiv:1912.00582"),
    Task("arc_easy", "ARC-Easy, test: grade-school science questions, four options",
         "allenai/ai2_arc", _ARC_REVISION, ["ARC-Easy/test-00000-of-00001.parquet"],
         _arc, '"Question: <question>\\nAnswer:" then " <option>"',
         cite="Clark et al., Think you have solved question answering? Try ARC, the AI2 reasoning challenge, 2018, "
              "arXiv:1803.05457"),
    Task("arc_challenge", "ARC-Challenge, test: the science questions retrieval and co-occurrence methods fail",
         "allenai/ai2_arc", _ARC_REVISION, ["ARC-Challenge/test-00000-of-00001.parquet"],
         _arc, '"Question: <question>\\nAnswer:" then " <option>"',
         cite="Clark et al., Think you have solved question answering? Try ARC, the AI2 reasoning challenge, 2018, "
              "arXiv:1803.05457"),
    Task("lambada", "LAMBADA, test, OpenAI's detokenized copy: the last word of a passage of narrative",
         "EleutherAI/lambada_openai", "900124bf3b8235c6daf21033af9948b3f07346c4", ["default/test/default.parquet"],
         _lambada, "the passage without its last word, then the last word; right when every token of the word is "
         "the model's first choice", headline="greedy",
         cite="Paperno et al., The LAMBADA dataset: word prediction requiring a broad discourse context, ACL 2016, "
              "arXiv:1606.06031"),
    Task("piqa", "PIQA, validation: which of two ways of doing an everyday physical task works",
         "ybisk/piqa", "142c51238b3ca2bc61e9a075913871b8b600e8e1", ["plain_text/validation/0000.parquet"],
         _piqa, '"Question: <goal>\\nAnswer:" then " <solution>"',
         cite="Bisk et al., PIQA: reasoning about physical commonsense in natural language, AAAI 2020, arXiv:1911.11641"),
    Task("winogrande", "WinoGrande, validation: which of two names a blank refers to",
         "allenai/winogrande", "01e74176c63542e6b0bcb004dcdea22d94fb67b5",
         ["winogrande_xl/validation-00000-of-00001.parquet"],
         _winogrande, "the sentence up to the blank with an option filled in, then the rest of the sentence",
         headline="acc",
         cite="Sakaguchi et al., WinoGrande: an adversarial Winograd schema challenge at scale, 2019, "
              "arXiv:1907.10641"),
    Task("sciq", "SciQ, test: science exam questions with a supporting paragraph, four options",
         "allenai/sciq", "2c94ad3e1aafab77146f384e23536f97a4849815", ["data/test-00000-of-00001.parquet"],
         _sciq, '"<support>\\nQuestion: <question>\\nAnswer:" then " <option>"',
         cite="Welbl et al., Crowdsourcing multiple choice science questions, W-NUT 2017, arXiv:1707.06209"),
    Task("commonsense_qa", "CommonsenseQA, validation: everyday knowledge, five options",
         "tau/commonsense_qa", "94630fe30dad47192a8546eb75f094926d47e155", ["data/validation-00000-of-00001.parquet"],
         _commonsense_qa, '"Question: <question>\\nAnswer:" then " <option>"',
         cite="Talmor et al., CommonsenseQA: a question answering challenge targeting commonsense knowledge, NAACL "
              "2019, arXiv:1811.00937"),
]}


def fixed_subset(task: str, n_items: int, size: int) -> list[int]:
    """`size` item indices of a task, the same on every machine: the items whose hashes sort first.

    The hosted models (scripts/eval_openrouter.py) are asked exactly these items, and `--limit` uses them too.
    """
    key = lambda i: hashlib.sha256(f"{task}:{i}".encode()).digest()
    return sorted(sorted(range(n_items), key=key)[:size])


# ---------------------------------------------------------------------------------------------------------------
# Scoring.
# ---------------------------------------------------------------------------------------------------------------


def _credit(scores: np.ndarray, gold: int) -> float:
    """1 if the right option has the best score; options tied for the best share the credit."""
    best = scores == scores.max()
    return float(best[gold]) / float(best.sum())


def _mean_and_se(values: np.ndarray) -> tuple[float, float]:
    n = len(values)
    return float(values.mean()), float(values.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0


def run_task(model: LanguageModel, name: str, limit: int | None = None, batch_tokens: int | None = None) -> dict:
    """Score one task under `model`. `limit` keeps a fixed subset of the items (for tests)."""
    t0 = time.time()
    task = TASKS[name]
    items = task.items()
    if limit is not None and limit < len(items):
        items = [items[i] for i in fixed_subset(name, len(items), limit)]

    # Encode every distinct string once.
    strings = sorted({text for item in items for pair in item.pairs for text in pair if text})
    tokens = dict(zip(strings, model.encode(strings)))
    room = model.window - 1  # places after the beginning-of-text token
    sequences, starts, n_cut = [], [], 0
    for item in items:
        for given, scored in item.pairs:
            target = tokens[scored]
            body = (tokens[given] if given else []) + target
            n_cut += len(body) > room
            body = body[-room:]  # a context too long for the window loses its beginning
            sequences.append([model.bos_id] + body)
            starts.append(max(1, len(body) + 1 - len(target)))
    scores = score_sequences(model, sequences, starts, batch_tokens=batch_tokens,
                             want_greedy=task.headline == "greedy", progress=f"{model.name} {name}")

    # Credit each item under the three rules.
    credit = {"acc": [], "acc_norm": [], "acc_per_token": []}
    k = 0
    for item in items:
        m = len(item.pairs)
        loglik = -scores.nll[k:k + m]
        n_bytes = np.array([len(scored.encode("utf-8")) for _, scored in item.pairs], dtype=np.float64)
        credit["acc"].append(_credit(loglik, item.gold))
        credit["acc_norm"].append(_credit(loglik / n_bytes, item.gold))
        credit["acc_per_token"].append(_credit(loglik / scores.n_scored[k:k + m], item.gold))
        k += m
    result: dict = {"task": name, "what": task.what, "form": task.form, "cite": task.cite,
                    "source": {"hub_id": task.hub_id, "revision": task.revision, "files": _summarize_files(task.files)},
                    "n": len(items), "chance": float(np.mean([1.0 / len(item.pairs) for item in items])),
                    "headline": task.headline}
    for rule, values in credit.items():
        result[rule], result[rule + "_se"] = _mean_and_se(np.array(values))

    if task.headline == "greedy":
        # One candidate an item (the true last word): right when the model's first choice at every token is it.
        correct = scores.greedy.astype(np.float64)
        result = {key: value for key, value in result.items() if not key.startswith("acc")}
        result["chance"] = 0.0
        result["word_perplexity"] = float(math.exp(scores.nll.mean()))  # of the last word, as GPT-2's table 3 has it
        result["mean_tokens_in_word"] = float(scores.n_scored.mean())
    else:
        correct = np.array(credit[task.headline])
    result["accuracy"], result["stderr"] = _mean_and_se(correct)

    groups = sorted({item.group for item in items} - {""})
    if groups:
        by_group = {g: _mean_and_se(correct[[i for i, item in enumerate(items) if item.group == g]]) for g in groups}
        result["by_group"] = {g: {"accuracy": a, "stderr": se} for g, (a, se) in by_group.items()}
        if all("/" in g for g in groups):  # "phenomenon/paradigm": also report the phenomena
            families = sorted({g.split("/")[0] for g in groups})
            result["by_family"] = {}
            for fam in families:
                a, se = _mean_and_se(correct[[i for i, item in enumerate(items) if item.group.startswith(fam + "/")]])
                result["by_family"][fam] = {"accuracy": a, "stderr": se}
            # the paradigms, not the items, as the sampled units: how much the score depends on which
            # constructions the benchmark happens to contain
            result["stderr_across_groups"] = _mean_and_se(np.array([a for a, _ in by_group.values()]))[1]

    first = items[0]
    result["example"] = {"given": first.pairs[first.gold][0], "options": [scored for _, scored in first.pairs],
                         "right": first.gold}
    if len({given for given, _ in first.pairs}) > 1:
        result["example"]["given"] = [given for given, _ in first.pairs]
    result["correct"] = "".join("1" if c == 1.0 else "0" for c in correct)  # item by item, in the dataset's order
    result["limit"] = limit
    result["n_sequences"] = len(sequences)
    result["n_tokens"] = int(sum(len(s) for s in sequences))
    result["n_cut_to_window"] = int(n_cut)
    result["window"] = model.window
    result["seconds"] = round(time.time() - t0, 2)
    return result


def _summarize_files(files: list[str]) -> list[str] | str:
    return files if len(files) <= 2 else f"{len(files)} files, <paradigm>/{files[0].split('/', 1)[1]}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="a run directory or checkpoint of ours, or hf:<repo>[@revision]")
    ap.add_argument("--tasks", default="all", help=f"comma-separated, or all: {', '.join(TASKS)}")
    ap.add_argument("--limit", type=int, default=None, help="a fixed subset of N items a task (for tests)")
    ap.add_argument("--out", default=None, help="JSON file to write or to add to (results/eval/<name>.json)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--precision", default="tf32", choices=["float32", "tf32", "bf16"])
    ap.add_argument("--tokenizer", default=None, help="for our checkpoints, when the run's config does not name it")
    ap.add_argument("--batch-tokens", type=int, default=None)
    args = ap.parse_args()

    model = load_model(args.model, device=args.device, precision=args.precision, tokenizer=args.tokenizer)
    for name in (list(TASKS) if args.tasks == "all" else args.tasks.split(",")):
        result = run_task(model, name, limit=args.limit, batch_tokens=args.batch_tokens)
        print(f"{model.name:40s} {name:15s} {100 * result['accuracy']:6.2f}% (se {100 * result['stderr']:.2f}, chance "
              f"{100 * result['chance']:.0f}%, n {result['n']})  {result['n_tokens']} tokens  {result['seconds']:.0f}s",
              flush=True)
        if args.out:
            merge_result(args.out, model.info, "benchmarks", name, result)


if __name__ == "__main__":
    main()
