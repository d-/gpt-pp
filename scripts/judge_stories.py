"""Grade story completions with a hosted model: the TinyStories evaluation.

Loss says how well a small model predicts stories; it does not say whether the stories it writes hold together.
Eldan and Li ("TinyStories: how small can language models be and still speak coherent English?", 2023,
arXiv:2305.07759, section 3) therefore let a large model mark them like a teacher: the small model is given the
beginning of a story, usually cut in the middle of a sentence, and completes it; the grader reads the beginning
and the completion, writes a short assessment, and then grades grammar, creativity and consistency with the
beginning out of ten (we also ask for the plot, as the paper does for its instruction-following models, and for
the age the writing suggests). The paper generates ten completions a prompt at temperature 1 and averages.

Input: a JSONL file, one completion a line: {"model", "prompt_id", "prompt", "completion"}.
Output: the same lines with the grades (JSONL), and a summary a model: the mean of each grade with a standard
error over the prompts (completions of one prompt are not independent, so the prompts are the sampled units).

    python scripts/judge_stories.py --write-prompts prompts.jsonl      # the paper's 44 story beginnings
    OPENROUTER_API_KEY=... python scripts/judge_stories.py --completions c.jsonl --out results/eval/stories/c.scores.jsonl
    OPENROUTER_API_KEY=... python scripts/judge_stories.py --self-test # a handful of made-up completions

Only the stories are sent to the grader: no system prompt, no model names, no header beyond the authorization.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# Python puts a script's own directory first on the module path, and this directory holds a file named queue.py,
# which would then be imported in place of the standard library's `queue` (by torch, by the thread pool). Put the
# repository first and this directory last, before anything else is imported.
sys.path[:] = [str(REPO)] + [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent] + [
    str(Path(__file__).resolve().parent)]

from concurrent.futures import ThreadPoolExecutor  # noqa: E402

import requests  # noqa: E402

from eval_openrouter import API, request_settings  # noqa: E402  (the same way of asking: least hidden thinking)

GRADER = "openai/gpt-6.1-terra"  # a mid-priced model of November 2026; the paper used GPT-4
GRADES = ("grammar", "creativity", "consistency", "plot")
AGE_GROUPS = {"A": "3 or under", "B": "4-5", "C": "6-7", "D": "8-9", "E": "10-12", "F": "13-16"}

# The exercise, worded after the transcript in section 3 of the paper.
EXERCISE = """In the following exercise, the student is given the beginning of a story. The student needs to complete it into a full story. The exercise tests the student's language abilities and creativity. The symbol *** marks the separator between the prescribed beginning and the student's completion:

{prompt}***{completion}

First give a short assessment of the part written by the student (the part after the *** symbol): is it grammatically correct, is it consistent with the beginning of the story, and does the student complete the sentence that the separator splits? Then grade the student's completion in terms of grammar, creativity, consistency with the story's beginning and whether the plot makes sense, each out of 10, and give your best guess of the student's age as reflected by the completion, from these groups: A: 3 or under. B: 4-5. C: 6-7. D: 8-9. E: 10-12. F: 13-16.

End your answer with one line in exactly this format:
Grammar: x/10, Creativity: x/10, Consistency: x/10, Plot: x/10, Age group: X"""

PROMPTS_SOURCE = ("roneneldan/TinyStories", "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64", "Evaluation prompts.yaml")

# Made-up completions for --self-test. The first story beginning of the paper's set (a sick bunny who "was so
# weak she could"), completed well, in broken grammar, off the subject, and not at all; then the second beginning
# (a girl tries to open an old cabinet, "when suddenly"), completed well and in broken grammar.
SELF_TEST = [
    ("a coherent completion",
     " not even hop to the door. Lucy lay down in her soft bed and felt very sad about the feast. Then her friend "
     "Mia the bunny came to see her. Mia had brought a basket of carrots and a warm blanket. \"We will have the "
     "feast here,\" said Mia. All the bunny friends came and sat around Lucy's bed. They ate, they laughed, and "
     "Lucy felt much better. She learned that good friends bring the party to you."),
    ("broken grammar",
     " not go feast she. Lucy bed in lay and sad was very the. Friend bunny come carrots bring is. They eated "
     "and laughs and Lucy feeled more good after are. The end was happy they is all."),
    ("fluent but about something else",
     " see the big red truck. The truck was driving to the city to bring apples to the store. The driver, a tall "
     "man named Bob, loved his truck very much. He washed it every day. One day the truck got a flat tire, and "
     "Bob fixed it with his tools. Then he drove home and had dinner."),
    ("nothing but repetition", " the the the the the the the the the the the the the the the the the the the the"),
]
SELF_TEST_SECOND = [
    ("a coherent completion",
     " the door flew open and a little gray kitten jumped out. The girl laughed. \"So you were the strange "
     "noise!\" she said. She gave the kitten some milk and made a soft bed for it inside the old cabinet. From "
     "that day on, the cabinet was the kitten's house, and the girl visited it every morning."),
    ("broken grammar",
     " door open kitten jump. Girl laugh is. She milk give and bed make it for. Cabinet house kitten be now, "
     "girl visit all morning the."),
]


def paper_prompts() -> list[str]:
    """The story beginnings the paper's authors wrote for this evaluation (44 of them), from the dataset's repository."""
    import yaml
    from huggingface_hub import hf_hub_download

    repo, revision, filename = PROMPTS_SOURCE
    with open(hf_hub_download(repo, filename, repo_type="dataset", revision=revision), encoding="utf-8") as f:
        return [str(p) for p in yaml.safe_load(f)]


def read_grades(reply: str) -> dict | None:
    """The grades in a grader's reply: the last line of the agreed form. None if the reply has no such line."""
    found = None
    for found in re.finditer(r"Grammar:\s*(\d+(?:\.\d+)?)\s*/\s*10\W+Creativity:\s*(\d+(?:\.\d+)?)\s*/\s*10\W+"
                             r"Consistency:\s*(\d+(?:\.\d+)?)\s*/\s*10\W+Plot:\s*(\d+(?:\.\d+)?)\s*/\s*10\W+"
                             r"Age group:\s*\(?([A-F])\b", reply or "", flags=re.IGNORECASE):
        pass
    if found is None:
        return None
    grades = {name: float(found.group(k + 1)) for k, name in enumerate(GRADES)}
    if any(not 0 <= g <= 10 for g in grades.values()):
        return None
    return {**grades, "age_group": found.group(5).upper()}


def grade(key: str, grader: str, row: dict, settings: dict, tries: int = 3) -> dict:
    """One completion to the grader. Returns the row with its grades, the assessment and the cost."""
    body = {"model": grader, **settings,
            "messages": [{"role": "user", "content": EXERCISE.format(prompt=row["prompt"], completion=row["completion"])}]}
    out = {key_: row[key_] for key_ in row if key_ not in ("prompt", "completion")}
    out.update({name: None for name in (*GRADES, "age_group")})
    out.update({"cost": 0.0, "assessment": "", "error": None})
    for attempt in range(tries):
        try:
            response = requests.post(f"{API}/chat/completions", json=body, timeout=180,
                                     headers={"Authorization": f"Bearer {key}"})
            data = response.json()
            if response.status_code == 200 and data.get("choices"):
                reply = data["choices"][0]["message"].get("content") or ""
                out["cost"] += float((data.get("usage") or {}).get("cost") or 0.0)
                grades = read_grades(reply)
                if grades:
                    out.update(grades)
                    out.update({"assessment": reply.strip(), "error": None})
                    return out
                out["error"] = "the reply has no line of grades"
            else:
                out["error"] = f"HTTP {response.status_code}: {json.dumps(data.get('error') or 'no reply in the body')[:300]}"
                if response.status_code in (400, 401, 402, 403, 404):
                    break
        except (requests.RequestException, ValueError) as exc:
            out["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        time.sleep(2.0 * (attempt + 1))
    return out


def summarize(rows: list[dict]) -> dict:
    """A model's mean grades. The standard error is over prompts: each prompt's completions are averaged first."""
    summary = {}
    for model in sorted({r["model"] for r in rows}):
        graded = [r for r in rows if r["model"] == model and r["grammar"] is not None]
        entry: dict = {"n_completions": sum(r["model"] == model for r in rows), "n_graded": len(graded),
                       "n_prompts": len({r["prompt_id"] for r in graded})}
        for name in GRADES:
            by_prompt: dict = {}
            for r in graded:
                by_prompt.setdefault(r["prompt_id"], []).append(r[name])
            means = [sum(v) / len(v) for v in by_prompt.values()]
            if means:
                mean = sum(means) / len(means)
                var = sum((m - mean) ** 2 for m in means) / (len(means) - 1) if len(means) > 1 else 0.0
                entry[name] = {"mean": mean, "stderr": math.sqrt(var / len(means))}
        ages = [r["age_group"] for r in graded]
        entry["age_groups"] = {f"{g} ({AGE_GROUPS[g]})": ages.count(g) for g in AGE_GROUPS if g in ages}
        summary[model] = entry
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--completions", help="JSONL: {model, prompt_id, prompt, completion} a line")
    ap.add_argument("--out", help="JSONL of grades to write (default: <completions>.scores.jsonl)")
    ap.add_argument("--grader", default=GRADER, help="OpenRouter model id of the grader")
    ap.add_argument("--cap", type=float, default=0.5, help="US dollars the run may spend")
    ap.add_argument("--max-tokens", type=int, default=600, help="output tokens of a grading")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--write-prompts", metavar="FILE", help="write the paper's story beginnings as JSONL and exit")
    ap.add_argument("--self-test", action="store_true", help="grade a handful of made-up completions")
    args = ap.parse_args()

    if args.write_prompts:
        with open(args.write_prompts, "w", encoding="utf-8") as f:
            for k, prompt in enumerate(paper_prompts()):
                f.write(json.dumps({"prompt_id": k, "prompt": prompt}, ensure_ascii=False) + "\n")
        print(f"{k + 1} story beginnings -> {args.write_prompts}")
        return

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("set OPENROUTER_API_KEY in the environment")
    if args.self_test:
        first, second = paper_prompts()[:2]
        rows = [{"model": label, "prompt_id": 0, "prompt": first, "completion": text} for label, text in SELF_TEST]
        rows += [{"model": label, "prompt_id": 1, "prompt": second, "completion": text} for label, text in SELF_TEST_SECOND]
    elif args.completions:
        with open(args.completions, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    else:
        sys.exit("give --completions FILE, --self-test or --write-prompts FILE")

    listing = next(m for m in requests.get(f"{API}/models", timeout=60).json()["data"] if m["id"] == args.grader)
    settings = request_settings(listing, args.max_tokens)
    price = listing["pricing"]
    longest = max(len(EXERCISE.format(prompt=r["prompt"], completion=r["completion"])) for r in rows)
    bound = longest / 2 * float(price["prompt"]) + args.max_tokens * float(price["completion"])  # a request, at most

    graded, spent = [], 0.0
    with ThreadPoolExecutor(args.workers) as pool:
        for start in range(0, len(rows), args.workers):
            batch = rows[start:start + args.workers]
            if spent + 3 * bound * len(batch) > args.cap:  # three tries a completion at the very most
                print(f"stopped at {len(graded)} of {len(rows)}: the next batch could pass the cap of ${args.cap}")
                break
            done = list(pool.map(lambda r: grade(key, args.grader, r, settings), batch))
            spent += sum(d["cost"] for d in done)
            graded += done

    summary = {"grader": args.grader, "settings": {k: v for k, v in settings.items() if k != "usage"},
               "date": time.strftime("%Y-%m-%d"), "cost_usd": round(spent, 6),
               "n_ungraded": sum(g["grammar"] is None for g in graded), "models": summarize(graded)}
    for model, entry in summary["models"].items():
        marks = "  ".join(f"{name} {entry[name]['mean']:.1f}" for name in GRADES if name in entry)
        print(f"{model:36s} {marks}  ages {entry['age_groups']}  ({entry['n_graded']} graded)")
    print(f"grader {args.grader}; ${spent:.4f}; {summary['n_ungraded']} ungraded")

    if args.self_test:
        models = summary["models"]
        good, broken, other = (models[label] for label, _ in SELF_TEST[:3])
        assert good["grammar"]["mean"] > broken["grammar"]["mean"] + 2, "broken grammar must cost grammar marks"
        assert good["consistency"]["mean"] > other["consistency"]["mean"] + 2, "a change of subject must cost consistency"
        assert good["plot"]["mean"] > models[SELF_TEST[3][0]]["plot"]["mean"] + 2, "repetition is not a plot"
        print("self-test passed: the grader tells the four completions apart")
        return
    out = Path(args.out or str(Path(args.completions).with_suffix("")) + ".scores.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for g in graded:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    print(f"grades -> {out}\nsummary -> {out.with_suffix('.summary.json')}")


if __name__ == "__main__":
    main()
