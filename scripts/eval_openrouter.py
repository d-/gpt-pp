"""Ask hosted models the multiple-choice items our small models take by log-likelihood.

A hosted chat model gives no log-probabilities, so it is asked the question in words and its letter is read from
the reply. The items are a fixed subset of three benchmarks (`gptpp.benchmarks.fixed_subset`: the same items on
every machine): 200 of HellaSwag, 200 of ARC-Easy and 300 BLiMP pairs as a forced choice between the two
sentences, their order decided by a hash of the item's number. The local models' accuracy on exactly these items
can be read off their own result files (`correct`, item by item).

Only the public benchmark items are sent: no system prompt, no description of what they are for, and no HTTP
header beyond the authorization. The key is read from the environment and never printed or stored.

Every request reports its own cost (OpenRouter's `usage.cost`, in US dollars); the script keeps a running total
and stops asking before a cap is passed, for each model and for the whole run. A model that would cost more than
its share is asked fewer items: a pilot of a few items measures its cost, and the largest nested prefix of the
fixed subset that fits is used.

    OPENROUTER_API_KEY=... python scripts/eval_openrouter.py --cap 5.0
    python scripts/eval_openrouter.py --table
    python scripts/eval_openrouter.py --reread       # read the stored replies again, after a change of read_letter
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# Python puts a script's own directory first on the module path. Put the repository first and this directory
# last, before anything else is imported, so that `gptpp` is found and nothing here shadows a library module.
sys.path[:] = [str(REPO)] + [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent] + [
    str(Path(__file__).resolve().parent)]

from concurrent.futures import ThreadPoolExecutor  # noqa: E402

import requests  # noqa: E402

from gptpp import benchmarks  # noqa: E402
from gptpp.evaluate import dump_json  # noqa: E402

API = "https://openrouter.ai/api/v1"
RESULTS = REPO / "results" / "eval" / "hosted"
SUBSET = {"hellaswag": 200, "arc_easy": 200, "blimp": 300}
LETTERS = "ABCDE"
KEPT = 4000  # characters of a reply kept in the result file, so that it can be read again

# Reference models, chosen on 2026-11-14 from the public list (GET /api/v1/models) by what the list itself says
# of them: OpenAI's "fast, cost-efficient", "mid-tier" and "flagship" models of the GPT-6 series, Anthropic's
# "most capable model", and two models whose weights are public (a dense 27B model and a 1.7B one).
# `share` is the part of the cap a model may spend; `max_tokens` overrides the default where a model thinks at
# length before it answers and cannot be told not to.
MODELS: list[dict] = [
    {"id": "openai/gpt-6-luna", "tier": "small", "share": 0.04},
    {"id": "openai/gpt-6.1-terra", "tier": "mid", "share": 0.08},
    {"id": "openai/gpt-6-astra", "tier": "frontier", "share": 0.34},
    {"id": "anthropic/claude-fable-5.2", "tier": "frontier", "share": 0.34},
    {"id": "qwen/qwen3.8-27b", "tier": "open weights, 27B", "share": 0.06},
    {"id": "liquid/lfm-3-1.7b", "tier": "open weights, 1.7B", "share": 0.10, "max_tokens": 2048},
]
EFFORTS = ["none", "minimal", "low", "medium", "high", "xhigh", "max"]  # least thinking first


# ---------------------------------------------------------------------------------------------------------------
# The questions.
# ---------------------------------------------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _items(task: str) -> list[benchmarks.Item]:
    return benchmarks.TASKS[task].items()


def questions(task: str, size: int) -> list[dict]:
    """The first `size` items of a task's fixed subset, worded for a chat model."""
    items = _items(task)
    out = []
    for index in benchmarks.fixed_subset(task, len(items), size):
        item = items[index]
        options = [scored.strip() for _, scored in item.pairs]
        gold = item.gold
        if task == "hellaswag":
            head = f"Which ending continues the text most plausibly?\n\nText: {item.pairs[0][0]}"
        elif task == "arc_easy":
            head = item.pairs[0][0].removeprefix("Question: ").removesuffix("\nAnswer:")
        else:  # blimp: the acceptable sentence is listed first in the data; a hash decides the order shown
            head = "Which of these two sentences is grammatically acceptable English?"
            if hashlib.sha256(f"blimp-order:{index}".encode()).digest()[0] & 1:
                options, gold = options[::-1], 1 - gold
        lines = "\n".join(f"{LETTERS[k]}. {option}" for k, option in enumerate(options))
        prompt = f"{head}\n\n{lines}\n\nAnswer with the letter only."
        out.append({"task": task, "index": index, "prompt": prompt, "n_options": len(options), "gold": LETTERS[gold]})
    return out


def read_letter(reply: str | None, n_options: int) -> str | None:
    """The one letter a reply comes down to, or None.

    Tried in this order: the reply is a lone letter ("B", "(B)", "**B.**"); it names its answer ("Answer: B",
    "the answer is B": the last such statement); it begins with a lettered option ("B. pulls the rope"); its last
    line is a lone letter, as when an explanation ends with the answer; it is one short sentence in which a single
    letter stands by itself ("Sentence B is acceptable."). A reply that names two letters ("A, B") or none
    ("Neither."), or a long one that does not end with its letter, is no answer.
    """
    if not reply:
        return None
    text, allowed = reply.strip(), LETTERS[:n_options]
    lone = rf"\W*([{allowed}])\W*"
    named = re.findall(rf"(?i:answer)\W*(?i:is)?\W*([{allowed}])(?![A-Za-z])", text)
    if match := re.fullmatch(lone, text):
        return match.group(1)
    if named:
        return named[-1]
    if match := re.match(rf"\W*([{allowed}])[.):]", text) or re.fullmatch(lone, text.splitlines()[-1]):
        return match.group(1)
    standing = set(re.findall(rf"(?<![A-Za-z'])([{allowed}])(?![A-Za-z'])", text))
    return standing.pop() if len(standing) == 1 and len(text) <= 80 else None


# ---------------------------------------------------------------------------------------------------------------
# Asking.
# ---------------------------------------------------------------------------------------------------------------


def request_settings(listing: dict, max_tokens: int) -> dict:
    """The settings sent with every request to a model: as little thinking as it allows, and no sampling noise
    where the model accepts a temperature."""
    supported = set(listing.get("supported_parameters") or [])
    settings: dict = {"max_tokens": max_tokens, "usage": {"include": True}}
    if "temperature" in supported:
        settings["temperature"] = 0
    reasoning = listing.get("reasoning") or {}
    efforts = reasoning.get("supported_efforts") or []
    if efforts:
        settings["reasoning"] = {"effort": min(efforts, key=EFFORTS.index)}
    elif "reasoning" in supported and not reasoning.get("mandatory"):
        settings["reasoning"] = {"enabled": False}
    return settings


def ask(key: str, model: str, question: dict, settings: dict, tries: int = 3) -> dict:
    """One question to one model. Returns the reply, the letter read from it, and what the request cost."""
    body = {"model": model, "messages": [{"role": "user", "content": question["prompt"]}], **settings}
    error = None
    for attempt in range(tries):
        try:
            response = requests.post(f"{API}/chat/completions", json=body, timeout=120,
                                     headers={"Authorization": f"Bearer {key}"})
            data = response.json()
            if response.status_code == 200 and data.get("choices"):
                usage = data.get("usage") or {}
                reply = data["choices"][0]["message"].get("content")
                details = usage.get("completion_tokens_details") or {}
                return {"task": question["task"], "index": question["index"], "gold": question["gold"],
                        "reply": (reply or "")[:KEPT], "answer": read_letter(reply, question["n_options"]),
                        "cost": float(usage.get("cost") or 0.0), "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "reasoning_tokens": details.get("reasoning_tokens", 0),
                        "finish": data["choices"][0].get("finish_reason"), "provider": data.get("provider")}
            # only the error itself is kept: the rest of the body names the account
            error = f"HTTP {response.status_code}: {json.dumps(data.get('error') or 'no reply in the body')[:300]}"
            if response.status_code in (400, 401, 402, 403, 404):
                break  # the request itself is refused: asking again will not help
        except (requests.RequestException, ValueError) as exc:
            error = f"{type(exc).__name__}: {str(exc)[:200]}"
        time.sleep(2.0 * (attempt + 1))
    return {"task": question["task"], "index": question["index"], "gold": question["gold"], "reply": "",
            "answer": None, "cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0,
            "finish": "error", "error": error}


class Budget:
    """Dollars spent so far, against a cap that must not be passed."""

    def __init__(self, cap: float):
        self.cap, self.spent = cap, 0.0

    def room(self) -> float:
        return self.cap - self.spent


def worst_case(listing: dict, settings: dict, prompt_tokens: int = 400) -> float:
    """An upper bound on what one request can cost: a long prompt and every allowed output token, at list price."""
    price = listing["pricing"]
    return prompt_tokens * float(price["prompt"]) + settings["max_tokens"] * float(price["completion"])


def ask_all(key: str, listing: dict, todo: list[dict], settings: dict, budgets: list[Budget], workers: int) -> list[dict]:
    """Ask the questions a batch at a time. Stop before a batch could pass any of the caps, and stop when two
    batches in a row fail entirely (a provider that is down would otherwise be retried for hours)."""
    answers, failed_batches = [], 0
    bound = worst_case(listing, settings)
    with ThreadPoolExecutor(workers) as pool:
        for start in range(0, len(todo), workers):
            batch = todo[start:start + workers]
            if any(b.room() < bound * len(batch) for b in budgets):
                print(f"    stopped at {len(answers)} of {len(todo)}: the next batch could pass a cap", flush=True)
                break
            got = list(pool.map(lambda q: ask(key, listing["id"], q, settings), batch))
            for b in budgets:
                b.spent += sum(a["cost"] for a in got)
            answers += got
            failed_batches = failed_batches + 1 if all(a["finish"] == "error" for a in got) else 0
            if failed_batches == 2:
                print(f"    stopped at {len(answers)} of {len(todo)}: two batches in a row failed "
                      f"({got[0].get('error')})", flush=True)
                break
    return answers


def summarize(answers: list[dict]) -> dict:
    """Accuracy by task, over the questions the model replied to. A reply that cannot be read counts as wrong
    (`n_unread`); a request that failed never reached the model and is left out (`n_failed`)."""
    out = {}
    for task in SUBSET:
        rows = [a for a in answers if a["task"] == task and a["finish"] != "error"]
        if rows:
            n = len(rows)
            acc = sum(a["answer"] == a["gold"] for a in rows) / n
            out[task] = {"n": n, "accuracy": acc, "stderr": math.sqrt(acc * (1 - acc) / n),
                         "n_unread": sum(a["answer"] is None for a in rows),
                         "n_failed": sum(a["task"] == task and a["finish"] == "error" for a in answers),
                         "indices": [a["index"] for a in rows]}
    return out


def run_model(key: str, entry: dict, listing: dict, total: Budget, args: argparse.Namespace) -> dict:
    settings = request_settings(listing, entry.get("max_tokens", args.max_tokens))
    own = Budget(entry["share"] * args.cap)
    budgets = [own, total]
    print(f"{entry['id']}: settings {json.dumps({k: v for k, v in settings.items() if k != 'usage'})}, "
          f"share ${own.cap:.2f}", flush=True)

    # A pilot on the first items of each task measures what an item costs.
    pilot = [q for task in SUBSET for q in questions(task, args.pilot)]
    answers = ask_all(key, listing, pilot, settings, budgets, args.workers)
    if args.pilot_only:
        n = max(1, len(answers))
        used = sum(a["completion_tokens"] for a in answers) / n
        thinking = sum(a["reasoning_tokens"] for a in answers) / n
        print(f"    pilot: ${own.spent:.4f} for {len(answers)} items; a reply used {used:.0f} tokens "
              f"({thinking:.0f} of thinking); replies {[a['reply'][:12] for a in answers]}; "
              f"errors {sorted({a['error'] for a in answers if a.get('error')})}", flush=True)
        return {}
    per_item = {task: max(1e-9, sum(a["cost"] for a in answers if a["task"] == task) /
                          max(1, sum(a["task"] == task for a in answers))) for task in SUBSET}
    full_cost = sum(per_item[task] * (SUBSET[task] - args.pilot) for task in SUBSET)
    fraction = min(1.0, 0.9 * own.room() / full_cost) if answers and full_cost > 0 else 0.0
    sizes = {task: max(args.pilot, int(fraction * SUBSET[task])) for task in SUBSET}
    print(f"    pilot: ${own.spent:.4f} for {len(answers)} items; the whole subset would cost ${full_cost:.2f}; "
          f"asking {sizes}", flush=True)

    # The rest, the tasks interleaved in proportion, so that a run stopped early is still balanced.
    asked = {(a["task"], a["index"]) for a in answers}
    rest = [(rank / sizes[task], q) for task in SUBSET for rank, q in enumerate(questions(task, sizes[task]))
            if (q["task"], q["index"]) not in asked]
    answers += ask_all(key, listing, [q for _, q in sorted(rest, key=lambda pair: pair[0])], settings, budgets,
                       args.workers)

    return {
        "model": {"id": entry["id"], "name": listing.get("name"), "tier": entry["tier"],
                  "canonical_slug": listing.get("canonical_slug"), "hugging_face_id": listing.get("hugging_face_id") or None,
                  "listed": time.strftime("%Y-%m-%d", time.gmtime(listing["created"])),
                  "price_per_million_tokens": {"prompt": 1e6 * float(listing["pricing"]["prompt"]),
                                               "completion": 1e6 * float(listing["pricing"]["completion"])}},
        "date": time.strftime("%Y-%m-%d"),
        "settings": {k: v for k, v in settings.items() if k != "usage"},
        "prompt_example": pilot[0]["prompt"],
        "results": summarize(answers),
        "cost_usd": round(own.spent, 6),
        "tokens": {name: sum(a[name] for a in answers)
                   for name in ("prompt_tokens", "completion_tokens", "reasoning_tokens")},
        "providers": sorted({a["provider"] for a in answers if a.get("provider")}),
        "errors": sorted({a["error"] for a in answers if a.get("error")})[:5],
        "answers": [{k: a[k] for k in ("task", "index", "gold", "answer", "reply", "finish")} for a in answers],
    }


def reread(key: str | None, listing: dict, args: argparse.Namespace) -> None:
    """Read every stored reply again with `read_letter` as it is now, and write the results again.

    A reply that was stored cut short (more than `KEPT` characters) cannot be read again: its question is asked
    once more, if a key is given, and the new reply replaces it. The file records how many were.
    """
    for path in sorted(RESULTS.glob("*.json")):
        if path.name.startswith("run-"):
            continue
        record = json.loads(path.read_text())
        model = record["model"]["id"]
        asked = {(q["task"], q["index"]): q for task in SUBSET for q in questions(task, SUBSET[task])}
        cut = [a for a in record["answers"] if len(a["reply"]) >= KEPT and a["finish"] != "error"]
        again = {}
        if cut and key and model in listing:
            settings = {**record["settings"], "usage": {"include": True}}
            budget = Budget(args.cap)
            got = ask_all(key, listing[model], [asked[(a["task"], a["index"])] for a in cut], settings, [budget],
                          args.workers)
            again = {(a["task"], a["index"]): a for a in got if a["finish"] != "error"}
            record["cost_usd"] = round(record["cost_usd"] + budget.spent, 6)
            for name in record["tokens"]:
                record["tokens"][name] += sum(a[name] for a in got)
            record["asked_again"] = {"date": time.strftime("%Y-%m-%d"), "n": len(again), "cost_usd": round(budget.spent, 6),
                                     "why": "the first reply was stored cut short and could not be read again"}
        changed = 0
        for a in record["answers"]:
            new = again.get((a["task"], a["index"]))
            if new:
                a.update({k: new[k] for k in ("reply", "finish")})
            if new or len(a["reply"]) < KEPT:
                answer = read_letter(a["reply"], asked[(a["task"], a["index"])]["n_options"])
                changed += answer != a["answer"]
                a["answer"] = answer
        record["results"] = summarize(record["answers"])
        path.write_text(dump_json(record) + "\n")
        shown = "  ".join(f"{t} {100 * r['accuracy']:.1f}% (n {r['n']}, {r['n_unread']} unread)"
                          for t, r in record["results"].items())
        print(f"{model}: {len(cut)} replies stored cut short, {len(again)} asked again, {changed} read differently\n"
              f"    {shown}", flush=True)


# ---------------------------------------------------------------------------------------------------------------
# The table: hosted models beside the local ones, on the same items.
# ---------------------------------------------------------------------------------------------------------------


def table() -> str:
    """The hosted models, then the local ones scored by log-likelihood on the items of the full subset."""
    from eval_released_models import load_results  # the local models, in their usual order

    order = [m["id"] for m in MODELS]
    hosted = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("*.json")) if not p.name.startswith("run-")]
    hosted.sort(key=lambda record: order.index(record["model"]["id"]) if record["model"]["id"] in order else len(order))
    lines = ["| model | " + " | ".join(f"{t} (n)" for t in SUBSET) + " | cost |", "|:--|" + "--:|" * (len(SUBSET) + 1)]
    for record in hosted:
        cells = [f"{100 * r['accuracy']:.1f} ± {100 * r['stderr']:.1f} ({r['n']})" if (r := record["results"].get(t))
                 else "" for t in SUBSET]
        lines.append(f"| {record['model']['id']} ({record['model']['tier']}) | " + " | ".join(cells)
                     + f" | ${record['cost_usd']:.2f} |")
    for record in load_results(REPO / "results" / "eval"):
        done = record.get("benchmarks", {})
        if not all(t in done and done[t].get("limit") is None for t in SUBSET):
            continue
        cells = []
        for task, size in SUBSET.items():
            marks = done[task]["correct"]
            picked = [marks[i] == "1" for i in benchmarks.fixed_subset(task, len(marks), size)]
            acc = sum(picked) / len(picked)
            cells.append(f"{100 * acc:.1f} ± {100 * math.sqrt(acc * (1 - acc) / len(picked)):.1f} ({len(picked)})")
        name = record["model"].get("label", record["model"]["name"])
        lines.append(f"| {'**' + name + '**' if record['model'].get('kind') == 'gptpp' else name} (log-likelihood) | "
                     + " | ".join(cells) + " | |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cap", type=float, default=5.0, help="US dollars the whole run may spend")
    ap.add_argument("--models", default="all", help="comma-separated OpenRouter ids (default: the reference models)")
    ap.add_argument("--pilot", type=int, default=4, help="items a task asked first, to measure a model's cost")
    ap.add_argument("--max-tokens", type=int, default=512, help="output tokens a reply may use, thinking included")
    ap.add_argument("--workers", type=int, default=12, help="requests in flight at once")
    ap.add_argument("--pilot-only", action="store_true", help="ask the pilot items, print what they cost, write nothing")
    ap.add_argument("--reread", action="store_true",
                    help="read the stored replies again (after a change of read_letter) and write the results again")
    ap.add_argument("--table", action="store_true", help="print the table of results/eval/hosted and exit")
    args = ap.parse_args()
    if args.table:
        print(table())
        return

    key = os.environ.get("OPENROUTER_API_KEY")
    listing = {m["id"]: m for m in requests.get(f"{API}/models", timeout=60).json()["data"]}
    if args.reread:
        reread(key, listing, args)
        return
    if not key:
        sys.exit("set OPENROUTER_API_KEY in the environment")
    entries = MODELS if args.models == "all" else [
        next((m for m in MODELS if m["id"] == i), {"id": i, "tier": "", "share": 1.0 / len(args.models.split(","))})
        for i in args.models.split(",")]

    def account() -> float:  # what the key has spent in all, as OpenRouter counts it
        data = requests.get(f"{API}/key", headers={"Authorization": f"Bearer {key}"}, timeout=60).json()["data"]
        return float(data["usage"])

    started = time.time()
    before = account()
    total = Budget(args.cap)
    RESULTS.mkdir(parents=True, exist_ok=True)
    done = []
    for entry in entries:
        if entry["id"] not in listing:
            print(f"{entry['id']}: not in the model list today, skipped", flush=True)
            continue
        record = run_model(key, entry, listing[entry["id"]], total, args)
        if args.pilot_only:
            continue
        (RESULTS / (entry["id"].replace("/", "--") + ".json")).write_text(dump_json(record) + "\n")
        shown = "  ".join(f"{t} {100 * r['accuracy']:.1f}% (n {r['n']})" for t, r in record["results"].items())
        print(f"    {shown}  ${record['cost_usd']:.4f}; ${total.spent:.4f} in all", flush=True)
        # the run's account so far, after every model, so that an interrupted run keeps it
        done.append({"id": entry["id"], "cost_usd": record["cost_usd"]})
        run = {"date": time.strftime("%Y-%m-%d"), "cap_usd": args.cap, "spent_usd_by_request_costs": round(total.spent, 6),
               "spent_usd_by_account_counter": round(account() - before, 6),  # trails the requests by some seconds
               "models": done, "subset": SUBSET}
        (RESULTS / f"run-{time.strftime('%Y%m%d-%H%M%S', time.gmtime(started))}.json").write_text(dump_json(run) + "\n")

    print(f"${total.spent:.4f} in all" + (" (pilots only)" if args.pilot_only else ""), flush=True)


if __name__ == "__main__":
    main()
