"""Write the job lists of the lab's experiments (one JSON object a line, for scripts/queue.py).

    python experiments/make_jobs.py            # writes experiments/main.jsonl and experiments/small.jsonl

main.jsonl   the main run (three cards), then, once it is done, the same cards take one-card jobs: the second seed
             of the ladder, the recipe race at the base size, the scaling grid
small.jsonl  one-card jobs for the fourth card meanwhile: the ladder from the 2019 recipe to the 2026 recipe (first
             seed), one-at-a-time swaps around the 2026 recipe, the optimizers

Every job is a training run named after its --run, so a finished run is skipped when a queue is started again.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
T = "$PY -m gptpp.train"


def job(name: str, args: str, gpus: int = 1) -> dict:
    if gpus == 1:
        return {"name": name, "cmd": f"{T} --run {name} {args}"}
    return {"name": name, "gpus": gpus,
            "cmd": f"$PY -m torch.distributed.run --standalone --nproc_per_node={gpus} -m gptpp.train --run {name} {args}"}


# ---------------------------------------------------------------------------------------------- the main run
# 12 layers, 768 wide, the 2026 recipe, FineWeb-EDU, three cards: 3.0e9 tokens at 48 x 1024 x 3 tokens a step.
STEPS = round(3.0e9 / (48 * 1024 * 3))
LINEAR = ",".join(str(s) for s in range(2000, STEPS, 2000))
main = [job("gptpp-base", f"--preset gptpp-base --tokens 3.0e9 --set device_batch=48 --eval-tokens 50e6 "
                          f"--save-log 24 --save-steps {LINEAR}", gpus=3)]

# ---------------------------------------------------------------------------------------------- the ladder
# From the 2019 recipe to the 2026 recipe, one change a rung, cumulative. Rungs 0 to 6 keep the 2019 training
# recipe (AdamW, warmup, cosine); rung 7 switches the training recipe; rungs 8 and 9 finish the model.
LADDER_TOKENS = "100e6"
A = "--preset gpt2-2019-ladder"
steps = [
    ("0-gpt2-2019",   ""),                                   # the published design
    ("1-no-biases",   "bias=false"),
    ("2-rotary",      "pos=rope"),
    ("3-rmsnorm",     "norm=rmsnorm_fixed"),
    ("4-relu2",       "act=relu2"),
    ("5-qk-norm",     "qk_norm=true"),
    ("6-zero-init",   "init=zero_proj"),
]
small = []
acc: list[str] = []
ladder_rows = []
for name, change in steps:
    if change:
        acc.append(change)
    sets = (" --set " + " ".join(acc)) if acc else ""
    ladder_rows.append((name, f"{A}{sets}"))
B = f"{A} --train-preset gptpp-ladder --set " + " ".join(acc)
ladder_rows.append(("7-muon", B))                                                      # the 2026 training recipe
ladder_rows.append(("8-untied", B + " tie_embeddings=false embed_norm=true"))          # two matrices, normalized embedding
ladder_rows.append(("9-softcap", "--preset gptpp-ladder"))                             # = the 2026 recipe exactly
for seed in (0, 1):
    for name, args in ladder_rows:
        small.append(job(f"ladder-{name}-s{seed}", f"{args} --tokens {LADDER_TOKENS} --seed {seed} --eval-tokens 10e6"))

# ---------------------------------------------------------------------------------------------- one-at-a-time swaps
# Around the 2026 recipe at the ladder size, 50M tokens, one seed. The baseline is ladder-9-softcap at 50M: swap-base.
G = "--preset gptpp-ladder --tokens 50e6 --eval-tokens 10e6"
swaps = [("base", "")]
swaps += [(f"act-{a}", f"act={a}") for a in ("relu", "gelu", "silu", "swiglu", "geglu")]
swaps += [(f"pos-{p}", f"pos={p}") for p in ("learned", "sinusoidal", "alibi", "none")]
swaps += [("norm-layernorm", "norm=layernorm"), ("norm-rmsnorm-gain", "norm=rmsnorm"), ("norm-dyt", "norm=dyt"),
          ("norm-none", "norm=none embed_norm=false"), ("norm-post", "norm=layernorm norm_placement=post")]
swaps += [("attn-no-qk-norm", "qk_norm=false"), ("attn-qk-learned", "qk_scale_learned=true"),
          ("attn-gqa4", "n_kv_head=4"), ("attn-gqa2", "n_kv_head=2"), ("attn-mqa", "n_kv_head=1"),
          ("no-softcap", "logit_softcap=0"), ("tied", "tie_embeddings=true embed_norm=false"), ("init-gpt2", "init=gpt2"),
          ("no-embed-norm", "embed_norm=false"), ("bias", "bias=true")]
swaps += [(f"sched-{s}", f"schedule={s}") for s in ("cosine", "linear", "constant")]
swaps += [("no-warmup", "warmup_tokens=0")]
for name, change in swaps:
    small.append(job(f"swap-{name}", G + (f" --set {change}" if change else "")))
# optimizers on the 2026 model: each at three rates; the best rate is the optimizer's entry
OPT = "optimizer={o} lr={lr} lr_embedding={lr} lr_unembedding={lr} schedule=cosine warmup_tokens=4e6 final_lr_frac=0.1"
for o, rates, extra in (("adamw", ("1.2e-3", "2.4e-3", "4.8e-3"), "weight_decay=0.1 beta1=0.9 grad_clip=1.0"),
                        ("adam", ("1.2e-3", "2.4e-3", "4.8e-3"), "weight_decay=0.0 beta1=0.9 grad_clip=1.0"),
                        ("lion", ("1e-4", "3e-4", "1e-3"), "weight_decay=0.1 beta1=0.9 beta2=0.99"),
                        ("sgd", ("0.03", "0.1", "0.3"), "weight_decay=0.0 momentum=0.9 grad_clip=1.0")):
    for lr in rates:
        small.append(job(f"opt-{o}-lr{lr}", f"{G} --set {OPT.format(o=o, lr=lr)} {extra}"))
# warmup for the 2019 recipe, pre-norm and post-norm
small.append(job("warm-2019-base", "--preset gpt2-2019-ladder --tokens 50e6 --eval-tokens 10e6"))          # the baseline of the three below
small.append(job("warm-2019-none", "--preset gpt2-2019-ladder --tokens 50e6 --eval-tokens 10e6 --set warmup_tokens=0"))
small.append(job("warm-2019-post", "--preset gpt2-2019-ladder --tokens 50e6 --eval-tokens 10e6 --set norm_placement=post"))
small.append(job("warm-2019-post-none", "--preset gpt2-2019-ladder --tokens 50e6 --eval-tokens 10e6 --set norm_placement=post warmup_tokens=0"))

# ---------------------------------------------------------------------------------------------- the recipe race
# The 2019 recipe against the 2026 recipe at the base size, one card each, the same batch and tokens. The 2019
# recipe's rate was tuned only at the smaller sizes, so it gets a short sweep first.
R = "--set device_batch=48 --eval-tokens 20e6"
for lr in ("6e-4", "1.2e-3", "2.4e-3"):
    small.append(job(f"race-2019-lr{lr}-100M", f"--preset gpt2-2019-base --tokens 100e6 --eval-tokens 20e6 --set device_batch=48 lr={lr}"))
# the base preset of the 2019 recipe warms up over 32M tokens, a third of this race: two more runs with the ladder's
# warmup of 8M tokens, so that the comparison does not rest on a schedule tuned for a longer run
for lr in ("1.2e-3", "2.4e-3"):
    small.append(job(f"race-2019-lr{lr}-w8M-100M", f"--preset gpt2-2019-base --tokens 100e6 --eval-tokens 20e6 --set device_batch=48 lr={lr} warmup_tokens=8e6"))
small.append(job("race-2026-100M", f"--preset gptpp-base --tokens 100e6 {R} --save-log 12"))

# ---------------------------------------------------------------------------------------------- the scaling grid
# Six sizes by four token budgets, the 2026 recipe, context 512, FineWeb-EDU. Rates of the vocabulary matrices
# scale with width by the preset's rule.
SIZES = {"w128-l4": "d_model=128 n_layer=4 n_head=4", "w192-l6": "d_model=192 n_layer=6 n_head=6",
         "w256-l8": "d_model=256 n_layer=8 n_head=8", "w384-l8": "d_model=384 n_layer=8 n_head=8",
         "w512-l8": "", "w768-l12": "d_model=768 n_layer=12 n_head=12 device_batch=64 grad_accum=2"}
for budget in ("25e6", "50e6", "100e6", "200e6"):
    for size, sets in SIZES.items():
        if size == "w768-l12" and budget == "200e6":
            continue                      # 38 minutes of a card; the race and the main run give this size's longer budgets
        small.append(job(f"grid-{size}-{budget.replace('e6', 'M')}",
                         f"--preset gptpp-ladder --tokens {budget} --eval-tokens 12.5e6" + (f" --set {sets}" if sets else "")))

# Two queues run at once and must not share a job. The fourth card is shared with short analysis jobs and runs
# about three times slower than a free card, so it gets only what the figures need first: the rest of the first
# seed of the ladder and the position swaps. The main run's three cards, once it is done, take everything else in
# the order the figures need it: activations, warmup, schedules, optimizers, norms, attention, then the race, the
# grid and a second seed for four rungs of the ladder.
def named(prefixes, suffix=""):
    return [j for j in small if j["name"].startswith(prefixes) and j["name"].endswith(suffix)]

# (rung 3 ran out of memory on the crowded fourth card at 05:22 and is run again on the main run's cards; the
# position swaps moved there too once it was clear how slow the shared card is)
fourth = [j for j in named(("ladder-",), "-s0") if j["name"] != "ladder-3-rmsnorm-s0"]
second_seed = [j for j in named(("ladder-",), "-s1") if j["name"].split("-")[1] in ("0", "6", "7", "9")]
three = (named(("ladder-3-rmsnorm-s0",)) + named(("swap-base", "swap-pos-")) + named(("swap-act-",)) + named(("warm-",)) + named(("swap-sched-", "swap-no-warmup")) + named(("opt-",))
         + named(("swap-norm-",)) + named(("swap-attn-", "swap-no-softcap", "swap-tied", "swap-init-", "swap-no-embed-norm", "swap-bias"))
         + named(("race-",)) + named(("grid-",)) + second_seed)
dropped = [j["name"] for j in small if j not in fourth and j not in three]
assert not (set(j["name"] for j in fourth) & set(j["name"] for j in three))
(HERE / "main.jsonl").write_text("".join(json.dumps(j) + "\n" for j in main + three))
(HERE / "small.jsonl").write_text("".join(json.dumps(j) + "\n" for j in fourth))
print(len(main), "main job, then", len(three), "one-card jobs on its cards;", len(fourth), "on the fourth card; not run:", dropped)
