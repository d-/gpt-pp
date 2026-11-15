# gpt-pp

A small, modern GPT written from scratch in PyTorch, with the code that trains it on one to four GPUs, measures it
against released models, and takes it apart. It is the lab behind the series
["Keys and values"](https://dwood.ai/gpt), a twelve-part introduction to language models from the ground up; every
number and figure of the series is produced by the code in this repository, and the tables below are generated
from the runs' own summaries.

The model is one architecture with switches. Two settings of the switches have names: `gpt2-2019`, the design
published with GPT-2 and trained the way the 2022 reference implementations reproduce it, and `gptpp`, the recipe
of 2026. Every switch can be flipped alone, which is how each step between the two is measured.

## Results

The main model, `gptpp-base` (12 layers, 768 wide, 135.3M parameters), was trained on 2,999,992,320 tokens of FineWeb-EDU in 3.20 hours on 3 GPUs of 32 GB (9.6 card-hours, 263,957 tokens a second, bfloat16, compiled). Its held-out loss at the end, by the trainer's own evaluation, is 3.0249 nats a token, 0.9429 bits a byte (the trainer converts with the bytes per token of the whole held-out file, 4.6283; the tokens it scores have 4.6353, which gives 0.9415).

**Against released models, one protocol** (bits per byte on whole held-out documents, windows of at most 1,024 tokens; lower is better; benchmarks scored by log-likelihood, centered so that chance is 0 and a perfect score 100, mean of nine):

| model | parameters | FineWeb-EDU | WikiText-103 | TinyStories | fresh prose | benchmarks, centered mean |
|:--|--:|--:|--:|--:|--:|--:|
| TinyStories 33M | 69M | 2.875 | 3.380 | 0.436 | 2.716 | 7.1 |
| DistilGPT2 | 82M | 1.084 | 1.201 | 0.964 | 1.393 | 21.4 |
| the 2019 recipe at gpt-pp's shape, 100M tokens | 111M | 1.233 | 1.500 | 1.232 | 1.545 | 12.8 |
| GPT-2 small | 124M | 1.007 | 1.072 | 0.856 | 1.297 | 24.5 |
| SmolLM2 135M | 135M | 0.857 | 0.972 | 0.692 | 1.207 | 35.3 |
| **gpt-pp** | **135M** | **0.932** | **1.130** | **0.864** | **1.325** | **25.8** |
| gpt-pp's recipe, 100M tokens | 135M | 1.129 | 1.383 | 1.103 | 1.468 | 16.5 |
| Pythia 160M | 162M | 0.998 | 1.094 | 0.882 | 1.309 | 25.0 |
| GPT-2 medium | 355M | 0.925 | 0.975 | 0.759 | 1.229 | 30.2 |
| SmolLM2 360M | 362M | 0.788 | 0.869 | 0.618 | 1.132 | 45.4 |
| Qwen3 0.6B base | 596M | 0.817 | 0.888 | 0.638 | 1.155 | 45.8 |
| GPT-2 large | 774M | 0.881 | 0.928 | 0.707 | 1.188 | 33.8 |
| GPT-2 XL | 1558M | 0.854 | 0.898 | 0.679 | 1.163 | 37.8 |

FineWeb-EDU is the main model's training distribution (the documents scored were held out of it); the fresh prose is eight essays published after every model's training data. gpt-pp is ahead of GPT-2 small (124M) on the first and behind it on the other three sets; a released model of its own size trained far longer (SmolLM2 135M) is ahead of it on all four. The protocol, the benchmarks one by one and the hosted models scored on the same items are in [docs/EVAL.md](docs/EVAL.md).

Against nanochat, whose recipe this one draws on, nothing was run like for like. At equal tokens the two are level as far as their own figures can say: its 12-layer model of January 2026 reports 0.9825 bits per byte on its own validation split of the same corpus after 1.08B tokens (later entries of its log at that depth read 0.965 to 0.977), and this recipe with its schedule laid out for 1.08B tokens ends at 0.9742 on the trainer's held-out windows (run `gptpp-base-1B`, 3.4 card-hours). The figures are not one measurement (other documents, another tokenizer, a context of 2,048 tokens there), and this lab's own two ways of scoring one model are 0.011 apart, so neither recipe is shown ahead.

**The ladder**: an 8-layer, 512-wide model trained on 100M tokens, from the 2019 recipe to the 2026 recipe one change at a time (held-out loss, nats a token; two seeds where two were run):

| step | change | seed 0 | seed 1 | mean | against the step before |
|--:|:--|--:|--:|--:|--:|
| 0 | the 2019 recipe | 4.144 | 4.126 | 4.135 |  |
| 1 | no biases | 4.133 | 4.136 | 4.135 | -0.000 |
| 2 | rotary positions | 4.022 | 4.026 | 4.024 | -0.110 |
| 3 | RMS norm without parameters | 4.048 | 4.063 | 4.056 | +0.031 |
| 4 | squared ReLU | 4.026 | 4.020 | 4.023 | -0.033 |
| 5 | cosine scores (QK normalization) | 4.010 | 4.013 | 4.011 | -0.011 |
| 6 | zero-initialized writers | 3.985 | 3.983 | 3.984 | -0.027 |
| 7 | the 2026 training recipe (Muon, its rates and schedule) | 3.828 | 3.829 | 3.828 | -0.155 |
| 8 | untied tables, normalized embedding | 3.815 | 3.819 | 3.817 | -0.011 |
| 9 | logit soft cap: the 2026 recipe | 3.816 | 3.812 | 3.814 | -0.003 |

Two seeds of one step differ by 0.001 to 0.018 nats (root mean square 0.008): a difference under about 0.016 nats between two single runs cannot be told from noise.

Step 7 changes the whole training recipe, not the optimizer's rule alone (fourteen settings: the rule, the tables' rates, the decay, the clipping, the warmup, the schedule). Two more runs split it in seed 0: the same model under the 2026 training recipe with AdamW (its own warmup of 8M tokens and clipping) in place of Muon on the hidden matrices ends at 3.911, between 3.985 and 3.828: -0.074 for the rest of the recipe and -0.083 for Muon itself.

**One change at a time from the 2026 recipe** (the same 8-layer model, 50M tokens, one seed each; the recipe itself ends at 4.045 nats a token; the change in held-out loss when one component is swapped back or replaced, so a positive number is a component that earns its place):

| component | swapped for | change |
|:--|:--|--:|
| positions | learned table | +0.112 |
| positions | sinusoidal (added after the embedding norm) | +0.060 |
| positions | sinusoidal added before the embedding norm, where it fades as the token rows grow (first run; see docs/REVIEW.md) | +0.133 |
| positions | ALiBi | +0.002 |
| positions | none | +0.264 |
| activation | ReLU | +0.018 |
| activation | GELU | +0.001 |
| activation | SiLU | +0.007 |
| activation | SwiGLU | -0.008 |
| activation | GeGLU | -0.002 |
| normalization | LayerNorm | -0.019 |
| normalization | RMSNorm with a gain | -0.020 |
| normalization | dynamic tanh | +0.031 |
| normalization | post-norm | +0.088 |
| normalization | none | diverged (loss not finite at step 80) |
| attention | no QK normalization | +0.029 |
| attention | learned gain | +0.001 |
| attention | 4 key-value heads | +0.003 |
| attention | 2 key-value heads | +0.030 |
| attention | 1 key-value head | +0.050 |
| the rest | with biases | -0.021 |
| the rest | GPT-2's scaled initialization | -0.001 |
| the rest | tied tables | -0.000 |
| the rest | no soft cap | -0.002 |
| the rest | no embedding norm | +0.014 |
| schedule | cosine | +0.031 |
| schedule | linear | +0.011 |
| schedule | constant | +0.150 |
| schedule | no warmup | -0.008 |

At this size and budget the recipe's simplifications cost or gain two hundredths of a nat at most, and two of them cost: a gain on the norm and biases on the linear maps each lower the loss by about 0.02 when put back, in both of two seeds (below). The squared ReLU is 0.017 ahead of ReLU and level with GELU; two tables and the soft cap show nothing. Positions that attention can read as differences (rotary or ALiBi), the training recipe, QK normalization and pre-norm are what the loss depends on. Post-norm here also swaps in LayerNorm (against pre-norm with the same LayerNorm it is a tenth of a nat worse); dynamic tanh is at the best of three slopes (`swap-norm-dyt-a*`).

**Second seeds.** The recipe and 13 of these runs were made again with seed 1 (the recipe: 4.045 and 4.043). Under the 2026 recipe two seeds differ by 0.001 to 0.008 nats (root mean square 0.005), less than on the ladder, most of whose rungs are trained with AdamW. The change against the recipe of the same seed:

| swapped for | seed 0 | seed 1 |
|:--|--:|--:|
| learned table | +0.112 | +0.108 |
| sinusoidal (added after the embedding norm) | +0.060 | +0.058 |
| ReLU | +0.018 | +0.016 |
| GELU | +0.001 | -0.002 |
| SiLU | +0.007 | +0.006 |
| SwiGLU | -0.008 | -0.011 |
| LayerNorm | -0.019 | -0.021 |
| RMSNorm with a gain | -0.020 | -0.021 |
| no QK normalization | +0.029 | +0.032 |
| with biases | -0.021 | -0.013 |
| no embedding norm | +0.014 | +0.008 |
| linear | +0.011 | +0.011 |
| no warmup | -0.008 | +0.002 |

**The ladder's 2019 rungs with a longer warmup.** The 2019 recipe's warmup at this size was 8M tokens. With 32M tokens (runs `ladder-*-w32M`, seed 0) each of its seven rungs is 0.017 to 0.042 nats better, the steps between them hardly move, and step 7 is -0.140 in place of -0.157 in that seed: the table above keeps the runs as first made, and its 2019 side sits about 0.03 high.

**The rule for the hidden matrices** (the same 8-layer model, 50M tokens, seed 0; only the rule and its rate change and the two tables keep their AdamW rates; each rule under the recipe's schedule, a warmup of 2M tokens and no clipping, and under its own, a warmup of 8M tokens and clipping at norm one):

| rule | rates tried: held-out loss | best | behind Muon |
|:--|:--|--:|--:|
| Muon (the recipe) | 0.01: 4.067, 0.02: 4.045, 0.04: 4.071 | 4.045 |  |
| AdamW, its own schedule (warmup of 8M tokens, clipping at norm one) | 6e-4: 4.387, 1.2e-3: 4.225, 2.4e-3: 4.181, 4.8e-3: 4.693 | 4.181 | +0.136 |
| AdamW, its own schedule and a cosine to a tenth | 1.2e-3: 4.264, 2.4e-3: 4.207 | 4.207 | +0.162 |
| AdamW, the recipe's schedule | 3e-4: 4.534, 6e-4: 4.365, 1.2e-3: 4.214, 2.4e-3: 4.731, 4.8e-3: 5.091 | 4.214 | +0.169 |
| Adam (AdamW without weight decay), the recipe's schedule | 3e-4: 4.533, 6e-4: 4.364, 1.2e-3: 4.225, 2.4e-3: 4.729, 4.8e-3: 5.182 | 4.225 | +0.180 |
| Lion, its own schedule | 1e-4: 4.560, 3e-4: 4.484, 1e-3: 7.108 | 4.484 | +0.439 |
| Lion, the recipe's schedule | 3e-5: 4.838, 1e-4: 4.523, 3e-4: 4.613, 1e-3: 5.006 | 4.523 | +0.478 |
| SGD with Nesterov momentum, its own schedule | 0.3: 4.758, 1: 4.536, 3: 4.586 | 4.536 | +0.491 |
| SGD with Nesterov momentum, the recipe's schedule | 0.03: 5.243, 0.1: 4.949, 0.3: 4.719, 1: 6.542 | 4.719 | +0.673 |

The recipe's schedule was chosen with Muon; given the warmup and the clipping of its own recipe, AdamW closes a fifth of the gap, and SGD and Lion improve and stay far behind. In a second seed the best AdamW ends at 4.206 and the recipe at 4.043: Muon's advantage is 0.136 and 0.163 in the two seeds. With one rate for every parameter group, the way of 2019, AdamW and Adam end near 5.0 at every rate on this model (runs `optflat-*`): a caution about the parametrization, not a comparison of rules.

**Norm placement and warmup in the 2019 recipe** (the 8-layer model, 50M tokens, one seed each; a warmup of 8M tokens):

| | steps of 8,192 tokens, rate 6e-4: warmup (977 steps) | no warmup | steps of 65,536 tokens, rate 2.4e-3: warmup (122 steps) | no warmup |
|:--|--:|--:|--:|--:|
| pre-norm | 4.581 | 4.950 | 4.613 | 5.706 |
| post-norm | 4.537 | 7.696 | 7.690 | 7.690 |

A loss near 7.70 is a collapse to predicting token frequencies (their entropy is 7.701 nats). Post-norm trains, and as well as pre-norm, when its warmup is long enough in steps; without one it collapses where pre-norm only loses a few tenths.

**The two recipes at the main model's shape**, 100M tokens each (held-out loss, nats a token; bits a byte):

| run | held-out loss | bits a byte |
|:--|--:|--:|
| `race-2019-lr1.2e-3-100M` | 4.032 | 1.257 |
| `race-2019-lr1.2e-3-w8M-100M` | 4.073 | 1.269 |
| `race-2019-lr2.4e-3-100M` | 3.991 | 1.244 |
| `race-2019-lr2.4e-3-w8M-100M` | 4.088 | 1.274 |
| `race-2019-lr4.8e-3-100M` | 4.030 | 1.256 |
| `race-2019-lr6e-4-100M` | 4.155 | 1.295 |
| `race-2026-100M` | 3.654 | 1.139 |
| `race-2026-lr0.01-100M` | 3.653 | 1.139 |
| `race-2026-lr0.04-100M` | 3.679 | 1.147 |
| `race-2026-ropebase1e3-100M` | 3.673 | 1.145 |
| `race-2026-ropebase1e5-100M` | 3.649 | 1.137 |

An independent review of these experiments, made before publication, found several comparisons that were not what they were described as; the affected ones were run again and the tables above hold the corrected runs. What it found, what was run again and what remains open: [docs/REVIEW.md](docs/REVIEW.md).

Every run's log and summary is under [results/runs](results/runs); the series reports them in its parts [5](https://dwood.ai/gpt-05), [6](https://dwood.ai/gpt-06) and [7](https://dwood.ai/gpt-07).

## The two recipes

| component | `gpt2-2019` | `gptpp` (2026) |
|:--|:--|:--|
| position | learned vector per position, added to the embedding | rotary: queries and keys turned by their position |
| normalization | LayerNorm with gain and bias, before each sublayer | RMSNorm without parameters, also after the embedding |
| MLP | GELU, hidden width 4d | squared ReLU, hidden width 4d |
| attention score | q·k / sqrt(d_head) | 12 · cos(q, k) |
| biases | in every linear layer | none |
| embedding and unembedding | one shared matrix | two matrices |
| logits | unbounded | 15 · tanh(logits / 15) |
| initialization | N(0, 0.02²), writers to the stream scaled by 1/sqrt(2L) | writers to the stream start at zero |
| optimizer | AdamW on everything, weight decay 0.1 | Muon on the hidden matrices, AdamW on the rest, no weight decay |
| learning rate | linear warmup, cosine decay to a tenth | short warmup, flat, linear decay over the last 65% to a twentieth |
| gradient clipping | global norm 1 | none |

Why each choice was made, where the sources disagree and what was measured before choosing: [docs/DESIGN.md](docs/DESIGN.md).

## Sizes

| preset | layers | width | heads | context | vocabulary | parameters | trained on |
|:--|--:|--:|--:|--:|--:|--:|:--|
| `gptpp-pico` | 6 | 128 | 4 | 256 | 4,096 | 2.2M | TinyStories; small enough to run in a web page |
| `gptpp-ladder` | 8 | 512 | 8 | 512 | 32,768 | 58.7M | FineWeb-EDU; one model per design change |
| `gptpp-base` | 12 | 768 | 12 | 1,024 | 32,768 | 135.3M | FineWeb-EDU; the main model |

and the same three under `gpt2-2019-*`.

## Quick start

```sh
pip install -r requirements.txt
export GPTPP_WORK=/path/to/workspace          # data, checkpoints and logs go here, never into the repository

# tokenizers and data (public datasets from the Hugging Face Hub; licenses in docs/DATA.md)
python scripts/prepare_tinystories.py         # 543M tokens, vocabulary 4,096
python scripts/prepare_fineweb.py             # 9.9B tokens of FineWeb-EDU, vocabulary 32,768

# the smallest model, about ten minutes on one card
python -m gptpp.train --preset gptpp-pico --run pico
python -m gptpp.generate --run pico --prompt "Once upon a time" --max-new-tokens 200

# the main model on three or four cards
python -m torch.distributed.run --standalone --nproc_per_node=3 -m gptpp.train --preset gptpp-base --run gptpp-base --tokens 3.0e9 \
    --set device_batch=48 --eval-tokens 50e6 --save-log 24 --save-steps 2000,4000,6000,8000,10000,12000,14000,16000,18000,20000
# (device_batch=48: the preset's 64 sequences of 1,024 tokens do not fit on a 32 GB card)

# one change at a time: any field of the model or training configuration
python -m gptpp.train --preset gptpp-ladder --run swap-act-gelu --tokens 50e6 --set act=gelu
# another rule for the hidden matrices only (the two tables keep their AdamW rates): <adamw|adam|lion|sgd>+adamw
python -m gptpp.train --preset gptpp-ladder --run opt-adamw --tokens 50e6 --set optimizer=adamw+adamw lr=1.2e-3 weight_decay=0.1 beta1=0.9

# measure: bits per byte on held-out documents, and benchmarks scored by log-likelihood
python -m gptpp.evaluate   --model gptpp-base --docs fineweb-edu,wikitext103,tinystories,fresh --out results/eval/gptpp-base.json
python -m gptpp.benchmarks --model gptpp-base --tasks all --out results/eval/gptpp-base.json
```

A list of jobs runs unattended, one job a card, with `python scripts/job_queue.py experiments/main.jsonl --gpus 0,1,2,3`;
`experiments/make_jobs.py` writes the lists behind the results above.

## Weights

The trained models are on the Hugging Face Hub, every weight in float32 as trained, in the format of transformers:

| model | what it is |
|:--|:--|
| [david-rename/gpt-pp-base](https://huggingface.co/david-rename/gpt-pp-base) | the main model: 135M parameters, 3.0B tokens of FineWeb-EDU |
| [david-rename/gpt-pp-chat](https://huggingface.co/david-rename/gpt-pp-chat) | the main model after the chat stage (SmolTalk) |
| [david-rename/gpt-pico](https://huggingface.co/david-rename/gpt-pico) | gpt-pico, the smallest size (the preset `gptpp-pico`): 2.2M parameters, TinyStories |

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("david-rename/gpt-pp-base", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("david-rename/gpt-pp-base", trust_remote_code=True)
```

The 2026 recipe is transformers' NanoChat architecture with one addition, a gain on the normalized queries, which
the exported model carries as a few lines of code (`gptpp/hf/`, hence `trust_remote_code=True`). `scripts/export_hf.py`
writes a run in that format and checks its logits against gpt-pp (`tests/test_export_hf.py` does the same on a small
random model); in float64 the two agree to 1e-13.

They serve with vLLM (checked with 0.33.0, which runs them through its Transformers backend; it needs the `ninja`
build tool on the `PATH`): `vllm serve david-rename/gpt-pp-chat --trust-remote-code --max-model-len 1024`.

## What is in the repository

| path | what |
|:--|:--|
| `gptpp/model.py` | the model: one class, every historical variant behind a switch; hooks on every activation (`run_with_cache`, `run_with_hooks`) |
| `gptpp/hf/` | the code an exported model carries for transformers: NanoChat with the query gain |
| `gptpp/optim.py` | Muon and Lion; the choice among them and PyTorch's AdamW, Adam and SGD; parameter groups; schedules |
| `gptpp/train.py`, `data.py` | the trainer (one to four GPUs, bfloat16, compiled) and the token-shard loader |
| `gptpp/bpe.py`, `tokenizer.py` | byte-pair encoding from scratch, and the fast encoder checked against it |
| `gptpp/generate.py` | sampling: temperature, top-k, top-p, min-p; decoding with a key-value cache |
| `gptpp/evaluate.py`, `benchmarks.py` | bits per byte under one protocol for our models and released ones; nine benchmarks by log-likelihood |
| `gptpp/chat.py` | the stage after pretraining: from completing text to answering |
| `gptpp/interp/` | circuits (QK and OV, composition, path expansion), lenses, patching, feed-forward memories, sparse autoencoders, transcoders, attribution graphs, grokking, n-gram baselines |
| `gptpp/export/` | one exporter per figure of the series: the data each figure draws |
| `scripts/` | data preparation, tokenizer training, the job queue, evaluation of released and hosted models |
| `experiments/` | the job lists of every run reported |
| `results/` | the small files the runs wrote: logs, summaries, evaluations, interpretability measurements; `results/review/` holds the reviewer's short runs |
| `docs/` | [DESIGN.md](docs/DESIGN.md) (every decision, and what the runs measured about it), [DATA.md](docs/DATA.md), [EVAL.md](docs/EVAL.md), [REVIEW.md](docs/REVIEW.md) (an independent review of the experiments: what it found, what was run again, what is open), [KNOWN-ISSUES.md](docs/KNOWN-ISSUES.md) (quirks, known issues and bugs of the code, each with its evidence) |
| `tests/` | the test suite (`pytest -q`) |

## Data

TinyStories (Eldan and Li, 2023; CDLA-Sharing-1.0) and FineWeb-EDU (Penedo et al., 2024; ODC-By 1.0), both from
the Hugging Face Hub at pinned revisions; sources, hashes, licenses and corpus statistics are in
[docs/DATA.md](docs/DATA.md). No data, checkpoint or log is committed; the trained weights are on the Hugging Face Hub (see Weights).

## Lineage

The 2019 recipe follows GPT-2 (Radford et al., 2019) as nanoGPT reproduces it. The 2026 recipe draws on the
NanoGPT speedrun (modded-nanogpt: Muon, rotary positions, QK normalization, squared ReLU, zero-initialized
projections, untied embeddings, the logit soft cap) and on nanochat. What was taken from each, what was changed
and why is in [docs/DESIGN.md](docs/DESIGN.md), sections 3 and 4.
