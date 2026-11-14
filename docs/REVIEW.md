# Review of the experiments

Before the series was published, the experiments in this repository were reviewed by a second reader who had not
built them. For each comparison the review asked three things: what differs between the runs besides what is
named; whether the baseline was given a fair chance; and whether the evaluation is clean. The reviewer read the
runs' recorded configurations, opened checkpoints, and made short runs of their own (scripts and logs under
[results/review](../results/review); they replay the lab's data order and step size and reproduce a lab curve to
0.001 nats). This file records what was found, what was run again because of it, and what is still open. It was
written on 14 November 2026, after the last runs.

## Found, and run again

1. **The optimizer comparison handicapped AdamW.** The runs `opt-<rule>-lr*` change only the rule for the hidden
   matrices and keep the rest of the 2026 recipe, including its warmup of 2M tokens (30 steps) and no gradient
   clipping. That schedule was chosen with Muon. AdamW's own recipe at this size has a warmup of 8M tokens and
   clipping at norm one, and the cliff in AdamW's losses above rate 1.2e-3 was the short warmup. Run again as
   `opt-adamw-own-lr*`: the best AdamW improves from 4.214 to 4.181 and Muon's advantage on the finished model
   falls from 0.169 to 0.136 nats. No rule got a schedule of its own beyond that; "Adam" in these runs is AdamW
   without weight decay.
2. **The post-norm runs said nothing about warmup.** In the 2019 recipe at steps of 65,536 tokens (where a warmup
   of 8M tokens is 122 steps) the post-norm model collapsed within 40 steps to predicting token frequencies, with
   warmup and without, and a lower rate did not rescue it at that step size (`results/review/runs/post-*`). Run
   again at steps of 8,192 tokens and rate 6e-4, where the same warmup is 977 steps (`warm-2019-small-*`):
   post-norm trains with a warmup, a little better than pre-norm, and collapses without one, where pre-norm only
   loses a few tenths of a nat.
3. **The sinusoidal positions faded.** With `pos=sinusoidal` under the 2026 recipe the fixed sinusoids were added
   before the embedding norm, to token rows that grow during training: position's share of the energy entering
   the norm fell from 33% to 1.4%. The switch `pos_after_norm` adds them after the norm at unit amplitude
   (`tests/test_sinusoid_share_under_embed_norm.py`); the run `swap-pos-sinusoidal-after-norm` ends 0.060 nats
   behind rotary positions and 0.052 ahead of a learned table, where the first run was 0.133 behind rotary and
   behind the table.
4. **The ladder's seventh step is a whole training recipe.** From rung 6 to rung 7 fourteen training settings
   change together (Muon in place of AdamW on the hidden matrices, the tables' rates, beta1, the decay, the
   clipping, the warmup, the schedule), so the step cannot be credited to the rule. The runs `split-rung7-adamw-*`
   put AdamW (with its own warmup and clipping) back on the hidden matrices under the 2026 recipe and split the
   step in two.

## Found, and not run again

- **The scaling grid did not apply the width rule.** The recipe scales the tables' AdamW rates by
  sqrt(768 / width); `--set d_model=...` on a preset keeps the rates of the preset's own width, so every size of
  the grid (`grid-*`) trained with the rates of width 512 (0.3674 and 0.0098, where the rule gives 0.7348 and
  0.0196 at width 128 and 0.3000 and 0.0080 at width 768). Measured cost at width 128: about 0.03 nats at 12.5M
  tokens, a tilt of roughly 5% on the fitted slope. `tests/test_width_rule_under_overrides.py` states the defect
  (strict xfail). The fit's exponents should be read with that in mind; so should the fact that the two tables
  are 91% to 37% of all parameters across the grid.
- **The trainer's bits per byte use a constant.** The trainer divides by the bytes per token of the whole held-out
  file (4.6283 for FineWeb-EDU). The tokens it actually scores have 4.6353: the main run's 0.9429 would read
  0.9415 (TinyStories: 0.552 would read 0.547). The evaluation protocol of `gptpp.evaluate` counts the bytes of
  each document it scores and is unaffected.
- **Held-out text.** Held-out documents are disjoint from training at the level of documents. 4.1% of the held-out
  64-token blocks also occur verbatim in the 31 training shards the main run read (boilerplate); without them the
  main model's 3.025 nats becomes 3.040. Two of the 1,000 documents scored by the protocol have a near-copy in
  training (0.9322 becomes 0.9324 without them).
- **Untuned alternatives.** Rungs 1 to 6 of the ladder keep the rate chosen for rung 0 at 50M tokens, and the 2019
  side's warmup was never varied on the ladder (at 12 layers a warmup of 32M tokens beat 8M). Dynamic tanh is at
  its default slope. Gains and biases train at one untuned rate (0.01). `swap-norm-post` changes two switches
  (placement and LayerNorm) and `swap-tied` four (tying, the embedding norm, the table's rate, the
  initialization). The race's 2026 side is untuned at that size and has 24M more parameters.
- **Seeds.** One seed for every run but the ladder's. Two seeds of a rung read the same data in a different order
  and are scored on the same held-out tokens, so the spread (0.001 to 0.018 nats, root mean square 0.008) is
  initialization and order only. On the ladder, removing the biases, QK normalization and the soft cap are inside
  it; among the one-change runs, everything within about 0.01.

## Checked and sound

- `--seed` sets both the initialization and the data order.
- Rungs 0 to 6 and 8 to 9 of the ladder differ from their neighbor in exactly one model field, with identical
  training configuration, batch, tokens a step and data order; 24 of the 27 one-change runs differ from the recipe
  in one field. The same seed and configuration run twice gave 3.8156 and 3.8133, and 4.0451 and 4.0448.
- A learned position table trained at the embedding's rate is not handicapped by it (short runs at two other rates).
- The race: same data, tokens a step, context, evaluation windows and seed on both sides; the 2019 preset at
  GPT-2's vocabulary has 124,439,808 parameters and nanoGPT's optimizer settings.
- The main model's trainer loss recomputed from its checkpoint is 3.0249; the gap to the protocol's figure is the
  constant above, the scoring of the first token of a document, and windowing.
- The protocol uses one code path for every model, each model's own tokenizer, windows of at most 1,024 tokens,
  the original bytes, and the same 1,000 documents.

## Not checked

Runs made before the first commit (the first three ladder rungs of seed 0 and the main run record the commit, not
a dirty tree); released models' numbers; the chat stage; the small models trained for the circuits and for the
study of rare tokens.

## The runs made after the review

All on the 8-layer, 512-wide model, FineWeb-EDU, held-out loss in nats a token, one seed (seed 0) each.

**AdamW on the hidden matrices under its own warmup (8M tokens) and clipping (norm one)**, the rest the 2026 recipe, 50M tokens:

| rate | with the recipe's warmup (2M) and no clipping | with its own warmup (8M) and clipping at 1 |
|:--|--:|--:|
| 3e-4 | 4.534 | not run |
| 6e-4 | 4.365 | 4.387 |
| 1.2e-3 | 4.214 | 4.225 |
| 2.4e-3 | 4.731 | 4.181 |
| 4.8e-3 | 5.091 | 4.693 |

Best AdamW of either kind: 4.1815 (rate 2.4e-3, its own warmup and clipping). Muon in the recipe: 4.0451. **Muon's advantage over the best AdamW: +0.136 nats** (it was +0.169 against AdamW under the recipe's schedule). One seed each; the seed yardstick is about 0.016.

**The ladder's step from rung 6 to rung 7, split** (100M tokens; seed 0 for the new runs):

| training | held-out loss |
|:--|--:|
| rung 6: the 2019 training recipe (AdamW on everything at 2.4e-3, decay 0.1, clipping, warmup 8M, cosine) | 3.9848 (seed 0), 3.9827 (seed 1) |
| the 2026 training recipe with AdamW at 1.2e-3 on the hidden matrices (its warmup of 8M tokens and clipping) | 3.9133 |
| the 2026 training recipe with AdamW at 2.4e-3 on the hidden matrices (its warmup of 8M tokens and clipping) | 3.9110 |
| rung 7: the 2026 training recipe (Muon on the hidden matrices) | 3.8277 (seed 0), 3.8290 (seed 1) |

In seed 0 the step is -0.1571. With AdamW at its better rate under the 2026 recipe the loss is 3.9110: **-0.074 for everything in the recipe but the rule (the tables' own rates, no decay on them, the schedule), and -0.083 for Muon in place of AdamW on the hidden matrices.** One seed; two seeds of an AdamW-trained rung have differed by up to 0.018.

**Norm placement and warmup in the 2019 recipe at steps of 8,192 tokens, rate 6e-4** (50M tokens; a warmup of 8M tokens is 977 steps):

| | warmup of 8M tokens | no warmup |
|:--|--:|--:|
| pre-norm | 4.5808 | 4.9497 |
| post-norm | 4.5372 | 7.6962 |

For comparison, at the ladder's step of 65,536 tokens and rate 2.4e-3: pre-norm 4.613 with warmup and 5.706 without; post-norm 7.690 and 7.690 (collapsed to token frequencies, 7.701).

**Positions in the 2026 recipe, 50M tokens** (the sinusoid run again, added after the embedding norm):

| scheme | held-out loss | against the recipe |
|:--|--:|--:|
| rotary (the recipe) | 4.0451 |  |
| ALiBi | 4.0471 | +0.002 |
| sinusoids added after the embedding norm (new) | 4.1050 | +0.060 |
| learned table | 4.1572 | +0.112 |
| sinusoids added before the norm (the first run: a fading signal) | 4.1777 | +0.133 |
| none | 4.3093 | +0.264 |

At 10M, 20M, 30M, 40M tokens the new sinusoid run reads 5.222, 4.762, 4.422, 4.224; the learned table 5.427, 4.867, 4.506, 4.282.
