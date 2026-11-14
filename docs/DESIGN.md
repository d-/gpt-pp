# Design of the core: model, optimizers, trainer

This file records every design decision of the model (`gptpp/model.py`), the optimizers (`gptpp/optim.py`), the
data loader (`gptpp/data.py`) and the trainer (`gptpp/train.py`): what was chosen, where it comes from, why, and
what we did where the sources disagree. Numbers quoted from our own runs say how they were measured.

The model is one architecture with switches. Two settings of the switches have names:

- **`gpt2-2019`**: the design published with GPT-2 (Radford et al., 2019), trained the way nanoGPT reproduces it.
- **`gptpp`**: the 2026 recipe, the default of `ModelConfig` and `TrainConfig`.

Every switch can be flipped alone (`--set act=gelu`), which is how the effect of each step from 2019 to 2026 is
measured.

## 1. The two recipes side by side

| component | `gpt2-2019` | `gptpp` (2026) | field |
|:--|:--|:--|:--|
| position | learned vector per position, added to the token embedding | rotary: queries and keys rotated by their position | `pos` |
| normalization | LayerNorm with gain and bias, before each sublayer and before the unembedding | RMSNorm without parameters, in the same places and after the embedding | `norm`, `norm_placement`, `embed_norm` |
| MLP | GELU, hidden width 4d | squared ReLU, hidden width 4d | `act`, `mlp_mult` |
| attention score | q·k / sqrt(d_head) | 12 · cos(q, k) | `qk_norm`, `qk_scale` |
| biases | in every linear layer | none | `bias` |
| embedding and unembedding | one shared matrix | two matrices | `tie_embeddings` |
| logits | unbounded | 15 · tanh(logits / 15) | `logit_softcap` |
| initialization | N(0, 0.02²); matrices that write to the stream scaled by 1/sqrt(2L) | matrices that write to the stream start at zero | `init` |
| optimizer | AdamW on everything, betas (0.9, 0.95), weight decay 0.1 on matrices | Muon on the hidden matrices; AdamW, betas (0.8, 0.95), on the rest; no weight decay | `optimizer`, `lr*`, `weight_decay` |
| learning rate | 6e-4 at the base size (2.4e-3 at the ladder size, 6e-3 at pico: Section 5.3), linear warmup, cosine decay to a tenth | 0.02 for Muon, 0.3 and 0.008 (at width 768) for embedding and unembedding; short warmup, flat, linear decay over the last 65% to a twentieth | `schedule`, `warmup_tokens`, `cooldown_frac`, `final_lr_frac` |
| gradient clipping | global norm 1 | none | `grad_clip` |

Parameters (counted by `GPT.num_params()`; the test suite checks that the 2019 recipe with GPT-2's vocabulary of
50,257 has exactly the 124,439,808 parameters of GPT-2 small):

| preset | embedding | position | unembedding | attention | MLP | norm | total |
|:--|--:|--:|--:|--:|--:|--:|--:|
| `gpt2-2019-pico` | 524,288 | 32,768 | 0 (tied) | 396,288 | 790,272 | 3,328 | 1,746,944 |
| `gptpp-pico` | 524,288 | 0 | 524,288 | 393,216 | 786,432 | 0 | 2,228,224 |
| `gpt2-2019-ladder` | 16,777,216 | 262,144 | 0 (tied) | 8,404,992 | 16,797,696 | 17,408 | 42,259,456 |
| `gptpp-ladder` | 16,777,216 | 0 | 16,777,216 | 8,388,608 | 16,777,216 | 0 | 58,720,256 |
| `gpt2-2019-base` | 25,165,824 | 786,432 | 0 (tied) | 28,348,416 | 56,669,184 | 38,400 | 111,008,256 |
| `gptpp-base` | 25,165,824 | 0 | 25,165,824 | 28,311,552 | 56,623,104 | 0 | 135,266,304 |

The two recipes have the same trunk to within the biases and gains (84.9M parameters of attention and MLP at the
base size). The 2026 model is larger only by its second vocabulary matrix.

## 2. The decisions of the 2026 recipe

Each entry: what, source, reason, and what the code does exactly.

### 2.1 Rotary positions

*Source.* Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding", 2021, arXiv:2104.09864.

*Reason.* A learned or sinusoidal position vector is added to the token embedding, so every layer has to separate
"what" from "where" in the same residual stream, and a position beyond the trained length has either no vector
(learned) or one never seen in training. Rotary embedding puts position where it is used: each pair of coordinates of a
query or key is rotated by an angle proportional to its position, and the dot product of a query at m with a key at
n then depends on m − n only. Nothing is added to the stream and there are no parameters.

*In the code.* `rotate()` pairs coordinate i with coordinate i + d_head/2 and rotates the pair at position m by
m · base^(−2i/d_head), counter-clockwise, with base 10,000 (`rope_base`). Tables of cosines and sines are computed in
float64, stored as float32 buffers outside the checkpoint, and extended when a longer sequence arrives. Under
autocast the rotation is computed in float32 and cast back to bfloat16.

### 2.2 Pre-norm with a parameter-free RMSNorm, also after the embedding

*Source.* Pre-norm: GPT-2 (2019), which moved layer normalization to the input of each sublayer and added one
after the last block; analyzed by Xiong et al., "On Layer Normalization in the Transformer Architecture", 2020,
arXiv:2002.04745. RMSNorm: Zhang and Sennrich, 2019, arXiv:1910.07467. Dropping the gain: modded-nanogpt (record
of 14 October 2024). Normalizing the embedding: modded-nanogpt (record of 3 November 2024, the one that untied the
embedding). nanochat has both.

*Reason.* What normalization buys is that a sublayer's input has a fixed scale; centering (the difference between
LayerNorm and RMSNorm) adds cost and little else, and a gain was expected to be redundant with the linear layer that follows it (the runs say otherwise at 8 layers: Section 8). With no
parameters the layer also needs no optimizer group. The extra norm after the embedding puts every token on the
same sphere when it enters the stream: the rows of an untied embedding trained at a high learning rate do not
keep their scale (in our 8-layer probe their root-mean-square went from 1 at initialization to between 2.5 and 5.1,
1st to 99th percentile, in 763 steps).

*In the code.* `RMSNorm(gain=False)` is `x / sqrt(mean(x²) + 1e-6)`. It is applied to the input of attention and of
the MLP (`norm_placement=pre`), to the stream before the unembedding, and to the embedding (`embed_norm`). The
residual stream is kept in float32 under autocast; the norms read it in float32.

*Deviation from the brief.* `embed_norm` is not in the brief's list of the recipe. It is on in `gptpp` because it
belongs with the untied embedding (modded-nanogpt introduced the two in the same record, 3 November 2024, and
nanochat has both). In an early 50M-token probe the run without it was 0.007 nats worse (4.077 against 4.071, Section
5.3); in the finished recipe it is 0.014 worse (Section 8), a little over twice the largest gap seen between two
seeds. `--set embed_norm=false` removes it.

### 2.3 Squared-ReLU MLP of width 4

*Source.* So et al., "Primer: Searching for Efficient Transformers for Language Modeling", 2021, arXiv:2109.08668,
where squaring the ReLU was the most effective single change found by an evolutionary search. modded-nanogpt adopted
it on 14 October 2024 and nanochat uses it.

*Reason.* It is as cheap as ReLU, its derivative is continuous at zero, and its output grows quadratically with how
well the input matches the neuron, so strong matches count for more.

*In the code.* `relu2(x) = relu(x)²`, `fc`: d → 4d, `proj`: 4d → d, no biases.

### 2.4 QK normalization

*Source.* Henry et al., "Query-Key Normalization for Transformers", 2020, arXiv:2010.04245 (L2-normalize queries and
keys, multiply by a learned scalar); Loshchilov et al., "nGPT", 2024, arXiv:2410.01131 (the scale that restores unit
variance of the scores is sqrt(d_head)); modded-nanogpt (14 October 2024) and nanochat.

*Reason.* Without it the size of a score depends on the norms of the query and the key, so a head can saturate its
softmax by growing its weights, and the sharpness of attention is entangled with the weight scale that the
optimizer and weight decay also act on. With it a score is a cosine times a scale: bounded, and independent of the
norm of the weights.

*In the code.* Queries and keys are RMS-normalized over the head dimension, the query is multiplied by a gain,
and the usual division by sqrt(d_head) follows; the gain is chosen so that

    score = qk_scale · cos(q, k).

`qk_scale` is the inverse temperature of the softmax: the largest possible gap between two scores is twice the
scale. It is the constant 12 for every head. With `qk_scale_learned` each head multiplies it by a learned factor
that starts at one (a parameter of the "gains" optimizer group). Normalization comes before the rotation; the
order does not matter, since a rotation preserves norms.

*Why 12.* The sources disagree (Section 3), and all of them were tuned at a head dimension of 128. We measured
(Section 5.3). On the 8-layer model (head dimension 64, context 512) a scale of 8, which is sqrt(d_head), the value
that gives unit-variance scores at initialization, was worse than 11.5 or 16 by 0.014 to 0.019 nats, where two
seeds of the same configuration differed by 0.0006; 11.5 and 16 were within 0.005 of each other. On pico (head
dimension 32, context 256) 5.7 was worse than the others by 0.012 to 0.015, and 8.1, 11.3 and 15 were within 0.003.
When the scale is learned, starting from sqrt(d_head), the heads end at 14.9 on average on the 8-layer model (8.2
in the first layer, 14 to 18 above it) and at 12.6 on pico, and the loss is lower by a further 0.004 to 0.005. A
constant of 12 sits on the flat part of both curves and keeps the recipe free of learned gains.

### 2.5 No biases

*Source.* nanoGPT's `bias=False` option ("a bit better and faster"), modded-nanogpt, nanochat; PaLM (Chowdhery et al.,
2022, arXiv:2204.02311) reports removing biases for stability.

*Reason.* With normalized inputs and a wide layer a bias was expected to add almost nothing, and every remaining
parameter is then a matrix or an embedding, which is what the optimizers are built for. (Measured, Section 8: with
biases put back the finished recipe is about 0.02 nats better at 8 layers and 50M tokens.)

### 2.6 Untied embedding and unembedding

*Source.* Tying: Press and Wolf, 2017, arXiv:1608.05859; Inan et al., 2017, arXiv:1611.01462; GPT-2 ties. Untying:
modded-nanogpt, record of 3 November 2024; nanochat.

*Reason.* The embedding is a lookup table whose rows are updated only when their token occurs; the unembedding is a
dense layer feeding the loss. Untied, each gets its own geometry and its own learning rate (they differ by a factor
of almost forty here). The price is V · d parameters (25.2M at the base size), which are cheap: the extra embedding costs a
lookup, no matrix product.

### 2.7 Zero-initialized residual projections

*Source.* Zhang, Dauphin and Ma, "Fixup Initialization", 2019, arXiv:1901.09321; Bachlechner et al., "ReZero is All
You Need", 2020, arXiv:2003.04887; modded-nanogpt (14 October 2024) and nanochat.

*Reason.* If the matrices that write into the residual stream (the attention output matrix and the MLP's second
matrix) start at zero, every block starts as the identity. The network begins as a bigram model, embed then
unembed, and the blocks grow in as their gradients ask; no depth-dependent scaling is needed.

*In the code (`init=zero_proj`).* Those matrices are zero. The other hidden matrices are uniform in ±1/sqrt(fan_in),
the default of `torch.nn.Linear`. An untied embedding is N(0, 1), the scale of the normalized stream; an untied
unembedding is N(0, 0.001²), so the first logits are near zero and the first loss is log V. A tied matrix keeps
GPT-2's N(0, 0.02²). The test suite checks that the model at initialization is exactly embed then unembed.

### 2.8 Logit soft-cap

*Source.* Bello et al., "Neural Combinatorial Optimization with Reinforcement Learning", 2016, arXiv:1611.09940 (tanh
clipping of logits); Gemma 2, 2024, arXiv:2408.00118 (soft-cap of 30 on the final logits); modded-nanogpt (6 November
2024 at 30, lowered to 15 on 4 January 2025); nanochat (15).

*Reason.* Cross-entropy rewards pushing the logit of a sure token ever higher. The cap bounds the logits smoothly, so
that capacity is not spent there and no single token can produce an enormous loss or gradient.

*In the code.* `logits = 15 · tanh(logits / 15)`, computed in float32.

### 2.9 Muon for the hidden matrices, AdamW for the rest

*Source.* Jordan et al., "Muon: An optimizer for hidden layers in neural networks", 2024,
https://kellerjordan.github.io/posts/muon/; Bernstein and Newhouse, "Old Optimizer, New Norm: An Anthology", 2024,
arXiv:2409.20325; the implementations in modded-nanogpt and nanochat, and https://github.com/KellerJordan/Muon. Our
implementation is our own (`gptpp/optim.py`).

*What it does.* For each hidden matrix: an exponential moving average of the gradient (momentum 0.95), the Nesterov
combination of that average and the current gradient, then orthogonalization: the update U S Vᵀ is replaced by
approximately U Vᵀ, all singular values set near one, by five Newton-Schulz steps
X ← aX + b(XXᵀ)X + c(XXᵀ)²X with (a, b, c) = (3.4445, −4.7750, 2.0315). The step is
lr · sqrt(max(1, rows/columns)) times that matrix.

*Reason.* The gradient of a hidden matrix is close to low rank: a few directions carry most of it. Adam rescales
coordinates; Muon rescales directions, giving every singular direction of the update the same size. The step is
steepest descent under the spectral norm, and `lr` is the size of the step in that norm whatever the size of the
gradient.

*Choices.*
- Tuned coefficients, five steps: singular values above about 0.0015 of the Frobenius norm end in [0.68, 1.20]
  (checked in `tests/test_optim.py`). The iteration runs in bfloat16 on the GPU, as in the references.
- Q, K, V and O are separate matrices (the Muon write-up reports that orthogonalizing them separately is better).
- Matrices of one shape are stacked and orthogonalized in one batched call: three calls a step for the default model.
- Embedding, unembedding and any gains or biases use AdamW with betas (0.8, 0.95) and eps 1e-10. Rates at width
  768: embedding 0.3, unembedding 0.008, nanochat's current values, scaled by sqrt(768 / d_model) as nanochat
  scales them (0.367 and 0.0098 at width 512; 0.735 and 0.0196 at width 128); gains and biases (absent from the
  default recipe) 0.01. Against nanochat's values of October 2025 (0.2 and 0.004) these were better by 0.010 nats
  on the 8-layer model and by 0.019 on pico; without the scaling by width pico was worse by 0.066 (Section 5.3).
- Muon's rate of 0.02 was the best of 0.01, 0.02 and 0.04 at 65,536 tokens a step on both sizes (Section 5.3), an
  eighth of the batch the references tuned it at, so it is not scaled with the batch (`lr_ref_batch` is off).
- No weight decay anywhere in the default.
- If the embedding is tied to the unembedding (the 2019 model trained the 2026 way), the one matrix is trained at
  the unembedding's rate. At the embedding's rate, 37 to 50 times larger, the loss rose in the first 14 steps
  instead of falling (from 10.7 to 12.5 in one configuration, from 10.5 to 145 in another).
- The cost. Muon's extra matrix products cost about 5 · d_model / (tokens per step per card) of a step's
  FLOPs: under 1% at the 524,288-token batches of the references, 4% to 12% at ours (Section 5.2).

### 2.10 Schedule

Linear warmup over the first 2M tokens, then constant, then linear decay over the last 65% of the budget to 5% of the
peak (`schedule=trapezoid`). The warmup protects the first Adam steps, which divide by a second moment estimated
from a handful of gradients. The shape is the one both references converged on (modded-nanogpt: constant, then linear
to zero over the last 70%; nanochat: 40 steps of warmup, linear decay over the last 65% to 0.05).

All schedules are functions of tokens seen, read at the middle of each step, so they do not depend on the batch size
or the number of cards.

### 2.11 Precision and speed

- bfloat16 autocast for the matrix products; the residual stream, the norms, the soft-cap and the loss in float32.
- TF32 for the remaining float32 products; `torch.compile` on the whole forward pass.
- Attention through `F.scaled_dot_product_attention` (FlashAttention kernels; Dao et al., 2022, arXiv:2205.14135).
- Fused AdamW. The data loader assembles a batch on the CPU while the card works on the previous step.

## 3. Where the sources disagree, and what we chose

| question | GPT-2 / nanoGPT | modded-nanogpt | nanochat | ours | why |
|:--|:--|:--|:--|:--|:--|
| scale of QK-normalized scores | no QK norm | 0.12 · d_head · cos with d_head 128 (15.4 cos) | sqrt(d_head) · cos in October 2025; 1.44 · sqrt(d_head) · cos now | `qk_scale` · cos with `qk_scale` = 12, whatever the head dimension | measured: Section 2.4 |
| rotary base | no rotary | 1,024, and half of the pairs not rotated | 10,000, then 100,000 | 10,000 | the value of the paper; ours to vary (`rope_base`) |
| hidden-matrix initialization | N(0, 0.02²) | N(0, 1/(3 fan_in)) | uniform with standard deviation 1/sqrt(fan_in); 0.4 of that for the first MLP matrix | uniform ±1/sqrt(fan_in), variance 1/(3 fan_in) | the PyTorch default; at width 768 its standard deviation is 0.0208, GPT-2's 0.02, so `init` changes one thing |
| unembedding initialization | tied, 0.02 | zero | zero in 2025, 0.001 now | N(0, 0.001²) | from exactly zero the first step moves only the unembedding |
| embedding initialization | 0.02 | 1 | 1 in 2025, 0.8 now | 1 | the scale of the normalized stream |
| soft-cap | none | 30 · tanh(x/30), then 15; 15 · x / sqrt(x² + 225) in the simple baseline | 15 · tanh(x/15) | 15 · tanh(x/15) | the common form (Gemma 2 uses it too) |
| AdamW rates of embedding, unembedding | 6e-4 for all | 0.7, 0.004 (simple baseline) | 0.2, 0.004 in 2025; 0.3, 0.008 now; times sqrt(768/d) | 0.3, 0.008, times sqrt(768/d) | measured: Section 2.9 |
| Muon rate | no Muon | 0.025 with weight decay 0.05 (simple baseline) | 0.02; times sqrt(batch / 524,288) now | 0.02, not scaled with the batch | measured: Section 2.9 |
| AdamW rate of the 2019 recipe | 6e-4 for GPT-2 small on 300B tokens | | | 6e-4 at the base size; 2.4e-3 (ladder) and 6e-3 (pico) | measured: at 6e-4 the 8-layer model was 0.50 nats behind its own best after 50M tokens (Section 5.3); the base size keeps 6e-4 as its preset, and the race later tried four rates there (2.4e-3 was the best, with a warmup of 32M tokens: README, "The two recipes at the main model's shape") |
| weight decay | 0.1 on matrices and embeddings | 0.05 on Muon's matrices, 0.001 on Adam's | 0 in 2025; a scheduled "cautious" decay now | 0 | the simplest setting that both references started from |
| schedule | warmup 2,000 steps, cosine to 10% | constant, linear to 0 over the last 70% | warmup 40 steps, constant, linear to 5% over the last 65% | warmup 2M tokens, constant, linear to 5% over the last 65% | follows nanochat; in tokens |
| order of training data | random windows with replacement | sequential | sequential; rows begin with BOS, documents packed by best fit | each shard once, its windows in a seeded shuffled order | without replacement and deterministic, with batches that mix the whole shard |
| which weight decay for "Adam" | (AdamW) | (AdamW) | (AdamW) | `adam` uses the L2 penalty, `adamw` the decoupled decay | so that the difference between the two can be measured |

Read on 14 November 2026: nanochat `nanochat/gpt.py`, `nanochat/optim.py`, `scripts/base_train.py` (master) and the
same files as of October 2025; modded-nanogpt `README.md` (the record table) and
`records/track_3_optimization/train_gpt_simple.py`; `KellerJordan/Muon/muon.py`; nanoGPT `model.py`, `train.py`,
`config/train_gpt2.py`.

## 4. What the references have that we left out

The recipe is limited to components the series explains one by one. Not adopted, all of them available as later
experiments: value embeddings and the skip connection from the embedding to every block; the "smear" and "backout"
terms; sliding-window attention; per-layer residual scalars; the Polar Express coefficients (Amsel et al., 2025,
arXiv:2505.16932), NorMuon, cautious weight decay and Muon's momentum warmup; distributing Muon's work across cards;
batch-size schedules; FP8 layers and hand-written kernels; BOS-aligned batches.

## 5. Measurements

All on this machine, 14 November 2026: four cards that the driver reports as "NVIDIA Graphics Device"
(compute capability 12.0, 32 GB each), PyTorch 2.9.1 with CUDA 13.0. Every card is capped at 165 W: under a
sustained load its clock falls from 2,415 MHz to about 1,730 MHz, so a benchmark of a few seconds reads about 5%
faster than a run. The rates below are sustained, as the trainer reports them (the first logging interval, which
pays for compilation, is left out).

### 5.1 Throughput, memory, compile time

| run | preset | cards × sequences × tokens | tokens a step | tokens a second | peak memory a card | compile |
|:--|:--|:--|--:|--:|--:|--:|
| `core-probe-pico-final` | `gptpp-pico` | 1 × 256 × 256 | 65,536 | 2,223,727 | 2.6 GB | 13 s |
| `core-probe-pico-gpt2-lr6` | `gpt2-2019-pico` | 1 × 256 × 256 | 65,536 | 2,429,199 | 2.3 GB | 2 s |
| `core-probe-base` | `gptpp-ladder` | 1 × 128 × 512 | 65,536 | 238,095 | 15.4 GB | 6 s |
| `core-probe-final` | `gptpp-ladder` | 1 × 128 × 512 | 65,536 | 231,710 | 15.4 GB | 14 s |
| `core-tput-ladder-b192` | `gptpp-ladder` | 1 × 192 × 512 | 98,304 | 237,020 | 22.8 GB | 16 s |
| `core-probe-gpt2-lr24` | `gpt2-2019-ladder` | 1 × 128 × 512 | 65,536 | 252,323 | 13.8 GB | 2 s |
| `core-tput-base-b48` | `gptpp-base` | 1 × 48 × 1024 | 49,152 | 89,590 | 22.5 GB | 15 s |
| `core-tput-base-b56` | `gptpp-base` | 1 × 56 × 1024 | 57,344 | 93,203 | 26.0 GB | 23 s |
| `core-tput-base-accum2` | `gptpp-base` | 1 × 48 × 1024 × 2 | 98,304 | 92,742 | 23.0 GB | 2 s |
| `core-tput-gpt2-base` | `gpt2-2019-base` | 1 × 48 × 1024 | 49,152 | 100,416 | 20.0 GB | 20 s |
| `core-ddp-ladder` | `gptpp-ladder` | 2 × 128 × 512 | 131,072 | 450,965 | 15.8 GB | 22 s |
| `core-ddp-base` | `gptpp-base` | 2 × 48 × 1024 | 98,304 | 177,069 | 22.4 GB | 26 s |

"Compile" is what compilation cost that run: 13 to 26 s when the compiler's cache is cold, about 2 s when it already
holds the graph. Reading the table:

- **Largest batch on a 32 GB card.** Base size, context 1,024: 56 sequences use 26.0 GB and 48 use 22.5 GB; 64 do
  not fit. Ladder size, context 512: 192 sequences use 22.8 GB, 128 use 15.4 GB. The rate does not depend on the
  batch in this range.
- **Run-to-run variation.** The same configuration measured 89.6k and 94.2k tokens a second (base, 48 sequences) and
  between 232k and 240k (ladder, 128 sequences) in different runs, on different cards and with different neighbours
  on the machine: read the rates to about 3%.
- **Two cards.** About 94% of twice the one-card rate at both sizes; the rest is the exchange of gradients. Four cards
  were not measured.
- **Muon against AdamW.** The 2019 recipe (AdamW, tied embedding) runs 5% to 10% faster than the 2026 recipe at the
  same size.
- **ALiBi** passes a (heads, T, T) bias to the attention kernel instead of asking it for the causal mask. At 32
  sequences of 512 tokens that cost 6% of the rate and no memory (193k against 205k tokens a second, 4.3 GB against
  4.4 GB); it was not measured at the full batch of 128.
- Earlier runs of the same configurations, before the tables' runs: base size with 32 sequences, 92,855 tokens a
  second and 15.5 GB; with 48 sequences, 94,213 and 22.5 GB; with 64 sequences, out of memory.

### 5.2 Where a step's time goes

Forward and backward pass and optimizer step, timed apart over a few seconds on one card:

| | `gptpp-ladder`, 128 × 512 | `gptpp-base`, 32 × 1,024 |
|:--|--:|--:|
| forward and backward | 250 ms | 293 ms |
| of which unembedding, soft-cap and loss | 84 ms | 57 ms |
| optimizer step, Muon and AdamW | 11 ms | 35 ms |
| optimizer step if everything used AdamW | 3 ms | 6 ms |
| loading the batch | 0.3 ms | 0.2 ms |

The unembedding and the loss take a third of the 8-layer model's step: 32,768 logits for each of 65,536 tokens.
The optimizer step is 4% of the total at the ladder size and 11% at the base size with 32 sequences (about 7% with
48), nearly all of it Muon's orthogonalization: its cost per step is fixed, so its share falls as the batch grows,
and it could be divided among the cards, which we did not do.

### 5.3 Short probes of the defaults

One card, one seed each unless said, `gptpp-ladder` on 50M tokens of FineWeb-EDU (763 steps of 65,536 tokens,
about four minutes) and `gptpp-pico` on 60M tokens of TinyStories (916 steps, half a minute). The loss is the
trainer's held-out loss in nats (1,048,576 fixed tokens). These are short runs: they say which default to start
from, not how the choices rank after billions of tokens. The logs are in `$GPTPP_WORK/ckpt/core-probe-*`.

The 8-layer model:

| run | what differs from the default at the time | held-out loss | bits per byte |
|:--|:--|--:|--:|
| `core-probe-base` | the defaults at the time: `qk_scale` 8, embedding and unembedding rates 0.245 and 0.0049, seed 0 | 4.0707 | 1.2689 |
| `core-probe-seed1` | seed 1 | 4.0701 | 1.2687 |
| `core-probe-qk144` | `qk_scale` 11.5 | 4.0517 | 1.2630 |
| `core-probe-qk2` | `qk_scale` 16 | 4.0563 | 1.2644 |
| `core-probe-qklearned` | `qk_scale` learned per head, starting at 8 | 4.0477 | 1.2617 |
| `core-probe-noembednorm` | `embed_norm` off | 4.0774 | 1.2710 |
| `core-probe-lr01` | Muon rate 0.01 instead of 0.02 | 4.0906 | 1.2751 |
| `core-probe-lr04` | Muon rate 0.04 | 4.1247 | 1.2857 |
| `core-probe-emb2026` | embedding and unembedding rates 0.367 and 0.0098 | 4.0608 | 1.2658 |
| `core-probe-final` | **the defaults now**: `qk_scale` 12, rates 0.367 and 0.0098 | 4.0452 | 1.2609 |
| `core-probe-gpt2` | the 2019 recipe (`gpt2-2019-ladder`), AdamW rate 6e-4 | 5.1250 | 1.5975 |
| `core-probe-gpt2-lr12` | the 2019 recipe, rate 1.2e-3 | 4.8127 | 1.5002 |
| `core-probe-gpt2-lr24` | the 2019 recipe, rate 2.4e-3 (**its default now**) | 4.6294 | 1.4430 |
| `core-probe-gpt2-lr48` | the 2019 recipe, rate 4.8e-3 | 5.0548 | 1.5756 |

Pico:

| run | what differs from the default at the time | held-out loss | bits per byte |
|:--|:--|--:|--:|
| `core-probe-pico-qk1` | the defaults at the time: `qk_scale` 5.7, embedding and unembedding rates 0.49 and 0.0098 | 1.7616 | 0.6310 |
| `core-probe-pico-qk144` | `qk_scale` 8.1 | 1.7479 | 0.6261 |
| `core-probe-pico-qk2` | `qk_scale` 11.3 | 1.7469 | 0.6257 |
| `core-probe-pico-qk265` | `qk_scale` 15 | 1.7495 | 0.6267 |
| `core-probe-pico-qklearned` | `qk_scale` learned per head, starting at 5.7 | 1.7424 | 0.6241 |
| `core-probe-pico-lr01` | Muon rate 0.01 instead of 0.02 | 1.7775 | 0.6367 |
| `core-probe-pico-lr04` | Muon rate 0.04 | 1.7826 | 0.6385 |
| `core-probe-pico-lrflat` | rates 0.2 and 0.004 (not scaled by width) | 1.8275 | 0.6546 |
| `core-probe-pico-emb2026` | rates 0.735 and 0.0196 | 1.7427 | 0.6242 |
| `core-probe-pico-emb2026x2` | rates 1.47 and 0.039 | 1.7398 | 0.6232 |
| `core-probe-pico-final` | **the defaults now**: `qk_scale` 12, rates 0.735 and 0.0196 | 1.7271 | 0.6186 |
| `core-probe-pico-gpt2-lr1` | the 2019 recipe (`gpt2-2019-pico`), AdamW rate 1e-3 | 2.5157 | 0.9011 |
| `core-probe-pico-gpt2-lr3` | the 2019 recipe, rate 3e-3 | 2.1212 | 0.7598 |
| `core-probe-pico-gpt2-lr6` | the 2019 recipe, rate 6e-3 (**its default now**) | 2.0103 | 0.7201 |
| `core-probe-pico-gpt2-lr12` | the 2019 recipe, rate 1.2e-2 | 1.9978 | 0.7156 |

Every probe except the two `*-final` runs ran when `qk_scale` was still counted in units of sqrt(d_head): their
`config.json` says 1 (or 1.44, 2, 2.65 in the `qk*` runs), which are the 8, 11.5 and 16 (ladder) and 5.7, 8.1, 11.3
and 15 (pico) of the tables. The checkpoints of those runs were deleted, since the code now would misread them;
their logs, configurations and summaries remain.

What the probes say:

1. Two seeds of the same configuration ended 0.0006 nats apart and were never more than 0.005 apart at the earlier
   evaluations. Differences of 0.01 are probably real; differences of 0.005 are not established.
2. Attention scale: Section 2.4. The learned scales ended at 14.9 on average on the ladder and 12.6 on pico.
3. Embedding and unembedding rates, and Muon's rate: Section 2.9.
4. `embed_norm`: Section 2.2.
5. The defaults together moved the ladder from 4.0707 to 4.0452 and pico from 1.7616 to 1.7271.
6. The 2019 recipe needs a much higher AdamW rate on short budgets than the 6e-4 of nanoGPT's 300B-token run: the
   best of the sweep was 2.4e-3 at the ladder size and 6e-3 to 1.2e-2 at pico, and those are now the presets'
   rates (6e-3 at pico). At 6e-4 the ladder's 2019 baseline would look 0.50 nats worse than it is. The base size was
   not swept and keeps 6e-4.
7. After 50M tokens the 8-layer model stands at 4.045 with the 2026 recipe and 4.629 with the 2019 recipe at its
   best rate; pico after 60M tokens of TinyStories at 1.727 and 1.998.

## 6. The trainer's conventions

- **Budget in tokens.** steps = round(tokens / (device_batch · seq_len · n_gpu · grad_accum)).
- **Data.** One pass, in shard order, without replacement; inside a shard the windows are visited in an order
  shuffled by the seed. A step's windows depend on the seed and on how many windows a step takes, not on how they
  are split among cards (`tests/test_train_smoke.py` checks that two cards with batch 8 reproduce the losses of
  one card with batch 16; they agree to 1e-4).
- **Evaluation.** The first floor(val_tokens / seq_len) windows of `val.bin`, every `eval_tokens`, in the mode the
  model trains in (bfloat16 autocast). It is a monitor; the bits per byte reported in tables come from
  `gptpp/evaluate.py`, which scores whole documents.
- **Determinism.** The seed fixes the initialization and the data order. Two runs with the same seed agree to about
  1e-4 in the loss after a hundred steps, not bit for bit (some CUDA kernels are not deterministic).
- **Logs.** `log.jsonl` is flushed line by line. A run that dies writes an `error` record and no `summary.json`; a
  loss that is not finite stops the run at the next log line. SIGTERM is handled like Ctrl-C and leaves an
  `interrupted_<step>.pt`.
- **Several cards.** `python -m torch.distributed.run --standalone --nproc_per_node=4 -m gptpp.train ...`.
  Every rank holds the full optimizer state and takes the same step on the averaged gradient; `summary.json`
  records `rank_spread`, the largest difference between the ranks' parameter checksums at the end (zero in our runs).
- **Mixing recipes.** `--set` changes single fields; `--train-preset` takes the whole training recipe (optimizer,
  rates, schedule) from another preset, so `--preset gpt2-2019-ladder --train-preset gptpp-ladder` is the rung
  of the ladder that changes only the optimizer.

## 7. The analysis interface

`forward_explicit` computes the same function as `forward` with attention written out, and hands every activation
to a hook. The names follow "A mathematical framework for transformer circuits": `embed`, `resid_pre.l`,
`attn.l.q|k|v|scores|pattern|z|result`, `attn_out.l`, `resid_mid.l`, `mlp.l.pre|post`, `mlp_out.l`, `resid_post.l`,
`final_norm`, `logits`.

- `q` and `k` are the linear projections, before QK normalization and rotation, so they are linear in the
  (normalized) residual stream. With grouped-query attention `k` and `v` have `n_kv_head` heads.
- `scores` include the mask (−inf on future keys) and the ALiBi term.
- `z` is each head's weighted average of values, `result` what each head writes to the stream (z through that head's
  slice of the output matrix). `result` is large, is computed only when asked for by name, and cannot be replaced.
- In a gated MLP, `mlp.l.pre` is the gate's pre-activation and `mlp.l.post` the gated product.
- Cached tensors stay in the autograd graph, so gradients with respect to activations are available.
- KV-cached decoding uses the same path with a `KVCache`; `generate` never lets the context exceed `block_size`
  (it rebuilds the cache from the most recent half when the window is full).

The tests assert, for every switch of the model: causality on both paths, the two paths agree, cached decoding
equals the full forward pass, the cache satisfies the residual identities, and patching the clean stream into a
corrupted run restores the clean logits.

<!-- MEASURED -->
## 8. What the runs measured about each decision

Generated from the runs' summaries by the series' tooling; the README has the same numbers as tables. Two
measurements of each decision on the 8-layer, 512-wide model: the step it made on the ladder (the 2019 recipe
changed one decision at a time, 100M tokens a rung, mean of the seeds that were run), and the change in
held-out loss when the old choice is put back into the finished recipe (50M tokens, one seed; the recipe ends at 4.045
nats a token). A negative step on the ladder means the decision helped; a positive number in the last column
means putting the old choice back hurt.

| decision (section) | step on the ladder | old choice put back into the 2026 recipe |
|:--|--:|:--|
| no biases (2.5) | -0.000 | with biases -0.021 |
| rotary positions (2.1) | -0.110 | learned table +0.112; sinusoid, added after the embedding norm +0.060; ALiBi +0.002; none +0.264 |
| RMSNorm without parameters (2.2) | +0.031 | LayerNorm -0.019; RMSNorm with a gain -0.020; dynamic tanh +0.031; post-norm +0.088; none: diverged |
| squared ReLU (2.3) | -0.033 | ReLU +0.018; GELU +0.001; SiLU +0.007; SwiGLU -0.008; GeGLU -0.002 |
| QK normalization (2.4) | -0.011 | off +0.029; a learned gain +0.001 |
| zero-initialized projections (2.7) | -0.027 | GPT-2's scaled initialization -0.001 |
| the 2026 training recipe: Muon, its rates and schedule (2.9, 2.10) | -0.155 | AdamW with its own warmup and clipping, at its best rate (2.4e-3) +0.136; AdamW at its best rate (1.2e-3) +0.169; Adam at its best rate (1.2e-3) +0.180; Lion at its best rate (1e-4) +0.478; SGD at its best rate (0.3) +0.673 |
| untied tables, normalized embedding (2.6, 2.2) | -0.011 | tied, no embedding norm -0.000; untied, no embedding norm +0.014 |
| logit soft-cap (2.8) | -0.003 | no cap -0.002 |
| schedule (2.10) |  | cosine +0.031; linear +0.011; constant +0.150; no warmup -0.008 |

Two seeds of one ladder rung differ by 0.001 to 0.018 nats (root mean square 0.008), so a difference under about 0.016 between two single runs cannot be told from noise, and 0.024 or more is safe to call real.

What this says about Section 2, at this size and budget: rotary positions, Muon, QK normalization and pre-norm are
what the loss depends on. The parameter-free norm (2.2) and the missing biases (2.5) did not measure better than
what they replaced: a gain in the norm and biases each measured about 0.02 nats better when put back, in both of
two seeds (under the 2026 recipe two seeds differ by 0.008 at most); on the ladder the parameter-free norm cost 0.03
and removing the biases was worth nothing. The squared ReLU (2.3) gained 0.03 on the ladder and ties GELU in the
finished recipe. Untying the tables, the zero initialization and the soft cap show nothing in the finished
recipe. The ladder and the put-back runs disagree on some decisions because a step on
the ladder is taken in the company of the earlier recipe (the 2019 optimizer until rung 7) and a put-back in the
company of the finished one. The reasons given in Section 2 are the reasons the choices were made; this table is
what they turned out to be worth.

The step of the training recipe on the ladder changes fourteen settings at once. Two more runs split it in seed 0: the rung-6 model under the 2026 training recipe with AdamW (its own warmup and clipping) in place of Muon ends at 3.911, between 3.985 and 3.828: -0.074 for the rest of the recipe, -0.083 for Muon itself.

The optimizer row compares rules for the hidden matrices only (`optimizer=<rule>+adamw`): the embedding, the
unembedding and the schedule stay the recipe's. With one rate for every parameter group, the way of 2019, AdamW
and Adam end near 5.0 nats at every rate on this model (runs `optflat-*`), because the normalized embedding needs
a rate hundreds of times larger than a hidden matrix: that is a property of the model, not of the optimizer.

These tables hold the runs as corrected after an independent review of the experiments ([REVIEW.md](REVIEW.md)):
the sinusoid added after the embedding norm (before it, the signal fades as the token rows grow), AdamW under its
own warmup and clipping, and the split of the training recipe's step.
<!-- /MEASURED -->
