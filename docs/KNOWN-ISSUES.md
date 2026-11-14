# Known issues, quirks and bugs

This file lists what a reader of the code should know before trusting a line of it or comparing it with a paper,
with GPT-2, nanoGPT or nanochat code, or with PyTorch's defaults. It was compiled on 14 November 2026 by reading
the code at commit `1890c9ae92b5d20836dd64fcceddca1606f1c8e0`, the tests, the run records under `results/` and
[REVIEW.md](REVIEW.md), together with the source and documentation of the pinned PyTorch. Nothing was run for it:
every item names the function it is about and the evidence that was read, which is a test that pins the
behaviour, a passage of the documentation, a run record, or the library's source. The items have stable ids
(K01 to K65, and I01 to I25 for the interpretability tools), which the series that this repository accompanies
uses when it cites them. What could not be
supported by evidence is listed apart, under "Unverified".

Each item has a kind:

- **quirk**: deliberate or harmless, but a reader who compares would trip on it;
- **known issue**: a limitation or defect still present at this commit;
- **bug, fixed**: found and fixed, with the commit and what it did to published numbers;
- **PyTorch**: a behaviour of the library that the code relies on or works around.

## The pin

- The code: commit `1890c9a` of 14 November 2026. The runs under `results/runs/` record the commits they were
  launched at: `d866bce` (26 runs), `b66f6b9` (83), `702f0af` (36), `5ecb05b` (60), and none for 6 (K42).
- The versions: `requirements.txt` pins `torch==2.9.1`. Every run's `config.json` records the same hardware
  entry: `"torch": "2.9.1+cu130"`, `"cuda": "13.0"`, `"python": "3.13.9"`, `"gpu": "NVIDIA Graphics Device"`
  (211 of 211 runs). The cards have 32 GB each and compute capability 12.0 (DESIGN.md, Section 5).
- The newest PyTorch when this was written is 2.15.0 (28 October 2026; it needs Python 3.11 or newer). The pin
  is of 12 November 2025; 2.10, 2.11, 2.12, 2.13 and 2.14 lie between.

What follows is taken from the release notes of PyTorch 2.10 to 2.15 and was not tested. No note removes or
renames an API that this code calls. The changes that could alter behaviour off the pin are these.

- `F.scaled_dot_product_attention`: 2.10 adds a FlashAttention-4 kernel, and 2.15 prefers the cuDNN kernel on
  some cards and fixes a defect of that kernel for models with fewer than 8 key-value heads on one family of
  cards. The docstring says the output can differ with the kernel chosen, so the last bits of a run can differ
  by version.
- Fused `AdamW` and `Adam`: 2.11 switches the CUDA kernels to fused multiply-add instructions, which changes
  the last bits of the tables' updates.
- `torch.compile`: 2.13 fixes dropped casts between float32 and bfloat16 in compiled code. Releases 2.13 and
  2.14 produced correlated dropout masks in compiled CUDA code, fixed in 2.15; the recipe has no dropout.
- `torch.distributed`: in 2.15 `backend="nccl"` is a rewritten implementation that binds a process to the card
  `LOCAL_RANK % device_count`, which is the card the trainer already uses. The launcher's default port changes
  in 2.12, but not with `--standalone`.
- TensorFloat-32: the flags `torch.backends.cuda.matmul.allow_tf32` and `torch.backends.cudnn.allow_tf32`,
  which this code sets, are described by the 2.9 documentation as going to be deprecated in favour of
  `fp32_precision`. No release note up to 2.15 removes them.
- Wheels: before 2.11 the default wheel on PyPI was built for CUDA 12, so `pip install torch==2.9.1` is not the
  `+cu130` build that the runs used. From 2.15 no wheel is built for CUDA 13.0.
- New and unused: `torch.optim.Muon` (present at the pin, K23) and `nn.LinearCrossEntropyLoss` (2.13).

## The model (`gptpp/model.py`)

### K01. `GPT.forward` returns the loss alone when targets are given

- Kind: quirk.
- Where: `GPT.forward`.
- What: With `targets`, `forward` returns one scalar, the mean cross-entropy in nats over the targets that are
  not -1. Without them it returns the logits. It never returns the pair of logits and loss. The ignored target
  is -1 and not PyTorch's default of -100, which `gptpp/evaluate.py` uses for its own masking.
- Evidence: `tests/test_model.py::test_forward_shapes_and_loss`. nanoGPT returns the pair, and nanochat returns
  the loss alone; both ignore -1.
- Effect: none measured.

### K02. Two paths compute one function

- Kind: quirk.
- Where: `CausalSelfAttention.forward` and `forward_explicit`; `GPT.forward` and `GPT.forward_explicit`.
- What: `forward` is the fast path. Attention is one call of `F.scaled_dot_product_attention`, nothing is
  recorded, and this is what the trainer compiles. `forward_explicit` writes attention out, hands every
  activation to a hook, and serves every recorded or replaced activation and all decoding with the key-value
  cache. Some analyses take the untouched model's logits or loss from the fast path and an intervened one from
  the explicit path (`circuits.py`, `measure_circuits.py`, `export/activation_patching.py`). The equality of the two
  is pinned in float64 on the CPU, to 1e-5 in the logits, over the 31 variants of `tests/conftest.py`.
- Evidence: `tests/test_model.py::test_fast_path_equals_explicit_path`,
  `::test_two_paths_agree_in_training_mode_without_dropout`, `::test_causality`,
  `::test_cached_decoding_equals_full_forward`.
- Effect: none in float64. In the arithmetic of training the two differ (K03).

### K03. Under bfloat16 autocast the two paths differ, and sampled text parts

- Kind: known issue.
- Where: `CausalSelfAttention.forward_explicit` against `forward`; `GPT.causal_bias`.
- What: On the explicit path the scores are a bfloat16 product to which the float32 mask of `causal_bias` is
  added. PyTorch promotes the sum to float32, and the softmax then runs in float32. On the fast path scores,
  softmax and average happen inside one kernel on bfloat16 inputs. The two paths therefore differ by the
  rounding of bfloat16. The test allows 0.15 between their logits and 0.3 against float32 on a small model.
- Evidence: `tests/test_model.py::test_two_paths_agree_under_bfloat16_autocast`;
  `results/interp/part-7-measured.json`, field `writing` (`"same_tokens": false`, `"tokens_in_common": 101`);
  the docstring of `gptpp/export/part7_measure.py`; the autocast lists of PyTorch 2.9.1
  (`aten/src/ATen/autocast_mode.h`).
- Effect: on the main model, 1,024 tokens sampled at temperature 0.8 with one seed by `generate` (explicit
  path, cache) and by repeated calls of the fast path are the same for 101 tokens and then part. Greedy decoding
  in float64 is identical on the two paths (`::test_generate`). An activation read by an analysis under autocast
  is an explicit-path number.

### K04. What runs in which precision under autocast

- Kind: PyTorch.
- Where: `gptpp/train.py`, `train` (the autocast context); `GPT.unembed_logits`; `rotate`;
  `CausalSelfAttention.position`.
- What: Autocast casts by operation. `linear`, `matmul`, `einsum` and `scaled_dot_product_attention` run in
  bfloat16. `softmax`, `layer_norm` and the cross-entropy run in float32. `rms_norm` is on neither list and
  runs in the type of its input. The residual stream therefore stays in float32, because the embedding is
  float32 and every addition of a bfloat16 sublayer output is promoted. The block norms read it in float32, the
  normalization of queries and keys is applied to bfloat16 tensors, the rotation is computed in float32 and cast
  back, and the logits are a bfloat16 product cast to float32 before the cap and the loss.
- Evidence: `aten/src/ATen/autocast_mode.h` and `aten/src/ATen/native/layer_norm.cpp` at tag v2.9.1 of
  PyTorch; DESIGN.md, Sections 2.1, 2.2 and 2.11; the dtype assertion in
  `tests/test_model.py::test_two_paths_agree_under_bfloat16_autocast`.
- Effect: none measured beyond K03. Logits carry the rounding of bfloat16 although their type is float32.

### K05. The fused attention call

- Kind: PyTorch.
- Where: `CausalSelfAttention.forward`.
- What: `F.scaled_dot_product_attention` applies dropout whatever the module's mode, so the code passes
  `dropout_p=self.dropout if self.training else 0.0`. The function raises if both `attn_mask` and `is_causal`
  are given, so the code passes the flag without ALiBi and the table, which contains the causal mask, with it.
  The docstring calls the function beta and says that its output may differ with the kernel chosen.
- Evidence: the docstring of `scaled_dot_product_attention` in `torch/nn/functional.py` (2.9.1);
  `tests/test_model.py::test_dropout_only_in_training_mode`.
- Effect: none measured. The recipe's dropout is 0.

### K06. Query-key normalization is a root-mean-square normalization, with the gain folded into the query

- Kind: quirk.
- Where: `CausalSelfAttention.__init__` (`qk_gain`, `qk_factor`) and `CausalSelfAttention.position`.
- What: Queries and keys are normalized to unit root mean square with `F.rms_norm(..., eps=1e-6)`, so their
  length is sqrt(d_head) and not one, as it is in the paper that introduced the method (arXiv:2010.04245). The
  kernel divides by sqrt(d_head), and the query is multiplied by `qk_scale / sqrt(d_head)`, so that a score is
  `qk_scale` times the cosine. The kernel's own `scale` argument is not used. With `qk_scale_learned` each head
  has one more factor, `qk_factor`, which starts at one and belongs to the "gains" group.
- Evidence: `tests/test_model.py::test_qk_norm_bounds_the_scores`; DESIGN.md, Section 2.4.
- Effect: none on the function, up to the 1e-6 inside the root. The switch itself is worth 0.029 and 0.032 nats
  in two seeds on the 8-layer model (README).

### K07. With ALiBi the float mask takes attention off FlashAttention

- Kind: PyTorch.
- Where: `CausalSelfAttention.forward` (`attn_mask`); `alibi_table`.
- What: The function accepts a mask as a tensor, but the FlashAttention kernel does not. The next kernel in
  PyTorch's order is memory-efficient attention, and the change of kernel gives no message. Which kernel ran
  was read from the source and not observed. Under autocast the ALiBi table is cast to bfloat16 on the fast
  path, while the explicit path adds it in float32.
- Evidence: `check_for_attn_mask` in `aten/src/ATen/native/transformers/sdp_utils_cpp.h`, the constraints in
  `.../cuda/sdp_utils.cpp` and the order in `aten/src/ATen/Context.h`, all at tag v2.9.1; DESIGN.md, Section 5.1.
- Effect: 193,000 against 205,000 tokens a second at 32 sequences of 512 tokens, with the same memory. It was
  not measured at the full batch.

### K08. Rotary positions pair coordinate i with coordinate i + d_head/2

- Kind: quirk.
- Where: `rotary_table`, `rotate`, `CausalSelfAttention.position`.
- What: `rotate` splits a head's vector into two halves and rotates the pair (i, i + d_head/2) by its position
  times base^(-2i/d_head), counter-clockwise. The paper (arXiv:2104.09864) pairs neighbouring coordinates. The
  tables are computed in float64 and kept as float32 buffers outside the checkpoint. The rotation follows the
  normalization of queries and keys and applies to queries and keys only.
- Evidence: `tests/test_model.py::test_rotation_depends_on_relative_position_only`,
  `::test_rotary_model_scores_are_functions_of_the_offset`; DESIGN.md, Section 2.1. nanochat uses the same
  split and rotates the other way.
- Effect: none on what can be learned, since the query and key matrices absorb a fixed permutation of
  coordinates. Weights are not interchangeable with an implementation that pairs neighbours.

### K09. What the hooks hand out

- Kind: quirk.
- Where: `CausalSelfAttention.forward_explicit`, `MLP.forward_explicit`, `Block.forward_explicit`,
  `GPT.run_with_cache`.
- What: `attn.l.q` and `attn.l.k` are the linear projections, before the normalization and before the rotation.
  `attn.l.scores` include the -inf of the mask and the ALiBi term. With fewer key-value heads, `k` and `v` have
  `n_kv_head` heads. `attn.l.result` is computed only when it is asked for and cannot be replaced: a hook on it
  is accepted and its return value is dropped. In a gated MLP, `mlp.l.pre` is what enters the gate's
  nonlinearity. Cached tensors stay in the autograd graph.
- Evidence: the docstring of `run_with_cache`; DESIGN.md, Section 7;
  `tests/test_model.py::test_cache_is_the_residual_decomposition`, `::test_cache_shapes`.
- Effect: none measured.

### K10. Grouped-query attention copies keys and values

- Kind: quirk.
- Where: `CausalSelfAttention.share`.
- What: With fewer key-value heads than query heads, keys and values are repeated with `repeat_interleave`
  before the kernel. PyTorch's `enable_gqa` argument is not used. The cache stores the heads before the copy, so
  the cache is smaller and the arithmetic of a step is not.
- Evidence: `tests/test_model.py::test_grouped_query_attention` and the shape assertion at the end of
  `::test_cached_decoding_equals_full_forward`.
- Effect: on the 8-layer model, 50M tokens and one seed, 4, 2 and 1 key-value heads cost 0.003, 0.030 and
  0.050 nats (README).

### K11. The key-value cache grows by concatenation and bought no speed at this size

- Kind: known issue.
- Where: `KVCache.extend`; `gptpp/generate.py`, `generate`.
- What: `extend` concatenates the new keys and values onto the stored ones at every token, in every layer.
  Nothing is preallocated, and the cache has no limit of its own; its callers stop or rebuild it at
  `block_size` (K56).
- Evidence: `results/interp/part-7-measured.json`, field `writing`; the message of commit `8bfeabc`.
- Effect: on one card the 12-layer model wrote 1,024 tokens at 167 tokens a second with the cache and at 176 by
  running the whole context through the fast path at every step (the better of two timings each). The cause was
  not profiled.

### K12. The norms are the repository's own modules, and the layout is not GPT-2's

- Kind: quirk.
- Where: `LayerNorm`, `RMSNorm`, `DynamicTanh`, `make_norm`, `CausalSelfAttention.__init__`, `gelu`.
- What: The norms are thin modules over the functionals, and their parameter is named `gain` where PyTorch's
  modules and nanoGPT name it `weight`. `LayerNorm` always has its gain, has a bias only when `bias` is set, and
  uses an epsilon of 1e-5. `RMSNorm` computes `x / sqrt(mean(x^2) + 1e-6)`, where the default epsilon of
  `nn.RMSNorm` is the machine epsilon. Attention has four separate matrices where GPT-2 has one fused matrix
  for queries, keys and values, and the 2019 preset uses the exact GELU and not the tanh approximation of
  GPT-2's released code. Released GPT-2 weights therefore do not load into the 2019 preset.
- Evidence: `tests/test_model.py::test_norms`, `::test_activations`,
  `::test_the_2019_preset_has_the_parameter_count_of_gpt2_small`; `aten/src/ATen/native/layer_norm.cpp` at tag
  v2.9.1.
- Effect: none measured.

### K13. Sinusoids added before the embedding norm fade

- Kind: bug, fixed in commit `8bfeabc` by a switch. The default keeps the old arrangement.
- Where: `GPT.embed_tokens`; `ModelConfig.pos_after_norm`.
- What: With `pos=sinusoidal` under the 2026 recipe the fixed sinusoids were added to the token vector before
  the embedding norm. The token rows grow during training, so position's share of the energy that enters the
  norm fell from 33% to 1.4%. With `pos_after_norm=true` the sinusoids are added after the norm at unit
  amplitude. The default is still `false`, so that the first run can be reproduced, and `pos=sinusoidal` alone
  still fades.
- Evidence: `tests/test_sinusoid_share_under_embed_norm.py`; REVIEW.md, item 3; the runs
  `swap-pos-sinusoidal`, `swap-pos-sinusoidal-after-norm` and its second seed.
- Effect: the first run ended at 4.1777 nats, 0.133 behind rotary positions and behind a learned table
  (4.1572). With the switch the run ends at 4.1050 and 4.1019 in two seeds, 0.060 and 0.058 behind rotary and
  ahead of the table. Every published number for sinusoids is the second.

### K14. Position tables are scaled by the embedding's scale, are not saved, and end at `block_size` when learned

- Kind: quirk.
- Where: `sinusoidal_table`, `GPT.__init__`, `GPT.extend_positions`.
- What: The sinusoidal table is multiplied by `embed_std`, which is 1 for an untied embedding of the 2026
  recipe and 0.02 otherwise. The sinusoidal, rotary and ALiBi tables are non-persistent buffers: they are absent
  from checkpoints, rebuilt at construction and extended when a longer sequence arrives. A learned table has no
  row past `block_size`, and the model raises `ValueError` there.
- Evidence: `tests/test_model.py::test_sequences_longer_than_block_size`,
  `::test_learned_positions_end_at_block_size`, `::test_checkpoint_round_trip`.
- Effect: none measured.

### K15. `init` switches more than the matrices that write to the stream

- Kind: quirk.
- Where: `GPT.init_weights`.
- What: Under `zero_proj` the two matrices that write to the stream start at zero, so the untrained model is
  the embedding followed by the unembedding; in an untied model, whose unembedding starts near zero, its first
  loss is ln V. The same switch also draws every other
  hidden matrix uniformly in plus or minus 1/sqrt(fan_in), which is a standard deviation of 0.0255 at width 512
  against 0.02 under `gpt2`, an untied embedding at N(0, 1) and an untied unembedding at N(0, 0.001^2). A tied
  table keeps 0.02 whatever `init` says. Under `gpt2` the writers have 0.02/sqrt(N), with N twice the number of
  layers, or the number of layers in a model without MLPs.
- Evidence: `tests/test_model.py::test_zero_proj_init_makes_every_block_the_identity`, `::test_gpt2_init`;
  DESIGN.md, Sections 2.7 and 3.
- Effect: rung 6 of the ladder is this whole switch: 0.027 nats on the means of two seeds, and 0.008 in one
  seed with the longer warmup. Put back into the finished recipe, the `gpt2` initialization changes the loss by
  0.001 in one seed.

### K16. Post-norm is the 2017 arrangement, and it collapses at the ladder's step size

- Kind: known issue.
- Where: `Block`; `GPT.__init__` (`final_norm` is the identity under post-norm).
- What: `norm_placement=post` normalizes after each addition and has no final norm. In the 2019 recipe at steps
  of 65,536 tokens the post-norm model fell within 40 steps to predicting token frequencies, with the warmup of
  8M tokens and without it, and a warmup of 32M tokens alone does not rescue it. The same model trains at steps
  of 8,192 tokens and rate 6e-4 with a warmup, and at the ladder's step with the longer warmup and rate 6e-4.
  The run `swap-norm-post` changes two switches, the placement and the kind of norm.
- Evidence: REVIEW.md, item 2 and the tables at its end; the runs `warm-2019-*`.
- Effect: 7.690 nats at the ladder's step, where the entropy of token frequencies is 7.701. At steps of 8,192
  tokens post-norm reaches 4.537 with a warmup (pre-norm 4.581) and 7.696 without (pre-norm 4.950). At the
  ladder's step with 32M tokens of warmup and rate 6e-4 it reaches 5.255 (pre-norm 4.613).

### K17. The soft cap is inside the model

- Kind: quirk.
- Where: `GPT.unembed_logits`.
- What: `forward`, the `logits` hook and `generate` return capped logits, 15 tanh(z / 15). The cap is
  computed in float32 (`unembed_logits` casts with `.float()`); under the trainer's autocast the product it caps
  was made in bfloat16. An analysis that multiplies the stream by `model.unembed.weight` itself reads logits
  before the cap.
- Evidence: `tests/test_model.py::test_soft_cap_bounds_the_logits`. Through the cap: `gptpp/interp/lens.py`,
  `sae.py` and the ablations of `dashboard.py`. Before it: the logit effects of `dashboard.py`, `memories.py`
  and `transcoder.py`, `patching.py`, and `attribution_graph.py`.
- Effect: none measured. The cap itself moves the held-out loss by 0.003 nats on the ladder and 0.002 as a
  swap, which is inside the seed-to-seed spread.

### K18. The four parameter groups

- Kind: quirk.
- Where: `GPT.param_groups`.
- What: "matrices" is every `nn.Linear` weight inside the blocks. "embedding" is the token table and a learned
  position table. "unembedding" is empty when the tables are tied. "gains" is everything else: the gains and
  biases of every norm, the linear biases, `qk_factor` and the slope of dynamic tanh. In the default 2026 recipe
  the "gains" group is empty.
- Evidence: `tests/test_model.py::test_param_groups_partition_the_parameters`; `derived.groups` in
  `results/runs/gptpp-base/config.json`.
- Effect: none measured. A learned position table trains at the embedding's rate, which the review found to be
  no handicap (REVIEW.md, "Checked and sound").

### K19. A checkpoint holds weights and the model's configuration, and nothing else

- Kind: quirk.
- Where: `GPT.from_checkpoint`; `gptpp/train.py`, `save_checkpoint`; `gptpp/config.py`, `_Config.from_dict`.
- What: A checkpoint is a dictionary with `model`, `model_config`, `step` and `tokens`. It is loaded with
  `weights_only=True`, the prefixes `module.` and `_orig_mod.` are stripped, and the model is returned in
  evaluation mode. There is no optimizer state. `from_dict` refuses an unknown field and fills a missing one
  with today's default, with a warning.
- Evidence: `tests/test_model.py::test_checkpoint_round_trip`, `::test_config_from_dict_is_strict`.
- Effect: none measured. A default that changes its meaning reinterprets old records (K20).

### K20. The unit of `qk_scale` changed during the build

- Kind: known issue.
- Where: `ModelConfig.qk_scale`; DESIGN.md, Section 5.3.
- What: The short probes that chose the defaults ran when `qk_scale` was counted in units of sqrt(d_head).
  Their recorded configurations say 1, 1.44, 2 and 2.65 for what the tables call 8, 11.5 and 16 on the 8-layer
  model and 5.7, 8.1, 11.3 and 15 on the smallest. Their checkpoints were deleted, because today's code would
  misread them.
- Evidence: DESIGN.md, Section 5.3, the paragraph after its two tables.
- Effect: none on published numbers, which give the scales in today's unit.

## The optimizers (`gptpp/optim.py`)

### K21. `orthogonalize` takes five tuned steps, in bfloat16 on a card, and does not converge to one

- Kind: quirk.
- Where: `orthogonalize`, `NS_TUNED`.
- What: On CUDA the iteration runs in bfloat16 and the result is cast back; on the CPU it keeps the gradient's
  type. A tall matrix is transposed so that the Gram matrix is the small one. The input is divided by its
  Frobenius norm plus 1e-7. With the tuned coefficients, five steps send every singular value above about 0.0015
  of the Frobenius norm into the interval from 0.68 to 1.20, not to one, and multiply smaller ones by at most
  485.
- Evidence: `tests/test_optim.py::test_the_tuned_polynomial`,
  `::test_tuned_newton_schulz_keeps_the_singular_vectors`, `::test_orthogonalize_batches_shapes_and_scale`,
  `::test_orthogonalize_in_bfloat16_on_the_gpu`.
- Effect: none measured against an exact orthogonalization.

### K22. `Muon.step` as written

- Kind: quirk.
- Where: `Muon.step`.
- What: Matrices of one shape are stacked and orthogonalized in one batched call, three calls a step for the
  default model. The momentum buffer is one tensor for each group and shape, kept outside `Optimizer.state`, so
  `state_dict()` does not carry it. The Nesterov form is `grad.lerp_(buffer, momentum)`. The step is
  `lr * sqrt(max(1, rows / columns))` with rows the output width, so the first matrix of the MLP moves twice as
  far as the others. The decoupled decay multiplies by `1 - lr * weight_decay`, with `lr` and not the scaled
  step. `p.grad` is read without a check for `None`, and the `closure` argument is ignored.
- Evidence: `tests/test_optim.py::test_muon_first_step`, `::test_muon_momentum_and_nesterov`,
  `::test_muon_weight_decay_is_decoupled`, `::test_muon_stacks_matrices_of_equal_shape`; DESIGN.md, Section 2.9.
- Effect: the optimizer's step is 4% of a step's time on the 8-layer model and 11% on the 12-layer model with
  32 sequences (DESIGN.md, Section 5.2).

### K23. PyTorch has its own Muon, and this code does not use it

- Kind: PyTorch.
- Where: `Muon` in `gptpp/optim.py`; `torch/optim/_muon.py` in PyTorch 2.9.1.
- What: `torch.optim.Muon` exists at the pin, with the same coefficients, five steps, momentum 0.95 and the
  Nesterov form. It differs from the class here in its defaults (rate 1e-3, weight decay 0.1), in handling one
  matrix at a time, in running the iteration in bfloat16 on the CPU as well, in clamping the norm from below
  where this code adds 1e-7, in skipping parameters without a gradient, and in offering a choice of step
  scaling. Its docstring writes the momentum as a sum, while its code computes the same running average as this
  code.
- Evidence: `torch/optim/_muon.py` (2.9.1). The release notes of 2.15 list a fix to it for bfloat16 gradients.
- Effect: none. No run used it.

### K24. A tied table trains at the unembedding's rate and decay

- Kind: quirk.
- Where: `build_optimizers`.
- What: When embedding and unembedding are one matrix, the "embedding" group, which then also holds a learned
  position table, takes the unembedding's rate and weight decay. At the embedding's rate, 37 to 50 times
  larger, the loss rose in the first 14 steps.
- Evidence: `tests/test_optim.py::test_build_optimizers_cover_every_parameter_once`;
  `tests/test_train_smoke.py::test_train_preset_trains_the_2019_model_the_2026_way`; DESIGN.md, Section 2.9.
- Effect: this is rung 7 of the ladder, where the one table trains at 0.0098. Rung 8 gives the embedding
  0.3674.

### K25. The optimizer names, and what the runs called "Adam" were

- Kind: quirk.
- Where: `build_optimizers`; `OPTIMIZERS` in `gptpp/config.py`.
- What: `adam` is `torch.optim.Adam`, whose weight decay is a penalty added to the gradient. `adamw` is
  `torch.optim.AdamW`, which at 2.9.1 is `Adam` with decoupled decay and multiplies the weights by
  `1 - lr * weight_decay`. The runs reported as "Adam" used `optimizer=adam+adamw` with a weight decay of 0,
  which is the same update as AdamW without decay. Adam with a penalty was never run. `muon+adamw` returns the
  AdamW optimizer first and Muon second, so AdamW steps first. `fused` is passed as true only when every
  parameter is on CUDA. `sgd` is `torch.optim.SGD` with Nesterov momentum whenever `nesterov` is true, which is
  the default, and the runs reported as "SGD with momentum" record `"nesterov": true`.
- Evidence: `torch/optim/adamw.py` and `torch/optim/adam.py` (2.9.1);
  `results/runs/opt-adam-lr1.2e-3/config.json`; README, "The rule for the hidden matrices".
- Effect: 4.225 nats for Adam against 4.214 for AdamW at their best rate under the recipe's schedule, one seed
  each.

### K26. `X+adamw` changes one line of the recipe and takes its constants from the defaults

- Kind: quirk.
- Where: `build_optimizers`, the branch for names that end in `+adamw`.
- What: The hidden matrices are trained by X with the run's rate, betas, epsilon and decay. The tables and the
  gains keep AdamW with the betas and epsilon of a default `TrainConfig()`, which are (0.8, 0.95) and 1e-10,
  whatever the run's configuration says. The AdamW runs of the comparison also kept the recipe's epsilon of
  1e-10 on the matrices, because no override changed it. The plain names (`adamw`, `adam`, `sgd`, `lion`) put all four groups into
  one optimizer, each group still at its own rate; the groups have one rate only when their rates are left unset
  (the 2019 preset) or set equal by hand.
- Evidence:
  `tests/test_optim.py::test_one_line_changed_rules_train_the_matrices_and_keep_the_recipes_adamw_for_the_rest`;
  `results/runs/opt-adamw-lr1.2e-3/config.json`.
- Effect: with the same rate set for every group, AdamW and Adam end near 5.0 nats at every rate (runs
  `optflat-*`),
  against 4.2 with the tables at their own rates.

### K27. The optimizer comparison handicapped AdamW

- Kind: bug, fixed. The defect was in the experiment and not in the code; the runs were made again in commits
  `8bfeabc` and `1890c9a`.
- Where: `experiments/review.jsonl`, `experiments/followup.jsonl`; the runs `opt-*-own-*`.
- What: The first comparison changed only the rule for the hidden matrices and kept the recipe's schedule, a
  warmup of 2M tokens and no clipping, which had been chosen with Muon. AdamW's own recipe at this size has a
  warmup of 8M tokens and clipping at norm one, and the cliff in its losses above rate 1.2e-3 was the short
  warmup. AdamW, Lion and SGD were run again under that warmup and clipping. Adam was not.
- Evidence: REVIEW.md, item 1 and the second round; README, "The rule for the hidden matrices".
- Effect: the best AdamW improved from 4.214 to 4.181 nats. Muon's advantage is 0.136 and 0.163 in two seeds
  and not 0.169. Lion and SGD improve and stay 0.44 and 0.49 behind.

### K28. The schedule is read in tokens, at the middle of the step, from zero

- Kind: quirk.
- Where: `lr_multiplier`, `set_lr`; the loop of `gptpp/train.py`, `train`.
- What: `lr_multiplier` is a function of the tokens seen and not of the step, and the trainer reads it at the
  middle of each step. The warmup is linear from zero, so the first step's rate is half a step's worth of warmup
  and not zero. The decay of the trapezoid begins at the later of the warmup's end and the point
  `1 - cooldown_frac` of the budget. The `lr` in the log is the rate of the hidden matrices.
- Evidence: `tests/test_optim.py::test_warmup_is_linear_from_zero`, `::test_schedule_shapes`,
  `::test_schedules_are_functions_of_tokens_not_steps`;
  `tests/test_train_smoke.py::test_samples_checkpoints_and_schedule`.
- Effect: none measured. The recipe's warmup of 2M tokens is 30 steps on the ladder and about 14 in the main
  run.

### K29. The recipe's constants

- Kind: quirk.
- Where: `TrainConfig` in `gptpp/config.py`.
- What: The recipe's AdamW has betas (0.8, 0.95) and an epsilon of 1e-10, where PyTorch's defaults are
  (0.9, 0.999) and 1e-8. Muon has momentum 0.95, five steps and a rate of 0.02. The rate is not scaled with the
  batch: it was chosen at 65,536 tokens a step and used unchanged at the 147,456 of the main run. `lr_gains` has
  no parameter to act on in the default recipe.
- Evidence: `train_config` in `results/runs/gptpp-base/config.json`; DESIGN.md, Sections 2.9 and 3.
- Effect: on the 12-layer model at 49,152 tokens a step, Muon at 0.01, 0.02 and 0.04 ends at 3.653, 3.654 and
  3.679 nats (100M tokens, one seed each).

### K64. `NS_CLASSIC` is not the classical Newton-Schulz quintic

- Kind: quirk.
- Where: `gptpp/optim.py`, the constant `NS_CLASSIC`; `tests/test_optim.py`.
- What: The constant is (2, -1.5, 0.5). The polynomial 2x - 1.5x^3 + 0.5x^5 has slope 2 at zero and, at 1, value 1
  and slope 0 but curvature 1. The classical fifth-order Newton-Schulz polynomial, which part 6 of the series calls
  the classical quintic, is (15x - 10x^3 + 3x^5) / 8, with slope 1.875 at zero and both slope and curvature 0 at 1.
  Only the tests use `NS_CLASSIC`. `Muon.step` passes no coefficients, so every run used the tuned ones.
- Evidence: the line; the tests that iterate it to the exact factor; arithmetic on the two polynomials.
- Effect: none on any run. In five steps the constant's polynomial multiplies a small singular value by 32, the
  classical quintic by 23.2.

## The trainer and the configuration (`gptpp/train.py`, `gptpp/config.py`)

### K30. The budget is rounded to whole steps

- Kind: quirk.
- Where: `train`.
- What: The number of steps is the budget divided by the tokens of a step, rounded, and the schedule is laid
  over that many steps. The tokens used therefore differ from the tokens asked for.
- Evidence: `derived` in `results/runs/*/config.json`;
  `tests/test_train_smoke.py::test_every_preset_trains_and_its_loss_falls`.
- Effect: 3.0e9 tokens asked became 2,999,992,320 (20,345 steps of 147,456); 100e6 became 100,007,936; 25e6
  became 24,969,216.

### K31. The width rule was not applied under `--set d_model=`

- Kind: bug, fixed in commit `1890c9a`.
- Where: `build_configs` in `gptpp/train.py`; `train_preset` in `gptpp/config.py`.
- What: The recipe scales the tables' AdamW rates by sqrt(768 / width). Before the fix `train_preset` went by
  the preset's own width, so a width set on the command line kept the preset's rates, and every size of the
  first scaling grid trained at the rates of width 512. `build_configs` now builds the model first and takes the
  training preset at its width. A rate given with `--set` still wins.
- Evidence: `tests/test_width_rule_under_overrides.py`; REVIEW.md, second round; the runs `grid-*` (first) and
  `gridr-*` (made again).
- Effect: the 19 runs at other widths were made again. At width 128 the rule is worth 0.049 nats after 25M
  tokens and 0.008 after 200M, and nothing that can be told from noise at widths 384 and 768. The fitted
  exponent for parameters fell from 0.56 to 0.52. One number of the series still comes from a model of the first
  grid: the copy test of the 128-wide, four-layer model (`results/interp/copy-test.json`, run
  `grid-w128-l4-200M`: 12.6 bits a token on the second copy against 20.8 on the first) was not repeated on the
  model trained again.

### K32. The width rule, and what has none

- Kind: quirk.
- Where: `train_preset`; `group_settings` in `gptpp/optim.py`.
- What: Under the 2026 recipe the embedding's rate is 0.3 times sqrt(768 / d), rounded to four decimals, and
  the unembedding's is 0.008 times sqrt(768 / d), rounded to five: 0.7348 and 0.0196 at width 128, 0.3674 and
  0.0098 at 512, 0.3 and 0.008 at 768. Muon's rate has no rule. The 2019 recipe has one rate and one warmup
  chosen for each size. A second scaling, by the square root of the batch (`lr_ref_batch`), exists and is off.
- Evidence: `tests/test_optim.py::test_group_settings`;
  `tests/test_width_rule_under_overrides.py::test_the_2019_recipe_has_no_width_rule`; DESIGN.md, Sections 2.9
  and 5.3.
- Effect: without the scaling the smallest model was 0.066 nats worse in a short probe (DESIGN.md, Section 2.9).

### K33. The base preset's batch does not fit on the cards that trained the main model

- Kind: known issue.
- Where: `_BATCH` in `gptpp/config.py`; the quick start of the README.
- What: The preset `gptpp-base` has 64 sequences a card at a context of 1,024, which runs out of memory on a
  card of 32 GB. The main run was launched with `--set device_batch=48 --eval-tokens 50e6 --save-log 24` and
  `--save-steps` for every 2,000th step. The quick start of the README gave the command without them until the
  commit that added this file; the preset itself still asks for 64.
- Evidence: the first line of `experiments/main.jsonl`; `argv` in `results/runs/gptpp-base/config.json`;
  DESIGN.md, Section 5.1.
- Effect: the main run's step is 3 times 48 times 1,024 tokens, 147,456. With `--save-log 24` alone the main run
  would have written 24 checkpoints, where it wrote 34.

### K34. TensorFloat-32 is switched on for the whole process

- Kind: PyTorch.
- Where: `train` in `gptpp/train.py`; `finetune` in `gptpp/chat.py`; `load_model` in `gptpp/evaluate.py`;
  the fixture `exact_float32_products` in `tests/conftest.py`.
- What: The trainer, the chat stage and the evaluator set `torch.backends.cuda.matmul.allow_tf32`, which is off
  by default, and `torch.backends.cudnn.allow_tf32`, which is on by default and concerns no layer of this
  model. They never restore them. A test that ran one of them loosened every later float32 comparison on a card
  until the fixture was added in commit `8bfeabc`. The documentation of PyTorch 2.9 describes these flags as
  going to be deprecated.
- Evidence: the notes on CUDA in the PyTorch 2.9 documentation; `aten/src/ATen/Context.h` at tag v2.9.1;
  EVAL.md, "Precision".
- Effect: on GPT-2 small and WikiText-103, 1.0719 bits per byte in float32 and 1.0720 with TensorFloat-32.

### K35. One initialization on every card, and runs that do not repeat bit for bit

- Kind: quirk.
- Where: `train` (the two calls of `torch.manual_seed`); `TrainLoader._next_shard` in `gptpp/data.py`.
- What: `torch.manual_seed(seed)` precedes the construction of the model on every rank. The seed then becomes
  `seed + 1 + rank`, which only dropout would see. `--seed` also fixes the order of the data. Two runs of one
  seed and configuration are not identical, because some CUDA kernels are not deterministic.
- Evidence: REVIEW.md, "Checked and sound"; `tests/test_train_smoke.py::test_same_seed_same_run`; DESIGN.md,
  Section 6; the note on reproducibility in the PyTorch 2.9 documentation.
- Effect: the same seed and configuration run twice gave 3.8156 and 3.8133 nats, and 4.0451 and 4.0448. Two
  different seeds differ by 0.001 to 0.008 under the 2026 recipe.

### K36. The model is compiled for training and evaluation, not for sampling or saving

- Kind: quirk.
- Where: `train`.
- What: `torch.compile(model)` is wrapped by `DistributedDataParallel`. The loss and the held-out evaluation
  call the compiled model. The sample written at each evaluation calls the uncompiled model through the explicit
  path. Checkpoints save the uncompiled model's state dict, so their keys carry no prefix. All 211 recorded runs
  were compiled.
- Evidence: `tests/test_train_smoke.py::test_command_line_with_compilation`; `compile_seconds` in
  `results/runs/gptpp-base/summary.json`.
- Effect: none measured on losses.

### K37. Several cards exchange gradients and nothing else

- Kind: quirk.
- Where: `train`, `rank_spread`; `TrainLoader.next_batch`.
- What: Every rank holds the whole model and the whole optimizer state, reads its slice of the step, and takes
  the same step on the averaged gradient. The orthogonalization of Muon is repeated on every rank. With gradient
  accumulation the trainer sets `require_backward_grad_sync` on the last pass only, which is the attribute that
  `DistributedDataParallel.no_sync()` toggles. The summary records `rank_spread`, the largest difference
  between the ranks' sums of all parameters.
- Evidence: `tests/test_train_smoke.py::test_two_cards`, which compares two cards of 8 sequences with one of 16
  and asserts the loss at initialization to 1e-4, every later record to 5e-3 and a `rank_spread` of zero;
  `::test_ranks_read_disjoint_slices_of_the_same_step`; `results/runs/gptpp-base/summary.json`.
- Effect: three cards read 263,957 tokens a second, 98% of three times one card. Gradient accumulation on
  several cards is exercised by no test and no run.

### K38. Clipping acts on all parameters, before both optimizers

- Kind: quirk.
- Where: `train`.
- What: With `grad_clip` above zero the global norm over every parameter is clipped once, before any optimizer
  steps. The 2019 recipe clips at norm one and the 2026 recipe does not clip. One step of Muon does not depend
  on the size of its gradient, but clipping rescales a gradient before it enters Muon's running average, by a
  factor that changes from step to step, so under `muon+adamw` it would change the mix inside the momentum and
  not only what AdamW does to the tables. No run combined the two.
  The loop of the chat stage does not clip at all.
- Evidence: `tests/test_optim.py::test_muon_ignores_the_size_of_the_gradient`;
  `tests/test_train_smoke.py::test_gradient_accumulation_and_other_optimizers`.
- Effect: with its own warmup and clipping the best AdamW gained 0.033 nats (K27).

### K39. The trainer's bits per byte divide by a constant

- Kind: known issue.
- Where: `train` (`val_bpb`).
- What: The trainer converts its loss with the bytes per token of the whole held-out file, with the
  beginning-of-document tokens counted: 4.6283 for FineWeb-EDU and 4.0278 for TinyStories. The 1,048,576 tokens
  that it scores have 4.6353 and 4.0640.
- Evidence: REVIEW.md, "The trainer's bits per byte use a constant"; EVAL.md, "The trainer's own bits per byte
  are a different number".
- Effect: the main run's 0.9429 would read 0.9415, and the smallest model's 0.552 would read 0.547. Comparisons
  between runs are unaffected, because every run uses the same constant.

### K40. The trainer's monitor is not the evaluation protocol

- Kind: known issue.
- Where: `evaluate` in `gptpp/train.py`; `ValLoader` in `gptpp/data.py`.
- What: The monitor is the mean loss over the first windows of the held-out stream, 1,048,576 tokens of it, cut
  without regard to documents, in bfloat16 autocast, with the beginning-of-document token among the targets. The
  protocol (K51) scores whole documents from their first byte, in float32 with TensorFloat-32 products, and
  never scores that token.
- Evidence: DESIGN.md, Section 6; REVIEW.md, "Checked and sound";
  `tests/test_train_smoke.py::test_val_loader_is_fixed`; the field `val` of
  `results/interp/part-7-measured.json`.
- Effect: for the main model the two read 0.9429 and 0.9322 bits per byte. In windows of 512 tokens the same
  model reads 3.086 nats, against 3.025 in its own windows of 1,024.

### K41. A run cannot be resumed

- Kind: known issue.
- Where: `train`.
- What: An interruption leaves a file `interrupted_<step>.pt` with the weights only and no `summary.json`. A
  crash leaves an `error` record in the log. A loss that is not finite stops the run at the next log line.
- Evidence: `tests/test_train_smoke.py::test_a_crash_leaves_a_readable_log_and_no_summary`,
  `::test_a_completed_run_is_not_overwritten`.
- Effect: none measured.

### K42. A run records the commit, not the code

- Kind: known issue.
- Where: `git_commit` in `gptpp/train.py`.
- What: `config.json` stores the commit at launch and nothing about uncommitted changes. Six runs made before
  the first commit, the main run among them, record none. The grid made again records `5ecb05b`, although it
  ran with the fix of the width rule that was committed afterwards as `1890c9a`. The run
  `swap-pos-sinusoidal-after-norm` records `702f0af`, and its switch was committed afterwards as `8bfeabc`.
- Evidence: `git_commit` in `results/runs/*/config.json`; the history of `pos_after_norm`; REVIEW.md, "Not
  checked".
- Effect: none on numbers. The recorded configuration, with its rates and switches, is the evidence of what
  ran.

### K43. The vocabulary size comes from the data

- Kind: quirk.
- Where: `train`; `apply_overrides` in `gptpp/config.py`.
- What: If the preset's `vocab_size` differs from the one in the data's `meta.json`, the trainer replaces it
  and says so. Overrides are parsed as the declared type of their field, and an unknown key is an error.
- Evidence: `tests/test_train_smoke.py::test_every_preset_trains_and_its_loss_falls`;
  `tests/test_model.py::test_overrides_are_type_checked`.
- Effect: none measured.

## Data and tokenizer (`gptpp/data.py`, `gptpp/bpe.py`, `gptpp/tokenizer.py`, `scripts/prepare_*.py`)

### K44. The data is read once, shard by shard, in windows that ignore documents

- Kind: quirk.
- Where: `TrainLoader`.
- What: Shards are read in order. Inside a shard the windows are visited in a permutation drawn from the seed,
  the pass and the shard, without replacement, and the windows left over at the end of a shard are dropped.
  Windows are cut without regard to documents, and consecutive windows share one token, so that every token is
  a target once. Rank r of n takes the r-th slice of each step, so a step's windows do not depend on the number
  of cards.
- Evidence: `tests/test_train_smoke.py::test_train_loader_reads_every_window_once`,
  `::test_train_loader_is_deterministic_by_seed`, `::test_ranks_read_disjoint_slices_of_the_same_step`.
- Effect: the main run read the first 30 of 99 shards and 720 windows of the 31st, 30% of the training set,
  once.

### K45. Pretraining uses one special token, and it is also a target

- Kind: quirk.
- Where: the format described at the top of `gptpp/data.py`; `TOKEN_DTYPE`; `SPECIAL_TOKENS` in `gptpp/bpe.py`.
- What: Every document begins with `<|bos|>`, and none ends with a token of its own. Because windows cross
  documents, `<|bos|>` is a target at every document boundary, in training and in the trainer's monitor, and
  the model learns to end a document by writing it. Token ids are stored as unsigned 16-bit integers, so a
  vocabulary cannot exceed 65,536 tokens.
- Evidence: `tests/test_tokenizer.py::test_shards_obey_the_contract`; `continuation` in `gptpp/chat_eval.py`,
  which stops the base model when it writes `<|bos|>`; DATA.md.
- Effect: a part of the gap between the monitor and the protocol (K40).

### K46. Held-out text is disjoint from training by document, not by content

- Kind: known issue.
- Where: `scripts/prepare_fineweb.py`; `HELD_OUT` in `gptpp/evaluate.py`.
- What: The held-out documents are a prefix of one seeded permutation, and the training documents that have the
  text or the address of a held-out one are removed. Near copies remain. Documents that repeat another training
  document's text stay in the training set.
- Evidence: REVIEW.md, "Held-out text"; DATA.md, "How the datasets are built"; `results/review/heldout_decomp.py`.
- Effect: 4.1% of the held-out blocks of 64 tokens occur verbatim in the 31 shards that the main run read;
  without them its 3.025 nats become 3.040. Two of the 1,000 documents that the protocol scores have a near copy
  in training; without them 0.9322 becomes 0.9324.

### K47. What the corpus and the vocabulary lack

- Kind: quirk.
- Where: DATA.md, "Things to know" and "Measured statistics"; `results/data/*/stats.json`.
- What: The byte `{` does not occur once in the 9.9 billion training tokens of FineWeb-EDU. 50 of the 32,768
  tokens of `bpe32k` never occur in training: 46 byte tokens, `{` among them, and the four chat tokens. 86 of
  the 4,096 tokens of `bpe4k` never occur. `bpe32k` has no token for two spaces, and a number gets no leading
  space.
- Evidence: `ids_never_seen` in `results/data/fineweb-edu/bpe32k/stats.json`; the field `samples.brace` of
  `results/interp/part-7-measured.json`.
- Effect: after a line of Python the main model gives `{` a probability of 3.9e-12.

### K48. The merges are learned by the compiled library and checked against the plain-Python trainer

- Kind: quirk.
- Where: `train_merges` in `gptpp/tokenizer.py`; `train` and `train_fast` in `gptpp/bpe.py`.
- What: The production tokenizers were trained by the `tokenizers` library. It is fed pieces cut by the same
  Python pattern, as bytes written as the characters U+0000 to U+00FF, so that its ids and its rule for ties,
  the smaller pair of ids, are those of `bpe.py`. `bpe.train` recounts every pair at every round and is
  quadratic; `bpe.train_fast` learns the same merges and is the one that reaches a real corpus. `bpe32k` was
  learned from two billion characters of training documents and not from the whole training set.
- Evidence: `tests/test_tokenizer.py::test_reference_and_production_learn_the_same_merges`,
  `::test_reference_trainer_on_real_text`; `tests/test_bpe.py::test_fast_training_learns_the_same_merges`;
  DATA.md, "The reference and the production tokenizer are the same tokenizer".
- Effect: none. The two give the same tokenizer.

### K49. The split pattern and the special tokens

- Kind: quirk.
- Where: `GPT4_PATTERN` and `SPECIAL_TOKENS` in `gptpp/bpe.py`; `_build` and `Tokenizer.encode` in
  `gptpp/tokenizer.py`.
- What: The pattern is that of `cl100k_base`, with its possessive quantifiers written as greedy ones and its
  alternative `\s++$` left out. Digits are cut at most three at a time. The five special tokens take the last
  ids and are never produced from text: `<|bos|>` in a web page is seven characters. `BPE.encode` and
  `Tokenizer.encode` add `<|bos|>` only with `bos=True`, and the default is `False`; the module-level
  `bpe.encode` has no such argument.
- Evidence: `tests/test_bpe.py::test_split_pattern`, `::test_special_tokens`;
  `tests/test_tokenizer.py::test_text_is_never_read_as_special_tokens`; DATA.md.
- Effect: none measured.

### K50. Two regular-expression engines carry two Unicode tables

- Kind: known issue.
- Where: the splitting in `gptpp/bpe.py` and in `_build` of `gptpp/tokenizer.py`.
- What: `bpe.py` cuts text with Python's `regex` module, and the production tokenizer cuts it with the engine
  inside the library. They classify 17,480 code points differently, and around such a character the two encoders
  cut differently. Both still decode to the original text. A comment above `GPT4_PATTERN` and the module's
  docstring say that the two engines cut every text alike, which is true only of text without those characters.
- Evidence: `tests/test_tokenizer.py::test_the_two_regex_engines_differ_only_in_their_unicode_tables`,
  `::test_reference_and_production_encode_identically`; DATA.md.
- Effect: none on this data. A scan made once found none of these characters in either corpus.

### K65. The n-gram tables are not scored as the networks beside them are

- Kind: quirk.
- Where: `Level` and `ngram_statistics` in `scripts/corpus_stats.py`; a second copy of `Level` in
  `gptpp/export/counted_models.py`.
- What: A Kneser-Ney model of order n is scored on every token of the held-out stream from the fourth on, the
  beginning-of-document tokens included, with contexts that run across document boundaries, and its bits per
  token are divided by the bytes per token of the whole held-out file. The networks in the same tables are scored
  under the protocol (K51): whole documents, that token never scored, each document's exact bytes. Each level has
  its own discount, D = n1 / (n1 + 2 n2), and the level under the unigrams is the uniform distribution. `Level`
  exists twice with the same code and different docstrings.
- Evidence: the docstring of `ngram_statistics` and the `note` field it writes.
- Effect: not measured. The tables score nearly the whole held-out file, so the constant is the right divisor
  for them, unlike the trainer's case (K39).

## Evaluation (`gptpp/evaluate.py`, `gptpp/benchmarks.py`)

### K51. The protocol's bits per byte

- Kind: quirk.
- Where: `load_model`, `score_sequences`, `bits_per_byte`.
- What: Each document is encoded by the model's own tokenizer and cut into windows of the model's
  beginning-of-text token and at most 1,023 tokens. Every token is scored once; that token and an end token are
  never scored; the bits are divided by the documents' exact bytes. Windows do not overlap, so the first tokens
  of each are predicted from little context. Every model, this repository's own included, is scored in float32
  with TensorFloat-32 products and not in the bfloat16 it trained in.
- Evidence: `tests/test_evaluate.py::test_every_token_is_scored_once_whatever_the_window`,
  `::test_padding_does_not_change_a_score`, `::test_our_model_scores_equal_its_own_loss`; the field
  `precision` in `results/eval/*.json`; EVAL.md, Section 1.
- Effect: for GPT-2 small on WikiText-103 a token deep in a window costs 4.62 bits, against 4.87 on average.

### K52. How released models are scored

- Kind: quirk.
- Where: `HFModel`; the list `MODELS` in `scripts/eval_released_models.py`.
- What: Each released model uses its own tokenizer, its own beginning-of-text token and its own context, up to
  1,024 tokens. One model defines no such token, and its end-of-text token is used. One model is scored with a
  context of 512, its training length, and not the 2,048 of its configuration. bfloat16 is not safe for a model
  that was not trained in it.
- Evidence: `bos_source` in `results/eval/qwen3-0.6b-base.json`; EVAL.md, "Precision".
- Effect: one released model scores 1.1164 bits per byte in float32 and 1.2917 in bfloat16 on 20 articles.

### K53. Published numbers that the code does and does not reproduce

- Kind: known issue.
- Where: `run_task` in `gptpp/benchmarks.py`; EVAL.md, Section 6.
- What: For GPT-2 small the code reproduces the published perplexity of the last word on LAMBADA (35.11
  against 35.13) and HellaSwag within 0.55 points. It does not reproduce the published WikiText-103 perplexity
  of the largest GPT-2 (20.24 against 17.48), and LAMBADA comes out 10% to 17% above the published figures for
  the three larger sizes. Unlike the common harness, every benchmark sequence begins with the beginning-of-text
  token. A context too long for the window loses its beginning, and tied options share the credit.
- Evidence: EVAL.md, Sections 3 and 6; `tests/test_evaluate.py::test_a_model_that_cannot_tell_scores_chance`,
  `::test_a_long_context_is_cut_from_the_left`.
- Effect: as stated. The cause of the difference for the largest GPT-2 is not known.

### K54. What was not scored

- Kind: known issue.
- Where: `results/eval/`; EVAL.md, Sections 2, 4, 7 and 10.
- What: The run `gptpp-base-1B` has only the trainer's number and was never scored under the protocol. The set
  "fresh" is eight essays from one site. One released model is gated and absent. One hosted model answered 118
  of 144 questions, and another was asked 672 of 700. Of the 2,000 TinyStories documents that are scored, 695
  are also in the training file of the dataset's first version, on which one released model was trained.
- Evidence: the files under `results/eval/`; README, the paragraph on nanochat; EVAL.md.
- Effect: that released model scores 0.442 bits per byte on the shared stories and 0.431 on the others.

## Generation and chat (`gptpp/generate.py`, `gptpp/chat.py`, `gptpp/chat_eval.py`)

### K55. Sampling takes one distribution and applies every cut at once

- Kind: quirk.
- Where: `truncate`, `sample`, `generate`.
- What: The logits are divided by the temperature and turned into probabilities once. Top-k, top-p and min-p
  are each computed on that same distribution, a token must pass all that are set, and what is kept is
  renormalized once. Top-k keeps every token tied with the k-th. A temperature of 0 takes the most probable
  token. The seed feeds a generator created on the prompt's device, and all rows of a batch draw from it, so
  one seed need not give the same text on a CPU and on a card, or for another batch.
- Evidence: `tests/test_model.py::test_truncation_rules`, `::test_sample`, `::test_generate`; the note on
  reproducibility in the PyTorch 2.9 documentation ("results may not be reproducible between CPU and GPU
  executions, even when using identical seeds").
- Effect: none measured.

### K56. Generation never exceeds the trained context, and slides by halves

- Kind: quirk.
- Where: `generate`; `reply` in `gptpp/chat.py`.
- What: `generate` keeps the last `block_size` tokens of a prompt. When the cache is full it is discarded and
  rebuilt from the most recent half of the window, so after a full window a token is predicted from between
  half a window and a whole one. `reply` does not slide: it stops at the context and reports "length".
- Evidence: `tests/test_model.py::test_generate`; `tests/test_chat.py::test_generation_stops_at_the_end_token`.
- Effect: none measured.

### K57. The chat format, and what the loss covers

- Kind: quirk.
- Where: `turns`, `assemble`, `render`, `pack`, `rows_of`, `ChatLoader`, `answer`.
- What: A conversation is `<|bos|>` followed by user turns between `<|user_start|>` and `<|user_end|>` and
  assistant turns between `<|assistant_start|>` and `<|assistant_end|>`. The loss is counted on the assistant's
  text and its end token only. There is no system role: a leading system message is joined to the first user
  message. Whole conversations are packed into rows of the context length, and the rest of a row is filled with
  `<|bos|>`, which carries no loss. A row's later conversations can see its earlier ones. `answer` stops at any
  special token.
- Evidence: `tests/test_chat.py::test_the_mask_is_the_assistants_text_and_its_end_token`,
  `::test_the_format_round_trips`, `::test_pack_places_every_conversation_whole_and_once`,
  `::test_rows_hold_whole_conversations_and_targets_are_masked`.
- Effect: on held-out conversations the end token costs 26.4 nats before the stage and 0.24 after
  (`results/chat/chat-stage.json`).

### K58. The chat stage is the pretraining recipe at 0.8 of its rates, on one card

- Kind: quirk.
- Where: `stage_config`, `finetune`.
- What: The stage rebuilds the base run's optimizers with every peak rate multiplied by `--lr-scale`, a warmup
  of 1M tokens and a linear decay. It starts from fresh optimizer state, runs on one card, and has no gradient
  clipping. The loss is computed by the compiled model and the evaluations by the uncompiled one. A base
  checkpoint without its run's `config.json` falls back to a default `TrainConfig()`.
- Evidence: `results/runs/gptpp-chat/config.json`;
  `tests/test_chat.py::test_the_stage_teaches_a_tiny_model_to_answer_and_to_stop`; the runs
  `gptpp-chat-probe-s*`.
- Effect: the loss on held-out conversations falls from 2.52 to 1.44 nats, and plain text rises from 0.932 to
  1.213 bits per byte. The twenty answers were judged by a language model and not by a person.

## Scripts and tests

### K59. `scripts/queue.py` shadowed the standard library

- Kind: bug, fixed in commit `8bfeabc`.
- Where: `scripts/job_queue.py`; the same guard at the top of `scripts/eval_released_models.py`,
  `eval_openrouter.py`, `judge_stories.py` and `prepare_chat.py`.
- What: Python puts a script's directory first on the module path, so a file named `queue.py` in `scripts/` was
  imported in place of the standard library's `queue` by anything run from there. The file is now
  `job_queue.py`, and the scripts take their own directory off the path or move it to the end.
- Evidence: the message of commit `8bfeabc`;
  `tests/test_train_smoke.py::test_queue_skips_done_jobs_and_survives_failures`, whose job `uses-py` imports
  `queue`.
- Effect: none on results.

### K60. The default workspace is one machine's path

- Kind: known issue.
- Where: `DEFAULT_WORK` in `gptpp/__init__.py`; `WORK` in `scripts/job_queue.py`; `SHAKESPEARE_PLACES` in
  `tests/conftest.py`; the commands of DATA.md.
- What: Without `GPTPP_WORK` the package looks for data and checkpoints under an absolute path of the machine
  it was built on, and the test fixtures look for a text file under another. Set `GPTPP_WORK`, and either set
  `GPTPP_SHAKESPEARE` or let the fixture download the file.
- Evidence: the lines named.
- Effect: none on results.

### K61. What the tests pin, and how tightly

- Kind: quirk.
- Where: `tests/`.
- What: The tests of the model and of the optimizers run on the CPU in float64 with every parameter
  randomized, over 31 variants, none of which sets `pos_after_norm`. The tests of the trainer use a card if
  there is one, and the test on two cards is skipped without two. The tests of the prepared corpora are skipped
  where the corpora are absent. Some tolerances are looser than the prose that cites them: two cards against
  one agree to 5e-3 after the first evaluation, and the same seed run twice to 2e-2.
- Evidence: `VARIANTS` in `tests/conftest.py`; `test_two_cards` and `test_same_seed_same_run` in
  `tests/test_train_smoke.py`; the skips in `tests/test_tokenizer.py`.
- Effect: none.

### K62. Documentation that was behind the code

- Kind: bug, fixed (in the commit that added this file), except for one comment.
- Where: EVAL.md, Section 10; DESIGN.md, Section 6; the README; a comment above `SPECIAL_TOKENS` in
  `gptpp/bpe.py`.
- What: Until the commit that added this file, EVAL.md said that no checkpoint of this repository's own existed,
  although `results/eval/gptpp-base.json` holds the main model's scores. DESIGN.md said that the test on two
  cards agrees to 1e-4 (it asserts 1e-4 at initialization and 5e-3 after) and that two runs with one seed agree
  to about 1e-4 after a hundred steps (no record of that; see Unverified). The README listed sign descent under
  `gptpp/optim.py` (it is the Lion of that file with both betas at 0.9, configured in
  `gptpp/interp/frequency_buckets.py`), and its command for the main run lacked the batch and the checkpoint
  steps of K33. These four are corrected. Still open: a comment in `bpe.py` calls the four chat tokens reserved
  for a later chat stage, which exists.
- Evidence: the passages named, and `git log -p` of this file's first commit.
- Effect: none on results.

### K63. Comparisons that are not single changes

- Kind: known issue.
- Where: `experiments/make_jobs.py`; REVIEW.md.
- What: The seventh step of the ladder changes fourteen training settings at once, and the eighth also gives
  the embedding its own rate, its own initialization and 16.8M parameters. Rungs 1 to 6 keep the rate chosen for
  rung 0, and the warmup of 8M tokens on the 2019 side was too short. `swap-norm-post` changes two switches and
  `swap-tied` four. The grid, the race, the ladder with the longer warmup and seven swaps have one seed.
- Evidence: REVIEW.md, item 4 and "Found, and left as stated limits"; README, "The ladder's 2019 rungs with a
  longer warmup".
- Effect: with 32M tokens of warmup each 2019 rung is 0.017 to 0.042 nats better, and the seventh step is 0.140
  and not 0.157 in that seed. Of that step, Muon in place of AdamW is 0.083 and 0.086 in two seeds.

## Unverified

These statements are believed and were not supported by evidence that was read. Each says what would verify it.

1. Why the cache bought no speed (K11). The likely cause is the cost per token of twelve explicit layers in
   the interpreter and of 24 concatenations, but nothing was profiled. A profile of `generate` and of the
   loop without the cache on an idle card would settle it.
2. Which attention kernel ran (K05, K07). That the recipe's call used FlashAttention on these cards, and that
   the ALiBi run used memory-efficient attention, is read from the selection code of PyTorch 2.9.1 and was not
   observed. `torch.backends.cuda.can_use_flash_attention` with `debug=True` would show it.
3. `Muon.step` on a matrix without a gradient (K22). The code hands `p.grad` to `torch.stack` unchecked. That
   a missing gradient raises there was not run, and no test covers it.
4. The size of the rounding of the ALiBi table to bfloat16 on the fast path (K07). A comparison of the two
   paths on an ALiBi checkpoint under autocast, by position, would measure it.
5. Whether a compiled graph at the pin drops a rounding between float32 and bfloat16, which the release notes
   of 2.13 list as fixed. A comparison of compiled and uncompiled logits on one batch would show it.
6. That two runs with one seed agree to about 1e-4 after a hundred steps (DESIGN.md, Section 6). No record of
   that measurement was found. The tests allow 2e-2, and two pairs of full runs ended 0.0023 and 0.0003 apart.
7. That switching between training and evaluation mode compiles a second graph (K36). This is read from the
   trainer's own accounting of `compile_seconds`.
8. The arithmetic inside the CUDA kernel of `rms_norm` on bfloat16 input (K04). The composite implementation
   casts up to float32 and back; the fused kernel was not read.
9. That `pip install torch==2.9.1` gives a build for CUDA 12. This is taken from the release notes of 2.11 and
   was not tried.
10. Gradient accumulation on several cards (K37). No test and no run exercises it.
11. What the sampler does with arguments outside their meant range (K55). This is read from `truncate` and
    `sample` and was not run, and no test covers it: `top_p=0.0` keeps no token, so the renormalization divides
    zero by zero, although the docstring says that the most likely token always survives; `top_k=0` asks for
    nothing, and "off" is `None`, not 0; a negative temperature is not rejected and inverts the ranking.

## Interpretability tools (`gptpp/interp/`)

The same kinds as above, for the analysis tools behind parts 8 to 10 of the series. The ids are I01 to I25.

### I01. What the hooks on queries and keys hold

- Kind: quirk.
- Where: `model.py:306-310, 319-323`; `GPT.run_with_cache`, `:635-638`.
- What: The hooks `attn.l.q` and `attn.l.k` hold the linear projections, before QK normalization and rotation;
  `scores` includes the mask and the ALiBi term; `result` is computed only when asked for and a hook on it replaces
  nothing. `patch_inputs` (`circuits.py:510-538`) relies on this: it swaps projections, and the model normalizes and
  turns them after.
- Evidence: docs/DESIGN.md section 7; `tests/test_model.py::test_cache_shapes`; the same hooks are K09 of
  CODE-NOTES.
- Effect: q k^T / sqrt(d) from a cache is not the model's score in a model with QK normalization or rotary
  positions.

### I02. Which logits each analysis reads: before or after the cap

- Kind: quirk.
- Where: `patching.py:59-64, 140-163`; `lens.py:62-65`; `attribution_graph.py:131-132`.
- What: The patching metric and direct attribution read logits before the cap (from `final_norm`); the lens applies
  the whole readout, cap included; an attribution graph takes its output nodes before the cap and weights them by
  probabilities after it.
- Evidence: the docstrings; results/interp/lens-patching-memories.json, "conventions";
  `tests/test_interp_lens.py::test_the_last_depth_is_the_model`.
- Effect: +7.90 in the series' Example 9.4.4 is a difference of uncapped logits; on the graph's prompt the top
  logit is 13.0 before the cap and 10.5 after, and the "logit of 28.9" that the series' Example 10.4.5 quotes is
  the graph's value of that output, the logit before the cap less the mean logit (13.0 + 15.9).

### I03. The circuit matrices are exact only for a stripped model

- Kind: known issue.
- Where: `circuits.py:65-70, 127-137, 214-233, 451-453`.
- What: `qk_matrix`, `qk_circuit`, `ov_circuit`, `composition` and `copying_statistic` are computed from the raw
  matrices of any model; they equal what the model computes only where `is_exact` holds (no MLP, no norm anywhere, no bias, no QK normalization, no cap, no dropout, rotary or no positions). `path_expansion` refuses other models.
- Evidence: module docstring 38-42; `test_the_path_expansion_refuses_a_model_it_is_not_exact_for`.
- Effect: none of these reads gpt-pp or pico; the circuit models are separate runs with `init=gpt2`, `norm=none`,
  `qk_norm=false` and the two table rates set by hand (results/runs/circ-attn2-repeats/config.json).

### I04. The baseline of the composition scores

- Kind: quirk.
- Where: `circuits.py:320-397`; `measure_circuits.py:131-137`.
- What: The baseline subtracted is the closed form 1/sqrt(d_model), 0.0625 at width 256, not the measured one of the
  paper (`composition_baseline` measures it and agrees to 4%). W_QK is taken at rotary offset 0, while `qk_circuit`
  defaults to offset 1. The permuted product has another baseline, about 1/sqrt(d_head).
- Evidence: `test_the_baseline_is_one_over_the_square_root_of_the_width`.
- Effect: the largest K score is 0.081 at offset 0 and 0.081 to 0.083 at offsets 1, 4, 16 and 64.

### I05. Prefix matching, copying and the in-context score are the lab's own definitions

- Kind: quirk.
- Where: `circuits.py:278-291, 613-634, 646-652`.
- What: Prefix matching adds the attention to the token after every earlier occurrence, so it grows with the number
  of repeats; the copying score is the lab's own formula on the weights; the in-context score is early minus late.
- Evidence: the docstrings.
- Effect: the same head scores 0.85 on 16 tokens given twice and 0.94 on 25 given four times.

### I06. What ablation replaces

- Kind: quirk.
- Where: `circuits.py:711-757`.
- What: Ablation replaces `attn.l.z`, a head's output before W_O, by zero or by its mean over the given sequences at
  each position; with `patterns` every layer's pattern is forced to its recorded value.
- Evidence: `test_a_full_ablation_changes_later_patterns_and_a_mean_ablation_uses_the_mean`.
- Effect: the 7.3 and 1.9 nats that the series' part 9 reports.

### I07. Patching sites: no head is patched alone

- Kind: known issue.
- Where: `patching.py:50, 67-116`.
- What: The sites are the stream entering a block, a layer's whole attention output and its MLP output, at one
  position. No head is patched alone, though the series' Definition 9.4.2 names "the output of a head". The two
  prompts must have the same number of tokens, and each map is one pair of prompts.
- Evidence: 
- Effect: heads enter the series' Example 9.4.4 by attribution only.

### I08. Direct attribution holds the final norm fixed

- Kind: quirk.
- Where: `patching.py:119-163`.
- What: Direct attribution holds the final norm's scale at its value on the prompt; the embedding's share is the
  hook `embed`, after its norm. Post-norm and dynamic tanh raise NotImplementedError.
- Evidence: `test_attributions_add_up_to_the_logit_difference`, `test_attribution_refuses_what_is_not_linear`.
- Effect: the shares add up to the metric to rounding.

### I09. The logit lens and the tuned lens

- Kind: quirk.
- Where: `lens.py:49-65, 181-222`.
- What: Depth 0 is the stream after the embedding's norm. The relative entropy runs from the final distribution to
  the lens, in bits. The tuned lens is 200 steps of Adam on 8 windows a step, its bias learned in units of each
  depth's root mean square; until commit 702f0af the map acted on the normalized stream.
- Evidence: docstrings; `git diff b66f6b9 702f0af -- gptpp/interp/lens.py`.
- Effect: the stream's root mean square is 1.0, 99, 1,072 and 4,568 after the embedding and blocks 1, 11 and 12, so
  the lens finds the token early and gives it almost no probability; the tuned fit "had not converged at the early
  blocks".

### I10. The tools do not share one precision

- Kind: quirk.
- Where: `sae.py:165-175, 500-515`; `transcoder.py:113-123, 335-338, 440-445`; `dashboard.py:231-236`;
  `memories.py:70-71`.
- What: The tools do not share one precision. Patching, the lens and the circuit scores run the model in float32, as
  the exporters call them. Dictionaries and transcoders read activations and splice under bfloat16 autocast.
  Transcoder training picks the k largest on bfloat16 pre-activations and evaluation in float32, and its clean loss
  is float32 beside replaced losses under autocast.
- Evidence: the lines; `gptpp/export/interp_common.py:21-36` loads without autocast.
- Effect: not measured.

### I11. The units of a dictionary

- Kind: quirk.
- Where: `sae.py:89-99, 142-162, 456-470, 478-479`.
- What: A dictionary reads `resid_post.l` at every token but the start mark and the first 16 of a window. The mean
  is subtracted and one constant sets the mean norm to sqrt(d), both from the first 32 batches; the bias starts at
  the geometric median of 32,768 samples. Activations, thresholds and clamp values are in those units.
- Evidence: `test_normalization_centers_scales_to_sqrt_d_and_restores`,
  `test_harvest_leaves_out_the_mark_and_the_first_tokens_of_a_window`.
- Effect: u_i of the series' equation (10.10) is the unit row divided by the scale.

### I12. The dictionaries were trained at four times the documented rate

- Kind: quirk.
- Where: `sae.py:35-36, 405, 642`.
- What: The docstring gives the rate 2e-4 / sqrt(n / 2^14); the command line's default is four times that
  (`--lr-scale 4.0`), and the series' dictionaries used it (0.0008 at n = 16,384), at a batch of 2,048 on 33,554,432
  vectors, where the command line's default for `--tokens` is 100e6.
- Evidence: the records beside the dictionaries.
- Effect: at eight times, dense and dead latents appeared; the error of gpt-pp's dictionary was still falling at the
  end.

### I13. Two meanings of "dead"

- Kind: known issue.
- Where: `sae.py:425, 445-449, 584`; `transcoder.py:386-390, 458`.
- What: "Dead" has two meanings: in training, no firing for 1,000,000 tokens (the autoencoder
  recomputes the mask at every step once a latent is dead and refreshes that flag every 64 steps; the transcoder
  refreshes its list of dead latents every 64 steps, so a revived one can stay in the auxiliary term for up to 63
  steps); in
  evaluation, no firing on the held-out tokens read, 9.8 million for gpt-pp, 5.1 million for pico, 262,144 for the
  transcoders, not the 10^7 of the series' Definition 10.3.3.
- Evidence: results/interp/dictionaries.json.
- Effect: "No latent is dead" is by the second.

### I14. The auxiliary loss differs between the autoencoder and the transcoder

- Kind: known issue.
- Where: `sae.py:376-381`; `transcoder.py:349-354`.
- What: The auxiliary loss differs between the two files: the autoencoder feeds the dead latents' pre-activations to
  the decoder as they are, negative ones included; the transcoder passes them through a ReLU. Both divide by the
  spread of the error in the batch, not by the data's variance.
- Evidence: the lines; `test_auxk_revives_dead_latents_and_without_it_they_stay_dead` covers the first only.
- Effect: not measured.

### I15. Transcoders: one a layer, and many latents that never fire

- Kind: known issue.
- Where: `transcoder.py:17-38, 311-317, 358-397`.
- What: One transcoder an MLP, not cross-layer; it reads the normalized stream the MLP reads. The start mark is
  trained on but left out of the constants and of the reported error. The rate is eight times the rule.
- Evidence: module docstring; the record of pico's transcoders.
- Effect: in blocks 2 to 6 of pico 50% to 61% of the latents never fire on held-out text (6% in block 1); they die
  in the last third of training.

### I16. Sparse sums through `F.embedding_bag`, and forward hooks on the fast path

- Kind: PyTorch.
- Where: `sae.py:356, 379`; `transcoder.py:96, 116, 323`.
- What: The k active rows are summed by `F.embedding_bag(..., per_sample_weights=, mode="sum")`, which carries
  gradients to the table and to the weights. The transcoders read and replace MLP outputs with
  `register_forward_hook` on the fast path, not with the lab's `Hooks`.
- Evidence: `test_a_topk_code_has_k_active_latents_and_decodes_to_the_sum_of_k_rows`;
  `test_what_is_read_of_the_mlps_is_what_the_model_computes`.
- Effect: none on numbers.

### I17. What the attribution graph holds fixed

- Kind: known issue.
- Where: `attribution_graph.py:114-151, 377-425`.
- What: The graph holds attention patterns and normalization denominators at the prompt's values and has no edge
  through the scores. It is built only for pre-norm models with MLPs and a norm that divides. `linear_effects` lets
  a later latent move with its pre-activation, right only while it stays among its token's k largest; `ablate` runs
  the model itself.
- Evidence: `test_what_is_not_supported_says_so`,
  `test_linear_effects_are_the_frozen_model_with_one_latent_taken_out`; results/interp/attribution-graph.json.
- Effect: on the figure's prompt the correlations are 0.27 and 0.54; three of 12 prompts pass.

### I18. Nodes, outputs and error of an attribution graph

- Kind: quirk.
- Where: `attribution_graph.py:124, 190-194, 326-342`.
- What: An embedding node is what enters the first block, after the embedding's norm. Outputs are at most ten tokens
  reaching 95%. On a pruned graph what was pruned counts as error.
- Evidence: module docstring 27-45.
- Effect: the replacement score of the figure's prompt is 0.912 before pruning and 0.488 after.

### I19. The grokking run uses neither the lab's trainer nor its optimizers

- Kind: quirk.
- Where: `grokking.py:65-82, 114-164, 171-174, 185-262`.
- What: The run does not use the lab's trainer or optimizers: full-batch `torch.optim.AdamW` on every parameter,
  biases and both tables included, rate 1e-3, decay 1, betas (0.9, 0.98), no schedule. The model is the lab's `GPT`
  with `bias=True`, every matrix redrawn from N(0, 1/d_model); the loss is taken in float64, on the p residues only.
  The key frequencies are read from the final embedding (a tenth of the strongest one's power) and applied to every
  saved state afterwards; the restricted loss keeps four products a frequency plus each class's mean and is reported
  on the held-out pairs.
- Evidence: module docstring; `test_the_model_is_the_papers_shape_and_a_short_run_fits_its_training_pairs`;
  results/interp/part-8.json.
- Effect: the run shares the model file with the series and nothing of its training recipe; one seed; measured at
  301 log-spaced steps, so "from step 8,129" is known to the spacing (the step before is 7,916).

### I20. Two kinds of n-gram model in one comparison

- Kind: quirk.
- Where: `ngram_order.py:42-88, 97-117`.
- What: The predictors are unsmoothed tables of the whole training stream, pairs counted across documents. The
  divergence runs from the table to the model, the finite direction. The levels drawn beside them are held-out
  losses of Kneser-Ney models on 200 million tokens. The pair table is dense, V by V.
- Evidence: module docstring; results/interp/part-8.json.
- Effect: two n-gram objects in one figure; the table suits a vocabulary of 4,096 only.

### I21. The optimizers of the rare-token runs

- Kind: quirk.
- Where: `frequency_buckets.py:24-35, 62-93, 141-151`.
- What: "SGD with momentum" is Nesterov momentum; "sign descent with momentum" is the lab's Lion at beta1 = beta2 =
  0.9; Adam keeps the recipe's epsilon, 1e-10; one rate for every group; a step is four passes of 64 sequences. A
  token joins the tenth that holds the middle of its share.
- Evidence: module docstring.
- Effect: buckets hold about a tenth each.

### I22. Reading an MLP as a memory

- Kind: quirk.
- Where: `memories.py:19-34, 46-47, 56-59, 69-115`.
- What: The memory coefficient is the hook `mlp.l.post`; a value's ranking is W_U v_i with no norm and no cap, right
  for an RMS norm without gain; gated MLPs are not covered. The pass over the text is stopped by an exception raised
  in the last hook.
- Evidence: module docstring; `test_top_triggers_are_the_brute_force_ones`.
- Effect: none on numbers.

### I23. Tools that run on one machine, and what the review did not cover

- Kind: known issue.
- Where: `gptpp/interp/copy_test.py:24, 29`; `measure_circuits.py:268`; docs/REVIEW.md, "Not checked".
- What: `copy_test.py` reads checkpoints from an absolute path of the machine the lab was built on whatever
  `$GPTPP_WORK` says (the package's own default is that path too: K60); `measure_circuits.py` reads an export of the site, from a
  directory named by `GPTPP_SITE`, and skips those numbers silently when it is absent. The review did not cover the circuit models or the rare-token runs, and each tool's model has one seed.
- Evidence: the lines; docs/REVIEW.md.
- Effect: the copy test runs on the series' machine only.

### I24. A latent's logit effects

- Kind: quirk.
- Where: `dashboard.py:66-74, 120-129, 216-238`.
- What: A latent's logit effects are its direct path only, without the division by the stream's length and without
  the cap; `removal` runs the model in float32 and differences capped logits. A record is cached with
  activations in float16.
- Evidence: the docstrings of `through_final_norm` and `removal`; `Codes.save`.
- Effect: two rankings of tokens a latent that need not agree.

### I25. A dashboard cache keyed by file date

- Kind: bug, fixed.
- Where: `dashboard.py:77-81, 107-117`.
- What: The record of a dictionary's activations was reused when its file was newer than the dictionary's;
  evaluating a dictionary again rewrites its file unchanged, so dates did not say which weights a record was made
  with. Since commit 702f0af a record carries a signature of the weights and is reused only when it matches.
- Evidence: the docstring of `signature`; `git diff b66f6b9 702f0af -- gptpp/interp/dashboard.py`.
- Effect: none known on a published number.
