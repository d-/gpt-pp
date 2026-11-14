# Evaluation

How the models of this repository are measured, and what the released models they are set beside score under
the same measurement. Everything here is produced by this code:

| what | code | writes |
|:--|:--|:--|
| bits per byte on held-out documents | `gptpp/evaluate.py` | `results/eval/<model>.json`, section `bits_per_byte` |
| benchmarks taken by log-likelihood | `gptpp/benchmarks.py` | the same file, section `benchmarks` |
| the released models, the tables | `scripts/eval_released_models.py` | one file a model, `summary.json` |
| hosted models on the same items | `scripts/eval_openrouter.py` | `results/eval/hosted/<model>.json` |
| stories graded by a hosted model | `scripts/judge_stories.py` | grades (JSONL) and a summary |
| the held-out set "fresh" | `scripts/make_fresh_corpus.py` | `$GPTPP_WORK/data/fresh/val_docs.jsonl` |

All numbers below were measured on 2026-11-14 on one GPU (NVIDIA Graphics Device, 32 GB), torch 2.9.1+cu130, transformers 5.23.0.
The tables are printed by `scripts/eval_released_models.py --table` and `scripts/eval_openrouter.py --table`.

## 1. Bits per byte

A language model gives a text a probability. Minus its base-two logarithm is the length, in bits, of the code an
arithmetic coder driven by the model would write for the text. Divided by the text's length in UTF-8 bytes it is
a number that does not depend on the tokenizer: a model with a vocabulary of 4,096 tokens and one with 151,936
can be compared on the same documents. Eight bits per byte is a model that knows nothing; English prose is about
one.

### The protocol

1. Each document is encoded by the model's own tokenizer, with no special tokens added.
2. The tokens are cut into consecutive windows. A window is the model's beginning-of-text token followed by at
   most `window - 1` tokens of the document, where `window = min(the model's context, 1024)`.
3. The model scores every token after the first: `-log p(token | beginning-of-text, earlier tokens of the window)`.
4. `bits per byte = (sum of the scores, in bits, over all documents) / (sum of the documents' UTF-8 bytes)`.

Also reported for each set: the loss in nats per token, the perplexity, tokens per byte and bytes per token
(`bits per byte = loss / ln 2 x tokens per byte`), the bits of every document (`doc_bits`, `doc_bytes`), and the
mean bits per token by position in the window (`by_position`).

### The choices, and why

| choice | what we do | why |
|:--|:--|:--|
| window | `min(context, 1024)` tokens, the beginning-of-text token included; windows do not overlap | 1,024 is the context of GPT-2, the oldest model in the table; a model that could read further is held to the same number of tokens. A model with a shorter context (TinyStories 33M: 512; our ladder models: 512; pico: 256) is scored in its own |
| beginning of text | every window begins with the model's own token: `<\|endoftext\|>` for the GPT-2 family, Pythia, SmolLM2 and TinyStories 33M, `<\|bos\|>` for ours. Qwen3 defines none: its end-of-text token, which separated its training documents, is used | the first token of a window is then predicted from something every model was trained to see, and every byte of the document is paid for |
| what is scored | every token of every document, once | the total is then the length of a real code for the documents: nothing is skipped at a window boundary (a test checks it) |
| what is not scored | the beginning-of-text token itself; no end-of-text token is appended | the first is given, not predicted; models differ in whether they were trained to predict the second |
| documents | a document starts a new window; documents with no text are dropped; the text is not normalized, stripped or de-tokenized | a short document (a story) is one window, scored from its first byte |
| padding | windows are sorted by length and padded on the right | in a causal model padding lies in the future of every real token; `padding_error` measures it for each model (below) |
| arithmetic | float32 weights and activations with TensorFloat-32 matrix products (`--precision tf32`); log-probabilities in float32, sums in float64 | twice as fast as exact float32 and equal to the digits we report (below) |
| which documents | the first N of a set whose order is already random (N in the table of section 2); the fingerprint of the documents is stored with every result | two results are comparable only when their fingerprints are equal; the tables warn when they are not |

A token that the tokenizer cannot reproduce (an unknown character, a normalization) would make the bits a code
for something other than the document's bytes. `n_docs_not_reproduced` counts the documents for which
`decode(encode(text)) != text`: it is 0 for every model and set here.

### What the windows cost

The first tokens of every window are predicted from little context. GPT-2 small, bits per token by position in
the window:

| set | 1 | 2-3 | 4-7 | 8-15 | 16-31 | 32-63 | 64-127 | 128-255 | 256-511 | 512-1023 | all |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| wikitext103 | 14.89 | 10.56 | 8.77 | 7.27 | 6.35 | 5.64 | 5.29 | 5.00 | 4.73 | 4.62 | 4.87 |
| fineweb-edu | 11.76 | 8.29 | 6.99 | 6.12 | 5.42 | 5.01 | 4.83 | 4.67 | 4.54 | 4.49 | 4.67 |

On documents much longer than a window (WikiText-103), a token deep in a window costs 4.62 bits against
4.87 on average: non-overlapping windows charge about 6% more than a window that always
had 512 or more tokens behind it would. This is the same for every model, and it is the price of scoring every
token exactly once.

### The trainer's own bits per byte are a different number

The trainer evaluates on a fixed first slice of the held-out tokens (1,048,576 of them, in windows of the
context length that start anywhere in the stream) and converts its loss with one constant, the bytes per token
of the whole held-out file: 4.6283 for FineWeb-EDU, 4.0278 for TinyStories. The tokens it actually scores have
4.6353 and 4.0640 bytes each, so its figure is a little high: the main run's 0.9429 would read 0.9415, and
pico's 0.552 would read 0.547. It is kept as it is because every curve and every run of the lab is converted
the same way, which is what a comparison between runs needs; a comparison with any other model is made under
the protocol above, which counts the bytes of each document it scores (0.9322 for the main model). The two also
differ by the windows: the trainer's start mid-document, the protocol's at the start of a document.

### Precision

GPT-2 small on WikiText-103: 1.0719 bits per byte in exact float32, 1.0720 with TensorFloat-32 products, 1.0751 in bfloat16 autocast. TensorFloat-32 changes the fourth decimal; bfloat16 autocast, the
way our own models train, changes the third for this model, which was not trained in it. The released models
and ours are all evaluated in `tf32`.
On the first 20 articles (float32 / tf32 / bf16): Pythia 160M 1.1164 / 1.1164 / 1.2917, SmolLM2 135M 0.9802 / 0.9802 / 0.9804, Qwen3 0.6B 0.9020 / 0.9020 / 0.9020. TensorFloat-32 equals float32 to four decimals in every case; bfloat16 is safe for the models that were trained in it and costs Pythia 160M sixteen percent. (`python -m gptpp.evaluate --model hf:<id> --docs wikitext103 [--max-docs 20] --precision <p>` prints each of these.)

Scoring a sequence alone or inside a padded batch changes its score by at most 1e-04 nats per token
(exact float32, random sequences that cost about 10 nats a token; one value a model in its result file,
`padding_error_nats_per_token`): rounding, not a leak of the padding.

### Standard errors

The documents are the sampled units. The standard error of bits per byte is that of a ratio of two sums over
documents (the linearization of the ratio estimator, Cochran 1977). With 60 articles or 8 essays it is wide; it
says how much the number would move with other documents of the same kind, not how well two models can be told
apart. For that, `scripts/eval_released_models.py --compare A B` gives the difference on the same documents with
its paired standard error, which is several times smaller: GPT-2 medium against GPT-2 small on fineweb-edu is -0.0823 ± 0.0008 bits per byte, where each model's own standard error is 0.0065. For the benchmarks the same command gives the paired difference in accuracy item by item.

## 2. The held-out sets

| name | source | documents used | bytes | fingerprint |
|:--|:--|--:|--:|:--|
| `wikitext103` | `Salesforce/wikitext` at `b08601e04326`, `wikitext-103-raw-v1/test-00000-of-00001.parquet` | 60 of 60 | 1,287,656 | `8be9b692dca47dc9` |
| `tinystories` | `roneneldan/TinyStories` at `f54c09fd2331`, `TinyStoriesV2-GPT4-valid.txt`, as written by `scripts/prepare_tinystories.py` to `$GPTPP_WORK/data/tinystories/val_docs.jsonl` (27,630 stories) | 2,000 of 27,630 | 1,615,901 | `9deec9fb3736a4f0` |
| `fineweb-edu` | `HuggingFaceFW/fineweb-edu` at `87f09149ef47`, `sample/10BT`, as held out by `scripts/prepare_fineweb.py` in `$GPTPP_WORK/data/fineweb-edu/val_docs.jsonl` (9,649 documents) | 1,000 of 9,649 | 4,579,235 | `1eb74dad69d54434` |
| `fresh` | `scripts/make_fresh_corpus.py`, `$GPTPP_WORK/data/fresh/val_docs.jsonl` | 8 of 8 | 430,307 | `dbfdc02fa9921f78` |

- **wikitext103.** The rows of the parquet file are the lines of the original file; an article begins at a line
  ` = Title = ` that stands between two empty rows (60 articles, as Merity et al. 2016 count them). The text is
  the raw version, as distributed: ` @-@ `, spaced punctuation and all.
- **tinystories.** The data preparation takes the second version of the dataset (stories written by GPT-4 only),
  removes from the training split any story that is also in the validation split, and writes the validation
  stories in a seeded random order; we score the first 2,000. The dataset's two versions share stories: 695 of these 2,000 are, word for word, in the training file of the first version, on which TinyStories 33M was trained (section 6; `scripts/eval_released_models.py --tinystories-overlap`, `results/eval/tinystories_overlap.json`).
- **fineweb-edu.** The documents that the data preparation holds out of our own training set (and whose text or
  URL occurs nowhere in it), in a seeded random order; we score the first 1,000. They are held out of *our*
  training data only: SmolLM2 was trained on FineWeb-Edu itself, and the other models on web text that may
  include the same pages.
- **fresh.** The only set no released model can have read: essays published on the web in October and November
  2026. `scripts/make_fresh_corpus.py` keeps paragraphs of plain prose and drops the front matter, HTML blocks
  (figures, scripts, styles, tables), Liquid tags, code, mathematics, headings, lists, tables, block quotations
  (they quote older texts) and sidenotes; a paragraph that still contains markup, code or math after cleaning is
  dropped whole. Eight essays give 500 paragraphs and 430,307 bytes; 397 blocks are dropped (169 headings, lists, quotations and tables; 135 with markup, code or math left in them; 52 shorter than 200 characters; 36 that do not end as a sentence does; 5 that are mostly figures), and three pages of the same months that hold no prose are skipped. Short quotations inside a paragraph remain. The text stays on this machine: it is
  scored locally and never sent to a hosted model.

What each released model can have read of these sets is recorded in its result file (`model.seen`) and listed in
section 4.

## 3. The benchmarks

A model that only predicts the next token answers a multiple-choice question by scoring each option as a
continuation. The sequence scored is `[beginning-of-text] + tokens(given text) + tokens(option)`, the two parts
encoded separately; if it exceeds the window the given text loses its beginning. The same code scores our
checkpoints and the Hugging Face models.

| task | Hub dataset | revision | file | items | chance | the option credited |
|:--|:--|:--|:--|--:|--:|:--|
| `hellaswag` | `Rowan/hellaswag` | `218ec52e09a7` | `data/validation-00000-of-00001.parquet` | 10,042 | 25.0% | highest log-probability per byte |
| `blimp` | `nyu-mll/blimp` | `877fba0801ff` | `<paradigm>/train-00000-of-00001.parquet`, 67 files | 67,000 | 50.0% | highest total log-probability |
| `arc_easy` | `allenai/ai2_arc` | `210d026faf99` | `ARC-Easy/test-00000-of-00001.parquet` | 2,376 | 25.0% | highest log-probability per byte |
| `arc_challenge` | `allenai/ai2_arc` | `210d026faf99` | `ARC-Challenge/test-00000-of-00001.parquet` | 1,172 | 25.0% | highest log-probability per byte |
| `lambada` | `EleutherAI/lambada_openai` | `900124bf3b82` | `default/test/default.parquet` | 5,153 | 0.0% | every token of the last word is the first choice |
| `piqa` | `ybisk/piqa` | `142c51238b3c` | `plain_text/validation/0000.parquet` | 1,838 | 50.0% | highest log-probability per byte |
| `winogrande` | `allenai/winogrande` | `01e74176c635` | `winogrande_xl/validation-00000-of-00001.parquet` | 1,267 | 50.0% | highest total log-probability |
| `sciq` | `allenai/sciq` | `2c94ad3e1aaf` | `data/test-00000-of-00001.parquet` | 1,000 | 25.0% | highest log-probability per byte |
| `commonsense_qa` | `tau/commonsense_qa` | `94630fe30dad` | `data/validation-00000-of-00001.parquet` | 1,221 | 20.0% | highest log-probability per byte |

Formats (the given text, then the option):

- `hellaswag`: "<activity>: <context>" then " <ending>". HellaSwag, validation: which of four endings continues a described scene. Zellers et al., HellaSwag: can a machine really finish your sentence?, ACL 2019, arXiv:1905.07830.
- `blimp`: each sentence scored whole, from the beginning-of-text token. BLiMP, all 67 paradigms: which sentence of a minimal pair is grammatical. Warstadt et al., BLiMP: the benchmark of linguistic minimal pairs for English, TACL 2020, arXiv:1912.00582.
- `arc_easy`: "Question: <question>\nAnswer:" then " <option>". ARC-Easy, test: grade-school science questions, four options. Clark et al., Think you have solved question answering? Try ARC, the AI2 reasoning challenge, 2018, arXiv:1803.05457.
- `arc_challenge`: "Question: <question>\nAnswer:" then " <option>". ARC-Challenge, test: the science questions retrieval and co-occurrence methods fail. Clark et al., Think you have solved question answering? Try ARC, the AI2 reasoning challenge, 2018, arXiv:1803.05457.
- `lambada`: the passage without its last word, then the last word; right when every token of the word is the model's first choice. LAMBADA, test, OpenAI's detokenized copy: the last word of a passage of narrative. Paperno et al., The LAMBADA dataset: word prediction requiring a broad discourse context, ACL 2016, arXiv:1606.06031.
- `piqa`: "Question: <goal>\nAnswer:" then " <solution>". PIQA, validation: which of two ways of doing an everyday physical task works. Bisk et al., PIQA: reasoning about physical commonsense in natural language, AAAI 2020, arXiv:1911.11641.
- `winogrande`: the sentence up to the blank with an option filled in, then the rest of the sentence. WinoGrande, validation: which of two names a blank refers to. Sakaguchi et al., WinoGrande: an adversarial Winograd schema challenge at scale, 2019, arXiv:1907.10641.
- `sciq`: "<support>\nQuestion: <question>\nAnswer:" then " <option>". SciQ, test: science exam questions with a supporting paragraph, four options. Welbl et al., Crowdsourcing multiple choice science questions, W-NUT 2017, arXiv:1707.06209.
- `commonsense_qa`: "Question: <question>\nAnswer:" then " <option>". CommonsenseQA, validation: everyday knowledge, five options. Talmor et al., CommonsenseQA: a question answering challenge targeting commonsense knowledge, NAACL 2019, arXiv:1811.00937.

Three rules are computed for every item and stored (`acc`, `acc_norm`, `acc_per_token`): the option with the
highest total log-probability, the highest per UTF-8 byte of the option, the highest per token. One is the
task's headline (`accuracy`, with `stderr`, the per-item marks `correct` and the breakdown by group):

- **per byte** where the options are free text of unequal length (HellaSwag, ARC, PIQA, SciQ, CommonsenseQA). A
  long option is then not punished for its length, and the normalization does not depend on the tokenizer. The
  total log-probability prefers short options, which helps where the right answers are short (ARC-Easy, SciQ)
  and hurts where they are long (ARC-Challenge, where it puts GPT-2 small below chance); see the table of
  section 5.
- **total** where the two candidates are the same length or the same text: BLiMP (two whole sentences, as in
  the benchmark's paper) and WinoGrande (the same rest of sentence after either name).
- **LAMBADA** has one candidate: an item is right when every token of the last word is the model's first choice
  given the passage. The perplexity of the last word is also stored.

Options that tie share the credit, so a model that cannot tell the options apart scores exactly the chance level
(a test checks it). The standard error is that of a mean over items. For BLiMP, `stderr_across_groups` treats
the 67 paradigms as the sample instead: it is 13 to 16 times larger and is the one to use for the question
"would another set of constructions give the same score".

The item formats of HellaSwag, ARC, PIQA, SciQ, WinoGrande and LAMBADA are the ones in common use (those of
EleutherAI's lm-evaluation-harness, written again here), with one difference: every sequence begins with the
beginning-of-text token, which the harness omits when a context is given. CommonsenseQA is scored on the answer
text, not on a letter.

The **centered mean** of a model is the mean over the nine tasks of `(accuracy - chance) / (1 - chance)`: 0 for
guessing everywhere, 100 for a perfect score everywhere.

## 4. The released models

| model | Hub id | revision | parameters | without embeddings | layers x width | window | bytes per token on fineweb-edu | year |
|:--|:--|:--|--:|--:|:--|--:|--:|--:|
| GPT-2 small | `openai-community/gpt2` | `607a30d783df` | 124.4M | 85.8M | 12 x 768 | 1024 | 4.64 | 2019 |
| GPT-2 medium | `openai-community/gpt2-medium` | `6dcaa7a952f7` | 354.8M | 303.4M | 24 x 1024 | 1024 | 4.64 | 2019 |
| GPT-2 large | `openai-community/gpt2-large` | `32b71b12589c` | 774.0M | 709.7M | 36 x 1280 | 1024 | 4.64 | 2019 |
| GPT-2 XL | `openai-community/gpt2-xl` | `15ea56dee5df` | 1557.6M | 1477.2M | 48 x 1600 | 1024 | 4.64 | 2019 |
| DistilGPT2 | `distilbert/distilgpt2` | `2290a62682d0` | 81.9M | 43.3M | 6 x 768 | 1024 | 4.64 | 2019 |
| Pythia 160M | `EleutherAI/pythia-160m` | `50f5173d932e` | 162.3M | 85.1M | 12 x 768 | 1024 | 4.65 | 2023 |
| SmolLM2 135M | `HuggingFaceTB/SmolLM2-135M` | `93efa2f097d5` | 134.5M | 106.2M | 30 x 576 | 1024 | 4.56 | 2025 |
| TinyStories 33M | `roneneldan/TinyStories-33M` | `2ad0a164221b` | 68.5M | 29.9M | 4 x 768 | 512 | 4.64 | 2023 |
| SmolLM2 360M | `HuggingFaceTB/SmolLM2-360M` | `f8027fd0eaee` | 361.8M | 314.6M | 32 x 960 | 1024 | 4.56 | 2025 |
| Qwen3 0.6B base | `Qwen/Qwen3-0.6B-Base` | `da87bfb608c1` | 596.0M | 440.5M | 28 x 1024 | 1024 | 4.64 | 2025 |

Parameters are counted once where the embedding and the unembedding share a matrix; "without embeddings" leaves
out the token embedding (and the unembedding where it is a separate matrix). `google/gemma-3-270m` is not in the
table: its repository is gated and the token in use has not been given access (HTTP 403, `GatedRepoError`). Once the license is accepted on the Hub for that account, `scripts/eval_released_models.py --models gemma-3-270m` adds it.

What is public about the training data of each (from its model card on the Hub, read on 2026-11-14), and which of
our held-out sets it can have read:

- **GPT-2 small** (Radford et al., Language models are unsupervised multitask learners, 2019). WebText: 40 GB of text from the web pages linked from Reddit posts with at least 3 karma; all Wikipedia pages removed; never released. Sequences of 1,024 tokens; the amount of training was not disclosed. Could it have read our held-out sets? `wikitext103`: no (Wikipedia was removed from WebText); `tinystories`: no (written in 2023); `fineweb-edu`: possible (web pages of 2017 and earlier); `fresh`: no.
- **GPT-2 medium** (Radford et al., Language models are unsupervised multitask learners, 2019). WebText, as GPT-2 small. Could it have read our held-out sets? `wikitext103`: no (Wikipedia was removed from WebText); `tinystories`: no (written in 2023); `fineweb-edu`: possible (web pages of 2017 and earlier); `fresh`: no.
- **GPT-2 large** (Radford et al., Language models are unsupervised multitask learners, 2019). WebText, as GPT-2 small. Could it have read our held-out sets? `wikitext103`: no (Wikipedia was removed from WebText); `tinystories`: no (written in 2023); `fineweb-edu`: possible (web pages of 2017 and earlier); `fresh`: no.
- **GPT-2 XL** (Radford et al., Language models are unsupervised multitask learners, 2019). WebText, as GPT-2 small. Could it have read our held-out sets? `wikitext103`: no (Wikipedia was removed from WebText); `tinystories`: no (written in 2023); `fineweb-edu`: possible (web pages of 2017 and earlier); `fresh`: no.
- **DistilGPT2** (Sanh et al., DistilBERT, a distilled version of BERT, 2019, arXiv:1910.01108). OpenWebText, an open reproduction of WebText, with GPT-2 small as the teacher (knowledge distillation). Could it have read our held-out sets? `wikitext103`: not stated; `tinystories`: no (written in 2023); `fineweb-edu`: possible (web pages); `fresh`: no.
- **Pythia 160M** (Biderman et al., Pythia: a suite for analyzing large language models across training and scaling, 2023, arXiv:2304.01373). The Pile (825 GiB of English text from 22 sources), not deduplicated: 143,000 steps of 2,097,152 tokens, about 300 billion tokens, just under one epoch. Could it have read our held-out sets? `wikitext103`: possible (the Pile's 22 sources are listed in Gao et al. 2020, arXiv:2101.00027); `tinystories`: no (the Pile is of 2020); `fineweb-edu`: possible (web pages); `fresh`: no.
- **SmolLM2 135M** (Allal et al., SmolLM2: when smol goes big, 2025, arXiv:2502.02737). 2 trillion tokens of FineWeb-Edu, DCLM, The Stack and further filtered datasets. Could it have read our held-out sets? `wikitext103`: possible; `tinystories`: possible; `fineweb-edu`: yes (FineWeb-Edu is in its training data; our held-out documents are held out of our training set, not of its); `fresh`: no.
- **TinyStories 33M** (Eldan and Li, TinyStories: how small can language models be and still speak coherent English?, 2023, arXiv:2305.07759). TinyStories (training split) only, with a context of 512 tokens. Could it have read our held-out sets? `wikitext103`: no; `tinystories`: the validation split is held out; the training split is its only data; `fineweb-edu`: no; `fresh`: no.
- **SmolLM2 360M** (Allal et al., SmolLM2: when smol goes big, 2025, arXiv:2502.02737). 4 trillion tokens of FineWeb-Edu, DCLM, The Stack and further filtered datasets. Could it have read our held-out sets? `wikitext103`: possible; `tinystories`: possible; `fineweb-edu`: yes (FineWeb-Edu is in its training data; our held-out documents are held out of our training set, not of its); `fresh`: no.
- **Qwen3 0.6B base** (Yang et al. (Qwen Team), Qwen3 technical report, 2025, arXiv:2505.09388). The Qwen3 family is pre-trained on 36 trillion tokens in 119 languages (web, code, books, synthetic data); what the 0.6B model itself saw is not stated. Could it have read our held-out sets? `wikitext103`: possible; `tinystories`: possible; `fineweb-edu`: possible; `fresh`: no.

## 5. Results

Bits per byte (lower is better), with the standard error over documents:

| model | parameters | window | wikitext103 | tinystories | fineweb-edu | fresh |
|:--|--:|--:|--:|--:|--:|--:|
| GPT-2 small | 124M | 1024 | 1.072 ± 0.010 | 0.856 ± 0.002 | 1.007 ± 0.007 | 1.297 ± 0.030 |
| GPT-2 medium | 355M | 1024 | 0.975 ± 0.011 | 0.759 ± 0.002 | 0.925 ± 0.007 | 1.229 ± 0.034 |
| GPT-2 large | 774M | 1024 | 0.928 ± 0.010 | 0.707 ± 0.002 | 0.881 ± 0.006 | 1.188 ± 0.036 |
| GPT-2 XL | 1558M | 1024 | 0.898 ± 0.010 | 0.679 ± 0.002 | 0.854 ± 0.006 | 1.163 ± 0.037 |
| DistilGPT2 | 82M | 1024 | 1.201 ± 0.011 | 0.964 ± 0.002 | 1.084 ± 0.007 | 1.393 ± 0.028 |
| Pythia 160M | 162M | 1024 | 1.094 ± 0.010 | 0.882 ± 0.002 | 0.998 ± 0.007 | 1.309 ± 0.034 |
| SmolLM2 135M | 135M | 1024 | 0.972 ± 0.012 | 0.692 ± 0.002 | 0.857 ± 0.006 | 1.207 ± 0.039 |
| TinyStories 33M | 69M | 512 | 3.380 ± 0.032 | 0.436 ± 0.002 | 2.875 ± 0.025 | 2.716 ± 0.052 |
| SmolLM2 360M | 362M | 1024 | 0.869 ± 0.010 | 0.618 ± 0.002 | 0.788 ± 0.006 | 1.132 ± 0.040 |
| Qwen3 0.6B base | 596M | 1024 | 0.888 ± 0.009 | 0.638 ± 0.002 | 0.817 ± 0.007 | 1.155 ± 0.039 |

Accuracy in percent with the standard error over items; `chance` is the level of guessing:

| model | hellaswag | blimp | arc_easy | arc_challenge | lambada | piqa | winogrande | sciq | commonsense_qa | centered mean |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| *chance* | *25.0* | *50.0* | *25.0* | *25.0* | *0.0* | *50.0* | *50.0* | *25.0* | *20.0* | *0.0* |
| GPT-2 small | 31.3 ± 0.5 | 82.5 ± 0.1 | 40.5 ± 1.0 | 22.2 ± 1.2 | 33.7 ± 0.7 | 62.6 ± 1.1 | 50.5 ± 1.4 | 68.2 ± 1.5 | 30.1 ± 1.3 | 24.5 |
| GPT-2 medium | 39.9 ± 0.5 | 82.6 ± 0.1 | 43.7 ± 1.0 | 25.4 ± 1.3 | 43.9 ± 0.7 | 65.9 ± 1.1 | 53.0 ± 1.4 | 72.3 ± 1.4 | 33.5 ± 1.4 | 30.2 |
| GPT-2 large | 45.6 ± 0.5 | 83.5 ± 0.1 | 46.8 ± 1.0 | 25.5 ± 1.3 | 48.5 ± 0.7 | 69.4 ± 1.1 | 54.5 ± 1.4 | 73.6 ± 1.4 | 34.8 ± 1.4 | 33.8 |
| GPT-2 XL | 51.4 ± 0.5 | 83.3 ± 0.1 | 50.2 ± 1.0 | 27.1 ± 1.3 | 51.4 ± 0.7 | 70.3 ± 1.1 | 57.6 ± 1.4 | 79.4 ± 1.3 | 38.0 ± 1.4 | 37.8 |
| DistilGPT2 | 27.6 ± 0.4 | 79.5 ± 0.2 | 38.4 ± 1.0 | 23.1 ± 1.2 | 25.1 ± 0.6 | 59.7 ± 1.1 | 52.1 ± 1.4 | 65.6 ± 1.5 | 29.7 ± 1.3 | 21.4 |
| Pythia 160M | 30.7 ± 0.5 | 78.5 ± 0.2 | 40.3 ± 1.0 | 24.5 ± 1.3 | 35.7 ± 0.7 | 62.0 ± 1.1 | 52.3 ± 1.4 | 71.5 ± 1.4 | 31.5 ± 1.3 | 25.0 |
| SmolLM2 135M | 43.7 ± 0.5 | 80.4 ± 0.2 | 58.2 ± 1.0 | 29.0 ± 1.3 | 43.6 ± 0.7 | 68.7 ± 1.1 | 53.4 ± 1.4 | 81.5 ± 1.2 | 35.6 ± 1.4 | 35.3 |
| TinyStories 33M | 27.5 ± 0.4 | 63.6 ± 0.2 | 27.6 ± 0.9 | 23.0 ± 1.2 | 11.4 ± 0.4 | 54.0 ± 1.2 | 52.1 ± 1.4 | 24.9 ± 1.4 | 27.4 ± 1.3 | 7.1 |
| SmolLM2 360M | 56.5 ± 0.5 | 81.4 ± 0.2 | 67.3 ± 1.0 | 36.9 ± 1.4 | 54.5 ± 0.7 | 72.5 ± 1.0 | 59.6 ± 1.4 | 87.6 ± 1.0 | 43.1 ± 1.4 | 45.4 |
| Qwen3 0.6B base | 53.4 ± 0.5 | 80.5 ± 0.2 | 67.5 ± 1.0 | 40.0 ± 1.4 | 55.0 ± 0.7 | 70.6 ± 1.1 | 58.2 ± 1.4 | 92.0 ± 0.9 | 47.5 ± 1.4 | 45.8 |

The three rules side by side (`acc` / `acc_norm` / `acc_per_token`), for the tasks where they differ:

| model | hellaswag | arc_easy | arc_challenge | piqa | sciq | commonsense_qa |
|:--|--:|--:|--:|--:|--:|--:|
| GPT-2 small | 29.0 / 31.3 / 31.0 | 43.8 / 40.5 / 38.5 | 19.2 / 22.2 / 23.1 | 63.1 / 62.6 / 62.0 | 77.6 / 68.2 / 67.9 | 30.0 / 30.1 / 28.4 |
| GPT-2 medium | 33.3 / 39.9 / 39.6 | 49.4 / 43.7 / 42.3 | 21.4 / 25.4 / 24.7 | 67.3 / 65.9 / 67.1 | 79.3 / 72.3 / 68.2 | 32.6 / 33.5 / 30.3 |
| GPT-2 large | 36.6 / 45.6 / 44.8 | 53.3 / 46.8 / 45.8 | 21.6 / 25.5 / 24.4 | 70.3 / 69.4 / 69.7 | 82.6 / 73.6 / 71.0 | 34.5 / 34.8 / 32.3 |
| GPT-2 XL | 40.3 / 51.4 / 50.3 | 56.4 / 50.2 / 48.7 | 24.7 / 27.1 / 28.2 | 71.1 / 70.3 / 71.2 | 85.1 / 79.4 / 74.6 | 36.7 / 38.0 / 33.3 |
| DistilGPT2 | 27.4 / 27.6 / 27.6 | 40.4 / 38.4 / 35.8 | 19.1 / 23.1 / 23.5 | 60.3 / 59.7 / 59.5 | 73.1 / 65.6 / 66.0 | 26.1 / 29.7 / 26.6 |
| Pythia 160M | 28.8 / 30.7 / 29.8 | 44.6 / 40.3 / 38.7 | 20.1 / 24.5 / 23.8 | 61.8 / 62.0 / 62.5 | 77.5 / 71.5 / 71.1 | 27.0 / 31.5 / 27.9 |
| SmolLM2 135M | 35.6 / 43.7 / 43.0 | 64.4 / 58.2 / 58.7 | 28.2 / 29.0 / 30.7 | 67.9 / 68.7 / 68.6 | 84.8 / 81.5 / 80.7 | 35.0 / 35.6 / 32.0 |
| TinyStories 33M | 27.1 / 27.5 / 27.0 | 28.1 / 27.6 / 27.9 | 18.9 / 23.0 / 23.0 | 55.0 / 54.0 / 54.1 | 26.6 / 24.9 / 31.7 | 20.1 / 27.4 / 24.7 |
| SmolLM2 360M | 43.2 / 56.5 / 55.6 | 70.2 / 67.3 / 64.9 | 35.2 / 36.9 / 37.5 | 71.9 / 72.5 / 72.8 | 91.0 / 87.6 / 87.2 | 43.0 / 43.1 / 39.7 |
| Qwen3 0.6B base | 40.8 / 53.4 / 52.0 | 68.9 / 67.5 / 65.3 | 35.8 / 40.0 / 37.5 | 70.3 / 70.6 / 71.3 | 94.1 / 92.0 / 91.2 | 46.0 / 47.5 / 45.2 |

BLiMP by phenomenon (accuracy in percent; 67 paradigms of 1,000 pairs in 13 groups):

| model | anaphor agreement | argument structure | binding | control raising | determiner noun agreement | ellipsis | filler gap dependency | irregular forms | island effects | npi licensing | quantifiers | s-selection | subject verb agreement |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| GPT-2 small | 99.6 | 83.9 | 79.3 | 80.7 | 95.5 | 84.9 | 81.0 | 95.8 | 70.7 | 78.2 | 71.8 | 83.4 | 87.4 |
| GPT-2 medium | 99.4 | 83.9 | 77.8 | 83.0 | 96.3 | 86.4 | 81.3 | 94.8 | 71.8 | 74.7 | 74.0 | 82.0 | 88.3 |
| GPT-2 large | 99.2 | 84.2 | 80.6 | 83.0 | 96.2 | 89.2 | 81.1 | 94.8 | 72.7 | 77.5 | 77.5 | 81.4 | 87.6 |
| GPT-2 XL | 99.2 | 84.0 | 81.1 | 83.1 | 96.4 | 88.4 | 81.7 | 93.5 | 72.0 | 75.0 | 78.2 | 82.7 | 86.9 |
| DistilGPT2 | 97.2 | 82.6 | 76.3 | 76.2 | 92.3 | 79.5 | 76.2 | 95.6 | 65.1 | 76.4 | 67.7 | 83.2 | 87.7 |
| Pythia 160M | 98.2 | 78.8 | 74.0 | 79.0 | 92.1 | 85.8 | 77.1 | 84.5 | 63.8 | 72.5 | 70.7 | 79.7 | 87.2 |
| SmolLM2 135M | 98.0 | 83.4 | 73.2 | 83.6 | 96.7 | 86.3 | 78.7 | 85.4 | 68.9 | 69.2 | 71.4 | 81.7 | 87.7 |
| TinyStories 33M | 89.8 | 68.9 | 66.1 | 62.7 | 72.3 | 62.5 | 63.0 | 85.7 | 51.3 | 50.0 | 65.3 | 66.6 | 58.8 |
| SmolLM2 360M | 99.2 | 83.6 | 75.4 | 84.2 | 96.1 | 86.6 | 75.9 | 91.4 | 67.6 | 79.3 | 69.9 | 80.6 | 88.3 |
| Qwen3 0.6B base | 98.7 | 82.3 | 74.8 | 83.9 | 95.6 | 87.9 | 77.3 | 95.6 | 71.1 | 68.1 | 68.2 | 80.5 | 87.8 |

What the tables say, without interpretation:

- On three of the four sets the lowest bits per byte among the released models is SmolLM2 360M's; on `tinystories` it is TinyStories 33M's, which is between 2.7 and 3.4 bits per byte on everything else.
- Every model but TinyStories 33M is 0.29 to 0.35 bits per byte higher on `fresh` than on `fineweb-edu`. The order of the models is the same on both sets except for two pairs of neighbors: SmolLM2 135M is ahead of GPT-2 large on `fineweb-edu` and behind it on `fresh`, and Pythia 160M is ahead of GPT-2 small on `fineweb-edu` and behind it on `fresh`. `fresh` is eight documents: its standard errors are four to seven times those of `fineweb-edu`.
- At the size of our main model, GPT-2 small scores 1.007 on `fineweb-edu` and SmolLM2 135M 0.857; FineWeb-Edu is part of SmolLM2's training data.
- BLiMP does not separate the four sizes of GPT-2 (82.5 to 83.5), and the models of 2025 are below them (80.4 to 81.4). ARC-Challenge and WinoGrande are within four points of chance for every model under 300 million parameters.
- The rule matters as much as the model on some tasks: SciQ for GPT-2 small is 77.6 by total log-probability and 68.2 per byte; ARC-Challenge is 19.2 (below chance) and 22.2.

## 6. Checks against published numbers

| model | WikiText-103 word perplexity: published, ours on raw text, ours on de-tokenized text | LAMBADA last-word perplexity: published, ours | HellaSwag acc: harness, ours | HellaSwag acc_norm: harness, ours |
|:--|--:|--:|--:|--:|
| GPT-2 small | 37.50, 49.22, 36.94 | 35.13, 35.11 | 28.92, 29.03 | 31.14, 31.35 |
| GPT-2 medium | 26.37, 34.56, 26.65 | 15.60, 17.15 | , 33.30 | , 39.90 |
| GPT-2 large | 22.05, 29.17, 22.52 | 10.87, 12.01 | , 36.57 | , 45.58 |
| GPT-2 XL | 17.48, 26.18, 20.24 | 8.63, 10.13 | 40.04, 40.34 | 50.89, 51.44 |

- **HellaSwag.** For GPT-2 small and XL the harness figures (as quoted in the header of `hellaswag.py` in
  build-nanogpt) and ours agree to within 0.55 points on both rules. The remaining difference is
  the beginning-of-text token.
- **LAMBADA.** The perplexity of the last word is the quantity of table 3 of the GPT-2 paper. GPT-2 small:
  35.11 against the published 35.13. The larger models come out 10 to 17% above the published figures.
  The paper's accuracy column uses a stop-word filter and is not comparable; without it the paper gives 52.66%
  for the largest model, and we measure 51.37%.
- **WikiText-103.** The published perplexities are per word of the benchmark's own count (245,569 = 241,211
  words + 4,358 line ends; we recover the count exactly from the raw file), zero-shot, on de-tokenized text.
  Ours, on the raw text, are higher by a nearly constant factor for the three smaller models and by more for
  GPT-2 XL (1.31, 1.31, 1.32; GPT-2 XL: 1.50). Scored on text de-tokenized by our own rules (`scripts/eval_released_models.py --wikitext-detokenized`, `results/eval/wikitext103_detokenized.json`) and under our protocol otherwise, the three smaller models come within 2.2% of the published figures: 36.94, 26.65 and 22.52 against 37.50, 26.37 and 22.05. The released GPT-2 XL checkpoint does not (20.24 against 17.48), and we do not know why.
- **TinyStories.** TinyStories 33M scores 0.442 bits per byte on the 695 held-out stories that are also in its training file and 0.431 on the other 1,305; GPT-2 small, which has seen neither, scores 0.859 and 0.854. There is no sign that it gains from having seen them: it scores slightly worse on them, as GPT-2 small does. The whole set is therefore used.

## 7. Hosted models on the same items

A hosted chat model returns no log-probabilities. It is asked the question in words and its letter is read from
the reply. The items are a fixed subset of three of the tasks (`benchmarks.fixed_subset`: the items whose hashes
sort first, the same on every machine): 200 of HellaSwag, 200 of ARC-Easy, 300 BLiMP pairs as a forced choice
between the two sentences, their order shown decided by a hash of the item's number. The prompt is the item and
"Answer with the letter only."; there is no system prompt, nothing about the purpose, and no HTTP header beyond
the authorization. Each model is asked with as little hidden reasoning as it allows (`effort: none` where it can
be switched off, its lowest effort otherwise) and with temperature 0 where it accepts one. A reply with no
readable letter counts as wrong.

The models were chosen on 2026-11-14 from OpenRouter's public list by what the list says of them: OpenAI's "fast,
cost-efficient" model (GPT-6 Luna), its "mid-tier" model (GPT-6.1 Terra) and its "flagship" (GPT-6 Astra),
Anthropic's "most capable model" (Claude Fable 5.2), and two models whose weights are public (Qwen3.8 27B, a
dense model of 27 billion parameters, and LFM3 1.7B). The local models are scored by log-likelihood on the same items (read from
`correct` in their result files).

| model | hellaswag (n) | arc_easy (n) | blimp (n) | cost |
|:--|--:|--:|--:|--:|
| openai/gpt-6-luna (small) | 90.5 ± 2.1 (200) | 98.5 ± 0.9 (200) | 88.7 ± 1.8 (300) | $0.01 |
| openai/gpt-6.1-terra (mid) | 94.5 ± 1.6 (200) | 99.0 ± 0.7 (200) | 95.0 ± 1.3 (300) | $0.08 |
| openai/gpt-6-astra (frontier) | 97.5 ± 1.1 (200) | 99.5 ± 0.5 (200) | 96.7 ± 1.0 (300) | $1.17 |
| anthropic/claude-fable-5.2 (frontier) | 98.4 ± 0.9 (192) | 99.5 ± 0.5 (192) | 98.3 ± 0.8 (288) | $1.45 |
| qwen/qwen3.8-27b (open weights, 27B) | 96.0 ± 1.4 (200) | 99.0 ± 0.7 (200) | 87.7 ± 1.9 (300) | $0.01 |
| liquid/lfm-3-1.7b (open weights, 1.7B) | 63.9 ± 8.0 (36) | 100.0 ± 0.0 (34) | 68.8 ± 6.7 (48) | $0.02 |
| GPT-2 small (log-likelihood) | 28.0 ± 3.2 (200) | 46.0 ± 3.5 (200) | 79.7 ± 2.3 (300) | |
| GPT-2 medium (log-likelihood) | 35.5 ± 3.4 (200) | 46.5 ± 3.5 (200) | 82.0 ± 2.2 (300) | |
| GPT-2 large (log-likelihood) | 41.0 ± 3.5 (200) | 51.5 ± 3.5 (200) | 83.7 ± 2.1 (300) | |
| GPT-2 XL (log-likelihood) | 47.5 ± 3.5 (200) | 52.5 ± 3.5 (200) | 83.0 ± 2.2 (300) | |
| DistilGPT2 (log-likelihood) | 22.5 ± 3.0 (200) | 43.5 ± 3.5 (200) | 79.0 ± 2.4 (300) | |
| Pythia 160M (log-likelihood) | 25.0 ± 3.1 (200) | 48.0 ± 3.5 (200) | 77.0 ± 2.4 (300) | |
| SmolLM2 135M (log-likelihood) | 38.0 ± 3.4 (200) | 64.5 ± 3.4 (200) | 79.0 ± 2.4 (300) | |
| TinyStories 33M (log-likelihood) | 23.5 ± 3.0 (200) | 32.0 ± 3.3 (200) | 63.7 ± 2.8 (300) | |
| SmolLM2 360M (log-likelihood) | 52.0 ± 3.5 (200) | 73.0 ± 3.1 (200) | 79.7 ± 2.3 (300) | |
| Qwen3 0.6B base (log-likelihood) | 54.0 ± 3.5 (200) | 71.5 ± 3.2 (200) | 78.7 ± 2.4 (300) | |

- **Cost.** The requests of the six models cost $2.74 by OpenRouter's own figure for each request; with the pilots, a first test and a first attempt at LFM3 that was stopped, the evaluation spent $2.79 against a cap of $5 (the account's counter, which stood at 0 before: $2.794, of which $0.008 is the test of the story grader).
- **Items.** Claude Fable 5.2 was asked 672 of the 700 items (192, 192, 288): its share of the cap ran out. The items a model was asked are always a prefix of the fixed subset, and are listed in its file.
- **Replies that are not a letter.** GPT-6 Luna answered "Neither." to 8 BLiMP pairs and GPT-6 Astra "A, B" or "A and B" to 6: no answer, counted wrong. Claude Fable 5.2 wrote an explanation in 32 replies: 30 end with their letter and are read by it, 2 do not and count as wrong. The first run kept only the first 200 characters of a reply, so 38 of its items were asked a second time ($0.20) and the replies kept whole; `--reread` reads the stored replies again.
- **LFM3 1.7B is incomplete.** Its provider answered 118 of the 144 questions it was sent and then returned HTTP 429 and 504 ("temporarily rate-limited upstream"); the script stops after two failed batches. Its row rests on 36, 34 and 48 items, and 13 of its replies ran out of tokens while thinking.
- **Reading.** A hosted model's answer to an item can change from one call to the next (most of these models accept no temperature); each item was asked once.

## 8. Stories graded by a hosted model

`scripts/judge_stories.py` implements the evaluation of Eldan and Li (2023, section 3): the small model is given
the beginning of a story, usually cut inside a sentence, and completes it; a hosted grader reads the beginning
and the completion separated by `***`, writes a short assessment, and grades grammar, creativity and consistency
with the beginning out of 10 (also the plot, and the age the writing suggests, from six groups). The paper
generates ten completions for each of its story beginnings at temperature 1 and averages the grades; the
beginnings (44 of them) are in the dataset's repository and `--write-prompts` writes them out. The summary gives
each model's mean grades with a standard error over prompts.

Tested on made-up completions of two of the paper's story beginnings (`--self-test`), with `openai/gpt-6.1-terra` as the grader and its hidden reasoning switched off (the written assessment is the reasoning, as in the paper):

| completion | grammar | creativity | consistency | plot |
|:--|--:|--:|--:|--:|
| coherent (2 beginnings) | 10.0 | 7.5 | 9.5 | 9.0 |
| broken grammar (2 beginnings) | 1.5 | 4.0 | 8.0 | 6.5 |
| fluent, about something else | 9.0 | 4.0 | 2.0 | 4.0 |
| one word repeated | 0.0 | 0.0 | 0.0 | 0.0 |

The test cost $0.005; a grading costs about $0.0009, so the paper's protocol (44 beginnings, ten completions each) is about $0.40 a model. No model of ours has been graded yet.

## 9. Commands

`$PY` is the lab's Python; run from the root of the repository.

```
# the fresh held-out set (once; scoring it again after a change means scoring every model again)
$PY scripts/make_fresh_corpus.py --posts '<site>/_posts/2026-1*.md'

# released models: everything missing from results/eval/ (about 40 GPU minutes for all ten)
CUDA_VISIBLE_DEVICES=2 HF_TOKEN=... $PY scripts/eval_released_models.py

# one of our checkpoints: bits per byte, then the benchmarks, into one file
CUDA_VISIBLE_DEVICES=2 $PY -m gptpp.evaluate   --model $GPTPP_WORK/ckpt/<run> --docs fineweb-edu,wikitext103,tinystories,fresh --out results/eval/<run>.json
CUDA_VISIBLE_DEVICES=2 $PY -m gptpp.benchmarks --model $GPTPP_WORK/ckpt/<run> --tasks all --out results/eval/<run>.json

# the tables (our models in bold), summary.json, and a paired comparison
$PY scripts/eval_released_models.py --table
$PY scripts/eval_released_models.py --compare <run> gpt2

# hosted models (strict cap in dollars) and their table
OPENROUTER_API_KEY=... $PY scripts/eval_openrouter.py --cap 5
$PY scripts/eval_openrouter.py --table

# stories
$PY scripts/judge_stories.py --write-prompts prompts.jsonl
OPENROUTER_API_KEY=... $PY scripts/judge_stories.py --completions <file>.jsonl --cap 2

# the checks of sections 1, 2 and 6
$PY scripts/eval_released_models.py --padding-check --device cpu
$PY scripts/eval_released_models.py --tinystories-overlap
CUDA_VISIBLE_DEVICES=2 $PY scripts/eval_released_models.py --wikitext-detokenized

# tests (24; on a GPU they score the whole WikiText-103 test set with GPT-2 small, about 20 seconds)
CUDA_VISIBLE_DEVICES=2 $PY -m pytest -q tests/test_evaluate.py
```

A checkpoint is a run directory (its `model.pt`) or a `ckpt_<step>.pt` file; the tokenizer is read from the
run's `config.json` (any key named `tokenizer`), else guessed from the vocabulary size, else given with
`--tokenizer`.

## 10. What is not here

- **Our own models.** No checkpoint existed when the sections above were written; the main model was scored afterwards under the same protocol (`results/eval/gptpp-base.json`), and the path is also tested on an untrained checkpoint in the trainer's format. Section 9 has the commands. The run `gptpp-base-1B` was never scored under the protocol.
- **Gemma 3 270M**: gated (section 4). **LFM3 1.7B**: incomplete (section 7).
- **The Children's Book Test**, the other benchmark of the GPT-2 paper's table 3, is not implemented: ten candidates after twenty sentences of context are about twelve million tokens a model.
- **Few-shot prompts.** Every benchmark is zero-shot.
- **Repeats.** Log-likelihood scoring has no seed: a second run gives the same numbers up to float32 rounding. The hosted models were asked once.
- **The published WikiText-103 figure of GPT-2 XL** is not reproduced (section 6).
