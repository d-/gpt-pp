# Data and tokenizers

What was downloaded, what was made from it, how to make it again, and what was measured on it.
Every number below is copied from a file that a script wrote (`meta.json` and `stats.json` in each
dataset directory, the latter also committed under `results/data/`; `meta.json` and
`reference_check.json` beside each tokenizer), except where a sentence says it comes from a one-off
measurement. Downloaded and prepared on 14 November 2026.

## What is on disk

`$GPTPP_WORK` is `/home/ubuntu/d-work/gpt` unless the environment says otherwise.

| what | where | size |
|:--|:--|:--|
| tokenizer `bpe4k` | `$GPTPP_WORK/data/tokenizers/bpe4k/` | 4,096 tokens: 256 bytes, 3,835 merges, 5 special |
| tokenizer `bpe32k` | `$GPTPP_WORK/data/tokenizers/bpe32k/` | 32,768 tokens: 256 bytes, 32,507 merges, 5 special |
| TinyStories, tokenized | `$GPTPP_WORK/data/tinystories/bpe4k/` | 542,905,221 training tokens in 6 shards; 5,481,226 held-out tokens; 1.1 GB |
| FineWeb-EDU, tokenized | `$GPTPP_WORK/data/fineweb-edu/bpe32k/` | 9,896,904,130 training tokens in 99 shards; 10,000,019 held-out tokens; 19.8 GB |
| held-out text | `$GPTPP_WORK/data/<dataset>/val_docs.jsonl` | 27,630 stories; 9,649 documents |
| downloads | `$GPTPP_WORK/data/raw/<dataset>/` | 2.25 GB; 28.5 GB |

A dataset directory holds `train_0000.bin`, `train_0001.bin`, ... (unsigned 16-bit little-endian
token ids, whole documents, a shard is closed at the first document boundary past 100 million
tokens), `val.bin` (the held-out documents), `meta.json` (the fields of Contract 1 and a few more)
and `stats.json`. Every document begins with `<|bos|>`. `val_docs.jsonl` has one `{"text": ...}` a
line, in the order of `val.bin` (for FineWeb-EDU also the document's `id` and `url`).

## Sources and licenses

### TinyStories

- Repository: <https://huggingface.co/datasets/roneneldan/TinyStories>, revision
  `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64` (last modified 12 August 2024).
- Files: `TinyStoriesV2-GPT4-train.txt` (2,227,753,162 bytes, SHA-256 `6418d412de72888f52b5142c761ac21a582f7d1166f0bfbdb5f03ccfdec90443`)
  and `TinyStoriesV2-GPT4-valid.txt` (22,502,601 bytes, SHA-256 `6874bae9a4c1a4e7edcf0e53b86c17817e9cf881fc75ff2368da457b80c0585d`).
  Both hashes equal the ones the Hub lists (compared once, by hand). This is the second version of
  the dataset, written by GPT-4 alone; the stories are separated by `<|endoftext|>`.
- License: CDLA-Sharing-1.0 (the dataset card declares `license: cdla-sharing-1.0`).
- Paper: Eldan and Li, "TinyStories: how small can language models be and still speak coherent
  English?", 2023, arXiv:2305.07759.

### FineWeb-EDU

- Repository: <https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu>, revision
  `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` (last modified 11 July 2025).
- Files: the subset `sample/10BT`, 14 Parquet files `sample/10BT/000_00000.parquet` to
  `sample/10BT/013_00000.parquet`, 28,518,193,415 bytes together, 9,672,101 documents. The sizes
  and SHA-256 hashes of all 14 equal the ones the Hub lists at that revision (compared once, by hand).
- License: ODC-By 1.0. The dataset card (its `README.md` at the revision above) declares
  `license: odc-by` and says: "The dataset is released under the **Open Data Commons Attribution
  License (ODC-By) v1.0** [license](https://opendatacommons.org/licenses/by/1-0/). The use of this
  dataset is also subject to [CommonCrawl's Terms of Use](https://commoncrawl.org/terms-of-use)."
- Paper: Penedo, Kydlíček, Ben allal, Lozhkov, Mitchell, Raffel, Von Werra and Wolf, "The FineWeb
  datasets: decanting the web for the finest text data at scale", 2024, arXiv:2406.17557.
- Why this subset: it is the publisher's own random sample, it is the size we need (about ten
  billion tokens against the six billion asked for) and it is 14 files. The download took 4 minutes
  10 seconds with the files fetched in parallel. The alternative, the pre-shuffled repository
  `karpathy/fineweb-edu-100b-shuffle` (1,823 shards of about 94 MB), was not timed.

## The tokenizers

Byte-level byte-pair encoding. `gptpp/bpe.py` is the algorithm in plain Python; `gptpp/tokenizer.py`
is the same tokenizer run by the Hugging Face `tokenizers` library (version 0.23.3).

- **Pre-tokenization.** The text is cut by the pattern of GPT-4 (`cl100k_base` in tiktoken) with
  its possessive quantifiers written as greedy ones and its alternative `\s++$` left out, two
  constructs that Oniguruma, the engine inside `tokenizers`, reads differently from Python's
  `regex`. The first change never alters the cut. The second alters it only inside the last run of
  whitespace of a text, and only if that run holds a line break followed by more whitespace:
  compared with the original pattern, our pattern cuts all 37,279 held-out documents of the two
  datasets identically (a one-off comparison, not part of the scripts). No normalization. The exact
  pattern is in each tokenizer's `meta.json`:

  ```
  '(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s
  ```

- **Token ids.** 0..255 are the bytes; 256 and up are the merges in the order learned; the last
  five ids are `<|bos|>`, `<|user_start|>`, `<|user_end|>`, `<|assistant_start|>`,
  `<|assistant_end|>`. So `<|bos|>` is 4091 in `bpe4k` and 32763 in `bpe32k`.
- **Special tokens are never read from text.** They have ids in the vocabulary of `tokenizer.json`
  but are not registered as "added tokens", so the seven characters `<|bos|>` in a web page are
  encoded as text. A special id enters a sequence only through `encode(..., bos=True)` or when the
  caller writes it.
- **Ties.** Among equally frequent pairs the one with the smaller ids is merged first.
- **Files.** `tokenizer.json` (loadable by `tokenizers` and `transformers` with none of our code),
  `merges.txt` (line k: the two ids merged into token 256 + k; this alone defines the tokenizer),
  `meta.json`, `reference_check.json`.

| | `bpe4k` | `bpe32k` |
|:--|:--|:--|
| trained on | TinyStories V2, the whole training split | FineWeb-EDU `sample/10BT`: the last 423,491 documents of the shuffled order, all training documents |
| characters | 2,184,987,346 | 2,000,001,075 |
| training time | 25 s | 52 s |
| SHA-256 of the training text | `6418d412...ec90443` (the training file) | `5d967add...71bc272` (`tokenizer_text.parquet`, as written by pyarrow 26.0.1) |

The first 30 merges of `bpe4k`: ` t`, `he`, ` a`, ` s`, ` w`, `nd`, ` the`, `ed`, ` b`, ` to`,
` and`, ` h`, ` f`, `in`, ` T`, ` wa`, `re`, `it`, `ou`, ` l`, ` d`, ` c`, ` p`, `ay`, ` m`, `er`,
` was`, `om`, ` The`, ` he`.

The first 30 merges of `bpe32k`: ` t`, ` a`, `he`, `in`, `re`, ` the`, `on`, `er`, ` o`, `at`,
` s`, `en`, ` c`, `es`, ` w`, `is`, `nd`, `it`, `or`, ` p`, `al`, ` b`, `ed`, ` of`, `an`, ` f`,
`ing`, ` in`, `ou`, `ar`.

### The reference and the production tokenizer are the same tokenizer

Three things were checked.

1. **Same merges on a small corpus** (`tests/test_tokenizer.py`, `tests/test_bpe.py`): the plain
   loop `bpe.train`, the incremental `bpe.train_fast` and the library's trainer learn identical
   merges at several vocabulary sizes, ties included.
2. **Same merges on the real training text** (`scripts/train_tokenizer.py --verify`, results in
   `reference_check.json`): `bpe.train_fast`, pure Python on one core, learns all 3,835 merges of
   `bpe4k` from the 2.18 billion characters of TinyStories in 157 s, and all 32,507 merges of
   `bpe32k` from its two billion characters in 368 s. Both lists are identical to the production
   lists.
3. **Same ids** (`scripts/corpus_stats.py`, field `checks` of `stats.json`): `bpe.BPE` encodes each
   of the 27,630 held-out stories and each of the 9,649 held-out FineWeb-EDU documents to exactly
   the tokens of `val.bin`.

How the agreement is arranged: the text is cut into pieces by Python's `regex` module for both
trainers, and each piece is handed to the library's trainer as its UTF-8 bytes written as the
characters U+0000..U+00FF. The trainer sorts its initial alphabet by code point, so its ids are
ours, and its rule for ties (the smaller pair of ids) is the rule of `bpe.py`.

Where the two can differ, and why. Encoding (not training) uses two regular expression engines:
`bpe.py` asks Python's `regex` module whether a character is a letter, a digit or whitespace, the
library asks Oniguruma, and each carries its own copy of the Unicode tables. With `regex`
2026.11.9 and `tokenizers` 0.23.3 they agree on the 25 whitespace characters and on every letter
and digit except 17,480 code points that `regex` already classifies as letters (17,144) or digits
(336) and Oniguruma does not: evidently characters added to Unicode after the version Oniguruma
carries (all of them are unassigned in Python 3.13's own tables, Unicode 15.1). On a text
containing one of those characters the two encoders cut differently around it, so the ids differ
there; both still decode to the original text. Apart from those characters the cuts are identical:
a test puts every code point of planes 0 to 3 and 14 in six contexts and compares the two engines.
None of the 17,480 characters occurs anywhere in the 9,672,101 documents of the FineWeb-EDU sample
or in the two TinyStories files (a one-off scan, not part of the scripts), so on this data the two
encoders agree on every document.

### A port to JavaScript

A page that runs a model needs the same cut. The pattern as written is refused by JavaScript
engines without regular expression modifiers (Node 22 rejects `(?i:...)`), and JavaScript's `\s` is
another set of characters (it has U+FEFF and lacks U+0085). This spelling, with the `u` flag, cuts
like the Python pattern:

```js
const WS = String.raw`\t\n\v\f\r \x85\xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000`;
const PATTERN = String.raw`'(?:[sdmtSDMT\u017f]|[lL][lL]|[vV][eE]|[rR][eE])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^${WS}\p{L}\p{N}]+[\r\n]*|[${WS}]*[\r\n]|[${WS}]+(?![^${WS}])|[${WS}]`;
const pieces = text.match(new RegExp(PATTERN, "gu")) ?? [];
```

(U+017F, the long s, is there because the case-insensitive `[sdmt]` of the two other engines
matches it.) Checked once in Node 22.23.4 (Unicode 17.0) against the cuts of Python's `regex`, not
part of the scripts: identical on 3,000 held-out documents of each dataset, on the strings of the
tests and on 50,000 random strings of whitespace, apostrophes and contraction letters; different
only around 12,823 code points that the two Unicode tables classify differently, none of which
occurs in the data.

## How the datasets are built

**TinyStories** (`scripts/prepare_tinystories.py`). Each file is split at `<|endoftext|>` and each
story stripped of surrounding whitespace. Training stories whose text also occurs in the
validation file would be dropped (there are none). Each split is put in an order drawn from
`numpy.random.default_rng(0)`; the validation split is the held-out set.

**FineWeb-EDU** (`scripts/prepare_fineweb.py`). One permutation of the 9,672,101 documents, from
`numpy.random.default_rng(0)`, fixes everything:

- the held-out set is the shortest prefix of the permutation with at least ten million tokens: 9,649
  documents, 10,000,019 tokens;
- the training set is the rest, in the order of the permutation, minus the 960 documents that have
  the text (815) or the URL (702) of a held-out document; 9,661,492 documents remain;
- the tokenizer is trained on the last 423,491 documents of the permutation (two billion characters),
  which are all training documents.

The dataset was deduplicated one crawl at a time, so a page fetched in several crawls is in it
several times: 403,945 documents of the sample repeat the text of another and 351,489 repeat a URL.
These repeats are left in the training set, except those of held-out documents. After that, no
training document has exactly the tokens of a held-out document (checked over every shard). Near
duplicates under different URLs cannot be excluded.

## Measured statistics

From `stats.json`. Token counts include one `<|bos|>` a document; bytes are the UTF-8 bytes of the
text.

| | TinyStories, `bpe4k` | FineWeb-EDU, `bpe32k` |
|:--|--:|--:|
| training tokens | 542,905,221 | 9,896,904,130 |
| training documents | 2,717,495 | 9,661,492 |
| training bytes | 2,185,895,233 | 45,923,179,704 |
| held-out tokens | 5,481,226 | 10,000,019 |
| held-out documents | 27,630 | 9,649 |
| held-out bytes | 22,077,118 | 46,283,256 |
| bytes per token, held-out (the trainer's `bytes_per_token_val`) | 4.0278 | 4.6283 |
| bytes per token, held-out, `<|bos|>` not counted | 4.0482 | 4.6328 |
| bytes per token, training | 4.0263 | 4.6402 |
| unigram entropy, bits per token | 8.569 | 11.110 |
| unigram entropy, bits per byte | 2.128 | 2.394 |
| tokens of the vocabulary never used in training | 86 | 50 |
| Zipf exponent (least squares over ranks 10 to 1,000) | 1.169 | 0.883 |
| document length in tokens: mean, median, 99th percentile, longest | 198.8, 177, 572, 1,860 | 1,023.4, 620, 7,875, 172,051 |
| documents longer than 1,024 tokens (share of all tokens in them) | 0.02% (0.1%) | 26.8% (64.8%) |

For comparison, the `token_count` column that FineWeb-EDU ships says the same 9,672,101 documents
are 9,972,918,177 GPT-2 tokens (vocabulary 50,257); `bpe32k` writes them in 9,898,298,520 tokens
(9,907,970,621 with the `<|bos|>` tokens, from the log of the run), 0.75% fewer with a vocabulary
35% smaller.

**n-gram models.** Counted on the first 200,000,000 tokens of the training stream, scored on the
held-out stream (every token from the fourth on). Interpolated Kneser-Ney smoothing, one discount
per level, D = n1 / (n1 + 2 n2), the uniform distribution below the unigrams, nothing tuned on the
held-out set. "Sample" is the entropy of the n-th token given the n - 1 before it under the raw
counts of the sample itself; it falls towards zero with n because most long n-grams occur once.

TinyStories, `bpe4k`:

| n | distinct n-grams | seen once | held-out n-grams not in sample | bits per token, sample | bits per token, held-out | perplexity | bits per byte |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 3,996 | 19 | 0.00% | 8.568 | 8.565 | 378.6 | 2.126 |
| 2 | 1,127,841 | 368,819 | 0.22% | 5.149 | 5.164 | 35.8 | 1.282 |
| 3 | 9,832,757 | 5,150,883 | 2.72% | 3.885 | 4.062 | 16.7 | 1.008 |
| 4 | 32,591,218 | 21,614,465 | 11.06% | 2.977 | 3.612 | 12.2 | 0.897 |

FineWeb-EDU, `bpe32k`:

| n | distinct n-grams | seen once | held-out n-grams not in sample | bits per token, sample | bits per token, held-out | perplexity | bits per byte |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 32,707 | 12 | 0.00% | 11.111 | 11.110 | 2,210.4 | 2.400 |
| 2 | 15,097,774 | 8,407,269 | 4.98% | 7.784 | 8.166 | 287.2 | 1.764 |
| 3 | 76,558,188 | 58,045,771 | 31.78% | 5.078 | 7.436 | 173.2 | 1.607 |
| 4 | 141,394,600 | 124,542,435 | 65.77% | 2.337 | 7.299 | 157.5 | 1.577 |

**General-purpose compressors** on the held-out text (the documents joined by line breaks and
compressed as one stream; bits per byte = 8 x compressed bytes / held-out bytes):

| compressor | TinyStories | FineWeb-EDU |
|:--|--:|--:|
| gzip -9 (zlib 1.3.1) | 2.401 | 3.020 |
| bzip2 -9 | 1.609 | 2.395 |
| zstd -19 (zstandard 0.24.0) | 1.545 | 2.267 |
| zstd -22 | 1.519 | 2.204 |
| xz -9e (LZMA2) | 1.510 | 2.176 |

`stats.json` also has the count of every token by rank with its id (the table of a Zipf plot), the
200 most frequent tokens with their text, histograms of document lengths (bins growing by a factor
2^(1/4)) and of token lengths in bytes, and the SHA-256 of `val.bin` and of every shard.

## Reproduce

```
PY=/home/ubuntu/d-work/envs/gpt/bin/python
cd gpt-pp
export HF_TOKEN=...                                   # only needed if the Hub asks for it

$PY scripts/prepare_tinystories.py --download-only
$PY scripts/train_tokenizer.py --name bpe4k --vocab-size 4096 \
    --input /home/ubuntu/d-work/gpt/data/raw/tinystories/TinyStoriesV2-GPT4-train.txt \
    --separator "<|endoftext|>" --trained-on "TinyStories V2 (GPT-4), the whole training split"
$PY scripts/prepare_tinystories.py --tokenizer bpe4k

$PY scripts/prepare_fineweb.py --tokenizer bpe32k     # downloads, trains bpe32k, writes the shards

$PY scripts/corpus_stats.py                            # stats.json for both
$PY scripts/train_tokenizer.py --name bpe4k --vocab-size 4096 --verify --separator "<|endoftext|>" \
    --input /home/ubuntu/d-work/gpt/data/raw/tinystories/TinyStoriesV2-GPT4-train.txt
$PY scripts/train_tokenizer.py --name bpe32k --vocab-size 32768 --verify \
    --input /home/ubuntu/d-work/gpt/data/fineweb-edu/tokenizer_text.parquet
$PY -m pytest -q tests/test_bpe.py tests/test_tokenizer.py
```

Time on this machine (96 cores, 360 GB of memory): the downloads 2 minutes 11 seconds and 4 minutes
10 seconds; `bpe4k` 27 s; the TinyStories shards 28 s; the whole FineWeb-EDU pipeline a little over
four minutes (scan 12 s, tokenizer text 14 s, `bpe32k` 52 s, tokenizing 9.9 billion tokens on 92
processes 150 s, assembling and writing 25 s); the statistics 61 s and 166 s.

The seed is 0 everywhere. The shards are a function of the downloaded files, the seed and the
merges, and the merges are a function of the training text, so a second run writes the same bytes.
This was tested once: a second run of the commands above from the downloaded files, in a separate
workspace and with both tokenizers trained again, wrote `merges.txt`, `tokenizer.json`,
`val_docs.jsonl`, `val.bin` and all 105 shards with the SHA-256 of the first run. `stats.json`
lists the SHA-256 of every file to compare against (`val.bin`: `f0ca459b...4966fcd` for
TinyStories, `9983fdee...518dc84` for FineWeb-EDU).

## Things to know

- **FineWeb-EDU has no `{`.** The byte `{` does not occur once in the 9.9 billion training tokens,
  `}` occurs 7,395 times and the tab character 42 times. A model trained on it has never seen an
  opening brace, a JSON object or indented code. (Why the dataset is so is not verified here.)
- Runs of spaces are rare in FineWeb-EDU too, and `bpe32k` has no token for two or more spaces:
  four spaces of indentation are four tokens.
- The single space is the ninth most frequent token of FineWeb-EDU (1.2%): the pattern gives a
  number no leading space, so ` 2026` is ` `, `202`, `6`.
- Other files in the dataset directory of FineWeb-EDU: `docs.npz` (for each of the 9,672,101
  documents, in the order of the Parquet files: characters, bytes, a 64-bit hash of the text and of
  the URL, GPT-2 tokens) and `tokenizer_text.parquet` (the text `bpe32k` was trained on).
