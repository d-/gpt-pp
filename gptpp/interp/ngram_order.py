"""Statistics of increasing order: checkpoints of a language model measured against n-gram predictors.

A network trained by gradient descent fits the low-order statistics of its data first. For a language model the
moments of the (one-hot) data are n-gram frequencies, so "low order first" reads: the frequencies of single
tokens, then of pairs, then longer contexts (Belrose et al., 2024, arXiv:2402.04362; the in-context version of
the same staircase is Edelman et al., 2024, arXiv:2402.11004). The test is to hold each checkpoint against the
two predictors that use nothing but those statistics:

    the unigram predictor   p(next) = the share of that token in the training set, whatever came before
    the bigram predictor    p(next | previous token) = the share of that pair among the pairs that begin alike

and to measure, on held-out text, at every position,

    the loss                -log p_model(the token that came), the usual cross-entropy
    KL(unigram || model)    how many extra nats the model would pay to encode tokens drawn from the unigram
    KL(bigram  || model)    predictor (from the bigram predictor) at that position: zero when the model's
                            next-token distribution is that predictor's, large when the model has moved on

and, on text sampled from each predictor (independent tokens; a first-order Markov chain), the model's loss: it
bottoms out at the entropy of the predictor when the model has learned exactly that much and no more.

    tables = count_tables(TokenData("tinystories/bpe4k"))           # every training shard, about 20 s
    predictors = Predictors(*tables)
    row = measure(model, windows, predictors)                        # one checkpoint, all in nats per token

The divergence is taken from the n-gram predictor to the model, in that order, because the model's softmax gives
every token some probability while the tables are full of zeros: the other order is infinite without smoothing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from ..data import TokenData


def count_tables(data: TokenData, shards: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Counts of every token (V,) and of every pair of adjacent tokens (V, V) [previous, next] over the training
    shards. The stream is counted as the trainer reads it: documents back to back, each begun by `<|bos|>`."""
    vocab = data.vocab_size
    unigram = np.zeros(vocab, dtype=np.int64)
    bigram = np.zeros(vocab * vocab, dtype=np.int64)
    for i in range(len(data.train_files) if shards is None else shards):
        tokens = np.asarray(data.shard(i)).astype(np.int64)
        unigram += np.bincount(tokens, minlength=vocab)
        bigram += np.bincount(tokens[:-1] * vocab + tokens[1:], minlength=vocab * vocab)
    return unigram, bigram.reshape(vocab, vocab)


def entropy(p: Tensor) -> Tensor:
    """Entropy in nats along the last axis, with 0 log 0 = 0."""
    return -torch.xlogy(p, p).sum(dim=-1)


@dataclass
class Predictors:
    """The unigram and bigram predictors of a corpus, from its counts, by maximum likelihood.

    A previous token that never occurs in the corpus has no row of counts; its row is the unigram distribution.
    """

    unigram: Tensor  # (V,) probabilities
    bigram: Tensor   # (V, V) rows: p(next | previous)

    def __init__(self, unigram_counts: np.ndarray, bigram_counts: np.ndarray):
        counts = torch.as_tensor(unigram_counts, dtype=torch.float64)
        self.unigram = counts / counts.sum()
        pairs = torch.as_tensor(bigram_counts, dtype=torch.float64)
        rows = pairs.sum(dim=1, keepdim=True)
        self.bigram = torch.where(rows > 0, pairs / rows.clamp(min=1), self.unigram.expand_as(pairs))
        self.row_share = (rows.squeeze(1) / rows.sum())  # how often each token is the previous one

    def to(self, device: str | torch.device) -> "Predictors":
        self.unigram, self.bigram, self.row_share = (t.to(device) for t in (self.unigram, self.bigram, self.row_share))
        return self

    def unigram_entropy(self) -> float:
        """Nats per token of independent draws from the unigram distribution."""
        return entropy(self.unigram).item()

    def bigram_entropy(self) -> float:
        """Nats per token of the Markov chain: the entropy of each row, weighted by how often the row is used."""
        return (self.row_share * entropy(self.bigram)).sum().item()


@torch.no_grad()
def log_probs(model, idx: Tensor) -> Tensor:
    """The model's next-token log-probabilities at every position of idx (B, T): (B, T, V), float32."""
    return F.log_softmax(model(idx).float(), dim=-1)


@torch.no_grad()
def measure(model, windows: Tensor, predictors: Predictors, batch: int = 32) -> dict[str, float]:
    """One checkpoint on held-out windows (N, T + 1): its loss and its divergence from each predictor, in nats per
    token, averaged over all N * T positions.

    At a position whose last token is x the bigram predictor is row x of the table; the divergence from a
    predictor q to the model m is sum_v q(v) (log q(v) - log m(v)) = -H(q) - sum_v q(v) log m(v).
    """
    uni, bi = predictors.unigram.float(), predictors.bigram.float()
    minus_h_uni = -entropy(predictors.unigram)
    minus_h_row = -entropy(predictors.bigram)          # (V,), float64
    total = {"loss": 0.0, "kl_unigram": 0.0, "kl_bigram": 0.0}
    for start in range(0, windows.size(0), batch):
        chunk = windows[start:start + batch]
        idx, targets = chunk[:, :-1], chunk[:, 1:]
        logp = log_probs(model, idx)
        total["loss"] += -logp.gather(-1, targets.unsqueeze(-1)).double().sum().item()
        total["kl_unigram"] += (minus_h_uni - (logp @ uni).double()).sum().item()
        total["kl_bigram"] += (minus_h_row[idx] - (bi[idx] * logp).sum(dim=-1).double()).sum().item()
    positions = windows.size(0) * (windows.size(1) - 1)
    return {name: value / positions for name, value in total.items()}


@torch.no_grad()
def loss_on(model, sequences: Tensor, batch: int = 64) -> float:
    """The model's mean loss in nats per token on sequences (N, T + 1), every position a target once."""
    total = 0.0
    for start in range(0, sequences.size(0), batch):
        chunk = sequences[start:start + batch]
        logp = log_probs(model, chunk[:, :-1])
        total += -logp.gather(-1, chunk[:, 1:].unsqueeze(-1)).double().sum().item()
    return total / (sequences.size(0) * (sequences.size(1) - 1))


def sample_unigram(predictors: Predictors, n: int, length: int, seed: int) -> Tensor:
    """n sequences of `length` tokens drawn independently from the unigram distribution: text with the corpus's
    token frequencies and nothing else."""
    generator = torch.Generator(device=predictors.unigram.device).manual_seed(seed)
    return torch.multinomial(predictors.unigram.float(), n * length, replacement=True, generator=generator).view(n, length)


def sample_bigram(predictors: Predictors, n: int, length: int, first: int, seed: int) -> Tensor:
    """n sequences of `length` tokens from the first-order Markov chain of the bigram table, each begun by the
    token `first` (the beginning-of-text token): text with the corpus's pair frequencies and nothing else."""
    generator = torch.Generator(device=predictors.bigram.device).manual_seed(seed)
    table = predictors.bigram.float()
    out = torch.empty(n, length, dtype=torch.long, device=table.device)
    out[:, 0] = first
    for t in range(1, length):
        out[:, t] = torch.multinomial(table[out[:, t - 1]], 1, generator=generator).squeeze(1)
    return out


def continue_with(predictors: Predictors, order: int, last: int, n: int, seed: int) -> list[int]:
    """What a predictor writes after a text that ends in the token `last`: n tokens, drawn independently from the
    unigram distribution (order 1) or along the chain of the bigram table from `last` (order 2)."""
    return (sample_unigram(predictors, 1, n, seed)[0] if order == 1 else sample_bigram(predictors, 1, n + 1, last, seed)[0, 1:]).tolist()


def held_out_windows(data: TokenData, n: int, length: int) -> Tensor:
    """The first n windows of `length` tokens of the held-out stream, one after another: (n, length) int64."""
    tokens = np.asarray(data.val_tokens()[: n * length]).astype(np.int64)
    return torch.from_numpy(tokens).view(n, length)


def nats_to_bits(x: float) -> float:
    return x / math.log(2)
