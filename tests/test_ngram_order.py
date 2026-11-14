"""The n-gram predictors and the divergences of gptpp/interp/ngram_order.py, and the sequences and the loss by
position of gptpp/export/loss_by_position.py, on cases worked by hand."""
import math

import numpy as np
import torch

from gptpp.export import loss_by_position as lp
from gptpp.interp import ngram_order as ng


class Fixed(torch.nn.Module):
    """A 'model' whose next-token distribution depends on the last token only: logits = table[last token]."""

    def __init__(self, log_table: torch.Tensor):
        super().__init__()
        self.table = log_table

    def forward(self, idx):
        return self.table[idx]


def predictors():
    # the stream 0 1 0 1 0 2 over a vocabulary of 4: token 3 never occurs
    stream = np.array([0, 1, 0, 1, 0, 2])
    unigram = np.bincount(stream, minlength=4)
    bigram = np.zeros((4, 4), dtype=np.int64)
    for a, b in zip(stream[:-1], stream[1:]):
        bigram[a, b] += 1
    return ng.Predictors(unigram, bigram)


def test_tables_by_hand():
    p = predictors()
    assert torch.allclose(p.unigram, torch.tensor([3, 2, 1, 0], dtype=torch.float64) / 6)
    assert torch.allclose(p.bigram[0], torch.tensor([0, 2, 1, 0], dtype=torch.float64) / 3)     # after 0: 1 twice, 2 once
    assert torch.allclose(p.bigram[1], torch.tensor([1.0, 0, 0, 0], dtype=torch.float64))        # after 1: always 0
    assert torch.allclose(p.bigram[3], p.unigram)                                                # a row never seen falls back
    h = -(3 / 6 * math.log(3 / 6) + 2 / 6 * math.log(2 / 6) + 1 / 6 * math.log(1 / 6))
    assert abs(p.unigram_entropy() - h) < 1e-12
    after_zero = -(2 / 3 * math.log(2 / 3) + 1 / 3 * math.log(1 / 3))
    assert abs(p.bigram_entropy() - 3 / 5 * after_zero) < 1e-12                                  # 0 is the previous token 3 times in 5


def test_divergences_of_three_models():
    p = predictors()
    windows = torch.tensor([[0, 1, 0, 1, 0, 2], [1, 0, 2, 0, 1, 0]])
    eps = 1e-12
    as_bigram = Fixed(torch.log(p.bigram.float() + eps))
    as_unigram = Fixed(torch.log(p.unigram.float() + eps).expand(4, 4))
    uniform = Fixed(torch.zeros(4, 4))
    m = ng.measure(as_bigram, windows, p)
    assert abs(m["kl_bigram"]) < 1e-5 and m["kl_unigram"] > 0.1                 # the bigram predictor itself: no divergence from it
    m = ng.measure(as_unigram, windows, p)
    assert abs(m["kl_unigram"]) < 1e-5 and m["kl_bigram"] > 0.1
    m = ng.measure(uniform, windows, p)
    assert abs(m["loss"] - math.log(4)) < 1e-6
    assert abs(m["kl_unigram"] - (math.log(4) - p.unigram_entropy())) < 1e-6     # from q to uniform: log V - H(q)
    assert abs(ng.loss_on(uniform, windows) - math.log(4)) < 1e-6
    assert abs(ng.nats_to_bits(math.log(4)) - 2) < 1e-12


def test_samples_follow_the_tables():
    p = predictors()
    chain = ng.sample_bigram(p, 64, 50, first=1, seed=0)
    assert (chain[:, 0] == 1).all() and (chain[:, 1] == 0).all()                 # after 1 always 0
    assert not ((chain[:, :-1] == 1) & (chain[:, 1:] != 0)).any() and not (chain == 3).any()
    draws = ng.sample_unigram(p, 64, 50, seed=0)
    assert not (draws == 3).any() and abs((draws == 0).float().mean().item() - 0.5) < 0.05
    assert ng.continue_with(p, 2, last=1, n=5, seed=0)[0] == 0 and len(ng.continue_with(p, 1, last=1, n=7, seed=0)) == 7


def test_repeated_sequences_and_loss_by_position():
    pool = torch.tensor([10, 11, 12, 13])
    seqs = lp.repeated_sequences(pool, bos=99, half=6, n=5, seed=0)
    assert seqs.shape == (5, 13) and (seqs[:, 0] == 99).all()
    assert torch.equal(seqs[:, 1:7], seqs[:, 7:]) and set(seqs[:, 1:].flatten().tolist()) <= {10, 11, 12, 13}
    uniform = Fixed(torch.zeros(100, 100))
    by_position = lp.loss_by_position(uniform, seqs, batch=2)
    assert by_position.shape == (12,) and torch.allclose(by_position, torch.full((12,), math.log(100), dtype=torch.float64))
    copier = torch.full((100, 100), -30.0)                                        # a model that always predicts 11 after 10
    copier[10, 11] = 30.0
    seq = torch.tensor([[99, 10, 11, 10, 11]])
    loss = lp.loss_by_position(Fixed(copier), seq, batch=1)
    assert loss[1] < 1e-6 and loss[3] < 1e-6 and loss[0] > 1                      # position 2 and 4 (the 11 after a 10) cost nothing


def test_windows_of_text_stay_inside_one_document():
    class Data:                                           # three documents of 5, 12 and 20 tokens of text, each after a <|bos|> (id 9)
        meta = {"bos_id": 9}

        def val_tokens(self):
            return np.concatenate([[9], np.arange(100, 105), [9], np.arange(200, 212), [9], np.arange(300, 320)]).astype(np.uint16)

    windows = lp.document_windows(Data(), context=8, n=5, seed=0, margin=2)
    assert windows.shape == (2, 9)                        # only the documents with 8 + 2 tokens of text and their <|bos|> qualify
    for row, base in zip(windows.tolist(), (200, 300)):
        assert 9 not in row and row == list(range(row[0], row[0] + 9)) and base <= row[0] and row[-1] < base + (12 if base == 200 else 20)
