"""Scratch wrapper of the review: the trainer with the fixed sinusoids added AFTER the embedding norm, at unit
amplitude on a unit-rms token vector, so that the position signal keeps a third of the energy however the token
table grows. (In the lab's arrangement the sinusoids are added before the norm, to a table whose rows grow.)"""
import sys
import gptpp.train as T
from gptpp.model import GPT
def embed_tokens(self, idx, t0=0):
    x = self.embed(idx)
    if self.embed_norm is not None:
        x = self.embed_norm(x)
    return x + self.pos_table[t0:t0 + idx.size(1)] / self.embed_std
GPT.embed_tokens = embed_tokens
if __name__ == "__main__":
    T.main(sys.argv[1:])
