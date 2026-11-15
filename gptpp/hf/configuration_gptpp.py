"""The configuration of a gpt-pp model in Hugging Face transformers: NanoChat's, with the gain on the queries.

gpt-pp's 2026 recipe is the architecture that transformers calls NanoChat (rotary positions, RMS norms without
gains on the embedding, before each sublayer and before the unembedding, QK normalization, squared ReLU, untied
tables, logits capped by 15 tanh(z / 15)), with one addition: each normalized query is multiplied by `qk_gain`, so
that an attention score is `qk_gain * sqrt(head_dim)` times the cosine of query and key. Every gpt-pp model has a
score of 12 times the cosine, so `qk_gain = 12 / sqrt(head_dim)` (1.5 for a head of 64, 2.12 for a head of 32).
"""

from transformers.models.nanochat.configuration_nanochat import NanoChatConfig


class GptPPConfig(NanoChatConfig):
    model_type = "gptpp"
    qk_gain: float = 1.0


__all__ = ["GptPPConfig"]
