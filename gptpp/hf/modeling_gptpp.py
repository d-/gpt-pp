"""A gpt-pp model in Hugging Face transformers: NanoChat, with the queries multiplied by a gain after their norm.

The gain cannot be put into the weights, because it acts after a normalization; it lives in the module that
normalizes the queries. Everything else (attention kernels, the key-value cache, generation) is NanoChat's.
"""

from transformers.models.nanochat.modeling_nanochat import NanoChatForCausalLM, NanoChatModel, NanoChatRMSNorm

from .configuration_gptpp import GptPPConfig


class GptPPQueryNorm(NanoChatRMSNorm):
    """RMS normalization of each query head over its coordinates, then the gain."""

    def __init__(self, eps: float, gain: float):
        super().__init__(eps=eps)
        self.gain = gain

    def forward(self, x):
        return super().forward(x) * self.gain

    def extra_repr(self):
        return f"eps={self.eps}, gain={self.gain}"


def _add_query_gain(model: NanoChatModel, config: GptPPConfig) -> None:
    for layer in model.layers:
        layer.self_attn.q_norm = GptPPQueryNorm(config.rms_norm_eps, config.qk_gain)


class GptPPModel(NanoChatModel):
    config_class = GptPPConfig

    def __init__(self, config: GptPPConfig):
        super().__init__(config)
        _add_query_gain(self, config)


class GptPPForCausalLM(NanoChatForCausalLM):
    config_class = GptPPConfig

    def __init__(self, config: GptPPConfig):
        super().__init__(config)
        _add_query_gain(self.model, config)


__all__ = ["GptPPForCausalLM", "GptPPModel"]
