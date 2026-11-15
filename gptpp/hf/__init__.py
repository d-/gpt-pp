"""gpt-pp's models in the format of Hugging Face transformers.

`configuration_gptpp.py` and `modeling_gptpp.py` are the code that an exported model carries with it (see
scripts/export_hf.py): they are copied next to the weights and loaded with `trust_remote_code=True`. They import
nothing from gpt-pp, so that the exported folder stands alone.
"""
