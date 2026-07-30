"""Loaders for the three model families evaluated in the CWS paper.

TinyLlama-1.1B and LLaMA-7B are both `LlamaForCausalLM`, so they share a
loader. HGRN-1.3B is a gated-recurrent SSM with no attention, served by the
`flash-linear-attention` (fla) library, which registers its architectures
with `transformers`'s `AutoModelForCausalLM` on import.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_REGISTRY = {
    "tinyllama-1.1b": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    "llama-7b": "openlm-research/open_llama_7b",
    "hgrn-1.3b": "fla-hub/hgrn-1.3B-100B",
}


def load_model_and_tokenizer(name: str, dtype=torch.bfloat16, device="cpu"):
    """Load one of `MODEL_REGISTRY`'s models by short name.

    `name` may also be an arbitrary HF hub id or local path, in which case
    it is used directly (bypassing the registry) so custom checkpoints work
    without editing this file.
    """
    hub_id = MODEL_REGISTRY.get(name.lower(), name)

    if name.lower() == "hgrn-1.3b" or "hgrn" in name.lower():
        try:
            import fla  # noqa: F401  (registers HGRN with AutoModelForCausalLM)
        except ImportError as e:
            raise ImportError(
                "HGRN-1.3B requires the `flash-linear-attention` package "
                "(`pip install flash-linear-attention`) to register its "
                "architecture with transformers."
            ) from e

    tokenizer = AutoTokenizer.from_pretrained(hub_id)
    model = AutoModelForCausalLM.from_pretrained(
        hub_id, torch_dtype=dtype, trust_remote_code=True
    )
    model = model.to(device)
    model.eval()
    model.config.use_cache = False
    return model, tokenizer
