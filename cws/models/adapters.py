"""Architecture-agnostic access to a causal LM's sequential block list.

Both LLaMA-family transformers and FLA's HGRN (a gated-recurrent SSM with no
attention) expose their per-layer blocks as an `nn.ModuleList` reachable from
the top-level `PreTrainedModel`. Rather than hardcoding attention-specific
forward signatures, this module finds that list generically and captures
whatever positional/keyword arguments the model itself passes to block 0 (the
same "Catcher" trick used by the original SparseGPT/GPTQ codebases), so the
same driver runs unmodified over transformer and SSM architectures alike.
"""

import torch
import torch.nn as nn

_CANDIDATE_BLOCK_PATHS = [
    "model.layers",  # LLaMA / TinyLlama / most HF decoder-only models
    "model.model.layers",  # some FLA model wrappers
    "backbone.layers",  # Mamba-style HF wrappers
    "transformer.h",  # GPT-2 style
]


def _resolve_path(model: nn.Module, path: str):
    obj = model
    for attr in path.split("."):
        if not hasattr(obj, attr):
            return None
        obj = getattr(obj, attr)
    return obj


def find_blocks(model: nn.Module) -> nn.ModuleList:
    for path in _CANDIDATE_BLOCK_PATHS:
        obj = _resolve_path(model, path)
        if isinstance(obj, (nn.ModuleList, list)) and len(obj) > 1:
            return obj

    candidates = []
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > 1:
            candidates.append((name.count("."), name, module))
    if not candidates:
        raise ValueError(
            "Could not locate the model's sequential block list (tried "
            f"{_CANDIDATE_BLOCK_PATHS} and a generic nn.ModuleList search)."
        )
    candidates.sort(key=lambda x: x[0])
    return candidates[0][2]


def find_prunable_linears(block: nn.Module) -> dict[str, nn.Linear]:
    """All nn.Linear submodules inside one block, keyed by dotted name.

    This intentionally includes every Linear in the block (attention/QKVO
    projections, gated-recurrent input/output/gate projections, and MLP
    projections alike) so the same call covers both architectures without
    per-model name lists that could drift out of date.
    """
    linears = {}
    for name, module in block.named_modules():
        if isinstance(module, nn.Linear):
            linears[name] = module
    return linears


class InputCatcher(nn.Module):
    """Wraps block 0 to record the exact (args, kwargs) the model passes it,
    then aborts the forward pass -- avoids running the remaining blocks just
    to capture calibration inputs for block 0."""

    class _StopForward(Exception):
        pass

    def __init__(self, block: nn.Module, capture_list: list):
        super().__init__()
        self.block = block
        self.capture_list = capture_list

    def forward(self, *args, **kwargs):
        self.capture_list.append((args, kwargs))
        raise InputCatcher._StopForward()


@torch.no_grad()
def capture_block0_inputs(model: nn.Module, blocks: nn.ModuleList, calib_batches, device):
    """Run each calibration batch through the model up to block 0, capturing
    the (args, kwargs) tuple block 0 was called with. Returns a list of these
    tuples, one per calibration batch, plus the first positional/`hidden_states`
    tensor extracted out for convenience.
    """
    captured: list = []
    original_block0 = blocks[0]
    blocks[0] = InputCatcher(original_block0, captured)
    try:
        for batch in calib_batches:
            input_ids = batch.to(device)
            try:
                model(input_ids)
            except InputCatcher._StopForward:
                pass
    finally:
        blocks[0] = original_block0
    return captured


def hidden_states_of(args, kwargs):
    if len(args) > 0:
        return args[0]
    for key in ("hidden_states", "input"):
        if key in kwargs:
            return kwargs[key]
    raise ValueError("Could not find hidden_states in captured block call.")


def with_hidden_states(args, kwargs, new_hidden_states):
    if len(args) > 0:
        return (new_hidden_states,) + args[1:], kwargs
    for key in ("hidden_states", "input"):
        if key in kwargs:
            new_kwargs = dict(kwargs)
            new_kwargs[key] = new_hidden_states
            return args, new_kwargs
    raise ValueError("Could not find hidden_states in captured block call.")
