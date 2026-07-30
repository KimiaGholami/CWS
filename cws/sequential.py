"""Sequential (layer-by-layer) pruning driver.

Implements the SparseGPT/CWS recipe: process one block at a time, in
forward order. For each block: calibrate on the *already-pruned* output of
every prior block, prune and correct every Linear inside the block, then
re-run the block on the calibration data so the next block calibrates
against the real (sparsified) activations rather than the dense ones. This
means the Hessian error correction compounds correctly across the whole
network, block by block, instead of every layer being pruned independently
against dense-model statistics.
"""

import torch
import torch.nn as nn

from .baselines import magnitude_prune_layer, sparsegpt_prune_layer, wanda_prune_layer
from .cws_obs import cws_prune_layer
from .hessian import HessianAccumulator, build_hinv
from .models.adapters import (
    find_blocks,
    find_prunable_linears,
    hidden_states_of,
    with_hidden_states,
)

METHODS = ("cws", "sparsegpt", "wanda", "magnitude")


def default_compute_dtype(device: torch.device) -> torch.dtype:
    # MPS has no float64 kernel support; fall back to float32 there.
    return torch.float32 if torch.device(device).type == "mps" else torch.float64


def _extract_hidden(output):
    if isinstance(output, tuple):
        return output[0]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if hasattr(output, "hidden_states") and output.hidden_states is not None:
        return output.hidden_states
    return output


@torch.no_grad()
def prune_model_sequential(
    model: nn.Module,
    calib_inputs: list,
    method: str = "cws",
    sparsity: float = 0.5,
    blocksize: int | None = 128,
    percdamp: float = 0.01,
    device: str = "cpu",
    verbose: bool = True,
) -> dict:
    """Prune every Linear layer of `model` to `sparsity` using `method`.

    Args:
        model: a causal LM already loaded onto `device`.
        calib_inputs: output of `adapters.capture_block0_inputs` -- one
            (args, kwargs) tuple per calibration sample, as passed to
            block 0's forward.
        method: one of `METHODS`.
        sparsity: fraction of weights pruned per row.
        blocksize: CWS/SparseGPT column-block width (128 in the paper).
            `None` uses CWS's global variant (whole layer as one block);
            not supported for `sparsegpt`/`wanda`/`magnitude`.
        percdamp: Hessian damping fraction (paper uses 0.01).

    Returns:
        Per-block, per-layer achieved sparsity, for a quick sanity check.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}, expected one of {METHODS}")

    blocks = find_blocks(model)
    compute_dtype = default_compute_dtype(device)
    stats = {}

    hidden_states = [hidden_states_of(a, k) for a, k in calib_inputs]

    for block_idx, block in enumerate(blocks):
        block.to(device)
        linears = find_prunable_linears(block)
        accumulators = {name: HessianAccumulator(lin) for name, lin in linears.items()}
        for acc in accumulators.values():
            acc.register()

        for (args, kwargs), hs in zip(calib_inputs, hidden_states):
            call_args, call_kwargs = with_hidden_states(args, kwargs, hs)
            block(*call_args, **call_kwargs)

        for acc in accumulators.values():
            acc.remove()

        block_stats = {}
        for name, lin in linears.items():
            H = accumulators[name].H
            W = lin.weight.data

            if method == "cws":
                Hinv = build_hinv(H, percdamp, compute_dtype)
                W_new, mask = cws_prune_layer(W, Hinv, sparsity, blocksize)
            elif method == "sparsegpt":
                Hinv = build_hinv(H, percdamp, compute_dtype)
                W_new, mask = sparsegpt_prune_layer(
                    W, Hinv, sparsity, blocksize if blocksize else 128
                )
            elif method == "wanda":
                W_new, mask = wanda_prune_layer(W, torch.diagonal(H), sparsity)
            else:  # magnitude
                W_new, mask = magnitude_prune_layer(W, sparsity)

            lin.weight.data = W_new.to(lin.weight.dtype)
            block_stats[name] = 1.0 - mask.float().mean().item()

        stats[block_idx] = block_stats
        if verbose:
            avg = sum(block_stats.values()) / len(block_stats)
            print(f"[block {block_idx}] avg achieved sparsity {avg:.4f}")

        new_hidden_states = []
        for (args, kwargs), hs in zip(calib_inputs, hidden_states):
            call_args, call_kwargs = with_hidden_states(args, kwargs, hs)
            out = block(*call_args, **call_kwargs)
            new_hidden_states.append(_extract_hidden(out))
        hidden_states = new_hidden_states

    return stats
