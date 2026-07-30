"""Baseline one-shot pruning methods used for comparison against CWS.

Implements:
  - magnitude: |W| thresholding (Han et al., 2015)
  - wanda: |W| * ||X||_2 per input channel, no weight correction (Sun et al., 2023)
  - sparsegpt: shared (non-per-row) Hinv sequence, blockwise OBS correction
    (Frantar & Alistarh, 2023) -- the classical algorithm CWS generalizes.

RIA (Zhang et al., 2024) and AWP (Liu et al., 2025) are cited in the CWS
paper as additional baselines but are not reimplemented here: RIA's
relative-importance re-weighting and AWP's iterative projected-gradient mask
search were not fully specified in the materials available when this
codebase was written, and a guessed reimplementation risks silently
misrepresenting those methods. Add them under this module if/when a
reference implementation is available to check against.
"""

import math

import torch


@torch.no_grad()
def magnitude_prune_layer(W: torch.Tensor, sparsity: float) -> tuple[torch.Tensor, torch.Tensor]:
    d_out, d_in = W.shape
    k_prune = int(math.floor(d_in * sparsity))
    W = W.clone()
    mask = torch.ones_like(W, dtype=torch.bool)
    if k_prune == 0:
        return W, mask
    score = W.abs()
    thresh = torch.kthvalue(score, k_prune, dim=1, keepdim=True).values
    mask = score > thresh
    W = W * mask
    return W, mask


@torch.no_grad()
def wanda_prune_layer(
    W: torch.Tensor, H_diag: torch.Tensor, sparsity: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Wanda: score_ij = |W_ij| * sqrt(diag(H)_j), no OBS correction.

    `H_diag` is the (undamped) diagonal of the accumulated H = (2/N) X^T X;
    since diag(H) = 2 * E[x_j^2], sqrt(diag(H)) is proportional to the RMS
    activation norm Wanda's score uses (the constant factor cancels in the
    per-row ranking).
    """
    d_out, d_in = W.shape
    k_prune = int(math.floor(d_in * sparsity))
    W = W.clone()
    mask = torch.ones_like(W, dtype=torch.bool)
    if k_prune == 0:
        return W, mask
    scale = H_diag.clamp_min(0).sqrt().unsqueeze(0)
    score = W.abs() * scale
    thresh = torch.kthvalue(score, k_prune, dim=1, keepdim=True).values
    mask = score > thresh
    W = W * mask
    return W, mask


@torch.no_grad()
def sparsegpt_prune_layer(
    W: torch.Tensor,
    Hinv: torch.Tensor,
    sparsity: float,
    blocksize: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Classical SparseGPT: one shared Hinv sequence for every output row.

    Unlike `cws_prune_layer`, the mask for a column-block is chosen once
    up front from a static snapshot of diag(Hinv) (Eq. in Frantar &
    Alistarh, 2023, Alg. 1), and the same shared Hinv row is used to
    correct every output row -- there is no per-row local inverse-Hessian
    copy. Per the CWS paper, this is exactly the S'=emptyset special case
    of the CWS criterion.
    """
    device = W.device
    dtype = W.dtype
    compute_dtype = Hinv.dtype
    d_out, d_in = W.shape
    W = W.clone().to(compute_dtype)
    mask = torch.ones((d_out, d_in), dtype=torch.bool, device=device)

    n_blocks = math.ceil(d_in / blocksize)
    for b in range(n_blocks):
        start = b * blocksize
        end = min(start + blocksize, d_in)
        B = end - start

        W_block = W[:, start:end].clone()
        Hinv_block = Hinv[start:end, start:end].to(compute_dtype)
        Err_block = torch.zeros_like(W_block)
        block_mask = torch.ones((d_out, B), dtype=torch.bool, device=device)

        if sparsity > 0:
            diag = torch.diagonal(Hinv_block)
            tmp = W_block.pow(2) / diag.clamp_min(1e-10).pow(2).unsqueeze(0)
            k_prune = int(math.floor(B * sparsity))
            if k_prune > 0:
                thresh = torch.kthvalue(tmp, k_prune, dim=1, keepdim=True).values
                block_mask = tmp > thresh

        for i in range(B):
            w = W_block[:, i].clone()
            d = Hinv_block[i, i]
            w = torch.where(block_mask[:, i], w, torch.zeros_like(w))
            err = w / d
            if i < B - 1:
                W_block[:, i + 1 :] -= err.unsqueeze(1) * Hinv_block[i, i + 1 :].unsqueeze(0)
            Err_block[:, i] = err
            W_block[:, i] = w

        W[:, start:end] = W_block
        mask[:, start:end] = block_mask

        if end < d_in:
            W[:, end:] -= Err_block @ Hinv[start:end, end:].to(compute_dtype)

    W = W.to(dtype)
    W = W * mask
    return W, mask
