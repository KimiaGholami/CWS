"""Calibration-time activation-covariance (empirical Hessian) accumulation.

Implements H = (2/N) X^T X for a single nn.Linear, accumulated batch-by-batch
via a forward hook, matching the SparseGPT/CWS convention (Frantar & Alistarh,
2023, Sec 3.1; CWS paper Sec 3.1).
"""

import math

import torch
import torch.nn as nn


class HessianAccumulator:
    """Accumulates H = (2/N) X^T X for one Linear layer's input activations."""

    def __init__(self, layer: nn.Linear):
        self.layer = layer
        d_in = layer.weight.shape[1]
        device = layer.weight.device
        self.H = torch.zeros((d_in, d_in), device=device, dtype=torch.float32)
        self.nsamples = 0
        self._handle = None

    def add_batch(self, inp: torch.Tensor) -> None:
        if inp.dim() == 2:
            inp = inp.unsqueeze(0)
        inp = inp.reshape(-1, inp.shape[-1]).t().float()
        n_new = inp.shape[1]
        if n_new == 0:
            return
        self.H *= self.nsamples / (self.nsamples + n_new)
        self.nsamples += n_new
        inp = math.sqrt(2 / self.nsamples) * inp
        self.H += inp @ inp.t()

    def _hook(self, module, args, kwargs=None):
        inp = args[0]
        if isinstance(inp, tuple):
            inp = inp[0]
        self.add_batch(inp.detach())

    def register(self) -> None:
        self._handle = self.layer.register_forward_pre_hook(
            self._hook, with_kwargs=True
        )

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def damped_hessian(self, percdamp: float = 0.01) -> torch.Tensor:
        """Return H with a damping term λ = percdamp * mean(diag(H)) added,
        and dead input channels (zero activation) patched to identity so the
        Cholesky factorization doesn't fail on unused columns."""
        H = self.H.clone()
        dead = torch.diagonal(H) == 0
        H[dead, dead] = 1.0
        damp = percdamp * torch.diagonal(H).mean()
        diag_idx = torch.arange(H.shape[0], device=H.device)
        H[diag_idx, diag_idx] += damp
        return H


def build_hinv(H: torch.Tensor, percdamp: float = 0.01, compute_dtype=torch.float64):
    """Damp H and return CWS/SparseGPT's upper-triangular Hinv-sequence matrix."""
    from .cws_obs import compute_hinv_cholesky

    dead = torch.diagonal(H) == 0
    H = H.clone()
    H[dead, dead] = 1.0
    damp = percdamp * torch.diagonal(H).mean()
    diag_idx = torch.arange(H.shape[0], device=H.device)
    H[diag_idx, diag_idx] += damp
    return compute_hinv_cholesky(H.to(compute_dtype))
