"""Correctness check for the CWS OBS kernel: on correlated (non-diagonal)
covariance, CWS's cancellation-aware selection + OBS correction should
reconstruct the layer output at least as well as uncorrected magnitude
pruning, which is blind to the cross-channel correlations CWS is designed
to exploit."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cws.baselines import magnitude_prune_layer
from cws.cws_obs import compute_hinv_cholesky, cws_prune_layer


def make_correlated_covariance(d_in: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(d_in, d_in, generator=g)
    cov = A @ A.t() + d_in * torch.eye(d_in)  # SPD, off-diagonal correlation
    return cov


def output_reconstruction_error(W, W_pruned, Sigma_X):
    delta = W - W_pruned
    return torch.einsum("oi,ij,oj->o", delta, Sigma_X, delta).sum().item()


def run():
    torch.manual_seed(0)
    d_out, d_in = 16, 32
    W = torch.randn(d_out, d_in, dtype=torch.float64)
    H = make_correlated_covariance(d_in).double()

    Hinv = compute_hinv_cholesky(H)
    W_cws, mask_cws = cws_prune_layer(W, Hinv, sparsity=0.5, blocksize=None)
    W_mag, mask_mag = magnitude_prune_layer(W, sparsity=0.5)

    assert (mask_cws.float().mean(dim=1) == 0.5).all()
    assert (mask_mag.float().mean(dim=1) == 0.5).all()

    err_cws = output_reconstruction_error(W, W_cws, H)
    err_mag = output_reconstruction_error(W, W_mag.double(), H)

    print(f"CWS reconstruction error:       {err_cws:.4f}")
    print(f"Magnitude reconstruction error: {err_mag:.4f}")
    assert err_cws <= err_mag * 1.001, (
        "CWS should not do worse than magnitude pruning on correlated activations"
    )
    print("PASSED: CWS reconstructs the layer output at least as well as magnitude pruning")


if __name__ == "__main__":
    run()
