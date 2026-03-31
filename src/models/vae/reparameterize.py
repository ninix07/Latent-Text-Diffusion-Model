"""Reparameterization trick and KL divergence for the VAE."""

from __future__ import annotations

import torch


def reparameterize(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    deterministic: bool = False,
) -> torch.Tensor:
    """Sample z via the reparameterization trick.

    When *deterministic* is True the mean is returned directly.
    """
    if deterministic:
        return mu
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + std * eps


def kl_divergence(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """KL(q(z|x) || N(0, I)), summed over latent dims, averaged over positions.

    Parameters
    ----------
    mu, log_var : Tensor (B, L, D)
    mask : Tensor (B, L), optional
        1 for real tokens, 0 for padding.  When provided, KL is averaged only
        over real positions — consistent with the masked computation in loss.py.
        When None, all positions are averaged (backward-compatible default).
    """
    kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())  # (B, L, D)
    if mask is None:
        return kl_per_dim.mean(dim=(0, 1)).sum()
    mask_3d = mask.unsqueeze(-1).float()  # (B, L, 1)
    # sum over (B, L) masked positions → (D,), divide by total real positions, then sum over D
    n_real = mask_3d.sum().clamp(min=1)
    return (kl_per_dim * mask_3d).sum(dim=(0, 1)) / n_real
