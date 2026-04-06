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
    """KL(q(z|x) || N(0, I)), summed over latent dims, averaged over batch.

    Parameters
    ----------
    mu, log_var : Tensor (B, D) — pooled latent parameters.
    mask : Tensor, optional
        Ignored (kept for backward compatibility).  Masking is no longer
        needed because the encoder pools over real positions before
        producing mu/log_var.
    """
    kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())  # (B, D)
    return kl_per_dim.mean(dim=0).sum()
