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


def kl_divergence(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """KL(q(z|x) || N(0, I)), summed over dims, averaged over batch."""
    # per-sample: sum over all dims (seq_len * latent_dim)
    kl_per_sample = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=list(range(1, mu.ndim)))
    return kl_per_sample.mean()
