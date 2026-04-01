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
    mu: torch.Tensor, log_var: torch.Tensor, free_bits: float = 0.0
) -> torch.Tensor:
    """KL(q(z|x) || N(0, I)).

    When *free_bits* > 0, applies the free-bits technique (Kingma et al.
    2016): the per-dimension KL (averaged over batch and sequence) is
    clamped to at least *free_bits* nats before summing.  This prevents
    posterior collapse by guaranteeing every latent dimension carries a
    minimum amount of information.
    """
    kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())  # (B, L, D)

    if free_bits > 0.0:
        # Average over batch and sequence, clamp per dim, then sum over dims
        kl_dim_mean = kl_per_dim.mean(dim=list(range(0, kl_per_dim.ndim - 1)))  # (D,)
        kl_dim_clamped = torch.clamp(kl_dim_mean, min=free_bits)
        return kl_dim_clamped.sum()

    # Original: sum over all dims per sample, average over batch
    kl_per_sample = kl_per_dim.sum(dim=list(range(1, mu.ndim)))
    return kl_per_sample.mean()
