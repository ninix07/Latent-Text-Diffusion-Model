"""Forward (noising) process for latent diffusion.

Provides ``q_sample`` which adds noise to clean latents according to
a precomputed noise schedule.
"""

from __future__ import annotations

import torch
from torch import Tensor

from src.models.diffusion.noise_schedule import CosineNoiseSchedule


def q_sample(
    z0: Tensor,
    t: Tensor,
    schedule: CosineNoiseSchedule,
    noise: Tensor | None = None,
) -> Tensor:
    """Sample from the forward diffusion process q(z_t | z_0).

    z_t = sqrt(alphas_cumprod[t]) * z0 + sqrt(1 - alphas_cumprod[t]) * noise

    Parameters
    ----------
    z0 : Tensor
        Clean latent vectors, shape ``(B, D)`` or ``(B, ...)``.
    t : Tensor
        Integer timestep indices, shape ``(B,)``.
    schedule : CosineNoiseSchedule
        Precomputed noise schedule providing the required buffers.
    noise : Tensor or None
        Optional pre-sampled noise; if *None*, sampled from N(0, I).

    Returns
    -------
    Tensor
        Noised latents z_t, same shape as *z0*.
    """
    if noise is None:
        noise = torch.randn_like(z0)

    # Gather per-sample coefficients and reshape for broadcasting
    # t is (B,), we need shape (B, 1, ...) to broadcast with z0
    dims_to_add = z0.dim() - 1  # number of non-batch dimensions
    shape = (-1,) + (1,) * dims_to_add

    sqrt_alpha = schedule.sqrt_alphas_cumprod[t].view(*shape)
    sqrt_one_minus_alpha = schedule.sqrt_one_minus_alphas_cumprod[t].view(*shape)

    return sqrt_alpha * z0 + sqrt_one_minus_alpha * noise
