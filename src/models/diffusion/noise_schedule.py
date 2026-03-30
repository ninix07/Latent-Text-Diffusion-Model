"""Cosine noise schedule for the diffusion forward process.

Implements the cosine schedule from Nichol & Dhariwal (2021):
    f(t) = cos((t/T + s) / (1 + s) * pi/2)^2
    alphas_cumprod(t) = f(t) / f(0)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class CosineNoiseSchedule(nn.Module):
    """Precomputes and stores the cosine noise schedule as registered buffers.

    Parameters
    ----------
    num_timesteps : int
        Total number of diffusion timesteps *T*.
    cosine_s : float
        Small offset to prevent singularity near t = 0.
    """

    def __init__(self, num_timesteps: int = 1000, cosine_s: float = 0.008) -> None:
        super().__init__()
        self.num_timesteps = num_timesteps

        # Build timestep indices [0, 1, ..., T-1]
        steps = torch.arange(num_timesteps, dtype=torch.float64)

        # f(t) = cos((t/T + s) / (1+s) * pi/2)^2
        f_t = torch.cos(((steps / num_timesteps) + cosine_s) / (1.0 + cosine_s) * (math.pi / 2.0)) ** 2
        f_0 = f_t[0]

        # alphas_cumprod = f(t) / f(0), clipped to avoid exact 0 or 1
        alphas_cumprod = (f_t / f_0).clamp(min=1e-5, max=1.0 - 1e-5)

        # Enforce strict monotonic decrease: if clipping created ties,
        # nudge tied values so that alphas_cumprod[i] > alphas_cumprod[i+1].
        for i in range(num_timesteps - 1, 0, -1):
            if alphas_cumprod[i] >= alphas_cumprod[i - 1]:
                alphas_cumprod[i - 1] = alphas_cumprod[i] + 1e-7

        # Derived quantities
        sqrt_alphas_cumprod = alphas_cumprod.sqrt()
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod).sqrt()
        log_snr = torch.log(alphas_cumprod / (1.0 - alphas_cumprod))

        # Register as float32 buffers (not parameters)
        self.register_buffer("alphas_cumprod", alphas_cumprod.float())
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod.float())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod.float())
        self.register_buffer("log_snr", log_snr.float())

    # ------------------------------------------------------------------
    def snr(self, t: Tensor) -> Tensor:
        """Return the signal-to-noise ratio at timestep(s) *t*.

        SNR(t) = alphas_cumprod[t] / (1 - alphas_cumprod[t])

        Parameters
        ----------
        t : Tensor
            Integer timestep indices (may be batched).

        Returns
        -------
        Tensor
            SNR values, same shape as *t*.
        """
        alpha = self.alphas_cumprod[t]
        return alpha / (1.0 - alpha)

    def __repr__(self) -> str:
        return f"CosineNoiseSchedule(T={self.num_timesteps})"
