"""DDIM (Denoising Diffusion Implicit Models) sampler.

Implements the deterministic (eta=0) and stochastic (eta>0) sampling
procedure from Song et al. (2020): https://arxiv.org/abs/2010.02502
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from src.models.diffusion.noise_schedule import CosineNoiseSchedule


class DDIMSampler:
    """DDIM sampler that iterates over a subsequence of timesteps.

    Parameters
    ----------
    schedule : CosineNoiseSchedule
        Precomputed cosine noise schedule.
    num_inference_steps : int
        Number of DDIM steps (length of subsequence).
    eta : float
        Stochasticity parameter; 0.0 = fully deterministic DDIM.
    """

    def __init__(
        self,
        schedule: CosineNoiseSchedule,
        num_inference_steps: int,
        eta: float = 0.0,
    ) -> None:
        self.schedule = schedule
        self.num_inference_steps = num_inference_steps
        self.eta = eta

        T = schedule.num_timesteps
        # Evenly spaced indices from T-1 down to 0, length=num_inference_steps
        step_ratio = T // num_inference_steps
        timesteps = list(reversed(range(0, T, step_ratio)))[:num_inference_steps]
        # Ensure we always have exactly num_inference_steps entries
        self._timesteps: list[int] = timesteps

    # ------------------------------------------------------------------
    def get_timesteps(self) -> list[int]:
        """Return the DDIM timestep subsequence (high to low)."""
        return self._timesteps

    # ------------------------------------------------------------------
    def predict_z0(self, z_t: Tensor, eps_pred: Tensor, t_idx: int) -> Tensor:
        """Predict clean z0 from noisy z_t and predicted noise.

        z0_pred = (z_t - sqrt(1-ᾱ_t) * eps_pred) / sqrt(ᾱ_t)

        Parameters
        ----------
        z_t : Tensor
            Noisy latent, shape ``(B, seq_len, latent_dim)``.
        eps_pred : Tensor
            Predicted noise, same shape as z_t.
        t_idx : int
            Index into the DDIM subsequence.

        Returns
        -------
        Tensor
            Predicted clean latent, same shape as z_t.
        """
        t = self._timesteps[t_idx]
        sqrt_alpha_t = self.schedule.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_t = self.schedule.sqrt_one_minus_alphas_cumprod[t]
        return (z_t - sqrt_one_minus_alpha_t * eps_pred) / sqrt_alpha_t

    # ------------------------------------------------------------------
    def step(self, z_t: Tensor, eps_pred: Tensor, t_idx: int) -> Tensor:
        """Perform a single DDIM reverse step.

        Parameters
        ----------
        z_t : Tensor
            Current noisy latent, shape ``(B, seq_len, latent_dim)``.
        eps_pred : Tensor
            Predicted noise, same shape as z_t.
        t_idx : int
            Current index into the DDIM subsequence (0 = highest noise).

        Returns
        -------
        Tensor
            z_prev: latent at previous (less noisy) timestep.
        """
        # Final step: return predicted z0 directly
        if t_idx == len(self._timesteps) - 1:
            return self.predict_z0(z_t, eps_pred, t_idx)

        t = self._timesteps[t_idx]
        t_prev = self._timesteps[t_idx + 1]

        alpha_t = self.schedule.alphas_cumprod[t]
        alpha_t_prev = self.schedule.alphas_cumprod[t_prev]

        z0_pred = self.predict_z0(z_t, eps_pred, t_idx)

        # Sigma for stochastic DDIM
        sigma = self.eta * torch.sqrt(
            (1.0 - alpha_t_prev) / (1.0 - alpha_t) * (1.0 - alpha_t / alpha_t_prev)
        )

        # Direction pointing to z_t
        coeff_eps = torch.sqrt((1.0 - alpha_t_prev - sigma ** 2).clamp(min=0.0))

        z_prev = (
            torch.sqrt(alpha_t_prev) * z0_pred
            + coeff_eps * eps_pred
            + sigma * torch.randn_like(z_t)
        )
        return z_prev

    # ------------------------------------------------------------------
    def sample(
        self,
        denoiser_fn: Callable[[Tensor, Tensor], Tensor],
        z_shape: tuple[int, int, int],
        device: torch.device | str,
    ) -> Tensor:
        """Run the full DDIM sampling loop from pure noise to z0.

        Parameters
        ----------
        denoiser_fn : Callable
            ``denoiser_fn(z_t, t_tensor) -> eps_pred`` where t_tensor has
            shape ``(B,)`` containing integer timestep indices.
        z_shape : tuple[int, int, int]
            ``(B, seq_len, latent_dim)`` shape for the latent.
        device : device
            Target device.

        Returns
        -------
        Tensor
            Denoised latent z0, shape ``z_shape``.
        """
        z_t = torch.randn(z_shape, device=device)
        B = z_shape[0]

        for i, t in enumerate(self._timesteps):
            t_tensor = torch.full((B,), t, dtype=torch.long, device=device)
            eps_pred = denoiser_fn(z_t, t_tensor)
            z_t = self.step(z_t, eps_pred, i)

        return z_t
