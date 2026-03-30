"""Classifier-Free Guidance (CFG) sampler.

Combines conditioned and unconditioned noise predictions in a single
batched forward pass for efficiency.
"""

from __future__ import annotations

import torch
from torch import Tensor

from src.models.diffusion.denoiser import ConditionalDenoiser


class CFGSampler:
    """Wraps a ConditionalDenoiser to apply classifier-free guidance.

    Parameters
    ----------
    denoiser : ConditionalDenoiser
        The trained conditional denoiser model.
    guidance_scale : float
        CFG weight *w*. At w=0, output equals unconditional prediction.
        At w=1, output equals conditional prediction.
    """

    def __init__(self, denoiser: ConditionalDenoiser, guidance_scale: float) -> None:
        self.denoiser = denoiser
        self.guidance_scale = guidance_scale

    # ------------------------------------------------------------------
    def predict_noise(
        self,
        z_t: Tensor,
        t: Tensor,
        conditioning: Tensor,
        conditioning_mask: Tensor,
    ) -> Tensor:
        """Predict CFG-combined noise estimate.

        Both the conditioned and unconditioned passes are batched together
        in a single forward call through the denoiser.

        Parameters
        ----------
        z_t : Tensor
            Noisy latent, shape ``(B, seq_len, latent_dim)``.
        t : Tensor
            Integer timesteps, shape ``(B,)``.
        conditioning : Tensor
            Conditioning sequence, shape ``(B, C, denoiser_dim)``.
        conditioning_mask : Tensor
            Key-padding mask for conditioning, shape ``(B, C)``.

        Returns
        -------
        Tensor
            CFG noise prediction, same shape as z_t.
        """
        B = z_t.shape[0]

        # Build null (unconditional) conditioning
        null_cond = torch.zeros_like(conditioning)
        null_mask = torch.zeros_like(conditioning_mask)

        # Stack conditioned and unconditioned in the batch dimension
        z_double = torch.cat([z_t, z_t], dim=0)           # (2B, seq_len, latent_dim)
        t_double = torch.cat([t, t], dim=0)                # (2B,)
        cond_double = torch.cat([conditioning, null_cond], dim=0)  # (2B, C, D)
        mask_double = torch.cat([conditioning_mask, null_mask], dim=0)  # (2B, C)

        # Single forward pass
        eps_double = self.denoiser(z_double, t_double, cond_double, mask_double)

        eps_cond = eps_double[:B]
        eps_uncond = eps_double[B:]

        # CFG combination
        w = self.guidance_scale
        eps_cfg = eps_uncond + w * (eps_cond - eps_uncond)
        return eps_cfg
