"""Conditional denoiser (epsilon-prediction transformer).

Stacks multiple :class:`DenoiserBlock` layers, adds input/output projections,
sinusoidal positional encoding for the latent sequence, and a timestep MLP.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.models.diffusion.timestep_embedding import TimestepMLP
from src.models.diffusion.denoiser_block import DenoiserBlock
from src.models.positional import sinusoidal_encoding


class ConditionalDenoiser(nn.Module):
    """Transformer-based conditional denoiser for latent diffusion.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the latent space.
    denoiser_dim : int
        Internal hidden dimension.
    num_layers : int
        Number of :class:`DenoiserBlock` layers.
    num_heads : int
        Attention heads per block.
    ff_dim : int
        Feed-forward inner dimension.
    max_seq_len : int
        Maximum latent sequence length (for positional encoding).
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        latent_dim: int,
        denoiser_dim: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(latent_dim, denoiser_dim)

        # Sinusoidal positional encoding for latent positions
        pe = sinusoidal_encoding(max_seq_len, denoiser_dim).unsqueeze(0)  # (1, L, D)
        self.register_buffer("pos_encoding", pe)

        # Timestep embedding MLP
        self.time_mlp = TimestepMLP(denoiser_dim, denoiser_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DenoiserBlock(denoiser_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        # Output projection back to latent space
        self.output_proj = nn.Linear(denoiser_dim, latent_dim)

    def forward(
        self,
        z_t: Tensor,
        t: Tensor,
        conditioning: Tensor,
        conditioning_mask: Tensor | None = None,
    ) -> Tensor:
        """Predict noise given noisy latents, timestep, and conditioning.

        Parameters
        ----------
        z_t : Tensor
            Noisy latent sequence, shape ``(B, seq_len, latent_dim)``.
        t : Tensor
            Integer timesteps, shape ``(B,)``.
        conditioning : Tensor
            Conditioning sequence, shape ``(B, C, denoiser_dim)``.
        conditioning_mask : Tensor or None
            Key padding mask for conditioning, shape ``(B, C)``.

        Returns
        -------
        Tensor
            Predicted noise, shape ``(B, seq_len, latent_dim)``.
        """
        B, S, _ = z_t.shape

        # Project to denoiser dim and add positional encoding
        h = self.input_proj(z_t) + self.pos_encoding[:, :S, :]

        # Timestep conditioning
        t_emb = self.time_mlp(t)  # (B, denoiser_dim)

        # Transformer blocks
        for block in self.blocks:
            h = block(h, t_emb, conditioning, conditioning_mask)

        # Project back to latent dim
        return self.output_proj(h)
