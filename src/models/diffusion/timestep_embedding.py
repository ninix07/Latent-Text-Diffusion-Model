"""Sinusoidal timestep embeddings and MLP projection.

Encodes integer diffusion timesteps into continuous vectors using
sinusoidal frequency bands, then projects through an MLP.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal positional encoding for scalar timesteps.

    Parameters
    ----------
    dim : int
        Embedding dimension (must be even).
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        assert dim % 2 == 0, "Embedding dim must be even."
        half = dim // 2
        # log-spaced frequencies: exp(-log(10000) * i / (half - 1))
        freq = torch.exp(-math.log(10_000.0) * torch.arange(half, dtype=torch.float32) / (half - 1))
        self.register_buffer("freq", freq)  # (half,)

    def forward(self, t: Tensor) -> Tensor:
        """Encode integer timesteps.

        Parameters
        ----------
        t : Tensor
            Integer timesteps, shape ``(B,)`` or ``(B, 1)``.

        Returns
        -------
        Tensor
            Shape ``(B, dim)``.
        """
        t = t.float().view(-1, 1)  # (B, 1)
        angles = t * self.freq.unsqueeze(0)  # (B, half)
        return torch.cat([angles.sin(), angles.cos()], dim=-1)  # (B, dim)


class TimestepMLP(nn.Module):
    """Sinusoidal embedding followed by a two-layer MLP.

    Parameters
    ----------
    sinusoidal_dim : int
        Dimension of the sinusoidal embedding.
    output_dim : int
        Output dimension after MLP projection.
    """

    def __init__(self, sinusoidal_dim: int, output_dim: int) -> None:
        super().__init__()
        self.sinusoidal = SinusoidalTimestepEmbedding(sinusoidal_dim)
        self.mlp = nn.Sequential(
            nn.Linear(sinusoidal_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        """Embed timesteps and project through MLP.

        Returns
        -------
        Tensor
            Shape ``(B, output_dim)``.
        """
        return self.mlp(self.sinusoidal(t))
