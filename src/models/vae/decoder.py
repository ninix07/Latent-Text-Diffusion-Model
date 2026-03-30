"""VAE decoder: maps latent z back to hidden states in embed_dim space."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _sinusoidal_encoding(max_len: int, dim: int) -> torch.Tensor:
    """Return a (max_len, dim) sinusoidal positional-encoding table."""
    pos = torch.arange(max_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    pe = torch.zeros(max_len, dim)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class VAEDecoder(nn.Module):
    """Transformer decoder that maps latent z to hidden states.

    Uses ``nn.TransformerEncoder`` (not ``TransformerDecoder``) because we
    process all positions in parallel — there is no autoregressive masking.

    Output shape: ``(B, max_answer_len, embed_dim)``.  Does **not** produce
    logits; that is handled by :class:`OutputProjection`.
    """

    def __init__(
        self,
        latent_dim: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        max_answer_len: int,
    ) -> None:
        super().__init__()

        # Up-project from latent to embed space
        self.up_proj = nn.Linear(latent_dim, embed_dim)

        # Positional encoding
        pe = _sinusoidal_encoding(max_answer_len, embed_dim)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, L, D)

        # Transformer
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Decode latent vectors to hidden states.

        Parameters
        ----------
        z : Tensor (B, L, latent_dim)
        mask : Tensor (B, L)  — 1 for real, 0 for padding.

        Returns
        -------
        Tensor (B, L, embed_dim)
        """
        x = self.up_proj(z) + self.pe[:, : z.size(1), :]
        pad_mask = mask == 0
        x = self.transformer(x, src_key_padding_mask=pad_mask)
        return x
