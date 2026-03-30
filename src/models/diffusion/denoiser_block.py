"""Single denoiser transformer block with AdaLN-Zero conditioning.

Each block contains:
* Adaptive-layer-norm conditioned self-attention with alpha gating
* Adaptive-layer-norm conditioned cross-attention with alpha gating
* Feed-forward network (ungated residual)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.models.diffusion.adaln_block import AdaLNModulation, ada_layer_norm


class DenoiserBlock(nn.Module):
    """Transformer block for the conditional denoiser.

    Parameters
    ----------
    denoiser_dim : int
        Hidden dimension throughout the block.
    num_heads : int
        Number of attention heads.
    ff_dim : int
        Inner dimension of the feed-forward network.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        denoiser_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # --- Self-attention ---
        self.self_attn = nn.MultiheadAttention(
            embed_dim=denoiser_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln_self = nn.LayerNorm(denoiser_dim)

        # --- Cross-attention ---
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=denoiser_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln_cross = nn.LayerNorm(denoiser_dim)

        # --- Feed-forward ---
        self.ffn = nn.Sequential(
            nn.Linear(denoiser_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, denoiser_dim),
            nn.Dropout(dropout),
        )
        self.ln_ffn = nn.LayerNorm(denoiser_dim)

        # --- AdaLN modulation for 6 params ---
        # gamma1, beta1, alpha1 (self-attn), gamma2, beta2, alpha2 (cross-attn)
        self.modulation = AdaLNModulation(denoiser_dim, num_params=6)

    def forward(
        self,
        x: Tensor,
        t_emb: Tensor,
        conditioning: Tensor,
        conditioning_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Latent sequence, shape ``(B, S, D)``.
        t_emb : Tensor
            Timestep embedding, shape ``(B, D)``.
        conditioning : Tensor
            Conditioning sequence (encoder output), shape ``(B, C, D)``.
        conditioning_mask : Tensor or None
            Key padding mask for conditioning, shape ``(B, C)``.
            True means **ignore** that position.

        Returns
        -------
        Tensor
            Same shape as *x*.
        """
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.modulation(t_emb)

        # --- Self-attention with AdaLN-Zero ---
        x_norm = ada_layer_norm(x, gamma1, beta1, self.ln_self)
        sa_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + alpha1 * sa_out

        # --- Cross-attention with AdaLN-Zero ---
        x_norm = ada_layer_norm(x, gamma2, beta2, self.ln_cross)
        ca_out, _ = self.cross_attn(
            x_norm,
            conditioning,
            conditioning,
            key_padding_mask=conditioning_mask,
        )
        x = x + alpha2 * ca_out

        # --- FFN (no alpha gate) ---
        x = x + self.ffn(self.ln_ffn(x))

        return x
