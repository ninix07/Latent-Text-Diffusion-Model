"""Linear projection and segment embedding for conditioning signals."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConditioningProjection(nn.Module):
    """Project encoder hidden states into the denoiser dimension and add
    learned segment embeddings to distinguish question vs. context tokens.

    Parameters
    ----------
    encoder_dim : int
        Dimension of the frozen encoder hidden states.
    denoiser_dim : int
        Dimension expected by the denoiser transformer.
    """

    def __init__(self, encoder_dim: int, denoiser_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(encoder_dim, denoiser_dim)
        self.segment_embedding = nn.Embedding(2, denoiser_dim)

        # Initialise segment embeddings with small standard deviation so they
        # start near zero and do not dominate the projected signal.
        nn.init.normal_(self.segment_embedding.weight, std=0.02)

    def forward(
        self,
        h_q: torch.Tensor,
        q_mask: torch.Tensor,
        h_c: torch.Tensor,
        c_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project and concatenate question and context representations.

        Parameters
        ----------
        h_q : Tensor
            Question hidden states ``(B, q_len, encoder_dim)``.
        q_mask : Tensor
            Question attention mask ``(B, q_len)``.
        h_c : Tensor
            Context hidden states ``(B, c_len, encoder_dim)``.
        c_mask : Tensor
            Context attention mask ``(B, c_len)``.

        Returns
        -------
        conditioning : Tensor
            Concatenated projected representations ``(B, q_len + c_len, denoiser_dim)``.
        conditioning_mask : Tensor
            Concatenated attention masks ``(B, q_len + c_len)``.
        """
        # Project both into denoiser dimension.
        proj_q = self.projection(h_q)  # (B, q_len, denoiser_dim)
        proj_c = self.projection(h_c)  # (B, c_len, denoiser_dim)

        # Add segment embeddings: 0 for question, 1 for context.
        seg_q = self.segment_embedding(
            torch.zeros(h_q.shape[:2], dtype=torch.long, device=h_q.device)
        )
        seg_c = self.segment_embedding(
            torch.ones(h_c.shape[:2], dtype=torch.long, device=h_c.device)
        )

        proj_q = proj_q + seg_q
        proj_c = proj_c + seg_c

        # Concatenate along the sequence dimension.
        conditioning = torch.cat([proj_q, proj_c], dim=1)
        conditioning_mask = torch.cat([q_mask, c_mask], dim=1)

        return conditioning, conditioning_mask
