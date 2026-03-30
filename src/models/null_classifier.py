"""Null-answer classifier for latent diffusion outputs.

A lightweight MLP that scores whether a z0 latent corresponds to an
answerable question (high confidence) or a null/unanswerable one.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class NullClassifier(nn.Module):
    """Binary classifier operating on pooled latent representations.

    Architecture:
        1. Mean-pool z0 over the sequence dimension -> ``(B, latent_dim)``
        2. Linear(latent_dim, hidden_dim) -> ReLU
        3. Linear(hidden_dim, 1) -> Sigmoid

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the latent space.
    hidden_dim : int
        Hidden dimension of the MLP.
    """

    def __init__(self, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    # ------------------------------------------------------------------
    def forward(self, z0: Tensor) -> Tensor:
        """Compute per-sample answerability probability.

        Parameters
        ----------
        z0 : Tensor
            Latent representations, shape ``(B, seq_len, latent_dim)``.

        Returns
        -------
        Tensor
            Per-sample probabilities, shape ``(B,)``.
        """
        # Mean-pool over sequence dimension
        pooled = z0.mean(dim=1)            # (B, latent_dim)
        logit = self.mlp(pooled)           # (B, 1)
        return logit.squeeze(-1)           # (B,)

    # ------------------------------------------------------------------
    def predict(
        self, z0: Tensor, threshold: float = 0.5
    ) -> tuple[bool, float]:
        """Classify a single sample (or the first sample in a batch).

        Parameters
        ----------
        z0 : Tensor
            Latent, shape ``(B, seq_len, latent_dim)`` or
            ``(seq_len, latent_dim)``.  Only the first batch element is
            used when B > 1.
        threshold : float
            Decision boundary; default 0.5.

        Returns
        -------
        is_answerable : bool
            True if confidence >= threshold.
        confidence : float
            Raw classifier score in [0, 1].
        """
        with torch.no_grad():
            if z0.dim() == 2:
                z0 = z0.unsqueeze(0)      # add batch dim
            probs = self.forward(z0)      # (B,)
            conf = float(probs[0].item())
        return conf >= threshold, conf
