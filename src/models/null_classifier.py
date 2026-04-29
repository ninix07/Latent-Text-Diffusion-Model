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
        3. Linear(hidden_dim, 1)  → returns *logits* (no sigmoid)

    Sigmoid is intentionally NOT applied inside the module so that the
    training loss can use ``binary_cross_entropy_with_logits`` (numerically
    stable). Callers that need probabilities should use
    :meth:`predict_proba` or apply ``torch.sigmoid`` to the forward output.

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
        )

    # ------------------------------------------------------------------
    def forward(self, z0: Tensor) -> Tensor:
        """Return per-sample logits.

        Parameters
        ----------
        z0 : Tensor
            Latent representations, shape ``(B, latent_dim)`` (pooled),
            ``(B, K, latent_dim)`` (sequence latent — mean-pooled over K),
            or ``(B, seq_len, latent_dim)`` (legacy 3D — mean-pooled).

        Returns
        -------
        Tensor
            Per-sample *logits*, shape ``(B,)``. Apply ``torch.sigmoid`` to
            convert to probabilities.
        """
        if z0.dim() == 3:
            z0 = z0.mean(dim=1)  # (B, latent_dim)
        logit = self.mlp(z0)  # (B, 1)
        return logit.squeeze(-1)  # (B,)

    # ------------------------------------------------------------------
    def predict_proba(self, z0: Tensor) -> Tensor:
        """Return per-sample probabilities in ``[0, 1]``."""
        return torch.sigmoid(self.forward(z0))

    # ------------------------------------------------------------------
    def predict(self, z0: Tensor, threshold: float = 0.5) -> tuple[bool, float]:
        """Classify a single sample (or the first sample in a batch).

        Parameters
        ----------
        z0 : Tensor
            Latent, shape ``(B, latent_dim)``, ``(latent_dim,)``,
            ``(B, K, latent_dim)``, or ``(B, seq_len, latent_dim)``.
        threshold : float
            Decision boundary on the probability; default 0.5.

        Returns
        -------
        is_answerable : bool
            True if probability >= threshold.
        confidence : float
            Probability in ``[0, 1]``.
        """
        with torch.no_grad():
            if z0.dim() == 1:
                z0 = z0.unsqueeze(0)  # add batch dim
            probs = self.predict_proba(z0)  # (B,)
            conf = float(probs[0].item())
        return conf >= threshold, conf
