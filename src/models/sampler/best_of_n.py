"""Best-of-N sampling utility.

Generates N candidate latents and returns the one with the highest
confidence score according to a null classifier.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor


def best_of_n_sample(
    generate_fn: Callable[[], Tensor],
    n: int,
    null_classifier: Callable[[Tensor], Tensor],
) -> tuple[Tensor, Tensor]:
    """Sample N latents and return the highest-confidence one.

    Parameters
    ----------
    generate_fn : Callable[[], Tensor]
        Zero-argument function that returns a z0 tensor of shape
        ``(B, seq_len, latent_dim)``.
    n : int
        Number of candidate samples to generate.
    null_classifier : Callable[[Tensor], Tensor]
        Callable that accepts z0 ``(B, seq_len, latent_dim)`` and returns
        per-sample confidence scores of shape ``(B,)``.

    Returns
    -------
    best_z0 : Tensor
        The candidate z0 with the highest confidence, shape
        ``(B, seq_len, latent_dim)``.
    confidence : Tensor
        The corresponding confidence score, shape ``(B,)``.
    """
    best_z0: Tensor | None = None
    best_confidence: Tensor | None = None

    for _ in range(n):
        z0 = generate_fn()
        confidence = null_classifier(z0)

        if best_z0 is None or best_confidence is None:
            best_z0 = z0
            best_confidence = confidence
        else:
            # For each sample in the batch, pick whichever candidate is better
            improved = confidence > best_confidence  # (B,)
            # Broadcast improved to match z0 dimensions
            improved_z = improved.view(-1, 1, 1).expand_as(z0)
            best_z0 = torch.where(improved_z, z0, best_z0)
            best_confidence = torch.where(improved, confidence, best_confidence)

    assert best_z0 is not None and best_confidence is not None
    return best_z0, best_confidence
