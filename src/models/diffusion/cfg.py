"""Classifier-free guidance (CFG) dropout utilities.

During training, randomly drops out conditioning for a fraction of samples
so the model learns both conditional and unconditional generation.
"""

from __future__ import annotations

import torch
from torch import Tensor


def cfg_dropout_mask(batch_size: int, rate: float, device: torch.device | None = None) -> Tensor:
    """Generate a per-sample boolean mask for CFG dropout.

    Parameters
    ----------
    batch_size : int
        Number of samples.
    rate : float
        Probability of dropping a sample's conditioning.
    device : torch.device, optional
        Device for the returned mask. Should match the conditioning tensor.

    Returns
    -------
    BoolTensor
        Shape ``(batch_size,)``. ``True`` means drop (zero out) that sample.
    """
    return torch.rand(batch_size, device=device) < rate


def apply_cfg_dropout(
    conditioning: Tensor,
    conditioning_mask: Tensor,
    dropout_rate: float,
) -> tuple[Tensor, Tensor]:
    """Zero out conditioning for a random subset of samples.

    Parameters
    ----------
    conditioning : Tensor
        Shape ``(B, C, D)``.
    conditioning_mask : Tensor
        Shape ``(B, C)``.
    dropout_rate : float
        Fraction of samples to drop.

    Returns
    -------
    tuple of (Tensor, Tensor)
        Modified conditioning and mask with dropped samples zeroed out
        and their masks set to ``True`` (ignored).
    """
    drop = cfg_dropout_mask(conditioning.size(0), dropout_rate, device=conditioning.device)  # (B,)
    conditioning = conditioning.clone()
    conditioning_mask = conditioning_mask.clone()

    # Zero conditioning and mark all positions as masked for dropped samples
    conditioning[drop] = 0.0
    conditioning_mask[drop] = True

    return conditioning, conditioning_mask
