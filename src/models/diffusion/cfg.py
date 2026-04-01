"""Classifier-free guidance (CFG) dropout utilities.

During training, randomly drops out conditioning for a fraction of samples
so the model learns both conditional and unconditional generation.
"""

from __future__ import annotations

import torch
from torch import Tensor


def cfg_dropout_mask(
    batch_size: int,
    rate: float,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Generate a per-sample boolean mask for CFG dropout.

    Parameters
    ----------
    batch_size : int
        Number of samples.
    rate : float
        Probability of dropping a sample's conditioning.
    device : torch.device or str
        Device to create the mask on.

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
        and their masks set to ``False`` (attend to null conditioning).
    """
    drop = cfg_dropout_mask(conditioning.size(0), dropout_rate, device=conditioning.device)
    conditioning = conditioning.clone()
    conditioning_mask = conditioning_mask.clone()

    # Zero conditioning for dropped samples but keep mask as False so the
    # model attends to zero-valued keys (masking all keys causes NaN in softmax).
    conditioning[drop] = 0.0
    conditioning_mask[drop] = False

    return conditioning, conditioning_mask
