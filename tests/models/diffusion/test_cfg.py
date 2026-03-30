"""Tests for classifier-free guidance dropout utilities."""

import torch

from src.models.diffusion.cfg import apply_cfg_dropout, cfg_dropout_mask


def test_dropout_rate():
    """Rate=0.5 on 1000 samples should zero out roughly 50%."""
    torch.manual_seed(0)
    n = 1000
    mask = cfg_dropout_mask(n, rate=0.5)
    frac = mask.float().mean().item()
    assert 0.4 < frac < 0.6, f"Expected ~50% dropout, got {frac:.2%}"


def test_rate_zero_no_dropout():
    mask = cfg_dropout_mask(1000, rate=0.0)
    assert mask.sum().item() == 0


def test_rate_one_all_dropout():
    mask = cfg_dropout_mask(1000, rate=1.0)
    assert mask.sum().item() == 1000


def test_apply_cfg_dropout_zeros_conditioning():
    B, C, D = 16, 8, 32
    cond = torch.randn(B, C, D)
    mask = torch.zeros(B, C, dtype=torch.bool)

    torch.manual_seed(42)
    new_cond, new_mask = apply_cfg_dropout(cond, mask, dropout_rate=0.5)

    # Dropped samples should have all-zero conditioning
    dropped = new_mask.all(dim=-1)  # (B,) True where all positions masked
    for i in range(B):
        if dropped[i]:
            assert (new_cond[i] == 0).all()
