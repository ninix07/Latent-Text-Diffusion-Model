"""Tests for CFGSampler."""

from __future__ import annotations

import torch
import pytest

from src.models.diffusion.denoiser import ConditionalDenoiser
from src.models.sampler.cfg_sampler import CFGSampler

# Tiny dimensions matching the project's tiny_config
LATENT_DIM = 16
DENOISER_DIM = 32
SEQ_LEN = 10
COND_LEN = 8
B = 4
NUM_TIMESTEPS = 100


@pytest.fixture
def denoiser() -> ConditionalDenoiser:
    """Tiny ConditionalDenoiser for testing."""
    return ConditionalDenoiser(
        latent_dim=LATENT_DIM,
        denoiser_dim=DENOISER_DIM,
        num_layers=1,
        num_heads=2,
        ff_dim=128,
    )


@pytest.fixture
def inputs():
    """Return (z_t, t, conditioning, conditioning_mask)."""
    torch.manual_seed(42)
    z_t = torch.randn(B, SEQ_LEN, LATENT_DIM)
    t = torch.zeros(B, dtype=torch.long)
    conditioning = torch.randn(B, COND_LEN, DENOISER_DIM)
    conditioning_mask = torch.zeros(B, COND_LEN, dtype=torch.bool)
    return z_t, t, conditioning, conditioning_mask


# ------------------------------------------------------------------
def test_w0_equals_unconditional(denoiser: ConditionalDenoiser, inputs) -> None:
    """With guidance_scale=0, CFG output should equal unconditional prediction."""
    z_t, t, conditioning, conditioning_mask = inputs

    cfg = CFGSampler(denoiser, guidance_scale=0.0)
    denoiser.eval()

    with torch.no_grad():
        eps_cfg = cfg.predict_noise(z_t, t, conditioning, conditioning_mask)

        # Unconditional: null conditioning = zeros
        null_cond = torch.zeros_like(conditioning)
        null_mask = torch.zeros_like(conditioning_mask)
        eps_uncond = denoiser(z_t, t, null_cond, null_mask)

    assert torch.allclose(eps_cfg, eps_uncond, atol=1e-5), (
        "With guidance_scale=0, CFG output must equal unconditional prediction"
    )


def test_output_shape(denoiser: ConditionalDenoiser, inputs) -> None:
    """CFG output shape must match input z_t shape."""
    z_t, t, conditioning, conditioning_mask = inputs

    cfg = CFGSampler(denoiser, guidance_scale=3.0)
    denoiser.eval()

    with torch.no_grad():
        eps_cfg = cfg.predict_noise(z_t, t, conditioning, conditioning_mask)

    assert eps_cfg.shape == z_t.shape, (
        f"Expected shape {z_t.shape}, got {eps_cfg.shape}"
    )
