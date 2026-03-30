"""Tests for DDIMSampler."""

from __future__ import annotations

import torch
import pytest

from src.models.diffusion.noise_schedule import CosineNoiseSchedule
from src.models.sampler.ddim import DDIMSampler

# Tiny dimensions
LATENT_DIM = 16
SEQ_LEN = 10
B = 4
NUM_TIMESTEPS = 100
NUM_INFERENCE_STEPS = 5


@pytest.fixture
def schedule() -> CosineNoiseSchedule:
    return CosineNoiseSchedule(num_timesteps=NUM_TIMESTEPS)


@pytest.fixture
def sampler(schedule: CosineNoiseSchedule) -> DDIMSampler:
    return DDIMSampler(schedule, num_inference_steps=NUM_INFERENCE_STEPS, eta=0.0)


# ------------------------------------------------------------------
def test_timestep_subsequence_decreasing(sampler: DDIMSampler) -> None:
    """Timesteps must be strictly decreasing (high noise to low)."""
    timesteps = sampler.get_timesteps()
    for i in range(len(timesteps) - 1):
        assert timesteps[i] > timesteps[i + 1], (
            f"Timestep at index {i} ({timesteps[i]}) is not greater than "
            f"index {i+1} ({timesteps[i+1]})"
        )


def test_timestep_count(sampler: DDIMSampler) -> None:
    """Length of subsequence must equal num_inference_steps."""
    assert len(sampler.get_timesteps()) == NUM_INFERENCE_STEPS


def test_deterministic_eta0(schedule: CosineNoiseSchedule) -> None:
    """With eta=0 and identical inputs, two calls must return identical z0."""
    sampler = DDIMSampler(schedule, num_inference_steps=NUM_INFERENCE_STEPS, eta=0.0)
    torch.manual_seed(0)
    z_t = torch.randn(B, SEQ_LEN, LATENT_DIM)
    eps_pred = torch.randn(B, SEQ_LEN, LATENT_DIM)

    result1 = sampler.predict_z0(z_t, eps_pred, t_idx=0)
    result2 = sampler.predict_z0(z_t, eps_pred, t_idx=0)
    assert torch.allclose(result1, result2), "eta=0 predictions should be identical"


def test_predict_z0_shape(sampler: DDIMSampler) -> None:
    """predict_z0 output shape must match input shape."""
    z_t = torch.randn(B, SEQ_LEN, LATENT_DIM)
    eps_pred = torch.randn(B, SEQ_LEN, LATENT_DIM)
    z0 = sampler.predict_z0(z_t, eps_pred, t_idx=0)
    assert z0.shape == (B, SEQ_LEN, LATENT_DIM)


def test_sample_shape(schedule: CosineNoiseSchedule) -> None:
    """Full sampling loop must return correct shape."""
    sampler = DDIMSampler(schedule, num_inference_steps=NUM_INFERENCE_STEPS, eta=0.0)

    def dummy_denoiser(z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(z_t)

    z0 = sampler.sample(
        denoiser_fn=dummy_denoiser,
        z_shape=(B, SEQ_LEN, LATENT_DIM),
        device="cpu",
    )
    assert z0.shape == (B, SEQ_LEN, LATENT_DIM)
