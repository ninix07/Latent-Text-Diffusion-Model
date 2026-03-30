"""Tests for the cosine noise schedule."""

import torch
import pytest

from src.models.diffusion.noise_schedule import CosineNoiseSchedule


@pytest.fixture
def schedule() -> CosineNoiseSchedule:
    return CosineNoiseSchedule(num_timesteps=1000, cosine_s=0.008)


class TestSNR:
    """Signal-to-noise ratio properties."""

    def test_snr_monotonically_decreasing(self, schedule: CosineNoiseSchedule) -> None:
        t = torch.arange(0, schedule.num_timesteps - 1)
        t_next = t + 1
        assert (schedule.snr(t) > schedule.snr(t_next)).all()

    def test_snr_high_at_t0(self, schedule: CosineNoiseSchedule) -> None:
        snr_0 = schedule.snr(torch.tensor([0])).item()
        assert snr_0 > 100, f"SNR at t=0 should be > 100, got {snr_0}"

    def test_snr_low_at_tmax(self, schedule: CosineNoiseSchedule) -> None:
        snr_last = schedule.snr(torch.tensor([schedule.num_timesteps - 1])).item()
        assert snr_last < 0.01, f"SNR at t=T-1 should be < 0.01, got {snr_last}"


class TestAlphasCumprod:
    """Properties of the cumulative product of alphas."""

    def test_alphas_cumprod_bounded(self, schedule: CosineNoiseSchedule) -> None:
        ac = schedule.alphas_cumprod
        assert (ac > 0).all(), "alphas_cumprod must be strictly positive"
        assert (ac < 1).all(), "alphas_cumprod must be strictly less than 1"
        # Verify clipping bounds (with small float32 tolerance)
        assert ac.min().item() >= 1e-5 - 1e-7
        assert ac.max().item() <= 1.0 - 1e-5 + 1e-7


class TestBuffers:
    """Registered buffer checks."""

    def test_buffer_count(self, schedule: CosineNoiseSchedule) -> None:
        expected_buffers = {
            "alphas_cumprod",
            "sqrt_alphas_cumprod",
            "sqrt_one_minus_alphas_cumprod",
            "log_snr",
        }
        actual_buffers = {name for name, _ in schedule.named_buffers()}
        assert expected_buffers.issubset(actual_buffers), (
            f"Missing buffers: {expected_buffers - actual_buffers}"
        )
