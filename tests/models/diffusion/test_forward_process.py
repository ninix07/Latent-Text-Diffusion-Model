"""Tests for the forward diffusion process (q_sample)."""

import torch
import pytest

from src.models.diffusion.noise_schedule import CosineNoiseSchedule
from src.models.diffusion.forward_process import q_sample


@pytest.fixture
def schedule() -> CosineNoiseSchedule:
    return CosineNoiseSchedule(num_timesteps=1000, cosine_s=0.008)


@pytest.fixture
def z0() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(8, 64)


class TestQSample:

    def test_t0_preserves_signal(
        self, schedule: CosineNoiseSchedule, z0: torch.Tensor
    ) -> None:
        """At t=0, alphas_cumprod ~ 1 so z_t should be very close to z0."""
        t = torch.zeros(z0.shape[0], dtype=torch.long)
        z_t = q_sample(z0, t, schedule)
        # sqrt_alpha_cumprod[0] is very close to 1; the noise contribution is tiny
        # Use a tolerance that accounts for the small noise added
        sqrt_alpha_0 = schedule.sqrt_alphas_cumprod[0].item()
        assert sqrt_alpha_0 > 0.999, f"sqrt_alpha at t=0 should be ~1, got {sqrt_alpha_0}"
        # With random noise, z_t ~ sqrt_alpha*z0 + sqrt(1-alpha)*eps
        # Check that the dominant component is z0
        residual = (z_t - sqrt_alpha_0 * z0).abs().mean().item()
        assert residual < 0.15, f"Residual at t=0 too large: {residual}"

    def test_tmax_is_noise(
        self, schedule: CosineNoiseSchedule, z0: torch.Tensor
    ) -> None:
        """At t=T-1, alphas_cumprod ~ 0 so z_t should be approximately pure noise."""
        t = torch.full((z0.shape[0],), schedule.num_timesteps - 1, dtype=torch.long)
        z_t = q_sample(z0, t, schedule)
        assert z_t.mean().abs().item() < 0.5, "Mean of z_t at T-1 should be near 0"
        assert 0.5 < z_t.std().item() < 1.5, "Std of z_t at T-1 should be near 1"

    def test_known_noise_recovery(
        self, schedule: CosineNoiseSchedule, z0: torch.Tensor
    ) -> None:
        """Given known noise, we can recover it from z_t."""
        torch.manual_seed(42)
        noise = torch.randn_like(z0)
        t = torch.full((z0.shape[0],), 500, dtype=torch.long)

        z_t = q_sample(z0, t, schedule, noise=noise)

        sqrt_alpha = schedule.sqrt_alphas_cumprod[500]
        sqrt_one_minus_alpha = schedule.sqrt_one_minus_alphas_cumprod[500]
        recovered_noise = (z_t - sqrt_alpha * z0) / sqrt_one_minus_alpha

        assert torch.allclose(recovered_noise, noise, atol=1e-5), (
            "Should recover the exact noise from z_t"
        )

    def test_batched_timesteps(
        self, schedule: CosineNoiseSchedule, z0: torch.Tensor
    ) -> None:
        """Each sample in the batch can have a different timestep."""
        torch.manual_seed(7)
        noise = torch.randn_like(z0)
        t = torch.tensor([0, 100, 200, 300, 500, 700, 900, 999], dtype=torch.long)

        z_t = q_sample(z0, t, schedule, noise=noise)

        # Verify each sample individually
        for i in range(z0.shape[0]):
            ti = t[i]
            expected = (
                schedule.sqrt_alphas_cumprod[ti] * z0[i]
                + schedule.sqrt_one_minus_alphas_cumprod[ti] * noise[i]
            )
            assert torch.allclose(z_t[i], expected, atol=1e-5), (
                f"Mismatch at sample {i}, t={ti.item()}"
            )

    def test_reproducible_with_seed(
        self, schedule: CosineNoiseSchedule, z0: torch.Tensor
    ) -> None:
        """Same random seed produces identical z_t."""
        t = torch.full((z0.shape[0],), 250, dtype=torch.long)

        torch.manual_seed(123)
        z_t_a = q_sample(z0, t, schedule)

        torch.manual_seed(123)
        z_t_b = q_sample(z0, t, schedule)

        assert torch.equal(z_t_a, z_t_b), "Same seed should produce identical results"
