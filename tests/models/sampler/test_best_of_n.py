"""Tests for best_of_n_sample."""

from __future__ import annotations

import torch
import pytest

from src.models.sampler.best_of_n import best_of_n_sample

# Tiny dimensions
LATENT_DIM = 16
SEQ_LEN = 10
B = 4


def _make_generate_fn(tensor: torch.Tensor):
    """Return a generate_fn that always returns the given tensor."""
    def generate_fn() -> torch.Tensor:
        return tensor
    return generate_fn


def _make_counter_generate_fn(tensors: list[torch.Tensor]):
    """Return a generate_fn that cycles through the given tensors in order."""
    call_count = [0]

    def generate_fn() -> torch.Tensor:
        idx = call_count[0]
        call_count[0] += 1
        return tensors[idx]

    return generate_fn


# ------------------------------------------------------------------
def test_n1_returns_single() -> None:
    """n=1 should return the single generated sample."""
    torch.manual_seed(0)
    z0 = torch.randn(B, SEQ_LEN, LATENT_DIM)
    generate_fn = _make_generate_fn(z0)

    def classifier(z: torch.Tensor) -> torch.Tensor:
        return torch.ones(z.shape[0])

    best_z0, confidence = best_of_n_sample(generate_fn, n=1, null_classifier=classifier)

    assert best_z0.shape == (B, SEQ_LEN, LATENT_DIM)
    assert torch.allclose(best_z0, z0), "n=1 should return the single sample unchanged"
    assert confidence.shape == (B,)


def test_selects_highest_confidence() -> None:
    """best_of_n_sample should return the sample with the highest classifier score."""
    torch.manual_seed(1)

    # Two candidates: first has low confidence, second has high confidence
    z0_low = torch.zeros(B, SEQ_LEN, LATENT_DIM)    # all zeros -> score 0.1
    z0_high = torch.ones(B, SEQ_LEN, LATENT_DIM)    # all ones  -> score 0.9

    generate_fn = _make_counter_generate_fn([z0_low, z0_high])

    def mock_classifier(z: torch.Tensor) -> torch.Tensor:
        # Returns 0.1 for zero tensor, 0.9 for ones tensor
        mean_val = z.mean().item()
        if mean_val < 0.5:
            return torch.full((z.shape[0],), 0.1)
        else:
            return torch.full((z.shape[0],), 0.9)

    best_z0, confidence = best_of_n_sample(
        generate_fn, n=2, null_classifier=mock_classifier
    )

    assert torch.allclose(best_z0, z0_high), (
        "Should select the sample with the highest confidence score"
    )
    assert torch.allclose(confidence, torch.full((B,), 0.9)), (
        "Returned confidence should be the highest value"
    )
