"""Tests for NullClassifier."""

from __future__ import annotations

import torch
import pytest

from src.models.null_classifier import NullClassifier

# Tiny dimensions
LATENT_DIM = 16
HIDDEN_DIM = 32
SEQ_LEN = 10
B = 4


@pytest.fixture
def classifier() -> NullClassifier:
    return NullClassifier(latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM)


@pytest.fixture
def z0() -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(B, SEQ_LEN, LATENT_DIM)


# ------------------------------------------------------------------
def test_predict_proba_range(classifier: NullClassifier, z0: torch.Tensor) -> None:
    """predict_proba() output values must lie in [0, 1]."""
    classifier.eval()
    with torch.no_grad():
        probs = classifier.predict_proba(z0)
    assert (probs >= 0.0).all() and (probs <= 1.0).all(), (
        f"Probabilities out of [0,1]: min={probs.min().item():.4f}, "
        f"max={probs.max().item():.4f}"
    )


def test_output_shape(classifier: NullClassifier, z0: torch.Tensor) -> None:
    """forward() must return shape (B,)."""
    classifier.eval()
    with torch.no_grad():
        logits = classifier(z0)
    assert logits.shape == (B,), f"Expected shape ({B},), got {logits.shape}"


def test_trainable(classifier: NullClassifier, z0: torch.Tensor) -> None:
    """A backward pass should update the model's parameters."""
    classifier.train()
    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01)

    # Capture parameter values before update
    params_before = [p.clone().detach() for p in classifier.parameters()]

    logits = classifier(z0)
    target = torch.ones(B)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    loss.backward()
    optimizer.step()

    params_after = list(classifier.parameters())

    # At least one parameter should have changed
    any_changed = any(
        not torch.allclose(pb, pa.detach())
        for pb, pa in zip(params_before, params_after)
    )
    assert any_changed, "No parameters were updated after backward pass"


def test_predict_returns_types(classifier: NullClassifier, z0: torch.Tensor) -> None:
    """predict() must return a (bool, float) tuple."""
    is_answerable, confidence = classifier.predict(z0)
    assert isinstance(is_answerable, bool), (
        f"Expected bool for is_answerable, got {type(is_answerable)}"
    )
    assert isinstance(confidence, float), (
        f"Expected float for confidence, got {type(confidence)}"
    )
    assert 0.0 <= confidence <= 1.0, f"Confidence {confidence} outside [0,1]"
