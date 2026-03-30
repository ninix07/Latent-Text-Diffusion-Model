"""Tests for ConditioningProjection."""

import pytest
import torch

from src.models.encoder.projection import ConditioningProjection

ENCODER_DIM = 768
DENOISER_DIM = 512
B = 2
Q_LEN = 4
C_LEN = 6


@pytest.fixture
def projection():
    return ConditioningProjection(encoder_dim=ENCODER_DIM, denoiser_dim=DENOISER_DIM)


@pytest.fixture
def sample_inputs():
    h_q = torch.randn(B, Q_LEN, ENCODER_DIM)
    q_mask = torch.ones(B, Q_LEN, dtype=torch.long)
    h_c = torch.randn(B, C_LEN, ENCODER_DIM)
    c_mask = torch.ones(B, C_LEN, dtype=torch.long)
    return h_q, q_mask, h_c, c_mask


# --------------------------------------------------------------------- #
# Tests                                                                  #
# --------------------------------------------------------------------- #


def test_output_shape(projection, sample_inputs):
    """Conditioning tensor has shape (B, q_len + c_len, denoiser_dim)."""
    h_q, q_mask, h_c, c_mask = sample_inputs
    conditioning, _ = projection(h_q, q_mask, h_c, c_mask)

    assert conditioning.shape == (B, Q_LEN + C_LEN, DENOISER_DIM)


def test_mask_shape(projection, sample_inputs):
    """Conditioning mask has shape (B, q_len + c_len)."""
    h_q, q_mask, h_c, c_mask = sample_inputs
    _, conditioning_mask = projection(h_q, q_mask, h_c, c_mask)

    assert conditioning_mask.shape == (B, Q_LEN + C_LEN)


def test_segment_embeddings_differ(projection):
    """The two segment embeddings must not be identical."""
    seg0 = projection.segment_embedding(torch.tensor([0]))
    seg1 = projection.segment_embedding(torch.tensor([1]))

    assert not torch.equal(seg0, seg1), "Segment embeddings should differ"


def test_projection_is_trainable(projection):
    """All parameters of the projection module should be trainable."""
    for name, param in projection.named_parameters():
        assert param.requires_grad, f"{name} should be trainable"


def test_gradients_flow(projection, sample_inputs):
    """Backward from output should produce gradients on projection params."""
    h_q, q_mask, h_c, c_mask = sample_inputs
    conditioning, _ = projection(h_q, q_mask, h_c, c_mask)

    loss = conditioning.sum()
    loss.backward()

    for name, param in projection.named_parameters():
        assert param.grad is not None, f"{name} should have a gradient"
        assert param.grad.abs().sum() > 0, f"{name} gradient should be nonzero"
