"""Tests for VAEDecoder."""

import torch
import pytest

from src.models.vae.decoder import VAEDecoder

EMBED_DIM = 64
LATENT_DIM = 16
NUM_LAYERS = 1
NUM_HEADS = 2
DROPOUT = 0.0
MAX_ANSWER_LEN = 10
BATCH_SIZE = 4


@pytest.fixture
def decoder():
    return VAEDecoder(
        latent_dim=LATENT_DIM,
        embed_dim=EMBED_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        max_answer_len=MAX_ANSWER_LEN,
    )


@pytest.fixture
def sample_z():
    z = torch.randn(BATCH_SIZE, MAX_ANSWER_LEN, LATENT_DIM)
    mask = torch.ones(BATCH_SIZE, MAX_ANSWER_LEN, dtype=torch.long)
    return z, mask


def test_output_shape(decoder, sample_z):
    z, mask = sample_z
    out = decoder(z, mask)
    assert out.shape == (BATCH_SIZE, MAX_ANSWER_LEN, EMBED_DIM)


def test_gradient_flows(decoder, sample_z):
    z, mask = sample_z
    z.requires_grad_(True)
    out = decoder(z, mask)
    loss = out.sum()
    loss.backward()
    # Check that at least one decoder parameter has gradients
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in decoder.parameters())
    assert has_grad
