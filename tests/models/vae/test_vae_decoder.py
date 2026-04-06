"""Tests for VAEDecoder (causal with latent prefix injection)."""

import torch
import pytest

from src.models.vae.decoder import VAEDecoder

EMBED_DIM = 64
LATENT_DIM = 16
NUM_LAYERS = 1
NUM_HEADS = 2
DROPOUT = 0.0
MAX_ANSWER_LEN = 10
VOCAB_SIZE = 100
NUM_LATENT_TOKENS = 2
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
        vocab_size=VOCAB_SIZE,
        num_latent_tokens=NUM_LATENT_TOKENS,
    )


@pytest.fixture
def sample_inputs():
    token_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, MAX_ANSWER_LEN))
    z = torch.randn(BATCH_SIZE, LATENT_DIM)
    mask = torch.ones(BATCH_SIZE, MAX_ANSWER_LEN, dtype=torch.long)
    return token_ids, z, mask


def test_output_shape(decoder, sample_inputs):
    token_ids, z, mask = sample_inputs
    out = decoder(token_ids, z, mask)
    assert out.shape == (BATCH_SIZE, MAX_ANSWER_LEN, EMBED_DIM)


def test_gradient_flows(decoder, sample_inputs):
    token_ids, z, mask = sample_inputs
    z = z.clone().requires_grad_(True)
    out = decoder(token_ids, z, mask)
    loss = out.sum()
    loss.backward()
    # Check that at least one decoder parameter has gradients
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in decoder.parameters()
    )
    assert has_grad


def test_causal_masking(decoder, sample_inputs):
    """Changing a future token should not affect earlier hidden states."""
    token_ids, z, mask = sample_inputs
    out1 = decoder(token_ids, z, mask)

    # Modify last token
    token_ids_mod = token_ids.clone()
    token_ids_mod[:, -1] = (token_ids[:, -1] + 1) % VOCAB_SIZE
    out2 = decoder(token_ids_mod, z, mask)

    # All positions except the last should be identical (causal masking)
    assert torch.allclose(out1[:, :-1, :], out2[:, :-1, :], atol=1e-5)
