"""Tests for VAEEncoder."""

import torch
import pytest

from src.models.vae.encoder import VAEEncoder

# Tiny dimensions for fast tests
EMBED_DIM = 64
LATENT_DIM = 16
NUM_LAYERS = 1
NUM_HEADS = 2
DROPOUT = 0.0
MAX_ANSWER_LEN = 10
VOCAB_SIZE = 100
BATCH_SIZE = 4


@pytest.fixture
def pretrained_emb():
    return torch.randn(VOCAB_SIZE, EMBED_DIM)


@pytest.fixture
def encoder(pretrained_emb):
    return VAEEncoder(
        embed_dim=EMBED_DIM,
        latent_dim=LATENT_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        max_answer_len=MAX_ANSWER_LEN,
        pretrained_embeddings=pretrained_emb,
    )


@pytest.fixture
def sample_batch():
    ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, MAX_ANSWER_LEN))
    mask = torch.ones(BATCH_SIZE, MAX_ANSWER_LEN, dtype=torch.long)
    return ids, mask


def test_output_shapes(encoder, sample_batch):
    ids, mask = sample_batch
    mu, log_var = encoder(ids, mask)
    assert mu.shape == (BATCH_SIZE, LATENT_DIM)
    assert log_var.shape == (BATCH_SIZE, LATENT_DIM)


def test_pooled_output(encoder, sample_batch):
    """Encoder produces a single pooled vector per sample, not per-position."""
    ids, mask = sample_batch
    mu, log_var = encoder(ids, mask)
    assert mu.dim() == 2
    assert log_var.dim() == 2


def test_pretrained_embed_init(pretrained_emb):
    enc = VAEEncoder(
        embed_dim=EMBED_DIM,
        latent_dim=LATENT_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        max_answer_len=MAX_ANSWER_LEN,
        pretrained_embeddings=pretrained_emb,
    )
    assert torch.allclose(enc.embedding.weight, pretrained_emb)
