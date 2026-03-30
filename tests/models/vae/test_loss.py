"""Tests for VAE loss and beta schedule."""

import torch
import pytest

from src.models.vae.loss import compute_vae_loss, compute_beta

BATCH_SIZE = 4
SEQ_LEN = 10
VOCAB_SIZE = 100
LATENT_DIM = 16


def test_padding_ignored():
    """Changing logits/targets at padding positions should not affect loss."""
    torch.manual_seed(0)
    mu = torch.zeros(BATCH_SIZE, SEQ_LEN, LATENT_DIM)
    log_var = torch.zeros(BATCH_SIZE, SEQ_LEN, LATENT_DIM)

    # Mask: first 5 positions real, last 5 padding
    mask = torch.zeros(BATCH_SIZE, SEQ_LEN, dtype=torch.long)
    mask[:, :5] = 1

    logits = torch.randn(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
    target = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))

    total1, recon1, _ = compute_vae_loss(logits, target, mask, mu, log_var, beta=1.0)

    # Change padding positions
    logits2 = logits.clone()
    logits2[:, 5:, :] = torch.randn(BATCH_SIZE, 5, VOCAB_SIZE) * 100
    target2 = target.clone()
    target2[:, 5:] = 0

    total2, recon2, _ = compute_vae_loss(logits2, target2, mask, mu, log_var, beta=1.0)

    assert torch.allclose(recon1, recon2)


def test_beta_warmup_schedule():
    assert compute_beta(0, start=0.0, end=1.0, warmup_steps=100) == pytest.approx(0.0)
    assert compute_beta(50, start=0.0, end=1.0, warmup_steps=100) == pytest.approx(0.5)
    assert compute_beta(100, start=0.0, end=1.0, warmup_steps=100) == pytest.approx(1.0)
    assert compute_beta(200, start=0.0, end=1.0, warmup_steps=100) == pytest.approx(1.0)


def test_perfect_prediction_low_loss():
    """One-hot logits matching targets should give near-zero recon loss."""
    target = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    mask = torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.long)
    mu = torch.zeros(BATCH_SIZE, SEQ_LEN, LATENT_DIM)
    log_var = torch.zeros(BATCH_SIZE, SEQ_LEN, LATENT_DIM)

    # Create very confident logits
    logits = torch.full((BATCH_SIZE, SEQ_LEN, VOCAB_SIZE), -100.0)
    for b in range(BATCH_SIZE):
        for l in range(SEQ_LEN):
            logits[b, l, target[b, l]] = 100.0

    _, recon, _ = compute_vae_loss(logits, target, mask, mu, log_var, beta=1.0)
    assert recon.item() < 0.01
