"""Tests for reparameterize and kl_divergence."""

import torch
import pytest

from src.models.vae.reparameterize import reparameterize, kl_divergence

BATCH_SIZE = 4
SEQ_LEN = 10
LATENT_DIM = 16


def test_deterministic_returns_mu():
    mu = torch.randn(BATCH_SIZE, SEQ_LEN, LATENT_DIM)
    log_var = torch.randn(BATCH_SIZE, SEQ_LEN, LATENT_DIM)
    z = reparameterize(mu, log_var, deterministic=True)
    assert torch.allclose(z, mu)


def test_stochastic_differs():
    mu = torch.randn(BATCH_SIZE, SEQ_LEN, LATENT_DIM)
    log_var = torch.zeros(BATCH_SIZE, SEQ_LEN, LATENT_DIM)
    z1 = reparameterize(mu, log_var, deterministic=False)
    z2 = reparameterize(mu, log_var, deterministic=False)
    assert not torch.allclose(z1, z2)


def test_kl_zero_for_standard_normal():
    mu = torch.zeros(BATCH_SIZE, SEQ_LEN, LATENT_DIM)
    log_var = torch.zeros(BATCH_SIZE, SEQ_LEN, LATENT_DIM)
    kl = kl_divergence(mu, log_var)
    assert kl.item() == pytest.approx(0.0, abs=1e-5)


def test_kl_positive():
    mu = torch.randn(BATCH_SIZE, SEQ_LEN, LATENT_DIM)
    log_var = torch.randn(BATCH_SIZE, SEQ_LEN, LATENT_DIM)
    kl = kl_divergence(mu, log_var)
    assert kl.item() > 0


def test_free_bits_enforces_floor_on_collapsed_posterior():
    """When mu=0, log_var=0 (collapsed posterior), vanilla KL is 0.
    With free_bits > 0, each dimension is clamped to at least lambda nats,
    so the total KL must be >= free_bits * LATENT_DIM.
    """
    mu = torch.zeros(BATCH_SIZE, SEQ_LEN, LATENT_DIM)
    log_var = torch.zeros(BATCH_SIZE, SEQ_LEN, LATENT_DIM)

    kl_vanilla = kl_divergence(mu, log_var, free_bits=0.0)
    assert kl_vanilla.item() == pytest.approx(0.0, abs=1e-5)

    fb = 0.25
    kl_fb = kl_divergence(mu, log_var, free_bits=fb)
    assert kl_fb.item() >= fb * LATENT_DIM - 1e-5


def test_free_bits_does_not_reduce_large_kl():
    """Free bits should only raise low-KL dims; dims already above
    the threshold should keep their true KL value.

    The free-bits path uses a different aggregation (mean over B,L per dim
    then sum) than vanilla (sum per sample then mean over B), so we compare
    the clamped result against the *unclamped* per-dim aggregation directly.
    """
    mu = torch.randn(BATCH_SIZE, SEQ_LEN, LATENT_DIM) * 3.0
    log_var = torch.randn(BATCH_SIZE, SEQ_LEN, LATENT_DIM)

    kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
    kl_unclamped = kl_per_dim.mean(dim=[0, 1]).sum().item()

    kl_fb = kl_divergence(mu, log_var, free_bits=0.25)
    assert kl_fb.item() >= kl_unclamped - 1e-5
