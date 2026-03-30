"""Tests for the ConditionalDenoiser."""

import torch
import torch.nn as nn

from src.models.diffusion.denoiser import ConditionalDenoiser


B = 4
SEQ_LEN = 10
LATENT_DIM = 16
DENOISER_DIM = 32
NUM_HEADS = 2
FF_DIM = 128
NUM_LAYERS = 1
COND_LEN = 8


def _make_denoiser():
    return ConditionalDenoiser(
        latent_dim=LATENT_DIM,
        denoiser_dim=DENOISER_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        max_seq_len=64,
        dropout=0.0,
    )


def test_output_shape():
    model = _make_denoiser()
    z_t = torch.randn(B, SEQ_LEN, LATENT_DIM)
    t = torch.randint(0, 100, (B,))
    cond = torch.randn(B, COND_LEN, DENOISER_DIM)
    out = model(z_t, t, cond)
    assert out.shape == (B, SEQ_LEN, LATENT_DIM)


def _break_zero_init(model):
    """Nudge alpha gates away from zero so attention paths are active."""
    for block in model.blocks:
        with torch.no_grad():
            nn.init.uniform_(block.modulation.linear.weight, -0.1, 0.1)
            nn.init.uniform_(block.modulation.linear.bias, -0.1, 0.1)


def test_timestep_sensitivity():
    model = _make_denoiser()
    _break_zero_init(model)
    model.eval()
    z_t = torch.randn(B, SEQ_LEN, LATENT_DIM)
    cond = torch.randn(B, COND_LEN, DENOISER_DIM)

    with torch.no_grad():
        out1 = model(z_t, torch.tensor([10] * B), cond)
        out2 = model(z_t, torch.tensor([500] * B), cond)
    assert not torch.allclose(out1, out2)


def test_conditioning_sensitivity():
    model = _make_denoiser()
    _break_zero_init(model)
    model.eval()
    z_t = torch.randn(B, SEQ_LEN, LATENT_DIM)
    t = torch.tensor([50] * B)

    with torch.no_grad():
        out1 = model(z_t, t, torch.randn(B, COND_LEN, DENOISER_DIM))
        out2 = model(z_t, t, torch.randn(B, COND_LEN, DENOISER_DIM))
    assert not torch.allclose(out1, out2)


def test_gradient_flow():
    model = _make_denoiser()
    z_t = torch.randn(B, SEQ_LEN, LATENT_DIM)
    t = torch.randint(0, 100, (B,))
    cond = torch.randn(B, COND_LEN, DENOISER_DIM)

    out = model(z_t, t, cond)
    loss = out.sum()
    loss.backward()

    # Check that at least some parameters received gradients
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients flowed to model parameters"
    assert any(g.abs().sum() > 0 for g in grads), "All gradients are zero"
