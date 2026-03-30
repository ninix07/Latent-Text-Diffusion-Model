"""Tests for AdaLN-Zero modulation block."""

import torch

from src.models.diffusion.adaln_block import AdaLNModulation, ada_layer_norm


def test_alpha_initialized_zero():
    """At init, alpha outputs (indices 2 and 5) should be approximately zero."""
    cond_dim = 32
    mod = AdaLNModulation(cond_dim, num_params=6)
    t_emb = torch.randn(4, cond_dim)
    params = mod(t_emb)  # 6 tensors of (B, 1, cond_dim)

    # alpha1 is params[2], alpha2 is params[5]
    for idx in [2, 5]:
        assert torch.allclose(params[idx], torch.zeros_like(params[idx]), atol=1e-5), (
            f"Alpha at index {idx} should be ~0 at init"
        )


def test_modulation_shapes():
    cond_dim = 32
    num_params = 6
    B = 4
    mod = AdaLNModulation(cond_dim, num_params=num_params)
    t_emb = torch.randn(B, cond_dim)
    params = mod(t_emb)
    assert len(params) == num_params
    for p in params:
        assert p.shape == (B, 1, cond_dim)


def test_ada_layer_norm_identity():
    """With gamma=0 and beta=0, AdaLN reduces to plain LayerNorm."""
    import torch.nn as nn

    B, S, D = 4, 10, 32
    ln = nn.LayerNorm(D)
    x = torch.randn(B, S, D)
    gamma = torch.zeros(B, 1, D)
    beta = torch.zeros(B, 1, D)
    out = ada_layer_norm(x, gamma, beta, ln)
    expected = ln(x)
    assert torch.allclose(out, expected, atol=1e-6)
