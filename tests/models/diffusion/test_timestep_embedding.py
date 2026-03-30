"""Tests for sinusoidal timestep embeddings."""

import torch

from src.models.diffusion.timestep_embedding import SinusoidalTimestepEmbedding, TimestepMLP


def test_output_shape():
    dim = 32
    B = 4
    emb = SinusoidalTimestepEmbedding(dim)
    t = torch.randint(0, 1000, (B,))
    out = emb(t)
    assert out.shape == (B, dim)


def test_different_timesteps_differ():
    dim = 32
    emb = SinusoidalTimestepEmbedding(dim)
    out_10 = emb(torch.tensor([10]))
    out_500 = emb(torch.tensor([500]))
    assert not torch.allclose(out_10, out_500)


def test_same_timestep_same_output():
    dim = 32
    emb = SinusoidalTimestepEmbedding(dim)
    t = torch.tensor([42])
    out1 = emb(t)
    out2 = emb(t)
    assert torch.allclose(out1, out2)


def test_mlp_output_shape():
    B = 4
    sinusoidal_dim = 32
    output_dim = 64
    mlp = TimestepMLP(sinusoidal_dim, output_dim)
    t = torch.randint(0, 1000, (B,))
    out = mlp(t)
    assert out.shape == (B, output_dim)
