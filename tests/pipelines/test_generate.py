"""Tests for the GenerationPipeline."""

from __future__ import annotations

import torch
import pytest
from dataclasses import replace as _replace

from src.config.schema import Config, VAEArchConfig, DenoiserArchConfig, InferenceConfig
from src.pipelines.generate import GenerationPipeline


# ---------------------------------------------------------------------------
# Minimal mock components
# ---------------------------------------------------------------------------

class _MockEncoder:
    """Frozen encoder that returns random hidden states."""
    def parameters(self):
        return iter([torch.zeros(1)])
    def encode(self, ids, mask):
        B, L = ids.shape
        return torch.randn(B, L, 64)


class _MockProjection(torch.nn.Module):
    def __init__(self, out_dim=32):
        super().__init__()
        self.out_dim = out_dim
    def forward(self, h_q, q_mask, h_c, c_mask):
        B = h_q.size(0)
        L = h_q.size(1) + h_c.size(1)
        return torch.randn(B, L, self.out_dim), torch.zeros(B, L, dtype=torch.bool)


class _MockVAE(torch.nn.Module):
    def __init__(self, max_answer_len=10, vocab_size=100):
        super().__init__()
        self.max_answer_len = max_answer_len
        self.vocab_size = vocab_size
    def decode_to_tokens(self, z, mask, strategy="greedy", **kw):
        B, L, D = z.shape
        return torch.randint(0, self.vocab_size, (B, L))


class _MockDenoiser(torch.nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
    def forward(self, z_t, t, cond, cond_mask):
        return torch.randn_like(z_t)


class _MockDDIM:
    def sample(self, denoiser_fn, z_shape, device):
        return torch.randn(*z_shape)


class _MockSampler:
    def __init__(self):
        self.ddim = _MockDDIM()
        self.denoiser = _MockDenoiser()


class _MockNullClassifier(torch.nn.Module):
    def forward(self, z):
        return torch.ones(z.size(0)) * 0.9


class _MockTokenizer:
    pad_token_id = 0
    eos_token_id = 102

    def decode(self, ids, skip_special_tokens=True):
        return "mock answer"


def _make_pipeline(tiny_config: Config) -> GenerationPipeline:
    norm_stats = {
        "mean": torch.zeros(1, 1, tiny_config.vae_arch.latent_dim),
        "std": torch.ones(1, 1, tiny_config.vae_arch.latent_dim),
    }
    return GenerationPipeline(
        encoder=_MockEncoder(),
        projection=_MockProjection(out_dim=tiny_config.denoiser_arch.denoiser_dim),
        vae=_MockVAE(max_answer_len=tiny_config.vae_arch.max_answer_len),
        sampler=_MockSampler(),
        null_classifier=_MockNullClassifier(),
        normalization_stats=norm_stats,
        tokenizer=_MockTokenizer(),
        config=tiny_config,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_single_generation(tiny_config: Config):
    """generate() should return a list containing a dict with expected keys."""
    pipeline = _make_pipeline(tiny_config)
    L, C, Q = 10, tiny_config.encoder.max_context_len, tiny_config.encoder.max_question_len

    results = pipeline.generate(
        context_ids=torch.zeros(1, C, dtype=torch.long),
        context_mask=torch.ones(1, C, dtype=torch.long),
        question_ids=torch.zeros(1, Q, dtype=torch.long),
        question_mask=torch.ones(1, Q, dtype=torch.long),
    )
    assert isinstance(results, list)
    assert len(results) == 1
    result = results[0]
    assert isinstance(result, dict)
    assert "answer_text" in result
    assert "is_answerable" in result
    assert "confidence" in result
    assert isinstance(result["answer_text"], str)
    assert isinstance(result["is_answerable"], bool)
    assert isinstance(result["confidence"], float)


def test_batch_generation(tiny_config: Config):
    """generate_batch() should return a list of correct length."""
    pipeline = _make_pipeline(tiny_config)
    B = 3
    C, Q = tiny_config.encoder.max_context_len, tiny_config.encoder.max_question_len

    batch = {
        "context_ids": torch.zeros(B, C, dtype=torch.long),
        "context_mask": torch.ones(B, C, dtype=torch.long),
        "question_ids": torch.zeros(B, Q, dtype=torch.long),
        "question_mask": torch.ones(B, Q, dtype=torch.long),
    }
    results = pipeline.generate_batch(batch)
    assert isinstance(results, list)
    assert len(results) == B
    for r in results:
        assert "answer_text" in r
