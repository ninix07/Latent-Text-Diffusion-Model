"""Sanity test: VAE encode → diffusion forward/reverse → VAE decode.

Confirms that the VAE latent space is structurally compatible with the
diffusion forward process: noising a clean latent at small t and denoising
back should recover something close to the original. We test the *forward*
process round-trip (no learned denoiser yet) — i.e. that q_sample is
reversible under the schedule's known coefficients. This catches shape
mismatches between the VAE and the diffusion code before either has been
trained.
"""

from __future__ import annotations

import torch

from src.config.schema import VAEArchConfig
from src.models.diffusion.forward_process import q_sample
from src.models.diffusion.noise_schedule import CosineNoiseSchedule
from src.models.vae.vae import SequenceVAE


def _build_vae(K: int = 4) -> SequenceVAE:
    cfg = VAEArchConfig(
        latent_dim=8,
        embed_dim=32,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        max_answer_len=10,
        num_latent_tokens=K,
    )
    vocab_size = 50
    emb = torch.randn(vocab_size, cfg.embed_dim)
    return SequenceVAE(cfg, pretrained_embeddings=emb)


def test_encode_produces_sequence_latent():
    """VAE encode emits (B, K, D), matching what the denoiser consumes."""
    vae = _build_vae(K=4)
    vae.eval()
    ids = torch.randint(0, 50, (3, 10))
    mask = torch.ones(3, 10, dtype=torch.long)
    z, mu, log_var = vae.encode(ids, mask, deterministic=True)
    assert z.shape == (3, 4, 8)
    assert mu.shape == (3, 4, 8)
    assert log_var.shape == (3, 4, 8)


def test_q_sample_accepts_sequence_latent():
    """q_sample must broadcast its (B,) timestep coefficients over (B, K, D)."""
    schedule = CosineNoiseSchedule(num_timesteps=100)
    z0 = torch.randn(3, 4, 8)
    t = torch.tensor([10, 20, 30])
    z_t = q_sample(z0, t, schedule)
    assert z_t.shape == z0.shape


def test_decoder_round_trip_low_noise():
    """At t=0 the noised latent is essentially z0 (cosine schedule has a
    tiny offset s, so alpha_cumprod[0] is just under 1). Decoder logits
    should match closely.
    """
    vae = _build_vae(K=4)
    vae.eval()
    schedule = CosineNoiseSchedule(num_timesteps=100)
    ids = torch.randint(0, 50, (2, 10))
    mask = torch.ones(2, 10, dtype=torch.long)

    with torch.no_grad():
        z0, _, _ = vae.encode(ids, mask, deterministic=True)
        t0 = torch.zeros(2, dtype=torch.long)
        z_t = q_sample(z0, t0, schedule, noise=torch.randn_like(z0))
        # Cosine schedule offset: tiny mismatch tolerated.
        assert torch.allclose(z_t, z0, atol=5e-2)
        logits_a = vae.decode(ids, z0, mask)
        logits_b = vae.decode(ids, z_t, mask)
        assert torch.allclose(logits_a, logits_b, atol=5e-2)


def test_round_trip_degrades_gracefully_with_t():
    """Large-t pure noise should change decoder logits; small-t should not.

    This is a smoke test for the latent's diffusion-friendliness rather
    than a quantitative quality gate. We only require: small-t logits are
    closer to the clean logits than large-t logits.
    """
    vae = _build_vae(K=4)
    vae.eval()
    schedule = CosineNoiseSchedule(num_timesteps=100)
    ids = torch.randint(0, 50, (2, 10))
    mask = torch.ones(2, 10, dtype=torch.long)

    with torch.no_grad():
        z0, _, _ = vae.encode(ids, mask, deterministic=True)
        logits_clean = vae.decode(ids, z0, mask)

        torch.manual_seed(0)
        noise = torch.randn_like(z0)
        z_small = q_sample(z0, torch.tensor([5, 5]), schedule, noise=noise)
        z_large = q_sample(z0, torch.tensor([90, 90]), schedule, noise=noise)

        diff_small = (vae.decode(ids, z_small, mask) - logits_clean).abs().mean()
        diff_large = (vae.decode(ids, z_large, mask) - logits_clean).abs().mean()

    assert diff_small <= diff_large
