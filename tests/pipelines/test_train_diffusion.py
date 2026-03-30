"""Tests for the diffusion training pipeline."""

from __future__ import annotations

import torch
import pytest
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import replace as _replace

from src.config.schema import Config, DiffusionTrainingConfig
from src.pipelines.train_diffusion import train_diffusion


def _make_diffusion_loader(config: Config, n_batches: int = 4) -> DataLoader:
    """Loader with pre-built conditioning (bypasses encoder)."""
    B = config.diffusion_training.batch_size
    L = config.vae_arch.max_answer_len
    D = config.vae_arch.latent_dim
    cond_len = config.encoder.max_context_len + config.encoder.max_question_len
    cond_dim = config.denoiser_arch.denoiser_dim

    items = []
    for _ in range(n_batches * B):
        items.append({
            "z_normalized": torch.randn(L, D),
            "context_ids": torch.zeros(config.encoder.max_context_len, dtype=torch.long),
            "context_mask": torch.ones(config.encoder.max_context_len, dtype=torch.long),
            "question_ids": torch.zeros(config.encoder.max_question_len, dtype=torch.long),
            "question_mask": torch.ones(config.encoder.max_question_len, dtype=torch.long),
            "is_answerable": torch.tensor(1.0),
            # Pre-built conditioning bypasses encoder
            "conditioning": torch.randn(cond_len, cond_dim),
            "conditioning_mask": torch.zeros(cond_len, dtype=torch.bool),
        })

    class _ListDataset(torch.utils.data.Dataset):
        def __len__(self): return len(items)
        def __getitem__(self, i): return items[i]

    def _collate(batch):
        return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}

    return DataLoader(_ListDataset(), batch_size=B, collate_fn=_collate)


def test_one_epoch_runs(tiny_config: Config):
    """One epoch should complete without error."""
    cfg = _replace(
        tiny_config,
        diffusion_training=_replace(
            tiny_config.diffusion_training,
            epochs=1,
            val_every_n_steps=1000,
            checkpoint_every_n_steps=1000,
        ),
    )
    loader = _make_diffusion_loader(cfg, n_batches=2)
    result = train_diffusion(cfg, device=torch.device("cpu"),
                             train_loader=loader, val_loader=None)
    assert isinstance(result, dict)


def test_loss_is_mse(tiny_config: Config):
    """Loss value should be a finite scalar."""
    cfg = _replace(
        tiny_config,
        diffusion_training=_replace(
            tiny_config.diffusion_training,
            epochs=1,
            val_every_n_steps=1000,
            checkpoint_every_n_steps=1000,
        ),
    )
    loader = _make_diffusion_loader(cfg, n_batches=2)

    losses = []
    import src.pipelines.train_diffusion as td_mod
    orig_mse = torch.nn.functional.mse_loss

    def _track_mse(pred, target, **kw):
        loss = orig_mse(pred, target, **kw)
        losses.append(loss.item())
        return loss

    import torch.nn.functional as F
    F.mse_loss = _track_mse
    try:
        train_diffusion(cfg, device=torch.device("cpu"),
                        train_loader=loader, val_loader=None)
    finally:
        F.mse_loss = orig_mse

    assert len(losses) > 0
    assert all(torch.isfinite(torch.tensor(l)) for l in losses)


def test_encoder_not_in_graph(tiny_config: Config):
    """Encoder parameters must not require gradients."""
    from src.models.encoder.frozen_encoder import FrozenEncoder
    enc = FrozenEncoder(tiny_config.encoder.model_name, unfreeze_top_n=0)
    for p in enc.parameters():
        assert not p.requires_grad
