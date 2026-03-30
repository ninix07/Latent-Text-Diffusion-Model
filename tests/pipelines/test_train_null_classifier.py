"""Tests for the null classifier training pipeline."""

from __future__ import annotations

import torch
import pytest
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import replace as _replace

from src.config.schema import Config, NullClassifierConfig
from src.pipelines.train_null_classifier import train_null_classifier


def _make_separable_loaders(config: Config):
    """Return train/val loaders with linearly separable latents."""
    latent_dim = config.vae_arch.latent_dim
    seq_len = config.vae_arch.max_answer_len
    n = 200

    # Answerable: centered at +2; unanswerable: centered at -2
    z_ans = torch.randn(n // 2, seq_len, latent_dim) + 2.0
    z_unans = torch.randn(n // 2, seq_len, latent_dim) - 2.0
    z = torch.cat([z_ans, z_unans], dim=0)
    labels = torch.cat([torch.ones(n // 2), torch.zeros(n // 2)])

    idx = torch.randperm(n)
    z, labels = z[idx], labels[idx]

    split = int(0.8 * n)
    train_ds = TensorDataset(z[:split], labels[:split])
    val_ds = TensorDataset(z[split:], labels[split:])

    bs = config.null_classifier.batch_size
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)
    return train_loader, val_loader


def test_trains_to_convergence(tiny_config: Config):
    """With separable data, accuracy should exceed 0.7."""
    cfg = _replace(
        tiny_config,
        null_classifier=_replace(
            tiny_config.null_classifier,
            epochs=10,
            learning_rate=1e-2,
        ),
    )
    train_loader, val_loader = _make_separable_loaders(cfg)
    result = train_null_classifier(cfg, device=torch.device("cpu"),
                                   train_loader=train_loader,
                                   val_loader=val_loader)
    assert result["accuracy"] > 0.7, f"Expected >0.7 accuracy, got {result['accuracy']:.3f}"


def test_checkpoint_saved(tiny_config: Config, tmp_path):
    """A checkpoint file should be created after training."""
    from pathlib import Path
    from src.config.schema import PathConfig

    cfg = _replace(tiny_config, paths=PathConfig(checkpoint_dir=str(tmp_path)))
    train_loader, val_loader = _make_separable_loaders(cfg)
    train_null_classifier(cfg, device=torch.device("cpu"),
                          train_loader=train_loader,
                          val_loader=val_loader)
    ckpt = tmp_path / "null_classifier_final.pt"
    assert ckpt.exists(), f"Checkpoint not found at {ckpt}"
