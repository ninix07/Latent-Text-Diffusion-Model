"""Tests for LatentDataset."""

from __future__ import annotations

import torch
import pytest

from src.data.latent_dataset import LatentDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_latent_file(tmp_path, split: str, n: int = 8, seq_len: int = 10, latent_dim: int = 16):
    """Write a synthetic latent_dataset_{split}.pt to tmp_path."""
    samples = []
    for i in range(n):
        samples.append({
            "z_normalized": torch.randn(seq_len, latent_dim),
            "context_ids": torch.randint(0, 1000, (32,)),
            "context_mask": torch.ones(32, dtype=torch.long),
            "question_ids": torch.randint(0, 1000, (16,)),
            "question_mask": torch.ones(16, dtype=torch.long),
            "is_answerable": torch.tensor(float(i % 2)),
        })
    path = tmp_path / f"latent_dataset_{split}.pt"
    torch.save(samples, path)
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_loads_without_error(tmp_path):
    _make_latent_file(tmp_path, "train", n=8)
    ds = LatentDataset(str(tmp_path), "train")
    assert len(ds) > 0


def test_item_shapes(tmp_path):
    seq_len, latent_dim = 10, 16
    _make_latent_file(tmp_path, "train", n=4, seq_len=seq_len, latent_dim=latent_dim)
    ds = LatentDataset(str(tmp_path), "train")
    item = ds[0]
    assert item["z_normalized"].shape == (seq_len, latent_dim), (
        f"Expected ({seq_len}, {latent_dim}), got {item['z_normalized'].shape}"
    )


def test_metadata_present(tmp_path):
    _make_latent_file(tmp_path, "val", n=4)
    ds = LatentDataset(str(tmp_path), "val")
    item = ds[0]
    expected_keys = {
        "z_normalized",
        "context_ids",
        "context_mask",
        "question_ids",
        "question_mask",
        "is_answerable",
    }
    assert expected_keys.issubset(set(item.keys())), (
        f"Missing keys: {expected_keys - set(item.keys())}"
    )
