"""DataLoader factory functions."""

from __future__ import annotations

from typing import Tuple

from torch.utils.data import DataLoader

from src.config.schema import Config
from src.data.tokenization import create_tokenizer
from src.data.squad_dataset import SQuADDataset
from src.data.sampler import create_balanced_sampler


def create_squad_dataloaders(
    config: Config,
    tokenizer,
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation DataLoaders for SQuAD v2.

    The training loader uses a balanced sampler; the validation loader
    iterates sequentially (no sampler).
    """
    train_ds = SQuADDataset(
        split="train",
        tokenizer=tokenizer,
        max_context_len=config.encoder.max_context_len,
        max_question_len=config.encoder.max_question_len,
        max_answer_len=config.vae_arch.max_answer_len,
    )
    val_ds = SQuADDataset(
        split="validation",
        tokenizer=tokenizer,
        max_context_len=config.encoder.max_context_len,
        max_question_len=config.encoder.max_question_len,
        max_answer_len=config.vae_arch.max_answer_len,
    )

    train_sampler = create_balanced_sampler(train_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.vae_training.batch_size,
        sampler=train_sampler,
        num_workers=0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.vae_training.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    return train_loader, val_loader


def create_latent_dataloaders(
    config: Config,
) -> Tuple[DataLoader, DataLoader]:
    """Placeholder for latent dataloaders (implemented in Branch 9).

    Will load precomputed latent vectors from ``config.paths.latent_dir``.
    """
    raise NotImplementedError(
        f"LatentDataset not yet implemented. "
        f"Latent dir: {config.paths.latent_dir}"
    )
