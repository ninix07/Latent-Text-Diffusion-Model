"""DataLoader factory functions."""

from __future__ import annotations

from typing import Tuple

from torch.utils.data import DataLoader, default_collate

from src.config.schema import Config
from src.data.tokenization import create_tokenizer
from src.data.squad_dataset import SQuADDataset
from src.data.sampler import create_balanced_sampler

# Fields that contain plain strings or variable-length lists of strings —
# default_collate cannot handle these (it requires equal-length sequences).
_SQUAD_STR_KEYS = {"answer_text", "all_answer_texts"}


def _squad_collate(batch):
    """Collate SQuAD samples, keeping string fields as plain Python lists."""
    str_batch = {k: [sample[k] for sample in batch] for k in _SQUAD_STR_KEYS if k in batch[0]}
    tensor_batch = default_collate(
        [{k: v for k, v in sample.items() if k not in _SQUAD_STR_KEYS} for sample in batch]
    )
    tensor_batch.update(str_batch)
    return tensor_batch


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
        collate_fn=_squad_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.vae_training.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=_squad_collate,
    )
    return train_loader, val_loader


def create_latent_dataloaders(
    config: Config,
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation DataLoaders from precomputed latent files.

    Loads from ``config.paths.latent_dir``.
    """
    from src.data.latent_dataset import LatentDataset

    train_ds = LatentDataset(config.paths.latent_dir, "train")
    val_ds = LatentDataset(config.paths.latent_dir, "val")

    train_loader = DataLoader(
        train_ds,
        batch_size=config.diffusion_training.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.diffusion_training.batch_size,
        shuffle=False,
    )
    return train_loader, val_loader
