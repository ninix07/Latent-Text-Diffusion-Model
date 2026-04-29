"""Export precomputed latents from a trained VAE."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch

from src.config.schema import Config

logger = logging.getLogger(__name__)


def export_latents(
    config: Config,
    vae_checkpoint_path: str,
    device: Optional[torch.device] = None,
) -> None:
    """Encode the full SQuAD dataset with a frozen VAE and save latent files.

    Encoding is deterministic (uses mu only, no sampling).

    Steps
    -----
    1. Load frozen VAE from checkpoint.
    2. Encode train split, compute per-position per-dim normalisation stats.
    3. Encode val split.
    4. Run quality gate on the val split; raise RuntimeError if it fails.
    5. Save ``latent_dataset_train.pt``, ``latent_dataset_val.pt``,
       and ``normalization_stats.pt`` into ``config.paths.latent_dir``.

    Parameters
    ----------
    config : Config
    vae_checkpoint_path : str
        Path to a ``.pt`` checkpoint produced by ``train_vae``.
    device : torch.device, optional
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------ load VAE
    from src.models.vae.vae import SequenceVAE
    from src.training.checkpoint import load_checkpoint
    from src.config.schema import Config as CfgCls

    ckpt = load_checkpoint(vae_checkpoint_path)
    saved_cfg = CfgCls.from_dict(ckpt["config"])

    # ------------------------------------------------------------------ dataloaders
    from src.data.tokenization import create_tokenizer
    from src.data.loaders import create_squad_dataloaders

    # create_tokenizer adds [NULL_ANS] — must use it (not bare AutoTokenizer) so
    # vocab_size matches the checkpoint trained with the extended vocab.
    tokenizer = create_tokenizer(saved_cfg.encoder.model_name)
    vocab_size = len(tokenizer)

    # Placeholder embedding tensor — load_state_dict below overwrites it with
    # the trained embeddings from the checkpoint.
    placeholder_emb = torch.zeros(vocab_size, saved_cfg.vae_arch.embed_dim, device=device)

    vae = SequenceVAE(saved_cfg.vae_arch, pretrained_embeddings=placeholder_emb).to(
        device
    )
    vae.load_state_dict(ckpt["model_state_dict"])
    vae.eval()

    train_loader, val_loader = create_squad_dataloaders(saved_cfg, tokenizer)

    # ------------------------------------------------------------------ encode helper
    def _encode_split(loader) -> dict:
        latents_list, context_ids_list, context_mask_list = [], [], []
        question_ids_list, question_mask_list, is_ans_list = [], [], []

        with torch.no_grad():
            for batch in loader:
                answer_ids = batch["answer_ids"].to(device)
                answer_mask = batch["answer_mask"].to(device)

                # Deterministic encode: mu only
                _, mu, _ = vae.encode(answer_ids, answer_mask, deterministic=True)
                latents_list.append(mu.cpu())

                context_ids_list.append(batch["context_ids"])
                context_mask_list.append(batch["context_mask"])
                question_ids_list.append(batch["question_ids"])
                question_mask_list.append(batch["question_mask"])
                # is_answerable may be bool tensor or list
                is_ans = batch["is_answerable"]
                if not isinstance(is_ans, torch.Tensor):
                    is_ans = torch.tensor(is_ans, dtype=torch.bool)
                is_ans_list.append(is_ans.cpu())

        return {
            "latents_raw": torch.cat(latents_list, dim=0),
            "context_ids": torch.cat(context_ids_list, dim=0),
            "context_mask": torch.cat(context_mask_list, dim=0),
            "question_ids": torch.cat(question_ids_list, dim=0),
            "question_mask": torch.cat(question_mask_list, dim=0),
            "is_answerable": torch.cat(is_ans_list, dim=0),
        }

    logger.info("Encoding train split…")
    train_data = _encode_split(train_loader)

    logger.info("Encoding val split…")
    val_data = _encode_split(val_loader)

    # ------------------------------------------------------------------ normalisation stats
    # Computed from train split only: per-(K, D) mean/std so each latent
    # slot gets its own scale (analogous to SD's per-channel scaling).
    # latents shape: (N, K, D) — sequence of latent vectors per sentence
    train_latents = train_data["latents_raw"]
    norm_mean = train_latents.mean(dim=0)  # (K, D)
    norm_std = train_latents.std(dim=0).clamp(min=1e-6)  # (K, D)
    norm_stats = {"mean": norm_mean, "std": norm_std}

    # Apply normalisation and rename key to match LatentDataset expectation
    def _normalize(data: dict) -> dict:
        z_norm = (data["latents_raw"] - norm_mean) / norm_std
        result = {"z_normalized": z_norm}
        result.update({k: v for k, v in data.items() if k != "latents_raw"})
        return result

    train_data = _normalize(train_data)
    val_data = _normalize(val_data)

    # ------------------------------------------------------------------ quality gate
    from src.pipelines.quality_gate import run_quality_gate

    passed, report = run_quality_gate(vae, val_loader, saved_cfg, device)
    if not passed:
        failed = [k for k, v in report.items() if not v["passed"]]
        raise RuntimeError(f"Quality gate failed on checks: {failed}. Report: {report}")
    logger.info("Quality gate passed.")

    # ------------------------------------------------------------------ save
    out_dir = Path(config.paths.latent_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(train_data, out_dir / "latent_dataset_train.pt")
    torch.save(val_data, out_dir / "latent_dataset_val.pt")
    torch.save(norm_stats, out_dir / "normalization_stats.pt")
    logger.info("Saved latents to %s", out_dir)
