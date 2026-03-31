"""Training pipeline for the latent diffusion model.

Trains ConditionalDenoiser + ConditioningProjection with a frozen encoder.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config.schema import Config
from src.models.diffusion.denoiser import ConditionalDenoiser
from src.models.diffusion.noise_schedule import CosineNoiseSchedule
from src.models.diffusion.forward_process import q_sample
from src.models.diffusion.cfg import apply_cfg_dropout
from src.models.encoder.projection import ConditioningProjection
from src.training.ema import EMAManager
from src.training.optimizer import create_optimizer, create_scheduler
from src.training.grad_utils import clip_gradients, accumulation_step
from src.training.checkpoint import save_checkpoint
from src.utils.logging import init_wandb, log_wandb, finish_wandb

logger = logging.getLogger(__name__)


def _encode_batch(encoder, batch: dict, device: torch.device) -> tuple:
    """Encode question and context tokens; returns hidden states."""
    q_ids = batch["question_ids"].to(device)
    q_mask = batch["question_mask"].to(device)
    c_ids = batch["context_ids"].to(device)
    c_mask = batch["context_mask"].to(device)
    with torch.no_grad():
        h_q = encoder.encode(q_ids, q_mask)
        h_c = encoder.encode(c_ids, c_mask)
    return h_q, q_mask, h_c, c_mask


def _validate(
    denoiser: ConditionalDenoiser,
    projection: ConditioningProjection,
    encoder,
    schedule: CosineNoiseSchedule,
    val_loader,
    device: torch.device,
) -> dict[str, float]:
    """Compute average MSE loss over the validation set.

    Supports both real latent loaders (with encoder) and mock loaders that
    supply pre-built ``conditioning`` / ``conditioning_mask`` tensors.
    """
    denoiser.eval()
    projection.eval()
    total_loss = 0.0
    n = 0
    # Fixed-seed generator so validation MSE is deterministic across calls.
    rng = torch.Generator(device=device)
    rng.manual_seed(0)
    with torch.no_grad():
        for batch in val_loader:
            z0 = batch["z_normalized"].to(device)
            B = z0.size(0)

            if "conditioning" in batch and "conditioning_mask" in batch:
                conditioning = batch["conditioning"].to(device)
                cond_mask = batch["conditioning_mask"].to(device)
            elif encoder is not None:
                h_q, q_mask, h_c, c_mask = _encode_batch(encoder, batch, device)
                conditioning, cond_mask = projection(h_q, q_mask.bool(), h_c, c_mask.bool())
            else:
                continue  # no conditioning source available — skip batch

            t = torch.randint(
                0, schedule.num_timesteps, (B,), device=device, generator=rng
            )
            noise = torch.randn(z0.shape, device=device, generator=rng)
            z_t = q_sample(z0, t, schedule, noise)

            eps_pred = denoiser(z_t, t, conditioning, cond_mask)
            loss = F.mse_loss(eps_pred, noise)
            total_loss += loss.item()
            n += 1

    denoiser.train()
    projection.train()
    return {"val_mse": total_loss / max(n, 1)}


def train_diffusion(
    config: Config,
    device: Optional[torch.device] = None,
    train_loader=None,
    val_loader=None,
) -> dict[str, float]:
    """Train the diffusion denoiser.

    Parameters
    ----------
    config : Config
    device : torch.device, optional
    train_loader : DataLoader, optional
        If provided, skips loading data from disk (test / injection mode).
    val_loader : DataLoader, optional
        If provided, used for validation; otherwise skips validation.

    Returns
    -------
    dict
        Final validation metrics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from src.config.seed import seed_everything
    seed_everything(config.seed)

    cfg_d = config.diffusion_training
    cfg_ns = config.noise_schedule
    cfg_arch = config.denoiser_arch
    cfg_enc = config.encoder
    cfg_vae = config.vae_arch

    # ------------------------------------------------------------------
    # Load data from disk only when no loaders are injected
    # ------------------------------------------------------------------
    if train_loader is None:
        from src.data.latent_dataset import LatentDataset

        train_ds = LatentDataset(config.paths.latent_dir, "train")
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg_d.batch_size,
            shuffle=True,
            drop_last=True,
        )
        if val_loader is None:
            val_ds = LatentDataset(config.paths.latent_dir, "val")
            val_loader = DataLoader(
                val_ds,
                batch_size=cfg_d.batch_size,
                shuffle=False,
            )

    # ------------------------------------------------------------------
    # Build models
    # ------------------------------------------------------------------
    # Frozen encoder — only loaded when not in pure-test mode
    encoder = None
    if not _is_mock_loader(train_loader):
        from src.models.encoder.frozen_encoder import FrozenEncoder

        encoder = FrozenEncoder(
            model_name=cfg_enc.model_name,
            unfreeze_top_n=cfg_enc.unfreeze_top_n,
        ).to(device)
        encoder.eval()

    projection = ConditioningProjection(
        encoder_dim=cfg_enc.hidden_dim,
        denoiser_dim=cfg_arch.denoiser_dim,
    ).to(device)

    denoiser = ConditionalDenoiser(
        latent_dim=cfg_vae.latent_dim,
        denoiser_dim=cfg_arch.denoiser_dim,
        num_layers=cfg_arch.num_layers,
        num_heads=cfg_arch.num_heads,
        ff_dim=cfg_arch.ff_dim,
        max_seq_len=cfg_vae.max_answer_len + 1,
        dropout=cfg_arch.dropout,
    ).to(device)

    schedule = CosineNoiseSchedule(
        num_timesteps=cfg_ns.num_timesteps,
        cosine_s=cfg_ns.cosine_s,
    ).to(device)

    # ------------------------------------------------------------------
    # Optimizer & EMA
    # ------------------------------------------------------------------
    trainable_params = list(denoiser.parameters()) + list(projection.parameters())
    optimizer = create_optimizer(
        trainable_params, lr=cfg_d.learning_rate, weight_decay=cfg_d.weight_decay
    )

    # Combine denoiser + projection into a single module for EMA
    combined = torch.nn.ModuleList([denoiser, projection])
    ema = EMAManager(combined, decay=cfg_d.ema_decay, start_step=cfg_d.ema_start_step)

    total_steps = cfg_d.epochs * len(train_loader)
    scheduler = create_scheduler(
        optimizer, warmup_steps=cfg_d.warmup_steps, total_steps=max(total_steps, 1)
    )

    # ------------------------------------------------------------------
    # wandb
    # ------------------------------------------------------------------
    init_wandb(config.to_dict(), project="latent-diffusion-text-diffusion")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    step = 0
    metrics: dict[str, float] = {}

    denoiser.train()
    projection.train()

    for epoch in range(cfg_d.epochs):
        for batch in train_loader:
            step += 1

            z0 = batch["z_normalized"].to(device)
            B = z0.size(0)

            # Encode (or use pre-provided conditioning from mock loader)
            if "conditioning" in batch and "conditioning_mask" in batch:
                conditioning = batch["conditioning"].to(device)
                cond_mask = batch["conditioning_mask"].to(device)
            else:
                h_q, q_mask, h_c, c_mask = _encode_batch(encoder, batch, device)
                conditioning, cond_mask = projection(
                    h_q, q_mask.bool(), h_c, c_mask.bool()
                )

            # CFG dropout
            conditioning, cond_mask = apply_cfg_dropout(
                conditioning, cond_mask, cfg_d.cfg_dropout_rate
            )

            # Sample timestep and noise
            t = torch.randint(0, schedule.num_timesteps, (B,), device=device)
            noise = torch.randn_like(z0)
            z_t = q_sample(z0, t, schedule, noise)

            # Forward + loss
            eps_pred = denoiser(z_t, t, conditioning, cond_mask)
            loss = F.mse_loss(eps_pred, noise) / cfg_d.grad_accum_steps

            # Backprop
            loss.backward()
            if accumulation_step(step, cfg_d.grad_accum_steps):
                grad_norm = clip_gradients(combined, cfg_d.grad_clip_max_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                ema.update(step)
            else:
                grad_norm = 0.0

            # ---- wandb: train metrics ----
            log_wandb(
                {
                    "train/mse_loss": loss.item() * cfg_d.grad_accum_steps,
                    "train/grad_norm": (
                        grad_norm if isinstance(grad_norm, float) else float(grad_norm)
                    ),
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                },
                step=step,
            )

            # ----------------------------------------------------------
            # Validation
            # ----------------------------------------------------------
            if (
                step % cfg_d.val_every_n_steps == 0
                and val_loader is not None
            ):
                ema.apply()
                val_metrics = _validate(
                    denoiser, projection, encoder, schedule, val_loader, device
                )
                ema.restore()
                metrics.update(val_metrics)
                logger.info("step=%d  %s", step, val_metrics)
                log_wandb(
                    {
                        "val/mse": val_metrics["val_mse"],
                    },
                    step=step,
                )

            # ----------------------------------------------------------
            # Checkpoint
            # ----------------------------------------------------------
            if step % cfg_d.checkpoint_every_n_steps == 0:
                ckpt_path = (
                    Path(config.paths.checkpoint_dir) / f"diffusion_step_{step}.pt"
                )
                ema.apply()
                save_checkpoint(
                    path=ckpt_path,
                    model=combined,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    ema=ema,
                    config=config,
                    step=step,
                    metrics=metrics,
                )
                ema.restore()
                logger.info("Saved checkpoint (EMA weights): %s", ckpt_path)

    if not metrics:
        metrics = {"val_mse": float("nan")}
    finish_wandb()
    return metrics


def _is_mock_loader(loader) -> bool:
    """Return True if the loader's dataset contains pre-built conditioning."""
    try:
        sample = next(iter(loader.dataset))
        return "conditioning" in sample
    except Exception:
        return False


if __name__ == "__main__":
    from src.config.loader import create_config_from_cli

    logging.basicConfig(level=logging.INFO)
    cfg = create_config_from_cli()
    result = train_diffusion(cfg)
    print("Final metrics:", result)
