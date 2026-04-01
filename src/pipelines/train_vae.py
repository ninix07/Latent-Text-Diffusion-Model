"""Training pipeline for the Sequence VAE."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch

from src.config.schema import Config
from src.models.vae.vae import SequenceVAE
from src.models.vae.loss import compute_beta
from src.training.ema import EMAManager
from src.training.optimizer import create_optimizer, create_scheduler
from src.training.grad_utils import clip_gradients, accumulation_step
from src.training.checkpoint import save_checkpoint
from src.utils.logging import init_wandb, log_wandb, finish_wandb

logger = logging.getLogger(__name__)


def _validate(
    vae: SequenceVAE,
    val_loader,
    device: torch.device,
    beta: float = 1.0,
    free_bits: float = 0.0,
) -> dict[str, float]:
    """Run one pass over val_loader and return averaged metrics."""
    vae.eval()
    totals: dict[str, float] = {"total": 0.0, "recon": 0.0, "kl": 0.0}
    n_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            answer_ids = batch["answer_ids"].to(device)
            answer_mask = batch["answer_mask"].to(device)
            _, _, _, _, loss_dict = vae(
                answer_ids, answer_mask, beta=beta, free_bits=free_bits
            )
            for k in totals:
                totals[k] += loss_dict[k].item()
            n_batches += 1
    if n_batches == 0:
        return totals
    return {k: v / n_batches for k, v in totals.items()}


def train_vae(
    config: Config,
    device: Optional[torch.device] = None,
    train_loader=None,
    val_loader=None,
) -> dict[str, float]:
    """Train the SequenceVAE and return final validation metrics.

    Parameters
    ----------
    config : Config
    device : torch.device, optional
        Defaults to CUDA if available, else CPU.
    train_loader : DataLoader, optional
        If provided, skips building a dataloader from SQuAD (useful for tests).
    val_loader : DataLoader, optional
        If provided, skips building a dataloader from SQuAD (useful for tests).

    Returns
    -------
    dict
        Final val metrics: ``{total, recon, kl}``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------ data
    if train_loader is None or val_loader is None:
        from src.data.tokenization import create_tokenizer
        from src.data.loaders import create_squad_dataloaders

        tokenizer = create_tokenizer(config.encoder.model_name)
        train_loader, val_loader = create_squad_dataloaders(config, tokenizer)

    # ------------------------------------------------------------------ model
    from src.data.tokenization import create_tokenizer as _create_tok

    _tok = _create_tok(config.encoder.model_name)
    vocab_size = len(_tok)

    pretrained_emb = torch.randn(vocab_size, config.vae_arch.embed_dim) * 0.02
    pretrained_emb = pretrained_emb.to(device)

    vae = SequenceVAE(config.vae_arch, pretrained_embeddings=pretrained_emb).to(device)

    tc = config.vae_training
    optimizer = create_optimizer(
        vae.parameters(), lr=tc.learning_rate, weight_decay=tc.weight_decay
    )

    # Estimate total steps
    steps_per_epoch = max(len(train_loader), 1)
    total_steps = tc.epochs * steps_per_epoch
    scheduler = create_scheduler(optimizer, tc.warmup_steps, total_steps)

    ema = EMAManager(vae, decay=0.999, start_step=0)

    # ------------------------------------------------------------------ wandb
    init_wandb(config.to_dict(), project="latent-diffusion-text-vae")

    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0
    final_metrics: dict[str, float] = {}

    for epoch in range(tc.epochs):
        vae.train()
        for batch in train_loader:
            global_step += 1
            answer_ids = batch["answer_ids"].to(device)
            answer_mask = batch["answer_mask"].to(device)

            beta = compute_beta(
                global_step,
                start=tc.beta_start,
                end=tc.beta_end,
                warmup_steps=tc.beta_warmup_steps,
            )

            logits, z, mu, log_var, loss_dict = vae(
                answer_ids, answer_mask, beta=beta, free_bits=tc.free_bits
            )
            loss = loss_dict["total"] / tc.grad_accum_steps
            loss.backward()

            if accumulation_step(global_step, tc.grad_accum_steps):
                clip_gradients(vae, tc.grad_clip_max_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                ema.update(global_step)

            # ---- wandb: train metrics + latent vitals ----
            with torch.no_grad():
                std = torch.exp(0.5 * log_var)  # posterior σ
                # Per-dim variance of μ across batch+seq → dims with var > threshold are "active"
                mu_flat = mu.reshape(-1, mu.size(-1))  # (B*L, latent_dim)
                mu_dim_var = mu_flat.var(dim=0)  # (latent_dim,)
                active_dims = int((mu_dim_var > 0.01).sum().item())
                # Per-dim KL: -0.5*(1 + log_var - mu^2 - exp(log_var)), averaged over B*L
                kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
                kl_per_dim_mean = kl_per_dim.reshape(-1, kl_per_dim.size(-1)).mean(
                    dim=0
                )  # (latent_dim,)

            log_wandb(
                {
                    "train/loss": loss_dict["total"].item(),
                    "train/recon": loss_dict["recon"].item(),
                    "train/kl": loss_dict["kl"].item(),
                    "train/beta": beta,
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "latent/mu_mean": mu.mean().item(),
                    "latent/mu_std": mu.std().item(),
                    "latent/std_mean": std.mean().item(),
                    "latent/std_std": std.std().item(),
                    "latent/z_mean": z.mean().item(),
                    "latent/z_std": z.std().item(),
                    "latent/active_dims": active_dims,
                    "latent/kl_per_dim_max": kl_per_dim_mean.max().item(),
                    "latent/kl_per_dim_min": kl_per_dim_mean.min().item(),
                    "latent/kl_per_dim_std": kl_per_dim_mean.std().item(),
                    "epoch": epoch,
                },
                step=global_step,
            )

            # ---- validation ----
            if global_step % tc.val_every_n_steps == 0:
                val_metrics = _validate(
                    vae, val_loader, device, beta=beta, free_bits=tc.free_bits
                )
                logger.info(
                    "step=%d val_loss=%.4f recon=%.4f kl=%.4f",
                    global_step,
                    val_metrics["total"],
                    val_metrics["recon"],
                    val_metrics["kl"],
                )
                final_metrics = val_metrics
                log_wandb(
                    {
                        "val/loss": val_metrics["total"],
                        "val/recon": val_metrics["recon"],
                        "val/kl": val_metrics["kl"],
                    },
                    step=global_step,
                )

                if val_metrics["total"] < best_val_loss:
                    best_val_loss = val_metrics["total"]
                    patience_counter = 0
                    ckpt_path = (
                        Path(config.paths.checkpoint_dir) / "vae_best.pt"
                    )
                    save_checkpoint(
                        path=ckpt_path,
                        model=vae,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        ema=ema,
                        config=config,
                        step=global_step,
                        metrics=val_metrics,
                    )
                    logger.info("Saved VAE checkpoint: %s", ckpt_path)
                else:
                    patience_counter += 1

                if patience_counter >= tc.patience:
                    logger.info("Early stopping at step %d", global_step)
                    finish_wandb()
                    return final_metrics

                vae.train()

        # end-of-epoch validation
        val_metrics = _validate(
            vae, val_loader, device, beta=beta, free_bits=tc.free_bits
        )
        final_metrics = val_metrics
        logger.info("epoch=%d val_loss=%.4f", epoch, val_metrics["total"])
        log_wandb(
            {
                "val/loss": val_metrics["total"],
                "val/recon": val_metrics["recon"],
                "val/kl": val_metrics["kl"],
                "epoch": epoch,
            },
            step=global_step,
        )

        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            patience_counter = 0
            ckpt_path = Path(config.paths.checkpoint_dir) / "vae_best.pt"
            save_checkpoint(
                path=ckpt_path,
                model=vae,
                optimizer=optimizer,
                scheduler=scheduler,
                ema=ema,
                config=config,
                step=global_step,
                metrics=val_metrics,
            )
            logger.info("Saved VAE checkpoint: %s", ckpt_path)
        else:
            patience_counter += 1

        if patience_counter >= tc.patience:
            logger.info("Early stopping at epoch %d", epoch)
            break

    finish_wandb()
    return final_metrics


if __name__ == "__main__":
    from src.config.loader import create_config_from_cli

    logging.basicConfig(level=logging.INFO)
    cfg = create_config_from_cli()
    metrics = train_vae(cfg)
    print("Final metrics:", metrics)
