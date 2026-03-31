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
    tokenizer=None,
) -> dict[str, float]:
    """Run one pass over val_loader and return averaged metrics.

    When *tokenizer* is provided also computes reconstruction EM and F1:
    the VAE encodes each answer then decodes it; EM/F1 measure how faithfully
    the decoded text matches the original answer.
    """
    from src.evaluation.squad_metrics import compute_squad_metrics

    vae.eval()
    totals: dict[str, float] = {"total": 0.0, "recon": 0.0, "kl": 0.0}
    n_batches = 0
    all_preds: list[str] = []
    all_refs: list[list[str]] = []

    with torch.no_grad():
        for batch in val_loader:
            answer_ids = batch["answer_ids"].to(device)
            answer_mask = batch["answer_mask"].to(device)
            logits, _, _, _, loss_dict = vae(answer_ids, answer_mask, beta=beta)
            for k in totals:
                totals[k] += loss_dict[k].item()
            n_batches += 1

            if tokenizer is not None and "all_answer_texts" in batch:
                pred_ids = logits.argmax(dim=-1)  # (B, L)
                for i in range(pred_ids.size(0)):
                    # Truncate to real token positions so that tokens predicted
                    # at padding positions cannot corrupt the decoded string.
                    length = int(answer_mask[i].sum().item())
                    # skip_special_tokens removes [NULL_ANS] → empty string for
                    # unanswerable, which is the correct SQuAD prediction.
                    pred_text = tokenizer.decode(
                        pred_ids[i, :length].tolist(), skip_special_tokens=True
                    ).strip()
                    all_preds.append(pred_text)
                    all_refs.append(batch["all_answer_texts"][i])

    if n_batches == 0:
        return totals

    result = {k: v / n_batches for k, v in totals.items()}

    if tokenizer is not None and all_preds:
        squad = compute_squad_metrics(all_preds, all_refs)
        result["em"] = squad["em"]
        result["f1"] = squad["f1"]
        result["has_ans_em"] = squad["has_ans_em"]
        result["has_ans_f1"] = squad["has_ans_f1"]

    return result


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

    from src.config.seed import seed_everything
    seed_everything(config.seed)

    # ------------------------------------------------------------------ data
    from src.data.tokenization import create_tokenizer

    if train_loader is None or val_loader is None:
        from src.data.loaders import create_squad_dataloaders

        tokenizer = create_tokenizer(config.encoder.model_name)
        train_loader, val_loader = create_squad_dataloaders(config, tokenizer)
    else:
        tokenizer = create_tokenizer(config.encoder.model_name)

    # ------------------------------------------------------------------ model
    # Use the same tokenizer that the data loaders use so vocab_size includes
    # the [NULL_ANS] special token added by create_tokenizer.
    vocab_size = len(tokenizer)

    # Initialize embeddings at a scale comparable to sinusoidal PE (~0.7 std) so
    # that token identity is not drowned out by positional encoding at the input.
    pretrained_emb = torch.randn(vocab_size, config.vae_arch.embed_dim) * 0.5
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

        # Accumulators for epoch-averaged train metrics
        epoch_totals: dict[str, float] = {
            "loss": 0.0,
            "recon": 0.0,
            "kl": 0.0,
            "true_kl": 0.0,
            "grad_norm": 0.0,
            "mu_mean": 0.0,
            "mu_std": 0.0,
            "std_mean": 0.0,
            "std_std": 0.0,
            "z_mean": 0.0,
            "z_std": 0.0,
            "active_dims": 0.0,
            "kl_per_dim_max": 0.0,
            "kl_per_dim_min": 0.0,
            "kl_per_dim_std": 0.0,
        }
        epoch_steps = 0

        for batch in train_loader:
            global_step += 1
            epoch_steps += 1
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

            grad_norm = 0.0
            if accumulation_step(global_step, tc.grad_accum_steps):
                grad_norm = clip_gradients(vae, tc.grad_clip_max_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                ema.update(global_step)

            # Accumulate train metrics for epoch average
            with torch.no_grad():
                std = torch.exp(0.5 * log_var)
                mu_flat = mu.reshape(-1, mu.size(-1))
                active_dims = int((mu_flat.var(dim=0) > 0.01).sum().item())
                kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
                kl_per_dim_mean = kl_per_dim.reshape(-1, kl_per_dim.size(-1)).mean(
                    dim=0
                )
                # True KL (no free bits) — exposes posterior collapse
                true_kl = kl_per_dim_mean.sum().item()

            step_metrics = {
                "loss": loss_dict["total"].item(),
                "recon": loss_dict["recon"].item(),
                "kl": loss_dict["kl"].item(),
                "true_kl": true_kl,
                "grad_norm": grad_norm,
                "mu_mean": mu.mean().item(),
                "mu_std": mu.std().item(),
                "std_mean": std.mean().item(),
                "std_std": std.std().item(),
                "z_mean": z.mean().item(),
                "z_std": z.std().item(),
                "active_dims": active_dims,
                "kl_per_dim_max": kl_per_dim_mean.max().item(),
                "kl_per_dim_min": kl_per_dim_mean.min().item(),
                "kl_per_dim_std": kl_per_dim_mean.std().item(),
            }

            for k, v in step_metrics.items():
                epoch_totals[k] += v

            # ---- wandb: per-step live curves ----
            log_wandb(
                {
                    "train/loss": step_metrics["loss"],
                    "train/recon": step_metrics["recon"],
                    "train/kl": step_metrics["kl"],
                    "train/true_kl": step_metrics["true_kl"],
                    "train/grad_norm": step_metrics["grad_norm"],
                    "train/beta": beta,
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "latent/mu_mean": step_metrics["mu_mean"],
                    "latent/mu_std": step_metrics["mu_std"],
                    "latent/std_mean": step_metrics["std_mean"],
                    "latent/std_std": step_metrics["std_std"],
                    "latent/z_mean": step_metrics["z_mean"],
                    "latent/z_std": step_metrics["z_std"],
                    "latent/active_dims": step_metrics["active_dims"],
                    "latent/kl_per_dim_max": step_metrics["kl_per_dim_max"],
                    "latent/kl_per_dim_min": step_metrics["kl_per_dim_min"],
                    "latent/kl_per_dim_std": step_metrics["kl_per_dim_std"],
                    "epoch": epoch,
                },
                step=global_step,
            )

        # ---- end-of-epoch: log averaged train metrics + validate ----
        n = max(epoch_steps, 1)
        logger.info(
            "epoch=%d  train_loss=%.4f  recon=%.4f  kl=%.4f",
            epoch,
            epoch_totals["loss"] / n,
            epoch_totals["recon"] / n,
            epoch_totals["kl"] / n,
        )
        # Validate with EMA weights for a smoother, more stable estimate.
        # Use tc.beta_end so the val loss composition is fixed across all epochs
        # (early-epoch beta≈0 would make val loss look artificially good and cause
        # premature early stopping).
        ema.apply()
        val_metrics = _validate(vae, val_loader, device, beta=tc.beta_end, tokenizer=tokenizer)
        ema.restore()
        final_metrics = val_metrics
        logger.info(
            "epoch=%d  val_loss=%.4f  recon=%.4f  kl=%.4f  em=%.3f  f1=%.3f  has_ans_em=%.3f  has_ans_f1=%.3f",
            epoch,
            val_metrics["total"],
            val_metrics["recon"],
            val_metrics["kl"],
            val_metrics.get("em", float("nan")),
            val_metrics.get("f1", float("nan")),
            val_metrics.get("has_ans_em", float("nan")),
            val_metrics.get("has_ans_f1", float("nan")),
        )
        log_wandb(
            {
                "val/loss": val_metrics["total"],
                "val/recon": val_metrics["recon"],
                "val/kl": val_metrics["kl"],
                "val/em": val_metrics.get("em", float("nan")),
                "val/f1": val_metrics.get("f1", float("nan")),
                "val/has_ans_em": val_metrics.get("has_ans_em", float("nan")),
                "val/has_ans_f1": val_metrics.get("has_ans_f1", float("nan")),
                "epoch": epoch,
            },
            step=global_step,
        )

        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            patience_counter = 0
            ckpt_path = Path(config.paths.checkpoint_dir) / "vae_best.pt"
            # Save the EMA-smoothed model as the best checkpoint.
            ema.apply()
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
            ema.restore()
            logger.info("Saved VAE checkpoint (EMA weights): %s", ckpt_path)
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
