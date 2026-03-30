"""Training pipeline for the NullClassifier.

Trains a lightweight MLP that classifies whether a latent z corresponds
to an answerable question.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.config.schema import Config
from src.models.null_classifier import NullClassifier
from src.training.checkpoint import save_checkpoint
from src.training.optimizer import create_optimizer, create_scheduler
from src.utils.logging import init_wandb, log_wandb, finish_wandb

logger = logging.getLogger(__name__)


# Sentinel – used to build a no-op EMA-like object for save_checkpoint
class _FakeEMA:
    def state_dict(self):
        return {}

    def load_state_dict(self, s):
        pass


def _evaluate(
    model: NullClassifier,
    val_loader,
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute binary accuracy and AUC on the validation set."""
    model.eval()
    all_probs: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in val_loader:
            z, labels = batch[0].to(device), batch[1].to(device)
            probs = model(z)
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    probs_t = torch.cat(all_probs)
    labels_t = torch.cat(all_labels)

    preds = (probs_t >= threshold).float()
    accuracy = float((preds == labels_t).float().mean().item())

    # Simple AUC via ranking
    try:
        pos_mask = labels_t == 1
        neg_mask = labels_t == 0
        if pos_mask.sum() > 0 and neg_mask.sum() > 0:
            pos_scores = probs_t[pos_mask]
            neg_scores = probs_t[neg_mask]
            # U-statistic
            auc = float(
                (pos_scores.unsqueeze(1) > neg_scores.unsqueeze(0))
                .float()
                .mean()
                .item()
            )
        else:
            auc = float("nan")
    except Exception:
        auc = float("nan")

    model.train()
    return {"accuracy": accuracy, "auc": auc, "threshold": threshold}


def train_null_classifier(
    config: Config,
    device: Optional[torch.device] = None,
    train_loader=None,
    val_loader=None,
) -> dict[str, float]:
    """Train the NullClassifier on precomputed latents.

    Parameters
    ----------
    config : Config
    device : torch.device, optional
    train_loader : DataLoader, optional
        If provided, skips loading data from disk (test mode).
    val_loader : DataLoader, optional

    Returns
    -------
    dict
        ``{accuracy, threshold, auc}``
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = config.null_classifier
    threshold = cfg.threshold

    # ------------------------------------------------------------------
    # Load data from disk only when no loaders are injected
    # ------------------------------------------------------------------
    if train_loader is None:
        from src.data.latent_dataset import LatentDataset

        train_ds = LatentDataset(config.paths.latent_dir, "train")
        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False
        )
        if val_loader is None:
            val_ds = LatentDataset(config.paths.latent_dir, "val")
            val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    model = NullClassifier(
        latent_dim=config.vae_arch.latent_dim,
        hidden_dim=cfg.hidden_dim,
    ).to(device)

    optimizer = create_optimizer(
        model.parameters(), lr=cfg.learning_rate, weight_decay=0.01
    )
    total_steps = cfg.epochs * len(train_loader)
    scheduler = create_scheduler(
        optimizer, warmup_steps=10, total_steps=max(total_steps, 1)
    )

    # ------------------------------------------------------------------
    # wandb
    # ------------------------------------------------------------------
    init_wandb(config.to_dict(), project="latent-diffusion-text-null-clf")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    model.train()
    step = 0
    metrics: dict[str, float] = {}

    for epoch in range(cfg.epochs):
        for batch in train_loader:
            step += 1

            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                # TensorDataset-style: (z, labels)
                z = batch[0].to(device)
                labels = batch[1].to(device).float()
            else:
                # LatentDataset dict style
                z = batch["z_normalized"].to(device)
                labels = batch["is_answerable"].to(device).float()

            probs = model(z)
            loss = F.binary_cross_entropy(probs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # ---- wandb: train metrics ----
            log_wandb(
                {
                    "train/bce_loss": loss.item(),
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                },
                step=step,
            )

        logger.info("epoch=%d  bce_loss=%.4f", epoch + 1, loss.item())

    # ------------------------------------------------------------------
    # Final evaluation
    # ------------------------------------------------------------------
    if val_loader is not None:
        metrics = _evaluate(model, val_loader, device, threshold=threshold)
        log_wandb(
            {
                "val/accuracy": metrics["accuracy"],
                "val/auc": metrics["auc"],
                "val/threshold": metrics["threshold"],
            },
            step=step,
        )
    else:
        metrics = {
            "accuracy": float("nan"),
            "auc": float("nan"),
            "threshold": threshold,
        }

    # ------------------------------------------------------------------
    # Save final checkpoint
    # ------------------------------------------------------------------
    ckpt_path = Path(config.paths.checkpoint_dir) / "null_classifier_final.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    fake_ema = _FakeEMA()
    save_checkpoint(
        path=ckpt_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=fake_ema,
        config=config,
        step=step,
        metrics=metrics,
    )
    logger.info("Saved null classifier checkpoint: %s", ckpt_path)

    finish_wandb()
    return metrics


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Train NullClassifier")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg = Config()
    result = train_null_classifier(cfg)
    print("Final metrics:", result)
