"""Logging setup for training runs."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("latent_diffusion")


def setup_logging(level: str = "INFO") -> None:
    """Configure basic logging format."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def log_metrics(step: int, metrics: dict[str, Any], prefix: str = "train") -> None:
    """Log metrics dict at a given step."""
    parts = [f"{prefix}/step={step}"]
    for key, value in metrics.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.4f}")
        else:
            parts.append(f"{key}={value}")
    logger.info(" | ".join(parts))


try:
    import wandb

    def init_wandb(config: dict, project: str = "latent-diffusion-text") -> None:
        """Initialize Weights & Biases run."""
        wandb.init(project=project, config=config)

    def log_wandb(metrics: dict[str, Any], step: int) -> None:
        """Log metrics to W&B."""
        wandb.log(metrics, step=step)

    def finish_wandb() -> None:
        """Finish the current W&B run."""
        wandb.finish()

except ImportError:

    def init_wandb(config: dict, project: str = "latent-diffusion-text") -> None:
        logger.warning("wandb not installed, skipping W&B init")

    def log_wandb(metrics: dict[str, Any], step: int) -> None:
        pass

    def finish_wandb() -> None:
        pass
