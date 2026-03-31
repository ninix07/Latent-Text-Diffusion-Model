"""Logging setup for training runs."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("latent_diffusion")

# Set to False if wandb is not installed or fails to initialise at runtime.
_wandb_ok: bool = False

try:
    import wandb as _wandb
    _wandb_available = True
except ImportError:
    _wandb = None  # type: ignore[assignment]
    _wandb_available = False


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


def init_wandb(config: dict, project: str = "latent-diffusion-text") -> None:
    """Initialise a W&B run. Degrades gracefully if wandb is unavailable or auth fails."""
    global _wandb_ok
    if not _wandb_available:
        logger.warning("wandb is not installed — run `uv add wandb` to enable W&B logging.")
        _wandb_ok = False
        return
    try:
        _wandb.init(project=project, config=config, reinit=True)
        # Register global_step as the x-axis so it doesn't conflict with
        # wandb's internal auto-increment counter used by system metrics.
        _wandb.define_metric("global_step")
        _wandb.define_metric("*", step_metric="global_step")
        _wandb_ok = True
        logger.info("wandb initialised (project=%s, run=%s)", project, _wandb.run.name)
    except Exception as exc:
        logger.warning("wandb.init failed (%s) — W&B logging disabled for this run.", exc)
        _wandb_ok = False


def log_wandb(metrics: dict[str, Any], step: int) -> None:
    """Log metrics to W&B. No-op if wandb is not active."""
    if not _wandb_ok:
        return
    try:
        # Include global_step as a metric (not the step= kwarg) so it serves
        # as the x-axis without conflicting with the system monitor's counter.
        _wandb.log({"global_step": step, **metrics})
    except Exception as exc:
        logger.warning("wandb.log failed at step %d: %s", step, exc)


def finish_wandb() -> None:
    """Finish the current W&B run."""
    global _wandb_ok
    if not _wandb_ok:
        return
    try:
        _wandb.finish()
    except Exception as exc:
        logger.warning("wandb.finish failed: %s", exc)
    finally:
        _wandb_ok = False
