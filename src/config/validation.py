"""Cross-field validation for the Config object."""

from __future__ import annotations

from src.config.schema import Config


def validate_config(config: Config) -> None:
    """Validate cross-field constraints. Raises ValueError on failure."""
    # Denoiser ff_dim must be 4x denoiser_dim
    expected_ff = 4 * config.denoiser_arch.denoiser_dim
    if config.denoiser_arch.ff_dim != expected_ff:
        raise ValueError(
            f"ff_dim must be 4 * denoiser_dim ({expected_ff}), "
            f"got ff_dim={config.denoiser_arch.ff_dim}"
        )

    # Positive dimensions
    if config.vae_arch.latent_dim <= 0:
        raise ValueError(f"latent_dim must be > 0, got {config.vae_arch.latent_dim}")
    if config.vae_arch.max_answer_len <= 0:
        raise ValueError(f"max_answer_len must be > 0, got {config.vae_arch.max_answer_len}")

    # Schedule type
    valid_schedules = ("cosine", "linear")
    if config.noise_schedule.schedule_type not in valid_schedules:
        raise ValueError(
            f"schedule_type must be one of {valid_schedules}, "
            f"got '{config.noise_schedule.schedule_type}'"
        )

    # Prediction type
    valid_predictions = ("epsilon",)
    if config.noise_schedule.prediction_type not in valid_predictions:
        raise ValueError(
            f"prediction_type must be one of {valid_predictions}, "
            f"got '{config.noise_schedule.prediction_type}'"
        )

    # Beta schedule ordering
    if config.vae_training.beta_start >= config.vae_training.beta_end:
        raise ValueError(
            f"beta_start ({config.vae_training.beta_start}) must be < "
            f"beta_end ({config.vae_training.beta_end})"
        )

    # EMA decay range
    ema = config.diffusion_training.ema_decay
    if not (0.0 < ema < 1.0):
        raise ValueError(f"ema_decay must be in (0, 1), got {ema}")

    # Positive timesteps
    if config.noise_schedule.num_timesteps <= 0:
        raise ValueError(
            f"num_timesteps must be > 0, got {config.noise_schedule.num_timesteps}"
        )

    # Guidance scale non-negative
    if config.inference.guidance_scale < 0:
        raise ValueError(
            f"guidance_scale must be >= 0, got {config.inference.guidance_scale}"
        )
