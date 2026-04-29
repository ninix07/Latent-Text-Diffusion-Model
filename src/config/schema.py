"""Frozen dataclass definitions for all configuration sections."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, asdict
from typing import Optional


@dataclass(frozen=True)
class PathConfig:
    """Filesystem paths for data, checkpoints, and outputs."""

    data_dir: str = "data"  # Root data directory
    checkpoint_dir: str = "checkpoints"  # Where to save model checkpoints
    latent_dir: str = "latents"  # Precomputed latent vectors
    output_dir: str = "outputs"  # Generation outputs and logs


@dataclass(frozen=True)
class EncoderConfig:
    """Pretrained encoder settings."""

    model_name: str = "bert-base-uncased"  # HuggingFace model identifier
    hidden_dim: int = 768  # Encoder hidden dimension
    max_context_len: int = 384  # Max context token length
    max_question_len: int = 64  # Max question token length
    unfreeze_top_n: int = 0  # Number of top layers to unfreeze


@dataclass(frozen=True)
class VAEArchConfig:
    """VAE architecture hyperparameters."""

    latent_dim: int = 128  # Latent space dimensionality
    embed_dim: int = 768  # Internal embedding dimension
    num_layers: int = 4  # Transformer layers in encoder/decoder
    num_heads: int = 8  # Attention heads
    dropout: float = 0.1  # Dropout rate
    max_answer_len: int = 50  # Max answer token length
    num_latent_tokens: int = 8  # Pseudo-tokens for latent KV injection


@dataclass(frozen=True)
class VAETrainingConfig:
    """VAE training hyperparameters."""

    learning_rate: float = 1e-4  # Peak learning rate
    batch_size: int = 64  # Training batch size
    epochs: int = 30  # Maximum training epochs
    patience: int = 5  # Early stopping patience (val checks)
    warmup_steps: int = 500  # LR scheduler warmup
    weight_decay: float = 0.01  # AdamW weight decay
    grad_clip_max_norm: float = 1.0  # Gradient clipping threshold
    grad_accum_steps: int = 1  # Gradient accumulation steps
    beta_start: float = 0.01  # KL weight at start
    beta_end: float = 1.0  # KL weight at end
    beta_warmup_steps: int = 10000  # Steps to ramp beta
    beta_schedule: str = "cyclical"  # "monotonic" or "cyclical"
    beta_cycles: int = 40  # Number of cycles (only used with "cyclical")
    target_kl: Optional[float] = (
        20.0  # KL ceiling — clamp KL contribution at this value (None = disabled)
    )
    beta_cycle_ratio: float = 0.5  # Fraction of cycle spent ramping
    free_bits: float = 0.01  # Min KL per latent dim (free bits)
    ema_decay: float = 0.999  # EMA decay rate for validation weights
    val_every_n_steps: int = 500  # Validation frequency (steps)
    noise_aug_sigma: float = 0.0  # Extra Gaussian noise std added to z before
    # decode (decoder noise robustness for diffusion-time latents)
    noise_aug_prob: float = 0.0  # Per-step probability of applying noise aug


@dataclass(frozen=True)
class QualityGateConfig:
    """Thresholds for latent quality gate checks."""

    min_recon_accuracy: float = 0.85  # Minimum token reconstruction accuracy
    min_mean_kl: float = 0.1  # Minimum mean KL divergence
    min_active_dims: int = 10  # Minimum active latent dimensions
    min_centroid_distance: float = 0.5  # Min L2 distance between ans/no-ans centroids
    active_dim_variance_threshold: float = 0.1  # Variance threshold for "active" dim
    max_dead_slots: int = 0  # Max collapsed latent slots (per-slot zero active dims)
    min_active_in_any_slot: int = 1  # Min active dims required in the weakest slot


@dataclass(frozen=True)
class DenoiserArchConfig:
    """Denoiser transformer architecture."""

    denoiser_dim: int = 512  # Internal model dimension
    num_layers: int = 6  # Transformer blocks
    num_heads: int = 8  # Attention heads
    ff_dim: int = 2048  # Feed-forward inner dimension (4x denoiser_dim)
    dropout: float = 0.1  # Dropout rate


@dataclass(frozen=True)
class NoiseScheduleConfig:
    """Noise schedule parameters."""

    num_timesteps: int = 1000  # Total diffusion timesteps T
    schedule_type: str = "cosine"  # Schedule type: "cosine" or "linear"
    cosine_s: float = 0.008  # Cosine schedule offset
    prediction_type: str = "epsilon"  # What the model predicts: "epsilon"


@dataclass(frozen=True)
class DiffusionTrainingConfig:
    """Diffusion model training hyperparameters."""

    learning_rate: float = 1e-4  # Peak learning rate
    batch_size: int = 128  # Training batch size
    epochs: int = 100  # Maximum training epochs
    warmup_steps: int = 1000  # LR scheduler warmup
    weight_decay: float = 0.01  # AdamW weight decay
    grad_clip_max_norm: float = 1.0  # Gradient clipping threshold
    grad_accum_steps: int = 1  # Gradient accumulation steps
    ema_decay: float = 0.9999  # EMA decay rate
    ema_start_step: int = 1000  # Step to start EMA updates
    cfg_dropout_rate: float = 0.1  # Classifier-free guidance dropout
    val_every_n_steps: int = 1000  # Validation frequency
    checkpoint_every_n_steps: int = 5000  # Checkpoint frequency


@dataclass(frozen=True)
class NullClassifierConfig:
    """Null answer classifier settings."""

    hidden_dim: int = 256  # MLP hidden dimension
    learning_rate: float = 1e-3  # Learning rate
    epochs: int = 20  # Training epochs
    batch_size: int = 256  # Batch size
    threshold: float = 0.5  # Default decision threshold


@dataclass(frozen=True)
class InferenceConfig:
    """Inference and sampling settings."""

    num_inference_steps: int = 50  # DDIM sampling steps
    guidance_scale: float = 3.0  # Classifier-free guidance weight
    eta: float = 0.0  # DDIM stochasticity (0=deterministic)
    best_of_n: int = 1  # Number of samples for best-of-N
    decoding_strategy: str = "greedy"  # "greedy" or "nucleus"
    nucleus_top_p: float = 0.9  # Top-p for nucleus sampling
    nucleus_temperature: float = 1.0  # Temperature for nucleus sampling


@dataclass(frozen=True)
class Config:
    """Top-level configuration combining all sections."""

    seed: int = 42  # Global random seed
    paths: PathConfig = field(default_factory=PathConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    vae_arch: VAEArchConfig = field(default_factory=VAEArchConfig)
    vae_training: VAETrainingConfig = field(default_factory=VAETrainingConfig)
    quality_gate: QualityGateConfig = field(default_factory=QualityGateConfig)
    denoiser_arch: DenoiserArchConfig = field(default_factory=DenoiserArchConfig)
    noise_schedule: NoiseScheduleConfig = field(default_factory=NoiseScheduleConfig)
    diffusion_training: DiffusionTrainingConfig = field(
        default_factory=DiffusionTrainingConfig
    )
    null_classifier: NullClassifierConfig = field(default_factory=NullClassifierConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Config:
        def _checked(dc_cls, data: dict) -> dict:
            valid = {f.name for f in fields(dc_cls)}
            unknown = set(data) - valid
            if unknown:
                raise ValueError(
                    f"Unknown keys for {dc_cls.__name__}: {sorted(unknown)}. "
                    f"Valid keys: {sorted(valid)}"
                )
            return data

        return cls(
            seed=d.get("seed", 42),
            paths=PathConfig(**_checked(PathConfig, d.get("paths", {}))),
            encoder=EncoderConfig(**_checked(EncoderConfig, d.get("encoder", {}))),
            vae_arch=VAEArchConfig(**_checked(VAEArchConfig, d.get("vae_arch", {}))),
            vae_training=VAETrainingConfig(
                **_checked(VAETrainingConfig, d.get("vae_training", {}))
            ),
            quality_gate=QualityGateConfig(
                **_checked(QualityGateConfig, d.get("quality_gate", {}))
            ),
            denoiser_arch=DenoiserArchConfig(
                **_checked(DenoiserArchConfig, d.get("denoiser_arch", {}))
            ),
            noise_schedule=NoiseScheduleConfig(
                **_checked(NoiseScheduleConfig, d.get("noise_schedule", {}))
            ),
            diffusion_training=DiffusionTrainingConfig(
                **_checked(DiffusionTrainingConfig, d.get("diffusion_training", {}))
            ),
            null_classifier=NullClassifierConfig(
                **_checked(NullClassifierConfig, d.get("null_classifier", {}))
            ),
            inference=InferenceConfig(
                **_checked(InferenceConfig, d.get("inference", {}))
            ),
        )
