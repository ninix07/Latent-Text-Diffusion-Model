"""CLI entry point for evaluating the full generation pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch

from src.config.schema import Config
from src.evaluation.squad_metrics import compute_squad_metrics
from src.evaluation.null_metrics import null_confusion_matrix

logger = logging.getLogger(__name__)


def evaluate(
    config: Config,
    vae_checkpoint: str,
    diffusion_checkpoint: str,
    null_classifier_checkpoint: str,
    device: Optional[torch.device] = None,
    max_examples: int = 100,
) -> dict:
    """Load all models and evaluate on the val split.

    Returns a dict with EM, F1, and null prediction metrics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from src.models.encoder.frozen_encoder import FrozenEncoder
    from src.models.encoder.projection import ConditioningProjection
    from src.models.vae.vae import SequenceVAE
    from src.models.diffusion.denoiser import ConditionalDenoiser
    from src.models.diffusion.noise_schedule import CosineNoiseSchedule
    from src.models.sampler.ddim import DDIMSampler
    from src.models.sampler.cfg_sampler import CFGSampler
    from src.models.null_classifier import NullClassifier
    from src.data.tokenization import create_tokenizer, get_null_token_id
    from src.training.checkpoint import load_checkpoint
    from src.pipelines.generate import GenerationPipeline

    # Load models
    encoder = FrozenEncoder(config.encoder.model_name).to(device)
    projection = ConditioningProjection(
        encoder_dim=config.encoder.hidden_dim,
        denoiser_dim=config.denoiser_arch.denoiser_dim,
    ).to(device)

    vae_ckpt = load_checkpoint(vae_checkpoint)
    # create_tokenizer adds [NULL_ANS] — must use it so vocab_size matches the
    # checkpoint (bare AutoTokenizer gives 30522, training used 30523).
    _tokenizer_for_vae = create_tokenizer(config.encoder.model_name)
    _vocab_size = len(_tokenizer_for_vae)
    _pretrained_emb = torch.randn(_vocab_size, config.vae_arch.embed_dim) * 0.5
    vae = SequenceVAE(config.vae_arch, pretrained_embeddings=_pretrained_emb)
    vae.load_state_dict(vae_ckpt["model_state_dict"])
    vae.to(device)
    vae.eval()

    diff_ckpt = load_checkpoint(diffusion_checkpoint)
    denoiser = ConditionalDenoiser(
        latent_dim=config.vae_arch.latent_dim,
        denoiser_dim=config.denoiser_arch.denoiser_dim,
        num_layers=config.denoiser_arch.num_layers,
        num_heads=config.denoiser_arch.num_heads,
        ff_dim=config.denoiser_arch.ff_dim,
        max_seq_len=config.vae_arch.num_latent_tokens,
        dropout=config.denoiser_arch.dropout,
    )
    # Checkpoint was saved with combined = ModuleList([denoiser, projection]),
    # so keys are prefixed "0." for denoiser and "1." for projection.
    combined_state = diff_ckpt["model_state_dict"]
    denoiser.load_state_dict(
        {k[2:]: v for k, v in combined_state.items() if k.startswith("0.")}
    )
    projection.load_state_dict(
        {k[2:]: v for k, v in combined_state.items() if k.startswith("1.")}
    )
    denoiser.to(device)
    denoiser.eval()

    schedule = CosineNoiseSchedule(
        config.noise_schedule.num_timesteps,
        cosine_s=config.noise_schedule.cosine_s,
    ).to(device)
    ddim = DDIMSampler(schedule, config.inference.num_inference_steps, config.inference.eta)
    cfg_sampler = CFGSampler(denoiser, guidance_scale=config.inference.guidance_scale, ddim=ddim)

    null_ckpt = load_checkpoint(null_classifier_checkpoint)
    null_clf = NullClassifier(config.vae_arch.latent_dim, config.null_classifier.hidden_dim)
    null_clf.load_state_dict(null_ckpt["model_state_dict"])
    null_clf.to(device)
    null_clf.eval()

    norm_stats = torch.load(
        Path(config.paths.latent_dir) / "normalization_stats.pt",
        map_location="cpu",
        weights_only=True,
    )

    tokenizer = create_tokenizer(config.encoder.model_name)

    pipeline = GenerationPipeline(
        encoder=encoder,
        projection=projection,
        vae=vae,
        sampler=cfg_sampler,
        null_classifier=null_clf,
        normalization_stats=norm_stats,
        tokenizer=tokenizer,
        config=config,
    )

    # Load val data
    from src.data.loaders import create_squad_dataloaders
    _, val_loader = create_squad_dataloaders(config, tokenizer)

    predictions, references, pred_ans, true_ans = [], [], [], []
    for batch in val_loader:
        if len(predictions) >= max_examples:
            break
        results = pipeline.generate_batch(batch)
        for j, r in enumerate(results):
            predictions.append(r["answer_text"])
            references.append(batch["all_answer_texts"][j] if "all_answer_texts" in batch else [""])
            pred_ans.append(r["is_answerable"])
            true_ans.append(bool(batch["is_answerable"][j].item()))

    metrics = compute_squad_metrics(predictions, references)
    null_m = null_confusion_matrix(pred_ans, true_ans)
    metrics["null_f1"] = null_m["f1"]
    return metrics


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--vae_checkpoint", required=True)
    parser.add_argument("--diffusion_checkpoint", required=True)
    parser.add_argument("--null_classifier_checkpoint", required=True)
    args = parser.parse_args()
    if args.config is not None:
        import yaml
        with open(args.config) as f:
            cfg = Config.from_dict(yaml.safe_load(f))
    else:
        cfg = Config()
    result = evaluate(cfg, args.vae_checkpoint, args.diffusion_checkpoint,
                      args.null_classifier_checkpoint)
    print(result)
