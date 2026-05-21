"""Training pipeline for LangVAE (replaces Stage 1 for the LangVAE path)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

from src.config.schema import Config
from src.models.vae.langvae_adapter import LangVAEAdapter
from src.utils.logging import init_wandb, log_wandb, finish_wandb
from pythae.trainers.training_callbacks import TrainingCallback

logger = logging.getLogger(__name__)


class WandbCallback(TrainingCallback):
    """Callback to log training metrics to Weights & Biases."""

    def on_log(self, training_config, logs, **kwargs):
        """Called after each epoch with loss metrics."""
        epoch = logs.get("epoch", None)
        train_loss = logs.get("train_epoch_loss", None)
        eval_loss = logs.get("eval_epoch_loss", None)

        if epoch is not None:
            metrics = {}
            if train_loss is not None:
                metrics["loss/train"] = train_loss
            if eval_loss is not None:
                metrics["loss/eval"] = eval_loss

            if metrics:  # Only log if we have metrics
                log_wandb(metrics, step=epoch)

            # Log with safe formatting for None values
            log_msg = f"Epoch {epoch}"
            if train_loss is not None:
                log_msg += f" - train_loss: {train_loss:.6f}"
            if eval_loss is not None:
                log_msg += f", eval_loss: {eval_loss:.6f}"
            logger.info(log_msg)


def _collect_answer_texts(squad_split: str) -> list[str]:
    """Load SQuAD v2 and return a flat list of answer strings."""
    from datasets import load_dataset
    from src.data.tokenization import NULL_TOKEN

    ds = load_dataset("squad_v2", split=squad_split)
    texts = []
    for example in ds:
        answers = example["answers"]["text"]
        texts.append(answers[0] if answers else NULL_TOKEN)
    return texts


def train_langvae(
    config: Config,
    device: Optional[torch.device] = None,
) -> LangVAEAdapter:
    """Train a LangVAE on SQuAD answer texts and return a LangVAEAdapter.

    The trained model is saved to ``config.langvae.checkpoint_dir``.

    Parameters
    ----------
    config : Config
    device : torch.device, optional
        Defaults to CUDA if available, else CPU.

    Returns
    -------
    LangVAEAdapter
        Wrapping the trained model.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from src.config.seed import seed_everything
    seed_everything(config.seed)

    lc = config.langvae

    # ------------------------------------------------------------------ imports
    from langvae import LangVAE
    from langvae.encoders import SentenceEncoder
    from langvae.decoders import SentenceDecoder
    from pythae.models.vae import VAEConfig
    from langvae.trainers import CyclicalScheduleKLThresholdTrainerConfig
    from langvae.pipelines import LanguageTrainingPipeline
    from transformers import AutoTokenizer

    # transformers>=4.36 returns DynamicCache instead of tuple-of-tuples for
    # past_key_values; langvae uses pkv[layer][k_or_v] which requires subscript.
    # Different transformers versions expose layers differently; handle each.
    try:
        from transformers.cache_utils import DynamicCache

        def _dyncache_layer_kv(self, idx):
            # 4.55+: .layers is list of CacheLayer with .keys / .values
            if hasattr(self, "layers"):
                layer = self.layers[idx]
                if hasattr(layer, "keys") and hasattr(layer, "values"):
                    k = layer.keys
                    v = layer.values
                    k = k() if callable(k) else k
                    v = v() if callable(v) else v
                    return (k, v)
            # 4.36-4.54: separate key_cache / value_cache lists
            if hasattr(self, "key_cache") and hasattr(self, "value_cache"):
                return (self.key_cache[idx], self.value_cache[idx])
            # legacy fallback
            if hasattr(self, "to_legacy_cache"):
                return self.to_legacy_cache()[idx]
            raise TypeError("Unsupported DynamicCache layout")

        @staticmethod
        def _dyncache_from_legacy(past_key_values):
            # Construct DynamicCache from legacy tuple-of-tuples format.
            cache = DynamicCache()
            for layer_idx, (k, v) in enumerate(past_key_values):
                if not hasattr(cache, "key_cache"):
                    cache.key_cache = []
                    cache.value_cache = []
                cache.key_cache.append(k)
                cache.value_cache.append(v)
            return cache

        # Force-override even if transformers added a different __getitem__.
        DynamicCache.__getitem__ = _dyncache_layer_kv
        if not hasattr(DynamicCache, "from_legacy_cache"):
            DynamicCache.from_legacy_cache = _dyncache_from_legacy
    except ImportError:
        pass

    # ------------------------------------------------------------------ tokeniser
    decoder_tokenizer = AutoTokenizer.from_pretrained(lc.decoder_model)
    if decoder_tokenizer.pad_token is None:
        decoder_tokenizer.pad_token = decoder_tokenizer.eos_token

    # ------------------------------------------------------------------ model
    encoder = SentenceEncoder(
        model_path=lc.encoder_model,
        latent_size=lc.latent_size,
        decoder_tokenizer=decoder_tokenizer,
        device=str(device),
    )
    decoder = SentenceDecoder(
        model_path=lc.decoder_model,
        latent_size=lc.latent_size,
        max_len=lc.max_len,
        device=device,
    )
    vae_config = VAEConfig(latent_dim=lc.latent_size)
    model = LangVAE(model_config=vae_config, encoder=encoder, decoder=decoder)

    # ------------------------------------------------------------------ datasets
    from langvae.data_conversion.tokenization import TokenizedDataSet

    logger.info("Loading SQuAD train answers…")
    train_texts = _collect_answer_texts("train")
    logger.info("Loading SQuAD val answers…")
    val_texts = _collect_answer_texts("validation")

    # Use langvae's TokenizedDataSet which returns one-hot encoded sparse tensors
    train_dataset = TokenizedDataSet(
        train_texts, decoder_tokenizer, max_len=lc.max_len,
        return_tensors=True, one_hot=True
    )
    val_dataset = TokenizedDataSet(
        val_texts, decoder_tokenizer, max_len=lc.max_len,
        return_tensors=True, one_hot=True
    )
    logger.info("Train: %d  Val: %d answer texts", len(train_texts), len(val_texts))

    # ------------------------------------------------------------------ trainer
    trainer_config = CyclicalScheduleKLThresholdTrainerConfig(
        learning_rate=lc.learning_rate,
        batch_size=lc.batch_size,
        num_epochs=lc.num_epochs,
        max_beta=lc.max_beta,
        target_kl=lc.kl_threshold,
        output_dir=lc.checkpoint_dir,
    )

    pipeline = LanguageTrainingPipeline(
        model=model,
        training_config=trainer_config,
    )

    logger.info("Starting LangVAE training (encoder=%s, decoder=%s, latent=%d)…",
                lc.encoder_model, lc.decoder_model, lc.latent_size)

    # ------------------------------------------------------------------ wandb setup
    init_wandb(
        config={
            "encoder_model": lc.encoder_model,
            "decoder_model": lc.decoder_model,
            "latent_size": lc.latent_size,
            "batch_size": lc.batch_size,
            "learning_rate": lc.learning_rate,
            "num_epochs": lc.num_epochs,
            "max_beta": lc.max_beta,
            "kl_threshold": lc.kl_threshold,
        },
        project="latent-diffusion-text-langvae",
    )

    # ------------------------------------------------------------------ callbacks
    callbacks = [WandbCallback()]

    # Note: pipeline expects raw datasets, not DataLoaders. It creates DataLoaders internally
    # with the collate_fn from training_config
    pipeline(train_data=train_dataset, eval_data=val_dataset, callbacks=callbacks)

    # ------------------------------------------------------------------ save
    ckpt_dir = Path(lc.checkpoint_dir)
    adapter = LangVAEAdapter(model, decoder_tokenizer, lc.latent_size, lc.max_len)
    adapter.save(ckpt_dir)
    logger.info("LangVAE saved to %s", ckpt_dir)

    # ------------------------------------------------------------------ wandb finalize
    log_wandb({"checkpoint_dir": str(ckpt_dir)}, step=0)
    finish_wandb()

    return adapter


if __name__ == "__main__":
    from src.config.loader import create_config_from_cli

    # Configure detailed logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("langvae_training.log"),
        ],
    )

    cfg = create_config_from_cli()
    train_langvae(cfg)
