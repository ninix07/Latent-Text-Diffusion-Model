"""Training pipeline for LangVAE (replaces Stage 1 for the LangVAE path)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

from src.config.schema import Config
from src.models.vae.langvae_adapter import LangVAEAdapter
from src.utils.logging import init_wandb, log_wandb, finish_wandb, is_wandb_active
from pythae.trainers.training_callbacks import TrainingCallback

logger = logging.getLogger(__name__)


class CustomWandbCallback(TrainingCallback):
    """Callback to log training metrics to Weights & Biases."""

    def on_log(self, training_config, logs, **kwargs):
        """Called after each epoch with loss metrics."""
        # pythae BaseTrainer passes epoch via kwargs["global_step"], not in logs.
        epoch = kwargs.get("global_step")
        train_loss = logs.get("train_epoch_loss", None)
        eval_loss = logs.get("eval_epoch_loss", None)

        metrics = {}
        if train_loss is not None:
            metrics["loss/train_epoch"] = train_loss
        if eval_loss is not None:
            metrics["loss/eval_epoch"] = eval_loss

        if metrics and epoch is not None:
            # Carry epoch as a value, not as the wandb step. The per-batch
            # forward wrapper logs on wandb's auto-incrementing step axis; mixing
            # a small epoch index in as an explicit step would be non-monotonic
            # and wandb would drop these epoch points.
            metrics["epoch"] = epoch
            # Surface whether the metrics actually reached W&B — a failed
            # init_wandb() makes log_wandb() a silent no-op, which looks like
            # "training is fine but no graphs appear".
            logger.info(
                " custom wandb log -> epoch=%s metrics=%s (wandb_active=%s)",
                epoch, metrics, is_wandb_active(),
            )
            log_wandb(metrics)

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
    from langvae.decoders import SentenceDecoder
    from pythae.models.vae import VAEConfig
    from langvae.trainers import CyclicalScheduleKLThresholdTrainerConfig
    from langvae.pipelines import LanguageTrainingPipeline
    from transformers import AutoTokenizer
    from src.models.vae.seq_sentence_encoder import SeqSentenceEncoder

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
    # Flat latent dim consumed by pythae's KL = K * D. The K=num_latent_tokens
    # slots are produced by the SeqSentenceEncoder's Perceiver query pool and
    # the downstream LangVAEAdapter reshapes back to (B, K, D) for diffusion.
    flat_latent_dim = lc.num_latent_tokens * lc.latent_size

    encoder = SeqSentenceEncoder(
        model_path=lc.encoder_model,
        latent_size=lc.latent_size,
        decoder_tokenizer=decoder_tokenizer,
        num_latent_tokens=lc.num_latent_tokens,
        device=str(device),
    )
    decoder = SentenceDecoder(
        model_path=lc.decoder_model,
        latent_size=flat_latent_dim,
        max_len=lc.max_len,
        device=device,
    )
    vae_config = VAEConfig(latent_dim=flat_latent_dim)
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
            "num_latent_tokens": lc.num_latent_tokens,
            "flat_latent_dim": flat_latent_dim,
            "batch_size": lc.batch_size,
            "learning_rate": lc.learning_rate,
            "num_epochs": lc.num_epochs,
            "max_beta": lc.max_beta,
            "kl_threshold": lc.kl_threshold,
        },
        project="latent-diffusion-text-langvae",
    )

    # ------------------------------------------------------------------ callbacks
    callbacks = [CustomWandbCallback()]

    # ------------------------------------------------------------------ per-step W&B logging
    # The trainer's on_log callback only fires at epoch end — one point per
    # epoch, and nothing at all when a run crashes mid-epoch (which is exactly
    # how the NaN failure looked: only system charts, no loss curve). Wrap the
    # model's forward to emit per-batch losses while training. This is version-
    # proof: it does not depend on which trainer/callback the installed langvae
    # build happens to invoke, only on the ModelOutput contract (loss/recon_loss/
    # reg_loss) and model.cur_beta, which are stable across langvae versions.
    if lc.log_every_n_steps and lc.log_every_n_steps > 0:
        _orig_forward = model.forward
        _step = {"n": 0}

        def _logging_forward(inputs, **kwargs):
            out = _orig_forward(inputs, **kwargs)
            # Only log on training passes; eval batches set model.training False.
            if model.training:
                n = _step["n"]
                if n % lc.log_every_n_steps == 0:
                    log_wandb({
                        "loss/train_step": float(out.loss),
                        "loss/recon_step": float(out.recon_loss),
                        "loss/kl_step": float(out.reg_loss),
                        "beta": float(model.cur_beta),
                    })
                _step["n"] = n + 1
            return out

        model.forward = _logging_forward

    # Note: pipeline expects raw datasets, not DataLoaders. It creates DataLoaders internally
    # with the collate_fn from training_config
    pipeline(train_data=train_dataset, eval_data=val_dataset, callbacks=callbacks)

    # ------------------------------------------------------------------ save
    ckpt_dir = Path(lc.checkpoint_dir)
    adapter = LangVAEAdapter(
        model,
        decoder_tokenizer,
        latent_size=lc.latent_size,
        max_len=lc.max_len,
        num_latent_tokens=lc.num_latent_tokens,
    )
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
