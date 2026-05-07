"""Training pipeline for LangVAE (replaces Stage 1 for the LangVAE path)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

from src.config.schema import Config
from src.models.vae.langvae_adapter import LangVAEAdapter

logger = logging.getLogger(__name__)


class _AnswerTextDataset(Dataset):
    """Minimal pythae-compatible dataset wrapping tokenised answer strings.

    LangVAE's LanguageTrainingPipeline expects each sample to be a dict
    with a ``data`` key containing a 1-D integer tensor of token IDs (encoded
    with the decoder tokenizer so that SentenceEncoder.recode() can convert
    them to the encoder's token space internally).
    """

    def __init__(self, texts: list[str], tokenizer, max_len: int) -> None:
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {"data": enc["input_ids"].squeeze(0)}  # (max_len,) int64


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
    try:
        from transformers.cache_utils import DynamicCache
        if not hasattr(DynamicCache, "__getitem__"):
            def _dyncache_getitem(self, idx):
                return self.to_legacy_cache()[idx]
            DynamicCache.__getitem__ = _dyncache_getitem
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
    logger.info("Loading SQuAD train answers…")
    train_texts = _collect_answer_texts("train")
    logger.info("Loading SQuAD val answers…")
    val_texts = _collect_answer_texts("validation")

    train_dataset = _AnswerTextDataset(train_texts, decoder_tokenizer, lc.max_len)
    val_dataset = _AnswerTextDataset(val_texts, decoder_tokenizer, lc.max_len)
    logger.info("Train: %d  Val: %d answer texts", len(train_texts), len(val_texts))

    # ------------------------------------------------------------------ trainer
    trainer_config = CyclicalScheduleKLThresholdTrainerConfig(
        learning_rate=lc.learning_rate,
        batch_size=lc.batch_size,
        num_epochs=lc.num_epochs,
        max_beta=lc.max_beta,
        threshold=lc.kl_threshold,
        output_dir=lc.checkpoint_dir,
    )

    pipeline = LanguageTrainingPipeline(
        vae_model=model,
        trainer_config=trainer_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    logger.info("Starting LangVAE training (encoder=%s, decoder=%s, latent=%d)…",
                lc.encoder_model, lc.decoder_model, lc.latent_size)
    pipeline.train()

    # ------------------------------------------------------------------ save
    ckpt_dir = Path(lc.checkpoint_dir)
    adapter = LangVAEAdapter(model, decoder_tokenizer, lc.latent_size, lc.max_len)
    adapter.save(ckpt_dir)
    logger.info("LangVAE saved to %s", ckpt_dir)

    return adapter


if __name__ == "__main__":
    from src.config.loader import create_config_from_cli

    logging.basicConfig(level=logging.INFO)
    cfg = create_config_from_cli()
    train_langvae(cfg)
