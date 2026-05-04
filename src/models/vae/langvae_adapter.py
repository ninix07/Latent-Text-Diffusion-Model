"""Adapter wrapping langvae.LangVAE to match the SequenceVAE interface.

SequenceVAE interface used by export_latents and generate:
    encode(token_ids, mask, deterministic) -> (z, mu, log_var)   shapes (B,D)
    decode_to_tokens(z, strategy, ...)     -> token_id tensor     shape  (B,L)

LangVAEAdapter replaces decode_to_tokens with decode_sentences, which
generate.py detects via hasattr and handles accordingly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class LangVAEAdapter:
    """Thin wrapper around a trained langvae.LangVAE model.

    Parameters
    ----------
    model : langvae.LangVAE
        A fully initialised (and optionally pretrained) LangVAE instance.
    decoder_tokenizer : PreTrainedTokenizer
        The decoder's tokenizer — SentenceEncoder.recode() expects token IDs
        from this tokenizer and converts them to the encoder tokenizer space
        internally.
    latent_size : int
        Dimensionality of the latent space.
    max_len : int
        Max sequence length (in decoder-tokenizer tokens).
    """

    def __init__(self, model, decoder_tokenizer, latent_size: int = 128, max_len: int = 50):
        self.model = model
        self.decoder_tokenizer = decoder_tokenizer
        self.latent_size = latent_size
        self.max_len = max_len

    # ------------------------------------------------------------------
    # Encode

    @torch.no_grad()
    def encode_from_texts(
        self,
        texts: list[str],
        deterministic: bool = True,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Encode a list of raw answer strings.

        Returns (z, mu, log_var) each of shape (B, latent_size).
        When deterministic=True z is the mean (mu), not sampled.
        """
        device = self._device

        # Tokenize with decoder tokenizer — SentenceEncoder.recode() will
        # convert these IDs to encoder (BERT) token IDs internally.
        enc = self.decoder_tokenizer(
            texts,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)  # (B, L) — decoder vocab IDs

        # SentenceEncoder.forward() expects a tensor x of shape (B, L)
        # with integer token IDs (pythae's "one-hot" convention for discrete data).
        enc_out = self.model.encoder(input_ids)

        mu = enc_out.embedding          # (B, latent_size)
        log_var = enc_out.log_covariance  # (B, latent_size)

        if deterministic:
            return mu, mu, log_var

        std = torch.exp(0.5 * log_var)
        z = mu + std * torch.randn_like(std)
        return z, mu, log_var

    # ------------------------------------------------------------------
    # Decode

    @torch.no_grad()
    def decode_sentences(self, z0: torch.Tensor) -> list[str]:
        """Decode latent vectors to answer strings.

        Parameters
        ----------
        z0 : (B, latent_size) float tensor on any device.

        Returns
        -------
        list[str] of length B.
        """
        return self.model.decode_sentences(z0)

    # ------------------------------------------------------------------
    # Persistence

    def save(self, path: str | Path) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path))
        logger.info("Saved LangVAE to %s", path)

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        device: str | torch.device = "cpu",
        latent_size: int = 128,
        max_len: int = 50,
    ) -> "LangVAEAdapter":
        from langvae import LangVAE

        model = LangVAE.load_from_folder(str(path))
        model = model.to(device)
        decoder_tokenizer = model.decoder.tokenizer
        return cls(model, decoder_tokenizer, latent_size=latent_size, max_len=max_len)

    # ------------------------------------------------------------------

    @property
    def _device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")
