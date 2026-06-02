"""Adapter wrapping langvae.LangVAE to match the SequenceVAE interface.

SequenceVAE interface used by export_latents and generate:
    encode(token_ids, mask, deterministic) -> (z, mu, log_var)
        Shapes: (B, K, D) when num_latent_tokens > 1, else (B, D).
    decode_to_tokens(z, strategy, ...)     -> token_id tensor     shape  (B,L)

LangVAEAdapter replaces decode_to_tokens with decode_sentences, which
generate.py detects via hasattr and handles accordingly.

The underlying LangVAE model stores latents as a single flat vector of
size ``num_latent_tokens * latent_size`` (so pythae's KL term sums over
every slot×dim coordinate). This adapter reshapes between the flat
representation pythae expects and the ``(B, K, D)`` sequence latent the
diffusion denoiser expects.
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
        Per-slot latent dimensionality (D).
    max_len : int
        Max sequence length (in decoder-tokenizer tokens).
    num_latent_tokens : int
        Number of latent slots K. When 1 the adapter is shape-compatible
        with the original single-vector LangVAE behaviour.
    """

    def __init__(
        self,
        model,
        decoder_tokenizer,
        latent_size: int = 128,
        max_len: int = 50,
        num_latent_tokens: int = 1,
    ) -> None:
        self.model = model
        self.decoder_tokenizer = decoder_tokenizer
        self.latent_size = latent_size
        self.max_len = max_len
        self.num_latent_tokens = num_latent_tokens

    # ------------------------------------------------------------------
    # Shape helpers

    def _unflatten(self, flat: torch.Tensor) -> torch.Tensor:
        """``(B, K*D)`` → ``(B, K, D)``. Pass through when K == 1."""
        if self.num_latent_tokens <= 1:
            return flat
        B = flat.shape[0]
        return flat.reshape(B, self.num_latent_tokens, self.latent_size)

    def _flatten(self, z: torch.Tensor) -> torch.Tensor:
        """``(B, K, D)`` → ``(B, K*D)``. Accept already-flat input untouched."""
        if z.dim() == 2:
            return z
        B = z.shape[0]
        return z.reshape(B, -1)

    # ------------------------------------------------------------------
    # Encode

    @torch.no_grad()
    def encode_from_texts(
        self,
        texts: list[str],
        deterministic: bool = True,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Encode a list of raw answer strings.

        Returns ``(z, mu, log_var)`` each of shape ``(B, K, D)`` when
        ``num_latent_tokens > 1``, else ``(B, D)``.
        When ``deterministic=True`` z is the mean (mu), not sampled.
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

        mu_flat = enc_out.embedding          # (B, K*D)
        log_var_flat = enc_out.log_covariance  # (B, K*D)

        mu = self._unflatten(mu_flat)
        log_var = self._unflatten(log_var_flat)

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
        z0 : tensor on any device.
            Accepts ``(B, K, D)`` (sequence latent from diffusion) or
            ``(B, K*D)`` (flat — passes straight through to LangVAE).

        Returns
        -------
        list[str] of length B.
        """
        return self.model.decode_sentences(self._flatten(z0))

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
        num_latent_tokens: int = 1,
    ) -> "LangVAEAdapter":
        from langvae import LangVAE

        model = LangVAE.load_from_folder(str(path))
        model = model.to(device)
        decoder_tokenizer = model.decoder.tokenizer
        return cls(
            model,
            decoder_tokenizer,
            latent_size=latent_size,
            max_len=max_len,
            num_latent_tokens=num_latent_tokens,
        )

    # ------------------------------------------------------------------

    @property
    def _device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")
