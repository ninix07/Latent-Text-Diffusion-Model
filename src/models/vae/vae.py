"""SequenceVAE — top-level module composing encoder, decoder, and output head."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.config.schema import VAEArchConfig
from .encoder import VAEEncoder
from .decoder import VAEDecoder
from .output_head import OutputProjection
from .reparameterize import reparameterize
from .loss import compute_vae_loss
from .decoding import greedy_decode, beam_search_decode, nucleus_decode


class SequenceVAE(nn.Module):
    """Full sequence VAE: encode token ids → latent → decode → logits.

    Parameters
    ----------
    config : VAEArchConfig
    pretrained_embeddings : Tensor, optional
        Shape ``(vocab_size, embed_dim)``.
    """

    def __init__(
        self,
        config: VAEArchConfig,
        pretrained_embeddings: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = config

        if pretrained_embeddings is not None:
            vocab_size = pretrained_embeddings.size(0)
        else:
            raise ValueError("pretrained_embeddings is required to infer vocab_size")

        self.encoder = VAEEncoder(
            embed_dim=config.embed_dim,
            latent_dim=config.latent_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=config.dropout,
            max_answer_len=config.max_answer_len,
            pretrained_embeddings=pretrained_embeddings,
        )
        self.decoder = VAEDecoder(
            latent_dim=config.latent_dim,
            embed_dim=config.embed_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=config.dropout,
            max_answer_len=config.max_answer_len,
        )
        self.output_head = OutputProjection(
            embed_dim=config.embed_dim,
            vocab_size=vocab_size,
            pretrained_embeddings=pretrained_embeddings,
        )

    def encode(
        self,
        token_ids: torch.Tensor,
        mask: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode to latent space. Returns (z, μ, log_var)."""
        mu, log_var = self.encoder(token_ids, mask)
        z = reparameterize(mu, log_var, deterministic=deterministic)
        return z, mu, log_var

    def decode(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Decode latent z to logits."""
        hidden = self.decoder(z, mask)
        return self.output_head(hidden)

    def forward(
        self,
        token_ids: torch.Tensor,
        mask: torch.Tensor,
        beta: float = 1.0,
        free_bits: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Full forward pass.

        Returns
        -------
        (logits, z, μ, log_var, loss_dict)
            loss_dict has keys ``"total"``, ``"recon"``, ``"kl"``.
        """
        z, mu, log_var = self.encode(token_ids, mask)
        logits = self.decode(z, mask)
        total, recon, kl = compute_vae_loss(
            logits, token_ids, mask, mu, log_var, beta, free_bits=free_bits
        )
        loss_dict = {"total": total, "recon": recon, "kl": kl}
        return logits, z, mu, log_var, loss_dict

    def decode_to_tokens(
        self,
        z: torch.Tensor,
        mask: torch.Tensor,
        strategy: str = "greedy",
        **kwargs,
    ) -> torch.Tensor:
        """Decode latent z to token ids using the given strategy."""
        logits = self.decode(z, mask)
        if strategy == "greedy":
            return greedy_decode(logits)
        elif strategy == "beam_search":
            return beam_search_decode(logits, **kwargs)
        elif strategy == "nucleus":
            return nucleus_decode(logits, **kwargs)
        else:
            raise ValueError(f"Unknown decoding strategy: {strategy}")
