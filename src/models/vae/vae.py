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


class SequenceVAE(nn.Module):
    """Full sequence VAE: encode token ids → pooled latent → causal decode → logits.

    The encoder pools the sequence into a single latent vector ``(B, D)``.
    The decoder is a **causal** transformer that injects the latent via a
    prefix of pseudo-tokens (KV cache injection).  During training the
    decoder is teacher-forced; during generation it decodes autoregressively.

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
            vocab_size=vocab_size,
            num_latent_tokens=config.num_latent_tokens,
        )
        self.output_head = OutputProjection(
            embed_dim=config.embed_dim,
            vocab_size=vocab_size,
            pretrained_embeddings=pretrained_embeddings,
        )

        # NOTE: weight tying between encoder embeddings and output head has
        # been intentionally removed.  The encoder and decoder now have
        # separate embeddings so reconstruction gradients do not conflict
        # with the encoder's latent-space training signal (Bug 8).

    def encode(
        self,
        token_ids: torch.Tensor,
        mask: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode to pooled latent space.  Returns ``(z, μ, log_var)`` each ``(B, D)``."""
        mu, log_var = self.encoder(token_ids, mask)
        z = reparameterize(mu, log_var, deterministic=deterministic)
        return z, mu, log_var

    def decode(
        self,
        token_ids: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Teacher-forced decode: latent z + shifted target → logits."""
        hidden = self.decoder(token_ids, z, mask)
        return self.output_head(hidden)

    def forward(
        self,
        token_ids: torch.Tensor,
        mask: torch.Tensor,
        beta: float = 1.0,
        free_bits: float = 0.0,
        target_kl: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Full forward pass (teacher-forced).

        Returns
        -------
        (logits, z, μ, log_var, loss_dict)
            loss_dict has keys ``"total"``, ``"recon"``, ``"kl"``.
        """
        z, mu, log_var = self.encode(token_ids, mask)
        # Pass the REAL mask so the decoder properly masks padding (Bug 2).
        # The causal architecture prevents the decoder from exploiting mask
        # boundaries because it can only see previous positions.
        logits = self.decode(token_ids, z, mask)
        total, recon, kl = compute_vae_loss(
            logits, token_ids, mask, mu, log_var, beta, free_bits, target_kl
        )
        loss_dict = {"total": total, "recon": recon, "kl": kl}
        return logits, z, mu, log_var, loss_dict

    def decode_to_tokens(
        self,
        z: torch.Tensor,
        strategy: str = "greedy",
        max_len: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Autoregressively decode latent z to token ids.

        Parameters
        ----------
        z : (B, latent_dim) — pooled latent.
        strategy : ``"greedy"`` or ``"nucleus"``.
        max_len : int, optional — defaults to ``config.max_answer_len``.
        """
        if max_len is None:
            max_len = self.config.max_answer_len
        return self.decoder.generate(
            z,
            max_len=max_len,
            output_head=self.output_head,
            strategy=strategy,
            **kwargs,
        )
