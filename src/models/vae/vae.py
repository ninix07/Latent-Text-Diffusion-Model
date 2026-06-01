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
    """Full sequence VAE: encode token ids → sequence latent ``(B, K, D)`` →
    causal decode → logits.

    The encoder produces ``num_latent_tokens`` latent vectors (one per
    Perceiver query).  The decoder is a **causal** transformer that injects
    the K latent vectors as a KV-cache prefix.  During training the decoder
    is teacher-forced; during generation it decodes autoregressively.

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
            num_latent_tokens=config.num_latent_tokens,
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

        # Tie decoder token embedding ↔ output head weight. The decoder learns
        # a token representation and the output head needs to score the same
        # representations — tying halves the parameter count for the V×D
        # matrix and gives consistent gradients to the shared weight. The
        # encoder embedding is intentionally NOT tied: its gradients come
        # from the latent-space training signal, which would conflict with
        # reconstruction gradients (the original "Bug 8").
        self.decoder.token_embedding.weight = self.output_head.linear.weight

    def encode(
        self,
        token_ids: torch.Tensor,
        mask: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode to sequence latent space.

        Returns ``(z, μ, log_var)`` each of shape ``(B, K, latent_dim)``.
        """
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
        target_kl: float | None = None,
        noise_aug_sigma: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Full forward pass (teacher-forced).

        Parameters
        ----------
        noise_aug_sigma : float
            Extra Gaussian noise std to add to the sampled latent before
            decoding. The KL is still computed against the un-perturbed
            posterior, so this only affects the decoder's gradient signal.
            Used to make the decoder robust to slightly-imperfect latents
            produced at diffusion sampling time. ``0.0`` disables it.

        Returns
        -------
        (logits, z, μ, log_var, loss_dict)
            loss_dict has keys ``"total"``, ``"recon"``, ``"kl"``.
        """
        z, mu, log_var = self.encode(token_ids, mask)

        # Optional decoder noise augmentation: only the latent fed to the
        # decoder is perturbed. mu/log_var are unchanged so the KL term
        # continues to regularise the *true* posterior.
        z_decode = z
        if noise_aug_sigma > 0.0 and self.training:
            z_decode = z + torch.randn_like(z) * noise_aug_sigma

        # Pass the REAL mask so the decoder properly masks padding (Bug 2).
        # The causal architecture prevents the decoder from exploiting mask
        # boundaries because it can only see previous positions.
        logits = self.decode(token_ids, z_decode, mask)
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
        z : (B, K, latent_dim) — sequence of latent vectors from the encoder.
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
