"""VAE encoder: embeds token ids and produces per-position μ and log_var."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.positional import sinusoidal_encoding as _sinusoidal_encoding


class VAEEncoder(nn.Module):
    """Transformer encoder that maps token ids to latent (μ, log_var).

    Output shape: ``(B, max_answer_len, latent_dim)`` — one latent vector per
    position with no sequence compression.
    """

    def __init__(
        self,
        embed_dim: int,
        latent_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        max_answer_len: int,
        pretrained_embeddings: torch.Tensor | None = None,
    ) -> None:
        super().__init__()

        # --- Embedding ---
        if pretrained_embeddings is not None:
            vocab_size, emb_d = pretrained_embeddings.shape
            self.embedding = nn.Embedding(vocab_size, emb_d)
            with torch.no_grad():
                self.embedding.weight.copy_(pretrained_embeddings)
        else:
            raise ValueError("vocab_size must be inferred from pretrained_embeddings or set explicitly")

        # --- Positional encoding (sinusoidal, non-learnable) ---
        pe = _sinusoidal_encoding(max_answer_len, embed_dim)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, L, D)

        # --- Transformer ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- Projection to latent space ---
        self.proj = nn.Linear(embed_dim, latent_dim)
        self.mu_head = nn.Linear(latent_dim, latent_dim)
        self.logvar_head = nn.Linear(latent_dim, latent_dim)

    # Allow construction without pretrained embeddings by accepting vocab_size
    @classmethod
    def from_vocab_size(
        cls,
        vocab_size: int,
        embed_dim: int,
        latent_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        max_answer_len: int,
    ) -> "VAEEncoder":
        emb = torch.randn(vocab_size, embed_dim)
        return cls(embed_dim, latent_dim, num_layers, num_heads, dropout, max_answer_len, emb)

    def forward(
        self,
        token_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode tokens to latent parameters.

        Parameters
        ----------
        token_ids : Tensor (B, L)
        mask : Tensor (B, L)  — 1 for real tokens, 0 for padding.

        Returns
        -------
        (μ, log_var) each of shape (B, L, latent_dim)
        """
        x = self.embedding(token_ids) + self.pe[:, : token_ids.size(1), :]

        # TransformerEncoder expects src_key_padding_mask where True = ignore
        pad_mask = mask == 0
        x = self.transformer(x, src_key_padding_mask=pad_mask)

        h = self.proj(x)
        mu = self.mu_head(h)
        log_var = self.logvar_head(h).clamp(-6.0, 4.0)
        return mu, log_var
