"""VAE decoder: causal transformer with latent KV-prefix injection."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.positional import sinusoidal_encoding as _sinusoidal_encoding


class VAEDecoder(nn.Module):
    """Causal transformer decoder with latent prefix (KV cache injection).

    The pooled latent *z* is projected into ``num_latent_tokens`` pseudo-tokens
    that are **prepended** to the decoder sequence.  A custom attention mask
    ensures:

    * All positions can attend to the latent prefix tokens.
    * Decoder token positions obey causal (autoregressive) ordering.

    During training the decoder receives the target tokens shifted right
    (a learned ``<start>`` embedding is prepended, the last token is dropped).
    This forces each position to predict the next token using only previous
    tokens **and** the latent — the decoder cannot bypass *z*.

    Output shape: ``(B, L, embed_dim)``.  Does **not** produce logits; that
    is handled by :class:`OutputProjection`.
    """

    def __init__(
        self,
        latent_dim: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        max_answer_len: int,
        vocab_size: int,
        num_latent_tokens: int = 8,
    ) -> None:
        super().__init__()
        self.num_latent_tokens = num_latent_tokens
        self.embed_dim = embed_dim

        # Decoder's own token embedding (separate from encoder — avoids
        # conflicting gradient streams, see Bug 8).
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        # Learned start-of-sequence embedding
        self.start_embed = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Positional encoding — applied only to decoder token positions,
        # NOT to latent prefix tokens.  This prevents PE from competing
        # with the latent signal (Bug 3).
        pe = _sinusoidal_encoding(max_answer_len, embed_dim)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, L, D)

        # Project pooled latent z → K prefix tokens for KV injection
        self.latent_proj = nn.Linear(latent_dim, num_latent_tokens * embed_dim)

        # Transformer encoder used with a custom causal+prefix mask so that
        # the architecture is effectively a causal decoder with KV injection.
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

    # ------------------------------------------------------------------
    # Mask helpers
    # ------------------------------------------------------------------

    def _build_attn_mask(self, K: int, L: int, device: torch.device) -> torch.Tensor:
        """Additive attention mask of shape ``(K+L, K+L)``.

        * Latent prefix rows (0 … K-1): attend to all prefix tokens, but
          **not** to decoder tokens (keeps prefix representations clean).
        * Decoder rows (K … K+L-1): attend to **all** prefix tokens plus
          causally to previous decoder tokens (position i sees K … K+i).

        Convention: 0 → attend, ``-inf`` → block.
        """
        total = K + L
        mask = torch.zeros(total, total, device=device)
        # Prefix tokens should not attend to decoder tokens
        mask[:K, K:] = float("-inf")
        # Causal mask for decoder positions — use bool then where to avoid
        # 0 * -inf = nan from triu multiplication.
        causal_bool = torch.triu(
            torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1
        )
        mask[K:, K:] = torch.where(causal_bool, float("-inf"), 0.0)
        return mask

    def _project_latent(self, z: torch.Tensor) -> torch.Tensor:
        """``(B, latent_dim)`` → ``(B, K, embed_dim)``."""
        B = z.size(0)
        return self.latent_proj(z).view(B, self.num_latent_tokens, self.embed_dim)

    # ------------------------------------------------------------------
    # Forward (teacher-forced training)
    # ------------------------------------------------------------------

    def forward(
        self,
        token_ids: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Teacher-forced decode: predict each token from previous tokens + z.

        Parameters
        ----------
        token_ids : Tensor (B, L) — target answer tokens.
        z         : Tensor (B, latent_dim) — pooled latent.
        mask      : Tensor (B, L) — 1 for real tokens, 0 for padding.

        Returns
        -------
        Tensor (B, L, embed_dim)
        """
        B, L = token_ids.shape
        K = self.num_latent_tokens

        # Shift right: [<start>, tok_0, …, tok_{L-2}]
        tok_emb = self.token_embedding(token_ids[:, :-1])  # (B, L-1, D)
        start = self.start_embed.expand(B, -1, -1)  # (B, 1, D)
        dec_input = torch.cat([start, tok_emb], dim=1)  # (B, L, D)
        dec_input = dec_input + self.pe[:, :L, :]

        # Latent prefix
        prefix = self._project_latent(z)  # (B, K, D)

        # Concatenate: [prefix | decoder_tokens]
        x = torch.cat([prefix, dec_input], dim=1)  # (B, K+L, D)

        # Attention mask (float additive): causal + prefix blocking
        attn_mask = self._build_attn_mask(K, L, x.device)
        # Padding mask (bool): True = ignore. Latent prefix is never padded.
        latent_pad = torch.zeros(B, K, device=mask.device, dtype=torch.bool)
        # Shift mask to align with dec_input: <start> is always real, and
        # dec_input[i] = token_ids[i-1], so we take mask[:, :-1] as the tail.
        shifted_mask = torch.cat(
            [torch.ones(B, 1, dtype=mask.dtype, device=mask.device), mask[:, :-1]],
            dim=1,
        )
        dec_pad = shifted_mask == 0
        full_pad_mask = torch.cat([latent_pad, dec_pad], dim=1)

        x = self.transformer(
            x, mask=attn_mask, src_key_padding_mask=full_pad_mask, is_causal=False
        )

        # Return only the decoder positions
        return x[:, K:, :]  # (B, L, D)

    # ------------------------------------------------------------------
    # Autoregressive generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        z: torch.Tensor,
        max_len: int,
        output_head: nn.Module,
        strategy: str = "greedy",
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """Autoregressively generate token ids from latent *z*.

        Parameters
        ----------
        z          : (B, latent_dim) — pooled latent.
        max_len    : int — maximum tokens to generate.
        output_head: OutputProjection module (hidden → logits).
        strategy   : ``"greedy"`` or ``"nucleus"``.
        temperature: softmax temperature (nucleus only).
        top_p      : nucleus probability mass (nucleus only).

        Returns
        -------
        Tensor (B, max_len)
        """
        import torch.nn.functional as F

        B = z.size(0)
        K = self.num_latent_tokens
        device = z.device

        prefix = self._project_latent(z)  # (B, K, D)
        start = self.start_embed.expand(B, -1, -1)  # (B, 1, D)
        dec_input = start + self.pe[:, :1, :]  # (B, 1, D)

        generated: list[torch.Tensor] = []

        for step in range(max_len):
            L_so_far = step + 1
            x = torch.cat([prefix, dec_input], dim=1)  # (B, K+L_so_far, D)
            attn_mask = self._build_attn_mask(K, L_so_far, device)

            out = self.transformer(x, mask=attn_mask)
            last_hidden = out[:, K + step : K + step + 1, :]  # (B, 1, D)
            logits = output_head(last_hidden).squeeze(1)  # (B, V)

            if strategy == "greedy":
                next_id = logits.argmax(dim=-1)  # (B,)
            else:  # nucleus
                scaled = logits / max(temperature, 1e-8)
                probs = F.softmax(scaled, dim=-1)
                sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
                cumsum = sorted_probs.cumsum(dim=-1)
                cutoff = (cumsum - sorted_probs) > top_p
                sorted_probs[cutoff] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                sampled = torch.multinomial(sorted_probs, 1).squeeze(-1)
                next_id = sorted_idx.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)

            generated.append(next_id)

            # Embed and append for next step
            next_emb = self.token_embedding(next_id).unsqueeze(1)  # (B, 1, D)
            next_emb = next_emb + self.pe[:, L_so_far : L_so_far + 1, :]
            dec_input = torch.cat([dec_input, next_emb], dim=1)

        return torch.stack(generated, dim=1)  # (B, max_len)
