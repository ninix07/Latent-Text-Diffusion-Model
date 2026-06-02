"""SeqSentenceEncoder — LangVAE encoder that emits K latent slots.

The stock :class:`langvae.encoders.SentenceEncoder` mean/last/CLS-pools the
pretrained encoder output into a single ``(B, hidden_size)`` vector and
projects it to ``(B, 2 * latent_size)``. The downstream latent-text
diffusion denoiser expects a sequence latent of shape
``(B, K, latent_size)``, so the pooled single-vector latent forces the
diffusion path to operate at sequence length 1 — effectively an MLP
denoiser, throwing away the whole point of sequence-aware diffusion.

This subclass replaces the pooling with K learnable Perceiver-style
queries that cross-attend to the full BERT hidden states. Each query
becomes one latent slot, and a per-slot linear projects to
``2 * latent_size``. The K slot mean/log-variance tensors are flattened
to ``(B, K * latent_size)`` for pythae's flat-latent KL computation; the
downstream :class:`LangVAEAdapter` reshapes back to ``(B, K, D)`` for
the diffusion pipeline.
"""

from __future__ import annotations

import torch
from torch import nn, Tensor
from pythae.models.base.base_utils import ModelOutput
from langvae.encoders import SentenceEncoder

# Bound on encoder log-variance. exp(15) ~ 3.3e6 stays finite even summed over
# flat_latent_dim slots; wide enough that a well-behaved posterior never hits it.
LOG_VAR_CLAMP = 15.0


class SeqSentenceEncoder(SentenceEncoder):
    """Sequence-latent encoder for LangVAE.

    Parameters
    ----------
    model_path, latent_size, decoder_tokenizer, device, ... :
        Same as :class:`SentenceEncoder`. ``latent_size`` is the
        *per-slot* latent dimension.
    num_latent_tokens : int
        Number of latent slots K.
    num_query_heads : int
        Number of attention heads in the Perceiver query cross-attention.
    """

    def __init__(
        self,
        model_path: str,
        latent_size: int,
        decoder_tokenizer,
        num_latent_tokens: int = 16,
        num_query_heads: int = 8,
        caching: bool = False,
        automodel_preset: dict = None,
        device: str = "cpu",
        args=None,
    ) -> None:
        super().__init__(
            model_path=model_path,
            latent_size=latent_size,
            decoder_tokenizer=decoder_tokenizer,
            caching=caching,
            automodel_preset=automodel_preset,
            device=device,
            args=args,
        )

        self.num_latent_tokens = num_latent_tokens
        self.per_slot_latent = latent_size

        hidden_size = self._encoder[0].config.hidden_size

        # Replace the parent's LazyLinear (built for the single-vector pooled
        # path and never called here) with a concrete Linear so torch.save can
        # serialise the module without hitting unmaterialised lazy params.
        self.linear = nn.Linear(hidden_size, 2 * latent_size, bias=False, device=device)
        self.linear.requires_grad_(False)

        # Perceiver-style query cross-attention onto BERT hidden states.
        self.latent_queries = nn.Parameter(
            torch.randn(1, num_latent_tokens, hidden_size) * 0.02
        )
        self.query_norm = nn.LayerNorm(hidden_size)
        self.kv_norm = nn.LayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_query_heads,
            dropout=0.0,
            batch_first=True,
        )

        # Per-slot projection to (mean, log_var). Replaces the parent's
        # self.linear which assumes a single pooled vector.
        self.slot_proj = nn.Linear(hidden_size, 2 * latent_size, bias=False)

        self.to(device)

    # ------------------------------------------------------------------
    # Helpers

    def _encode_full(self, tok_ids) -> tuple[Tensor, Tensor]:
        """Tokenise decoder IDs back to strings, run pretrained encoder, and
        return ``(last_hidden_state, attention_mask)`` — no pooling.
        """
        input_texts = self.decoder_tokenizer.batch_decode(
            tok_ids, clean_up_tokenization_spaces=False, skip_special_tokens=True
        )
        enc_toks = self.tokenizer(
            input_texts, padding=True, truncation=True, return_tensors="pt"
        )
        enc_attn_mask = enc_toks["attention_mask"].to(self.device)
        enc_ids = enc_toks["input_ids"].to(self.device)
        with torch.no_grad():
            enc_out = self.encoder(input_ids=enc_ids, attention_mask=enc_attn_mask)
        return enc_out.last_hidden_state, enc_attn_mask

    # ------------------------------------------------------------------
    # Forward

    def forward(self, x: Tensor, c=None) -> ModelOutput:
        # Fix for pythae device allocation bug (mirror parent class behaviour).
        self._encoder[0] = self._encoder[0].to(self.device)

        tok_ids = x
        if x.layout == torch.sparse_coo:
            tok_ids = [x[i].coalesce().indices()[1] for i in range(x.shape[0])]

        hidden, enc_attn_mask = self._encode_full(tok_ids)  # (B, L, H), (B, L)
        B = hidden.shape[0]

        kv_pad_mask = enc_attn_mask == 0  # True = ignore

        queries = self.latent_queries.expand(B, -1, -1)  # (B, K, H)
        q = self.query_norm(queries)
        kv = self.kv_norm(hidden)
        pooled, _ = self.cross_attn(
            q, kv, kv, key_padding_mask=kv_pad_mask, need_weights=False
        )
        pooled = pooled + queries  # residual on queries

        slot_out = self.slot_proj(pooled)  # (B, K, 2 * D)
        mean, log_var = slot_out.chunk(2, dim=-1)  # each (B, K, D)

        # NaN guard. pythae's KL sums exp(log_var) over all K*D=flat_latent_dim
        # slots; an unbounded log_var lets exp() overflow to +inf within an
        # epoch, which propagates to the loss as NaN and trips the trainer's
        # ArithmeticError. Clamp to a range wide enough not to distort a healthy
        # posterior but tight enough that exp() stays finite. langvae also clips
        # grads *before* backward (a no-op), so this clamp is the only bound on
        # the bottleneck statistics.
        log_var = log_var.clamp(-LOG_VAR_CLAMP, LOG_VAR_CLAMP)

        # Flatten for pythae's flat-latent KL computation.
        mean_flat = mean.reshape(B, -1)
        log_var_flat = log_var.reshape(B, -1)

        return ModelOutput(
            embedding=mean_flat,
            cvars_embedding=[],
            log_covariance=log_var_flat,
        )
