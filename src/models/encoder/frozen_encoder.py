"""Frozen pretrained BERT encoder for conditioning extraction."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class FrozenEncoder(nn.Module):
    """Wraps a pretrained BERT model with frozen parameters.

    The encoder is kept in eval mode at all times.  An optional
    ``unfreeze_top_n`` argument allows the top N encoder layers to remain
    trainable (useful for light fine-tuning).

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier (e.g. ``"bert-base-uncased"``).
    unfreeze_top_n : int, optional
        Number of top transformer layers to leave unfrozen.  Default ``0``
        means every parameter is frozen.
    """

    def __init__(self, model_name: str, unfreeze_top_n: int = 0) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self._freeze(unfreeze_top_n)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run a forward pass and return the last hidden state.

        Parameters
        ----------
        input_ids : Tensor
            Token ids of shape ``(B, seq_len)``.
        attention_mask : Tensor
            Attention mask of shape ``(B, seq_len)``.

        Returns
        -------
        Tensor
            Last hidden state of shape ``(B, seq_len, hidden_dim)``.
        """
        with torch.no_grad():
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        return outputs.last_hidden_state

    def get_embedding_table(self) -> torch.Tensor:
        """Return the token embedding weight matrix.

        Returns
        -------
        Tensor
            Weight matrix of shape ``(vocab_size, hidden_dim)``.
        """
        return self.bert.embeddings.word_embeddings.weight

    # ------------------------------------------------------------------
    # Override train to keep eval mode
    # ------------------------------------------------------------------

    def train(self, mode: bool = True) -> "FrozenEncoder":
        """Override to keep the encoder permanently in eval mode."""
        # Always call super with mode=False so all sub-modules stay in eval.
        super().train(False)
        return self

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _freeze(self, unfreeze_top_n: int = 0) -> None:
        """Freeze all parameters, optionally unfreezing top N encoder layers.

        Parameters
        ----------
        unfreeze_top_n : int
            Number of top BERT encoder layers to leave trainable.
        """
        # Freeze everything first.
        for param in self.bert.parameters():
            param.requires_grad = False

        # Optionally unfreeze the top N encoder layers.
        if unfreeze_top_n > 0:
            layers = self.bert.encoder.layer
            total = len(layers)
            for layer in layers[total - unfreeze_top_n :]:
                for param in layer.parameters():
                    param.requires_grad = True

        # Ensure eval mode.
        self.bert.eval()
