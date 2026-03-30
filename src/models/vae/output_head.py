"""Output projection: cosine-similarity logits with learned temperature."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OutputProjection(nn.Module):
    """Produce logits via cosine similarity with a learned temperature.

    Parameters
    ----------
    embed_dim : int
        Hidden dimension of decoder output.
    vocab_size : int
        Vocabulary size.
    pretrained_embeddings : Tensor, optional
        Weight matrix of shape ``(vocab_size, embed_dim)`` to initialise the
        projection.  A **detached copy** is used.
    """

    LOG_TAU_MAX = 4.6  # ≈ ln(100)

    def __init__(
        self,
        embed_dim: int,
        vocab_size: int,
        pretrained_embeddings: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(embed_dim, vocab_size, bias=False)

        if pretrained_embeddings is not None:
            with torch.no_grad():
                self.linear.weight.copy_(pretrained_embeddings.detach())

        self.log_tau = nn.Parameter(torch.zeros(1))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute cosine-similarity logits scaled by temperature.

        Parameters
        ----------
        hidden_states : Tensor (B, L, embed_dim)

        Returns
        -------
        Tensor (B, L, vocab_size)
        """
        h = F.normalize(hidden_states, dim=-1)
        w = F.normalize(self.linear.weight, dim=-1)
        cos_sim = F.linear(h, w)  # (B, L, V)

        tau = torch.exp(torch.clamp(self.log_tau, max=self.LOG_TAU_MAX))
        return cos_sim * tau
