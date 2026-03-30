"""Adaptive Layer Normalization (AdaLN-Zero) building blocks.

Provides modulation parameters conditioned on timestep embeddings so that
each denoiser block can shift and scale its hidden states.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class AdaLNModulation(nn.Module):
    """Produce modulation parameters from a conditioning vector.

    Outputs *num_params* vectors of size *cond_dim*. When used for
    AdaLN-Zero with self-attn and cross-attn the typical value is
    ``num_params=6`` (gamma1, beta1, alpha1, gamma2, beta2, alpha2).

    The alpha-related output weights and biases are initialized to zero
    so that the gated residual starts as identity (AdaLN-Zero trick).

    Parameters
    ----------
    cond_dim : int
        Dimension of the conditioning vector (and each output chunk).
    num_params : int
        Number of modulation vectors to produce.
    """

    def __init__(self, cond_dim: int, num_params: int = 6) -> None:
        super().__init__()
        self.cond_dim = cond_dim
        self.num_params = num_params
        self.silu = nn.SiLU()
        self.linear = nn.Linear(cond_dim, num_params * cond_dim)

        # Zero-init the alpha-related slices (indices 2 and 5 for 6 params).
        # In the general case, every third param starting from index 2 is alpha.
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        # Re-initialise non-alpha slices with default init
        for i in range(num_params):
            if i % 3 != 2:  # not an alpha slot
                start = i * cond_dim
                end = (i + 1) * cond_dim
                nn.init.xavier_uniform_(self.linear.weight[start:end])
                nn.init.zeros_(self.linear.bias[start:end])

    def forward(self, t_emb: Tensor) -> tuple[Tensor, ...]:
        """Compute modulation parameters.

        Parameters
        ----------
        t_emb : Tensor
            Conditioning vector, shape ``(B, cond_dim)``.

        Returns
        -------
        tuple of Tensor
            *num_params* tensors each of shape ``(B, 1, cond_dim)``
            (unsqueezed for broadcasting over sequence length).
        """
        out = self.linear(self.silu(t_emb))  # (B, num_params * cond_dim)
        chunks = out.chunk(self.num_params, dim=-1)  # tuple of (B, cond_dim)
        return tuple(c.unsqueeze(1) for c in chunks)  # (B, 1, cond_dim)


def ada_layer_norm(
    x: Tensor,
    gamma: Tensor,
    beta: Tensor,
    layer_norm: nn.LayerNorm,
) -> Tensor:
    """Apply adaptive layer normalisation.

    ``AdaLN(x) = LayerNorm(x) * (1 + gamma) + beta``

    Parameters
    ----------
    x : Tensor
        Input tensor, shape ``(B, S, D)``.
    gamma, beta : Tensor
        Modulation tensors, shape ``(B, 1, D)``.
    layer_norm : nn.LayerNorm
        Standard layer norm module.

    Returns
    -------
    Tensor
        Same shape as *x*.
    """
    return layer_norm(x) * (1.0 + gamma) + beta
