"""VAE loss utilities: reconstruction + KL with beta warmup."""

from __future__ import annotations

import torch
import torch.nn.functional as F

def compute_vae_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    mask: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    beta: float,
    free_bits: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute total VAE loss = recon + beta * kl.

    Parameters
    ----------
    logits : (B, L, V)
    target_ids : (B, L)
    mask : (B, L) — 1 for real tokens, 0 for padding.
    mu, log_var : (B, L, latent_dim)
    beta : float
    free_bits : float
        Minimum KL per latent dimension (prevents full posterior collapse).

    Returns
    -------
    (total_loss, recon_loss, kl_loss)
    """
    # Reconstruction: cross-entropy ignoring padding
    B, L, V = logits.shape
    logits_flat = logits.reshape(-1, V)
    target_flat = target_ids.reshape(-1)
    mask_flat = mask.reshape(-1).float()

    ce = F.cross_entropy(logits_flat, target_flat, reduction="none")
    recon = (ce * mask_flat).sum() / mask_flat.sum().clamp(min=1)

    # KL: masked mean over real (non-padding) positions, sum over latent dims.
    # Averaging over padding positions adds noise because the encoder still
    # produces mu/log_var at those positions despite them carrying no content.
    mask_3d = mask.unsqueeze(-1).float()  # (B, L, 1)
    kl_raw = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())  # (B, L, D)
    kl_per_dim = (kl_raw * mask_3d).sum(dim=(0, 1)) / mask_3d.sum().clamp(min=1)  # (D,)
    if free_bits > 0.0:
        kl = kl_per_dim.clamp(min=free_bits).sum()
    else:
        kl = kl_per_dim.sum()

    total = recon + beta * kl
    return total, recon, kl


def compute_beta(
    step: int,
    start: float,
    end: float,
    warmup_steps: int,
) -> float:
    """Linear beta warmup from *start* to *end* over *warmup_steps*.

    Returns *start* at step 0 and *end* at step >= warmup_steps.
    """
    if warmup_steps <= 0:
        return end
    t = min(step / warmup_steps, 1.0)
    return start + (end - start) * t
