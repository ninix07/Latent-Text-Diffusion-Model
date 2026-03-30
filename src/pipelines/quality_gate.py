"""Quality gate: checks latent space quality before exporting latents."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from src.config.schema import Config


def run_quality_gate(
    vae,
    dataloader,
    config: Config,
    device: torch.device,
) -> Tuple[bool, dict]:
    """Evaluate latent quality and return a pass/fail report.

    Checks performed
    ----------------
    1. ``recon_accuracy`` — fraction of tokens correctly reconstructed.
    2. ``mean_kl`` — mean KL divergence across all positions/dims.
    3. ``active_dims`` — number of latent dims with variance > threshold.
    4. ``centroid_distance`` — L2 distance between answerable / unanswerable
       latent centroids (mean over sequence positions).

    Parameters
    ----------
    vae : SequenceVAE
        Model in eval mode.
    dataloader : DataLoader
        Provides batches with keys ``answer_ids``, ``answer_mask``,
        ``is_answerable``.
    config : Config
    device : torch.device

    Returns
    -------
    (all_passed, report)
        ``report`` maps check name -> ``{value, passed, threshold}``.
    """
    qg = config.quality_gate
    vae.eval()

    all_token_correct = 0
    all_token_total = 0
    all_kl: list[torch.Tensor] = []
    all_mu: list[torch.Tensor] = []  # (B, L, D) stacked
    all_answerable: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in dataloader:
            answer_ids = batch["answer_ids"].to(device)
            answer_mask = batch["answer_mask"].to(device)
            is_answerable = batch["is_answerable"].to(device)

            logits, z, mu, log_var, loss_dict = vae(answer_ids, answer_mask)

            # --- Token reconstruction accuracy ---
            pred_ids = logits.argmax(dim=-1)  # (B, L)
            correct = ((pred_ids == answer_ids) * answer_mask).sum()
            total = answer_mask.sum()
            all_token_correct += correct.item()
            all_token_total += total.item()

            # --- KL ---
            # kl from loss_dict is already mean-reduced
            # recompute per-sample for aggregation
            kl_per_pos = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())  # (B,L,D)
            mask_exp = answer_mask.unsqueeze(-1).float()
            kl_masked = (kl_per_pos * mask_exp).sum() / mask_exp.sum().clamp(min=1)
            all_kl.append(kl_masked.cpu())

            # --- Collect mu for active dims + centroid distance ---
            # Collapse to (B, D) by mean-pooling over sequence
            mu_pooled = (mu * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1)  # (B, D)
            all_mu.append(mu_pooled.cpu())
            all_answerable.append(is_answerable.cpu())

    recon_accuracy = all_token_correct / max(all_token_total, 1)
    mean_kl = float(torch.stack(all_kl).mean())

    mu_cat = torch.cat(all_mu, dim=0)         # (N, D)
    ans_cat = torch.cat(all_answerable, dim=0)  # (N,)

    # Active dims: variance across samples
    variances = mu_cat.var(dim=0)  # (D,)
    active_dims = int((variances > qg.active_dim_variance_threshold).sum().item())

    # Centroid distance
    ans_mask = ans_cat.bool()
    unans_mask = ~ans_mask
    if ans_mask.sum() > 0 and unans_mask.sum() > 0:
        centroid_ans = mu_cat[ans_mask].mean(dim=0)
        centroid_unans = mu_cat[unans_mask].mean(dim=0)
        centroid_distance = float(torch.norm(centroid_ans - centroid_unans).item())
    else:
        centroid_distance = 0.0

    report = {
        "recon_accuracy": {
            "value": recon_accuracy,
            "passed": recon_accuracy >= qg.min_recon_accuracy,
            "threshold": qg.min_recon_accuracy,
        },
        "mean_kl": {
            "value": mean_kl,
            "passed": mean_kl >= qg.min_mean_kl,
            "threshold": qg.min_mean_kl,
        },
        "active_dims": {
            "value": active_dims,
            "passed": active_dims >= qg.min_active_dims,
            "threshold": qg.min_active_dims,
        },
        "centroid_distance": {
            "value": centroid_distance,
            "passed": centroid_distance >= qg.min_centroid_distance,
            "threshold": qg.min_centroid_distance,
        },
    }

    all_passed = all(v["passed"] for v in report.values())
    return all_passed, report
