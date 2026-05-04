"""Export precomputed latents from a trained VAE."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch

from src.config.schema import Config

logger = logging.getLogger(__name__)


def export_latents(
    config: Config,
    vae_checkpoint_path: str,
    device: Optional[torch.device] = None,
) -> None:
    """Encode the full SQuAD dataset with a frozen VAE and save latent files.

    Encoding is deterministic (uses mu only, no sampling).

    Steps
    -----
    1. Load frozen VAE from checkpoint.
    2. Encode train split, compute per-position per-dim normalisation stats.
    3. Encode val split.
    4. Run quality gate on the val split; raise RuntimeError if it fails.
    5. Save ``latent_dataset_train.pt``, ``latent_dataset_val.pt``,
       and ``normalization_stats.pt`` into ``config.paths.latent_dir``.

    Parameters
    ----------
    config : Config
    vae_checkpoint_path : str
        Path to a ``.pt`` checkpoint (SequenceVAE) or a directory (LangVAE).
    device : torch.device, optional
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(vae_checkpoint_path)

    # -------------------------------------------------------------- detect VAE type
    # A LangVAE checkpoint is a directory; a SequenceVAE checkpoint is a .pt file.
    use_langvae = ckpt_path.is_dir()

    if use_langvae:
        _export_latents_langvae(config, ckpt_path, device)
    else:
        _export_latents_sequence_vae(config, str(vae_checkpoint_path), device)


# ============================================================= SequenceVAE path


def _export_latents_sequence_vae(
    config: Config,
    vae_checkpoint_path: str,
    device: torch.device,
) -> None:
    from src.models.vae.vae import SequenceVAE
    from src.training.checkpoint import load_checkpoint
    from src.config.schema import Config as CfgCls

    ckpt = load_checkpoint(vae_checkpoint_path)
    saved_cfg = CfgCls.from_dict(ckpt["config"])

    from src.data.tokenization import create_tokenizer
    from src.data.loaders import create_squad_dataloaders

    tokenizer = create_tokenizer(saved_cfg.encoder.model_name)
    vocab_size = len(tokenizer)

    pretrained_emb = torch.randn(vocab_size, saved_cfg.vae_arch.embed_dim) * 0.5
    pretrained_emb = pretrained_emb.to(device)

    vae = SequenceVAE(saved_cfg.vae_arch, pretrained_embeddings=pretrained_emb).to(device)
    vae.load_state_dict(ckpt["model_state_dict"])
    vae.eval()

    train_loader, val_loader = create_squad_dataloaders(saved_cfg, tokenizer)

    def _encode_split(loader) -> dict:
        latents_list, context_ids_list, context_mask_list = [], [], []
        question_ids_list, question_mask_list, is_ans_list = [], [], []

        with torch.no_grad():
            for batch in loader:
                answer_ids = batch["answer_ids"].to(device)
                answer_mask = batch["answer_mask"].to(device)

                _, mu, _ = vae.encode(answer_ids, answer_mask, deterministic=True)
                latents_list.append(mu.cpu())

                context_ids_list.append(batch["context_ids"])
                context_mask_list.append(batch["context_mask"])
                question_ids_list.append(batch["question_ids"])
                question_mask_list.append(batch["question_mask"])
                is_ans = batch["is_answerable"]
                if not isinstance(is_ans, torch.Tensor):
                    is_ans = torch.tensor(is_ans, dtype=torch.bool)
                is_ans_list.append(is_ans.cpu())

        return {
            "latents_raw": torch.cat(latents_list, dim=0),
            "context_ids": torch.cat(context_ids_list, dim=0),
            "context_mask": torch.cat(context_mask_list, dim=0),
            "question_ids": torch.cat(question_ids_list, dim=0),
            "question_mask": torch.cat(question_mask_list, dim=0),
            "is_answerable": torch.cat(is_ans_list, dim=0),
        }

    logger.info("Encoding train split…")
    train_data = _encode_split(train_loader)

    logger.info("Encoding val split…")
    val_data = _encode_split(val_loader)

    _normalise_and_save(config, train_data, val_data)

    # Quality gate uses SequenceVAE-specific logit format
    from src.pipelines.quality_gate import run_quality_gate

    passed, report = run_quality_gate(vae, val_loader, saved_cfg, device)
    if not passed:
        failed = [k for k, v in report.items() if not v["passed"]]
        raise RuntimeError(f"Quality gate failed on checks: {failed}. Report: {report}")
    logger.info("Quality gate passed.")


# ================================================================ LangVAE path


def _export_latents_langvae(
    config: Config,
    ckpt_dir: Path,
    device: torch.device,
) -> None:
    from src.models.vae.langvae_adapter import LangVAEAdapter
    from src.data.tokenization import create_tokenizer
    from src.data.loaders import create_squad_dataloaders

    lc = config.langvae
    adapter = LangVAEAdapter.from_pretrained(
        ckpt_dir, device=device, latent_size=lc.latent_size, max_len=lc.max_len
    )

    # We still need the context/question IDs from the SQuAD dataloader.
    # Answer texts come from batch["answer_text"] (already in SQuADDataset).
    tokenizer = create_tokenizer(config.encoder.model_name)
    train_loader, val_loader = create_squad_dataloaders(config, tokenizer)

    def _encode_split(loader) -> dict:
        latents_list, context_ids_list, context_mask_list = [], [], []
        question_ids_list, question_mask_list, is_ans_list = [], [], []

        for batch in loader:
            answer_texts: list[str] = list(batch["answer_text"])

            _, mu, _ = adapter.encode_from_texts(answer_texts, deterministic=True)
            latents_list.append(mu.cpu())

            context_ids_list.append(batch["context_ids"])
            context_mask_list.append(batch["context_mask"])
            question_ids_list.append(batch["question_ids"])
            question_mask_list.append(batch["question_mask"])
            is_ans = batch["is_answerable"]
            if not isinstance(is_ans, torch.Tensor):
                is_ans = torch.tensor(is_ans, dtype=torch.bool)
            is_ans_list.append(is_ans.cpu())

        return {
            "latents_raw": torch.cat(latents_list, dim=0),
            "context_ids": torch.cat(context_ids_list, dim=0),
            "context_mask": torch.cat(context_mask_list, dim=0),
            "question_ids": torch.cat(question_ids_list, dim=0),
            "question_mask": torch.cat(question_mask_list, dim=0),
            "is_answerable": torch.cat(is_ans_list, dim=0),
        }

    logger.info("LangVAE: encoding train split…")
    train_data = _encode_split(train_loader)

    logger.info("LangVAE: encoding val split…")
    val_data = _encode_split(val_loader)

    _normalise_and_save(config, train_data, val_data)

    # LangVAE quality gate: round-trip EM on a small val subset
    _run_langvae_quality_gate(adapter, val_loader)


def _run_langvae_quality_gate(adapter, val_loader, n_batches: int = 5) -> None:
    """Simple round-trip quality gate: encode → decode → check EM ≥ 0.50."""
    from src.evaluation.squad_metrics import compute_squad_metrics

    preds, refs = [], []
    for i, batch in enumerate(val_loader):
        if i >= n_batches:
            break
        texts: list[str] = list(batch["answer_text"])
        _, mu, _ = adapter.encode_from_texts(texts, deterministic=True)
        decoded = adapter.decode_sentences(mu)
        preds.extend(decoded)
        # all_answer_texts may be absent on train loader; fall back to answer_text
        if "all_answer_texts" in batch:
            refs.extend(list(batch["all_answer_texts"]))
        else:
            refs.extend([[t] for t in texts])

    metrics = compute_squad_metrics(preds, refs)
    em = metrics["em"]
    logger.info("LangVAE round-trip quality gate: EM=%.3f (threshold=0.50)", em)
    if em < 0.50:
        raise RuntimeError(
            f"LangVAE quality gate failed: round-trip EM={em:.3f} < 0.50. "
            "Check model training or latent_size configuration."
        )
    logger.info("LangVAE quality gate passed.")


# ================================================================ shared helpers


def _normalise_and_save(config: Config, train_data: dict, val_data: dict) -> None:
    """Compute normalisation stats from train, apply to both splits, save."""
    train_latents = train_data["latents_raw"]
    norm_mean = train_latents.mean(dim=0)
    norm_std = train_latents.std(dim=0).clamp(min=1e-6)
    norm_stats = {"mean": norm_mean, "std": norm_std}

    def _normalize(data: dict) -> dict:
        z_norm = (data["latents_raw"] - norm_mean) / norm_std
        result = {"z_normalized": z_norm}
        result.update({k: v for k, v in data.items() if k != "latents_raw"})
        return result

    train_data = _normalize(train_data)
    val_data = _normalize(val_data)

    out_dir = Path(config.paths.latent_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(train_data, out_dir / "latent_dataset_train.pt")
    torch.save(val_data, out_dir / "latent_dataset_val.pt")
    torch.save(norm_stats, out_dir / "normalization_stats.pt")
    logger.info("Saved latents to %s", out_dir)
