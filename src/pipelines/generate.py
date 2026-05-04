"""End-to-end inference pipeline for the Latent Diffusion Text Model."""

from __future__ import annotations

from typing import Optional

import torch


class GenerationPipeline:
    """Full inference pipeline: encode → diffuse → classify → decode."""

    def __init__(
        self,
        encoder,
        projection,
        vae,
        sampler,
        null_classifier,
        normalization_stats: dict,
        tokenizer,
        config,
    ):
        self.encoder = encoder
        self.projection = projection
        self.vae = vae
        self.sampler = sampler
        self.null_classifier = null_classifier
        self.norm_mean = normalization_stats["mean"]
        self.norm_std = normalization_stats["std"]
        self.tokenizer = tokenizer
        self.config = config

    @torch.no_grad()
    def generate(
        self,
        context_ids: torch.Tensor,
        context_mask: torch.Tensor,
        question_ids: torch.Tensor,
        question_mask: torch.Tensor,
    ) -> dict:
        """Generate an answer for a single example (or batch).

        Returns
        -------
        dict
            Keys: answer_text (str), is_answerable (bool), confidence (float).
        """
        device = next(self.encoder.parameters()).device
        context_ids = context_ids.to(device)
        context_mask = context_mask.to(device)
        question_ids = question_ids.to(device)
        question_mask = question_mask.to(device)

        # 1. Encode question + context
        h_q = self.encoder.encode(question_ids, question_mask)
        h_c = self.encoder.encode(context_ids, context_mask)
        conditioning, cond_mask = self.projection(
            h_q, question_mask.bool(), h_c, context_mask.bool()
        )

        # 2. DDIM sampling (with optional best-of-N)
        B = context_ids.size(0)
        K = self.config.vae_arch.num_latent_tokens
        D = self.config.vae_arch.latent_dim
        z_shape = (B, K, D)

        def denoiser_fn(z_t, t_tensor):
            return self.sampler.predict_noise(z_t, t_tensor, conditioning, cond_mask)

        # 3. Denormalize helper. Diffusion runs in normalized latent space;
        # we denormalize only for the VAE decoder. The null classifier was
        # trained on z_normalized, so it must see normalized latents too.
        mean = self.norm_mean.to(device)
        std = self.norm_std.to(device)

        n_samples = self.config.inference.best_of_n
        if n_samples > 1:
            from src.models.sampler.best_of_n import best_of_n_sample

            def _generate_z0_norm() -> torch.Tensor:
                return self.sampler.ddim.sample(denoiser_fn, z_shape, device)

            z0_norm, confidence = best_of_n_sample(
                _generate_z0_norm, n_samples, self.null_classifier
            )
        else:
            z0_norm = self.sampler.ddim.sample(denoiser_fn, z_shape, device)
            confidence = self.null_classifier(z0_norm)  # (B,)
        z0 = z0_norm * std + mean
        threshold = self.config.null_classifier.threshold
        is_answerable = (confidence >= threshold).tolist()
        confidence_vals = confidence.tolist()

        # 4. VAE decode → answer strings
        # LangVAEAdapter exposes decode_sentences() and returns text directly.
        # SequenceVAE exposes decode_to_tokens() and needs detokenization.
        if hasattr(self.vae, "decode_sentences"):
            answer_texts = self.vae.decode_sentences(z0)
        else:
            strategy = self.config.inference.decoding_strategy
            # Beam search not yet implemented; fall back to greedy.
            if strategy == "beam_search":
                strategy = "greedy"
            token_ids = self.vae.decode_to_tokens(
                z0,
                strategy=strategy,
                temperature=self.config.inference.nucleus_temperature,
                top_p=self.config.inference.nucleus_top_p,
            )
            answer_texts = [
                self.tokenizer.decode(token_ids[i].tolist(), skip_special_tokens=True)
                for i in range(B)
            ]

        # 5. Package results
        results = []
        for i in range(B):
            ans_bool = (
                bool(is_answerable[i])
                if isinstance(is_answerable, list)
                else bool(is_answerable)
            )
            conf = (
                float(confidence_vals[i])
                if isinstance(confidence_vals, list)
                else float(confidence_vals)
            )
            text = answer_texts[i] if ans_bool else ""
            results.append(
                {"answer_text": text, "is_answerable": ans_bool, "confidence": conf}
            )

        return results

    @torch.no_grad()
    def generate_batch(self, batch: dict) -> list[dict]:
        """Generate answers for a batch dict."""
        return self.generate(
            batch["context_ids"],
            batch["context_mask"],
            batch["question_ids"],
            batch["question_mask"],
        )
