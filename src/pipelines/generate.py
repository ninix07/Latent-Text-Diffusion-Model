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

        # 2. DDIM sampling
        B = context_ids.size(0)
        L = self.config.vae_arch.max_answer_len
        D = self.config.vae_arch.latent_dim
        z_shape = (B, L, D)

        def denoiser_fn(z_t, t_tensor):
            return self.sampler.denoiser(z_t, t_tensor, conditioning, cond_mask)

        z0_normalized = self.sampler.ddim.sample(denoiser_fn, z_shape, device)

        # 3. Denormalize
        mean = self.norm_mean.to(device)
        std = self.norm_std.to(device)
        z0 = z0_normalized * std + mean

        # 4. Null classification
        confidence = self.null_classifier(z0)  # (B,)
        threshold = self.config.null_classifier.threshold
        is_answerable = (confidence >= threshold).tolist()
        confidence_vals = confidence.tolist()

        # 5. VAE decode → token IDs
        mask = torch.ones(B, L, dtype=torch.long, device=device)
        strategy = self.config.inference.decoding_strategy
        token_ids = self.vae.decode_to_tokens(
            z0, mask,
            strategy=strategy,
            beam_width=self.config.inference.beam_width,
        )

        # 6. Detokenize
        results = []
        for i in range(B):
            ans_bool = bool(is_answerable[i]) if isinstance(is_answerable, list) else bool(is_answerable)
            conf = float(confidence_vals[i]) if isinstance(confidence_vals, list) else float(confidence_vals)
            if ans_bool:
                ids = token_ids[i].tolist()
                text = self.tokenizer.decode(ids, skip_special_tokens=True)
            else:
                text = ""
            results.append({"answer_text": text, "is_answerable": ans_bool, "confidence": conf})

        return results[0] if B == 1 else results

    @torch.no_grad()
    def generate_batch(self, batch: dict) -> list[dict]:
        """Generate answers for a batch dict."""
        return self.generate(
            batch["context_ids"],
            batch["context_mask"],
            batch["question_ids"],
            batch["question_mask"],
        )
