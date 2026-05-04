# Latent Diffusion Text Model

A latent diffusion model for text generation, built on SQuAD-style question answering. The system encodes answers into a continuous latent space via a VAE, trains a conditional denoiser to generate latents from question+context conditioning, and decodes back to text.

Two VAE backends are supported:
- **SequenceVAE** — custom transformer encoder/decoder trained from scratch
- **LangVAE** — pre-trained frozen BERT encoder + GPT-2/Llama decoder; only bottleneck projection layers are trained (~5% of parameters)

## Architecture

```
Question + Context ──► Frozen BERT Encoder ──► Conditioning Projection ─┐
                                                                        ▼
                                              Denoiser (Transformer) ◄── Noise Schedule
                                                        │
                                                        ▼
                                              Latent z₀ (denoised)
                                                        │
                                        ┌───────────────┼───────────────┐
                                        ▼                               ▼
                                Null Classifier                   VAE Decoder
                              (answerable?)              (z → text, via SequenceVAE
                                                          or LangVAE)
```

**Pipeline stages (same for both VAE backends):**
1. **VAE** — Compress answer text into 128-dim Gaussian latent vectors
2. **Export Latents** — Encode dataset with frozen VAE; compute per-dim normalization stats
3. **Diffusion** — Train a conditional denoiser on exported latents
4. **Null Classifier** — Binary MLP to detect unanswerable questions from latents
5. **Inference** — DDIM sampling + CFG → denormalize → null-check → VAE decode → text

## Prerequisites

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** (recommended) or pip

## Setup

```bash
# Clone the repository
git clone <repo-url>
cd Latent-Text-Diffusion-Model

# Create virtual environment and install dependencies
uv sync

# Or with pip
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Optional: Weights & Biases

SequenceVAE and diffusion training log to [wandb](https://wandb.ai). Without it, logging degrades gracefully to no-ops.

```bash
uv pip install wandb
wandb login
```

## Configuration

Configs live in `configs/` and are merged in order. Base config sets shared paths and encoder settings; stage-specific configs override architecture and training params.

```
configs/
├── base.yaml                      # Paths, encoder (BERT), seed
├── vae/
│   ├── default.yaml               # SequenceVAE arch + training + quality gate
│   └── langvae.yaml               # LangVAE encoder/decoder models + training
├── diffusion/default.yaml         # Denoiser arch + noise schedule + training
├── null_classifier/default.yaml
└── inference/default.yaml         # DDIM steps, CFG scale, decoding strategy
```

Override any parameter via CLI with dot notation:

```bash
--vae_arch.latent_dim 128 --langvae.decoder_model gpt2
```

---

## Training — SequenceVAE path

### Step 1A: Train SequenceVAE

```bash
python -m src.pipelines.train_vae \
    --config configs/base.yaml configs/vae/default.yaml
```

Trains encoder + decoder from scratch on SQuAD answer spans. Checkpoint saved to `checkpoints/vae_best.pt`. Logs to wandb project `latent-diffusion-text-vae`.

**Metrics tracked:** `train/loss`, `train/recon`, `train/kl`, `train/beta`, `latent/active_dims`, `latent/mu_mean`, `latent/z_std`

### Step 2A: Export Latents (SequenceVAE)

```bash
python -m src.pipelines.export_latents \
    --config configs/base.yaml configs/vae/default.yaml \
    --vae_checkpoint checkpoints/vae_best.pt
```

Outputs: `latents/latent_dataset_{train,val}.pt`, `latents/normalization_stats.pt`

Runs a quality gate (recon accuracy ≥ 0.85, active dims ≥ 10, centroid distance ≥ 0.5). Raises `RuntimeError` if gate fails.

---

## Training — LangVAE path

LangVAE trains only the bottleneck projection layers on top of frozen BERT (encoder) and GPT-2 (decoder). Training is faster and requires less GPU memory.

### Step 1B: Train LangVAE

```bash
python -m src.pipelines.train_langvae \
    --config configs/base.yaml configs/vae/langvae.yaml
```

Trains on SQuAD answer texts. Checkpoint directory saved to `checkpoints/langvae/`. Uses LangVAE's built-in cyclical KL scheduler.

**Key config options (`configs/vae/langvae.yaml`):**

| Key | Default | Description |
|-----|---------|-------------|
| `langvae.encoder_model` | `bert-base-cased` | HF encoder (frozen) |
| `langvae.decoder_model` | `gpt2` | HF decoder (frozen) |
| `langvae.latent_size` | `128` | Must match `vae_arch.latent_dim` |
| `langvae.num_epochs` | `10` | Training epochs |
| `langvae.kl_threshold` | `0.5` | Cyclical KL annealing threshold |

To use a larger decoder (better quality, more VRAM):

```bash
python -m src.pipelines.train_langvae \
    --config configs/base.yaml configs/vae/langvae.yaml \
    --langvae.decoder_model meta-llama/Llama-3.2-3B
```

### Step 2B: Export Latents (LangVAE)

Pass the **checkpoint directory** (not a `.pt` file) — `export_latents` auto-detects the VAE type:

```bash
python -m src.pipelines.export_latents \
    --config configs/base.yaml configs/vae/langvae.yaml \
    --vae_checkpoint checkpoints/langvae
```

Runs a round-trip quality gate (encode → decode → EM ≥ 0.50). Raises `RuntimeError` if gate fails.

---

## Training — Shared stages (Step 3–4)

These stages are identical regardless of which VAE was used.

### Step 3: Train the Diffusion Model

```bash
python -m src.pipelines.train_diffusion \
    --config configs/base.yaml configs/vae/default.yaml configs/diffusion/default.yaml
```

Trains the conditional denoiser on precomputed normalized latents. Logs to wandb project `latent-diffusion-text-diffusion`.

**Metrics tracked:** `train/mse_loss`, `train/grad_norm`, `train/lr`, `val/mse`

### Step 4: Train the Null Classifier

```bash
python -m src.pipelines.train_null_classifier \
    --config configs/base.yaml configs/vae/default.yaml configs/null_classifier/default.yaml
```

Trains a binary MLP on exported latents to distinguish answerable vs. unanswerable. Logs to wandb project `latent-diffusion-text-null-clf`.

**Metrics tracked:** `train/bce_loss`, `val/accuracy`, `val/auc`

### Step 5: Evaluate

**SequenceVAE:**
```bash
python -m src.evaluate \
    --vae_checkpoint checkpoints/vae_best.pt \
    --diffusion_checkpoint checkpoints/diffusion_best.pt \
    --null_classifier_checkpoint checkpoints/null_classifier_best.pt
```

**LangVAE:**
```bash
python -m src.evaluate \
    --vae_checkpoint checkpoints/langvae \
    --diffusion_checkpoint checkpoints/diffusion_best.pt \
    --null_classifier_checkpoint checkpoints/null_classifier_best.pt
```

Reports Exact Match (EM), F1, and null prediction F1 on the SQuAD v2 validation set.

---

## Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run a specific module
uv run pytest tests/models/vae/test_vae_integration.py -v
```

---

## Project Structure

```
├── configs/
│   ├── base.yaml
│   ├── vae/
│   │   ├── default.yaml           # SequenceVAE config
│   │   └── langvae.yaml           # LangVAE config
│   ├── diffusion/
│   ├── null_classifier/
│   └── inference/
├── src/
│   ├── config/                    # Config schema, loader, validation
│   ├── data/                      # Tokenization, dataloaders, latent dataset
│   ├── models/
│   │   ├── vae/
│   │   │   ├── vae.py             # SequenceVAE (encoder + decoder + output head)
│   │   │   ├── langvae_adapter.py # LangVAE wrapper (encode_from_texts / decode_sentences)
│   │   │   ├── encoder.py
│   │   │   ├── decoder.py
│   │   │   ├── loss.py
│   │   │   └── decoding.py
│   │   ├── diffusion/             # Denoiser, noise schedule, forward process, CFG
│   │   ├── encoder/               # Frozen BERT + conditioning projection
│   │   ├── sampler/               # DDIM + CFG sampler
│   │   └── null_classifier.py
│   ├── pipelines/
│   │   ├── train_vae.py           # Stage 1 — SequenceVAE
│   │   ├── train_langvae.py       # Stage 1 — LangVAE
│   │   ├── export_latents.py      # Stage 2 — auto-detects VAE type
│   │   ├── train_diffusion.py     # Stage 3
│   │   ├── train_null_classifier.py # Stage 4
│   │   ├── quality_gate.py        # SequenceVAE quality checks
│   │   └── generate.py            # Stage 5 — inference pipeline
│   ├── training/                  # Optimizer, EMA, grad utils, checkpoints
│   ├── evaluation/                # SQuAD metrics, null metrics
│   └── utils/                     # W&B logging
├── tests/
├── pyproject.toml
└── README.md
```

## VAE Backend Comparison

| | SequenceVAE | LangVAE |
|--|-------------|---------|
| **Encoder** | Trained from scratch | BERT (frozen) |
| **Decoder** | Trained from scratch | GPT-2 / Llama (frozen) |
| **Trainable params** | ~100% | ~5% |
| **Training time** | High | Low |
| **Checkpoint format** | `.pt` file | directory |
| **Quality gate** | Token recon + KL + active dims | Round-trip EM ≥ 0.50 |
| **Decoder output** | Token IDs → detokenize | Text strings directly |
