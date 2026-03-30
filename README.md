# Latent Diffusion Text Model

A latent diffusion model for text generation, built on SQuAD-style question answering. The system encodes answers into a continuous latent space via a Sequence VAE, trains a conditional denoiser to generate latents from question+context conditioning, and decodes back to text.

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
                              (answerable?)                     (z → tokens)
```

**Pipeline stages:**
1. **VAE** — Sequence VAE compresses answer tokens into a continuous latent space
2. **Export Latents** — Encode the dataset with the frozen VAE; compute normalization stats
3. **Diffusion** — Train a conditional denoiser on the exported latents
4. **Null Classifier** — Binary MLP to detect unanswerable questions from latents
5. **Inference** — DDIM sampling + CFG → denormalize → null-check → VAE decode → text

## Prerequisites

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** (recommended) or pip

## Setup

```bash
# Clone the repository
git clone <repo-url>
cd "Latent Diffusion Text Model"

# Create virtual environment and install dependencies with uv
uv sync

# Or with pip
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Optional: Weights & Biases

All training pipelines log to [wandb](https://wandb.ai) if installed. Without it, logging gracefully degrades to no-ops.

```bash
uv pip install wandb
wandb login
```

## Configuration

Configs live in `configs/` and are merged in order. The base config sets shared paths and encoder settings; stage-specific configs override architecture and training params.

```
configs/
├── base.yaml                  # Paths, encoder, seed
├── vae/default.yaml           # VAE arch + training + quality gate
├── diffusion/default.yaml     # Denoiser arch + noise schedule + training
├── null_classifier/default.yaml
└── inference/default.yaml     # DDIM steps, CFG scale, decoding strategy
```

You can override any parameter via CLI with dot notation:

```bash
--vae_training.learning_rate 3e-4 --vae_arch.latent_dim 128
```

## Training

Training follows a strict sequential pipeline. Each stage depends on the output of the previous one.

### Step 1: Train the VAE

```bash
python -m src.pipelines.train_vae \
    --config configs/base.yaml configs/vae/default.yaml
```

This trains the Sequence VAE on SQuAD answer spans. Logs to wandb project `latent-diffusion-text-vae`.

**Key metrics tracked:** `train/loss`, `train/recon`, `train/kl`, `train/beta`, latent vitals (`latent/active_dims`, `latent/mu_mean`, `latent/z_std`, etc.)

### Step 2: Export Latents

After VAE training, encode the full dataset into latent vectors and run the quality gate:

```bash
python -m src.pipelines.export_latents \
    --config configs/base.yaml configs/vae/default.yaml \
    --vae_checkpoint checkpoints/<your_vae_checkpoint>.pt
```

This will:
- Encode train/val splits deterministically (using mu)
- Compute per-position normalization stats
- Run the quality gate (recon accuracy, mean KL, active dims, centroid distance)
- Save `latent_dataset_train.pt`, `latent_dataset_val.pt`, and `normalization_stats.pt` to the `latents/` directory

### Step 3: Train the Diffusion Model

```bash
python -m src.pipelines.train_diffusion \
    --config configs/base.yaml configs/vae/default.yaml configs/diffusion/default.yaml
```

Trains the conditional denoiser on precomputed latents. Logs to wandb project `latent-diffusion-text-diffusion`.

**Key metrics tracked:** `train/mse_loss`, `train/grad_norm`, `train/lr`, `val/mse`

### Step 4: Train the Null Classifier

```bash
python -m src.pipelines.train_null_classifier \
    --config configs/base.yaml configs/vae/default.yaml configs/null_classifier/default.yaml
```

Trains a binary MLP on the exported latents to distinguish answerable vs. unanswerable questions. Logs to wandb project `latent-diffusion-text-null-clf`.

**Key metrics tracked:** `train/bce_loss`, `val/accuracy`, `val/auc`

### Step 5: Evaluate

Run end-to-end evaluation on the validation set:

```bash
python -m src.evaluate \
    --vae_checkpoint checkpoints/<vae>.pt \
    --diffusion_checkpoint checkpoints/<diffusion>.pt \
    --null_classifier_checkpoint checkpoints/<null_clf>.pt
```

Reports Exact Match (EM), F1, and null prediction F1.

## Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run a specific test module
uv run pytest tests/models/vae/test_vae_integration.py -v
```

## Project Structure

```
├── configs/                    # YAML configuration files
│   ├── base.yaml
│   ├── vae/
│   ├── diffusion/
│   ├── null_classifier/
│   └── inference/
├── src/
│   ├── config/                 # Config schema, loader, validation
│   ├── data/                   # Tokenization, dataloaders, latent dataset
│   ├── models/
│   │   ├── vae/                # Encoder, decoder, reparameterize, loss, decoding
│   │   ├── diffusion/          # Denoiser, noise schedule, forward process, CFG
│   │   ├── encoder/            # Frozen BERT encoder + conditioning projection
│   │   ├── sampler/            # DDIM + CFG sampler
│   │   └── null_classifier.py
│   ├── pipelines/              # Training & inference entry points
│   │   ├── train_vae.py
│   │   ├── export_latents.py
│   │   ├── train_diffusion.py
│   │   ├── train_null_classifier.py
│   │   ├── quality_gate.py
│   │   └── generate.py
│   ├── training/               # Optimizer, scheduler, EMA, grad utils, checkpoints
│   ├── evaluation/             # SQuAD metrics, null metrics
│   └── utils/                  # Logging (wandb integration)
├── tests/                      # Unit and integration tests
├── pyproject.toml
└── README.md
```
