# RQ-VAE Explorer Design

Interactive training visualization tool for exploring RQ-VAE loss functions.

## Overview

A browser-based tool that lets you:
- Train an RQ-VAE on MNIST with a 2D latent space
- Watch codebook centers evolve in real-time
- Adjust loss function weights during training
- Monitor codebook health (active vs dead vectors)

## Stack

- **JAX + Flax** — model implementation
- **Gradio** — browser UI
- **uv** — package management
- **MNIST** — dataset (via tensorflow-datasets)

## Project Structure

```
rq_vae_loss_exploration/
├── pyproject.toml
├── src/
│   └── rq_vae_explorer/
│       ├── __init__.py
│       ├── model/
│       │   ├── __init__.py
│       │   ├── encoder.py      # Flax CNN encoder → 2D latent
│       │   ├── decoder.py      # 2D latent → 28x28 reconstruction
│       │   ├── quantizer.py    # Residual quantization logic
│       │   └── rqvae.py        # Combines encoder + quantizer + decoder
│       ├── training/
│       │   ├── __init__.py
│       │   ├── losses.py       # Reconstruction, commitment, codebook losses
│       │   ├── trainer.py      # Training loop with JAX jit
│       │   └── state.py        # Shared mutable state for UI ↔ trainer
│       ├── data/
│       │   ├── __init__.py
│       │   └── mnist.py        # MNIST dataset loading
│       └── ui/
│           ├── __init__.py
│           ├── app.py          # Gradio app entry point
│           ├── plots.py        # 2D codebook + latent visualization
│           └── controls.py     # Loss weight sliders
└── scripts/
    └── run.py
```

## Model Architecture

### Encoder (MNIST 28x28 → 2D latent)

```
Input: (batch, 28, 28, 1)
  → Conv 32 filters, 3x3, stride 2, ReLU  → (batch, 14, 14, 32)
  → Conv 64 filters, 3x3, stride 2, ReLU  → (batch, 7, 7, 64)
  → Flatten → Dense 128 → ReLU
  → Dense 2  → (batch, 2)
```

### Decoder (2D latent → 28x28 reconstruction)

```
Input: (batch, 2)
  → Dense 128 → ReLU
  → Dense 7*7*64 → Reshape → (batch, 7, 7, 64)
  → ConvTranspose 32 filters, 3x3, stride 2, ReLU → (batch, 14, 14, 32)
  → ConvTranspose 1 filter, 3x3, stride 2, Sigmoid → (batch, 28, 28, 1)
```

### Quantizer (residual quantization)

```
For each of D levels (default 2):
  1. Find nearest codebook vector to current residual
  2. Quantized output += codebook vector
  3. Residual = input - quantized output (for next level)

Codebook shape: (D, K, 2) → D levels × K=16 vectors × 2D
```

Straight-through estimator for gradients (gradients pass through quantization as identity).

**Default configuration:**
- K = 16 codebook vectors per level
- D = 2 residual quantization levels
- Both configurable via UI

## Loss Functions

```python
total_loss = recon_loss + λ_commit * commit_loss + λ_codebook * codebook_loss
```

| Loss | Formula | Purpose |
|------|---------|---------|
| Reconstruction | `MSE(input, reconstructed)` | Output quality |
| Commitment | `MSE(z_e, stop_gradient(z_q))` | Encoder commits to codebook |
| Codebook | `MSE(stop_gradient(z_e), z_q)` | Codebook moves toward encoder |

**Default weights:**
- `λ_commit = 0.25`
- `λ_codebook = 1.0`

Weights adjustable via UI sliders during training — changes take effect on the next batch.

## UI Layout

```
┌─────────────────────────────────────────────────────────────┐
│  [Start Training] [Pause] [Reset]            Step: 1234     │
├─────────────────────────┬───────────────────────────────────┤
│                         │                                   │
│   2D Codebook Plot      │   Sample Reconstructions          │
│   ● Active centers      │   (Input → Reconstructed)         │
│   ○ Dead centers        │                                   │
│   · Encoder outputs     │                                   │
│     (colored by digit)  │                                   │
│                         │                                   │
├─────────────────────────┼───────────────────────────────────┤
│  Codebook Health        │  Loss Plot (over time)            │
│  Level 1: 14/16 active  │  📉 recon ── commit ── codebook   │
│  Level 2: 12/16 active  │                                   │
├─────────────────────────┴───────────────────────────────────┤
│  λ_commit:    [====○================] 0.25                  │
│  λ_codebook:  [==========○==========] 1.00                  │
└─────────────────────────────────────────────────────────────┘
```

### Codebook health tracking

- Rolling window (last 100 batches) of assignment counts per vector
- Vector is "dead" if <1% of assignments in that window
- Displayed both visually (○ vs ●) and in stats panel

## Training Loop & UI Interaction

**Training runs in a background thread:**

```python
while training:
    batch = next(data_iterator)

    # Read current weights from shared state
    lambdas = state.get_lambdas()

    # JIT-compiled train step
    params, opt_state, metrics, codebook = train_step(
        params, opt_state, batch, lambdas
    )

    # Update shared state for UI
    state.update(
        codebook=codebook,
        losses=metrics,
        step=step,
        assignments=assignments,
    )

    # Sample reconstructions every 50 steps
    if step % 50 == 0:
        state.update(reconstructions=reconstruct(params, sample_batch))
```

**UI polling:** Gradio refreshes plots every 500ms while training.

**Thread safety:** Python threading with locks. JAX releases GIL during computation, so this works without multiprocessing.

## Dependencies

```toml
[project]
name = "rq-vae-explorer"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "jax[cpu]",
    "flax",
    "optax",
    "tensorflow-datasets",
    "gradio",
    "matplotlib",
    "numpy",
]

[project.optional-dependencies]
gpu = ["jax[cuda12]"]

[project.scripts]
rq-vae-explorer = "rq_vae_explorer.ui.app:main"
```

## Running

```bash
# Setup
uv sync

# Run (CPU)
uv run rq-vae-explorer

# With GPU
uv sync --extra gpu
uv run rq-vae-explorer
```

## Future Considerations

### Additional tunable parameters (planned)

- Codebook size (K) and quantization depth (D) adjustable via UI
- Learning rate
- Other loss formulas (MSE vs BCE, EMA updates vs gradient-based codebook learning)

### Dead codebook remediation (planned)

Currently: detection and visualization only.

Future experiments:
- Random reinitialization from encoder outputs
- Splitting popular codebook vectors
- Entropy regularization to encourage uniform usage
- EMA decay thresholds

The quantizer module is isolated specifically to accommodate these experiments without disrupting the rest of the codebase.
