# RQ-VAE Explorer

An interactive browser-based training visualization tool for exploring Residual Quantized Variational Autoencoders (RQ-VAE). Watch codebook centers evolve in real-time, adjust loss weights on-the-fly, and build intuition for how different hyperparameters affect training dynamics.

## Features

- **Real-Time 2D Codebook Visualization** - See codebook vectors and encoder outputs in a 2D latent space, with points colored by MNIST digit class
- **Interactive Loss Tuning** - Adjust commitment and codebook loss weights during training and observe immediate effects
- **Live Training Metrics** - Monitor reconstruction, commitment, and codebook losses as curves update in real-time
- **Codebook Health Monitoring** - Track active vs dead codebook vectors to detect codebook collapse
- **Sample Reconstructions** - View side-by-side comparisons of original images and their reconstructions

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/rq_vae_loss_exploration.git
cd rq_vae_loss_exploration

# Install dependencies with uv
uv sync

# For GPU support (CUDA 12)
uv sync --extra gpu
```

## Usage

```bash
# Launch the interactive web UI
uv run rq-vae-explorer
```

This starts a Gradio interface (typically at `http://127.0.0.1:7860`) where you can:

1. Click **Start** to begin training
2. Adjust **λ_commit** and **λ_codebook** sliders to tune loss weights
3. Watch the codebook visualization, loss curves, and reconstructions update in real-time
4. Click **Stop** to pause or **Reset** to reinitialize the model

## Architecture

### Model

The RQ-VAE maps 28×28 MNIST images to a 2D latent space for easy visualization:

```
Encoder: MNIST (28×28) → Conv layers → Dense → 2D latent
Quantizer: 2D latent → Residual quantization (2 levels × 16 vectors)
Decoder: Quantized latent → Dense → ConvTranspose layers → Reconstruction (28×28)
```

### Loss Function

```
total_loss = reconstruction_loss + λ_commit × commitment_loss + λ_codebook × codebook_loss
```

| Loss | Purpose |
|------|---------|
| **Reconstruction** | MSE between input and output images |
| **Commitment** | Encourages encoder outputs to stay close to codebook vectors |
| **Codebook** | Moves codebook vectors toward encoder outputs |

## Project Structure

```
src/rq_vae_explorer/
├── model/
│   ├── encoder.py      # CNN encoder (28×28 → 2D)
│   ├── decoder.py      # Transposed CNN decoder (2D → 28×28)
│   ├── quantizer.py    # Multi-level residual quantization
│   └── rqvae.py        # Full model
├── training/
│   ├── losses.py       # Loss function implementations
│   ├── trainer.py      # Background training loop
│   └── state.py        # Thread-safe UI ↔ trainer communication
├── data/
│   └── mnist.py        # Dataset loading and batching
└── ui/
    ├── app.py          # Main Gradio application
    ├── plots.py        # Visualization functions
    └── controls.py     # UI components
```

## Development

```bash
# Run tests
uv run pytest tests/ -v

# Run a specific test
uv run pytest tests/test_rqvae.py::test_name -v
```

## Tech Stack

- **JAX/Flax** - Neural network computation with JIT compilation
- **Optax** - Gradient-based optimization (Adam)
- **Gradio** - Interactive web interface
- **Matplotlib** - Plotting and visualization
- **TensorFlow Datasets** - MNIST data loading

## License

MIT
