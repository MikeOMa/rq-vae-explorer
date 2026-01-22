"""Loss functions for RQ-VAE training."""

import jax
import jax.numpy as jnp


def compute_losses(
    x: jnp.ndarray,
    x_recon: jnp.ndarray,
    z_e: jnp.ndarray,
    z_q: jnp.ndarray,
    lambda_commit: float = 0.25,
    lambda_codebook: float = 1.0,
) -> dict[str, jnp.ndarray]:
    """Compute all RQ-VAE loss components.

    Loss = recon_loss + λ_commit * commit_loss + λ_codebook * codebook_loss

    Args:
        x: Original input images (batch, H, W, C)
        x_recon: Reconstructed images (batch, H, W, C)
        z_e: Encoder output before quantization (batch, latent_dim)
        z_q: Quantized latent vectors (batch, latent_dim)
        lambda_commit: Weight for commitment loss
        lambda_codebook: Weight for codebook loss

    Returns:
        Dict with keys: total, recon, commit, codebook
    """
    # Reconstruction loss: MSE between input and reconstruction
    recon_loss = jnp.mean((x - x_recon) ** 2)

    # Commitment loss: encourages encoder to commit to codebook vectors
    # Gradient only flows to encoder (z_q is stopped)
    commit_loss = jnp.mean((z_e - jax.lax.stop_gradient(z_q)) ** 2)

    # Codebook loss: moves codebook vectors toward encoder outputs
    # Gradient only flows to codebook (z_e is stopped)
    codebook_loss = jnp.mean((jax.lax.stop_gradient(z_e) - z_q) ** 2)

    # Total loss
    total_loss = recon_loss + lambda_commit * commit_loss + lambda_codebook * codebook_loss

    return {
        "total": total_loss,
        "recon": recon_loss,
        "commit": commit_loss,
        "codebook": codebook_loss,
    }
