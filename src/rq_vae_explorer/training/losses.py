"""Loss functions for RQ-VAE training."""

import jax
import jax.numpy as jnp


def sinkhorn_loss(
    points: jnp.ndarray,
    codebook: jnp.ndarray,
    epsilon: float = 0.05,
    num_iters: int = 20,
) -> jnp.ndarray:
    """Compute Sinkhorn (entropy-regularized optimal transport) loss.

    Args:
        points: Batch of points (batch_size, latent_dim)
        codebook: Codebook vectors (num_codes, latent_dim)
        epsilon: Entropy regularization strength (lower = sharper)
        num_iters: Number of Sinkhorn iterations

    Returns:
        Scalar Wasserstein distance approximation
    """
    batch_size = points.shape[0]
    num_codes = codebook.shape[0]

    # Cost matrix: squared Euclidean distances (batch_size, num_codes)
    diff = points[:, None, :] - codebook[None, :, :]  # (batch, codes, dim)
    C = jnp.sum(diff**2, axis=-1)  # (batch, codes)

    # Kernel matrix
    K = jnp.exp(-C / epsilon)

    # Uniform marginals
    a = jnp.ones(batch_size) / batch_size
    b = jnp.ones(num_codes) / num_codes

    # Sinkhorn iterations
    u = jnp.ones(batch_size)
    v = jnp.ones(num_codes)

    for _ in range(num_iters):
        u = a / (K @ v + 1e-8)
        v = b / (K.T @ u + 1e-8)

    # Transport plan and cost
    transport = u[:, None] * K * v[None, :]
    loss = jnp.sum(transport * C)

    return loss


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
    total_loss = (
        recon_loss + lambda_commit * commit_loss + lambda_codebook * codebook_loss
    )

    return {
        "total": total_loss,
        "recon": recon_loss,
        "commit": commit_loss,
        "codebook": codebook_loss,
    }
