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
    codebook: jnp.ndarray | None = None,
    z_q1: jnp.ndarray | None = None,
    lambda_commit: float = 0.25,
    lambda_codebook: float = 1.0,
    lambda_wasserstein: float = 0.0,
    sinkhorn_epsilon: float = 0.05,
) -> dict[str, jnp.ndarray]:
    """Compute all RQ-VAE loss components.

    Loss = recon + lambda_commit * commit + lambda_codebook * codebook
           + lambda_wasserstein * wasserstein

    Args:
        x: Original input images (batch, H, W, C)
        x_recon: Reconstructed images (batch, H, W, C)
        z_e: Encoder output before quantization (batch, latent_dim)
        z_q: Quantized latent vectors (batch, latent_dim)
        codebook: Full codebook (num_levels, num_codes, latent_dim)
        z_q1: Level 1 quantized output (batch, latent_dim)
        lambda_commit: Weight for commitment loss
        lambda_codebook: Weight for codebook loss
        lambda_wasserstein: Weight for Wasserstein loss (0 = disabled)
        sinkhorn_epsilon: Sinkhorn entropy regularization

    Returns:
        Dict with keys: total, recon, commit, codebook, wasserstein
    """
    # Reconstruction loss: MSE between input and reconstruction
    recon_loss = jnp.mean((x - x_recon) ** 2)

    # Commitment loss: encourages encoder to commit to codebook vectors
    commit_loss = jnp.mean((z_e - jax.lax.stop_gradient(z_q)) ** 2)

    # Codebook loss: moves codebook vectors toward encoder outputs
    codebook_loss = jnp.mean((jax.lax.stop_gradient(z_e) - z_q) ** 2)

    # Wasserstein loss: optimal transport between encoder outputs and codebook
    # Always compute when codebook/z_q1 provided (JAX JIT requires static control flow)
    # lambda_wasserstein=0 will zero out the contribution to total loss
    if codebook is not None and z_q1 is not None:
        # Level 1: transport between z_e and codebook[0]
        w_loss_l1 = sinkhorn_loss(z_e, codebook[0], sinkhorn_epsilon)

        # Level 2: transport between residuals and codebook[1]
        residuals = z_e - z_q1
        w_loss_l2 = sinkhorn_loss(residuals, codebook[1], sinkhorn_epsilon)

        wasserstein_loss = w_loss_l1 + w_loss_l2
    else:
        wasserstein_loss = jnp.array(0.0)

    # Total loss
    total_loss = (
        recon_loss
        + lambda_commit * commit_loss
        + lambda_codebook * codebook_loss
        + lambda_wasserstein * wasserstein_loss
    )

    return {
        "total": total_loss,
        "recon": recon_loss,
        "commit": commit_loss,
        "codebook": codebook_loss,
        "wasserstein": wasserstein_loss,
    }
