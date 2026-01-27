import jax.numpy as jnp
from rq_vae_explorer.training.losses import compute_losses


def test_compute_losses_returns_all_components():
    # Mock inputs
    x = jnp.ones((4, 28, 28, 1)) * 0.5
    x_recon = jnp.ones((4, 28, 28, 1)) * 0.6
    z_e = jnp.ones((4, 2))
    z_q = jnp.ones((4, 2)) * 1.1

    losses = compute_losses(
        x=x,
        x_recon=x_recon,
        z_e=z_e,
        z_q=z_q,
        lambda_commit=0.25,
        lambda_codebook=1.0,
    )

    assert "total" in losses
    assert "recon" in losses
    assert "commit" in losses
    assert "codebook" in losses

    # All losses should be non-negative
    assert losses["total"] >= 0
    assert losses["recon"] >= 0
    assert losses["commit"] >= 0
    assert losses["codebook"] >= 0


def test_compute_losses_weights_affect_total():
    x = jnp.ones((4, 28, 28, 1)) * 0.5
    x_recon = jnp.ones((4, 28, 28, 1)) * 0.6
    z_e = jnp.ones((4, 2))
    z_q = jnp.ones((4, 2)) * 1.1

    losses_low = compute_losses(
        x, x_recon, z_e, z_q, lambda_commit=0.1, lambda_codebook=0.1
    )
    losses_high = compute_losses(
        x, x_recon, z_e, z_q, lambda_commit=1.0, lambda_codebook=1.0
    )

    # Higher weights should give higher total loss
    assert losses_high["total"] > losses_low["total"]

    # Component losses should be the same
    assert jnp.isclose(losses_high["recon"], losses_low["recon"])
    assert jnp.isclose(losses_high["commit"], losses_low["commit"])
    assert jnp.isclose(losses_high["codebook"], losses_low["codebook"])


def test_sinkhorn_loss_basic():
    """Sinkhorn loss computes optimal transport between points and codebook."""
    from rq_vae_explorer.training.losses import sinkhorn_loss

    # Simple case: 4 points, 4 codebook entries
    points = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    codebook = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    loss = sinkhorn_loss(points, codebook, epsilon=0.05, num_iters=20)

    # Perfect match should give near-zero loss
    assert loss >= 0
    assert loss < 0.1


def test_sinkhorn_loss_nonzero_for_mismatch():
    """Sinkhorn loss is higher when points don't match codebook."""
    from rq_vae_explorer.training.losses import sinkhorn_loss

    # Use moderate distances that work with epsilon=0.05
    points = jnp.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1]])
    codebook = jnp.array([[1.0, 1.0], [1.1, 1.0], [1.0, 1.1], [1.1, 1.1]])

    loss = sinkhorn_loss(points, codebook, epsilon=0.1, num_iters=20)

    # Offset should give positive loss
    assert loss > 0.5


def test_compute_losses_with_wasserstein():
    """compute_losses includes wasserstein when lambda > 0."""
    x = jnp.ones((4, 28, 28, 1)) * 0.5
    x_recon = jnp.ones((4, 28, 28, 1)) * 0.6
    z_e = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    z_q = z_e * 1.1
    z_q1 = z_e * 0.5
    codebook = jnp.ones((2, 16, 2)) * 0.5  # (num_levels, num_codes, latent_dim)

    losses = compute_losses(
        x=x,
        x_recon=x_recon,
        z_e=z_e,
        z_q=z_q,
        codebook=codebook,
        z_q1=z_q1,
        lambda_commit=0.25,
        lambda_codebook=1.0,
        lambda_wasserstein=0.5,
        sinkhorn_epsilon=0.05,
    )

    assert "wasserstein" in losses
    assert losses["wasserstein"] >= 0


def test_compute_losses_wasserstein_not_in_total_when_disabled():
    """wasserstein loss doesn't contribute to total when lambda_wasserstein=0."""
    x = jnp.ones((4, 28, 28, 1)) * 0.5
    x_recon = jnp.ones((4, 28, 28, 1)) * 0.6
    z_e = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    z_q = z_e * 1.1
    z_q1 = z_e * 0.5
    codebook = jnp.ones((2, 16, 2)) * 0.5

    losses_disabled = compute_losses(
        x=x,
        x_recon=x_recon,
        z_e=z_e,
        z_q=z_q,
        codebook=codebook,
        z_q1=z_q1,
        lambda_commit=0.25,
        lambda_codebook=1.0,
        lambda_wasserstein=0.0,
        sinkhorn_epsilon=0.05,
    )

    losses_enabled = compute_losses(
        x=x,
        x_recon=x_recon,
        z_e=z_e,
        z_q=z_q,
        codebook=codebook,
        z_q1=z_q1,
        lambda_commit=0.25,
        lambda_codebook=1.0,
        lambda_wasserstein=0.5,
        sinkhorn_epsilon=0.05,
    )

    # Wasserstein is computed but doesn't contribute when lambda=0
    assert losses_enabled["total"] > losses_disabled["total"]
    # Component losses should be the same
    assert jnp.isclose(losses_enabled["recon"], losses_disabled["recon"])
    assert jnp.isclose(losses_enabled["commit"], losses_disabled["commit"])
    assert jnp.isclose(losses_enabled["codebook"], losses_disabled["codebook"])
