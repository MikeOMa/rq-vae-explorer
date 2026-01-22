import jax
import jax.numpy as jnp
from rq_vae_explorer.model import RQVAE


def test_rqvae_forward_pass():
    model = RQVAE(latent_dim=2, num_codes=16, num_levels=2)

    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.zeros((1, 28, 28, 1))
    variables = model.init(rng, dummy_input)

    batch = jax.random.uniform(jax.random.PRNGKey(1), (4, 28, 28, 1))
    output, aux = model.apply(variables, batch)

    # Reconstruction should have same shape as input
    assert output.shape == (4, 28, 28, 1)

    # Check auxiliary outputs
    assert "z_e" in aux
    assert "z_q" in aux
    assert "indices" in aux
    assert "codebook" in aux

    assert aux["z_e"].shape == (4, 2)
    assert aux["z_q"].shape == (4, 2)


def test_rqvae_reconstruction_range():
    model = RQVAE(latent_dim=2, num_codes=16, num_levels=2)

    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.zeros((1, 28, 28, 1))
    variables = model.init(rng, dummy_input)

    batch = jax.random.uniform(jax.random.PRNGKey(1), (4, 28, 28, 1))
    output, _ = model.apply(variables, batch)

    # Output should be in [0, 1] due to sigmoid
    assert output.min() >= 0.0
    assert output.max() <= 1.0
