import jax
import jax.numpy as jnp
from rq_vae_explorer.model.quantizer import ResidualQuantizer


def test_quantizer_output_shape():
    quantizer = ResidualQuantizer(num_codes=16, num_levels=2, latent_dim=2)

    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.zeros((1, 2))
    variables = quantizer.init(rng, dummy_input)

    batch = jax.random.normal(jax.random.PRNGKey(1), (4, 2))
    output, aux = quantizer.apply(variables, batch)

    assert output.shape == (4, 2)
    assert "indices" in aux
    assert "codebook" in aux


def test_quantizer_indices_valid():
    quantizer = ResidualQuantizer(num_codes=16, num_levels=2, latent_dim=2)

    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.zeros((1, 2))
    variables = quantizer.init(rng, dummy_input)

    batch = jax.random.normal(jax.random.PRNGKey(1), (4, 2))
    _, aux = quantizer.apply(variables, batch)

    indices = aux["indices"]
    # Shape: (batch, num_levels)
    assert indices.shape == (4, 2)
    # All indices should be in valid range
    assert jnp.all(indices >= 0)
    assert jnp.all(indices < 16)


def test_quantizer_codebook_shape():
    quantizer = ResidualQuantizer(num_codes=16, num_levels=2, latent_dim=2)

    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.zeros((1, 2))
    variables = quantizer.init(rng, dummy_input)

    batch = jax.random.normal(jax.random.PRNGKey(1), (4, 2))
    _, aux = quantizer.apply(variables, batch)

    codebook = aux["codebook"]
    # Shape: (num_levels, num_codes, latent_dim)
    assert codebook.shape == (2, 16, 2)


def test_quantizer_returns_z_q1():
    """Test that quantizer returns level 1 quantized output."""
    quantizer = ResidualQuantizer(num_codes=16, num_levels=2, latent_dim=2)
    rng = jax.random.PRNGKey(42)
    z = jax.random.normal(rng, (4, 2))

    params = quantizer.init(rng, z)
    _, aux = quantizer.apply(params, z)

    assert "z_q1" in aux, "aux should contain z_q1"
    assert aux["z_q1"].shape == (4, 2), "z_q1 should have same shape as input"
    # z_q1 should be different from z_q (unless only 1 level)
    # Just check it exists and has right shape
