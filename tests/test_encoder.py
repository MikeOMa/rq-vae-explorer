import jax
import jax.numpy as jnp
from rq_vae_explorer.model.encoder import Encoder


def test_encoder_output_shape():
    encoder = Encoder(latent_dim=2)

    # Initialize with dummy input
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.zeros((1, 28, 28, 1))
    params = encoder.init(rng, dummy_input)

    # Test forward pass
    batch = jnp.zeros((4, 28, 28, 1))
    output = encoder.apply(params, batch)

    assert output.shape == (4, 2)


def test_encoder_different_latent_dims():
    for latent_dim in [2, 4, 8]:
        encoder = Encoder(latent_dim=latent_dim)
        rng = jax.random.PRNGKey(0)
        dummy_input = jnp.zeros((1, 28, 28, 1))
        params = encoder.init(rng, dummy_input)

        batch = jnp.zeros((2, 28, 28, 1))
        output = encoder.apply(params, batch)

        assert output.shape == (2, latent_dim)
