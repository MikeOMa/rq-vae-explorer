import jax
import jax.numpy as jnp
from rq_vae_explorer.model.decoder import Decoder


def test_decoder_output_shape():
    decoder = Decoder()

    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.zeros((1, 2))
    params = decoder.init(rng, dummy_input)

    batch = jnp.zeros((4, 2))
    output = decoder.apply(params, batch)

    assert output.shape == (4, 28, 28, 1)


def test_decoder_output_range():
    decoder = Decoder()

    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.zeros((1, 2))
    params = decoder.init(rng, dummy_input)

    # Random latent vectors
    batch = jax.random.normal(jax.random.PRNGKey(1), (4, 2))
    output = decoder.apply(params, batch)

    # Output should be in [0, 1] due to sigmoid
    assert output.min() >= 0.0
    assert output.max() <= 1.0


def test_decoder_different_latent_dims():
    for latent_dim in [2, 4, 8]:
        decoder = Decoder(latent_dim=latent_dim)
        rng = jax.random.PRNGKey(0)
        dummy_input = jnp.zeros((1, latent_dim))
        params = decoder.init(rng, dummy_input)

        batch = jnp.zeros((2, latent_dim))
        output = decoder.apply(params, batch)

        assert output.shape == (2, 28, 28, 1)
