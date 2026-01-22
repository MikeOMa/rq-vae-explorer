"""CNN Decoder for MNIST images."""

from flax import linen as nn
import jax.numpy as jnp


class Decoder(nn.Module):
    """Decodes latent vectors to 28x28 MNIST images.

    Architecture:
        Dense 128 -> Dense 7*7*64 -> Reshape (7, 7, 64)
        ConvTranspose 32 filters, 3x3, stride 2 -> (14, 14, 32)
        ConvTranspose 1 filter, 3x3, stride 2, sigmoid -> (28, 28, 1)

    Attributes:
        latent_dim: Dimension of the input latent vector (default 2)
    """
    latent_dim: int = 2

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        """Forward pass.

        Args:
            z: Latent vectors of shape (batch, latent_dim)

        Returns:
            Reconstructed images of shape (batch, 28, 28, 1)
        """
        batch_size = z.shape[0]

        # Dense layers
        x = nn.Dense(features=128)(z)
        x = nn.relu(x)

        x = nn.Dense(features=7 * 7 * 64)(x)
        x = nn.relu(x)

        # Reshape for conv transpose
        x = x.reshape((batch_size, 7, 7, 64))

        # Conv transpose layers
        x = nn.ConvTranspose(features=32, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.relu(x)

        x = nn.ConvTranspose(features=1, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.sigmoid(x)

        return x
