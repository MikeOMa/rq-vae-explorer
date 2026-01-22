"""CNN Encoder for MNIST images."""

from flax import linen as nn
import jax.numpy as jnp


class Encoder(nn.Module):
    """Encodes 28x28 MNIST images to a latent vector.

    Architecture:
        Conv 32 filters, 3x3, stride 2 -> (14, 14, 32)
        Conv 64 filters, 3x3, stride 2 -> (7, 7, 64)
        Flatten -> Dense 128 -> Dense latent_dim

    Attributes:
        latent_dim: Dimension of the output latent vector (default 2)
    """

    latent_dim: int = 2

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass.

        Args:
            x: Input images of shape (batch, 28, 28, 1)

        Returns:
            Latent vectors of shape (batch, latent_dim)
        """
        # Conv layers
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.relu(x)

        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.relu(x)

        # Flatten and dense layers
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)

        x = nn.Dense(features=self.latent_dim)(x)

        return x
