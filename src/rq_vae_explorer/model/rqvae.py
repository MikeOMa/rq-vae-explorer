"""Full RQ-VAE model combining encoder, quantizer, and decoder."""

from flax import linen as nn
import jax.numpy as jnp

from .encoder import Encoder
from .decoder import Decoder
from .quantizer import ResidualQuantizer


class RQVAE(nn.Module):
    """Residual Quantized Variational Autoencoder.

    Combines encoder, residual quantizer, and decoder into a full model.

    Attributes:
        latent_dim: Dimension of latent space
        num_codes: Number of codebook vectors per level
        num_levels: Number of residual quantization levels
    """

    latent_dim: int = 2
    num_codes: int = 16
    num_levels: int = 2

    def setup(self):
        self.encoder = Encoder(latent_dim=self.latent_dim)
        self.quantizer = ResidualQuantizer(
            num_codes=self.num_codes,
            num_levels=self.num_levels,
            latent_dim=self.latent_dim,
        )
        self.decoder = Decoder(latent_dim=self.latent_dim)

    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, dict]:
        """Forward pass through the full model.

        Args:
            x: Input images of shape (batch, 28, 28, 1)

        Returns:
            Tuple of:
                - Reconstructed images of shape (batch, 28, 28, 1)
                - Auxiliary dict with quantizer outputs
        """
        # Encode
        z_e = self.encoder(x)

        # Quantize
        z_q, aux = self.quantizer(z_e)

        # Decode
        x_recon = self.decoder(z_q)

        return x_recon, aux

    def encode(self, x: jnp.ndarray) -> jnp.ndarray:
        """Encode images to latent vectors (before quantization)."""
        return self.encoder(x)

    def decode(self, z: jnp.ndarray) -> jnp.ndarray:
        """Decode latent vectors to images."""
        return self.decoder(z)
