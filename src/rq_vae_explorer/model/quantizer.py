"""Residual Vector Quantization module."""

from flax import linen as nn
import jax
import jax.numpy as jnp


class ResidualQuantizer(nn.Module):
    """Residual Vector Quantizer for RQ-VAE.

    Performs multi-level residual quantization:
    1. Find nearest codebook vector to input
    2. Compute residual = input - quantized
    3. Repeat for next level on residual
    4. Sum all quantized vectors for final output

    Uses straight-through estimator for gradients.

    Attributes:
        num_codes: Number of codebook vectors per level (K)
        num_levels: Number of quantization levels (D)
        latent_dim: Dimension of latent vectors
    """

    num_codes: int = 16
    num_levels: int = 2
    latent_dim: int = 2

    def setup(self):
        # Initialize codebook: (num_levels, num_codes, latent_dim)
        self.codebook = self.param(
            "codebook",
            nn.initializers.uniform(scale=1.0),
            (self.num_levels, self.num_codes, self.latent_dim),
        )

    def __call__(self, z: jnp.ndarray) -> tuple[jnp.ndarray, dict]:
        """Quantize input vectors using residual quantization.

        Args:
            z: Input latent vectors of shape (batch, latent_dim)

        Returns:
            Tuple of:
                - Quantized vectors of shape (batch, latent_dim)
                - Auxiliary dict with:
                    - indices: Selected codebook indices (batch, num_levels)
                    - codebook: Current codebook values
                    - z_e: Original encoder output (for loss computation)
        """
        quantized = jnp.zeros_like(z)
        residual = z
        all_indices = []

        for level in range(self.num_levels):
            level_codebook = self.codebook[level]  # (num_codes, latent_dim)

            # Find nearest codebook vector
            # distances: (batch, num_codes)
            distances = jnp.sum(
                (residual[:, None, :] - level_codebook[None, :, :]) ** 2,
                axis=-1,
            )
            indices = jnp.argmin(distances, axis=-1)  # (batch,)
            all_indices.append(indices)

            # Get quantized vectors for this level
            level_quantized = level_codebook[indices]  # (batch, latent_dim)

            # Accumulate quantized output
            quantized = quantized + level_quantized

            # Compute residual for next level
            residual = residual - level_quantized

        # Straight-through estimator: gradients flow through as if identity
        quantized_st = z + jax.lax.stop_gradient(quantized - z)

        indices = jnp.stack(all_indices, axis=-1)  # (batch, num_levels)

        aux = {
            "indices": indices,
            "codebook": self.codebook,
            "z_e": z,
            "z_q": quantized,
        }

        return quantized_st, aux
