"""Training loop for RQ-VAE."""

import threading
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from rq_vae_explorer.model import RQVAE
from rq_vae_explorer.data import load_mnist, create_data_iterator
from rq_vae_explorer.training.losses import compute_losses
from rq_vae_explorer.training.state import TrainingState


class Trainer:
    """Manages RQ-VAE training with live parameter updates.

    Runs training in a background thread while allowing the UI
    to read metrics and update hyperparameters.
    """

    def __init__(
        self,
        state: TrainingState,
        latent_dim: int = 2,
        num_codes: int = 16,
        num_levels: int = 2,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
    ):
        self.state = state
        self.batch_size = batch_size
        self.num_codes = num_codes
        self.num_levels = num_levels

        # Initialize model
        self.model = RQVAE(
            latent_dim=latent_dim,
            num_codes=num_codes,
            num_levels=num_levels,
        )

        # Initialize parameters
        rng = jax.random.PRNGKey(0)
        dummy_input = jnp.zeros((1, 28, 28, 1))
        self.params = self.model.init(rng, dummy_input)

        # Initialize optimizer
        self.optimizer = optax.adam(learning_rate)
        self.opt_state = self.optimizer.init(self.params)

        # Load data
        (train_images, train_labels), _ = load_mnist()
        self.train_data = (train_images, train_labels)
        self.data_iterator = create_data_iterator(
            self.train_data, batch_size=batch_size
        )

        # Keep a fixed sample for reconstructions
        self._sample_batch = next(self.data_iterator)

        # JIT compile training step
        self._train_step_jit = jax.jit(self._train_step_inner)

        # Background thread
        self._thread: threading.Thread | None = None

        # Assignment tracking (rolling window)
        self._assignment_window: list[np.ndarray] = []
        self._window_size = 100

    def _train_step_inner(
        self,
        params: Any,
        opt_state: Any,
        batch: jnp.ndarray,
        lambda_commit: float,
        lambda_codebook: float,
    ) -> tuple[
        Any, Any, dict, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
    ]:
        """Inner training step (JIT compiled)."""

        def loss_fn(params):
            x_recon, aux = self.model.apply(params, batch)
            losses = compute_losses(
                x=batch,
                x_recon=x_recon,
                z_e=aux["z_e"],
                z_q=aux["z_q"],
                lambda_commit=lambda_commit,
                lambda_codebook=lambda_codebook,
            )
            return losses["total"], (losses, aux, x_recon)

        (loss, (losses, aux, x_recon)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(params)

        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return (
            params,
            opt_state,
            losses,
            aux["codebook"],
            aux["indices"],
            aux["z_e"],
            aux["z_q1"],
            aux["z_q"],
        )

    def train_step(self) -> None:
        """Execute a single training step."""
        batch_images, batch_labels = next(self.data_iterator)
        lambda_commit, lambda_codebook = self.state.get_lambdas()

        (
            self.params,
            self.opt_state,
            losses,
            codebook,
            indices,
            encoder_outputs,
            z_q1,
            z_q,
        ) = self._train_step_jit(
            self.params,
            self.opt_state,
            batch_images,
            lambda_commit,
            lambda_codebook,
        )

        # Convert to numpy for state
        codebook_np = np.array(codebook)
        encoder_outputs_np = np.array(encoder_outputs)
        labels_np = np.array(batch_labels)
        indices_np = np.array(indices)
        z_q1_np = np.array(z_q1)
        z_q_np = np.array(z_q)

        # Update assignment tracking
        self._update_assignment_tracking(indices_np)

        # Update state
        step = self.state.step + 1
        self.state.update(
            step=step,
            codebook=codebook_np,
            encoder_outputs=encoder_outputs_np,
            encoder_labels=labels_np,
            assignment_counts=self._get_assignment_counts(),
            z_q1=z_q1_np,
            z_q=z_q_np,
        )
        self.state.add_losses({k: float(v) for k, v in losses.items()})

        # Update reconstructions periodically
        if step % 50 == 0:
            self._update_reconstructions()

    def _update_assignment_tracking(self, indices: np.ndarray) -> None:
        """Track codebook assignments for dead vector detection."""
        # indices shape: (batch, num_levels)
        self._assignment_window.append(indices)
        if len(self._assignment_window) > self._window_size:
            self._assignment_window.pop(0)

    def _get_assignment_counts(self) -> np.ndarray:
        """Get assignment counts per codebook vector."""
        if not self._assignment_window:
            return np.zeros((self.num_levels, self.num_codes))

        counts = np.zeros((self.num_levels, self.num_codes))
        for indices in self._assignment_window:
            for level in range(self.num_levels):
                level_indices = indices[:, level]
                for idx in level_indices:
                    counts[level, idx] += 1

        return counts

    def _update_reconstructions(self) -> None:
        """Update sample reconstructions for UI display."""
        sample_images, _ = self._sample_batch
        x_recon, _ = self.model.apply(self.params, sample_images[:8])

        self.state.update(
            reconstructions=np.array(x_recon),
            sample_inputs=np.array(sample_images[:8]),
        )

    def _training_loop(self) -> None:
        """Background training loop."""
        self.state.start_training()

        try:
            while not self.state.should_stop:
                self.train_step()
        finally:
            self.state.training_stopped()

    def start(self) -> None:
        """Start training in background thread."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._thread = threading.Thread(target=self._training_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop training."""
        self.state.stop_training()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def reset(self) -> None:
        """Reset model and state for new training run."""
        self.stop()

        # Reinitialize parameters
        rng = jax.random.PRNGKey(0)
        dummy_input = jnp.zeros((1, 28, 28, 1))
        self.params = self.model.init(rng, dummy_input)
        self.opt_state = self.optimizer.init(self.params)

        # Reset data iterator
        self.data_iterator = create_data_iterator(
            self.train_data, batch_size=self.batch_size
        )

        # Reset assignment tracking
        self._assignment_window = []

        # Reset state
        self.state.reset()
