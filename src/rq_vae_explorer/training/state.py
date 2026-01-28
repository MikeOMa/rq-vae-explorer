"""Thread-safe shared state for training loop and UI communication."""

import threading
from dataclasses import dataclass, field
import numpy as np


@dataclass
class TrainingState:
    """Thread-safe state shared between trainer and UI.

    The trainer reads lambda values and writes metrics.
    The UI reads metrics and writes lambda values.
    """

    # Loss weights (UI writes, trainer reads)
    _lambda_commit: float = 0.25
    _lambda_codebook: float = 1.0
    _lambda_wasserstein: float = 0.0
    _sinkhorn_epsilon: float = 0.2

    # Training status
    _step: int = 0
    _is_training: bool = False
    _should_stop: bool = False

    # Metrics (trainer writes, UI reads)
    _loss_history: dict[str, list[float]] = field(
        default_factory=lambda: {
            "total": [],
            "recon": [],
            "commit": [],
            "codebook": [],
            "wasserstein": [],
        }
    )
    _codebook: np.ndarray | None = None
    _encoder_outputs: np.ndarray | None = None
    _encoder_labels: np.ndarray | None = None
    _reconstructions: np.ndarray | None = None
    _sample_inputs: np.ndarray | None = None
    _decoded_codebooks: np.ndarray | None = None

    # Assignment tracking for dead codebook detection
    _assignment_counts: np.ndarray | None = None
    _z_q1: np.ndarray | None = None
    _z_q: np.ndarray | None = None

    # Codebook trajectory history for debugging
    _codebook_history: list[np.ndarray] = field(default_factory=list)
    _codebook_history_steps: list[int] = field(default_factory=list)
    _codebook_history_max: int = 200
    _codebook_history_interval: int = 50

    # Gradient EMA for debugging (smoothed gradient magnitudes per codebook vector)
    _codebook_grad_ema: np.ndarray | None = None
    _grad_ema_alpha: float = 0.05

    # Thread lock
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # --- Lambda getters/setters (thread-safe) ---

    @property
    def lambda_commit(self) -> float:
        with self._lock:
            return self._lambda_commit

    @property
    def lambda_codebook(self) -> float:
        with self._lock:
            return self._lambda_codebook

    def set_lambda_commit(self, value: float) -> None:
        with self._lock:
            self._lambda_commit = value

    def set_lambda_codebook(self, value: float) -> None:
        with self._lock:
            self._lambda_codebook = value

    def get_lambdas(self) -> tuple[float, float]:
        """Get both lambda values atomically."""
        with self._lock:
            return self._lambda_commit, self._lambda_codebook

    @property
    def lambda_wasserstein(self) -> float:
        with self._lock:
            return self._lambda_wasserstein

    @property
    def sinkhorn_epsilon(self) -> float:
        with self._lock:
            return self._sinkhorn_epsilon

    def set_lambda_wasserstein(self, value: float) -> None:
        with self._lock:
            self._lambda_wasserstein = value

    def set_sinkhorn_epsilon(self, value: float) -> None:
        with self._lock:
            self._sinkhorn_epsilon = value

    def get_all_lambdas(self) -> tuple[float, float, float, float]:
        """Get all lambda values atomically."""
        with self._lock:
            return (
                self._lambda_commit,
                self._lambda_codebook,
                self._lambda_wasserstein,
                self._sinkhorn_epsilon,
            )

    # --- Training status ---

    @property
    def step(self) -> int:
        with self._lock:
            return self._step

    @property
    def is_training(self) -> bool:
        with self._lock:
            return self._is_training

    @property
    def should_stop(self) -> bool:
        with self._lock:
            return self._should_stop

    def start_training(self) -> None:
        with self._lock:
            self._is_training = True
            self._should_stop = False

    def stop_training(self) -> None:
        with self._lock:
            self._should_stop = True

    def training_stopped(self) -> None:
        with self._lock:
            self._is_training = False

    def reset(self) -> None:
        """Reset all state for a new training run."""
        with self._lock:
            self._step = 0
            self._is_training = False
            self._should_stop = False
            self._loss_history = {
                "total": [],
                "recon": [],
                "commit": [],
                "codebook": [],
                "wasserstein": [],
            }
            self._codebook = None
            self._encoder_outputs = None
            self._encoder_labels = None
            self._reconstructions = None
            self._sample_inputs = None
            self._decoded_codebooks = None
            self._assignment_counts = None
            self._z_q1 = None
            self._z_q = None
            self._codebook_history = []
            self._codebook_history_steps = []
            self._codebook_grad_ema = None

    # --- Metrics updates ---

    def update(
        self,
        step: int | None = None,
        codebook: np.ndarray | None = None,
        encoder_outputs: np.ndarray | None = None,
        encoder_labels: np.ndarray | None = None,
        reconstructions: np.ndarray | None = None,
        sample_inputs: np.ndarray | None = None,
        decoded_codebooks: np.ndarray | None = None,
        assignment_counts: np.ndarray | None = None,
        z_q1: np.ndarray | None = None,
        z_q: np.ndarray | None = None,
    ) -> None:
        """Update state with new values from trainer."""
        with self._lock:
            if step is not None:
                self._step = step
            if codebook is not None:
                self._codebook = codebook
            if encoder_outputs is not None:
                self._encoder_outputs = encoder_outputs
            if encoder_labels is not None:
                self._encoder_labels = encoder_labels
            if reconstructions is not None:
                self._reconstructions = reconstructions
            if sample_inputs is not None:
                self._sample_inputs = sample_inputs
            if decoded_codebooks is not None:
                self._decoded_codebooks = decoded_codebooks
            if assignment_counts is not None:
                self._assignment_counts = assignment_counts
            if z_q1 is not None:
                self._z_q1 = z_q1
            if z_q is not None:
                self._z_q = z_q

    def add_losses(self, losses: dict[str, float]) -> None:
        """Add a new loss record to history."""
        with self._lock:
            for key, value in losses.items():
                if key in self._loss_history:
                    self._loss_history[key].append(float(value))

    # --- Metrics getters ---

    def get_loss_history(self) -> dict[str, list[float]]:
        with self._lock:
            return {k: list(v) for k, v in self._loss_history.items()}

    def get_codebook(self) -> np.ndarray | None:
        with self._lock:
            return self._codebook.copy() if self._codebook is not None else None

    def get_encoder_outputs(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        with self._lock:
            outputs = (
                self._encoder_outputs.copy()
                if self._encoder_outputs is not None
                else None
            )
            labels = (
                self._encoder_labels.copy()
                if self._encoder_labels is not None
                else None
            )
            return outputs, labels

    def get_reconstructions(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        with self._lock:
            recons = (
                self._reconstructions.copy()
                if self._reconstructions is not None
                else None
            )
            inputs = (
                self._sample_inputs.copy() if self._sample_inputs is not None else None
            )
            return recons, inputs

    def get_assignment_counts(self) -> np.ndarray | None:
        with self._lock:
            return (
                self._assignment_counts.copy()
                if self._assignment_counts is not None
                else None
            )

    def get_quantized_outputs(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Get z_q1 and z_q arrays."""
        with self._lock:
            z_q1 = self._z_q1.copy() if self._z_q1 is not None else None
            z_q = self._z_q.copy() if self._z_q is not None else None
            return z_q1, z_q

    def get_decoded_codebooks(self) -> np.ndarray | None:
        """Get decoded codebook combination images."""
        with self._lock:
            return (
                self._decoded_codebooks.copy()
                if self._decoded_codebooks is not None
                else None
            )

    # --- Codebook debug tracking ---

    def update_codebook_history(self, codebook: np.ndarray, step: int) -> None:
        """Record a codebook snapshot for trajectory visualization."""
        with self._lock:
            if step % self._codebook_history_interval == 0:
                self._codebook_history.append(codebook.copy())
                self._codebook_history_steps.append(step)
                if len(self._codebook_history) > self._codebook_history_max:
                    self._codebook_history.pop(0)
                    self._codebook_history_steps.pop(0)

    def update_grad_ema(self, grad_norms: np.ndarray) -> None:
        """Update exponential moving average of gradient magnitudes."""
        with self._lock:
            if self._codebook_grad_ema is None:
                self._codebook_grad_ema = grad_norms.copy()
            else:
                self._codebook_grad_ema = (
                    1 - self._grad_ema_alpha
                ) * self._codebook_grad_ema + self._grad_ema_alpha * grad_norms

    def get_codebook_history(self) -> tuple[list[np.ndarray], list[int]]:
        """Get codebook trajectory history."""
        with self._lock:
            return (
                [cb.copy() for cb in self._codebook_history],
                list(self._codebook_history_steps),
            )

    def get_codebook_grad_ema(self) -> np.ndarray | None:
        """Get smoothed gradient magnitudes."""
        with self._lock:
            return (
                self._codebook_grad_ema.copy()
                if self._codebook_grad_ema is not None
                else None
            )
