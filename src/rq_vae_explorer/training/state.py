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
    _sinkhorn_epsilon: float = 0.05

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

    # Assignment tracking for dead codebook detection
    _assignment_counts: np.ndarray | None = None
    _z_q1: np.ndarray | None = None
    _z_q: np.ndarray | None = None

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
            self._assignment_counts = None
            self._z_q1 = None
            self._z_q = None

    # --- Metrics updates ---

    def update(
        self,
        step: int | None = None,
        codebook: np.ndarray | None = None,
        encoder_outputs: np.ndarray | None = None,
        encoder_labels: np.ndarray | None = None,
        reconstructions: np.ndarray | None = None,
        sample_inputs: np.ndarray | None = None,
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
