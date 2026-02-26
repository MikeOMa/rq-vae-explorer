"""MLflow experiment tracking for RQ-VAE training."""

import threading
from typing import Any


class MLflowTracker:
    """Optional MLflow tracking integration.

    Wraps MLflow operations with enable/disable support and thread-safe
    access to run metadata for display in the UI.

    Usage::

        tracker = MLflowTracker(experiment_name="rq-vae-explorer", enabled=True)
        tracker.start_run({"learning_rate": 1e-3, "batch_size": 64})
        tracker.log_metrics({"loss_total": 0.5}, step=1)
        tracker.end_run()
    """

    def __init__(
        self,
        experiment_name: str = "rq-vae-explorer",
        run_name: str | None = None,
        log_interval: int = 10,
        enabled: bool = False,
    ):
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.log_interval = log_interval
        self.enabled = enabled

        self._run_id: str | None = None
        self._tracking_uri: str | None = None
        self._lock = threading.Lock()

    @property
    def run_id(self) -> str | None:
        with self._lock:
            return self._run_id

    @property
    def tracking_uri(self) -> str | None:
        with self._lock:
            return self._tracking_uri

    def start_run(self, params: dict[str, Any] | None = None) -> None:
        """Start a new MLflow run and log initial parameters."""
        if not self.enabled:
            return

        try:
            import mlflow

            mlflow.set_experiment(self.experiment_name)
            run = mlflow.start_run(run_name=self.run_name)

            if params:
                mlflow.log_params(params)

            with self._lock:
                self._run_id = run.info.run_id
                self._tracking_uri = mlflow.get_tracking_uri()
        except Exception as e:
            print(f"MLflow start_run failed: {e}")

    def end_run(self) -> None:
        """End the current MLflow run."""
        if not self.enabled:
            return

        try:
            import mlflow

            mlflow.end_run()
        except Exception as e:
            print(f"MLflow end_run failed: {e}")
        finally:
            with self._lock:
                self._run_id = None

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Log metrics at the given step.

        Respects ``log_interval`` — only logs when ``step % log_interval == 0``.
        """
        if not self.enabled:
            return

        if step % self.log_interval != 0:
            return

        try:
            import mlflow

            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            print(f"MLflow log_metrics failed: {e}")
