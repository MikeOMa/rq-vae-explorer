"""Tests for MLflowTracker."""

import pytest
from unittest.mock import patch, MagicMock
from rq_vae_explorer.training.mlflow_tracker import MLflowTracker


def test_tracker_disabled_by_default():
    tracker = MLflowTracker()
    assert not tracker.enabled
    assert tracker.run_id is None


def test_tracker_does_nothing_when_disabled():
    tracker = MLflowTracker(enabled=False)
    # Should not raise even without mlflow installed/configured
    tracker.start_run({"lr": 1e-3})
    tracker.log_metrics({"loss": 0.5}, step=10)
    tracker.end_run()
    assert tracker.run_id is None


def test_log_metrics_respects_interval():
    tracker = MLflowTracker(enabled=True, log_interval=10)

    logged_calls = []

    with patch("mlflow.set_experiment"), \
         patch("mlflow.start_run") as mock_start, \
         patch("mlflow.get_tracking_uri", return_value="mlruns"), \
         patch("mlflow.log_metrics", side_effect=lambda m, step: logged_calls.append(step)), \
         patch("mlflow.end_run"):

        mock_run = MagicMock()
        mock_run.info.run_id = "abc123"
        mock_start.return_value.__enter__ = lambda s: mock_run
        mock_start.return_value.__exit__ = MagicMock(return_value=False)
        mock_start.return_value = mock_run

        tracker.start_run()

        # Only steps divisible by log_interval should be logged
        for step in range(1, 31):
            tracker.log_metrics({"loss": 0.5}, step=step)

    assert logged_calls == [10, 20, 30]


def test_log_metrics_skipped_when_disabled():
    tracker = MLflowTracker(enabled=False, log_interval=1)

    with patch("mlflow.log_metrics") as mock_log:
        tracker.log_metrics({"loss": 0.5}, step=1)
        mock_log.assert_not_called()


def test_start_run_logs_params():
    tracker = MLflowTracker(enabled=True, experiment_name="test-exp")
    params = {"latent_dim": 2, "batch_size": 64}

    with patch("mlflow.set_experiment") as mock_set_exp, \
         patch("mlflow.start_run") as mock_start, \
         patch("mlflow.get_tracking_uri", return_value="mlruns"), \
         patch("mlflow.log_params") as mock_log_params, \
         patch("mlflow.end_run"):

        mock_run = MagicMock()
        mock_run.info.run_id = "run-xyz"
        mock_start.return_value = mock_run

        tracker.start_run(params)

        mock_set_exp.assert_called_once_with("test-exp")
        mock_log_params.assert_called_once_with(params)
        assert tracker.run_id == "run-xyz"
        assert tracker.tracking_uri == "mlruns"


def test_end_run_clears_run_id():
    tracker = MLflowTracker(enabled=True)

    with patch("mlflow.set_experiment"), \
         patch("mlflow.start_run") as mock_start, \
         patch("mlflow.get_tracking_uri", return_value="mlruns"), \
         patch("mlflow.log_params"), \
         patch("mlflow.end_run"):

        mock_run = MagicMock()
        mock_run.info.run_id = "run-abc"
        mock_start.return_value = mock_run

        tracker.start_run()
        assert tracker.run_id == "run-abc"

        tracker.end_run()
        assert tracker.run_id is None


def test_tracker_survives_mlflow_errors():
    tracker = MLflowTracker(enabled=True)

    with patch("mlflow.set_experiment", side_effect=Exception("connection error")):
        # Should not raise
        tracker.start_run({"lr": 1e-3})

    assert tracker.run_id is None


def test_tracker_log_metrics_survives_errors():
    tracker = MLflowTracker(enabled=True, log_interval=1)

    with patch("mlflow.set_experiment"), \
         patch("mlflow.start_run") as mock_start, \
         patch("mlflow.get_tracking_uri", return_value="mlruns"), \
         patch("mlflow.log_params"), \
         patch("mlflow.log_metrics", side_effect=Exception("write error")):

        mock_run = MagicMock()
        mock_run.info.run_id = "run-err"
        mock_start.return_value = mock_run

        tracker.start_run()
        # Should not raise
        tracker.log_metrics({"loss": 0.5}, step=1)
