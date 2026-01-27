import threading
import time
import numpy as np
from rq_vae_explorer.training.state import TrainingState


def test_training_state_initial_values():
    state = TrainingState()

    assert state.lambda_commit == 0.25
    assert state.lambda_codebook == 1.0
    assert state.step == 0
    assert state.is_training is False


def test_training_state_update_lambdas():
    state = TrainingState()

    state.set_lambda_commit(0.5)
    state.set_lambda_codebook(2.0)

    assert state.lambda_commit == 0.5
    assert state.lambda_codebook == 2.0


def test_training_state_thread_safety():
    state = TrainingState()
    results = []

    def reader():
        for _ in range(100):
            results.append(state.lambda_commit)
            time.sleep(0.001)

    def writer():
        for i in range(100):
            state.set_lambda_commit(float(i))
            time.sleep(0.001)

    t1 = threading.Thread(target=reader)
    t2 = threading.Thread(target=writer)

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Should complete without errors
    assert len(results) == 100


def test_training_state_loss_history():
    state = TrainingState()

    state.add_losses({"total": 1.0, "recon": 0.5, "commit": 0.3, "codebook": 0.2})
    state.add_losses({"total": 0.8, "recon": 0.4, "commit": 0.2, "codebook": 0.2})

    history = state.get_loss_history()
    assert len(history["total"]) == 2
    assert history["total"][0] == 1.0
    assert history["total"][1] == 0.8


def test_state_stores_quantized_outputs():
    """Test that state can store and retrieve z_q1 and z_q."""
    state = TrainingState()

    z_q1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    z_q = np.array([[1.5, 2.5], [3.5, 4.5]])

    state.update(z_q1=z_q1, z_q=z_q)

    retrieved_z_q1, retrieved_z_q = state.get_quantized_outputs()

    assert retrieved_z_q1 is not None
    assert retrieved_z_q is not None
    np.testing.assert_array_equal(retrieved_z_q1, z_q1)
    np.testing.assert_array_equal(retrieved_z_q, z_q)


def test_training_state_wasserstein_params():
    """State tracks lambda_wasserstein and sinkhorn_epsilon."""
    state = TrainingState()

    # Default values
    assert state.lambda_wasserstein == 0.0
    assert state.sinkhorn_epsilon == 0.05

    # Setters work
    state.set_lambda_wasserstein(0.5)
    state.set_sinkhorn_epsilon(0.1)

    assert state.lambda_wasserstein == 0.5
    assert state.sinkhorn_epsilon == 0.1


def test_training_state_get_all_lambdas():
    """get_all_lambdas returns all four parameters."""
    state = TrainingState()
    state.set_lambda_commit(0.3)
    state.set_lambda_codebook(0.8)
    state.set_lambda_wasserstein(0.5)
    state.set_sinkhorn_epsilon(0.1)

    commit, codebook, wasserstein, epsilon = state.get_all_lambdas()

    assert commit == 0.3
    assert codebook == 0.8
    assert wasserstein == 0.5
    assert epsilon == 0.1
