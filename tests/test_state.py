import threading
import time
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
