import time
from rq_vae_explorer.training.trainer import Trainer
from rq_vae_explorer.training.state import TrainingState


def test_trainer_initialization():
    state = TrainingState()
    trainer = Trainer(state=state, latent_dim=2, num_codes=16, num_levels=2)

    assert trainer.state is state
    assert trainer.model is not None
    assert trainer.params is not None


def test_trainer_single_step():
    state = TrainingState()
    trainer = Trainer(state=state, latent_dim=2, num_codes=16, num_levels=2)

    # Run one training step
    trainer.train_step()

    assert state.step == 1
    history = state.get_loss_history()
    assert len(history["total"]) == 1


def test_trainer_background_training():
    state = TrainingState()
    trainer = Trainer(state=state, latent_dim=2, num_codes=16, num_levels=2)

    # Start training in background
    trainer.start()
    time.sleep(2.0)  # Let it run briefly (JIT compilation takes time)

    assert state.is_training
    assert state.step > 0

    # Stop training
    trainer.stop()
    time.sleep(0.2)  # Wait for thread to finish

    assert not state.is_training


def test_trainer_uses_wasserstein_params():
    """Trainer passes wasserstein params to loss computation."""
    state = TrainingState()
    state.set_lambda_wasserstein(0.5)
    state.set_sinkhorn_epsilon(0.1)

    trainer = Trainer(state=state)
    trainer.train_step()

    # Check wasserstein is in loss history and is non-zero (actually computed)
    history = state.get_loss_history()
    assert "wasserstein" in history
    assert len(history["wasserstein"]) == 1
    assert history["wasserstein"][0] > 0  # Must be computed, not default 0
