"""Entry point for RQ-VAE Explorer - starts the FastAPI training server."""

import uvicorn

from rq_vae_explorer.training.state import TrainingState
from rq_vae_explorer.training.trainer import Trainer
from rq_vae_explorer.ui import api


def main():
    """Start the FastAPI server. Connect with the Rust UI (rq-vae-explorer-ui)."""
    state = TrainingState()
    trainer = Trainer(
        state=state,
        latent_dim=2,
        num_codes=4,
        num_levels=2,
    )
    api.init_app(state, trainer)
    print("RQ-VAE Explorer API running on http://127.0.0.1:7860")
    print("Start the Rust UI with: cd ui-rust && cargo run --release")
    uvicorn.run(api.app, host="127.0.0.1", port=7860)


if __name__ == "__main__":
    main()
