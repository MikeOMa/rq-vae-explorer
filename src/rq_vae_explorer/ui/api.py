"""FastAPI HTTP server exposing training state and controls for the Rust UI."""

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastapi import FastAPI
from fastapi.responses import Response

from rq_vae_explorer.training.state import TrainingState
from rq_vae_explorer.training.trainer import Trainer
from rq_vae_explorer.ui.plots import (
    get_codebook_health,
    plot_codebook_2d,
    plot_codebook_trajectory,
    plot_decoded_codebooks,
    plot_gradient_magnitudes,
    plot_loss_curves,
    plot_reconstructions,
)

app = FastAPI(title="RQ-VAE Explorer API")

# Initialized at startup via init_app()
_state: TrainingState | None = None
_trainer: Trainer | None = None


def init_app(state: TrainingState, trainer: Trainer) -> None:
    global _state, _trainer
    _state = state
    _trainer = trainer


def _fig_to_png(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# --- Health check ---


@app.get("/health")
def health():
    return {"ok": True}


# --- Training controls ---


@app.post("/training/start")
def start():
    _trainer.start()
    return {"step": _state.step, "is_training": _state.is_training}


@app.post("/training/stop")
def stop():
    _trainer.stop()
    return {"step": _state.step, "is_training": _state.is_training}


@app.post("/training/reset")
def reset():
    _trainer.reset()
    return {"step": _state.step, "is_training": _state.is_training}


# --- State endpoint ---


@app.get("/training/state")
def get_state():
    assignment_counts = _state.get_assignment_counts()
    health = get_codebook_health(assignment_counts)
    loss_history = _state.get_loss_history()

    latest_losses: dict[str, float] = {}
    for key, values in loss_history.items():
        latest_losses[key] = float(values[-1]) if values else 0.0

    return {
        "step": _state.step,
        "is_training": _state.is_training,
        "health": health,
        "latest_losses": latest_losses,
        "loss_history_len": len(loss_history.get("total", [])),
    }


# --- Parameter updates ---


@app.post("/params")
def update_params(
    lambda_commit: float | None = None,
    lambda_codebook: float | None = None,
    lambda_wasserstein: float | None = None,
    sinkhorn_epsilon: float | None = None,
):
    if lambda_commit is not None:
        _state.set_lambda_commit(lambda_commit)
    if lambda_codebook is not None:
        _state.set_lambda_codebook(lambda_codebook)
    if lambda_wasserstein is not None:
        _state.set_lambda_wasserstein(lambda_wasserstein)
    if sinkhorn_epsilon is not None:
        _state.set_sinkhorn_epsilon(sinkhorn_epsilon)
    return {"ok": True}


# --- Plot endpoints (return PNG images) ---


@app.get("/plots/codebook")
def plot_codebook(mode: str = "Residuals"):
    plt.close("all")
    codebook = _state.get_codebook()
    if codebook is None:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "No data yet", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("2D Codebook Visualization")
        return Response(content=_fig_to_png(fig), media_type="image/png")

    encoder_outputs, labels = _state.get_encoder_outputs()
    z_q1, z_q = _state.get_quantized_outputs()
    assignment_counts = _state.get_assignment_counts()

    fig = plot_codebook_2d(
        codebook,
        encoder_outputs,
        labels,
        assignment_counts,
        z_q1=z_q1,
        z_q=z_q,
        mode=mode,
    )
    return Response(content=_fig_to_png(fig), media_type="image/png")


@app.get("/plots/reconstructions")
def plot_recons():
    plt.close("all")
    recons, inputs = _state.get_reconstructions()
    fig = plot_reconstructions(inputs, recons)
    return Response(content=_fig_to_png(fig), media_type="image/png")


@app.get("/plots/loss")
def plot_loss():
    plt.close("all")
    history = _state.get_loss_history()
    fig = plot_loss_curves(history)
    return Response(content=_fig_to_png(fig), media_type="image/png")


@app.get("/plots/trajectory")
def plot_trajectory(level: int = 0):
    plt.close("all")
    codebook_history, history_steps = _state.get_codebook_history()
    assignment_counts = _state.get_assignment_counts()
    fig = plot_codebook_trajectory(
        codebook_history, history_steps, assignment_counts, level=level
    )
    return Response(content=_fig_to_png(fig), media_type="image/png")


@app.get("/plots/gradients")
def plot_gradients():
    plt.close("all")
    grad_ema = _state.get_codebook_grad_ema()
    assignment_counts = _state.get_assignment_counts()
    fig = plot_gradient_magnitudes(grad_ema, assignment_counts)
    return Response(content=_fig_to_png(fig), media_type="image/png")


@app.get("/plots/decoded")
def plot_decoded():
    plt.close("all")
    decoded_codebooks = _state.get_decoded_codebooks()
    fig = plot_decoded_codebooks(decoded_codebooks, num_codes=4)
    return Response(content=_fig_to_png(fig), media_type="image/png")
