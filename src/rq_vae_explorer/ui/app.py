"""Main Gradio application for RQ-VAE Explorer."""

import gradio as gr
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend

from rq_vae_explorer.training.state import TrainingState
from rq_vae_explorer.training.trainer import Trainer
from rq_vae_explorer.ui.plots import (
    plot_codebook_2d,
    plot_loss_curves,
    plot_reconstructions,
    get_codebook_health,
)
from rq_vae_explorer.ui.controls import (
    create_lambda_sliders,
    create_wasserstein_sliders,
    create_training_controls,
    format_health_text,
    format_step_text,
)


def create_app() -> gr.Blocks:
    """Create the Gradio application.

    Returns:
        Gradio Blocks app
    """
    # Shared state and trainer (created once)
    state = TrainingState()
    trainer = Trainer(
        state=state,
        latent_dim=2,
        num_codes=16,
        num_levels=2,
    )

    with gr.Blocks(title="RQ-VAE Explorer") as app:
        gr.Markdown("# RQ-VAE Explorer")
        gr.Markdown(
            "Interactive training visualization for Residual Quantized VAE on MNIST"
        )

        # Training controls
        with gr.Row():
            start_btn, stop_btn, reset_btn = create_training_controls()
            step_text = gr.Markdown("**Step:** 0 | ⏸ Paused")

        # Main visualization area
        with gr.Row():
            with gr.Column(scale=1):
                codebook_plot = gr.Plot(label="2D Codebook Visualization")
                mode_dropdown = gr.Dropdown(
                    choices=["Residuals", "Cumulative"],
                    value="Residuals",
                    label="Plot 2 Mode",
                )
                health_text = gr.Markdown("**Codebook Health**\nNo data yet")

            with gr.Column(scale=1):
                recon_plot = gr.Plot(label="Sample Reconstructions")
                loss_plot = gr.Plot(label="Loss Curves")

        # Parameter controls
        with gr.Row():
            lambda_commit, lambda_codebook = create_lambda_sliders()
        with gr.Row():
            lambda_wasserstein, sinkhorn_epsilon = create_wasserstein_sliders()

        # --- Event handlers ---

        def on_start():
            trainer.start()
            return format_step_text(state.step, True)

        def on_stop():
            trainer.stop()
            return format_step_text(state.step, False)

        def on_reset():
            trainer.reset()
            return (
                format_step_text(0, False),
                None,  # codebook_plot
                None,  # recon_plot
                None,  # loss_plot
                "**Codebook Health**\nNo data yet",
            )

        def on_lambda_commit_change(value):
            state.set_lambda_commit(value)

        def on_lambda_codebook_change(value):
            state.set_lambda_codebook(value)

        def on_lambda_wasserstein_change(value):
            state.set_lambda_wasserstein(value)

        def on_sinkhorn_epsilon_change(value):
            state.set_sinkhorn_epsilon(value)

        def refresh_ui(mode: str):
            """Refresh all UI components with current state."""
            codebook = state.get_codebook()
            encoder_outputs, labels = state.get_encoder_outputs()
            z_q1, z_q = state.get_quantized_outputs()
            recons, inputs = state.get_reconstructions()
            history = state.get_loss_history()
            assignment_counts = state.get_assignment_counts()

            # Codebook plot
            if codebook is not None:
                codebook_fig = plot_codebook_2d(
                    codebook,
                    encoder_outputs,
                    labels,
                    assignment_counts,
                    z_q1=z_q1,
                    z_q=z_q,
                    mode=mode,
                )
            else:
                codebook_fig = None

            # Reconstruction plot
            recon_fig = plot_reconstructions(inputs, recons)

            # Loss plot
            loss_fig = plot_loss_curves(history)

            # Health text
            health = get_codebook_health(assignment_counts)
            health_str = format_health_text(health)

            # Step text
            step_str = format_step_text(state.step, state.is_training)

            return step_str, codebook_fig, recon_fig, loss_fig, health_str

        # Wire up event handlers
        start_btn.click(on_start, outputs=[step_text])
        stop_btn.click(on_stop, outputs=[step_text])
        reset_btn.click(
            on_reset,
            outputs=[step_text, codebook_plot, recon_plot, loss_plot, health_text],
        )

        lambda_commit.change(on_lambda_commit_change, inputs=[lambda_commit])
        lambda_codebook.change(on_lambda_codebook_change, inputs=[lambda_codebook])
        lambda_wasserstein.change(
            on_lambda_wasserstein_change, inputs=[lambda_wasserstein]
        )
        sinkhorn_epsilon.change(on_sinkhorn_epsilon_change, inputs=[sinkhorn_epsilon])

        # Auto-refresh while app is open
        timer = gr.Timer(0.5)
        timer.tick(
            refresh_ui,
            inputs=[mode_dropdown],
            outputs=[step_text, codebook_plot, recon_plot, loss_plot, health_text],
        )

    return app


def main():
    """Entry point for the application."""
    app = create_app()
    app.launch()


if __name__ == "__main__":
    main()
