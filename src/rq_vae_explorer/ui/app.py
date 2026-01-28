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
    plot_codebook_trajectory,
    plot_gradient_magnitudes,
    plot_decoded_codebooks,
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
        num_codes=4,
        num_levels=2,
    )

    # Cache for debug plots (to avoid regenerating every tick)
    debug_plot_cache = {
        "last_step": -1,
        "trajectory_l1": None,
        "trajectory_l2": None,
        "gradient": None,
        "decoded": None,
    }

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

        # Debug visualizations (collapsible)
        with gr.Accordion("Debug: Codebook Dynamics", open=False):
            with gr.Row():
                debug_refresh_interval = gr.Slider(
                    minimum=50,
                    maximum=1000,
                    value=300,
                    step=50,
                    label="Debug Plot Refresh Interval (steps)",
                    info="How often to update debug plots",
                )
            with gr.Row():
                trajectory_plot_l1 = gr.Plot(label="L1 Codebook Trajectory")
                trajectory_plot_l2 = gr.Plot(label="L2 Codebook Trajectory")
            with gr.Row():
                gradient_plot = gr.Plot(label="Gradient Magnitudes")
                decoded_codebooks_plot = gr.Plot(label="Decoded Codebook Combinations")

        # --- Event handlers ---

        def on_start():
            trainer.start()
            return format_step_text(state.step, True)

        def on_stop():
            trainer.stop()
            return format_step_text(state.step, False)

        def on_reset():
            trainer.reset()
            # Reset debug plot cache
            debug_plot_cache["last_step"] = -1
            debug_plot_cache["trajectory_l1"] = None
            debug_plot_cache["trajectory_l2"] = None
            debug_plot_cache["gradient"] = None
            debug_plot_cache["decoded"] = None
            return (
                format_step_text(0, False),
                None,  # codebook_plot
                None,  # recon_plot
                None,  # loss_plot
                "**Codebook Health**\nNo data yet",
                None,  # trajectory_plot_l1
                None,  # trajectory_plot_l2
                None,  # gradient_plot
                None,  # decoded_codebooks_plot
            )

        def on_lambda_commit_change(value):
            state.set_lambda_commit(value)

        def on_lambda_codebook_change(value):
            state.set_lambda_codebook(value)

        def on_lambda_wasserstein_change(value):
            state.set_lambda_wasserstein(value)

        def on_sinkhorn_epsilon_change(value):
            state.set_sinkhorn_epsilon(value)

        def refresh_ui(mode: str, refresh_interval: int):
            """Refresh all UI components with current state."""
            import matplotlib.pyplot as plt

            plt.close("all")  # Prevent memory leak from accumulating figures

            current_step = state.step
            codebook = state.get_codebook()
            encoder_outputs, labels = state.get_encoder_outputs()
            z_q1, z_q = state.get_quantized_outputs()
            recons, inputs = state.get_reconstructions()
            history = state.get_loss_history()
            assignment_counts = state.get_assignment_counts()

            # Codebook plot (always update - it's the main visualization)
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
            step_str = format_step_text(current_step, state.is_training)

            # Debug plots - only update every N steps
            steps_since_last = current_step - debug_plot_cache["last_step"]
            if (
                steps_since_last >= refresh_interval
                or debug_plot_cache["last_step"] < 0
            ):
                codebook_history, history_steps = state.get_codebook_history()
                grad_ema = state.get_codebook_grad_ema()
                decoded_codebooks = state.get_decoded_codebooks()

                debug_plot_cache["trajectory_l1"] = plot_codebook_trajectory(
                    codebook_history, history_steps, assignment_counts, level=0
                )
                debug_plot_cache["trajectory_l2"] = plot_codebook_trajectory(
                    codebook_history, history_steps, assignment_counts, level=1
                )
                debug_plot_cache["gradient"] = plot_gradient_magnitudes(
                    grad_ema, assignment_counts
                )
                debug_plot_cache["decoded"] = plot_decoded_codebooks(
                    decoded_codebooks, num_codes=4
                )
                debug_plot_cache["last_step"] = current_step

            return (
                step_str,
                codebook_fig,
                recon_fig,
                loss_fig,
                health_str,
                debug_plot_cache["trajectory_l1"],
                debug_plot_cache["trajectory_l2"],
                debug_plot_cache["gradient"],
                debug_plot_cache["decoded"],
            )

        # Wire up event handlers
        start_btn.click(on_start, outputs=[step_text])
        stop_btn.click(on_stop, outputs=[step_text])
        reset_btn.click(
            on_reset,
            outputs=[
                step_text,
                codebook_plot,
                recon_plot,
                loss_plot,
                health_text,
                trajectory_plot_l1,
                trajectory_plot_l2,
                gradient_plot,
                decoded_codebooks_plot,
            ],
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
            inputs=[mode_dropdown, debug_refresh_interval],
            outputs=[
                step_text,
                codebook_plot,
                recon_plot,
                loss_plot,
                health_text,
                trajectory_plot_l1,
                trajectory_plot_l2,
                gradient_plot,
                decoded_codebooks_plot,
            ],
        )

    return app


def main():
    """Entry point for the application."""
    app = create_app()
    app.launch()


if __name__ == "__main__":
    main()
