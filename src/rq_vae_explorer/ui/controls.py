"""UI control components for Gradio."""

import gradio as gr


def create_lambda_sliders() -> tuple[gr.Slider, gr.Slider]:
    """Create sliders for loss weight parameters.

    Returns:
        Tuple of (lambda_commit_slider, lambda_codebook_slider)
    """
    lambda_commit = gr.Slider(
        minimum=0.0,
        maximum=2.0,
        value=0.25,
        step=0.05,
        label="λ_commit (commitment loss weight)",
        info="Higher = encoder commits harder to codebook vectors",
    )

    lambda_codebook = gr.Slider(
        minimum=0.0,
        maximum=2.0,
        value=1.0,
        step=0.05,
        label="λ_codebook (codebook loss weight)",
        info="Higher = codebook vectors move faster toward encoder outputs",
    )

    return lambda_commit, lambda_codebook


def create_wasserstein_sliders() -> tuple[gr.Slider, gr.Slider]:
    """Create sliders for Wasserstein loss parameters.

    Returns:
        Tuple of (lambda_wasserstein_slider, sinkhorn_epsilon_slider)
    """
    lambda_wasserstein = gr.Slider(
        minimum=0.0,
        maximum=1.0,
        value=0.0,
        step=0.01,
        label="λ_wasserstein (optimal transport weight)",
        info="0 = off (traditional VQ), higher = pull all codes toward data",
    )

    sinkhorn_epsilon = gr.Slider(
        minimum=0.01,
        maximum=0.2,
        value=0.05,
        step=0.01,
        label="Sinkhorn ε (transport softness)",
        info="Lower = sharper transport, higher = softer",
    )

    return lambda_wasserstein, sinkhorn_epsilon


def create_training_controls() -> tuple[gr.Button, gr.Button, gr.Button]:
    """Create training control buttons.

    Returns:
        Tuple of (start_btn, stop_btn, reset_btn)
    """
    with gr.Row():
        start_btn = gr.Button("▶ Start Training", variant="primary")
        stop_btn = gr.Button("⏹ Stop")
        reset_btn = gr.Button("↺ Reset")

    return start_btn, stop_btn, reset_btn


def format_health_text(health: list[dict[str, int]]) -> str:
    """Format codebook health statistics as text.

    Args:
        health: List of dicts with level, active, dead, total keys

    Returns:
        Formatted text string
    """
    if not health:
        return "No data yet"

    lines = ["**Codebook Health**"]
    for h in health:
        status = "✓" if h["dead"] == 0 else "⚠"
        lines.append(f"{status} Level {h['level']}: {h['active']}/{h['total']} active")
        if h["dead"] > 0:
            lines.append(f"   ({h['dead']} dead)")

    return "\n".join(lines)


def format_step_text(step: int, is_training: bool) -> str:
    """Format current step and training status.

    Args:
        step: Current training step
        is_training: Whether training is active

    Returns:
        Formatted text string
    """
    status = "🟢 Training" if is_training else "⏸ Paused"
    return f"**Step:** {step:,} | {status}"
