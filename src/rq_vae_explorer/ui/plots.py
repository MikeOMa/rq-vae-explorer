"""Plotting functions for the UI."""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


# Color map for digits 0-9
DIGIT_COLORS = list(mcolors.TABLEAU_COLORS.values())


def plot_codebook_2d(
    codebook: np.ndarray,
    encoder_outputs: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    assignment_counts: np.ndarray | None = None,
    threshold_pct: float = 0.01,
    z_q1: np.ndarray | None = None,
    z_q: np.ndarray | None = None,
    mode: str = "Residuals",
) -> plt.Figure:
    """Plot codebook vectors and encoder outputs in 2D.

    Args:
        codebook: Codebook vectors (num_levels, num_codes, 2)
        encoder_outputs: Encoder outputs z_e (num_samples, 2)
        labels: Labels for encoder outputs (num_samples,)
        assignment_counts: Assignment counts (num_levels, num_codes)
        threshold_pct: Threshold for dead codebook detection
        z_q1: Level 1 quantized outputs (num_samples, 2)
        z_q: Final quantized outputs (num_samples, 2)
        mode: "Residuals" or "Cumulative" for plot 2

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # --- Plot 1: Level 1 (always the same) ---
    ax1 = axes[0]
    level_codebook = codebook[0]  # (num_codes, 2)

    # Determine dead codebooks for level 1
    if assignment_counts is not None:
        total_assignments = assignment_counts[0].sum()
        threshold = threshold_pct * total_assignments
        is_dead = assignment_counts[0] < threshold
    else:
        is_dead = np.zeros(len(level_codebook), dtype=bool)

    # Plot encoder outputs on plot 1
    if encoder_outputs is not None:
        _scatter_points(ax1, encoder_outputs, labels, marker="o")

    # Plot level 1 codebook vectors
    _plot_codebook_centers(ax1, level_codebook, is_dead)

    ax1.set_title("Level 1")
    ax1.set_xlabel("Latent dim 1")
    ax1.set_ylabel("Latent dim 2")
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Mode-dependent ---
    ax2 = axes[1]

    if mode == "Cumulative":
        _plot_cumulative(
            ax2,
            codebook[0],
            encoder_outputs,
            z_q,
            labels,
            assignment_counts,
            threshold_pct,
        )
    else:  # Residuals
        _plot_residuals(
            ax2,
            codebook,
            encoder_outputs,
            z_q1,
            labels,
            assignment_counts,
            threshold_pct,
        )

    plt.tight_layout()
    return fig


def _scatter_points(
    ax: plt.Axes,
    points: np.ndarray,
    labels: np.ndarray | None,
    marker: str = "o",
    alpha: float = 0.3,
    size: int = 10,
) -> None:
    """Scatter points colored by label."""
    if labels is not None:
        for digit in range(10):
            mask = labels == digit
            if mask.any():
                ax.scatter(
                    points[mask, 0],
                    points[mask, 1],
                    c=DIGIT_COLORS[digit],
                    alpha=alpha,
                    s=size,
                    marker=marker,
                    label=str(digit) if marker == "o" else None,
                )
    else:
        ax.scatter(
            points[:, 0],
            points[:, 1],
            alpha=alpha,
            s=size,
            c="gray",
            marker=marker,
        )


def _plot_codebook_centers(
    ax: plt.Axes,
    codebook: np.ndarray,
    is_dead: np.ndarray,
) -> None:
    """Plot codebook centers (active filled, dead hollow)."""
    active_mask = ~is_dead
    dead_mask = is_dead

    if active_mask.any():
        ax.scatter(
            codebook[active_mask, 0],
            codebook[active_mask, 1],
            c="black",
            s=100,
            marker="o",
            edgecolors="white",
            linewidths=2,
            zorder=10,
            label="Active",
        )

    if dead_mask.any():
        ax.scatter(
            codebook[dead_mask, 0],
            codebook[dead_mask, 1],
            c="white",
            s=100,
            marker="o",
            edgecolors="gray",
            linewidths=2,
            zorder=10,
            label="Dead",
        )


def _plot_residuals(
    ax: plt.Axes,
    codebook: np.ndarray,
    encoder_outputs: np.ndarray | None,
    z_q1: np.ndarray | None,
    labels: np.ndarray | None,
    assignment_counts: np.ndarray | None,
    threshold_pct: float,
) -> None:
    """Plot level 2 with residuals (z_e - z_q1)."""
    level_codebook = codebook[1] if codebook.shape[0] > 1 else codebook[0]

    # Determine dead codebooks for level 2
    if assignment_counts is not None and assignment_counts.shape[0] > 1:
        total_assignments = assignment_counts[1].sum()
        threshold = threshold_pct * total_assignments
        is_dead = assignment_counts[1] < threshold
    else:
        is_dead = np.zeros(len(level_codebook), dtype=bool)

    # Plot residuals
    if encoder_outputs is not None and z_q1 is not None:
        residuals = encoder_outputs - z_q1
        _scatter_points(ax, residuals, labels, marker="o")

    # Plot level 2 codebook
    _plot_codebook_centers(ax, level_codebook, is_dead)

    ax.set_title("Level 2 (Residuals)")
    ax.set_xlabel("Residual dim 1")
    ax.set_ylabel("Residual dim 2")
    ax.grid(True, alpha=0.3)


def _plot_cumulative(
    ax: plt.Axes,
    level1_codebook: np.ndarray,
    encoder_outputs: np.ndarray | None,
    z_q: np.ndarray | None,
    labels: np.ndarray | None,
    assignment_counts: np.ndarray | None,
    threshold_pct: float,
) -> None:
    """Plot cumulative view: z_e, z_q, and connecting lines."""
    # Determine dead codebooks for level 1 (shown in cumulative view)
    if assignment_counts is not None:
        total_assignments = assignment_counts[0].sum()
        threshold = threshold_pct * total_assignments
        is_dead = assignment_counts[0] < threshold
    else:
        is_dead = np.zeros(len(level1_codebook), dtype=bool)

    # Plot encoder outputs (circles)
    if encoder_outputs is not None:
        _scatter_points(ax, encoder_outputs, labels, marker="o", alpha=0.4)

    # Plot final quantized (x markers) and connecting lines
    if encoder_outputs is not None and z_q is not None:
        # Draw connecting lines
        if labels is not None:
            for digit in range(10):
                mask = labels == digit
                if mask.any():
                    for i in np.where(mask)[0]:
                        ax.plot(
                            [encoder_outputs[i, 0], z_q[i, 0]],
                            [encoder_outputs[i, 1], z_q[i, 1]],
                            c=DIGIT_COLORS[digit],
                            alpha=0.2,
                            linewidth=0.5,
                        )
        else:
            for i in range(len(encoder_outputs)):
                ax.plot(
                    [encoder_outputs[i, 0], z_q[i, 0]],
                    [encoder_outputs[i, 1], z_q[i, 1]],
                    c="gray",
                    alpha=0.2,
                    linewidth=0.5,
                )

        # Plot z_q points (x markers)
        _scatter_points(ax, z_q, labels, marker="x", alpha=0.6, size=20)

    # Plot level 1 codebook
    _plot_codebook_centers(ax, level1_codebook, is_dead)

    ax.set_title("Cumulative Quantization")
    ax.set_xlabel("Latent dim 1")
    ax.set_ylabel("Latent dim 2")
    ax.grid(True, alpha=0.3)


def plot_loss_curves(history: dict[str, list[float]]) -> plt.Figure:
    """Plot loss curves over training.

    Args:
        history: Dict with keys total, recon, commit, codebook, wasserstein

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    if not history.get("total"):
        ax.text(
            0.5, 0.5, "No data yet", ha="center", va="center", transform=ax.transAxes
        )
        return fig

    steps = range(len(history["total"]))

    ax.plot(steps, history["total"], label="Total", linewidth=2)
    ax.plot(steps, history["recon"], label="Reconstruction", linestyle="--")
    ax.plot(steps, history["commit"], label="Commitment", linestyle="--")
    ax.plot(steps, history["codebook"], label="Codebook", linestyle="--")

    # Only show wasserstein if it has non-zero values
    if history.get("wasserstein") and any(v > 0 for v in history["wasserstein"]):
        ax.plot(steps, history["wasserstein"], label="Wasserstein", linestyle="--")

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_reconstructions(
    inputs: np.ndarray | None,
    reconstructions: np.ndarray | None,
    num_samples: int = 8,
) -> plt.Figure:
    """Plot input images and their reconstructions side by side.

    Args:
        inputs: Input images (N, 28, 28, 1)
        reconstructions: Reconstructed images (N, 28, 28, 1)
        num_samples: Number of samples to show

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 1.5, 3))

    if inputs is None or reconstructions is None:
        for ax in axes.flat:
            ax.axis("off")
        axes[0, num_samples // 2].text(
            0.5, 0.5, "No data yet", ha="center", va="center"
        )
        return fig

    num_samples = min(num_samples, len(inputs))

    for i in range(num_samples):
        # Input
        axes[0, i].imshow(inputs[i, :, :, 0], cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("Input", fontsize=10)

        # Reconstruction
        axes[1, i].imshow(reconstructions[i, :, :, 0], cmap="gray", vmin=0, vmax=1)
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title("Recon", fontsize=10)

    plt.tight_layout()
    return fig


def get_codebook_health(
    assignment_counts: np.ndarray | None,
    threshold_pct: float = 0.01,
) -> list[dict[str, int]]:
    """Get codebook health statistics per level.

    Args:
        assignment_counts: Assignment counts (num_levels, num_codes)
        threshold_pct: Threshold percentage for dead detection

    Returns:
        List of dicts with active/dead counts per level
    """
    if assignment_counts is None:
        return []

    health = []
    for level in range(assignment_counts.shape[0]):
        total = assignment_counts[level].sum()
        threshold = threshold_pct * total if total > 0 else 0

        is_dead = assignment_counts[level] < threshold
        num_codes = len(assignment_counts[level])

        health.append(
            {
                "level": level + 1,
                "active": int(num_codes - is_dead.sum()),
                "dead": int(is_dead.sum()),
                "total": num_codes,
            }
        )

    return health


# Colors for individual codebook vectors (4 codes)
CODEBOOK_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def plot_codebook_trajectory(
    history: list[np.ndarray],
    steps: list[int],
    assignment_counts: np.ndarray | None = None,
    threshold_pct: float = 0.01,
    level: int = 0,
) -> plt.Figure:
    """Plot trajectory of codebook vectors over training.

    Args:
        history: List of codebook snapshots (num_levels, num_codes, 2)
        steps: Step numbers corresponding to each snapshot
        assignment_counts: Current assignment counts (num_levels, num_codes)
        threshold_pct: Threshold for dead codebook detection
        level: Which level to plot (0 = L1, 1 = L2)

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    level_name = f"L{level + 1}"

    if not history or len(history) < 2:
        ax.text(
            0.5,
            0.5,
            "Not enough data yet",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title(f"{level_name} Codebook Trajectory")
        return fig

    num_codes = history[0].shape[1]

    # Determine dead codebooks from current assignment counts
    if assignment_counts is not None and level < assignment_counts.shape[0]:
        total_assignments = assignment_counts[level].sum()
        threshold = threshold_pct * total_assignments
        is_dead = assignment_counts[level] < threshold
    else:
        is_dead = np.zeros(num_codes, dtype=bool)

    # Plot trajectory for each codebook vector at the specified level
    for code_idx in range(num_codes):
        positions = np.array([h[level, code_idx] for h in history])
        color = CODEBOOK_COLORS[code_idx % len(CODEBOOK_COLORS)]
        linestyle = "--" if is_dead[code_idx] else "-"
        alpha = 0.5 if is_dead[code_idx] else 0.8

        # Draw trajectory line
        ax.plot(
            positions[:, 0],
            positions[:, 1],
            color=color,
            linestyle=linestyle,
            alpha=alpha,
            linewidth=2,
            label=f"Code {code_idx}" + (" (dead)" if is_dead[code_idx] else ""),
        )

        # Mark start (small circle)
        ax.scatter(
            positions[0, 0],
            positions[0, 1],
            color=color,
            s=30,
            marker="o",
            alpha=alpha,
            zorder=5,
        )

        # Mark end (large marker)
        ax.scatter(
            positions[-1, 0],
            positions[-1, 1],
            color=color,
            s=100,
            marker="s",
            edgecolors="white",
            linewidths=1.5,
            alpha=1.0,
            zorder=10,
        )

    step_range = f"Steps {steps[0]}-{steps[-1]}" if steps else ""
    ax.set_title(f"{level_name} Codebook Trajectory ({step_range})")
    ax.set_xlabel("Latent dim 1")
    ax.set_ylabel("Latent dim 2")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_decoded_codebooks(
    decoded_images: np.ndarray | None,
    num_codes: int = 4,
) -> plt.Figure:
    """Plot decoded codebook combinations as a grid.

    Args:
        decoded_images: Decoded images (num_l1 * num_l2, 28, 28, 1)
        num_codes: Number of codes per level

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(num_codes, num_codes, figsize=(6, 6))

    if decoded_images is None:
        for ax in axes.flat:
            ax.axis("off")
        axes[0, 0].text(
            0.5,
            0.5,
            "No data yet",
            ha="center",
            va="center",
            transform=axes[0, 0].transAxes,
        )
        fig.suptitle("Decoded Codebook Combinations")
        return fig

    for i in range(num_codes):
        for j in range(num_codes):
            idx = i * num_codes + j
            ax = axes[i, j]
            if idx < len(decoded_images):
                ax.imshow(decoded_images[idx, :, :, 0], cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if i == 0:
                ax.set_title(f"L2-{j}", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"L1-{i}", fontsize=9)
                ax.yaxis.set_visible(True)
                ax.set_yticks([])

    fig.suptitle("Decoded: L1[row] + L2[col]", fontsize=11)
    plt.tight_layout()
    return fig


def plot_gradient_magnitudes(
    grad_ema: np.ndarray | None,
    assignment_counts: np.ndarray | None = None,
    threshold_pct: float = 0.01,
) -> plt.Figure:
    """Plot smoothed gradient magnitudes per codebook vector.

    Args:
        grad_ema: Smoothed gradient norms (num_levels, num_codes)
        assignment_counts: Assignment counts (num_levels, num_codes)
        threshold_pct: Threshold for dead codebook detection

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    if grad_ema is None:
        ax.text(
            0.5,
            0.5,
            "No gradient data yet",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Gradient Magnitudes (EMA)")
        return fig

    num_levels, num_codes = grad_ema.shape

    # Determine dead codebooks
    if assignment_counts is not None:
        is_dead = np.zeros_like(grad_ema, dtype=bool)
        for level in range(num_levels):
            total = assignment_counts[level].sum()
            threshold = threshold_pct * total if total > 0 else 0
            is_dead[level] = assignment_counts[level] < threshold
    else:
        is_dead = np.zeros_like(grad_ema, dtype=bool)

    # Create bar chart
    labels = []
    values = []
    colors = []

    for level in range(num_levels):
        for code_idx in range(num_codes):
            labels.append(f"L{level + 1}-{code_idx}")
            values.append(grad_ema[level, code_idx])
            if is_dead[level, code_idx]:
                colors.append("gray")
            else:
                colors.append(CODEBOOK_COLORS[code_idx % len(CODEBOOK_COLORS)])

    x = np.arange(len(labels))
    ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Gradient Norm (EMA)")
    ax.set_title("Gradient Magnitudes (gray = dead)")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig
