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
) -> plt.Figure:
    """Plot codebook vectors and encoder outputs in 2D.

    Args:
        codebook: Codebook vectors (num_levels, num_codes, 2)
        encoder_outputs: Encoder outputs to scatter (num_samples, 2)
        labels: Labels for encoder outputs (num_samples,)
        assignment_counts: Assignment counts (num_levels, num_codes)
        threshold_pct: Threshold for dead codebook detection

    Returns:
        Matplotlib figure
    """
    num_levels = codebook.shape[0]

    fig, axes = plt.subplots(1, num_levels, figsize=(5 * num_levels, 5))
    if num_levels == 1:
        axes = [axes]

    for level, ax in enumerate(axes):
        level_codebook = codebook[level]  # (num_codes, 2)

        # Determine dead codebooks
        if assignment_counts is not None:
            total_assignments = assignment_counts[level].sum()
            threshold = threshold_pct * total_assignments
            is_dead = assignment_counts[level] < threshold
        else:
            is_dead = np.zeros(len(level_codebook), dtype=bool)

        # Plot encoder outputs
        if encoder_outputs is not None:
            if labels is not None:
                for digit in range(10):
                    mask = labels == digit
                    if mask.any():
                        ax.scatter(
                            encoder_outputs[mask, 0],
                            encoder_outputs[mask, 1],
                            c=DIGIT_COLORS[digit],
                            alpha=0.3,
                            s=10,
                            label=str(digit),
                        )
            else:
                ax.scatter(
                    encoder_outputs[:, 0],
                    encoder_outputs[:, 1],
                    alpha=0.3,
                    s=10,
                    c="gray",
                )

        # Plot codebook vectors
        active_mask = ~is_dead
        dead_mask = is_dead

        # Active codebooks (filled circles)
        if active_mask.any():
            ax.scatter(
                level_codebook[active_mask, 0],
                level_codebook[active_mask, 1],
                c="black",
                s=100,
                marker="o",
                edgecolors="white",
                linewidths=2,
                zorder=10,
                label="Active",
            )

        # Dead codebooks (hollow circles)
        if dead_mask.any():
            ax.scatter(
                level_codebook[dead_mask, 0],
                level_codebook[dead_mask, 1],
                c="white",
                s=100,
                marker="o",
                edgecolors="gray",
                linewidths=2,
                zorder=10,
                label="Dead",
            )

        ax.set_title(f"Level {level + 1}")
        ax.set_xlabel("Latent dim 1")
        ax.set_ylabel("Latent dim 2")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_loss_curves(history: dict[str, list[float]]) -> plt.Figure:
    """Plot loss curves over training.

    Args:
        history: Dict with keys total, recon, commit, codebook

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    if not history.get("total"):
        ax.text(0.5, 0.5, "No data yet", ha="center", va="center", transform=ax.transAxes)
        return fig

    steps = range(len(history["total"]))

    ax.plot(steps, history["total"], label="Total", linewidth=2)
    ax.plot(steps, history["recon"], label="Reconstruction", linestyle="--")
    ax.plot(steps, history["commit"], label="Commitment", linestyle="--")
    ax.plot(steps, history["codebook"], label="Codebook", linestyle="--")

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

        health.append({
            "level": level + 1,
            "active": int(num_codes - is_dead.sum()),
            "dead": int(is_dead.sum()),
            "total": num_codes,
        })

    return health
