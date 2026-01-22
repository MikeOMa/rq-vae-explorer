import numpy as np
import matplotlib.pyplot as plt
from rq_vae_explorer.ui.plots import (
    plot_codebook_2d,
    plot_loss_curves,
    get_codebook_health,
)


def test_plot_codebook_2d_returns_figure():
    codebook = np.random.randn(2, 16, 2)  # (levels, codes, dim)
    encoder_outputs = np.random.randn(100, 2)
    labels = np.random.randint(0, 10, 100)
    assignment_counts = np.ones((2, 16)) * 10

    fig = plot_codebook_2d(codebook, encoder_outputs, labels, assignment_counts)

    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_loss_curves_returns_figure():
    history = {
        "total": [1.0, 0.8, 0.6],
        "recon": [0.5, 0.4, 0.3],
        "commit": [0.3, 0.2, 0.15],
        "codebook": [0.2, 0.2, 0.15],
    }

    fig = plot_loss_curves(history)

    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_get_codebook_health():
    # 2 levels, 16 codes each
    assignment_counts = np.array(
        [
            [
                10,
                10,
                10,
                0,
                0,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
            ],  # Level 0: 2 dead
            [
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                0,
                0,
                0,
                0,
                10,
                10,
                10,
                10,
            ],  # Level 1: 4 dead
        ]
    )

    health = get_codebook_health(assignment_counts, threshold_pct=0.01)

    assert health[0]["active"] == 14
    assert health[0]["dead"] == 2
    assert health[1]["active"] == 12
    assert health[1]["dead"] == 4
