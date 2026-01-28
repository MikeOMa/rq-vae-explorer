# Codebook Trajectory Debug Tools

## Problem

Dead codebooks stay dead even with Wasserstein loss enabled. The Wasserstein loss decreases over time, suggesting gradients are flowing somewhere, but dead codes appear frozen. We need visibility into:
1. Whether dead codes are moving at all (trajectory)
2. Whether dead codes are receiving gradient signal (gradient magnitudes)

## Design

### State Tracking (`state.py`)

Add fields:
- `_codebook_history: list[np.ndarray]` - snapshots of codebook positions
- `_codebook_history_steps: list[int]` - step numbers for each snapshot
- `_codebook_grad_ema: np.ndarray | None` - smoothed gradient magnitudes, shape `(num_levels, num_codes)`

Configuration:
- Record snapshots every 50 steps
- Keep max 200 snapshots (~10k steps of history)
- EMA smoothing factor alpha=0.05

New methods:
- `get_codebook_history() -> tuple[list[np.ndarray], list[int]]`
- `get_codebook_grad_ema() -> np.ndarray | None`
- `update_codebook_history(codebook: np.ndarray, step: int)`
- `update_grad_ema(grad_norms: np.ndarray)`

Reset clears history and EMA.

### Trainer Changes (`trainer.py`)

Gradient computation:
- Extract codebook gradients from `grads['params']['quantizer']['codebook']`
- Compute per-vector L2 norm: shape `(num_levels, num_codes)`
- Return from JIT-compiled function

EMA update each step:
```python
alpha = 0.05
smoothed = (1 - alpha) * current_ema + alpha * grad_norms
```

Snapshot recording:
- Every 50 steps: append codebook to history
- Cap at 200 snapshots

Config change:
- `num_codes=4` (down from 16) for easier visualization

### Plotting Functions (`plots.py`)

`plot_codebook_trajectory(history, steps, assignment_counts)`:
- Plot L1 codebook vectors only (4 vectors)
- Each vector gets unique color from tab10 palette
- Lines connect positions over time
- Start = small circle, end = large marker
- Dead vectors = dashed lines, active = solid
- Title shows step range covered

`plot_gradient_magnitudes(grad_ema, assignment_counts)`:
- 8 bars total: L1-0, L1-1, L1-2, L1-3, L2-0, L2-1, L2-2, L2-3
- Gray bars for dead codes, colored for active
- Linear y-axis scale
- Title indicates EMA smoothing

### UI Integration (`app.py`)

Layout:
- New collapsible accordion: "Debug: Codebook Dynamics"
- Contains two plots side-by-side: trajectory (left), gradients (right)

Refresh:
- Pull history and EMA in `refresh_ui()`
- Update both debug plots on timer tick

## Expected Outcome

After implementation, we can observe:
1. If dead codes move at all (trajectory shows movement or stasis)
2. If dead codes receive gradients (bar height > 0 or ~0)

This will reveal whether:
- Dead codes get no gradient (Wasserstein not reaching them)
- Dead codes get gradient but don't move enough (learning rate or magnitude issue)
- Encoder adapts faster than codebook (explaining why Wasserstein loss drops)
