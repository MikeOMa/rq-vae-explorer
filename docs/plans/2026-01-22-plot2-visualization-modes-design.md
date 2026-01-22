# Plot 2 Visualization Modes Design

Add a dropdown to switch Plot 2 between two visualization modes: Residuals and Cumulative.

## Current Behavior

Both plots show the same `encoder_outputs` (z_e) against their respective codebook levels. This is incorrect for level 2 — it operates on residuals, not raw encoder outputs.

## New Behavior

**Plot 1 (unchanged):** Level 1 — z_e points colored by digit + level 1 codebook centers

**Plot 2 (dropdown-controlled):**

| Mode | Points | Codebook | Title |
|------|--------|----------|-------|
| Residuals | (z_e - z_q1) colored by digit | Level 2 centers | "Level 2 (Residuals)" |
| Cumulative | z_e (circles) + z_q (x markers) + connecting lines | Level 1 centers | "Cumulative Quantization" |

## Data Requirements

The quantizer aux dict needs to include `z_q1` (level 1 quantized only):

| Field | Shape | Description |
|-------|-------|-------------|
| `z_e` | (N, 2) | Encoder outputs (existing) |
| `z_q` | (N, 2) | Final quantized z_q1 + z_q2 (existing) |
| `z_q1` | (N, 2) | Level 1 quantized only (NEW) |

Residuals computed in plotting code as `z_e - z_q1`.

## Code Changes

### 1. quantizer.py

Capture z_q1 during the quantization loop:

```python
z_q1 = None

for level in range(self.num_levels):
    # ... existing distance/index computation ...
    quantized = quantized + level_quantized

    if level == 0:
        z_q1 = quantized

    residual = residual - level_quantized

aux = {
    "indices": indices,
    "codebook": self.codebook,
    "z_e": z,
    "z_q": quantized,
    "z_q1": z_q1,  # NEW
}
```

### 2. state.py

Add new fields:

```python
_z_q1: np.ndarray | None = None
_z_q: np.ndarray | None = None
```

Update methods:
- `update()`: Accept `z_q1` and `z_q` parameters
- `reset()`: Clear both fields
- Add `get_quantized_outputs() -> tuple[np.ndarray | None, np.ndarray | None]`

### 3. trainer.py

Pass z_q1 and z_q from quantizer aux dict to state.update().

### 4. plots.py

Update `plot_codebook_2d` to accept:
- `mode: str` — "Residuals" or "Cumulative"
- `z_q1: np.ndarray | None`
- `z_q: np.ndarray | None`

Plot 2 rendering logic:
- **Residuals mode**: scatter (z_e - z_q1), show level 2 codebook
- **Cumulative mode**: scatter z_e (circles) + z_q (x markers) + connecting lines, show level 1 codebook

### 5. app.py

Add dropdown component:

```python
mode_dropdown = gr.Dropdown(
    choices=["Residuals", "Cumulative"],
    value="Residuals",
    label="Plot 2 Mode",
)
```

Wire dropdown value to plot refresh function.
