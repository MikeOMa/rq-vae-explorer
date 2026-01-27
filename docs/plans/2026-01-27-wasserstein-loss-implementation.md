# Wasserstein Loss Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add Sinkhorn-based Wasserstein loss to fight dead codebooks by providing gradients to all codes.

**Architecture:** New `sinkhorn_loss()` function in losses.py computes optimal transport between encoder outputs and codebook. Applied to both RQ levels. Controlled via `lambda_wasserstein` and `sinkhorn_epsilon` parameters flowing from UI → State → Trainer → Loss.

**Tech Stack:** JAX/Flax, Gradio, existing RQ-VAE codebase

---

## Task 1: Add Sinkhorn Loss Function

**Files:**
- Modify: `src/rq_vae_explorer/training/losses.py:1-52`
- Test: `tests/test_losses.py`

**Step 1: Write the failing test for sinkhorn_loss**

Add to `tests/test_losses.py`:

```python
def test_sinkhorn_loss_basic():
    """Sinkhorn loss computes optimal transport between points and codebook."""
    from rq_vae_explorer.training.losses import sinkhorn_loss

    # Simple case: 4 points, 4 codebook entries
    points = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    codebook = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    loss = sinkhorn_loss(points, codebook, epsilon=0.05, num_iters=20)

    # Perfect match should give near-zero loss
    assert loss >= 0
    assert loss < 0.1


def test_sinkhorn_loss_nonzero_for_mismatch():
    """Sinkhorn loss is higher when points don't match codebook."""
    from rq_vae_explorer.training.losses import sinkhorn_loss

    points = jnp.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1]])
    codebook = jnp.array([[5.0, 5.0], [6.0, 5.0], [5.0, 6.0], [6.0, 6.0]])

    loss = sinkhorn_loss(points, codebook, epsilon=0.05, num_iters=20)

    # Far apart should give significant loss
    assert loss > 1.0
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_losses.py::test_sinkhorn_loss_basic -v`
Expected: FAIL with "cannot import name 'sinkhorn_loss'"

**Step 3: Implement sinkhorn_loss**

Add to `src/rq_vae_explorer/training/losses.py` after the imports:

```python
def sinkhorn_loss(
    points: jnp.ndarray,
    codebook: jnp.ndarray,
    epsilon: float = 0.05,
    num_iters: int = 20,
) -> jnp.ndarray:
    """Compute Sinkhorn (entropy-regularized optimal transport) loss.

    Args:
        points: Batch of points (batch_size, latent_dim)
        codebook: Codebook vectors (num_codes, latent_dim)
        epsilon: Entropy regularization strength (lower = sharper)
        num_iters: Number of Sinkhorn iterations

    Returns:
        Scalar Wasserstein distance approximation
    """
    batch_size = points.shape[0]
    num_codes = codebook.shape[0]

    # Cost matrix: squared Euclidean distances (batch_size, num_codes)
    diff = points[:, None, :] - codebook[None, :, :]  # (batch, codes, dim)
    C = jnp.sum(diff ** 2, axis=-1)  # (batch, codes)

    # Kernel matrix
    K = jnp.exp(-C / epsilon)

    # Uniform marginals
    a = jnp.ones(batch_size) / batch_size
    b = jnp.ones(num_codes) / num_codes

    # Sinkhorn iterations
    u = jnp.ones(batch_size)
    v = jnp.ones(num_codes)

    for _ in range(num_iters):
        u = a / (K @ v + 1e-8)
        v = b / (K.T @ u + 1e-8)

    # Transport plan and cost
    transport = u[:, None] * K * v[None, :]
    loss = jnp.sum(transport * C)

    return loss
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_losses.py::test_sinkhorn_loss_basic tests/test_losses.py::test_sinkhorn_loss_nonzero_for_mismatch -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/rq_vae_explorer/training/losses.py tests/test_losses.py
git commit -m "feat(losses): add sinkhorn_loss for optimal transport"
```

---

## Task 2: Extend compute_losses with Wasserstein

**Files:**
- Modify: `src/rq_vae_explorer/training/losses.py:7-52`
- Test: `tests/test_losses.py`

**Step 1: Write failing test for wasserstein in compute_losses**

Add to `tests/test_losses.py`:

```python
def test_compute_losses_with_wasserstein():
    """compute_losses includes wasserstein when lambda > 0."""
    x = jnp.ones((4, 28, 28, 1)) * 0.5
    x_recon = jnp.ones((4, 28, 28, 1)) * 0.6
    z_e = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    z_q = z_e * 1.1
    z_q1 = z_e * 0.5
    codebook = jnp.ones((2, 16, 2)) * 0.5  # (num_levels, num_codes, latent_dim)

    losses = compute_losses(
        x=x,
        x_recon=x_recon,
        z_e=z_e,
        z_q=z_q,
        codebook=codebook,
        z_q1=z_q1,
        lambda_commit=0.25,
        lambda_codebook=1.0,
        lambda_wasserstein=0.5,
        sinkhorn_epsilon=0.05,
    )

    assert "wasserstein" in losses
    assert losses["wasserstein"] >= 0


def test_compute_losses_wasserstein_zero_when_disabled():
    """wasserstein loss is 0 when lambda_wasserstein=0."""
    x = jnp.ones((4, 28, 28, 1)) * 0.5
    x_recon = jnp.ones((4, 28, 28, 1)) * 0.6
    z_e = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    z_q = z_e * 1.1
    z_q1 = z_e * 0.5
    codebook = jnp.ones((2, 16, 2)) * 0.5

    losses = compute_losses(
        x=x,
        x_recon=x_recon,
        z_e=z_e,
        z_q=z_q,
        codebook=codebook,
        z_q1=z_q1,
        lambda_commit=0.25,
        lambda_codebook=1.0,
        lambda_wasserstein=0.0,
        sinkhorn_epsilon=0.05,
    )

    assert losses["wasserstein"] == 0.0
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_losses.py::test_compute_losses_with_wasserstein -v`
Expected: FAIL with "unexpected keyword argument 'codebook'"

**Step 3: Update compute_losses signature and implementation**

Replace entire `compute_losses` function in `src/rq_vae_explorer/training/losses.py`:

```python
def compute_losses(
    x: jnp.ndarray,
    x_recon: jnp.ndarray,
    z_e: jnp.ndarray,
    z_q: jnp.ndarray,
    codebook: jnp.ndarray | None = None,
    z_q1: jnp.ndarray | None = None,
    lambda_commit: float = 0.25,
    lambda_codebook: float = 1.0,
    lambda_wasserstein: float = 0.0,
    sinkhorn_epsilon: float = 0.05,
) -> dict[str, jnp.ndarray]:
    """Compute all RQ-VAE loss components.

    Loss = recon + lambda_commit * commit + lambda_codebook * codebook
           + lambda_wasserstein * wasserstein

    Args:
        x: Original input images (batch, H, W, C)
        x_recon: Reconstructed images (batch, H, W, C)
        z_e: Encoder output before quantization (batch, latent_dim)
        z_q: Quantized latent vectors (batch, latent_dim)
        codebook: Full codebook (num_levels, num_codes, latent_dim)
        z_q1: Level 1 quantized output (batch, latent_dim)
        lambda_commit: Weight for commitment loss
        lambda_codebook: Weight for codebook loss
        lambda_wasserstein: Weight for Wasserstein loss (0 = disabled)
        sinkhorn_epsilon: Sinkhorn entropy regularization

    Returns:
        Dict with keys: total, recon, commit, codebook, wasserstein
    """
    # Reconstruction loss: MSE between input and reconstruction
    recon_loss = jnp.mean((x - x_recon) ** 2)

    # Commitment loss: encourages encoder to commit to codebook vectors
    commit_loss = jnp.mean((z_e - jax.lax.stop_gradient(z_q)) ** 2)

    # Codebook loss: moves codebook vectors toward encoder outputs
    codebook_loss = jnp.mean((jax.lax.stop_gradient(z_e) - z_q) ** 2)

    # Wasserstein loss: optimal transport between encoder outputs and codebook
    if lambda_wasserstein > 0 and codebook is not None and z_q1 is not None:
        # Level 1: transport between z_e and codebook[0]
        w_loss_l1 = sinkhorn_loss(z_e, codebook[0], sinkhorn_epsilon)

        # Level 2: transport between residuals and codebook[1]
        residuals = z_e - z_q1
        w_loss_l2 = sinkhorn_loss(residuals, codebook[1], sinkhorn_epsilon)

        wasserstein_loss = w_loss_l1 + w_loss_l2
    else:
        wasserstein_loss = jnp.array(0.0)

    # Total loss
    total_loss = (
        recon_loss
        + lambda_commit * commit_loss
        + lambda_codebook * codebook_loss
        + lambda_wasserstein * wasserstein_loss
    )

    return {
        "total": total_loss,
        "recon": recon_loss,
        "commit": commit_loss,
        "codebook": codebook_loss,
        "wasserstein": wasserstein_loss,
    }
```

**Step 4: Run all loss tests to verify they pass**

Run: `uv run pytest tests/test_losses.py -v`
Expected: PASS (all 6 tests)

**Step 5: Commit**

```bash
git add src/rq_vae_explorer/training/losses.py tests/test_losses.py
git commit -m "feat(losses): add wasserstein loss to compute_losses"
```

---

## Task 3: Extend TrainingState with Wasserstein Parameters

**Files:**
- Modify: `src/rq_vae_explorer/training/state.py:16-67`
- Test: `tests/test_state.py`

**Step 1: Write failing test for new state parameters**

Add to `tests/test_state.py`:

```python
def test_training_state_wasserstein_params():
    """State tracks lambda_wasserstein and sinkhorn_epsilon."""
    state = TrainingState()

    # Default values
    assert state.lambda_wasserstein == 0.0
    assert state.sinkhorn_epsilon == 0.05

    # Setters work
    state.set_lambda_wasserstein(0.5)
    state.set_sinkhorn_epsilon(0.1)

    assert state.lambda_wasserstein == 0.5
    assert state.sinkhorn_epsilon == 0.1


def test_training_state_get_all_lambdas():
    """get_all_lambdas returns all four parameters."""
    state = TrainingState()
    state.set_lambda_commit(0.3)
    state.set_lambda_codebook(0.8)
    state.set_lambda_wasserstein(0.5)
    state.set_sinkhorn_epsilon(0.1)

    commit, codebook, wasserstein, epsilon = state.get_all_lambdas()

    assert commit == 0.3
    assert codebook == 0.8
    assert wasserstein == 0.5
    assert epsilon == 0.1
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_state.py::test_training_state_wasserstein_params -v`
Expected: FAIL with "has no attribute 'lambda_wasserstein'"

**Step 3: Add wasserstein parameters to TrainingState**

In `src/rq_vae_explorer/training/state.py`, add after line 18 (`_lambda_codebook`):

```python
    _lambda_wasserstein: float = 0.0
    _sinkhorn_epsilon: float = 0.05
```

Add new property/setter methods after `get_lambdas()` (around line 67):

```python
    @property
    def lambda_wasserstein(self) -> float:
        with self._lock:
            return self._lambda_wasserstein

    @property
    def sinkhorn_epsilon(self) -> float:
        with self._lock:
            return self._sinkhorn_epsilon

    def set_lambda_wasserstein(self, value: float) -> None:
        with self._lock:
            self._lambda_wasserstein = value

    def set_sinkhorn_epsilon(self, value: float) -> None:
        with self._lock:
            self._sinkhorn_epsilon = value

    def get_all_lambdas(self) -> tuple[float, float, float, float]:
        """Get all lambda values atomically."""
        with self._lock:
            return (
                self._lambda_commit,
                self._lambda_codebook,
                self._lambda_wasserstein,
                self._sinkhorn_epsilon,
            )
```

**Step 4: Update loss_history in reset() method**

In the `reset()` method around line 104, update the `_loss_history` initialization:

```python
            self._loss_history = {
                "total": [],
                "recon": [],
                "commit": [],
                "codebook": [],
                "wasserstein": [],
            }
```

Also update the initial `_loss_history` field default around line 26:

```python
    _loss_history: dict[str, list[float]] = field(
        default_factory=lambda: {
            "total": [],
            "recon": [],
            "commit": [],
            "codebook": [],
            "wasserstein": [],
        }
    )
```

**Step 5: Run state tests to verify they pass**

Run: `uv run pytest tests/test_state.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/rq_vae_explorer/training/state.py tests/test_state.py
git commit -m "feat(state): add lambda_wasserstein and sinkhorn_epsilon"
```

---

## Task 4: Update Trainer to Pass Wasserstein Parameters

**Files:**
- Modify: `src/rq_vae_explorer/training/trainer.py:74-96`
- Test: `tests/test_trainer.py`

**Step 1: Write failing test**

Add to `tests/test_trainer.py`:

```python
def test_trainer_uses_wasserstein_params():
    """Trainer passes wasserstein params to loss computation."""
    state = TrainingState()
    state.set_lambda_wasserstein(0.5)
    state.set_sinkhorn_epsilon(0.1)

    trainer = Trainer(state=state)
    trainer.train_step()

    # Check wasserstein is in loss history
    history = state.get_loss_history()
    assert "wasserstein" in history
    assert len(history["wasserstein"]) == 1
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_trainer.py::test_trainer_uses_wasserstein_params -v`
Expected: FAIL (wasserstein not in history or key error)

**Step 3: Update _train_step_inner signature**

In `src/rq_vae_explorer/training/trainer.py`, update the `_train_step_inner` method signature (around line 74):

```python
    def _train_step_inner(
        self,
        params: Any,
        opt_state: Any,
        batch: jnp.ndarray,
        lambda_commit: float,
        lambda_codebook: float,
        lambda_wasserstein: float,
        sinkhorn_epsilon: float,
    ) -> tuple[
        Any, Any, dict, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
    ]:
        """Inner training step (JIT compiled)."""

        def loss_fn(params):
            x_recon, aux = self.model.apply(params, batch)
            losses = compute_losses(
                x=batch,
                x_recon=x_recon,
                z_e=aux["z_e"],
                z_q=aux["z_q"],
                codebook=aux["codebook"],
                z_q1=aux["z_q1"],
                lambda_commit=lambda_commit,
                lambda_codebook=lambda_codebook,
                lambda_wasserstein=lambda_wasserstein,
                sinkhorn_epsilon=sinkhorn_epsilon,
            )
            return losses["total"], (losses, aux, x_recon)

        (loss, (losses, aux, x_recon)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(params)

        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return (
            params,
            opt_state,
            losses,
            aux["codebook"],
            aux["indices"],
            aux["z_e"],
            aux["z_q1"],
            aux["z_q"],
        )
```

**Step 4: Update train_step to pass new parameters**

Update the `train_step` method (around line 116):

```python
    def train_step(self) -> None:
        """Execute a single training step."""
        batch_images, batch_labels = next(self.data_iterator)
        lambda_commit, lambda_codebook, lambda_wasserstein, sinkhorn_epsilon = (
            self.state.get_all_lambdas()
        )

        (
            self.params,
            self.opt_state,
            losses,
            codebook,
            indices,
            encoder_outputs,
            z_q1,
            z_q,
        ) = self._train_step_jit(
            self.params,
            self.opt_state,
            batch_images,
            lambda_commit,
            lambda_codebook,
            lambda_wasserstein,
            sinkhorn_epsilon,
        )
```

**Step 5: Run trainer tests to verify they pass**

Run: `uv run pytest tests/test_trainer.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/rq_vae_explorer/training/trainer.py tests/test_trainer.py
git commit -m "feat(trainer): pass wasserstein params to loss computation"
```

---

## Task 5: Add UI Sliders for Wasserstein Parameters

**Files:**
- Modify: `src/rq_vae_explorer/ui/controls.py:6-30`
- Test: `tests/test_controls.py`

**Step 1: Write failing test**

Add to `tests/test_controls.py`:

```python
def test_create_wasserstein_sliders():
    """Wasserstein sliders are created with correct defaults."""
    from rq_vae_explorer.ui.controls import create_wasserstein_sliders

    lambda_w, epsilon = create_wasserstein_sliders()

    assert lambda_w.value == 0.0
    assert epsilon.value == 0.05
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_controls.py::test_create_wasserstein_sliders -v`
Expected: FAIL with "cannot import name 'create_wasserstein_sliders'"

**Step 3: Add create_wasserstein_sliders function**

Add to `src/rq_vae_explorer/ui/controls.py` after `create_lambda_sliders`:

```python
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
```

**Step 4: Run controls tests to verify they pass**

Run: `uv run pytest tests/test_controls.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/rq_vae_explorer/ui/controls.py tests/test_controls.py
git commit -m "feat(ui): add wasserstein parameter sliders"
```

---

## Task 6: Wire Up UI Sliders in App

**Files:**
- Modify: `src/rq_vae_explorer/ui/app.py:16-21,66-68,89-93,141-142`

**Step 1: Update imports**

In `src/rq_vae_explorer/ui/app.py`, update the controls import (line 16):

```python
from rq_vae_explorer.ui.controls import (
    create_lambda_sliders,
    create_wasserstein_sliders,
    create_training_controls,
    format_health_text,
    format_step_text,
)
```

**Step 2: Add wasserstein sliders to UI**

After line 67 (`lambda_commit, lambda_codebook = create_lambda_sliders()`), add:

```python
        with gr.Row():
            lambda_wasserstein, sinkhorn_epsilon = create_wasserstein_sliders()
```

**Step 3: Add event handlers for new sliders**

After line 93 (`def on_lambda_codebook_change`), add:

```python
        def on_lambda_wasserstein_change(value):
            state.set_lambda_wasserstein(value)

        def on_sinkhorn_epsilon_change(value):
            state.set_sinkhorn_epsilon(value)
```

**Step 4: Wire up new event handlers**

After line 142 (`lambda_codebook.change(...)`), add:

```python
        lambda_wasserstein.change(
            on_lambda_wasserstein_change, inputs=[lambda_wasserstein]
        )
        sinkhorn_epsilon.change(on_sinkhorn_epsilon_change, inputs=[sinkhorn_epsilon])
```

**Step 5: Run integration test to verify app still works**

Run: `uv run pytest tests/test_integration.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/rq_vae_explorer/ui/app.py
git commit -m "feat(ui): wire up wasserstein sliders in app"
```

---

## Task 7: Add Wasserstein to Loss Plot

**Files:**
- Modify: `src/rq_vae_explorer/ui/plots.py:253-284`
- Test: `tests/test_plots.py`

**Step 1: Write failing test**

Add to `tests/test_plots.py`:

```python
def test_plot_loss_curves_includes_wasserstein():
    """Loss plot includes wasserstein line when present."""
    history = {
        "total": [1.0, 0.9, 0.8],
        "recon": [0.5, 0.4, 0.3],
        "commit": [0.2, 0.2, 0.2],
        "codebook": [0.3, 0.3, 0.3],
        "wasserstein": [0.1, 0.1, 0.1],
    }

    fig = plot_loss_curves(history)
    ax = fig.axes[0]

    # Should have 5 lines (total, recon, commit, codebook, wasserstein)
    assert len(ax.lines) == 5
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_plots.py::test_plot_loss_curves_includes_wasserstein -v`
Expected: FAIL (only 4 lines)

**Step 3: Update plot_loss_curves to include wasserstein**

In `src/rq_vae_explorer/ui/plots.py`, update `plot_loss_curves` (around line 253):

```python
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
```

**Step 4: Run plot tests to verify they pass**

Run: `uv run pytest tests/test_plots.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/rq_vae_explorer/ui/plots.py tests/test_plots.py
git commit -m "feat(plots): show wasserstein loss in loss curves"
```

---

## Task 8: Run Full Test Suite and Manual Verification

**Step 1: Run all tests**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS

**Step 2: Run the app manually**

Run: `uv run rq-vae-explorer`

Verify:
- New sliders appear: "λ_wasserstein" and "Sinkhorn ε"
- Setting λ_wasserstein > 0 and training shows wasserstein in loss plot
- Dead codes start receiving gradients (over time, hollow circles should fill)

**Step 3: Final commit if any fixes needed**

```bash
git add -A
git commit -m "fix: address any issues found in manual testing"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Add sinkhorn_loss function | losses.py, test_losses.py |
| 2 | Extend compute_losses with wasserstein | losses.py, test_losses.py |
| 3 | Add state parameters | state.py, test_state.py |
| 4 | Update trainer to pass params | trainer.py, test_trainer.py |
| 5 | Add UI sliders | controls.py, test_controls.py |
| 6 | Wire up sliders in app | app.py |
| 7 | Add wasserstein to loss plot | plots.py, test_plots.py |
| 8 | Full test suite + manual verification | - |
