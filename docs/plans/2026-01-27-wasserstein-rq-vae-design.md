# Wasserstein RQ-VAE Design

## Problem

Dead codebooks: some codebook entries are never selected during training because they're too far from the data distribution. With standard VQ-VAE, only the "winning" (nearest) code receives gradients, so unused codes never move toward the data.

## Solution

Add a Wasserstein (optimal transport) loss that provides gradients to all codebook entries, pulling unused codes toward the data distribution.

## Design

### Loss Function

Extend `compute_losses()` with new parameters:

```python
def compute_losses(
    x, x_recon, z_e, z_q,
    codebook,                        # (num_levels, num_codes, latent_dim)
    z_q1,                            # level 1 quantized output
    lambda_commit=0.25,
    lambda_codebook=1.0,
    lambda_wasserstein=0.0,          # NEW: 0.0 = off, >0 = enable
    sinkhorn_epsilon=0.05,           # NEW: transport softness
) -> dict[str, jnp.ndarray]:
```

Total loss:
```
loss = recon + λ_commit * commit + λ_codebook * codebook + λ_wasserstein * wasserstein
```

Wasserstein term computed for both RQ levels:
- Level 1: transport between `z_e` and `codebook[0]`
- Level 2: transport between `z_e - z_q1` (residuals) and `codebook[1]`

### Sinkhorn Implementation

```python
def sinkhorn_loss(
    points: jnp.ndarray,    # (batch, latent_dim)
    codebook: jnp.ndarray,  # (num_codes, latent_dim)
    epsilon: float = 0.05,
    num_iters: int = 20,
) -> jnp.ndarray:
```

Algorithm:
1. Compute cost matrix: `C[i,j] = ||points[i] - codebook[j]||²`
2. Initialize uniform marginals: `a = 1/batch`, `b = 1/num_codes`
3. Sinkhorn iterations: alternating normalization on `K = exp(-C / epsilon)`
4. Return transport cost: `sum(transport_plan * C)`

The uniform target marginal creates pressure to use all codes equally.

Fixed iterations: 20 (sufficient for convergence, not exposed in UI).

### Training Integration

**TrainerConfig:**
```python
@dataclass
class TrainerConfig:
    learning_rate: float = 1e-3
    lambda_commit: float = 0.25
    lambda_codebook: float = 1.0
    lambda_wasserstein: float = 0.0   # NEW
    sinkhorn_epsilon: float = 0.05     # NEW
```

**Training step:** Pass `codebook` and `z_q1` from quantizer aux dict to `compute_losses()`.

**TrainingState:** Add `wasserstein` to loss history tracking.

### UI Changes

**New sliders:**
- `λ Wasserstein`: range 0.0-1.0, default 0.0, step 0.01
- `Sinkhorn ε`: range 0.01-0.2, default 0.05, step 0.01

**Loss plot:** Add wasserstein as fourth line (only when λ_wasserstein > 0).

## Configurability

| Setting | Traditional VQ-VAE | Pure Wasserstein | Hybrid |
|---------|-------------------|------------------|--------|
| λ_codebook | 1.0 | 0.0 | 1.0 |
| λ_wasserstein | 0.0 | 1.0 | 0.5 |

## Files to Modify

1. `src/rq_vae_explorer/training/losses.py` - Add `sinkhorn_loss()`, update `compute_losses()`
2. `src/rq_vae_explorer/training/trainer.py` - Add config params, pass to loss
3. `src/rq_vae_explorer/training/state.py` - Track wasserstein in history
4. `src/rq_vae_explorer/ui/controls.py` - Add two sliders
5. `src/rq_vae_explorer/ui/plotting.py` - Add wasserstein to loss plot

## Future Extensions

- Per-level Wasserstein weights (`λ_wasserstein_l1`, `λ_wasserstein_l2`) if needed for fine-grained control
