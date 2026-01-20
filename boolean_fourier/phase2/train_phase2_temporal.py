"""
Phase 2 Training: Temporal Layer with Hierarchical R Matrix
============================================================

Composes frozen Phase 1 logic masks into temporal masks via R matrix.
Uses pure nonnegative P (column-stochastic, row-budgeted) per mHC design.

Architecture:
- Frozen logic layer: 4 ternary masks [4, 4] from Phase 1
- Learnable R matrix: [4, 16] with Sinkhorn projection
- Temporal masks: composed as V_temporal = R.T @ V_logic

Key decisions:
- Pure nonnegative P (not S ⊙ P) - simpler, sufficient for hierarchy
- Temperature τ ≥ 0.05 for numerical stability
- k=2 sparsification target for NPU hardening
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from functools import partial
from typing import Dict, Tuple, List
import json
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import numpy as np

from temporal_dataset import (
    create_temporal_train_test_split,
    TEMPORAL_OP_NAMES,
)
from hierarchical_r import sinkhorn_rectangular, validate_rectangular_sinkhorn


# Configuration
SEQ_LEN = 8
N_TRAIN = 10000
N_TEST = 2000
BATCH_SIZE = 128
N_EPOCHS = 100
LEARNING_RATE = 0.05
THRESHOLD = 0.3

# Hierarchical parameters
N_PARENT = 4   # Logic operations
N_CHILD = 16   # Temporal operations
FEATURE_DIM = 8  # Extended Boolean Fourier basis

CHECKPOINT_DIR = Path("v5/checkpoints/phase2_temporal")


# ============================================================================
# Phase 1 Logic Masks (frozen)
# ============================================================================

# Expected masks from Phase 1 validation (canonical form)
PHASE1_LOGIC_MASKS = {
    'xor': jnp.array([0., 0., 0., 1.]),       # Pure parity
    'and': jnp.array([1., 1., 1., -1.]),      # sign(1 + a + b - ab)
    'or': jnp.array([-1., 1., 1., 1.]),       # sign(-1 + a + b + ab)
    'implies': jnp.array([-1., -1., 1., -1.]) # sign(-1 - a + b - ab)
}


def load_phase1_masks() -> jnp.ndarray:
    """
    Load frozen Phase 1 logic masks.
    Returns [4, 4] matrix where each row is a logic operation mask.
    """
    masks = jnp.stack([
        PHASE1_LOGIC_MASKS['xor'],
        PHASE1_LOGIC_MASKS['and'],
        PHASE1_LOGIC_MASKS['or'],
        PHASE1_LOGIC_MASKS['implies'],
    ], axis=0)  # [4, 4]
    return masks


# ============================================================================
# Extended Temporal Features
# ============================================================================

def temporal_fourier_features(a_seq: jnp.ndarray, b_seq: jnp.ndarray) -> jnp.ndarray:
    """
    Construct extended Boolean Fourier features for temporal processing.

    For each position i, compute features based on current and previous values:
    [1, a[i], a[i-1], a[i]*a[i-1], b[i], b[i-1], b[i]*b[i-1], a[i]*b[i]]

    This 8-dim basis captures both spatial (a,b interaction) and temporal (prev/curr) patterns.

    Args:
        a_seq: [batch, seq_len] in {-1, +1}
        b_seq: [batch, seq_len] in {-1, +1}

    Returns:
        features: [batch, seq_len, 8]
    """
    batch_size, seq_len = a_seq.shape

    # Shift sequences to get previous values (pad with 0 = neutral)
    a_prev = jnp.concatenate([jnp.zeros((batch_size, 1)), a_seq[:, :-1]], axis=1)
    b_prev = jnp.concatenate([jnp.zeros((batch_size, 1)), b_seq[:, :-1]], axis=1)

    # Build 8-dim feature basis
    ones = jnp.ones_like(a_seq)
    features = jnp.stack([
        ones,           # 0: constant
        a_seq,          # 1: a current
        a_prev,         # 2: a previous
        a_seq * a_prev, # 3: a temporal parity (toggle detection)
        b_seq,          # 4: b current
        b_prev,         # 5: b previous
        b_seq * b_prev, # 6: b temporal parity
        a_seq * b_seq,  # 7: spatial parity (XOR)
    ], axis=-1)  # [batch, seq_len, 8]

    return features


# ============================================================================
# Hierarchical Temporal Layer
# ============================================================================

class HierarchicalTemporalLayer(nn.Module):
    """
    Temporal layer that composes logic masks via R matrix.

    V_temporal = R.T @ V_logic
    where R is [n_parent, n_child] doubly-stochastic (Sinkhorn projected)
    """
    n_parent: int = 4
    n_child: int = 16
    feature_dim: int = 8
    temperature: float = 0.1  # Sinkhorn temperature

    @nn.compact
    def __call__(
        self,
        a_seq: jnp.ndarray,
        b_seq: jnp.ndarray,
        operation_id: int,
        logic_masks: jnp.ndarray,
        temperature: float = None,
        training: bool = True
    ) -> jnp.ndarray:
        """
        Forward pass for temporal layer.

        Args:
            a_seq: [batch, seq_len] in {-1, +1}
            b_seq: [batch, seq_len] in {-1, +1}
            operation_id: int in {0, ..., 15}
            logic_masks: [4, 4] frozen logic masks from Phase 1
            temperature: Sinkhorn temperature (overrides default if provided)
            training: whether in training mode

        Returns:
            output: [batch, seq_len] in {-1, +1}
        """
        temp = temperature if temperature is not None else self.temperature

        # Initialize log-R matrix (will be projected via Sinkhorn)
        log_R = self.param(
            'log_R',
            nn.initializers.normal(0.5),
            (self.n_parent, self.n_child)
        )

        # Project to doubly-stochastic via Sinkhorn
        # Uses rectangular Sinkhorn: cols sum to 1, rows sum to n_child/n_parent
        R = sinkhorn_rectangular(log_R / temp, n_iters=10, temperature=1.0)

        # Compose temporal masks from logic masks
        # logic_masks: [4, 4] -> we need to expand to [4, 8] for temporal features
        # For simplicity, pad logic masks to 8-dim (first 4 are logic, rest are zero)
        logic_masks_expanded = jnp.pad(
            logic_masks,
            ((0, 0), (0, self.feature_dim - logic_masks.shape[1])),
            mode='constant',
            constant_values=0
        )  # [4, 8]

        # Compose: temporal_masks = R.T @ logic_masks_expanded
        # R: [4, 16], logic_expanded: [4, 8]
        # temporal_masks: [16, 8]
        temporal_masks = R.T @ logic_masks_expanded  # [16, 8]

        # Select mask for this operation
        mask = temporal_masks[operation_id]  # [8]

        # Compute temporal features
        features = temporal_fourier_features(a_seq, b_seq)  # [batch, seq_len, 8]

        # Apply mask
        masked = features * mask  # [batch, seq_len, 8]

        # Sum over feature dimensions
        output = jnp.sum(masked, axis=-1)  # [batch, seq_len]

        # Apply activation
        if training:
            output = jnp.tanh(output * 10.0)  # Soft sign for differentiable training
        else:
            output = jnp.sign(output)
            output = jnp.where(output == 0, 1.0, output)

        return output

    def get_temporal_masks(self, logic_masks: jnp.ndarray, params: dict) -> jnp.ndarray:
        """Get composed temporal masks from current R matrix."""
        log_R = params['log_R']
        R = sinkhorn_rectangular(log_R / self.temperature, n_iters=10)

        logic_expanded = jnp.pad(
            logic_masks,
            ((0, 0), (0, self.feature_dim - logic_masks.shape[1]))
        )
        return R.T @ logic_expanded


# ============================================================================
# Training Functions
# ============================================================================

def compute_hamming_loss(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Hamming loss: fraction of mismatched bits."""
    return jnp.mean((1 - pred * target) / 2)


@partial(jax.jit, static_argnums=(5,))
def train_step(
    state: train_state.TrainState,
    a_seq: jnp.ndarray,
    b_seq: jnp.ndarray,
    target: jnp.ndarray,
    logic_masks: jnp.ndarray,
    op_id: int,
    temperature: float = 0.1
):
    """Single training step."""

    def loss_fn(params):
        pred = state.apply_fn(
            {'params': params},
            a_seq, b_seq, op_id, logic_masks,
            temperature=temperature, training=True
        )
        loss = compute_hamming_loss(pred, target)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def compute_accuracy(
    apply_fn,
    params: dict,
    a_seq: jnp.ndarray,
    b_seq: jnp.ndarray,
    target: jnp.ndarray,
    logic_masks: jnp.ndarray,
    op_id: int
) -> float:
    """Compute accuracy for a single operation."""
    pred = apply_fn(
        {'params': params},
        a_seq, b_seq, op_id, logic_masks,
        temperature=0.05, training=False
    )
    return float(jnp.mean(pred == target))


# ============================================================================
# Main Training Loop
# ============================================================================

def train_phase2():
    """Train Phase 2 temporal layer with hierarchical R matrix."""
    print("="*60)
    print("Phase 2: Hierarchical Temporal Layer Training")
    print("="*60)

    # Load frozen Phase 1 masks
    print("\nLoading Phase 1 logic masks...")
    logic_masks = load_phase1_masks()
    print(f"  Logic masks shape: {logic_masks.shape}")
    for i, name in enumerate(['XOR', 'AND', 'OR', 'IMPLIES']):
        print(f"    {name}: {logic_masks[i]}")

    # Generate temporal dataset
    print("\nGenerating temporal dataset...")
    train_data, test_data = create_temporal_train_test_split(N_TRAIN, N_TEST, SEQ_LEN)
    print(f"  Train samples: {N_TRAIN}")
    print(f"  Test samples: {N_TEST}")
    print(f"  Sequence length: {SEQ_LEN}")
    print(f"  Temporal operations: {len(TEMPORAL_OP_NAMES)}")

    # Initialize model
    print("\nInitializing model...")
    model = HierarchicalTemporalLayer(
        n_parent=N_PARENT,
        n_child=N_CHILD,
        feature_dim=FEATURE_DIM,
        temperature=0.1
    )

    rng = jax.random.PRNGKey(42)
    dummy_a = jnp.ones((1, SEQ_LEN))
    dummy_b = jnp.ones((1, SEQ_LEN))
    variables = model.init(rng, dummy_a, dummy_b, 0, logic_masks)

    print(f"  Parameters: {jax.tree_util.tree_map(lambda x: x.shape, variables['params'])}")

    # Show initial R matrix
    log_R = variables['params']['log_R']
    R_init = sinkhorn_rectangular(log_R / 0.1, n_iters=10)
    print(f"\nInitial R matrix stats:")
    print(f"  Shape: {R_init.shape}")
    print(f"  Column sums: {R_init.sum(axis=0)[:4]}... (should be ~1.0)")
    print(f"  Row sums: {R_init.sum(axis=1)} (should be ~{N_CHILD/N_PARENT})")

    # Create optimizer
    optimizer = optax.adam(LEARNING_RATE)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optimizer
    )

    # Training loop
    print("\n" + "="*60)
    print("Training")
    print("="*60)

    best_mean_acc = 0.0
    best_params = None

    for epoch in range(N_EPOCHS):
        # Temperature annealing (floor at 0.05 for stability)
        progress = epoch / (N_EPOCHS - 1)
        temperature = 0.3 * (0.05 / 0.3) ** progress
        temperature = max(temperature, 0.05)

        epoch_loss = 0.0
        n_batches = 0

        # Train on each temporal operation
        for op_id, op_name in TEMPORAL_OP_NAMES.items():
            a_seq, b_seq, target, _ = train_data[op_name]

            for start in range(0, len(a_seq), BATCH_SIZE):
                end = min(start + BATCH_SIZE, len(a_seq))
                a_batch = a_seq[start:end]
                b_batch = b_seq[start:end]
                t_batch = target[start:end]

                state, loss = train_step(
                    state, a_batch, b_batch, t_batch,
                    logic_masks, op_id, temperature
                )
                epoch_loss += float(loss)
                n_batches += 1

        avg_loss = epoch_loss / n_batches

        # Validation every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            accuracies = {}
            for op_id, op_name in TEMPORAL_OP_NAMES.items():
                a_seq, b_seq, target, _ = test_data[op_name]
                acc = compute_accuracy(
                    model.apply, state.params,
                    a_seq, b_seq, target, logic_masks, op_id
                )
                accuracies[op_name] = acc

            mean_acc = np.mean(list(accuracies.values()))

            print(f"\nEpoch {epoch+1}/{N_EPOCHS}")
            print(f"  Loss: {avg_loss:.4f} | Mean Acc: {mean_acc:.2%} | T: {temperature:.3f}")

            # Show worst 3 operations
            sorted_accs = sorted(accuracies.items(), key=lambda x: x[1])
            print(f"  Worst 3:")
            for name, acc in sorted_accs[:3]:
                print(f"    {name}: {acc:.2%}")

            if mean_acc > best_mean_acc:
                best_mean_acc = mean_acc
                best_params = state.params

            # Validate R matrix
            log_R = state.params['log_R']
            R = sinkhorn_rectangular(log_R / temperature, n_iters=10)
            validation = validate_rectangular_sinkhorn(R, N_PARENT, N_CHILD)
            print(f"  R matrix: col_err={validation['col_stochastic_error']:.4f}, row_err={validation['row_budget_error']:.4f}")

    # Final results
    print("\n" + "="*60)
    print("Training Complete")
    print("="*60)

    print("\nFinal Accuracies:")
    all_pass = True
    for op_id, op_name in TEMPORAL_OP_NAMES.items():
        a_seq, b_seq, target, _ = test_data[op_name]
        acc = compute_accuracy(
            model.apply, state.params,
            a_seq, b_seq, target, logic_masks, op_id
        )
        status = "OK" if acc > 0.80 else "LOW"
        if acc < 0.80:
            all_pass = False
        print(f"  [{status}] {op_name:15s}: {acc:.2%}")

    mean_final = np.mean([
        compute_accuracy(
            model.apply, state.params,
            test_data[op][0], test_data[op][1], test_data[op][2],
            logic_masks, op_id
        )
        for op_id, op in TEMPORAL_OP_NAMES.items()
    ])

    print(f"\nMean Accuracy: {mean_final:.2%}")
    print(f"Best Mean Accuracy: {best_mean_acc:.2%}")

    # Analyze learned R matrix
    print("\n" + "="*60)
    print("R Matrix Analysis")
    print("="*60)

    log_R = state.params['log_R']
    R_final = sinkhorn_rectangular(log_R / 0.05, n_iters=20)

    print(f"\nFinal R matrix [4, 16]:")
    print(f"  Column sums: {R_final.sum(axis=0)}")
    print(f"  Row sums: {R_final.sum(axis=1)}")

    # Show which logic operations contribute most to each temporal operation
    print("\nDominant logic parent for each temporal operation:")
    for op_id, op_name in TEMPORAL_OP_NAMES.items():
        col = R_final[:, op_id]
        dominant = int(np.argmax(col))
        logic_names = ['XOR', 'AND', 'OR', 'IMP']
        weight = float(col[dominant])
        print(f"  {op_name:15s} <- {logic_names[dominant]} ({weight:.2f})")

    # Save checkpoint
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / "phase2_params.npy"
    np.save(checkpoint_path, {
        'log_R': np.array(state.params['log_R']),
        'R_final': np.array(R_final),
        'accuracies': {op: float(compute_accuracy(
            model.apply, state.params,
            test_data[op][0], test_data[op][1], test_data[op][2],
            logic_masks, oid
        )) for oid, op in TEMPORAL_OP_NAMES.items()},
        'mean_accuracy': float(mean_final),
    })
    print(f"\nCheckpoint saved to {checkpoint_path}")

    return state, mean_final


if __name__ == "__main__":
    train_phase2()
