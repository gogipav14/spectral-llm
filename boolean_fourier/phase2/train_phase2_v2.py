"""
Phase 2 Training V2: Logic Composition via Hierarchical R
==========================================================

Uses the SAME 4-dim Boolean Fourier basis as Phase 1.
Each temporal operation is a weighted combination of the 4 frozen logic masks.

Architecture:
    temporal_mask[i] = sum_j R[j,i] * logic_mask[j]

Where:
    - logic_mask: [4, 4] frozen from Phase 1
    - R: [4, 16] = P * s[None, :] (column-sign Sinkhorn)
    - temporal_mask: [16, 4] composed masks

Key insight: Pure nonnegative P can only do convex combinations, NOT negations.
Solution: Use column-sign R = P * s where s ∈ {-1, +1}^n_child.
This allows XNOR = -XOR, NAND = -AND, etc.
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from functools import partial
from typing import Dict, Tuple
import json
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import numpy as np

from temporal_dataset_v2 import (
    create_temporal_v2_train_test_split,
    TEMPORAL_V2_OP_NAMES,
)
from hierarchical_r import (
    ColumnSignRMatrix,
    sinkhorn_rectangular,
    validate_rectangular_sinkhorn
)


# Configuration
N_BITS = 64
N_TRAIN = 2000  # Reduced for faster iteration
N_TEST = 500
BATCH_SIZE = 128
N_EPOCHS = 200  # More epochs for plateau detection
LEARNING_RATE = 0.05  # Lower LR for stability
LEARNING_RATE_RESTART = 0.1  # Higher LR for restarts

# Hierarchical parameters
N_PARENT = 4   # Logic operations
N_CHILD = 16   # Temporal operations
FEATURE_DIM = 4  # Same as Phase 1!
SINKHORN_ITERS = 20

# Plateau detection (checks every 10 epochs, so window of 5 = 50 epochs)
PLATEAU_WINDOW = 5
PLATEAU_THRESHOLD = 0.02  # If accuracy doesn't improve by this much
MICRO_RESTART_STD = 0.5  # Larger perturbation for more exploration

CHECKPOINT_DIR = Path("v5/checkpoints/phase2_v2")


# ============================================================================
# Phase 1 Logic Masks (frozen)
# ============================================================================

def load_phase1_masks() -> jnp.ndarray:
    """Load frozen Phase 1 logic masks."""
    masks = jnp.array([
        [0., 0., 0., 1.],      # XOR: pure parity
        [1., 1., 1., -1.],     # AND: sign(1 + a + b - ab)
        [-1., 1., 1., 1.],     # OR: sign(-1 + a + b + ab)
        [-1., -1., 1., -1.],   # IMPLIES: sign(-1 - a + b - ab)
    ], dtype=jnp.float32)  # [4, 4]
    return masks


# ============================================================================
# Boolean Fourier Features (same as Phase 1)
# ============================================================================

def boolean_fourier_features(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """
    Boolean Fourier basis: [1, a, b, ab] per bit.

    Args:
        a, b: [batch, n_bits] in {-1, +1}

    Returns:
        features: [batch, n_bits, 4]
    """
    ones = jnp.ones_like(a)
    ab = a * b
    return jnp.stack([ones, a, b, ab], axis=-1)


# ============================================================================
# Hierarchical Temporal Layer
# ============================================================================

class HierarchicalTemporalLayerV2(nn.Module):
    """
    Temporal layer that composes logic masks via R matrix with column-sign.

    Uses the same 4-dim Boolean Fourier basis as Phase 1.
    Each temporal operation is a weighted combination of logic masks.

    Key: Uses R = P * s[None, :] where:
    - P is column-stochastic, row-budgeted via Sinkhorn
    - s is column-sign vector in {-1, +1}^n_child

    This enables negations (XNOR = -XOR, NAND = -AND, etc.)
    """
    n_parent: int = 4
    n_child: int = 16
    sinkhorn_iters: int = 20
    use_signs: bool = True

    @nn.compact
    def __call__(
        self,
        a: jnp.ndarray,
        b: jnp.ndarray,
        operation_id: int,
        logic_masks: jnp.ndarray,
        temperature: float = 0.1,
        training: bool = True
    ) -> jnp.ndarray:
        """
        Forward pass.

        Args:
            a, b: [batch, n_bits] in {-1, +1}
            operation_id: int in {0, ..., 15}
            logic_masks: [4, 4] frozen logic masks
            temperature: Sinkhorn temperature
            training: whether in training mode

        Returns:
            output: [batch, n_bits] in {-1, +1}
        """
        # Use ColumnSignRMatrix for proper mHC-style composition
        R_module = ColumnSignRMatrix(
            n_parent=self.n_parent,
            n_child=self.n_child,
            sinkhorn_iters=self.sinkhorn_iters,
            temperature=temperature,
            use_signs=self.use_signs,
            name='R_matrix'
        )
        R = R_module(training=training)  # [4, 16]

        # Compose temporal masks from logic masks
        # R: [4, 16], logic_masks: [4, 4]
        # temporal_masks = R.T @ logic_masks = [16, 4]
        temporal_masks = R.T @ logic_masks  # [16, 4]

        # Select mask for this operation
        mask = temporal_masks[operation_id]  # [4]

        # Compute Boolean Fourier features (same as Phase 1)
        features = boolean_fourier_features(a, b)  # [batch, n_bits, 4]

        # Apply mask
        masked = features * mask  # [batch, n_bits, 4]

        # Sum over features
        output = jnp.sum(masked, axis=-1)  # [batch, n_bits]

        # Apply activation
        if training:
            output = jnp.tanh(output * 10.0)
        else:
            output = jnp.sign(output)
            output = jnp.where(output == 0, 1.0, output)

        return output


# ============================================================================
# Training Functions
# ============================================================================

def compute_hamming_loss(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Hamming loss."""
    return jnp.mean((1 - pred * target) / 2)


@partial(jax.jit, static_argnums=(5,))
def train_step(
    state: train_state.TrainState,
    a: jnp.ndarray,
    b: jnp.ndarray,
    target: jnp.ndarray,
    logic_masks: jnp.ndarray,
    op_id: int,
    temperature: float = 0.1
):
    """Single training step."""

    def loss_fn(params):
        pred = state.apply_fn(
            {'params': params},
            a, b, op_id, logic_masks,
            temperature=temperature, training=True
        )
        return compute_hamming_loss(pred, target)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def compute_accuracy(
    apply_fn,
    params: dict,
    a: jnp.ndarray,
    b: jnp.ndarray,
    target: jnp.ndarray,
    logic_masks: jnp.ndarray,
    op_id: int
) -> float:
    """Compute accuracy."""
    pred = apply_fn(
        {'params': params},
        a, b, op_id, logic_masks,
        temperature=0.05, training=False
    )
    return float(jnp.mean(pred == target))


# ============================================================================
# Main Training Loop
# ============================================================================

def train_phase2_v2():
    """Train Phase 2 V2."""
    print("="*60)
    print("Phase 2 V2: Logic Composition via Hierarchical R")
    print("="*60)

    # Load frozen Phase 1 masks
    print("\nLoading Phase 1 logic masks...")
    logic_masks = load_phase1_masks()
    print(f"  Shape: {logic_masks.shape}")
    for i, name in enumerate(['XOR', 'AND', 'OR', 'IMPLIES']):
        print(f"    {name}: {logic_masks[i]}")

    # Generate dataset
    print("\nGenerating temporal V2 dataset...")
    train_data, test_data = create_temporal_v2_train_test_split(N_TRAIN, N_TEST, N_BITS)
    print(f"  Train: {N_TRAIN}, Test: {N_TEST}, Bits: {N_BITS}")
    print(f"  Operations: {len(TEMPORAL_V2_OP_NAMES)}")

    # Initialize model
    print("\nInitializing model...")
    model = HierarchicalTemporalLayerV2(
        n_parent=N_PARENT,
        n_child=N_CHILD,
        sinkhorn_iters=SINKHORN_ITERS,
        use_signs=True  # CRITICAL: Enable column-sign for negations
    )

    rng = jax.random.PRNGKey(42)
    dummy_a = jnp.ones((1, N_BITS))
    dummy_b = jnp.ones((1, N_BITS))
    variables = model.init(rng, dummy_a, dummy_b, 0, logic_masks)

    # Show parameter structure
    param_shapes = jax.tree_util.tree_map(lambda x: x.shape, variables['params'])
    print(f"  Parameters: {param_shapes}")

    # Show initial R matrix
    log_alpha = variables['params']['R_matrix']['log_alpha']
    sign_logits = variables['params']['R_matrix']['sign_logits']
    P_init = sinkhorn_rectangular(log_alpha, n_iters=SINKHORN_ITERS, temperature=0.1)
    s_init = jnp.tanh(sign_logits / 0.1)

    print(f"\nInitial R = P * s[None, :] (column-sign):")
    print(f"  P shape: {P_init.shape}")
    print(f"  s shape: {s_init.shape}")
    print(f"  P range: [{float(P_init.min()):.3f}, {float(P_init.max()):.3f}]")
    print(f"  s range: [{float(s_init.min()):.3f}, {float(s_init.max()):.3f}]")

    # Validate initial P stability
    P_stability = validate_rectangular_sinkhorn(P_init, N_PARENT, N_CHILD)
    print(f"  P stability: col_err={P_stability['col_stochastic_error']:.4f}, "
          f"row_err={P_stability['row_budget_error']:.4f}, "
          f"stable={P_stability['is_stable']}")

    # Create optimizer
    optimizer = optax.adam(LEARNING_RATE)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optimizer
    )

    # Training loop
    print("\n" + "="*60)
    print("Training (with plateau detection)")
    print("="*60)

    best_mean_acc = 0.0
    acc_history = []
    n_restarts = 0

    def micro_restart(state, rng):
        """Perturb parameters to escape local minimum."""
        new_params = {}
        for key, value in state.params['R_matrix'].items():
            noise = jax.random.normal(rng, value.shape) * MICRO_RESTART_STD
            new_params[key] = value + noise
            rng, _ = jax.random.split(rng)
        return state.replace(params={'R_matrix': new_params})

    restart_rng = jax.random.PRNGKey(123)

    for epoch in range(N_EPOCHS):
        # Temperature annealing (floor at 0.05)
        progress = epoch / (N_EPOCHS - 1)
        temperature = 0.5 * (0.05 / 0.5) ** progress
        temperature = max(temperature, 0.05)

        epoch_loss = 0.0
        n_batches = 0

        # Train on each operation
        for op_id, op_name in TEMPORAL_V2_OP_NAMES.items():
            a, b, target, _ = train_data[op_name]

            for start in range(0, len(a), BATCH_SIZE):
                end = min(start + BATCH_SIZE, len(a))
                a_batch = a[start:end]
                b_batch = b[start:end]
                t_batch = target[start:end]

                state, loss = train_step(
                    state, a_batch, b_batch, t_batch,
                    logic_masks, op_id, temperature
                )
                epoch_loss += float(loss)
                n_batches += 1

        avg_loss = epoch_loss / n_batches

        # Validation every 10 epochs (more frequent for plateau detection)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            accuracies = {}
            for op_id, op_name in TEMPORAL_V2_OP_NAMES.items():
                a, b, target, _ = test_data[op_name]
                acc = compute_accuracy(
                    model.apply, state.params,
                    a, b, target, logic_masks, op_id
                )
                accuracies[op_name] = acc

            mean_acc = np.mean(list(accuracies.values()))

            print(f"\nEpoch {epoch+1}/{N_EPOCHS}")
            print(f"  Loss: {avg_loss:.4f} | Mean Acc: {mean_acc:.2%} | T: {temperature:.3f}")

            # Show worst 3
            sorted_accs = sorted(accuracies.items(), key=lambda x: x[1])
            print(f"  Worst 3:")
            for name, acc in sorted_accs[:3]:
                print(f"    {name}: {acc:.2%}")

            # R matrix stats (P and s components)
            log_alpha = state.params['R_matrix']['log_alpha']
            sign_logits = state.params['R_matrix']['sign_logits']
            P = sinkhorn_rectangular(log_alpha, n_iters=SINKHORN_ITERS, temperature=temperature)
            s = jnp.tanh(sign_logits / temperature)
            R = P * s[None, :]
            print(f"  R range: [{float(R.min()):.2f}, {float(R.max()):.2f}]")
            print(f"  Signs (s): {[f'{x:.2f}' for x in s[:8]]}")

            if mean_acc > best_mean_acc:
                best_mean_acc = mean_acc

            # Plateau detection
            acc_history.append(mean_acc)
            if len(acc_history) >= PLATEAU_WINDOW:
                recent = acc_history[-PLATEAU_WINDOW:]
                improvement = max(recent) - min(recent)
                if improvement < PLATEAU_THRESHOLD and mean_acc < 0.95:
                    # Plateau detected - micro-restart
                    n_restarts += 1
                    print(f"  ** PLATEAU DETECTED (restart #{n_restarts}) **")
                    restart_rng, key = jax.random.split(restart_rng)
                    state = micro_restart(state, key)
                    acc_history = []  # Reset history
                    # Reset optimizer state for fresh gradients
                    optimizer = optax.adam(LEARNING_RATE_RESTART)
                    state = train_state.TrainState.create(
                        apply_fn=model.apply,
                        params=state.params,
                        tx=optimizer
                    )

    # Final results
    print("\n" + "="*60)
    print("Training Complete")
    print("="*60)

    print("\nFinal Accuracies:")
    for op_id, op_name in TEMPORAL_V2_OP_NAMES.items():
        a, b, target, _ = test_data[op_name]
        acc = compute_accuracy(
            model.apply, state.params,
            a, b, target, logic_masks, op_id
        )
        status = "OK" if acc > 0.95 else "LOW"
        print(f"  [{status}] {op_name:25s}: {acc:.2%}")

    mean_final = np.mean([
        compute_accuracy(
            model.apply, state.params,
            test_data[op][0], test_data[op][1], test_data[op][2],
            logic_masks, op_id
        )
        for op_id, op in TEMPORAL_V2_OP_NAMES.items()
    ])

    print(f"\nMean Accuracy: {mean_final:.2%}")
    print(f"Best Mean Accuracy: {best_mean_acc:.2%}")
    print(f"Total Restarts: {n_restarts}")

    # Analyze R matrix
    print("\n" + "="*60)
    print("R Matrix Analysis (Column-Sign)")
    print("="*60)

    log_alpha = state.params['R_matrix']['log_alpha']
    sign_logits = state.params['R_matrix']['sign_logits']

    # Get P (nonnegative routing) and s (column signs)
    P_final = sinkhorn_rectangular(log_alpha, n_iters=SINKHORN_ITERS, temperature=0.05)
    s_final = jnp.sign(sign_logits)
    s_final = jnp.where(s_final == 0, 1.0, s_final)
    R_final = P_final * s_final[None, :]

    print("\nP matrix stability:")
    P_stability = validate_rectangular_sinkhorn(P_final, N_PARENT, N_CHILD)
    print(f"  col_err={P_stability['col_stochastic_error']:.4f}")
    print(f"  row_err={P_stability['row_budget_error']:.4f}")
    print(f"  stable={P_stability['is_stable']}")

    print("\nColumn signs (s):")
    print(f"  First 8:  {[int(x) for x in s_final[:8]]}")
    print(f"  Last 8:   {[int(x) for x in s_final[8:]]}")

    print("\nExpected sign patterns:")
    print("  Op 0-3 (pure logic): s = +1 (positive)")
    print("  Op 4-7 (negated): s = -1 (negative)")
    print("  Op 8-15 (composed): depends on composition")

    # Check sign patterns for negated operations
    print("\nNegation sign check (ops 4-7):")
    for i, name in enumerate(['xnor', 'nand', 'nor', 'not_implies']):
        sign = int(s_final[4 + i])
        expected = -1
        status = "OK" if sign == expected else f"WRONG (got {sign})"
        print(f"  {name}: sign = {sign} [{status}]")

    print("\nLearned R matrix columns (routing weights):")
    logic_names = ['XOR', 'AND', 'OR', 'IMP']
    for op_id, op_name in list(TEMPORAL_V2_OP_NAMES.items())[:8]:
        col = R_final[:, op_id]
        s = int(s_final[op_id])
        print(f"  {op_name:25s}: s={s:+d}, P=[{P_final[0,op_id]:.2f}, {P_final[1,op_id]:.2f}, {P_final[2,op_id]:.2f}, {P_final[3,op_id]:.2f}]")

    # Check if pure logic operations route correctly
    print("\nPure logic routing check:")
    for i, name in enumerate(['xor', 'and', 'or', 'implies']):
        col = jnp.abs(R_final[:, i])
        dominant = int(np.argmax(col))
        expected = i
        match = "OK" if dominant == expected else "WRONG"
        print(f"  {name}: dominant parent = {logic_names[dominant]} [{match}]")

    # Save checkpoint
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / "phase2_v2_params.npy"
    np.save(checkpoint_path, {
        'log_alpha': np.array(log_alpha),
        'sign_logits': np.array(sign_logits),
        'P_final': np.array(P_final),
        's_final': np.array(s_final),
        'R_final': np.array(R_final),
        'mean_accuracy': float(mean_final),
    })
    print(f"\nCheckpoint saved to {checkpoint_path}")

    return state, mean_final


if __name__ == "__main__":
    train_phase2_v2()
