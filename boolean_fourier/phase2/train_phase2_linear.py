"""
Phase 2 Training (Linear Operations Only)
==========================================

Simplified version that only trains on operations 0-7:
- Pure logic (XOR, AND, OR, IMPLIES): s=+1, route to corresponding parent
- Negations (XNOR, NAND, NOR, NOT_IMPLIES): s=-1, route to corresponding parent

All 8 operations are linear combinations of the 4 logic masks.
This isolates the hierarchical composition problem from the nonlinear operations.
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import numpy as np

from temporal_dataset_v2 import create_temporal_v2_train_test_split, TEMPORAL_V2_OP_NAMES
from hierarchical_r import ColumnSignRMatrix, sinkhorn_rectangular, validate_rectangular_sinkhorn


# Configuration
N_BITS = 64
N_TRAIN = 2000
N_TEST = 500
BATCH_SIZE = 128
N_EPOCHS = 200
LEARNING_RATE = 0.1

# Only train on linear operations (0-7)
LINEAR_OPS = {i: TEMPORAL_V2_OP_NAMES[i] for i in range(8)}
N_LINEAR_OPS = 8

# Hierarchical parameters
N_PARENT = 4
N_CHILD = N_LINEAR_OPS  # Only 8 children now
SINKHORN_ITERS = 20

# Plateau detection
PLATEAU_WINDOW = 5
PLATEAU_THRESHOLD = 0.02
MICRO_RESTART_STD = 0.5

CHECKPOINT_DIR = Path("v5/checkpoints/phase2_linear")


def load_phase1_masks() -> jnp.ndarray:
    """Load frozen Phase 1 logic masks."""
    return jnp.array([
        [0., 0., 0., 1.],      # XOR
        [1., 1., 1., -1.],     # AND
        [-1., 1., 1., 1.],     # OR
        [-1., -1., 1., -1.],   # IMPLIES
    ], dtype=jnp.float32)


def boolean_fourier_features(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Boolean Fourier basis: [1, a, b, ab] per bit."""
    ones = jnp.ones_like(a)
    ab = a * b
    return jnp.stack([ones, a, b, ab], axis=-1)


class LinearTemporalLayer(nn.Module):
    """
    Temporal layer for 8 linear operations.

    Each operation is either:
    - s=+1: direct copy of one logic mask
    - s=-1: negation of one logic mask
    """
    n_parent: int = 4
    n_child: int = 8
    sinkhorn_iters: int = 20

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
        """Forward pass."""
        # Use ColumnSignRMatrix
        R_module = ColumnSignRMatrix(
            n_parent=self.n_parent,
            n_child=self.n_child,
            sinkhorn_iters=self.sinkhorn_iters,
            temperature=temperature,
            use_signs=True,
            name='R_matrix'
        )
        R = R_module(training=training)  # [4, 8]

        # Compose temporal masks
        # R: [4, 8], logic_masks: [4, 4]
        # temporal_masks = R.T @ logic_masks = [8, 4]
        temporal_masks = R.T @ logic_masks

        # Select mask for this operation
        mask = temporal_masks[operation_id]  # [4]

        # Boolean Fourier features
        features = boolean_fourier_features(a, b)  # [batch, n_bits, 4]

        # Apply mask
        masked = features * mask  # [batch, n_bits, 4]
        output = jnp.sum(masked, axis=-1)  # [batch, n_bits]

        # Activation
        if training:
            output = jnp.tanh(output * 10.0)
        else:
            output = jnp.sign(output)
            output = jnp.where(output == 0, 1.0, output)

        return output


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
    """Training step."""
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


def compute_accuracy(apply_fn, params, a, b, target, logic_masks, op_id) -> float:
    """Compute accuracy."""
    pred = apply_fn(
        {'params': params},
        a, b, op_id, logic_masks,
        temperature=0.05, training=False
    )
    return float(jnp.mean(pred == target))


def train_linear_phase2():
    """Train Phase 2 on linear operations only."""
    print("=" * 60)
    print("Phase 2 (Linear Operations Only)")
    print("=" * 60)

    # Load logic masks
    print("\nLoading Phase 1 logic masks...")
    logic_masks = load_phase1_masks()

    # Generate dataset
    print("\nGenerating dataset...")
    train_data, test_data = create_temporal_v2_train_test_split(N_TRAIN, N_TEST, N_BITS)
    print(f"  Train: {N_TRAIN}, Test: {N_TEST}, Bits: {N_BITS}")
    print(f"  Operations: {len(LINEAR_OPS)}")

    # Expected routing pattern
    print("\nExpected routing (identity routing with signs):")
    print("  Op 0 (xor): s=+1, route to parent 0 (XOR)")
    print("  Op 1 (and): s=+1, route to parent 1 (AND)")
    print("  Op 2 (or): s=+1, route to parent 2 (OR)")
    print("  Op 3 (implies): s=+1, route to parent 3 (IMPLIES)")
    print("  Op 4 (xnor): s=-1, route to parent 0 (XOR)")
    print("  Op 5 (nand): s=-1, route to parent 1 (AND)")
    print("  Op 6 (nor): s=-1, route to parent 2 (OR)")
    print("  Op 7 (not_implies): s=-1, route to parent 3 (IMPLIES)")

    # Initialize model
    print("\nInitializing model...")
    model = LinearTemporalLayer(
        n_parent=N_PARENT,
        n_child=N_LINEAR_OPS,
        sinkhorn_iters=SINKHORN_ITERS
    )

    rng = jax.random.PRNGKey(42)
    dummy_a = jnp.ones((1, N_BITS))
    dummy_b = jnp.ones((1, N_BITS))
    variables = model.init(rng, dummy_a, dummy_b, 0, logic_masks)

    param_shapes = jax.tree_util.tree_map(lambda x: x.shape, variables['params'])
    print(f"  Parameters: {param_shapes}")

    # Create optimizer
    optimizer = optax.adam(LEARNING_RATE)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optimizer
    )

    # Training loop
    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)

    best_mean_acc = 0.0
    acc_history = []
    n_restarts = 0
    restart_rng = jax.random.PRNGKey(123)

    def micro_restart(state, rng):
        """Perturb parameters."""
        new_params = {}
        for key, value in state.params['R_matrix'].items():
            noise = jax.random.normal(rng, value.shape) * MICRO_RESTART_STD
            new_params[key] = value + noise
            rng, _ = jax.random.split(rng)
        return state.replace(params={'R_matrix': new_params})

    for epoch in range(N_EPOCHS):
        # Temperature annealing
        progress = epoch / (N_EPOCHS - 1)
        temperature = 0.5 * (0.05 / 0.5) ** progress
        temperature = max(temperature, 0.05)

        epoch_loss = 0.0
        n_batches = 0

        # Train on each LINEAR operation
        for op_id, op_name in LINEAR_OPS.items():
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

        # Validation every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            accuracies = {}
            for op_id, op_name in LINEAR_OPS.items():
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

            # Show signs
            sign_logits = state.params['R_matrix']['sign_logits']
            s = jnp.tanh(sign_logits / temperature)
            print(f"  Signs (s): {[f'{x:.2f}' for x in s]}")

            if mean_acc > best_mean_acc:
                best_mean_acc = mean_acc

            # Plateau detection
            acc_history.append(mean_acc)
            if len(acc_history) >= PLATEAU_WINDOW:
                recent = acc_history[-PLATEAU_WINDOW:]
                improvement = max(recent) - min(recent)
                if improvement < PLATEAU_THRESHOLD and mean_acc < 0.95:
                    n_restarts += 1
                    print(f"  ** PLATEAU DETECTED (restart #{n_restarts}) **")
                    restart_rng, key = jax.random.split(restart_rng)
                    state = micro_restart(state, key)
                    acc_history = []
                    optimizer = optax.adam(LEARNING_RATE)
                    state = train_state.TrainState.create(
                        apply_fn=model.apply,
                        params=state.params,
                        tx=optimizer
                    )

    # Final results
    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)

    print("\nFinal Accuracies:")
    all_perfect = True
    for op_id, op_name in LINEAR_OPS.items():
        a, b, target, _ = test_data[op_name]
        acc = compute_accuracy(
            model.apply, state.params,
            a, b, target, logic_masks, op_id
        )
        status = "OK" if acc > 0.99 else "LOW"
        if acc < 0.99:
            all_perfect = False
        print(f"  [{status}] {op_name:15s}: {acc:.2%}")

    mean_final = np.mean([
        compute_accuracy(
            model.apply, state.params,
            test_data[op][0], test_data[op][1], test_data[op][2],
            logic_masks, op_id
        )
        for op_id, op in LINEAR_OPS.items()
    ])

    print(f"\nMean Accuracy: {mean_final:.2%}")
    print(f"Best Mean Accuracy: {best_mean_acc:.2%}")
    print(f"Total Restarts: {n_restarts}")

    # Analyze R matrix
    print("\n" + "=" * 60)
    print("R Matrix Analysis")
    print("=" * 60)

    log_alpha = state.params['R_matrix']['log_alpha']
    sign_logits = state.params['R_matrix']['sign_logits']

    P_final = sinkhorn_rectangular(log_alpha, n_iters=SINKHORN_ITERS, temperature=0.05)
    s_final = jnp.sign(sign_logits)
    s_final = jnp.where(s_final == 0, 1.0, s_final)
    R_final = P_final * s_final[None, :]

    print("\nP matrix stability:")
    P_stability = validate_rectangular_sinkhorn(P_final, N_PARENT, N_LINEAR_OPS)
    print(f"  col_err={P_stability['col_stochastic_error']:.4f}")
    print(f"  row_err={P_stability['row_budget_error']:.4f}")

    print("\nColumn signs (s):")
    print(f"  {[int(x) for x in s_final]}")
    print("  Expected: [+1, +1, +1, +1, -1, -1, -1, -1]")

    print("\nRouting (dominant parent per operation):")
    logic_names = ['XOR', 'AND', 'OR', 'IMP']
    for op_id, op_name in LINEAR_OPS.items():
        col = jnp.abs(P_final[:, op_id])
        dominant = int(np.argmax(col))
        expected = op_id % 4  # Op 0,4->0, 1,5->1, 2,6->2, 3,7->3
        sign = int(s_final[op_id])
        match = "OK" if dominant == expected else "WRONG"
        print(f"  {op_name:15s}: parent={logic_names[dominant]}, s={sign:+d} [{match}]")

    if all_perfect:
        print("\n" + "=" * 60)
        print("SUCCESS: All linear operations learned!")
        print("=" * 60)

    # Save checkpoint
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / "phase2_linear_params.npy"
    np.save(checkpoint_path, {
        'log_alpha': np.array(log_alpha),
        'sign_logits': np.array(sign_logits),
        'P_final': np.array(P_final),
        's_final': np.array(s_final),
        'mean_accuracy': float(mean_final),
    })
    print(f"\nCheckpoint saved to {checkpoint_path}")

    return state, mean_final


if __name__ == "__main__":
    train_linear_phase2()
