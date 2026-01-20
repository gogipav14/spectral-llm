"""
Phase 2: All 16 Operations with Identity-like Initialization
=============================================================

Extends identity initialization to all 16 operations.

For linear operations (0-7): identity routing with known signs
For nonlinear operations (8-15): need to learn new routing patterns
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
from hierarchical_r import sinkhorn_rectangular, validate_rectangular_sinkhorn


# Configuration
N_BITS = 64
N_TRAIN = 2000
N_TEST = 500
BATCH_SIZE = 128
N_EPOCHS = 200
LEARNING_RATE = 0.05

N_PARENT = 4
N_CHILD = 16
SINKHORN_ITERS = 20

CHECKPOINT_DIR = Path("v5/checkpoints/phase2_all16")


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


def identity_like_init_16(key, shape, dtype=jnp.float32):
    """
    Initialize log_alpha for 16 children.

    First 8 children: identity-like (each child routes to one parent)
    Last 8 children: start uniform (will need to learn)
    """
    n_parent, n_child = shape  # [4, 16]

    # Start with small random values
    log_alpha = jax.random.normal(key, shape, dtype) * 0.1

    # Add large bias for first 8 children (identity pattern)
    identity_bias = 5.0
    for i in range(n_parent):
        log_alpha = log_alpha.at[i, i].add(identity_bias)  # Children 0-3
        log_alpha = log_alpha.at[i, i + n_parent].add(identity_bias)  # Children 4-7

    # Children 8-15: leave uniform (will learn)
    return log_alpha


def sign_init_16(key, shape):
    """
    Initialize signs for 16 children.

    First 4: +1 (pure logic)
    Next 4: -1 (negations)
    Last 8: 0 (will learn)
    """
    signs = jnp.array([
        2.0, 2.0, 2.0, 2.0,    # ops 0-3: +1
        -2.0, -2.0, -2.0, -2.0, # ops 4-7: -1
        0.0, 0.0, 0.0, 0.0,    # ops 8-11: unknown
        0.0, 0.0, 0.0, 0.0,    # ops 12-15: unknown
    ])
    return signs


class All16TemporalLayer(nn.Module):
    """Temporal layer for all 16 operations."""
    n_parent: int = 4
    n_child: int = 16
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
        # Identity-like initialized log_alpha
        log_alpha = self.param(
            'log_alpha',
            identity_like_init_16,
            (self.n_parent, self.n_child)
        )

        # Get P via Sinkhorn
        P = sinkhorn_rectangular(
            log_alpha,
            n_iters=self.sinkhorn_iters,
            temperature=temperature
        )

        # Learnable signs
        sign_logits = self.param(
            'sign_logits',
            sign_init_16,
            (self.n_child,)
        )

        if training:
            s = jnp.tanh(sign_logits / temperature)
        else:
            s = jnp.sign(sign_logits)
            s = jnp.where(s == 0, 1.0, s)

        # R = P * s[None, :]
        R = P * s[None, :]

        # Compose temporal masks
        temporal_masks = R.T @ logic_masks  # [16, 4]

        # Select mask for this operation
        mask = temporal_masks[operation_id]

        # Boolean Fourier features
        features = boolean_fourier_features(a, b)

        # Apply mask
        masked = features * mask
        output = jnp.sum(masked, axis=-1)

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


def train_all16():
    """Train Phase 2 with all 16 operations."""
    print("=" * 60)
    print("Phase 2: All 16 Operations (Identity-like Init)")
    print("=" * 60)

    # Load logic masks
    print("\nLoading Phase 1 logic masks...")
    logic_masks = load_phase1_masks()

    # Generate dataset
    print("\nGenerating dataset...")
    train_data, test_data = create_temporal_v2_train_test_split(N_TRAIN, N_TEST, N_BITS)
    print(f"  Train: {N_TRAIN}, Test: {N_TEST}")
    print(f"  Operations: {len(TEMPORAL_V2_OP_NAMES)}")

    # Initialize model
    print("\nInitializing model...")
    model = All16TemporalLayer(
        n_parent=N_PARENT,
        n_child=N_CHILD,
        sinkhorn_iters=SINKHORN_ITERS
    )

    rng = jax.random.PRNGKey(42)
    dummy_a = jnp.ones((1, N_BITS))
    dummy_b = jnp.ones((1, N_BITS))
    variables = model.init(rng, dummy_a, dummy_b, 0, logic_masks)

    print(f"  log_alpha shape: {variables['params']['log_alpha'].shape}")
    print(f"  sign_logits shape: {variables['params']['sign_logits'].shape}")

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

    for epoch in range(N_EPOCHS):
        # Temperature annealing
        progress = epoch / (N_EPOCHS - 1)
        temperature = 0.5 * (0.05 / 0.5) ** progress
        temperature = max(temperature, 0.05)

        epoch_loss = 0.0
        n_batches = 0

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

        # Validation every 20 epochs
        if (epoch + 1) % 20 == 0 or epoch == 0:
            accuracies = {}
            for op_id, op_name in TEMPORAL_V2_OP_NAMES.items():
                a, b, target, _ = test_data[op_name]
                acc = compute_accuracy(
                    model.apply, state.params,
                    a, b, target, logic_masks, op_id
                )
                accuracies[op_name] = acc

            mean_acc = np.mean(list(accuracies.values()))

            # Categorize by operation type
            linear_acc = np.mean([accuracies[TEMPORAL_V2_OP_NAMES[i]] for i in range(8)])
            nonlinear_acc = np.mean([accuracies[TEMPORAL_V2_OP_NAMES[i]] for i in range(8, 16)])

            print(f"\nEpoch {epoch+1}/{N_EPOCHS}")
            print(f"  Loss: {avg_loss:.4f} | Mean: {mean_acc:.2%}")
            print(f"  Linear (0-7): {linear_acc:.2%} | Nonlinear (8-15): {nonlinear_acc:.2%}")

            # Show worst 3
            sorted_accs = sorted(accuracies.items(), key=lambda x: x[1])
            print(f"  Worst 3:")
            for name, acc in sorted_accs[:3]:
                print(f"    {name}: {acc:.2%}")

            if mean_acc > best_mean_acc:
                best_mean_acc = mean_acc

    # Final results
    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)

    print("\nFinal Accuracies:")
    print("\nLinear operations (0-7):")
    for op_id in range(8):
        op_name = TEMPORAL_V2_OP_NAMES[op_id]
        a, b, target, _ = test_data[op_name]
        acc = compute_accuracy(
            model.apply, state.params,
            a, b, target, logic_masks, op_id
        )
        status = "OK" if acc > 0.99 else "LOW"
        print(f"  [{status}] {op_name:25s}: {acc:.2%}")

    print("\nNonlinear operations (8-15):")
    for op_id in range(8, 16):
        op_name = TEMPORAL_V2_OP_NAMES[op_id]
        a, b, target, _ = test_data[op_name]
        acc = compute_accuracy(
            model.apply, state.params,
            a, b, target, logic_masks, op_id
        )
        status = "OK" if acc > 0.99 else "LOW"
        print(f"  [{status}] {op_name:25s}: {acc:.2%}")

    mean_final = np.mean([
        compute_accuracy(
            model.apply, state.params,
            test_data[op][0], test_data[op][1], test_data[op][2],
            logic_masks, op_id
        )
        for op_id, op in TEMPORAL_V2_OP_NAMES.items()
    ])

    linear_final = np.mean([
        compute_accuracy(
            model.apply, state.params,
            test_data[TEMPORAL_V2_OP_NAMES[i]][0],
            test_data[TEMPORAL_V2_OP_NAMES[i]][1],
            test_data[TEMPORAL_V2_OP_NAMES[i]][2],
            logic_masks, i
        )
        for i in range(8)
    ])

    print(f"\nMean Accuracy (all 16): {mean_final:.2%}")
    print(f"Linear Accuracy (0-7): {linear_final:.2%}")
    print(f"Best Mean Accuracy: {best_mean_acc:.2%}")

    # Save checkpoint
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / "all16_params.npy"
    np.save(checkpoint_path, {
        'log_alpha': np.array(state.params['log_alpha']),
        'sign_logits': np.array(state.params['sign_logits']),
        'mean_accuracy': float(mean_final),
        'linear_accuracy': float(linear_final),
    })
    print(f"\nCheckpoint saved to {checkpoint_path}")

    return state, mean_final


if __name__ == "__main__":
    train_all16()
