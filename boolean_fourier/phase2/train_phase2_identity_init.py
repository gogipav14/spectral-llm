"""
Phase 2: Identity-Initialized Routing
======================================

Initialize log_alpha to encourage identity-like routing:
- Child 0-3 route to parents 0-3 (diagonal)
- Child 4-7 route to parents 0-3 (repeated diagonal)

This tests if the optimization can MAINTAIN correct routing while learning signs.
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
N_EPOCHS = 100
LEARNING_RATE = 0.05  # Lower LR to maintain routing

# Only train on linear operations (0-7)
LINEAR_OPS = {i: TEMPORAL_V2_OP_NAMES[i] for i in range(8)}
N_LINEAR_OPS = 8

N_PARENT = 4
N_CHILD = N_LINEAR_OPS
SINKHORN_ITERS = 20

CHECKPOINT_DIR = Path("v5/checkpoints/phase2_identity_init")


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


def identity_like_init(key, shape, dtype=jnp.float32):
    """
    Initialize log_alpha to encourage identity-like routing.

    For [4, 8] matrix:
    - High values on diagonal for children 0-3
    - High values on repeated diagonal for children 4-7
    """
    n_parent, n_child = shape
    # Start with small random values
    log_alpha = jax.random.normal(key, shape, dtype) * 0.1

    # Add large bias on the "identity" pattern
    identity_bias = 5.0  # Large positive value to encourage routing
    for i in range(n_parent):
        log_alpha = log_alpha.at[i, i].add(identity_bias)  # Children 0-3
        if i + n_parent < n_child:
            log_alpha = log_alpha.at[i, i + n_parent].add(identity_bias)  # Children 4-7

    return log_alpha


class IdentityInitTemporalLayer(nn.Module):
    """
    Temporal layer with identity-initialized routing.
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
        # Identity-initialized log_alpha
        log_alpha = self.param(
            'log_alpha',
            identity_like_init,
            (self.n_parent, self.n_child)
        )

        # Get P via Sinkhorn
        P = sinkhorn_rectangular(
            log_alpha,
            n_iters=self.sinkhorn_iters,
            temperature=temperature
        )

        # Learnable signs (initialized to expected pattern)
        # Expected: [+1, +1, +1, +1, -1, -1, -1, -1]
        sign_init = jnp.array([1., 1., 1., 1., -1., -1., -1., -1.])
        sign_logits = self.param(
            'sign_logits',
            lambda key, shape: sign_init * 2.0,  # Start near expected
            (self.n_child,)
        )

        if training:
            s = jnp.tanh(sign_logits / temperature)
        else:
            s = jnp.sign(sign_logits)
            s = jnp.where(s == 0, 1.0, s)

        # R = P * s[None, :]
        R = P * s[None, :]  # [4, 8]

        # Compose temporal masks
        temporal_masks = R.T @ logic_masks  # [8, 4]

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


def train_identity_init():
    """Train Phase 2 with identity-initialized routing."""
    print("=" * 60)
    print("Phase 2: Identity-Initialized Routing")
    print("=" * 60)

    # Load logic masks
    print("\nLoading Phase 1 logic masks...")
    logic_masks = load_phase1_masks()

    # Generate dataset
    print("\nGenerating dataset...")
    train_data, test_data = create_temporal_v2_train_test_split(N_TRAIN, N_TEST, N_BITS)
    print(f"  Train: {N_TRAIN}, Test: {N_TEST}")
    print(f"  Operations: {len(LINEAR_OPS)}")

    # Initialize model
    print("\nInitializing model (identity-initialized routing)...")
    model = IdentityInitTemporalLayer(
        n_parent=N_PARENT,
        n_child=N_LINEAR_OPS,
        sinkhorn_iters=SINKHORN_ITERS
    )

    rng = jax.random.PRNGKey(42)
    dummy_a = jnp.ones((1, N_BITS))
    dummy_b = jnp.ones((1, N_BITS))
    variables = model.init(rng, dummy_a, dummy_b, 0, logic_masks)

    print(f"  log_alpha shape: {variables['params']['log_alpha'].shape}")
    print(f"  sign_logits shape: {variables['params']['sign_logits'].shape}")

    # Show initial P
    log_alpha = variables['params']['log_alpha']
    P_init = sinkhorn_rectangular(log_alpha, n_iters=SINKHORN_ITERS, temperature=0.5)
    print(f"\nInitial P matrix (should be identity-like):")
    print(f"  P:\n{np.array(P_init).round(2)}")

    # Show initial signs
    sign_logits = variables['params']['sign_logits']
    s_init = jnp.tanh(sign_logits / 0.5)
    print(f"\nInitial signs: {[f'{x:+.2f}' for x in s_init]}")

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

            # Show signs
            sign_logits = state.params['sign_logits']
            s = jnp.tanh(sign_logits / temperature)
            print(f"  Signs (s): {[f'{x:+.2f}' for x in s]}")

            # Show routing (dominant parent per child)
            log_alpha = state.params['log_alpha']
            P = sinkhorn_rectangular(log_alpha, n_iters=SINKHORN_ITERS, temperature=temperature)
            routing = [int(np.argmax(P[:, i])) for i in range(N_LINEAR_OPS)]
            print(f"  Routing:  {routing}")
            print(f"  Expected: [0, 1, 2, 3, 0, 1, 2, 3]")

            if mean_acc > best_mean_acc:
                best_mean_acc = mean_acc

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

    # Final analysis
    print("\n" + "=" * 60)
    print("Final R Matrix Analysis")
    print("=" * 60)

    log_alpha = state.params['log_alpha']
    sign_logits = state.params['sign_logits']

    P_final = sinkhorn_rectangular(log_alpha, n_iters=SINKHORN_ITERS, temperature=0.05)
    s_final = jnp.sign(sign_logits)
    s_final = jnp.where(s_final == 0, 1.0, s_final)

    print(f"\nFinal P matrix:")
    print(f"{np.array(P_final).round(2)}")

    print(f"\nFinal signs: {[int(x) for x in s_final]}")
    print(f"Expected:    [+1, +1, +1, +1, -1, -1, -1, -1]")

    # Check routing correctness
    print("\nRouting check:")
    logic_names = ['XOR', 'AND', 'OR', 'IMP']
    expected_routing = [0, 1, 2, 3, 0, 1, 2, 3]
    expected_signs = [1, 1, 1, 1, -1, -1, -1, -1]

    n_routing_correct = 0
    n_sign_correct = 0
    for op_id, op_name in LINEAR_OPS.items():
        dominant = int(np.argmax(P_final[:, op_id]))
        sign = int(s_final[op_id])
        exp_route = expected_routing[op_id]
        exp_sign = expected_signs[op_id]
        route_ok = dominant == exp_route
        sign_ok = sign == exp_sign
        if route_ok:
            n_routing_correct += 1
        if sign_ok:
            n_sign_correct += 1
        status = "OK" if (route_ok and sign_ok) else "WRONG"
        print(f"  {op_name:15s}: route={logic_names[dominant]}, s={sign:+d} [{status}]")

    print(f"\nCorrect routing: {n_routing_correct}/{N_LINEAR_OPS}")
    print(f"Correct signs: {n_sign_correct}/{N_LINEAR_OPS}")

    if all_perfect:
        print("\n" + "=" * 60)
        print("SUCCESS: All operations learned!")
        print("=" * 60)

    # Save checkpoint
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / "identity_init_params.npy"
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
    train_identity_init()
