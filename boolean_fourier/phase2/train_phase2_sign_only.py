"""
Phase 2 Diagnostic: Learn Signs Only (Fixed Routing)
=====================================================

Initialize P as identity-like routing (each child routes to one parent)
and only learn the column signs.

This tests if the sign learning works when routing is correct.

For 8 children and 4 parents:
- Children 0-3 route to parents 0-3 (pure logic)
- Children 4-7 route to parents 0-3 (negated logic)
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


# Configuration
N_BITS = 64
N_TRAIN = 2000
N_TEST = 500
BATCH_SIZE = 128
N_EPOCHS = 100
LEARNING_RATE = 0.1

# Only train on linear operations (0-7)
LINEAR_OPS = {i: TEMPORAL_V2_OP_NAMES[i] for i in range(8)}

CHECKPOINT_DIR = Path("v5/checkpoints/phase2_sign_only")


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


class SignOnlyTemporalLayer(nn.Module):
    """
    Temporal layer with FIXED routing (identity-like P) and learnable signs.

    P is fixed as:
        [ [1,0,0,0, 1,0,0,0],  # XOR parent routes to op 0,4
          [0,1,0,0, 0,1,0,0],  # AND parent routes to op 1,5
          [0,0,1,0, 0,0,1,0],  # OR parent routes to op 2,6
          [0,0,0,1, 0,0,0,1] ] # IMPLIES parent routes to op 3,7

    Only signs s ∈ {-1, +1}^8 are learnable.
    """
    n_ops: int = 8

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
        # FIXED routing: identity-like (each child routes to one parent)
        # Child 0,4 -> Parent 0 (XOR)
        # Child 1,5 -> Parent 1 (AND)
        # Child 2,6 -> Parent 2 (OR)
        # Child 3,7 -> Parent 3 (IMPLIES)
        P = jnp.array([
            [1., 0., 0., 0., 1., 0., 0., 0.],  # XOR parent
            [0., 1., 0., 0., 0., 1., 0., 0.],  # AND parent
            [0., 0., 1., 0., 0., 0., 1., 0.],  # OR parent
            [0., 0., 0., 1., 0., 0., 0., 1.],  # IMPLIES parent
        ])  # [4, 8]

        # Learnable signs only
        sign_logits = self.param(
            'sign_logits',
            nn.initializers.zeros,  # Start at 0 (will be pushed to +1 or -1)
            (self.n_ops,)
        )

        if training:
            s = jnp.tanh(sign_logits / temperature)
        else:
            s = jnp.sign(sign_logits)
            s = jnp.where(s == 0, 1.0, s)

        # R = P * s[None, :]
        R = P * s[None, :]  # [4, 8]

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


def train_sign_only():
    """Train Phase 2 with fixed routing, learning signs only."""
    print("=" * 60)
    print("Phase 2 Diagnostic: Sign-Only Learning")
    print("=" * 60)

    # Load logic masks
    print("\nLoading Phase 1 logic masks...")
    logic_masks = load_phase1_masks()

    # Generate dataset
    print("\nGenerating dataset...")
    train_data, test_data = create_temporal_v2_train_test_split(N_TRAIN, N_TEST, N_BITS)
    print(f"  Train: {N_TRAIN}, Test: {N_TEST}")
    print(f"  Operations: {len(LINEAR_OPS)}")

    # Expected result
    print("\nExpected signs:")
    print("  Op 0 (xor): s=+1")
    print("  Op 1 (and): s=+1")
    print("  Op 2 (or): s=+1")
    print("  Op 3 (implies): s=+1")
    print("  Op 4 (xnor): s=-1")
    print("  Op 5 (nand): s=-1")
    print("  Op 6 (nor): s=-1")
    print("  Op 7 (not_implies): s=-1")

    # Initialize model
    print("\nInitializing model (fixed routing, learnable signs)...")
    model = SignOnlyTemporalLayer(n_ops=8)

    rng = jax.random.PRNGKey(42)
    dummy_a = jnp.ones((1, N_BITS))
    dummy_b = jnp.ones((1, N_BITS))
    variables = model.init(rng, dummy_a, dummy_b, 0, logic_masks)

    print(f"  Parameters: sign_logits {variables['params']['sign_logits'].shape}")

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
            print(f"  Expected:  [+1, +1, +1, +1, -1, -1, -1, -1]")

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

    # Final signs
    print("\n" + "=" * 60)
    print("Final Signs Analysis")
    print("=" * 60)

    sign_logits = state.params['sign_logits']
    s_final = jnp.sign(sign_logits)
    s_final = jnp.where(s_final == 0, 1.0, s_final)

    print(f"\nLearned signs: {[int(x) for x in s_final]}")
    print(f"Expected:      [+1, +1, +1, +1, -1, -1, -1, -1]")

    n_correct = 0
    expected = [1, 1, 1, 1, -1, -1, -1, -1]
    for i, (op_name, exp_sign) in enumerate(zip(LINEAR_OPS.values(), expected)):
        learned = int(s_final[i])
        match = "OK" if learned == exp_sign else "WRONG"
        if learned == exp_sign:
            n_correct += 1
        print(f"  {op_name:15s}: learned={learned:+d}, expected={exp_sign:+d} [{match}]")

    print(f"\nCorrect signs: {n_correct}/{len(expected)}")

    if all_perfect:
        print("\n" + "=" * 60)
        print("SUCCESS: All operations learned with correct signs!")
        print("=" * 60)

    # Save checkpoint
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / "sign_only_params.npy"
    np.save(checkpoint_path, {
        'sign_logits': np.array(sign_logits),
        's_final': np.array(s_final),
        'mean_accuracy': float(mean_final),
    })
    print(f"\nCheckpoint saved to {checkpoint_path}")

    return state, mean_final


if __name__ == "__main__":
    train_sign_only()
