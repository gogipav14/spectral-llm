"""
Phase 1 Training: Binary Logic Layer (Continuous Version)
==========================================================

Train with continuous masks first to verify the spectral hypothesis.
No ternary quantization during training - just learn the optimal mask weights.

After training converges, we can quantize to ternary for inference.
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from functools import partial

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import numpy as np
from pathlib import Path

from logic_dataset import generate_logic_dataset, create_train_test_split, OP_NAMES
from ternary_ops import ternary_quantize


# Configuration
N_BITS = 64
N_TRAIN = 10000
N_TEST = 2000
BATCH_SIZE = 128
N_EPOCHS = 100  # More epochs for continuous training
LEARNING_RATE = 0.1  # Larger LR for continuous training
THRESHOLD = 0.3

CHECKPOINT_DIR = Path("v5/checkpoints/phase1_continuous")


def boolean_fourier_features(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Construct Boolean Fourier features: [1, a, b, ab] per bit."""
    ones = jnp.ones_like(a)
    ab = a * b
    return jnp.stack([ones, a, b, ab], axis=-1)


class ContinuousLogicLayer(nn.Module):
    """
    Logic layer with continuous (non-quantized) masks for training.
    Uses soft sign (tanh) for differentiable training.
    """
    n_bits: int = 64
    temperature: float = 1.0  # Controls sharpness of tanh (lower = harder)

    @nn.compact
    def __call__(self, a, b, operation_id, training: bool = True):
        # Initialize 4 continuous masks
        xor_mask = self.param('xor_mask', nn.initializers.normal(1.0), (4,))
        and_mask = self.param('and_mask', nn.initializers.normal(1.0), (4,))
        or_mask = self.param('or_mask', nn.initializers.normal(1.0), (4,))
        implies_mask = self.param('implies_mask', nn.initializers.normal(1.0), (4,))

        masks = [xor_mask, and_mask, or_mask, implies_mask]
        mask = masks[operation_id]

        # Boolean Fourier features
        features = boolean_fourier_features(a, b)  # [batch, n_bits, 4]

        # Apply mask
        masked = features * mask  # [batch, n_bits, 4]

        # Sum over Fourier characters
        output = jnp.sum(masked, axis=-1)  # [batch, n_bits]

        if training:
            # Soft sign using tanh for differentiable training
            # Scale by 10 to make it sharper (closer to hard sign)
            output = jnp.tanh(output * 10.0 / self.temperature)
        else:
            # Hard sign for inference
            output = jnp.sign(output)
            output = jnp.where(output == 0, 1.0, output)

        return output


def compute_hamming_loss(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Hamming loss: fraction of mismatched bits."""
    return jnp.mean((1 - pred * target) / 2)


@partial(jax.jit, static_argnums=(4, 5))
def train_step(state, a, b, target, op_id, apply_fn):
    """Training step with gradient update."""

    def loss_fn(params):
        # Training mode: use soft sign
        pred = apply_fn({'params': params}, a, b, op_id, training=True)
        return compute_hamming_loss(pred, target)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)

    return state, loss


def compute_accuracy(apply_fn, params, a, b, target, op_id):
    """Compute accuracy for a single operation (inference mode)."""
    # Inference mode: use hard sign
    pred = apply_fn({'params': params}, a, b, op_id, training=False)
    return float(jnp.mean(pred == target))


def main():
    print("="*60)
    print("Phase 1: Continuous Logic Layer Training")
    print("="*60)

    print(f"\nConfiguration:")
    print(f"  n_bits: {N_BITS}")
    print(f"  n_train: {N_TRAIN}")
    print(f"  batch_size: {BATCH_SIZE}")
    print(f"  epochs: {N_EPOCHS}")
    print(f"  learning_rate: {LEARNING_RATE}")

    # Create dataset
    print("\nGenerating dataset...")
    train_data, test_data = create_train_test_split(N_TRAIN, N_TEST, N_BITS)

    # Initialize model
    print("\nInitializing model...")
    model = ContinuousLogicLayer(n_bits=N_BITS)

    rng = jax.random.PRNGKey(42)
    dummy_a = jnp.ones((1, N_BITS))
    dummy_b = jnp.ones((1, N_BITS))
    variables = model.init(rng, dummy_a, dummy_b, 0)

    print(f"  Parameters: {jax.tree_util.tree_map(lambda x: x.shape, variables['params'])}")

    # Show initial masks
    print("\nInitial masks (continuous):")
    for mask_name in ['xor_mask', 'and_mask', 'or_mask', 'implies_mask']:
        w = variables['params'][mask_name]
        print(f"  {mask_name}: {w}")

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

    best_accuracy = 0.0

    for epoch in range(N_EPOCHS):
        epoch_loss = 0.0
        n_batches = 0

        # Train on each operation
        for op_id, op_name in OP_NAMES.items():
            a, b, target, _ = train_data[op_name]

            for start in range(0, len(a), BATCH_SIZE):
                end = min(start + BATCH_SIZE, len(a))
                a_batch = a[start:end]
                b_batch = b[start:end]
                t_batch = target[start:end]

                state, loss = train_step(
                    state, a_batch, b_batch, t_batch, op_id, model.apply
                )
                epoch_loss += float(loss)
                n_batches += 1

        avg_loss = epoch_loss / n_batches

        # Validation every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            accuracies = {}
            for op_id, op_name in OP_NAMES.items():
                a, b, target, _ = test_data[op_name]
                acc = compute_accuracy(model.apply, state.params, a, b, target, op_id)
                accuracies[op_name] = acc

            mean_acc = np.mean(list(accuracies.values()))

            print(f"\nEpoch {epoch+1}/{N_EPOCHS}")
            print(f"  Loss: {avg_loss:.4f}")
            print(f"  Accuracies:")
            for op_name, acc in accuracies.items():
                status = "✓" if acc > 0.99 else " "
                print(f"    {status} {op_name.upper():8s}: {acc:.2%}")

            # Show masks
            print(f"  Continuous masks:")
            for mask_name in ['xor_mask', 'and_mask', 'or_mask', 'implies_mask']:
                w = state.params[mask_name]
                print(f"    {mask_name}: {w}")

            if mean_acc > best_accuracy:
                best_accuracy = mean_acc

    # Final results
    print("\n" + "="*60)
    print("Training Complete")
    print("="*60)

    print("\nFinal Continuous Masks:")
    for mask_name in ['xor_mask', 'and_mask', 'or_mask', 'implies_mask']:
        w = state.params[mask_name]
        w_ternary = ternary_quantize(w, THRESHOLD)
        print(f"  {mask_name}:")
        print(f"    Continuous: {w}")
        print(f"    Ternary:    {w_ternary}")

    print("\nFinal Accuracies:")
    for op_id, op_name in OP_NAMES.items():
        a, b, target, _ = test_data[op_name]
        acc = compute_accuracy(model.apply, state.params, a, b, target, op_id)
        status = "✓" if acc > 0.99 else "✗"
        print(f"  {status} {op_name.upper():8s}: {acc:.2%}")

    # Check success criteria
    print("\n" + "="*60)
    print("Success Criteria Check")
    print("="*60)

    # XOR parity concentration
    xor_mask = state.params['xor_mask']
    xor_ternary = ternary_quantize(xor_mask, THRESHOLD)
    total_energy = jnp.sum(jnp.abs(xor_ternary))
    if total_energy > 0:
        parity_concentration = float(jnp.abs(xor_ternary[3]) / total_energy)
    else:
        parity_concentration = 0.0

    all_acc_pass = all(
        compute_accuracy(model.apply, state.params, test_data[op][0],
                        test_data[op][1], test_data[op][2], op_id) > 0.99
        for op_id, op in OP_NAMES.items()
    )

    print(f"  [{'✓' if parity_concentration > 0.9 else '✗'}] XOR parity concentration > 90%: {parity_concentration:.0%}")
    print(f"  [{'✓' if all_acc_pass else '✗'}] All operations > 99% accuracy")

    if parity_concentration > 0.9 and all_acc_pass:
        print("\n✅ ALL CRITERIA PASSED! Spectral hypothesis validated.")
    else:
        print("\n⚠️  Some criteria not met.")

    # Save checkpoint
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / "final_params.npy"
    np.save(checkpoint_path, dict(state.params))
    print(f"\n✓ Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
