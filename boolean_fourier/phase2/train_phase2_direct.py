"""
Phase 2 Direct: Learn Masks Directly (Diagnostic)
==================================================

Instead of hierarchical composition, learn each temporal mask directly.
This tests whether the target masks are learnable at all.

After training, we can analyze if the learned masks look like linear
combinations of the frozen logic primitives.
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


# Configuration
N_BITS = 64
N_TRAIN = 2000
N_TEST = 500
BATCH_SIZE = 128
N_EPOCHS = 100
LEARNING_RATE = 0.1
THRESHOLD = 0.3

CHECKPOINT_DIR = Path("v5/checkpoints/phase2_direct")


def soft_ternary(w: jnp.ndarray, temperature: float = 1.0) -> jnp.ndarray:
    """Soft ternary quantization."""
    sign = jnp.tanh(w / temperature)
    gate = jax.nn.sigmoid((jnp.abs(w) - THRESHOLD) / temperature)
    return sign * gate


def ternary_quantize(w: jnp.ndarray) -> jnp.ndarray:
    """Hard ternary quantization."""
    return jnp.sign(w) * (jnp.abs(w) > THRESHOLD)


def boolean_fourier_features(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Boolean Fourier basis: [1, a, b, ab]."""
    ones = jnp.ones_like(a)
    ab = a * b
    return jnp.stack([ones, a, b, ab], axis=-1)


class DirectTemporalLayer(nn.Module):
    """
    Each temporal operation has its own learned 4-dim mask.
    No composition constraint - just direct learning.
    """
    n_ops: int = 16

    @nn.compact
    def __call__(
        self,
        a: jnp.ndarray,
        b: jnp.ndarray,
        operation_id: int,
        temperature: float = 1.0,
        training: bool = True
    ) -> jnp.ndarray:
        # 16 independent masks, each 4-dim
        all_masks = self.param(
            'masks',
            nn.initializers.normal(0.5),
            (self.n_ops, 4)
        )

        # Select mask for this operation
        raw_mask = all_masks[operation_id]

        # Apply soft/hard ternary
        if training:
            mask = soft_ternary(raw_mask, temperature)
        else:
            mask = ternary_quantize(raw_mask)

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


def ternary_attractor_loss(mask: jnp.ndarray) -> jnp.ndarray:
    """Pull towards {-1, 0, +1}."""
    dist_neg1 = (mask + 1) ** 2
    dist_zero = mask ** 2
    dist_pos1 = (mask - 1) ** 2
    min_dist = jnp.minimum(jnp.minimum(dist_neg1, dist_zero), dist_pos1)
    return jnp.mean(min_dist)


@partial(jax.jit, static_argnums=(4,))
def train_step(
    state: train_state.TrainState,
    a: jnp.ndarray,
    b: jnp.ndarray,
    target: jnp.ndarray,
    op_id: int,
    temperature: float = 1.0
):
    """Training step."""

    def loss_fn(params):
        pred = state.apply_fn(
            {'params': params},
            a, b, op_id,
            temperature=temperature, training=True
        )
        task_loss = compute_hamming_loss(pred, target)

        # Ternary attractor for the mask
        mask = params['masks'][op_id]
        attractor = ternary_attractor_loss(mask)

        return task_loss + 0.01 * attractor

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def compute_accuracy(apply_fn, params, a, b, target, op_id) -> float:
    """Compute accuracy."""
    pred = apply_fn({'params': params}, a, b, op_id, temperature=0.01, training=False)
    return float(jnp.mean(pred == target))


def train_phase2_direct():
    """Train Phase 2 with direct mask learning."""
    print("="*60)
    print("Phase 2 Direct: Learn Masks Directly")
    print("="*60)

    # Expected logic masks for comparison
    logic_masks = jnp.array([
        [0., 0., 0., 1.],      # XOR
        [1., 1., 1., -1.],     # AND
        [-1., 1., 1., 1.],     # OR
        [-1., -1., 1., -1.],   # IMPLIES
    ])

    # Generate dataset
    print("\nGenerating temporal V2 dataset...")
    train_data, test_data = create_temporal_v2_train_test_split(N_TRAIN, N_TEST, N_BITS)
    print(f"  Train: {N_TRAIN}, Test: {N_TEST}")

    # Initialize model
    print("\nInitializing model...")
    model = DirectTemporalLayer(n_ops=16)

    rng = jax.random.PRNGKey(42)
    dummy_a = jnp.ones((1, N_BITS))
    dummy_b = jnp.ones((1, N_BITS))
    variables = model.init(rng, dummy_a, dummy_b, 0)

    print(f"  Parameters: masks {variables['params']['masks'].shape}")

    # Create optimizer
    optimizer = optax.adam(LEARNING_RATE)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optimizer
    )

    # Training
    print("\n" + "="*60)
    print("Training")
    print("="*60)

    for epoch in range(N_EPOCHS):
        progress = epoch / (N_EPOCHS - 1)
        temperature = 1.0 * (0.05 / 1.0) ** progress
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
                    state, a_batch, b_batch, t_batch, op_id, temperature
                )
                epoch_loss += float(loss)
                n_batches += 1

        avg_loss = epoch_loss / n_batches

        if (epoch + 1) % 20 == 0 or epoch == 0:
            accuracies = {}
            for op_id, op_name in TEMPORAL_V2_OP_NAMES.items():
                a, b, target, _ = test_data[op_name]
                acc = compute_accuracy(model.apply, state.params, a, b, target, op_id)
                accuracies[op_name] = acc

            mean_acc = np.mean(list(accuracies.values()))
            print(f"\nEpoch {epoch+1}/{N_EPOCHS}")
            print(f"  Loss: {avg_loss:.4f} | Mean Acc: {mean_acc:.2%} | T: {temperature:.3f}")

            # Worst 3
            sorted_accs = sorted(accuracies.items(), key=lambda x: x[1])
            print(f"  Worst 3:")
            for name, acc in sorted_accs[:3]:
                print(f"    {name}: {acc:.2%}")

    # Final results
    print("\n" + "="*60)
    print("Training Complete")
    print("="*60)

    print("\nFinal Accuracies:")
    all_perfect = True
    for op_id, op_name in TEMPORAL_V2_OP_NAMES.items():
        a, b, target, _ = test_data[op_name]
        acc = compute_accuracy(model.apply, state.params, a, b, target, op_id)
        status = "OK" if acc > 0.99 else "LOW"
        if acc < 0.99:
            all_perfect = False
        print(f"  [{status}] {op_name:25s}: {acc:.2%}")

    mean_final = np.mean([
        compute_accuracy(model.apply, state.params,
                        test_data[op][0], test_data[op][1], test_data[op][2], op_id)
        for op_id, op in TEMPORAL_V2_OP_NAMES.items()
    ])

    print(f"\nMean Accuracy: {mean_final:.2%}")

    # Analyze learned masks
    print("\n" + "="*60)
    print("Learned Masks Analysis")
    print("="*60)

    learned_masks = state.params['masks']
    ternary_masks = ternary_quantize(learned_masks)

    print("\nLearned masks (ternary):")
    logic_names = ['XOR', 'AND', 'OR', 'IMP']

    for op_id, op_name in TEMPORAL_V2_OP_NAMES.items():
        mask = ternary_masks[op_id]
        print(f"  {op_name:25s}: [{mask[0]:5.1f}, {mask[1]:5.1f}, {mask[2]:5.1f}, {mask[3]:5.1f}]")

    # Check if first 4 match logic primitives
    print("\nComparison with logic primitives:")
    for i, (op_name, expected) in enumerate([
        ('xor', logic_masks[0]),
        ('and', logic_masks[1]),
        ('or', logic_masks[2]),
        ('implies', logic_masks[3]),
    ]):
        learned = ternary_masks[i]
        match = jnp.allclose(learned, expected, atol=0.5)
        status = "MATCH" if match else "DIFFER"
        print(f"  {op_name:8s}: learned={learned}, expected={expected} [{status}]")

    # Check negation patterns
    print("\nNegation patterns (should be negative of corresponding logic):")
    for i, (neg_name, pos_idx) in enumerate([
        ('xnor', 0), ('nand', 1), ('nor', 2), ('not_implies', 3)
    ]):
        neg_mask = ternary_masks[4 + i]
        pos_mask = ternary_masks[pos_idx]
        expected_neg = -pos_mask
        match = jnp.allclose(neg_mask, expected_neg, atol=0.5)
        status = "OK" if match else "DIFFER"
        print(f"  {neg_name:12s}: learned={neg_mask}, expected=-{TEMPORAL_V2_OP_NAMES[pos_idx]}={expected_neg} [{status}]")

    if all_perfect:
        print("\n" + "="*60)
        print("SUCCESS: All operations learned perfectly!")
        print("="*60)

    # Save checkpoint
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / "phase2_direct_params.npy"
    np.save(checkpoint_path, {
        'masks': np.array(learned_masks),
        'ternary_masks': np.array(ternary_masks),
        'mean_accuracy': float(mean_final),
    })
    print(f"\nCheckpoint saved to {checkpoint_path}")

    return state, mean_final


if __name__ == "__main__":
    train_phase2_direct()
