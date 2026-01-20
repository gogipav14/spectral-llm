"""
Phase 1 Training: Binary Logic Layer
=====================================

Train the minimal spectral proof architecture:
- 4 ternary masks over 4-dim Boolean Fourier basis
- Learn to separate XOR, AND, OR, IMPLIES by spectral signature

Success Criteria:
1. XOR mask concentrates >90% energy on parity character (ab)
2. All operations achieve >99% accuracy
3. Masks are nearly orthogonal (cosine similarity < 0.3)

Expected Results:
- XOR mask → [0, 0, 0, 1] (spike on parity)
- AND mask → [1, 1, 1, 1]
- OR mask → [1, 1, 1, -1]
- IMPLIES mask → [1, -1, 1, 1]
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
from tqdm import tqdm

# Local imports
from binary_logic_layer import BinaryLogicLayer, GROUND_TRUTH_OPS
from ternary_ops import ternary_quantize
from logic_dataset import generate_mixed_dataset, batch_iterator, create_train_test_split, OP_NAMES


# Configuration
N_BITS = 64
N_TRAIN = 10000
N_TEST = 2000
BATCH_SIZE = 128
N_EPOCHS = 20
LEARNING_RATE = 1e-2
THRESHOLD = 0.3

CHECKPOINT_DIR = Path("v5/checkpoints/phase1_logic")


def compute_accuracy(
    model: BinaryLogicLayer,
    params: dict,
    a: jnp.ndarray,
    b: jnp.ndarray,
    target: jnp.ndarray,
    op_id: int
) -> float:
    """Compute accuracy for a single operation."""
    pred = model.apply({'params': params}, a, b, op_id)
    return float(jnp.mean(pred == target))


def compute_hamming_loss(
    pred: jnp.ndarray,
    target: jnp.ndarray
) -> jnp.ndarray:
    """
    Hamming loss: fraction of mismatched bits.

    Note: We use a differentiable approximation since pred != target
    doesn't have gradients. We compare via (1 - pred * target) / 2.
    """
    # In {-1, +1} encoding:
    # If pred == target: pred * target = 1 → (1 - 1) / 2 = 0
    # If pred != target: pred * target = -1 → (1 - (-1)) / 2 = 1
    return jnp.mean((1 - pred * target) / 2)


def train_step(state, a, b, target, op_id, model):
    """Single training step with gradient update."""

    def loss_fn(params):
        pred = model.apply({'params': params}, a, b, op_id)
        loss = compute_hamming_loss(pred, target)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)

    return state, loss


@partial(jax.jit, static_argnums=(4, 5))
def train_step_jit(state, a, b, target, op_id, apply_fn):
    """JIT-compiled training step."""

    def loss_fn(params):
        pred = apply_fn({'params': params}, a, b, op_id)
        return compute_hamming_loss(pred, target)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)

    return state, loss


def validate_spectral_spike(params: dict) -> dict:
    """
    Validate that XOR mask concentrates on parity character.

    Returns dict with metrics for all masks.
    """
    results = {}

    for mask_name in ['xor_mask', 'and_mask', 'or_mask', 'implies_mask']:
        if mask_name in params:
            w = params[mask_name]['w']
            w_ternary = ternary_quantize(w, THRESHOLD)

            # Energy concentration
            total_energy = jnp.sum(jnp.abs(w_ternary))
            if total_energy > 0:
                energy_per_coord = jnp.abs(w_ternary) / total_energy
            else:
                energy_per_coord = jnp.zeros(4)

            results[mask_name] = {
                'continuous': w,
                'ternary': w_ternary,
                'energy': energy_per_coord,
                'parity_concentration': float(energy_per_coord[3]),  # ab coordinate
                'sparsity': float(jnp.mean(w_ternary == 0))
            }

    return results


def validate_orthogonality(params: dict) -> float:
    """
    Compute maximum cosine similarity between masks.

    Returns max off-diagonal cosine similarity.
    """
    masks = []
    for mask_name in ['xor_mask', 'and_mask', 'or_mask', 'implies_mask']:
        if mask_name in params:
            w = ternary_quantize(params[mask_name]['w'], THRESHOLD)
            masks.append(w)

    if len(masks) < 2:
        return 0.0

    masks = jnp.stack(masks)
    norms = jnp.linalg.norm(masks, axis=1, keepdims=True) + 1e-8
    normalized = masks / norms

    cosine_sim = normalized @ normalized.T
    off_diag = cosine_sim - jnp.diag(jnp.diag(cosine_sim))

    return float(jnp.abs(off_diag).max())


def main():
    print("="*60)
    print("Phase 1: Binary Logic Layer Training")
    print("="*60)

    # Setup
    print(f"\nConfiguration:")
    print(f"  n_bits: {N_BITS}")
    print(f"  n_train: {N_TRAIN}")
    print(f"  n_test: {N_TEST}")
    print(f"  batch_size: {BATCH_SIZE}")
    print(f"  epochs: {N_EPOCHS}")
    print(f"  learning_rate: {LEARNING_RATE}")

    # Create dataset
    print("\nGenerating dataset...")
    train_data, test_data = create_train_test_split(N_TRAIN, N_TEST, N_BITS)

    # Initialize model
    print("\nInitializing model...")
    model = BinaryLogicLayer(n_bits=N_BITS, threshold=THRESHOLD)

    rng = jax.random.PRNGKey(42)
    dummy_a = jnp.ones((1, N_BITS))
    dummy_b = jnp.ones((1, N_BITS))
    variables = model.init(rng, dummy_a, dummy_b, 0)

    print(f"  Parameters: {jax.tree_util.tree_map(lambda x: x.shape, variables['params'])}")

    # Show initial masks
    print("\nInitial masks (before training):")
    initial_metrics = validate_spectral_spike(variables['params'])
    for name, m in initial_metrics.items():
        print(f"  {name}: {m['ternary']} (parity: {m['parity_concentration']:.2%})")

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

        # Train on each operation separately
        for op_id, op_name in OP_NAMES.items():
            a, b, target, _ = train_data[op_name]

            # Simple epoch loop (small dataset fits in memory)
            for start in range(0, len(a), BATCH_SIZE):
                end = min(start + BATCH_SIZE, len(a))
                a_batch = a[start:end]
                b_batch = b[start:end]
                t_batch = target[start:end]

                state, loss = train_step_jit(
                    state, a_batch, b_batch, t_batch, op_id, model.apply
                )
                epoch_loss += float(loss)
                n_batches += 1

        avg_loss = epoch_loss / n_batches

        # Validation
        accuracies = {}
        for op_id, op_name in OP_NAMES.items():
            a, b, target, _ = test_data[op_name]
            acc = compute_accuracy(model, state.params, a, b, target, op_id)
            accuracies[op_name] = acc

        mean_acc = np.mean(list(accuracies.values()))

        # Print progress
        print(f"\nEpoch {epoch+1}/{N_EPOCHS}")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Accuracies:")
        for op_name, acc in accuracies.items():
            status = "✓" if acc > 0.99 else " "
            print(f"    {status} {op_name.upper():8s}: {acc:.2%}")

        # Validate spectral properties
        metrics = validate_spectral_spike(state.params)
        print(f"  Masks (ternary):")
        for name, m in metrics.items():
            op = name.replace('_mask', '').upper()
            parity = m['parity_concentration']
            status = "✓" if (name == 'xor_mask' and parity > 0.9) else " "
            print(f"    {status} {op:8s}: {m['ternary']} (parity: {parity:.0%})")

        ortho = validate_orthogonality(state.params)
        print(f"  Max cosine overlap: {ortho:.3f}")

        # Track best
        if mean_acc > best_accuracy:
            best_accuracy = mean_acc

    # Final results
    print("\n" + "="*60)
    print("Training Complete")
    print("="*60)

    print(f"\nFinal Accuracies:")
    for op_id, op_name in OP_NAMES.items():
        a, b, target, _ = test_data[op_name]
        acc = compute_accuracy(model, state.params, a, b, target, op_id)
        status = "✓" if acc > 0.99 else "✗"
        print(f"  {status} {op_name.upper():8s}: {acc:.2%}")

    print(f"\nFinal Masks:")
    final_metrics = validate_spectral_spike(state.params)
    for name, m in final_metrics.items():
        op = name.replace('_mask', '').upper()
        print(f"  {op:8s}: {m['ternary']}")
        print(f"           Continuous: {m['continuous']}")
        print(f"           Parity concentration: {m['parity_concentration']:.0%}")
        print(f"           Sparsity: {m['sparsity']:.0%}")

    print(f"\nOrthogonality: {validate_orthogonality(state.params):.3f}")

    # Check success criteria
    print("\n" + "="*60)
    print("Success Criteria Check")
    print("="*60)

    xor_parity = final_metrics['xor_mask']['parity_concentration']
    all_acc_pass = all(
        compute_accuracy(model, state.params, test_data[op][0], test_data[op][1], test_data[op][2], op_id) > 0.99
        for op_id, op in OP_NAMES.items()
    )
    ortho_pass = validate_orthogonality(state.params) < 0.3

    print(f"  [{'✓' if xor_parity > 0.9 else '✗'}] XOR parity concentration > 90%: {xor_parity:.0%}")
    print(f"  [{'✓' if all_acc_pass else '✗'}] All operations > 99% accuracy")
    print(f"  [{'✓' if ortho_pass else '✗'}] Mask orthogonality < 0.3: {validate_orthogonality(state.params):.3f}")

    if xor_parity > 0.9 and all_acc_pass and ortho_pass:
        print("\n✅ ALL CRITERIA PASSED! Phase 1 proof successful.")
    else:
        print("\n⚠️  Some criteria not met. May need more training or tuning.")

    # Save checkpoint
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / "final_params.npy"
    np.save(checkpoint_path, dict(state.params))
    print(f"\n✓ Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
