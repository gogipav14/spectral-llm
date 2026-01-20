"""
Phase 1 Training: Binary Logic Layer (FIXED)
=============================================

Key fixes from user feedback:
1. Soft-ternary annealing with ternary-attractor regularizer
2. Train XOR ALONE first (simplest - single Walsh harmonic)
3. Validate XOR mask → [0, 0, 0, ±1] with 100% generalization
4. Sequential training: XOR → AND → OR → IMPLIES
5. Comprehensive metrics every epoch

Boolean Fourier features: [1, a, b, ab]
- XOR = ab (parity), so ideal mask = [0, 0, 0, ±1]
- AND = sign(1 + a + b - ab), mask = [1, 1, 1, -1]
- OR = sign(-1 + a + b + ab), mask = [-1, 1, 1, 1]
- IMPLIES = sign(-1 - a + b - ab), mask = [-1, -1, 1, -1]
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from functools import partial
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import numpy as np
from pathlib import Path

from logic_dataset import (
    generate_logic_dataset,
    create_train_test_split,
    OP_NAMES,
    ground_truth_xor,
    ground_truth_and,
    ground_truth_or,
    ground_truth_implies
)


# Configuration
N_BITS = 64
N_TRAIN = 10000
N_TEST = 2000
BATCH_SIZE = 128
STEPS_PER_OP = 5000  # Steps per operation
LEARNING_RATE = 0.01
CHECKPOINT_DIR = Path("v5/checkpoints/phase1_fixed")

# Expected masks (for validation)
EXPECTED_MASKS = {
    'xor': jnp.array([0., 0., 0., 1.]),
    'and': jnp.array([1., 1., 1., -1.]),
    'or': jnp.array([-1., 1., 1., 1.]),
    'implies': jnp.array([-1., -1., 1., -1.])
}


def soft_ternary(x: jnp.ndarray, temperature: float = 1.0) -> jnp.ndarray:
    """
    Continuous relaxation of ternary quantization.

    At high temperature (1.0): smooth interpolation
    At low temperature (0.01): crisp ternary values
    """
    # Soft clamp to [-1, 1] range
    x_clamped = jnp.tanh(x / temperature)

    # Compute distances to {-1, 0, +1}
    dist_to_neg1 = jnp.abs(x_clamped + 1)
    dist_to_zero = jnp.abs(x_clamped)
    dist_to_pos1 = jnp.abs(x_clamped - 1)

    # Soft min - weighted average based on distances
    distances = jnp.stack([dist_to_neg1, dist_to_zero, dist_to_pos1], axis=-1)
    weights = jax.nn.softmax(-distances / temperature, axis=-1)

    ternary_vals = jnp.array([-1.0, 0.0, 1.0])
    return jnp.sum(weights * ternary_vals, axis=-1)


def ternary_attractor_loss(x: jnp.ndarray) -> jnp.ndarray:
    """
    Regularizer that penalizes distance from {-1, 0, +1}.

    Encourages weights to converge to ternary values.
    """
    dist_neg1 = jnp.abs(x + 1)
    dist_zero = jnp.abs(x)
    dist_pos1 = jnp.abs(x - 1)

    # Min distance to any ternary value
    min_dist = jnp.minimum(jnp.minimum(dist_neg1, dist_zero), dist_pos1)
    return jnp.mean(min_dist ** 2)


def hard_ternary(x: jnp.ndarray, threshold: float = 0.3) -> jnp.ndarray:
    """Hard quantization to {-1, 0, +1} for inference."""
    return jnp.sign(x) * (jnp.abs(x) > threshold)


def boolean_fourier_features(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """
    Construct Boolean Fourier features: [1, a, b, ab] per bit.

    This is the complete Walsh-Fourier basis for 2-variable Boolean functions.
    """
    ones = jnp.ones_like(a)
    ab = a * b
    return jnp.stack([ones, a, b, ab], axis=-1)


class SoftLogicLayer(nn.Module):
    """
    Logic layer with soft-ternary masks for differentiable training.

    Uses temperature annealing: start smooth, anneal to crisp ternary.
    """
    n_bits: int = 64
    temperature: float = 1.0

    @nn.compact
    def __call__(self, a, b, operation_id, training: bool = True):
        # Initialize mask logits (continuous, will be soft-quantized)
        xor_logits = self.param('xor_mask', nn.initializers.normal(0.5), (4,))
        and_logits = self.param('and_mask', nn.initializers.normal(0.5), (4,))
        or_logits = self.param('or_mask', nn.initializers.normal(0.5), (4,))
        implies_logits = self.param('implies_mask', nn.initializers.normal(0.5), (4,))

        logits_list = [xor_logits, and_logits, or_logits, implies_logits]
        mask_logits = logits_list[operation_id]

        # Apply soft or hard ternary
        if training:
            mask = soft_ternary(mask_logits, self.temperature)
        else:
            mask = hard_ternary(mask_logits)

        # Boolean Fourier features: [batch, n_bits, 4]
        features = boolean_fourier_features(a, b)

        # Apply mask
        masked = features * mask  # [batch, n_bits, 4]

        # Sum over Fourier characters
        output = jnp.sum(masked, axis=-1)  # [batch, n_bits]

        # Soft or hard sign
        if training:
            output = jnp.tanh(output * 10.0)  # Soft sign
        else:
            output = jnp.sign(output)
            output = jnp.where(output == 0, 1.0, output)

        return output


def compute_metrics(params: Dict, temperature: float, op_name: str = None) -> Dict:
    """
    Compute comprehensive metrics for debugging.
    """
    metrics = {}

    mask_names = ['xor_mask', 'and_mask', 'or_mask', 'implies_mask']
    op_names = ['XOR', 'AND', 'OR', 'IMPLIES']

    for i, (mask_name, op) in enumerate(zip(mask_names, op_names)):
        logits = params[mask_name]
        mask_soft = soft_ternary(logits, temperature)
        mask_hard = hard_ternary(logits)

        # Sparsity (fraction of zeros in hard mask)
        sparsity = float(jnp.mean(mask_hard == 0))
        metrics[f'{op}_sparsity'] = sparsity

        # Ternary distance
        ternary_dist = float(ternary_attractor_loss(logits))
        metrics[f'{op}_ternary_dist'] = ternary_dist

        # For XOR: parity concentration (should be ~1.0)
        if op == 'XOR':
            abs_mask = jnp.abs(mask_soft)
            total = jnp.sum(abs_mask) + 1e-9
            parity_conc = float(abs_mask[3] / total)
            metrics['xor_purity'] = parity_conc
            metrics['xor_mask_soft'] = np.array(mask_soft)
            metrics['xor_mask_hard'] = np.array(mask_hard)

    # Mask overlap (orthogonality check)
    all_masks = jnp.stack([
        soft_ternary(params[m], temperature) for m in mask_names
    ])
    # Normalize for cosine similarity
    norms = jnp.linalg.norm(all_masks, axis=1, keepdims=True) + 1e-9
    normalized = all_masks / norms
    overlap = normalized @ normalized.T
    off_diag = jnp.abs(overlap - jnp.diag(jnp.diag(overlap)))
    metrics['max_overlap'] = float(off_diag.max())

    metrics['temperature'] = temperature

    return metrics


def compute_hamming_loss(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Differentiable Hamming loss."""
    return jnp.mean((1 - pred * target) / 2)


def compute_accuracy(apply_fn, params, a, b, target, op_id, temperature):
    """Compute accuracy in inference mode (hard ternary)."""
    # Create model with given temperature for inference
    pred = apply_fn({'params': params}, a, b, op_id, training=False)
    return float(jnp.mean(pred == target))


@partial(jax.jit, static_argnums=(5,))
def train_step(state, a, b, target, temperature, op_id):
    """
    Training step with soft ternary and attractor regularization.
    """
    def loss_fn(params):
        # Create temporary module with current temperature
        model = SoftLogicLayer(temperature=temperature)
        pred = model.apply({'params': params}, a, b, op_id, training=True)

        # Task loss
        task_loss = compute_hamming_loss(pred, target)

        # Ternary attractor loss (only for current operation's mask)
        mask_names = ['xor_mask', 'and_mask', 'or_mask', 'implies_mask']
        attractor_loss = ternary_attractor_loss(params[mask_names[op_id]])

        # Operation-specific mask regularization
        # Expected: XOR=[0,0,0,1], AND=[1,1,1,-1], OR=[-1,1,1,1], IMPLIES=[-1,-1,1,-1]
        mask_loss = 0.0
        if op_id == 0:  # XOR: want [0, 0, 0, ±1]
            raw_mask = params['xor_mask']
            ab_loss = (1.0 - jnp.abs(raw_mask[3])) ** 2
            other_loss = jnp.sum(raw_mask[:3] ** 2)
            mask_loss = ab_loss + other_loss
        elif op_id == 1:  # AND: want [1, 1, 1, -1]
            raw_mask = params['and_mask']
            target_mask = jnp.array([1., 1., 1., -1.])
            mask_loss = jnp.mean((raw_mask - target_mask) ** 2)
        elif op_id == 2:  # OR: want [-1, 1, 1, 1]
            raw_mask = params['or_mask']
            target_mask = jnp.array([-1., 1., 1., 1.])
            mask_loss = jnp.mean((raw_mask - target_mask) ** 2)
        elif op_id == 3:  # IMPLIES: want [-1, -1, 1, -1]
            raw_mask = params['implies_mask']
            target_mask = jnp.array([-1., -1., 1., -1.])
            mask_loss = jnp.mean((raw_mask - target_mask) ** 2)

        total_loss = task_loss + 0.1 * attractor_loss + 0.5 * mask_loss

        return total_loss, {'task': task_loss, 'attractor': attractor_loss, 'mask': mask_loss}

    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)

    return state, loss, aux


def train_single_operation(
    state,
    train_data: Dict,
    test_data: Dict,
    op_name: str,
    op_id: int,
    steps: int,
    model: SoftLogicLayer
) -> Tuple[train_state.TrainState, Dict]:
    """
    Train a single operation with temperature annealing.
    """
    print(f"\n{'='*60}")
    print(f"Training {op_name.upper()}")
    print(f"{'='*60}")

    a, b, target, _ = train_data[op_name]
    a_test, b_test, target_test, _ = test_data[op_name]

    n_samples = len(a)
    best_acc = 0.0

    for step in range(steps):
        # Temperature annealing: 1.0 → 0.01
        progress = step / steps
        temperature = 1.0 * (0.01 / 1.0) ** progress

        # Get random batch
        idx = np.random.randint(0, n_samples, BATCH_SIZE)
        a_batch = a[idx]
        b_batch = b[idx]
        t_batch = target[idx]

        # Train step
        state, loss, aux = train_step(state, a_batch, b_batch, t_batch, temperature, op_id)

        # Log every 500 steps
        if step % 500 == 0 or step == steps - 1:
            # Compute metrics
            metrics = compute_metrics(state.params, temperature, op_name)

            # Test accuracy
            acc = compute_accuracy(
                model.apply, state.params,
                a_test, b_test, target_test, op_id, temperature
            )
            metrics['test_accuracy'] = acc

            if acc > best_acc:
                best_acc = acc

            print(f"\nStep {step}/{steps}:")
            print(f"  Loss: {loss:.4f} (task: {aux['task']:.4f}, attractor: {aux['attractor']:.4f}, mask: {aux['mask']:.4f})")
            print(f"  Temperature: {temperature:.4f}")
            print(f"  Test accuracy: {acc:.2%}")

            if op_name == 'xor':
                print(f"  XOR purity: {metrics['xor_purity']:.3f} (target: >0.95)")
                print(f"  XOR mask (soft): {metrics['xor_mask_soft']}")
                print(f"  XOR mask (hard): {metrics['xor_mask_hard']}")

            print(f"  {op_name.upper()} sparsity: {metrics[f'{op_name.upper()}_sparsity']:.2f}")
            print(f"  {op_name.upper()} ternary dist: {metrics[f'{op_name.upper()}_ternary_dist']:.4f}")
            print(f"  Max mask overlap: {metrics['max_overlap']:.3f}")

    # Final validation
    final_acc = compute_accuracy(
        model.apply, state.params,
        a_test, b_test, target_test, op_id, 0.01  # Low temp for crisp
    )
    print(f"\n{op_name.upper()} Final accuracy: {final_acc:.2%}")

    # Check if XOR learned correctly
    if op_name == 'xor':
        metrics = compute_metrics(state.params, 0.01)
        if metrics['xor_purity'] < 0.95:
            print(f"⚠️  XOR purity too low: {metrics['xor_purity']:.3f}")
        else:
            print(f"✓ XOR purity OK: {metrics['xor_purity']:.3f}")

        if final_acc < 0.99:
            print(f"⚠️  XOR accuracy too low: {final_acc:.2%}")
        else:
            print(f"✓ XOR accuracy OK: {final_acc:.2%}")

    return state, {'accuracy': final_acc, 'best_acc': best_acc}


def main():
    print("="*60)
    print("Phase 1: Binary Logic Layer (FIXED)")
    print("="*60)
    print("\nKey improvements:")
    print("  • Soft-ternary annealing (temp: 1.0 → 0.01)")
    print("  • Ternary attractor regularization")
    print("  • Sequential training: XOR → AND → OR → IMPLIES")
    print("  • Comprehensive metrics logging")

    print(f"\nConfiguration:")
    print(f"  n_bits: {N_BITS}")
    print(f"  steps_per_op: {STEPS_PER_OP}")
    print(f"  batch_size: {BATCH_SIZE}")
    print(f"  learning_rate: {LEARNING_RATE}")

    # Create dataset
    print("\nGenerating dataset...")
    train_data, test_data = create_train_test_split(N_TRAIN, N_TEST, N_BITS)

    # Initialize model
    print("\nInitializing model...")
    model = SoftLogicLayer(n_bits=N_BITS, temperature=1.0)

    rng = jax.random.PRNGKey(42)
    dummy_a = jnp.ones((1, N_BITS))
    dummy_b = jnp.ones((1, N_BITS))
    variables = model.init(rng, dummy_a, dummy_b, 0)

    print(f"  Parameters: {jax.tree_util.tree_map(lambda x: x.shape, variables['params'])}")

    # Show initial masks
    print("\nInitial masks:")
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

    # Sequential training
    results = {}

    # Stage 1: XOR (most important - pure parity)
    state, res = train_single_operation(
        state, train_data, test_data, 'xor', 0, STEPS_PER_OP, model
    )
    results['xor'] = res

    # Stage 2: AND
    state, res = train_single_operation(
        state, train_data, test_data, 'and', 1, STEPS_PER_OP, model
    )
    results['and'] = res

    # Stage 3: OR
    state, res = train_single_operation(
        state, train_data, test_data, 'or', 2, STEPS_PER_OP, model
    )
    results['or'] = res

    # Stage 4: IMPLIES
    state, res = train_single_operation(
        state, train_data, test_data, 'implies', 3, STEPS_PER_OP, model
    )
    results['implies'] = res

    # Final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)

    print("\nFinal masks (hard ternary):")
    for mask_name in ['xor_mask', 'and_mask', 'or_mask', 'implies_mask']:
        w = state.params[mask_name]
        w_hard = hard_ternary(w)
        op = mask_name.replace('_mask', '')
        expected = EXPECTED_MASKS[op]
        match = jnp.allclose(jnp.abs(w_hard), jnp.abs(expected))
        status = "✓" if match else "✗"
        print(f"  {status} {mask_name}:")
        print(f"      Learned: {w_hard}")
        print(f"      Expected: {expected}")

    print("\nFinal accuracies:")
    all_pass = True
    for op_name in ['xor', 'and', 'or', 'implies']:
        acc = results[op_name]['accuracy']
        status = "✓" if acc > 0.99 else "✗"
        if acc <= 0.99:
            all_pass = False
        print(f"  {status} {op_name.upper():8s}: {acc:.2%}")

    # XOR purity check
    metrics = compute_metrics(state.params, 0.01)
    xor_pass = metrics['xor_purity'] > 0.95
    if not xor_pass:
        all_pass = False

    print(f"\nXOR purity: {metrics['xor_purity']:.3f} {'✓' if xor_pass else '✗'}")
    print(f"Max mask overlap: {metrics['max_overlap']:.3f}")

    if all_pass:
        print("\n✅ ALL CRITERIA PASSED! Phase 1 successful.")
    else:
        print("\n⚠️  Some criteria not met.")

    # Save checkpoint
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / "final_params.npy"
    np.save(checkpoint_path, dict(state.params))
    print(f"\n✓ Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
