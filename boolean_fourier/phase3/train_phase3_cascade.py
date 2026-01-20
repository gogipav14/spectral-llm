"""
Phase 3 Option A: Cascade Composition
=====================================

2-stage architecture that reuses Phase 1/2 primitives:
  Stage 1: d = OP1(a, b)  using 2-var mask
  Stage 2: result = OP2(d, c)  using 2-var mask

This tests whether 3-variable operations can be decomposed into
cascades of 2-variable operations.

Example: (a AND b) OR c
  Stage 1: d = AND(a, b) with mask [1, 1, 1, -1]
  Stage 2: result = OR(d, c) with mask [-1, 1, 1, 1]

k=1 hardening: Each stage uses hard routing (argmax) at inference.
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

from logic3_dataset import create_phase3_train_test_split, CASCADE_OPS
from boolean_fourier_3var import PHASE3_OPERATIONS

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "v5"))
from hierarchical_r import sinkhorn_rectangular


# Configuration
N_BITS = 64
N_TRAIN = 2000
N_TEST = 500
BATCH_SIZE = 128
N_EPOCHS = 100
LEARNING_RATE = 0.05

# Cascade operations to train
# Each tuple: (op_name, stage1_primitive, stage2_primitive)
CASCADE_DECOMPOSITIONS = {
    'xor_ab_xor_c': ('xor', 'xor'),      # (a XOR b) XOR c
    'and_ab_or_c': ('and', 'or'),        # (a AND b) OR c
    'or_ab_and_c': ('or', 'and'),        # (a OR b) AND c
    'implies_ab_c': ('implies', 'implies'),  # (a IMPLIES b) IMPLIES c
    'xor_and_ab_c': ('and', 'xor'),      # (a AND b) XOR c
    'and_xor_ab_c': ('xor', 'and'),      # (a XOR b) AND c
}

# Phase 1 primitive masks (from validated Phase 1)
PRIMITIVE_MASKS = {
    'xor': jnp.array([0., 0., 0., 1.]),
    'and': jnp.array([1., 1., 1., -1.]),
    'or': jnp.array([-1., 1., 1., 1.]),
    'implies': jnp.array([-1., -1., 1., -1.]),
}

CHECKPOINT_DIR = Path("v6/checkpoints/phase3_cascade")


def boolean_fourier_2var(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """4-dim Boolean Fourier basis: [1, a, b, ab]"""
    ones = jnp.ones_like(a)
    ab = a * b
    return jnp.stack([ones, a, b, ab], axis=-1)


def apply_2var_mask(a: jnp.ndarray, b: jnp.ndarray, mask: jnp.ndarray, hard: bool = False) -> jnp.ndarray:
    """Apply a 2-variable mask and threshold."""
    features = boolean_fourier_2var(a, b)  # [batch, n_bits, 4]
    masked = features * mask  # [batch, n_bits, 4]
    output = jnp.sum(masked, axis=-1)  # [batch, n_bits]

    if hard:
        output = jnp.sign(output)
        output = jnp.where(output == 0, 1.0, output)
    else:
        output = jnp.tanh(output * 10.0)

    return output


class CascadeLayer(nn.Module):
    """
    2-stage cascade for 3-variable operations.

    Stage 1: d = OP1(a, b)
    Stage 2: result = OP2(d, c)

    Each stage selects from 4 primitive masks via soft routing.
    """
    n_primitives: int = 4
    sinkhorn_iters: int = 20

    @nn.compact
    def __call__(
        self,
        a: jnp.ndarray,
        b: jnp.ndarray,
        c: jnp.ndarray,
        operation_id: int,
        primitive_masks: jnp.ndarray,
        temperature: float = 0.1,
        training: bool = True
    ) -> jnp.ndarray:
        """
        Forward pass for cascade composition.

        primitive_masks: [4, 4] - the 4 Phase 1 masks stacked
        """
        n_cascade_ops = len(CASCADE_DECOMPOSITIONS)

        # Stage 1 routing: which primitive for (a, b)?
        # log_alpha_1: [4 primitives, n_cascade_ops]
        log_alpha_1 = self.param(
            'log_alpha_1',
            nn.initializers.normal(0.1),
            (self.n_primitives, n_cascade_ops)
        )

        # Stage 2 routing: which primitive for (d, c)?
        log_alpha_2 = self.param(
            'log_alpha_2',
            nn.initializers.normal(0.1),
            (self.n_primitives, n_cascade_ops)
        )

        # Sinkhorn projection
        P1 = sinkhorn_rectangular(log_alpha_1, n_iters=self.sinkhorn_iters, temperature=temperature)
        P2 = sinkhorn_rectangular(log_alpha_2, n_iters=self.sinkhorn_iters, temperature=temperature)

        # Get routing weights for this operation
        w1 = P1[:, operation_id]  # [4] weights over primitives for stage 1
        w2 = P2[:, operation_id]  # [4] weights over primitives for stage 2

        if training:
            # Soft combination of primitives
            # Stage 1: d = sum_i w1[i] * OP_i(a, b)
            d = jnp.zeros_like(a)
            for i in range(self.n_primitives):
                mask_i = primitive_masks[i]
                d_i = apply_2var_mask(a, b, mask_i, hard=False)
                d = d + w1[i] * d_i

            # Stage 2: result = sum_j w2[j] * OP_j(d, c)
            result = jnp.zeros_like(a)
            for j in range(self.n_primitives):
                mask_j = primitive_masks[j]
                r_j = apply_2var_mask(d, c, mask_j, hard=False)
                result = result + w2[j] * r_j

            return jnp.tanh(result * 5.0)

        else:
            # k=1 hard routing: argmax selection
            i_best = jnp.argmax(w1)
            j_best = jnp.argmax(w2)

            mask_1 = primitive_masks[i_best]
            mask_2 = primitive_masks[j_best]

            d = apply_2var_mask(a, b, mask_1, hard=True)
            result = apply_2var_mask(d, c, mask_2, hard=True)

            return result


def compute_hamming_loss(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Hamming loss."""
    return jnp.mean((1 - pred * target) / 2)


@partial(jax.jit, static_argnums=(6,))
def train_step(
    state: train_state.TrainState,
    a: jnp.ndarray,
    b: jnp.ndarray,
    c: jnp.ndarray,
    target: jnp.ndarray,
    primitive_masks: jnp.ndarray,
    op_id: int,
    temperature: float = 0.1
):
    """Training step."""
    def loss_fn(params):
        pred = state.apply_fn(
            {'params': params},
            a, b, c, op_id, primitive_masks,
            temperature=temperature, training=True
        )
        return compute_hamming_loss(pred, target)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def compute_accuracy(apply_fn, params, a, b, c, target, primitive_masks, op_id) -> float:
    """Compute accuracy with hard routing."""
    pred = apply_fn(
        {'params': params},
        a, b, c, op_id, primitive_masks,
        temperature=0.05, training=False
    )
    return float(jnp.mean(pred == target))


def train_cascade():
    """Train Phase 3 cascade composition."""
    print("=" * 60)
    print("Phase 3 Option A: Cascade Composition")
    print("2-stage: d = OP1(a,b), result = OP2(d,c)")
    print("=" * 60)

    # Stack primitive masks
    primitive_masks = jnp.stack([
        PRIMITIVE_MASKS['xor'],
        PRIMITIVE_MASKS['and'],
        PRIMITIVE_MASKS['or'],
        PRIMITIVE_MASKS['implies'],
    ])  # [4, 4]
    print(f"\nPrimitive masks loaded: {primitive_masks.shape}")

    # Map op names to indices
    cascade_ops_list = list(CASCADE_DECOMPOSITIONS.keys())
    op_name_to_id = {name: i for i, name in enumerate(cascade_ops_list)}

    # Generate dataset (only cascade operations)
    print("\nGenerating dataset...")
    train_data, test_data = create_phase3_train_test_split(N_TRAIN, N_TEST, N_BITS)
    print(f"  Train: {N_TRAIN}, Test: {N_TEST}")
    print(f"  Operations: {cascade_ops_list}")

    # Initialize model
    print("\nInitializing cascade model...")
    model = CascadeLayer(n_primitives=4, sinkhorn_iters=20)

    rng = jax.random.PRNGKey(42)
    dummy_a = jnp.ones((1, N_BITS))
    dummy_b = jnp.ones((1, N_BITS))
    dummy_c = jnp.ones((1, N_BITS))
    variables = model.init(rng, dummy_a, dummy_b, dummy_c, 0, primitive_masks)

    print(f"  log_alpha_1 shape: {variables['params']['log_alpha_1'].shape}")
    print(f"  log_alpha_2 shape: {variables['params']['log_alpha_2'].shape}")

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
        progress = epoch / (N_EPOCHS - 1)
        temperature = 0.5 * (0.05 / 0.5) ** progress
        temperature = max(temperature, 0.05)

        epoch_loss = 0.0
        n_batches = 0

        for op_name in cascade_ops_list:
            if op_name not in test_data:
                continue
            op_id = op_name_to_id[op_name]
            a, b, c, target, _ = train_data[op_name]

            for start in range(0, len(a), BATCH_SIZE):
                end = min(start + BATCH_SIZE, len(a))
                a_batch = a[start:end]
                b_batch = b[start:end]
                c_batch = c[start:end]
                t_batch = target[start:end]

                state, loss = train_step(
                    state, a_batch, b_batch, c_batch, t_batch,
                    primitive_masks, op_id, temperature
                )
                epoch_loss += float(loss)
                n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Validation every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            accuracies = {}
            for op_name in cascade_ops_list:
                if op_name not in test_data:
                    continue
                op_id = op_name_to_id[op_name]
                a, b, c, target, _ = test_data[op_name]
                acc = compute_accuracy(
                    model.apply, state.params,
                    a, b, c, target, primitive_masks, op_id
                )
                accuracies[op_name] = acc

            mean_acc = np.mean(list(accuracies.values())) if accuracies else 0

            print(f"\nEpoch {epoch+1}/{N_EPOCHS}")
            print(f"  Loss: {avg_loss:.4f} | Mean Acc: {mean_acc:.2%} | T: {temperature:.3f}")

            # Show each operation
            for op_name, acc in accuracies.items():
                expected = CASCADE_DECOMPOSITIONS[op_name]
                status = "OK" if acc > 0.99 else "LOW"
                print(f"  [{status}] {op_name:20s}: {acc:.2%} (expect {expected[0]}→{expected[1]})")

            if mean_acc > best_mean_acc:
                best_mean_acc = mean_acc

    # Final results
    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)

    # Analyze learned routing
    log_alpha_1 = state.params['log_alpha_1']
    log_alpha_2 = state.params['log_alpha_2']
    P1 = sinkhorn_rectangular(log_alpha_1, n_iters=20, temperature=0.05)
    P2 = sinkhorn_rectangular(log_alpha_2, n_iters=20, temperature=0.05)

    primitive_names = ['XOR', 'AND', 'OR', 'IMP']

    print("\nLearned cascade routing:")
    for op_name in cascade_ops_list:
        if op_name not in test_data:
            continue
        op_id = op_name_to_id[op_name]
        expected = CASCADE_DECOMPOSITIONS[op_name]

        # Get dominant primitive for each stage
        stage1_idx = int(jnp.argmax(P1[:, op_id]))
        stage2_idx = int(jnp.argmax(P2[:, op_id]))
        stage1_name = primitive_names[stage1_idx]
        stage2_name = primitive_names[stage2_idx]

        # Accuracy
        a, b, c, target, _ = test_data[op_name]
        acc = compute_accuracy(
            model.apply, state.params,
            a, b, c, target, primitive_masks, op_id
        )

        print(f"  {op_name:20s}: {stage1_name}→{stage2_name} ({acc:.2%}) | expected: {expected[0]}→{expected[1]}")

    print(f"\nBest Mean Accuracy: {best_mean_acc:.2%}")

    # Save checkpoint
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / "cascade_params.npy"
    np.save(checkpoint_path, {
        'log_alpha_1': np.array(state.params['log_alpha_1']),
        'log_alpha_2': np.array(state.params['log_alpha_2']),
        'P1': np.array(P1),
        'P2': np.array(P2),
        'best_accuracy': float(best_mean_acc),
    })
    print(f"\nCheckpoint saved to {checkpoint_path}")

    return state, best_mean_acc


if __name__ == "__main__":
    train_cascade()
