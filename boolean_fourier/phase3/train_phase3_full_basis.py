"""
Phase 3 Option B: Full 8-Dimensional Basis
==========================================

Direct mask learning over the 8-character Boolean Fourier basis:
  [1, a, b, c, ab, ac, bc, abc]

Each 3-variable operation learns a single ternary mask.
This tests whether all operations have sparse spectral signatures.

k=1 hardening: Uses hard sign() and ternary projection at inference.
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from functools import partial
from pathlib import Path
import json
from datetime import datetime

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import numpy as np

from logic3_dataset import (
    create_phase3_train_test_split,
    PURE_3VAR_OPS,
    CASCADE_OPS,
)
from boolean_fourier_3var import (
    boolean_fourier_3var,
    PHASE3_OPERATIONS,
    CHAR_NAMES_3VAR,
    spectral_sparsity,
)


# Configuration
N_BITS = 64
N_TRAIN = 2000
N_TEST = 500
BATCH_SIZE = 128
N_EPOCHS = 100
LEARNING_RATE = 0.05
TERNARY_THRESHOLD = 0.3

CHECKPOINT_DIR = Path("v6/checkpoints/phase3_full_basis")


def ternary_projection(w: jnp.ndarray, threshold: float = 0.3) -> jnp.ndarray:
    """Project to ternary {-1, 0, +1} with STE."""
    # Hard ternary projection
    w_ternary = jnp.where(
        jnp.abs(w) < threshold,
        0.0,
        jnp.sign(w)
    )
    # Straight-through estimator
    return w + jax.lax.stop_gradient(w_ternary - w)


class Phase3FullBasisLayer(nn.Module):
    """
    Direct 8-dim mask learning for 3-variable operations.

    Each operation has its own mask over [1, a, b, c, ab, ac, bc, abc].
    """
    n_operations: int = 10  # All Phase 3 operations
    ternary_threshold: float = 0.3

    @nn.compact
    def __call__(
        self,
        a: jnp.ndarray,
        b: jnp.ndarray,
        c: jnp.ndarray,
        operation_id: int,
        temperature: float = 0.1,
        training: bool = True
    ) -> jnp.ndarray:
        """
        Forward pass.

        a, b, c: [batch, n_bits] in {-1, +1}
        operation_id: int in [0, n_operations)
        """
        # Learnable masks: [n_operations, 8]
        mask_logits = self.param(
            'mask_logits',
            nn.initializers.normal(0.5),
            (self.n_operations, 8)
        )

        # Get mask for this operation
        logits = mask_logits[operation_id]  # [8]

        if training:
            # Soft ternary during training
            mask = jnp.tanh(logits / temperature)
        else:
            # Hard ternary at inference
            mask = ternary_projection(logits, self.ternary_threshold)

        # Boolean Fourier features
        features = boolean_fourier_3var(a, b, c)  # [batch, n_bits, 8]

        # Apply mask
        masked = features * mask  # [batch, n_bits, 8]
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
    c: jnp.ndarray,
    target: jnp.ndarray,
    op_id: int,
    temperature: float = 0.1
):
    """Training step."""
    def loss_fn(params):
        pred = state.apply_fn(
            {'params': params},
            a, b, c, op_id,
            temperature=temperature, training=True
        )
        return compute_hamming_loss(pred, target)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def compute_accuracy(apply_fn, params, a, b, c, target, op_id) -> float:
    """Compute accuracy with hard ternary masks."""
    pred = apply_fn(
        {'params': params},
        a, b, c, op_id,
        temperature=0.05, training=False
    )
    return float(jnp.mean(pred == target))


def train_full_basis(n_seeds: int = 3):
    """Train Phase 3 with full 8-dim basis."""
    print("=" * 70)
    print("Phase 3 Option B: Full 8-Dimensional Basis")
    print("Direct mask learning over [1, a, b, c, ab, ac, bc, abc]")
    print("=" * 70)

    # Generate dataset
    print("\nGenerating dataset...")
    train_data, test_data = create_phase3_train_test_split(N_TRAIN, N_TEST, N_BITS)
    print(f"  Train: {N_TRAIN}, Test: {N_TEST}")
    print(f"  Operations: {len(PHASE3_OPERATIONS)}")

    all_seed_results = []

    for seed in range(n_seeds):
        print(f"\n{'─'*60}")
        print(f"Seed {seed}")
        print(f"{'─'*60}")

        # Initialize model
        model = Phase3FullBasisLayer(
            n_operations=len(PHASE3_OPERATIONS),
            ternary_threshold=TERNARY_THRESHOLD
        )

        rng = jax.random.PRNGKey(seed * 42 + 123)
        dummy_a = jnp.ones((1, N_BITS))
        dummy_b = jnp.ones((1, N_BITS))
        dummy_c = jnp.ones((1, N_BITS))
        variables = model.init(rng, dummy_a, dummy_b, dummy_c, 0)

        # Create optimizer
        optimizer = optax.adam(LEARNING_RATE)
        state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=variables['params'],
            tx=optimizer
        )

        best_mean_acc = 0.0

        # Training loop
        for epoch in range(N_EPOCHS):
            progress = epoch / (N_EPOCHS - 1)
            temperature = 0.5 * (0.05 / 0.5) ** progress
            temperature = max(temperature, 0.05)

            epoch_loss = 0.0
            n_batches = 0

            for op_id, (op_name, _) in PHASE3_OPERATIONS.items():
                a, b, c, target, _ = train_data[op_name]

                for start in range(0, len(a), BATCH_SIZE):
                    end = min(start + BATCH_SIZE, len(a))
                    state, loss = train_step(
                        state,
                        a[start:end], b[start:end], c[start:end],
                        target[start:end], op_id, temperature
                    )
                    epoch_loss += float(loss)
                    n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)

            # Validation every 20 epochs
            if (epoch + 1) % 20 == 0 or epoch == 0:
                accuracies = {}
                for op_id, (op_name, _) in PHASE3_OPERATIONS.items():
                    a, b, c, target, _ = test_data[op_name]
                    acc = compute_accuracy(
                        model.apply, state.params,
                        a, b, c, target, op_id
                    )
                    accuracies[op_name] = acc

                mean_acc = np.mean(list(accuracies.values()))

                # Categorize
                pure_acc = np.mean([accuracies.get(name, 0) for name in PURE_3VAR_OPS if name in accuracies])
                cascade_acc = np.mean([accuracies.get(name, 0) for name in CASCADE_OPS if name in accuracies])

                print(f"\n  Epoch {epoch+1}/{N_EPOCHS}")
                print(f"    Loss: {avg_loss:.4f} | Mean: {mean_acc:.2%}")
                print(f"    Pure 3-var: {pure_acc:.2%} | Cascade: {cascade_acc:.2%}")

                # Show worst 3
                sorted_accs = sorted(accuracies.items(), key=lambda x: x[1])
                print(f"    Worst: ", end="")
                for name, acc in sorted_accs[:3]:
                    print(f"{name}={acc:.0%} ", end="")
                print()

                if mean_acc > best_mean_acc:
                    best_mean_acc = mean_acc

        # Seed final results
        final_accuracies = {}
        final_masks = {}
        for op_id, (op_name, _) in PHASE3_OPERATIONS.items():
            a, b, c, target, _ = test_data[op_name]
            acc = compute_accuracy(
                model.apply, state.params,
                a, b, c, target, op_id
            )
            final_accuracies[op_name] = acc

            # Get ternary mask
            mask_logits = state.params['mask_logits'][op_id]
            mask = ternary_projection(mask_logits, TERNARY_THRESHOLD)
            final_masks[op_name] = [int(x) for x in mask]

        all_seed_results.append({
            'accuracies': final_accuracies,
            'masks': final_masks,
            'best_mean_acc': best_mean_acc,
        })

        print(f"\n  Seed {seed} final mean: {np.mean(list(final_accuracies.values())):.2%}")

    # Aggregate results
    print("\n" + "=" * 70)
    print("FINAL RESULTS (all seeds)")
    print("=" * 70)

    # Per-operation stats
    print("\nPer-operation accuracy (mean ± std across seeds):")
    for op_id, (op_name, _) in PHASE3_OPERATIONS.items():
        accs = [r['accuracies'][op_name] for r in all_seed_results]
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        status = "OK" if mean_acc > 0.99 else "LOW" if mean_acc > 0.9 else "FAIL"
        print(f"  [{status}] {op_name:25s}: {mean_acc:.2%} ± {std_acc:.2%}")

    # Learned masks (from best seed)
    best_seed_idx = np.argmax([r['best_mean_acc'] for r in all_seed_results])
    best_masks = all_seed_results[best_seed_idx]['masks']

    print(f"\nLearned ternary masks (seed {best_seed_idx}, basis: {CHAR_NAMES_3VAR}):")
    for op_id, (op_name, _) in PHASE3_OPERATIONS.items():
        mask = best_masks[op_name]
        mask_str = ''.join(['+' if x > 0 else '-' if x < 0 else '0' for x in mask])
        support = sum(1 for v in mask if v != 0)
        sparsity = 1 - support / 8
        print(f"  {op_name:20s}: [{mask_str}] support={support} sparsity={sparsity:.0%}")

    # Overall stats
    all_mean_accs = [np.mean(list(r['accuracies'].values())) for r in all_seed_results]
    print(f"\nOverall mean accuracy: {np.mean(all_mean_accs):.2%} ± {np.std(all_mean_accs):.2%}")
    print(f"Best seed accuracy: {max(all_mean_accs):.2%}")

    # Success criteria
    success = np.mean(all_mean_accs) > 0.95
    print("\n" + "=" * 70)
    if success:
        print("✅ PHASE 3 OPTION B VALIDATION PASSED")
    else:
        print("⚠️  PHASE 3 OPTION B: NEEDS IMPROVEMENT")
    print("=" * 70)

    # Save results
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = CHECKPOINT_DIR / "phase3_full_basis_results.json"

    results_dict = {
        'n_seeds': n_seeds,
        'overall_mean_accuracy': float(np.mean(all_mean_accs)),
        'best_seed_accuracy': float(max(all_mean_accs)),
        'per_seed_results': [
            {
                'accuracies': {k: float(v) for k, v in r['accuracies'].items()},
                'masks': r['masks'],
                'best_mean_acc': float(r['best_mean_acc']),
            }
            for r in all_seed_results
        ],
        'timestamp': datetime.now().isoformat(),
    }

    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return all_seed_results, success


if __name__ == "__main__":
    results, success = train_full_basis(n_seeds=3)
