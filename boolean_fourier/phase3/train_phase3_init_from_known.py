"""
Phase 3: Initialize from Known Optimal Masks
=============================================

Like Phase 2A identity initialization, we start masks near the
brute-force-discovered optimal solutions.

This tests whether the optimization can MAINTAIN correct masks
(similar to Phase 2A maintaining identity routing).
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

from logic3_dataset import create_phase3_train_test_split
from boolean_fourier_3var import (
    boolean_fourier_3var,
    PHASE3_OPERATIONS,
    CHAR_NAMES_3VAR,
)


# Configuration
N_BITS = 64
N_TRAIN = 2000
N_TEST = 500
BATCH_SIZE = 128
N_EPOCHS = 50
LEARNING_RATE = 0.02  # Lower LR to maintain near-optimal masks
TERNARY_THRESHOLD = 0.3

# Known optimal masks from brute-force representability test
# Basis: [1, a, b, c, ab, ac, bc, abc]
KNOWN_OPTIMAL_MASKS = {
    'parity_3':     jnp.array([-1., 0., 0., 0., 0., 0., 0., 1.]),
    'majority_3':   jnp.array([-1., 0., 1., 1., 0., 0., 0., -1.]),
    'and_3':        jnp.array([-1., 0., 0., 1., 0., 1., 1., 1.]),
    'or_3':         jnp.array([-1., 1., 1., 1., -1., -1., -1., 1.]),
    'xor_ab_xor_c': jnp.array([-1., 0., 0., 0., 0., 0., 0., 1.]),
    'and_ab_or_c':  jnp.array([-1., 0., 0., 1., -1., 1., 1., 0.]),
    'or_ab_and_c':  jnp.array([-1., 0., 1., 1., 1., 0., -1., -1.]),
    'implies_ab_c': jnp.array([-1., 0., -1., 1., 0., 1., 0., 1.]),
    'xor_and_ab_c': jnp.array([-1., -1., 0., 1., 0., 1., 1., -1.]),
    'and_xor_ab_c': jnp.array([-1., 0., 0., 1., 1., 0., 0., -1.]),
}

CHECKPOINT_DIR = Path("v6/checkpoints/phase3_init_from_known")


def ternary_projection(w: jnp.ndarray, threshold: float = 0.3) -> jnp.ndarray:
    """Project to ternary {-1, 0, +1} with STE."""
    w_ternary = jnp.where(
        jnp.abs(w) < threshold,
        0.0,
        jnp.sign(w)
    )
    return w + jax.lax.stop_gradient(w_ternary - w)


def optimal_mask_init(key, shape, dtype=jnp.float32):
    """
    Initialize masks near known optimal solutions.

    Scale by 2.0 to push values firmly into ternary buckets.
    """
    n_ops = len(PHASE3_OPERATIONS)
    masks = jnp.zeros((n_ops, 8), dtype=dtype)

    for op_id, (op_name, _) in PHASE3_OPERATIONS.items():
        optimal = KNOWN_OPTIMAL_MASKS[op_name]
        # Scale by 2.0 to be clearly above/below threshold
        masks = masks.at[op_id].set(optimal * 2.0)

    # Add small noise for gradient flow
    noise = jax.random.normal(key, (n_ops, 8), dtype) * 0.1
    return masks + noise


class Phase3InitializedLayer(nn.Module):
    """
    Phase 3 layer with masks initialized to known optimal.
    """
    n_operations: int = 10
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
        """Forward pass."""
        # Masks initialized to known optimal solutions
        mask_logits = self.param(
            'mask_logits',
            optimal_mask_init,
            (self.n_operations, 8)
        )

        logits = mask_logits[operation_id]

        if training:
            mask = jnp.tanh(logits / temperature)
        else:
            mask = ternary_projection(logits, self.ternary_threshold)

        features = boolean_fourier_3var(a, b, c)
        masked = features * mask
        output = jnp.sum(masked, axis=-1)

        if training:
            output = jnp.tanh(output * 10.0)
        else:
            output = jnp.sign(output)
            output = jnp.where(output == 0, 1.0, output)

        return output


def compute_hamming_loss(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((1 - pred * target) / 2)


@partial(jax.jit, static_argnums=(5,))
def train_step(state, a, b, c, target, op_id, temperature=0.1):
    def loss_fn(params):
        pred = state.apply_fn(
            {'params': params}, a, b, c, op_id,
            temperature=temperature, training=True
        )
        return compute_hamming_loss(pred, target)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def compute_accuracy(apply_fn, params, a, b, c, target, op_id) -> float:
    pred = apply_fn(
        {'params': params}, a, b, c, op_id,
        temperature=0.05, training=False
    )
    return float(jnp.mean(pred == target))


def run_validation(n_seeds: int = 5):
    """Run Phase 3 validation with known-optimal initialization."""
    print("=" * 70)
    print("Phase 3: Initialize from Known Optimal Masks")
    print("Testing if optimization can MAINTAIN correct masks")
    print("=" * 70)

    # Generate dataset
    print("\nGenerating dataset...")
    train_data, test_data = create_phase3_train_test_split(N_TRAIN, N_TEST, N_BITS)
    print(f"  Train: {N_TRAIN}, Test: {N_TEST}")
    print(f"  Operations: {len(PHASE3_OPERATIONS)}")

    # Show initial masks
    print("\nKnown optimal masks:")
    for op_name, mask in KNOWN_OPTIMAL_MASKS.items():
        mask_str = ''.join(['+' if x > 0 else '-' if x < 0 else '0' for x in mask])
        print(f"  {op_name:20s}: [{mask_str}]")

    all_seed_results = []

    for seed in range(n_seeds):
        print(f"\n{'─'*60}")
        print(f"Seed {seed}")
        print(f"{'─'*60}")

        # Initialize model
        model = Phase3InitializedLayer(
            n_operations=len(PHASE3_OPERATIONS),
            ternary_threshold=TERNARY_THRESHOLD
        )

        rng = jax.random.PRNGKey(seed * 42 + 123)
        dummy_a = jnp.ones((1, N_BITS))
        dummy_b = jnp.ones((1, N_BITS))
        dummy_c = jnp.ones((1, N_BITS))
        variables = model.init(rng, dummy_a, dummy_b, dummy_c, 0)

        # Check initial accuracy
        print("\n  Initial accuracies:")
        init_accs = {}
        for op_id, (op_name, _) in PHASE3_OPERATIONS.items():
            a, b, c, target, _ = test_data[op_name]
            acc = compute_accuracy(model.apply, variables['params'], a, b, c, target, op_id)
            init_accs[op_name] = acc

        init_mean = np.mean(list(init_accs.values()))
        print(f"    Mean: {init_mean:.2%}")
        if init_mean < 0.99:
            worst = sorted(init_accs.items(), key=lambda x: x[1])[:3]
            print(f"    Worst: {', '.join([f'{n}={a:.0%}' for n,a in worst])}")

        # Create optimizer
        optimizer = optax.adam(LEARNING_RATE)
        state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=variables['params'],
            tx=optimizer
        )

        best_mean_acc = init_mean

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

            # Validation every 10 epochs
            if (epoch + 1) % 10 == 0:
                accuracies = {}
                for op_id, (op_name, _) in PHASE3_OPERATIONS.items():
                    a, b, c, target, _ = test_data[op_name]
                    acc = compute_accuracy(model.apply, state.params, a, b, c, target, op_id)
                    accuracies[op_name] = acc

                mean_acc = np.mean(list(accuracies.values()))
                n_perfect = sum(1 for a in accuracies.values() if a > 0.99)

                print(f"  Epoch {epoch+1}: mean={mean_acc:.2%}, perfect={n_perfect}/10")

                if mean_acc > best_mean_acc:
                    best_mean_acc = mean_acc

        # Final results
        final_accuracies = {}
        final_masks = {}
        for op_id, (op_name, _) in PHASE3_OPERATIONS.items():
            a, b, c, target, _ = test_data[op_name]
            acc = compute_accuracy(model.apply, state.params, a, b, c, target, op_id)
            final_accuracies[op_name] = acc

            mask_logits = state.params['mask_logits'][op_id]
            mask = ternary_projection(mask_logits, TERNARY_THRESHOLD)
            final_masks[op_name] = [int(x) for x in mask]

        # Check mask drift
        mask_drift = {}
        for op_id, (op_name, _) in PHASE3_OPERATIONS.items():
            optimal = KNOWN_OPTIMAL_MASKS[op_name]
            learned = jnp.array(final_masks[op_name])
            drift = float(jnp.sum(jnp.abs(learned - optimal)))
            mask_drift[op_name] = drift

        all_seed_results.append({
            'accuracies': final_accuracies,
            'masks': final_masks,
            'mask_drift': mask_drift,
            'best_mean_acc': best_mean_acc,
        })

        print(f"\n  Seed {seed} final: mean={np.mean(list(final_accuracies.values())):.2%}, drift={np.mean(list(mask_drift.values())):.2f}")

    # Aggregate results
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    # Per-operation stats
    print("\nPer-operation accuracy (mean ± std):")
    for op_id, (op_name, _) in PHASE3_OPERATIONS.items():
        accs = [r['accuracies'][op_name] for r in all_seed_results]
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        drifts = [r['mask_drift'][op_name] for r in all_seed_results]
        mean_drift = np.mean(drifts)
        status = "OK" if mean_acc > 0.99 else "LOW"
        print(f"  [{status}] {op_name:20s}: {mean_acc:.2%} ± {std_acc:.2%} (drift={mean_drift:.1f})")

    # Overall stats
    all_mean_accs = [np.mean(list(r['accuracies'].values())) for r in all_seed_results]
    all_drifts = [np.mean(list(r['mask_drift'].values())) for r in all_seed_results]

    print(f"\nOverall:")
    print(f"  Mean accuracy: {np.mean(all_mean_accs):.2%} ± {np.std(all_mean_accs):.2%}")
    print(f"  Mean mask drift: {np.mean(all_drifts):.2f} ± {np.std(all_drifts):.2f}")
    print(f"  Seeds with >99% mean: {sum(m > 0.99 for m in all_mean_accs)}/{n_seeds}")

    # Success criteria
    success = np.mean(all_mean_accs) > 0.99 and np.mean(all_drifts) < 1.0

    print("\n" + "=" * 70)
    if success:
        print("✅ PHASE 3 VALIDATION PASSED")
        print("   Known-optimal initialization achieves >99% with minimal drift")
    else:
        print("⚠️  PHASE 3: PARTIAL SUCCESS")
        print("   Check individual operation results above")
    print("=" * 70)

    # Save results
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = CHECKPOINT_DIR / "phase3_init_known_results.json"

    results_dict = {
        'n_seeds': n_seeds,
        'overall_mean_accuracy': float(np.mean(all_mean_accs)),
        'mean_mask_drift': float(np.mean(all_drifts)),
        'success': bool(success),
        'per_seed_results': [
            {
                'accuracies': {k: float(v) for k, v in r['accuracies'].items()},
                'masks': r['masks'],
                'mask_drift': {k: float(v) for k, v in r['mask_drift'].items()},
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
    results, success = run_validation(n_seeds=5)
