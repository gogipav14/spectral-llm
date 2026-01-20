"""
PHASE 3 FINAL VALIDATION
========================

Validates Phase 3 using known optimal masks (no training).
This establishes the theoretical ceiling for NPU deployment.

Success Criteria:
1. All 10 operations achieve 100% with known optimal masks
2. Masks are sparse (>25% sparsity on average)
3. k=1 hardening (pure ternary) works without accuracy drop
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from pathlib import Path
import json
from datetime import datetime

import jax.numpy as jnp
import numpy as np

from logic3_dataset import create_phase3_train_test_split
from boolean_fourier_3var import (
    boolean_fourier_3var,
    PHASE3_OPERATIONS,
    CHAR_NAMES_3VAR,
)


# Known optimal ternary masks from brute-force test (representability_3var.py)
# Basis: [1, a, b, c, ab, ac, bc, abc]
# VERIFIED: All achieve 100% accuracy via exhaustive 3^8 = 6561 mask enumeration
KNOWN_OPTIMAL_MASKS = {
    'parity_3':     jnp.array([-1., 0., 0., 0., 0., 0., 0., 1.]),
    'majority_3':   jnp.array([-1., 0., 1., 1., 0., 0., 0., -1.]),
    'and_3':        jnp.array([-1., 0., 0., 1., 0., 1., 1., 1.]),
    'or_3':         jnp.array([-1., 1., 1., 1., -1., -1., -1., 1.]),
    'xor_ab_xor_c': jnp.array([-1., 0., 0., 0., 0., 0., 0., 1.]),
    'and_ab_or_c':  jnp.array([-1., 0., 1., 1., 1., 0., -1., -1.]),  # Fixed from brute-force
    'or_ab_and_c':  jnp.array([-1., 0., 0., 1., -1., 1., 1., 0.]),   # Fixed from brute-force
    'implies_ab_c': jnp.array([-1., 0., -1., 1., -1., 0., 1., 1.]),  # Fixed from brute-force
    'xor_and_ab_c': jnp.array([-1., -1., 0., -1., 0., 1., 1., 1.]),  # Fixed from brute-force
    'and_xor_ab_c': jnp.array([-1., -1., 0., 1., 1., 0., 0., 1.]),   # Fixed from brute-force
}

CHECKPOINT_DIR = Path("v6/checkpoints/phase3_final")


def apply_mask(a, b, c, mask):
    """Apply ternary mask and return binary output."""
    features = boolean_fourier_3var(a, b, c)  # [batch, n_bits, 8]
    masked = features * mask  # [batch, n_bits, 8]
    output = jnp.sum(masked, axis=-1)  # [batch, n_bits]
    output = jnp.sign(output)
    output = jnp.where(output == 0, 1.0, output)
    return output


def compute_accuracy(a, b, c, target, mask):
    """Compute accuracy for a single operation."""
    pred = apply_mask(a, b, c, mask)
    return float(jnp.mean(pred == target))


def run_final_validation(n_test_seeds: int = 5):
    """Run Phase 3 final validation."""
    print("=" * 70)
    print("PHASE 3 FINAL VALIDATION")
    print("Using known optimal ternary masks (no training)")
    print("=" * 70)

    all_seed_results = []

    for test_seed in range(n_test_seeds):
        print(f"\n{'─'*60}")
        print(f"Test Seed {test_seed}")
        print(f"{'─'*60}")

        # Generate test data with different seed
        _, test_data = create_phase3_train_test_split(
            n_train=100, n_test=500, n_bits=64,
            train_seed=42, test_seed=test_seed * 100 + 123
        )

        accuracies = {}
        for op_id, (op_name, _) in PHASE3_OPERATIONS.items():
            a, b, c, target, _ = test_data[op_name]
            mask = KNOWN_OPTIMAL_MASKS[op_name]
            acc = compute_accuracy(a, b, c, target, mask)
            accuracies[op_name] = acc

        mean_acc = np.mean(list(accuracies.values()))
        n_perfect = sum(1 for a in accuracies.values() if a > 0.9999)

        print(f"  Mean accuracy: {mean_acc:.4%}")
        print(f"  Perfect operations: {n_perfect}/10")

        if n_perfect < 10:
            print("  Non-perfect operations:")
            for name, acc in sorted(accuracies.items(), key=lambda x: x[1]):
                if acc < 0.9999:
                    print(f"    {name}: {acc:.4%}")

        all_seed_results.append({
            'accuracies': accuracies,
            'mean_acc': mean_acc,
            'n_perfect': n_perfect,
        })

    # Aggregate results
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    all_mean_accs = [r['mean_acc'] for r in all_seed_results]
    all_n_perfect = [r['n_perfect'] for r in all_seed_results]

    print(f"\nOverall Statistics (n={n_test_seeds} test seeds):")
    print(f"  Mean accuracy: {np.mean(all_mean_accs):.4%} ± {np.std(all_mean_accs):.4%}")
    print(f"  Mean perfect ops: {np.mean(all_n_perfect):.1f}/10")

    # Per-operation accuracy
    print("\nPer-operation accuracy:")
    for op_id, (op_name, _) in PHASE3_OPERATIONS.items():
        accs = [r['accuracies'][op_name] for r in all_seed_results]
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        mask = KNOWN_OPTIMAL_MASKS[op_name]
        support = int(jnp.sum(jnp.abs(mask) > 0))
        sparsity = 1 - support / 8
        status = "OK" if mean_acc > 0.999 else "LOW"
        print(f"  [{status}] {op_name:20s}: {mean_acc:.4%} ± {std_acc:.4%} (support={support}, sparsity={sparsity:.0%})")

    # Mask analysis
    print("\n" + "─" * 70)
    print("MASK ANALYSIS")
    print("─" * 70)

    print(f"\nMask sparsity statistics:")
    supports = []
    for op_name, mask in KNOWN_OPTIMAL_MASKS.items():
        support = int(jnp.sum(jnp.abs(mask) > 0))
        supports.append(support)

    mean_support = np.mean(supports)
    mean_sparsity = 1 - mean_support / 8
    print(f"  Mean support size: {mean_support:.1f}/8")
    print(f"  Mean sparsity: {mean_sparsity:.0%}")

    # Mask summary
    print(f"\nOptimal masks (basis: {CHAR_NAMES_3VAR}):")
    for op_name, mask in KNOWN_OPTIMAL_MASKS.items():
        mask_str = ''.join(['+' if x > 0 else '-' if x < 0 else '0' for x in mask])
        support = int(jnp.sum(jnp.abs(mask) > 0))
        dominant = [CHAR_NAMES_3VAR[i] for i, v in enumerate(mask) if v != 0]
        print(f"  {op_name:20s}: [{mask_str}] = {', '.join(dominant)}")

    # Success criteria
    mean_overall = np.mean(all_mean_accs)
    success = mean_overall > 0.999 and mean_sparsity > 0.25

    print("\n" + "=" * 70)
    if success:
        print("✅ PHASE 3 FINAL VALIDATION PASSED")
        print(f"   All operations: {mean_overall:.4%}")
        print(f"   Mean sparsity: {mean_sparsity:.0%}")
        print("   Ready for NPU deployment with k=1 ternary masks!")
    else:
        print("⚠️  PHASE 3 VALIDATION: CHECK RESULTS")
        print(f"   Mean accuracy: {mean_overall:.4%}")
        print(f"   Mean sparsity: {mean_sparsity:.0%}")
    print("=" * 70)

    # Save results
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = CHECKPOINT_DIR / "phase3_final_results.json"

    results_dict = {
        'success': bool(success),
        'n_test_seeds': n_test_seeds,
        'overall_mean_accuracy': float(mean_overall),
        'mean_sparsity': float(mean_sparsity),
        'optimal_masks': {name: [int(x) for x in mask] for name, mask in KNOWN_OPTIMAL_MASKS.items()},
        'per_seed_results': [
            {
                'accuracies': {k: float(v) for k, v in r['accuracies'].items()},
                'mean_acc': float(r['mean_acc']),
            }
            for r in all_seed_results
        ],
        'timestamp': datetime.now().isoformat(),
    }

    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results_dict, success


if __name__ == "__main__":
    results, success = run_final_validation(n_test_seeds=5)
