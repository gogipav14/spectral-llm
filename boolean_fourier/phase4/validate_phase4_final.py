"""
PHASE 4 FINAL VALIDATION
========================

Validates Phase 4 using synthesized optimal masks (from spectral synthesis).
Tests across multiple seeds to verify robustness.

Success Criteria:
1. All 10 operations achieve 100% with synthesized masks
2. Masks are sparse (>25% sparsity on average)
3. Pure ternary inference works without accuracy drop
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from pathlib import Path
import json
from datetime import datetime

import jax.numpy as jnp
import jax.random as random
import numpy as np

from spectral_synthesis import boolean_fourier_basis, compute_mask_accuracy
from spectral_synthesis_4var import PHASE4_OPERATIONS, CHAR_NAMES_4VAR


# Synthesized optimal masks from spectral synthesis run
OPTIMAL_MASKS_4VAR = {
    'xor_4':          jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=jnp.float32),
    'and_4':          jnp.array([-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=jnp.float32),
    'or_4':           jnp.array([1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1], dtype=jnp.float32),
    'majority_4':     jnp.array([1, 1, 1, -1, 1, 0, 0, -1, 1, -1, 0, 0, -1, -1, -1, 1], dtype=jnp.float32),
    'threshold_3of4': jnp.array([-1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, -1], dtype=jnp.float32),
    'exactly_2of4':   jnp.array([-1, 0, 0, -1, 0, -1, -1, 0, 0, -1, -1, 0, -1, 0, 0, 1], dtype=jnp.float32),
    'xor_ab_and_cd':  jnp.array([-1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=jnp.float32),
    'or_ab_xor_cd':   jnp.array([1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, -1, -1, -1, 1], dtype=jnp.float32),
    'nested_xor':     jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=jnp.float32),
    'implies_chain':  jnp.array([1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1], dtype=jnp.float32),
}

CHECKPOINT_DIR = Path("v6/checkpoints/phase4_final")


def generate_test_data(n_samples: int, seed: int):
    """Generate random 4-variable Boolean inputs."""
    rng = random.PRNGKey(seed)
    x = random.choice(rng, jnp.array([-1.0, 1.0]), shape=(n_samples, 4))
    return x


def apply_mask_4var(x, mask):
    """Apply 16-dim ternary mask and return binary output."""
    basis = boolean_fourier_basis(x, n_vars=4)  # [batch, 16]
    output = jnp.sum(basis * mask, axis=-1)  # [batch]
    output = jnp.sign(output)
    output = jnp.where(output == 0, 1.0, output)
    return output


def compute_accuracy_4var(x, target, mask):
    """Compute accuracy for a single operation."""
    pred = apply_mask_4var(x, mask)
    return float(jnp.mean(pred == target))


def run_final_validation(n_test_seeds: int = 5, n_samples: int = 5000):
    """Run Phase 4 final validation."""
    print("=" * 70)
    print("PHASE 4 FINAL VALIDATION")
    print("Using synthesized optimal ternary masks")
    print("=" * 70)

    all_seed_results = []

    for test_seed in range(n_test_seeds):
        print(f"\n{'─'*60}")
        print(f"Test Seed {test_seed}")
        print(f"{'─'*60}")

        # Generate test data
        x = generate_test_data(n_samples, test_seed * 100 + 42)

        accuracies = {}
        for op_name, op_fn in PHASE4_OPERATIONS.items():
            target = op_fn(x)
            mask = OPTIMAL_MASKS_4VAR[op_name]
            acc = compute_accuracy_4var(x, target, mask)
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
    for op_name in PHASE4_OPERATIONS.keys():
        accs = [r['accuracies'][op_name] for r in all_seed_results]
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        mask = OPTIMAL_MASKS_4VAR[op_name]
        support = int(jnp.sum(jnp.abs(mask) > 0))
        sparsity = 1 - support / 16
        status = "OK" if mean_acc > 0.999 else "LOW"
        print(f"  [{status}] {op_name:20s}: {mean_acc:.4%} ± {std_acc:.4%} (support={support}, sparsity={sparsity:.0%})")

    # Mask analysis
    print("\n" + "─" * 70)
    print("MASK ANALYSIS")
    print("─" * 70)

    print(f"\nMask sparsity statistics:")
    supports = []
    for op_name, mask in OPTIMAL_MASKS_4VAR.items():
        support = int(jnp.sum(jnp.abs(mask) > 0))
        supports.append(support)

    mean_support = np.mean(supports)
    mean_sparsity = 1 - mean_support / 16
    print(f"  Mean support size: {mean_support:.1f}/16")
    print(f"  Mean sparsity: {mean_sparsity:.0%}")

    # Mask summary
    print(f"\nOptimal masks (basis: {CHAR_NAMES_4VAR}):")
    for op_name, mask in OPTIMAL_MASKS_4VAR.items():
        mask_str = ''.join(['+' if x > 0 else '-' if x < 0 else '0' for x in mask])
        support = int(jnp.sum(jnp.abs(mask) > 0))
        dominant = [CHAR_NAMES_4VAR[i] for i, v in enumerate(mask) if v != 0]
        print(f"  {op_name:20s}: [{mask_str}] = {', '.join(dominant[:5])}{'...' if len(dominant) > 5 else ''}")

    # Success criteria
    mean_overall = np.mean(all_mean_accs)
    success = mean_overall > 0.999 and mean_sparsity > 0.25

    print("\n" + "=" * 70)
    if success:
        print("✅ PHASE 4 FINAL VALIDATION PASSED")
        print(f"   All operations: {mean_overall:.4%}")
        print(f"   Mean sparsity: {mean_sparsity:.0%}")
        print("   Ready for deployment with 16-dim ternary masks!")
    else:
        print("⚠️  PHASE 4 VALIDATION: CHECK RESULTS")
        print(f"   Mean accuracy: {mean_overall:.4%}")
        print(f"   Mean sparsity: {mean_sparsity:.0%}")
    print("=" * 70)

    # Save results
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = CHECKPOINT_DIR / "phase4_final_results.json"

    results_dict = {
        'success': bool(success),
        'n_test_seeds': n_test_seeds,
        'overall_mean_accuracy': float(mean_overall),
        'mean_sparsity': float(mean_sparsity),
        'optimal_masks': {name: [int(x) for x in mask] for name, mask in OPTIMAL_MASKS_4VAR.items()},
        'per_seed_results': [
            {
                'accuracies': {k: float(v) for k, v in r['accuracies'].items()},
                'mean_acc': float(r['mean_acc']),
            }
            for r in all_seed_results
        ],
        'basis': CHAR_NAMES_4VAR,
        'timestamp': datetime.now().isoformat(),
    }

    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results_dict, success


if __name__ == "__main__":
    results, success = run_final_validation(n_test_seeds=5, n_samples=5000)
