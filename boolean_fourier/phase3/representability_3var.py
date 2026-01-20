"""
REPRESENTABILITY TEST: 3-Variable Operations
=============================================

Brute-force test: which 3-variable operations can be represented by
a single ternary mask over the 8-dim Boolean Fourier basis?

For basis [1, a, b, c, ab, ac, bc, abc] with sign() activation,
there are 3^8 = 6561 possible ternary masks to test.

This establishes the theoretical ceiling for single-layer Phase 3.
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax.numpy as jnp
import numpy as np
from itertools import product
from typing import Tuple, Dict

from logic3_dataset import create_phase3_train_test_split, PURE_3VAR_OPS, CASCADE_OPS
from boolean_fourier_3var import (
    boolean_fourier_3var,
    PHASE3_OPERATIONS,
    CHAR_NAMES_3VAR,
)


def test_mask_accuracy_3var(
    mask: jnp.ndarray,
    a: jnp.ndarray,
    b: jnp.ndarray,
    c: jnp.ndarray,
    target: jnp.ndarray
) -> float:
    """Test a single 8-dim mask against ground truth."""
    # Get 8-dim Boolean Fourier features
    features = boolean_fourier_3var(a, b, c)  # [n_samples, n_bits, 8]

    # Apply mask and sign
    masked = features * mask  # [n_samples, n_bits, 8]
    output = jnp.sum(masked, axis=-1)  # [n_samples, n_bits]
    output = jnp.sign(output)
    output = jnp.where(output == 0, 1.0, output)

    # Accuracy
    return float(jnp.mean(output == target))


def brute_force_best_mask_3var(
    a: jnp.ndarray,
    b: jnp.ndarray,
    c: jnp.ndarray,
    target: jnp.ndarray,
    values: Tuple[int, ...] = (-1, 0, 1),
    early_exit_threshold: float = 0.9999
) -> Tuple[jnp.ndarray, float]:
    """
    Find the best ternary mask for a 3-variable operation.

    Tests all 3^8 = 6561 combinations.
    """
    best_acc = 0.0
    best_mask = None
    n_tested = 0

    # 3^8 = 6561 combinations
    for mask_tuple in product(values, repeat=8):
        mask = jnp.array(mask_tuple, dtype=jnp.float32)
        acc = test_mask_accuracy_3var(mask, a, b, c, target)
        n_tested += 1

        if acc > best_acc:
            best_acc = acc
            best_mask = mask

        # Early exit if perfect
        if acc >= early_exit_threshold:
            return mask, acc

    return best_mask, best_acc


def run_representability_test_3var(verbose: bool = True):
    """
    Test representability of all Phase 3 operations.
    """
    print("=" * 70)
    print("REPRESENTABILITY TEST: 3-Variable Operations")
    print("Testing 3^8 = 6561 ternary masks per operation")
    print("=" * 70)

    # Generate test data
    print("\nGenerating test data...")
    _, test_data = create_phase3_train_test_split(1000, 500, n_bits=64)

    print("\n" + "─" * 70)
    print("Testing all operations...")
    print("─" * 70)

    results = {}
    representable = []
    not_representable = []

    for op_id, (op_name, op_fn) in PHASE3_OPERATIONS.items():
        a, b, c, target, _ = test_data[op_name]

        # Find best mask
        if verbose:
            print(f"\n{op_id:2d}. {op_name:25s}: searching...", end="", flush=True)

        best_mask, best_acc = brute_force_best_mask_3var(a, b, c, target)

        results[op_name] = {
            'best_mask': [int(x) for x in best_mask],
            'best_acc': best_acc,
            'op_id': op_id,
        }

        # Classify
        if best_acc > 0.99:
            representable.append(op_name)
            status = "✅ REPRESENTABLE"
        elif best_acc > 0.75:
            status = "⚠️  PARTIAL"
            not_representable.append(op_name)
        else:
            status = "❌ NOT REPRESENTABLE"
            not_representable.append(op_name)

        if verbose:
            print(f"\r{op_id:2d}. {op_name:25s}")
            print(f"    Best mask: {[int(x) for x in best_mask]} → {best_acc:.2%}")
            print(f"    {status}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nRepresentable by single ternary mask ({len(representable)}/{len(PHASE3_OPERATIONS)}):\"")
    for name in representable:
        r = results[name]
        mask_str = ''.join(['+' if x > 0 else '-' if x < 0 else '0' for x in r['best_mask']])
        print(f"  ✅ {name:25s}: [{mask_str}]")

    if not_representable:
        print(f"\nNOT representable ({len(not_representable)}/{len(PHASE3_OPERATIONS)}):")
        for name in not_representable:
            r = results[name]
            print(f"  ❌ {name:25s}: best = {r['best_acc']:.2%}")

    # Analysis by category
    print("\n" + "─" * 70)
    print("ANALYSIS BY CATEGORY")
    print("─" * 70)

    pure_rep = sum(1 for name in PURE_3VAR_OPS if name in representable)
    print(f"\nPure 3-variable operations: {pure_rep}/{len(PURE_3VAR_OPS)} representable")
    for name in PURE_3VAR_OPS:
        if name in results:
            r = results[name]
            status = "✅" if r['best_acc'] > 0.99 else "❌"
            print(f"  {status} {name:20s}: {r['best_acc']:.2%}")

    cascade_rep = sum(1 for name in CASCADE_OPS if name in representable)
    print(f"\nCascade compositions: {cascade_rep}/{len(CASCADE_OPS)} representable")
    for name in CASCADE_OPS:
        if name in results:
            r = results[name]
            status = "✅" if r['best_acc'] > 0.99 else "❌"
            print(f"  {status} {name:20s}: {r['best_acc']:.2%}")

    # Interpretation
    print("\n" + "─" * 70)
    print("INTERPRETATION")
    print("─" * 70)

    if len(representable) == len(PHASE3_OPERATIONS):
        print("\n✅ ALL operations are representable by single ternary mask!")
        print("   Phase 3 can achieve 100% with correct mask learning.")
    else:
        print(f"\n⚠️  {len(not_representable)} operations NOT representable.")
        print("   These may require 2-stage cascade architecture.")

    # Show spectral sparsity of best masks
    print("\n" + "─" * 70)
    print("SPECTRAL SPARSITY OF LEARNED MASKS")
    print("─" * 70)
    print(f"Basis: {CHAR_NAMES_3VAR}")

    for name, r in sorted(results.items(), key=lambda x: x[1]['op_id']):
        mask = r['best_mask']
        support = sum(1 for v in mask if v != 0)
        sparsity = 1 - support / 8
        dominant = [CHAR_NAMES_3VAR[i] for i, v in enumerate(mask) if abs(v) == max(abs(x) for x in mask)]
        print(f"  {name:20s}: support={support}, sparsity={sparsity:.0%}, dominant={dominant}")

    print("\n" + "=" * 70)

    return results, representable, not_representable


if __name__ == "__main__":
    results, representable, not_representable = run_representability_test_3var()
