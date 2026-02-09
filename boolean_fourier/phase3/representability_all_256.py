"""
UNIVERSAL TERNARY REPRESENTABILITY TEST: All 256 Three-Variable Boolean Functions
==================================================================================

Tests whether EVERY Boolean function f:{-1,+1}^3 -> {-1,+1} has a ternary PTF:
    f(x) = sign(sum_S w_S chi_S(x)),  w_S in {-1, 0, +1}

Method: Exhaustive enumeration of 3^8 = 6561 ternary masks for each of the
256 functions. Total configurations: 256 x 6561 = 1,679,616.

If all 256 pass, this proves Theorem 2 (Ternary Representability for n=3).
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax.numpy as jnp
import numpy as np
from itertools import product
import json
from pathlib import Path
from datetime import datetime


def boolean_fourier_3var_flat(a, b, c):
    """Compute 8-dim Fourier basis for flat (non-batched) inputs."""
    return jnp.array([1, a, b, c, a*b, a*c, b*c, a*b*c], dtype=jnp.float32)


def enumerate_all_inputs_3var():
    """All 8 inputs in {-1,+1}^3."""
    inputs = []
    for a in [-1, 1]:
        for b in [-1, 1]:
            for c in [-1, 1]:
                inputs.append((a, b, c))
    return inputs


def truth_table_to_function(tt_index):
    """
    Convert a truth table index (0-255) to a Boolean function.

    Each index encodes 8 output bits for the 8 inputs in lexicographic order.
    Convention: output in {-1, +1} where bit=0 -> +1, bit=1 -> -1.
    """
    outputs = []
    for bit_pos in range(8):
        bit = (tt_index >> bit_pos) & 1
        outputs.append(-1 if bit == 1 else 1)
    return outputs


def test_all_256():
    """Test all 256 Boolean functions for ternary representability."""

    # Precompute all inputs and their Fourier basis vectors
    inputs = enumerate_all_inputs_3var()
    basis_vectors = []
    for a, b, c in inputs:
        basis_vectors.append(boolean_fourier_3var_flat(a, b, c))
    basis_matrix = jnp.stack(basis_vectors)  # (8, 8)

    # Precompute all 6561 ternary masks
    all_masks = []
    for mask_tuple in product([-1, 0, 1], repeat=8):
        all_masks.append(jnp.array(mask_tuple, dtype=jnp.float32))
    all_masks = jnp.stack(all_masks)  # (6561, 8)

    # Precompute all mask outputs: sign(basis_matrix @ mask^T) for all masks
    # Shape: (8_inputs, 6561_masks)
    raw_outputs = basis_matrix @ all_masks.T  # (8, 6561)
    mask_predictions = jnp.sign(raw_outputs)
    # Handle sign(0) = 0 -> treat as +1
    mask_predictions = jnp.where(mask_predictions == 0, 1.0, mask_predictions)

    print("=" * 70)
    print("UNIVERSAL TERNARY REPRESENTABILITY TEST")
    print(f"Testing all 256 Boolean functions on 3 variables")
    print(f"Against all 6561 ternary masks (3^8)")
    print(f"Total configurations: {256 * 6561:,}")
    print("=" * 70)

    results = {}
    n_representable = 0
    failures = []

    for func_id in range(256):
        # Get truth table for this function
        tt = truth_table_to_function(func_id)
        target = jnp.array(tt, dtype=jnp.float32)  # (8,)

        # Check each mask: does mask_predictions[:, mask_id] == target?
        # Broadcast: (8, 6561) vs (8, 1)
        matches = (mask_predictions == target[:, None])  # (8, 6561)
        all_correct = matches.all(axis=0)  # (6561,) - True if mask is perfect

        is_representable = bool(all_correct.any())

        if is_representable:
            # Find first matching mask
            best_idx = int(jnp.argmax(all_correct.astype(jnp.int32)))
            best_mask = all_masks[best_idx]
            support = int(jnp.sum(best_mask != 0))
            n_representable += 1
        else:
            best_mask = None
            support = -1
            failures.append(func_id)

        results[func_id] = {
            'representable': is_representable,
            'truth_table': tt,
            'best_mask': [int(v) for v in best_mask] if best_mask is not None else None,
            'support_size': support,
        }

        if func_id % 32 == 0:
            print(f"  Tested {func_id}/256 functions... ({n_representable} representable so far)")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nRepresentable: {n_representable}/256 ({100*n_representable/256:.1f}%)")

    if n_representable == 256:
        print("\nTHEOREM CONFIRMED: All 256 three-variable Boolean functions")
        print("have ternary PTF representations.")
    else:
        print(f"\n{len(failures)} functions NOT representable:")
        for fid in failures:
            print(f"  Function {fid}: truth table = {results[fid]['truth_table']}")

    # Sparsity analysis
    supports = [r['support_size'] for r in results.values() if r['representable']]
    if supports:
        print(f"\nSparsity analysis (representable functions):")
        print(f"  Mean support: {np.mean(supports):.1f}/8")
        print(f"  Min support:  {np.min(supports)}/8")
        print(f"  Max support:  {np.max(supports)}/8")

        # Distribution
        from collections import Counter
        support_dist = Counter(supports)
        print(f"\n  Support distribution:")
        for s in sorted(support_dist.keys()):
            print(f"    support={s}: {support_dist[s]} functions ({100*support_dist[s]/len(supports):.1f}%)")

    # Save results
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True, parents=True)

    output = {
        'n_variables': 3,
        'n_functions': 256,
        'n_representable': n_representable,
        'n_masks_tested': 6561,
        'total_configurations': 256 * 6561,
        'all_representable': n_representable == 256,
        'failures': failures,
        'support_distribution': {str(k): v for k, v in Counter(supports).items()} if supports else {},
        'mean_support': float(np.mean(supports)) if supports else 0,
        'results': results,
        'timestamp': datetime.now().isoformat(),
    }

    output_path = output_dir / 'representability_all_256.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {output_path}")

    return output


if __name__ == '__main__':
    test_all_256()
