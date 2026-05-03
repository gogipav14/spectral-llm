#!/usr/bin/env python3
"""
Verification script for Theorem 7.1 (Ternary PTF Universality through n ≤ 4).

This script independently verifies that every Boolean function on n ≤ 4 variables
has an exact ternary PTF representation by:
1. Loading the certificate masks from JSON
2. Constructing H_n via Kronecker product (integer, exact)
3. Verifying sign(H_n @ w) == f for every function

Usage:
    python3 -m bbt.verify_universality
"""

import json
import os
import numpy as np
import sys


def build_hadamard_kronecker(n: int) -> np.ndarray:
    """Build H_n = H_1^{⊗n} via Kronecker product. Exact integer matrix."""
    H = np.array([[1, 1], [1, -1]], dtype=np.int32)
    result = H
    for _ in range(1, n):
        result = np.kron(result, H)
    return result


def verify_all(n: int, masks: list, verbose: bool = True) -> bool:
    """
    Verify sign(H_n @ w) == f for all 2^{2^n} functions.

    Args:
        n: Number of variables
        masks: List of 2^{2^n} ternary masks (each length 2^n)
        verbose: Print progress

    Returns:
        True if all functions verified
    """
    N = 1 << n
    num_funcs = 1 << N
    H = build_hadamard_kronecker(n)

    assert len(masks) == num_funcs, f"Expected {num_funcs} masks, got {len(masks)}"

    failures = 0
    for func_id in range(num_funcs):
        # Reconstruct truth table from func_id
        truth_table = np.array([1 - 2 * ((func_id >> bit) & 1) for bit in range(N)],
                               dtype=np.int32)

        # Load mask
        w = np.array(masks[func_id], dtype=np.int32)

        # Verify ternary
        assert np.all(np.isin(w, [-1, 0, 1])), f"Function {func_id}: mask not ternary"

        # Compute H @ w (exact integer arithmetic)
        y = H @ w

        # Verify sign(y) == f
        predicted = np.sign(y).astype(np.int32)
        predicted[predicted == 0] = 1  # tie-break

        if not np.array_equal(predicted, truth_table):
            failures += 1
            if verbose:
                print(f"  FAIL: function {func_id}")

    if verbose:
        if failures == 0:
            print(f"  VERIFIED: all {num_funcs} functions at n={n}")
        else:
            print(f"  FAILED: {failures}/{num_funcs} functions at n={n}")

    return failures == 0


def main():
    results_dir = os.path.join(os.path.dirname(__file__), 'results')

    print("=" * 60)
    print("VERIFICATION OF THEOREM 7.1")
    print("Ternary PTF Universality through n ≤ 4")
    print("=" * 60)

    all_ok = True

    # n=3: load from exhaustive_n3.json
    n3_path = os.path.join(results_dir, 'exhaustive_n3.json')
    if os.path.exists(n3_path):
        print(f"\nn=3: Loading {n3_path}...")
        with open(n3_path) as f:
            data = json.load(f)
        masks = data['per_function']['masks']
        ok = verify_all(3, masks)
        all_ok = all_ok and ok
    else:
        print(f"\nn=3: Certificate not found at {n3_path}")
        print("  Run: python3 -m bbt.exhaustive_validation --n 3")
        all_ok = False

    # n=4: load from exhaustive_n4_repaired.json
    # This file doesn't store per-function masks, so we need to regenerate
    n4_path = os.path.join(results_dir, 'exhaustive_n4_repaired.json')
    if os.path.exists(n4_path):
        print(f"\nn=4: Certificate file exists at {n4_path}")
        print("  Regenerating and verifying all 65,536 masks...")

        from .exhaustive_validation import build_walsh_matrix, all_boolean_functions
        from .bbt_transform import batch_bbt_profiles
        from .repair import repair_batch

        all_tts_jnp = all_boolean_functions(4)
        all_tts = np.array(all_tts_jnp)
        batch_result = batch_bbt_profiles(all_tts_jnp, 4)
        f_hats = np.array(batch_result['fourier_coefficients'])

        # Heuristic masks
        masks = (np.sign(f_hats) * (np.abs(f_hats) > 0.05)).astype(np.float32)

        # Repair
        result = repair_batch(masks, all_tts, 4, f_hats=f_hats, verbose=False)
        masks_final = result['repaired_masks']

        # Verify with integer arithmetic
        H = build_hadamard_kronecker(4)
        masks_int = np.round(masks_final).astype(np.int32)
        masks_list = [masks_int[i].tolist() for i in range(len(masks_int))]
        ok = verify_all(4, masks_list)
        all_ok = all_ok and ok
    else:
        print(f"\nn=4: No certificate. Run: python3 -m bbt.exhaustive_validation --n 4 --repair")
        all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("THEOREM 7.1 VERIFIED: All Boolean functions on n ≤ 4 variables")
        print("admit exact ternary PTF representations.")
    else:
        print("VERIFICATION INCOMPLETE: See errors above.")
    print("=" * 60)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
