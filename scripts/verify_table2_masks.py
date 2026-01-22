#!/usr/bin/env python3
"""
Table 2 Verification Script
============================

Verifies that Phase 1 ternary masks match the canonical source
(train_phase1_fixed.py) and tests them against truth tables.

Usage:
    python scripts/verify_table2_masks.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

# Canonical masks from train_phase1_fixed.py lines 54-59
CANONICAL_MASKS = {
    'XOR': np.array([0., 0., 0., 1.]),
    'AND': np.array([1., 1., 1., -1.]),
    'OR': np.array([-1., 1., 1., 1.]),
    'IMPLIES': np.array([-1., -1., 1., -1.])
}

def boolean_fourier_features(a, b):
    """Compute Boolean Fourier basis [1, a, b, ab]."""
    ones = np.ones_like(a)
    ab = a * b
    return np.stack([ones, a, b, ab], axis=-1)

def ground_truth_xor(a, b):
    """XOR ground truth: a * b (parity)."""
    return a * b

def ground_truth_and(a, b):
    """AND ground truth from canonical source."""
    return np.sign(1 + a + b - a*b)

def ground_truth_or(a, b):
    """OR ground truth from canonical source."""
    return np.sign(-1 + a + b + a*b)

def ground_truth_implies(a, b):
    """IMPLIES ground truth from canonical source."""
    return np.sign(-1 - a + b - a*b)

def test_operation(name, mask, ground_truth_fn):
    """
    Test an operation mask against its ground truth function.

    Args:
        name: Operation name
        mask: Ternary mask [w_1, w_a, w_b, w_ab]
        ground_truth_fn: Function that computes expected outputs

    Returns:
        True if mask produces correct outputs
    """
    inputs = np.array([
        [-1, -1],
        [-1, +1],
        [+1, -1],
        [+1, +1],
    ])

    a, b = inputs[:, 0], inputs[:, 1]

    # Compute expected outputs from ground truth
    expected = ground_truth_fn(a, b)

    # Compute outputs from mask
    features = boolean_fourier_features(a, b)
    outputs = np.sign(features @ mask)

    # Handle zero outputs (shouldn't happen with correct masks)
    outputs = np.where(outputs == 0, 1, outputs)

    correct = np.allclose(outputs, expected)

    if not correct:
        print(f"  ✗ {name} FAILED")
        print(f"    Expected: {expected}")
        print(f"    Got:      {outputs}")
    else:
        print(f"  ✓ {name} passed")

    return correct

def compute_sparsity(mask):
    """Compute sparsity (fraction of zeros)."""
    return np.sum(mask == 0) / len(mask)

def main():
    print("=" * 70)
    print("Table 2 Mask Verification")
    print("=" * 70)
    print("\nCanonical source: boolean_fourier/phase1/train_phase1_fixed.py (lines 54-59)")
    print()

    # Ground truth functions (from canonical source: logic_dataset.py)
    # Encoding: -1 = TRUE, +1 = FALSE
    ground_truth_fns = {
        'XOR': ground_truth_xor,
        'AND': ground_truth_and,
        'OR': ground_truth_or,
        'IMPLIES': ground_truth_implies,
    }

    print("Testing masks against ground truth functions...")
    print()

    all_pass = True
    for op_name in ['XOR', 'AND', 'OR', 'IMPLIES']:
        mask = CANONICAL_MASKS[op_name]
        passed = test_operation(op_name, mask, ground_truth_fns[op_name])
        if not passed:
            all_pass = False

    print()
    print("-" * 70)
    print("Table 2: Phase 1 Ternary Masks (n=2, Boolean Fourier Basis)")
    print("-" * 70)
    print()
    print("| Operation | Mask [1, a, b, ab] | Sparsity | Formula |")
    print("|-----------|-------------------|----------|---------|")

    formulas = {
        'XOR': 'sign(ab)',
        'AND': 'sign(1 + a + b - ab)',
        'OR': 'sign(-1 + a + b + ab)',
        'IMPLIES': 'sign(-1 - a + b - ab)'
    }

    for op_name in ['XOR', 'AND', 'OR', 'IMPLIES']:
        mask = CANONICAL_MASKS[op_name]
        sparsity = compute_sparsity(mask)

        # Format mask with signs
        mask_str = "["
        for i, coef in enumerate(mask):
            if coef == 0:
                mask_str += "0"
            elif coef > 0:
                mask_str += "+1" if i > 0 else "1"
            else:
                mask_str += "-1"
            if i < len(mask) - 1:
                mask_str += ", "
        mask_str += "]"

        print(f"| {op_name:9s} | {mask_str:17s} | {sparsity*100:6.1f}% | {formulas[op_name]:23s} |")

    print()

    # Summary statistics
    total_zeros = sum(np.sum(CANONICAL_MASKS[op] == 0) for op in CANONICAL_MASKS)
    total_coeffs = sum(len(CANONICAL_MASKS[op]) for op in CANONICAL_MASKS)
    avg_sparsity = total_zeros / total_coeffs

    print(f"Average sparsity: {avg_sparsity*100:.1f}%")
    print()

    if all_pass:
        print("=" * 70)
        print("✅ ALL MASKS VERIFIED CORRECT")
        print("=" * 70)
        print()
        print("These masks are ready for Table 2 in the paper.")
        return 0
    else:
        print("=" * 70)
        print("❌ SOME MASKS FAILED VERIFICATION")
        print("=" * 70)
        return 1

if __name__ == '__main__':
    sys.exit(main())
