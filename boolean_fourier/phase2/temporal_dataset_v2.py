"""
Temporal Dataset V2: Logic Compositions
========================================

16 temporal patterns that compose the 4 frozen logic primitives (XOR, AND, OR, IMPLIES).
Uses the SAME 4-dim Boolean Fourier basis as Phase 1.

Each temporal operation should be learnable as a weighted combination of logic masks:
    temporal_mask[i] = sum_j R[j,i] * logic_mask[j]

Operations:
- 0-3: Pure logic (identical to Phase 1)
- 4-7: Negated logic (NOT of each)
- 8-11: Conditional compositions
- 12-15: Complex compositions
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Tuple, List


# Operation names and IDs
TEMPORAL_V2_OP_NAMES = {
    # Pure logic (same as Phase 1 - sanity check)
    0: 'xor',
    1: 'and',
    2: 'or',
    3: 'implies',

    # Negated logic
    4: 'xnor',        # NOT XOR = -(a*b)
    5: 'nand',        # NOT AND
    6: 'nor',         # NOT OR
    7: 'not_implies', # NOT IMPLIES

    # Conditional compositions
    8: 'if_a_then_xor_else_and',
    9: 'if_a_then_and_else_or',
    10: 'xor_of_and_with_b',
    11: 'and_of_xor_with_a',

    # Complex compositions
    12: 'or_of_and_with_xor',
    13: 'majority_vote',
    14: 'parity_of_and_or',
    15: 'xor_implies_and',
}


# Ground truth implementations in {-1, +1} encoding
def compute_xor(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """XOR: a XOR b = a * b in {-1,+1}"""
    return a * b


def compute_and(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """AND: a AND b - true only when both are true (both +1)"""
    # In {-1,+1}: AND = sign(1 + a + b - a*b)
    # Truth table: (-1,-1)->-1, (-1,+1)->-1, (+1,-1)->-1, (+1,+1)->+1
    return jnp.sign(1 + a + b - a*b)


def compute_or(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """OR: a OR b - true when either is true"""
    # In {-1,+1}: OR = sign(-1 + a + b + a*b)
    # Truth table: (-1,-1)->-1, (-1,+1)->+1, (+1,-1)->+1, (+1,+1)->+1
    return jnp.sign(-1 + a + b + a*b)


def compute_implies(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """IMPLIES: a IMPLIES b - false only when a is true and b is false"""
    # In {-1,+1}: IMPLIES = sign(-1 - a + b - a*b)
    # Truth table: (-1,-1)->+1, (-1,+1)->+1, (+1,-1)->-1, (+1,+1)->+1
    return jnp.sign(-1 - a + b - a*b)


# Negated operations
def compute_xnor(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """XNOR: NOT XOR = -(a*b)"""
    return -compute_xor(a, b)


def compute_nand(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """NAND: NOT AND"""
    return -compute_and(a, b)


def compute_nor(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """NOR: NOT OR"""
    return -compute_or(a, b)


def compute_not_implies(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """NOT IMPLIES"""
    return -compute_implies(a, b)


# Conditional compositions
def compute_if_a_then_xor_else_and(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """IF a THEN XOR ELSE AND"""
    xor_result = compute_xor(a, b)
    and_result = compute_and(a, b)
    return jnp.where(a == 1, xor_result, and_result)


def compute_if_a_then_and_else_or(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """IF a THEN AND ELSE OR"""
    and_result = compute_and(a, b)
    or_result = compute_or(a, b)
    return jnp.where(a == 1, and_result, or_result)


def compute_xor_of_and_with_b(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """XOR of (a AND b) with b"""
    and_result = compute_and(a, b)
    return compute_xor(and_result, b)


def compute_and_of_xor_with_a(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """AND of (a XOR b) with a"""
    xor_result = compute_xor(a, b)
    return compute_and(xor_result, a)


# Complex compositions
def compute_or_of_and_with_xor(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """OR of (a AND b) with (a XOR b)"""
    and_result = compute_and(a, b)
    xor_result = compute_xor(a, b)
    return compute_or(and_result, xor_result)


def compute_majority_vote(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Majority vote of XOR, AND, OR"""
    xor_result = compute_xor(a, b)
    and_result = compute_and(a, b)
    or_result = compute_or(a, b)
    vote_sum = xor_result + and_result + or_result
    return jnp.sign(vote_sum)


def compute_parity_of_and_or(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Parity (XOR) of AND and OR results"""
    and_result = compute_and(a, b)
    or_result = compute_or(a, b)
    return compute_xor(and_result, or_result)


def compute_xor_implies_and(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """(a XOR b) IMPLIES (a AND b)"""
    xor_result = compute_xor(a, b)
    and_result = compute_and(a, b)
    return compute_implies(xor_result, and_result)


# Map operation ID to ground truth function
TEMPORAL_V2_GROUND_TRUTH = {
    0: compute_xor,
    1: compute_and,
    2: compute_or,
    3: compute_implies,
    4: compute_xnor,
    5: compute_nand,
    6: compute_nor,
    7: compute_not_implies,
    8: compute_if_a_then_xor_else_and,
    9: compute_if_a_then_and_else_or,
    10: compute_xor_of_and_with_b,
    11: compute_and_of_xor_with_a,
    12: compute_or_of_and_with_xor,
    13: compute_majority_vote,
    14: compute_parity_of_and_or,
    15: compute_xor_implies_and,
}


def generate_temporal_v2_dataset(
    n_samples: int = 10000,
    n_bits: int = 64,
    seed: int = 42
) -> Dict[str, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]]:
    """
    Generate temporal V2 dataset (logic compositions).

    Returns:
        Dictionary mapping operation name to (a, b, target, op_id)
    """
    rng = np.random.default_rng(seed)

    # Generate binary inputs in {-1, +1}
    a = 2 * rng.integers(0, 2, (n_samples, n_bits)) - 1
    b = 2 * rng.integers(0, 2, (n_samples, n_bits)) - 1

    a = jnp.array(a, dtype=jnp.float32)
    b = jnp.array(b, dtype=jnp.float32)

    dataset = {}
    for op_id, op_name in TEMPORAL_V2_OP_NAMES.items():
        ground_truth_fn = TEMPORAL_V2_GROUND_TRUTH[op_id]
        target = ground_truth_fn(a, b)
        # Ensure no zeros in target
        target = jnp.where(target == 0, 1.0, target)
        dataset[op_name] = (a, b, target, op_id)

    return dataset


def create_temporal_v2_train_test_split(
    n_train: int = 10000,
    n_test: int = 2000,
    n_bits: int = 64,
    seed: int = 42
) -> Tuple[Dict, Dict]:
    """Create train/test split for temporal V2 dataset."""
    train_data = generate_temporal_v2_dataset(n_train, n_bits, seed)
    test_data = generate_temporal_v2_dataset(n_test, n_bits, seed + 10000)
    return train_data, test_data


def validate_temporal_v2_operations(n_bits: int = 64):
    """Validate that temporal V2 operations are correct."""
    print("="*60)
    print("Temporal V2 (Logic Compositions) Validation")
    print("="*60)

    # Generate small test set for verification
    n_test = 100
    rng = np.random.default_rng(42)
    a = 2 * rng.integers(0, 2, (n_test, n_bits)) - 1
    b = 2 * rng.integers(0, 2, (n_test, n_bits)) - 1
    a = jnp.array(a, dtype=jnp.float32)
    b = jnp.array(b, dtype=jnp.float32)

    print(f"\nTest shape: a={a.shape}, b={b.shape}")

    # Verify truth tables for first 4 bits of first sample
    print("\n--- Truth Table Verification (first sample, first 4 bits) ---")
    print(f"a: {a[0, :4].astype(int)}")
    print(f"b: {b[0, :4].astype(int)}")

    for op_id, op_name in TEMPORAL_V2_OP_NAMES.items():
        fn = TEMPORAL_V2_GROUND_TRUTH[op_id]
        result = fn(a, b)
        print(f"{op_id:2d}. {op_name:25s}: {result[0, :4].astype(int)}")

    # Verify outputs are in {-1, +1}
    print("\n--- Output Range Check ---")
    all_valid = True
    for op_id, op_name in TEMPORAL_V2_OP_NAMES.items():
        fn = TEMPORAL_V2_GROUND_TRUTH[op_id]
        result = fn(a, b)
        # Allow small numerical errors
        valid = jnp.all(jnp.abs(jnp.abs(result) - 1) < 0.01)
        if not valid:
            all_valid = False
            unique = jnp.unique(result)
            print(f"  {op_name}: FAIL (unique values: {unique})")

    if all_valid:
        print("  All operations output valid {-1, +1} values")

    # Verify negation relationships
    print("\n--- Negation Relationships ---")
    xor = TEMPORAL_V2_GROUND_TRUTH[0](a, b)
    xnor = TEMPORAL_V2_GROUND_TRUTH[4](a, b)
    print(f"  XOR + XNOR = 0: {jnp.allclose(xor + xnor, 0)}")

    and_result = TEMPORAL_V2_GROUND_TRUTH[1](a, b)
    nand = TEMPORAL_V2_GROUND_TRUTH[5](a, b)
    print(f"  AND + NAND = 0: {jnp.allclose(and_result + nand, 0)}")

    or_result = TEMPORAL_V2_GROUND_TRUTH[2](a, b)
    nor = TEMPORAL_V2_GROUND_TRUTH[6](a, b)
    print(f"  OR + NOR = 0: {jnp.allclose(or_result + nor, 0)}")

    print("\n" + "="*60)


if __name__ == "__main__":
    validate_temporal_v2_operations()

    print("\nGenerating full dataset...")
    train, test = create_temporal_v2_train_test_split(n_train=1000, n_test=200)

    print(f"\nTrain: {len(train)} operations")
    print(f"Test: {len(test)} operations")

    for op_name in list(train.keys())[:4]:
        a, b, t, oid = train[op_name]
        print(f"  {op_name}: a={a.shape}, target={t.shape}")
