"""
3-Variable Boolean Logic Dataset
================================

Generates synthetic examples for Phase 3 operations.
All inputs/outputs in {-1, +1} Boolean Fourier convention.

Operations:
- Pure 3-variable: parity_3, majority_3, and_3, or_3
- Cascade compositions: (a OP1 b) OP2 c

Each operation has a known spectral signature in the 8-dim Boolean Fourier basis.
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax.numpy as jnp
import numpy as np
from typing import Dict, Tuple
from boolean_fourier_3var import (
    PHASE3_OPERATIONS,
    boolean_fourier_3var,
    CHAR_NAMES_3VAR,
)


# Expected ternary masks for pure 3-variable operations
# Basis: [1, a, b, c, ab, ac, bc, abc]
EXPECTED_3VAR_MASKS = {
    # Pure operations
    'parity_3': [0, 0, 0, 0, 0, 0, 0, 1],      # Pure abc character (XOR)
    'majority_3': [0, 1, 1, 1, 0, 0, 0, 0],    # Low-degree dominant

    # AND_3: true only when all are +1
    # In {-1,+1}: AND_3 = (1 + a + b + c + ab + ac + bc + abc) / 8 thresholded
    # After sign(), only needs: [1, 1, 1, 1, 1, 1, 1, -1] or similar
    'and_3': [1, 1, 1, 1, 1, 1, 1, -1],

    # OR_3: true when any is +1
    # In {-1,+1}: OR_3 = (-1 + a + b + c - ab - ac - bc + abc) / 8 thresholded
    'or_3': [-1, 1, 1, 1, -1, -1, -1, 1],
}


def generate_3var_examples(
    n_samples: int = 10000,
    n_bits: int = 64,
    seed: int = 42
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Generate random 3-variable Boolean inputs.

    Returns:
        a, b, c: Each [n_samples, n_bits] in {-1, +1}
    """
    rng = np.random.default_rng(seed)

    a = 2 * rng.integers(0, 2, (n_samples, n_bits)) - 1
    b = 2 * rng.integers(0, 2, (n_samples, n_bits)) - 1
    c = 2 * rng.integers(0, 2, (n_samples, n_bits)) - 1

    return (
        jnp.array(a, dtype=jnp.float32),
        jnp.array(b, dtype=jnp.float32),
        jnp.array(c, dtype=jnp.float32),
    )


def create_phase3_dataset(
    n_samples: int = 10000,
    n_bits: int = 64,
    seed: int = 42
) -> Dict[str, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int]]:
    """
    Create complete Phase 3 dataset.

    Returns:
        Dict[op_name -> (a, b, c, target, op_id)]
    """
    a, b, c = generate_3var_examples(n_samples, n_bits, seed)

    dataset = {}
    for op_id, (op_name, op_fn) in PHASE3_OPERATIONS.items():
        target = op_fn(a, b, c)
        dataset[op_name] = (a, b, c, target, op_id)

    return dataset


def create_phase3_train_test_split(
    n_train: int = 2000,
    n_test: int = 500,
    n_bits: int = 64,
    train_seed: int = 42,
    test_seed: int = 123
) -> Tuple[Dict, Dict]:
    """
    Create train/test split for Phase 3.

    Returns:
        train_data, test_data: Dict[op_name -> (a, b, c, target, op_id)]
    """
    train_data = create_phase3_dataset(n_train, n_bits, train_seed)
    test_data = create_phase3_dataset(n_test, n_bits, test_seed)

    return train_data, test_data


def get_operation_spectral_signature(op_name: str) -> dict:
    """
    Get theoretical spectral signature for an operation.

    Returns dict with character names and expected mask values.
    """
    if op_name in EXPECTED_3VAR_MASKS:
        mask = EXPECTED_3VAR_MASKS[op_name]
        return {
            'mask': mask,
            'characters': {name: val for name, val in zip(CHAR_NAMES_3VAR, mask)},
            'support_size': sum(1 for v in mask if v != 0),
            'sparsity': sum(1 for v in mask if v == 0) / 8,
        }
    return None


# Operation categories for analysis
PURE_3VAR_OPS = ['parity_3', 'majority_3', 'and_3', 'or_3']
CASCADE_OPS = ['xor_ab_xor_c', 'and_ab_or_c', 'or_ab_and_c',
               'implies_ab_c', 'xor_and_ab_c', 'and_xor_ab_c']


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 3 Dataset Generation Test")
    print("=" * 60)

    # Generate dataset
    train_data, test_data = create_phase3_train_test_split(1000, 200, n_bits=64)

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_data)} operations")
    print(f"  Test: {len(test_data)} operations")

    # Verify each operation
    print("\n" + "-" * 60)
    print("Operation Verification")
    print("-" * 60)

    for op_name, (a, b, c, target, op_id) in test_data.items():
        # Check output is in {-1, +1}
        valid = jnp.all((target == -1) | (target == 1))
        n_pos = jnp.sum(target == 1) / target.size

        # Get spectral info if available
        sig = get_operation_spectral_signature(op_name)
        sparsity_str = f"sparsity={sig['sparsity']:.0%}" if sig else "unknown"

        print(f"  {op_id:2d}. {op_name:20s}: valid={valid}, +1 ratio={n_pos:.2%}, {sparsity_str}")

    # Show expected masks
    print("\n" + "-" * 60)
    print("Expected Ternary Masks (basis: [1, a, b, c, ab, ac, bc, abc])")
    print("-" * 60)

    for op_name, mask in EXPECTED_3VAR_MASKS.items():
        support = sum(1 for v in mask if v != 0)
        print(f"  {op_name:15s}: {mask}  (support={support})")

    print("\n" + "=" * 60)
