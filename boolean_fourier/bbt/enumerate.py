"""
BBT Phase B1: Boolean Function Enumeration and NPN Classes
============================================================

Enumerates all Boolean functions on n variables and computes
NPN equivalence classes (Negation of inputs, Permutation of inputs,
Negation of output).

n=3: 256 functions → 14 NPN classes
n=4: 65536 functions → 222 NPN classes
"""

import numpy as np
import jax.numpy as jnp
from itertools import permutations
from typing import Dict, List, Tuple


# ============================================================================
# Boolean Function Enumeration
# ============================================================================

def all_boolean_functions(n: int) -> jnp.ndarray:
    """
    Generate all 2^{2^n} Boolean functions on n variables.

    Each function is a truth table in {-1, +1} encoding.

    Args:
        n: Number of variables

    Returns:
        Shape (2^{2^n}, 2^n) array of truth tables
    """
    N = 1 << n          # 2^n = number of inputs
    num_funcs = 1 << N   # 2^{2^n} = number of functions

    # Generate all truth tables: func_id's binary representation → {-1,+1}
    # Bit j of func_id gives the output for input j: 0→+1, 1→-1
    func_ids = np.arange(num_funcs, dtype=np.int64)
    truth_tables = np.zeros((num_funcs, N), dtype=np.float32)

    for bit in range(N):
        # For each input position, extract the corresponding output bit
        truth_tables[:, bit] = 1.0 - 2.0 * ((func_ids >> bit) & 1)

    return jnp.array(truth_tables)


def all_inputs(n: int) -> np.ndarray:
    """
    Generate all 2^n input vectors in {-1, +1}^n.

    Input i has bits: x_0 = (-1)^{bit_0(i)}, ..., x_{n-1} = (-1)^{bit_{n-1}(i)}

    Returns:
        Shape (2^n, n) array
    """
    N = 1 << n
    inputs = np.zeros((N, n), dtype=np.float32)
    for bit in range(n):
        inputs[:, bit] = 1.0 - 2.0 * ((np.arange(N) >> bit) & 1)
    return inputs


# ============================================================================
# NPN Equivalence Classes
# ============================================================================

def _apply_npn_transform(
    truth_table: np.ndarray,
    n: int,
    perm: Tuple[int, ...],
    input_negations: Tuple[bool, ...],
    output_negation: bool
) -> int:
    """
    Apply an NPN transform to a truth table and return its integer ID.

    Args:
        truth_table: Array of {-1, +1}, length 2^n
        n: Number of variables
        perm: Permutation of variables (tuple of length n)
        input_negations: Which inputs to negate (tuple of n bools)
        output_negation: Whether to negate output

    Returns:
        Integer ID of the transformed function
    """
    N = 1 << n
    new_tt = np.zeros(N, dtype=np.float32)

    for i in range(N):
        # Decode input i into bits
        bits = [(i >> b) & 1 for b in range(n)]

        # Apply input negations then permutation
        # New input j maps to: perm[j]-th original bit, possibly negated
        new_i = 0
        for j in range(n):
            src_bit = bits[perm[j]]
            if input_negations[perm[j]]:
                src_bit = 1 - src_bit
            new_i |= (src_bit << j)

        val = truth_table[new_i]
        if output_negation:
            val = -val
        new_tt[i] = val

    # Convert to integer ID
    func_id = 0
    for bit in range(N):
        if new_tt[bit] < 0:  # -1 maps to bit=1
            func_id |= (1 << bit)
    return func_id


def npn_canonical_id(truth_table: np.ndarray, n: int) -> int:
    """
    Compute the canonical (minimum) function ID under NPN equivalence.

    This is the lexicographically smallest function in the NPN orbit.
    """
    N = 1 << n
    min_id = None

    # Convert current truth table to ID
    def tt_to_id(tt):
        fid = 0
        for bit in range(N):
            if tt[bit] < 0:
                fid |= (1 << bit)
        return fid

    for perm in permutations(range(n)):
        for neg_mask in range(1 << n):
            input_negs = tuple(bool((neg_mask >> b) & 1) for b in range(n))
            for out_neg in [False, True]:
                fid = _apply_npn_transform(truth_table, n, perm, input_negs, out_neg)
                if min_id is None or fid < min_id:
                    min_id = fid

    return min_id


def compute_npn_classes(n: int) -> Dict[int, List[int]]:
    """
    Compute all NPN equivalence classes for n-variable Boolean functions.

    Args:
        n: Number of variables

    Returns:
        Dict mapping canonical_id → list of function IDs in that class
    """
    N = 1 << n
    num_funcs = 1 << N

    # For each function, compute its canonical ID
    all_tts = np.zeros((num_funcs, N), dtype=np.float32)
    for func_id in range(num_funcs):
        for bit in range(N):
            all_tts[func_id, bit] = 1.0 - 2.0 * ((func_id >> bit) & 1)

    classes = {}
    assigned = np.full(num_funcs, -1, dtype=np.int64)

    for func_id in range(num_funcs):
        if assigned[func_id] >= 0:
            continue

        canonical = npn_canonical_id(all_tts[func_id], n)
        if canonical not in classes:
            classes[canonical] = []

        # Find all functions in this orbit
        for perm in permutations(range(n)):
            for neg_mask in range(1 << n):
                input_negs = tuple(bool((neg_mask >> b) & 1) for b in range(n))
                for out_neg in [False, True]:
                    fid = _apply_npn_transform(all_tts[func_id], n, perm, input_negs, out_neg)
                    if assigned[fid] < 0:
                        assigned[fid] = canonical
                        classes[canonical].append(fid)

    return classes


def npn_class_representatives(n: int) -> List[Tuple[int, np.ndarray]]:
    """
    Get one representative truth table per NPN class.

    Returns:
        List of (canonical_id, truth_table) tuples
    """
    classes = compute_npn_classes(n)
    N = 1 << n
    reps = []
    for canonical_id in sorted(classes.keys()):
        tt = np.array([1.0 - 2.0 * ((canonical_id >> bit) & 1) for bit in range(N)],
                      dtype=np.float32)
        reps.append((canonical_id, tt))
    return reps


if __name__ == "__main__":
    print("=== Boolean Function Enumeration ===\n")

    # n=2
    n = 2
    all_funcs = all_boolean_functions(n)
    print(f"n={n}: {len(all_funcs)} functions, each with {all_funcs.shape[1]} entries")

    # n=3: enumerate and count NPN classes
    n = 3
    all_funcs = all_boolean_functions(n)
    print(f"n={n}: {len(all_funcs)} functions, each with {all_funcs.shape[1]} entries")

    print(f"\nComputing NPN classes for n={n}...")
    classes = compute_npn_classes(n)
    print(f"  Found {len(classes)} NPN classes")
    print(f"  Class sizes: {sorted([len(v) for v in classes.values()])}")
    print(f"  Total functions covered: {sum(len(v) for v in classes.values())}")

    # Show representatives
    reps = npn_class_representatives(n)
    print(f"\n  Representatives (canonical_id, class_size):")
    for cid, tt in reps:
        cls_size = len(classes[cid])
        print(f"    id={cid:3d} ({cid:08b}), size={cls_size}")
