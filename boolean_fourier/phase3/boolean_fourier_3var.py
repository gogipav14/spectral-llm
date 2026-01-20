"""
Boolean Fourier Basis for 3-Variable Functions
===============================================

Complete Walsh-Fourier basis for functions f: {-1,+1}³ → {-1,+1}

Basis characters (8 total):
    χ_∅(a,b,c) = 1           (constant)
    χ_a(a,b,c) = a           (degree 1)
    χ_b(a,b,c) = b           (degree 1)
    χ_c(a,b,c) = c           (degree 1)
    χ_ab(a,b,c) = ab         (degree 2)
    χ_ac(a,b,c) = ac         (degree 2)
    χ_bc(a,b,c) = bc         (degree 2)
    χ_abc(a,b,c) = abc       (degree 3, parity)

Any Boolean function f(a,b,c) has a unique expansion:
    f = Σ_S ĥat{f}(S) χ_S

For NPU: basis is computed via products (no multipliers needed for {-1,+1}).
"""

import jax.numpy as jnp
from typing import Tuple


def boolean_fourier_3var(a: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
    """
    Compute 8-dim Boolean Fourier basis for 3 variables.

    Args:
        a, b, c: [batch, n_bits] in {-1, +1}

    Returns:
        features: [batch, n_bits, 8] basis vectors

    Character ordering:
        [1, a, b, c, ab, ac, bc, abc]
    """
    ones = jnp.ones_like(a)
    ab = a * b
    ac = a * c
    bc = b * c
    abc = a * b * c

    return jnp.stack([ones, a, b, c, ab, ac, bc, abc], axis=-1)


def boolean_fourier_2var(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """
    Compute 4-dim Boolean Fourier basis for 2 variables.

    Args:
        a, b: [batch, n_bits] in {-1, +1}

    Returns:
        features: [batch, n_bits, 4] basis vectors

    Character ordering:
        [1, a, b, ab]
    """
    ones = jnp.ones_like(a)
    ab = a * b
    return jnp.stack([ones, a, b, ab], axis=-1)


# Character names for analysis
CHAR_NAMES_3VAR = ['1', 'a', 'b', 'c', 'ab', 'ac', 'bc', 'abc']
CHAR_DEGREES_3VAR = [0, 1, 1, 1, 2, 2, 2, 3]

CHAR_NAMES_2VAR = ['1', 'a', 'b', 'ab']
CHAR_DEGREES_2VAR = [0, 1, 1, 2]


def spectral_sparsity(mask: jnp.ndarray, threshold: float = 0.1) -> dict:
    """
    Analyze spectral sparsity of a mask.

    Returns:
        dict with support_size, energy_concentration, degree_distribution
    """
    abs_mask = jnp.abs(mask)
    total_energy = jnp.sum(abs_mask)

    # Support size (nonzero coefficients)
    support = jnp.sum(abs_mask > threshold)

    # Top-1 concentration
    top1_energy = jnp.max(abs_mask)
    top1_concentration = top1_energy / (total_energy + 1e-8)

    # Degree distribution (for 3-var)
    if len(mask) == 8:
        degrees = jnp.array(CHAR_DEGREES_3VAR)
    else:
        degrees = jnp.array(CHAR_DEGREES_2VAR)

    degree_energy = {}
    for d in range(max(degrees) + 1):
        mask_d = degrees == d
        energy_d = jnp.sum(abs_mask * mask_d)
        degree_energy[f'degree_{d}'] = float(energy_d / (total_energy + 1e-8))

    return {
        'support_size': int(support),
        'total_chars': len(mask),
        'sparsity': float(1 - support / len(mask)),
        'top1_concentration': float(top1_concentration),
        **degree_energy
    }


# =============================================================================
# Ground Truth 3-Variable Functions
# =============================================================================

def compute_parity_3(a: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
    """3-way XOR / parity: a ⊕ b ⊕ c"""
    return a * b * c  # In {-1,+1}: XOR is product


def compute_majority_3(a: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
    """Majority vote: sign(a + b + c)"""
    result = jnp.sign(a + b + c)
    return jnp.where(result == 0, 1.0, result)


def compute_and_3(a: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
    """3-way AND: true only when all are +1"""
    # Correct formula: AND_3 = sign(-3 + a + b + c + ab + ac + bc + abc)
    # Note: coefficient -3 is outside ternary {-1,0,+1}, so AND_3 may not be
    # representable by single ternary mask. Using direct computation instead.
    result = jnp.sign(-3 + a + b + c + a*b + a*c + b*c + a*b*c)
    return jnp.where(result == 0, 1.0, result)


def compute_or_3(a: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
    """3-way OR: true when any is +1"""
    # Correct formula: OR_3 = sign(3 + a + b + c - ab - ac - bc - abc)
    # Note: coefficient 3 is outside ternary {-1,0,+1}, so OR_3 may not be
    # representable by single ternary mask. Using direct computation instead.
    result = jnp.sign(3 + a + b + c - a*b - a*c - b*c - a*b*c)
    return jnp.where(result == 0, 1.0, result)


# Cascade compositions (Option A style)
def compute_xor_ab_xor_c(a: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
    """(a XOR b) XOR c = a*b*c (same as 3-way parity)"""
    return a * b * c


def compute_and_ab_or_c(a: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
    """(a AND b) OR c"""
    ab_and = jnp.sign(-1 + a + b + a*b)  # AND(a,b) = sign(-1 + a + b + ab)
    ab_and = jnp.where(ab_and == 0, 1.0, ab_and)
    result = jnp.sign(1 + ab_and + c - ab_and*c)  # OR with c = sign(1 + x + y - xy)
    return jnp.where(result == 0, 1.0, result)


def compute_or_ab_and_c(a: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
    """(a OR b) AND c"""
    ab_or = jnp.sign(1 + a + b - a*b)  # OR(a,b) = sign(1 + a + b - ab)
    ab_or = jnp.where(ab_or == 0, 1.0, ab_or)
    result = jnp.sign(-1 + ab_or + c + ab_or*c)  # AND with c = sign(-1 + x + y + xy)
    return jnp.where(result == 0, 1.0, result)


def compute_implies_ab_c(a: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
    """(a IMPLIES b) IMPLIES c"""
    ab_imp = jnp.sign(1 - a + b + a*b)  # IMPLIES(a,b) = sign(1 - a + b + ab)
    ab_imp = jnp.where(ab_imp == 0, 1.0, ab_imp)
    result = jnp.sign(1 - ab_imp + c + ab_imp*c)  # IMPLIES with c
    return jnp.where(result == 0, 1.0, result)


def compute_xor_and_ab_c(a: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
    """(a AND b) XOR c"""
    ab_and = jnp.sign(-1 + a + b + a*b)  # AND(a,b) = sign(-1 + a + b + ab)
    ab_and = jnp.where(ab_and == 0, 1.0, ab_and)
    return ab_and * c  # XOR (product never produces 0 in {-1,+1})


def compute_and_xor_ab_c(a: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
    """(a XOR b) AND c"""
    ab_xor = a * b  # XOR(a,b)
    result = jnp.sign(-1 + ab_xor + c + ab_xor*c)  # AND with c = sign(-1 + x + y + xy)
    return jnp.where(result == 0, 1.0, result)


# Operation registry
PHASE3_OPERATIONS = {
    # Pure 3-variable (sparse spectrum expected)
    0: ('parity_3', compute_parity_3),          # Should be pure abc character
    1: ('majority_3', compute_majority_3),       # Low-degree dominant
    2: ('and_3', compute_and_3),                # All characters
    3: ('or_3', compute_or_3),                  # All characters

    # Cascade compositions (Option A test cases)
    4: ('xor_ab_xor_c', compute_xor_ab_xor_c),  # = parity_3
    5: ('and_ab_or_c', compute_and_ab_or_c),
    6: ('or_ab_and_c', compute_or_ab_and_c),
    7: ('implies_ab_c', compute_implies_ab_c),
    8: ('xor_and_ab_c', compute_xor_and_ab_c),
    9: ('and_xor_ab_c', compute_and_xor_ab_c),
}


if __name__ == "__main__":
    import numpy as np

    print("="*60)
    print("Boolean Fourier 3-Variable Basis Test")
    print("="*60)

    # Generate test data
    n_samples, n_bits = 100, 64
    rng = np.random.default_rng(42)
    a = 2 * rng.integers(0, 2, (n_samples, n_bits)) - 1
    b = 2 * rng.integers(0, 2, (n_samples, n_bits)) - 1
    c = 2 * rng.integers(0, 2, (n_samples, n_bits)) - 1

    a, b, c = jnp.array(a, dtype=jnp.float32), jnp.array(b, dtype=jnp.float32), jnp.array(c, dtype=jnp.float32)

    # Test basis
    features = boolean_fourier_3var(a, b, c)
    print(f"\nBasis shape: {features.shape}")
    print(f"Characters: {CHAR_NAMES_3VAR}")

    # Verify operations
    print("\n" + "-"*60)
    print("Operation Verification")
    print("-"*60)

    for op_id, (name, fn) in PHASE3_OPERATIONS.items():
        result = fn(a, b, c)
        # Verify output is in {-1, +1}
        valid = jnp.all((result == -1) | (result == 1))
        print(f"{op_id:2d}. {name:20s}: valid={valid}")

    print("\n" + "="*60)
