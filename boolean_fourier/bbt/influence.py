"""
BBT Phase A1: Fourier Influence Computation
============================================

Computes coordinate influences Inf_ℓ(f) from Boolean Fourier coefficients.

    Inf_ℓ(f) = Σ_{S∋ℓ} f̂(S)²  =  Pr_x[f(x) ≠ f(x⊕e_ℓ)]

Key identity: I(f) = Σ_ℓ Inf_ℓ(f) = Σ_S |S| · f̂(S)²
"""

import os
import sys
import functools

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

# Import FWHT from existing infrastructure
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase5', 'exact'))
from fwht_benchmark import fwht_jax


# ============================================================================
# Fourier Coefficients
# ============================================================================

def fourier_coefficients(truth_table: jnp.ndarray, n: int) -> jnp.ndarray:
    """
    Compute normalized Fourier coefficients via FWHT.

    Args:
        truth_table: Values in {-1, +1}, length 2^n
        n: Number of variables

    Returns:
        Array of 2^n Fourier coefficients f̂(S), normalized so Σ f̂(S)² = 1
        for Boolean-valued functions.
    """
    return fwht_jax(truth_table, n)


# ============================================================================
# Coordinate Influence
# ============================================================================

def _build_membership_masks(n: int) -> jnp.ndarray:
    """
    Build binary mask matrix M of shape (n, 2^n) where M[ℓ, S] = 1 iff ℓ ∈ S.

    S is represented as an integer whose binary expansion encodes the subset.
    """
    N = 1 << n
    indices = jnp.arange(N, dtype=jnp.int32)
    masks = jnp.array([(indices >> ell) & 1 for ell in range(n)], dtype=jnp.float32)
    return masks  # shape (n, N)


def coordinate_influence(f_hat: jnp.ndarray, n: int, ell: int) -> float:
    """
    Compute Inf_ℓ(f) = Σ_{S: ℓ∈S} f̂(S)².

    Args:
        f_hat: Fourier coefficients, length 2^n
        n: Number of variables
        ell: Coordinate index (0-indexed)

    Returns:
        Inf_ℓ(f) ∈ [0, 1]
    """
    N = 1 << n
    mask = (jnp.arange(N, dtype=jnp.int32) >> ell) & 1
    return float(jnp.sum(f_hat ** 2 * mask))


def all_influences(f_hat: jnp.ndarray, n: int) -> jnp.ndarray:
    """
    Compute all coordinate influences simultaneously.

    Args:
        f_hat: Fourier coefficients, length 2^n
        n: Number of variables

    Returns:
        Array of shape (n,) with Inf_ℓ(f) for ℓ = 0, ..., n-1
    """
    masks = _build_membership_masks(n)  # (n, N)
    f_hat_sq = f_hat ** 2               # (N,)
    return masks @ f_hat_sq             # (n,)


def total_influence(f_hat: jnp.ndarray, n: int) -> float:
    """
    Compute total influence I(f) = Σ_ℓ Inf_ℓ(f) = Σ_S |S| · f̂(S)².

    Args:
        f_hat: Fourier coefficients, length 2^n
        n: Number of variables

    Returns:
        I(f) ∈ [0, n]
    """
    N = 1 << n
    indices = jnp.arange(N, dtype=jnp.int32)
    # popcount: number of set bits in each index
    popcounts = jnp.zeros(N, dtype=jnp.float32)
    for bit in range(n):
        popcounts = popcounts + ((indices >> bit) & 1).astype(jnp.float32)
    return float(jnp.sum(f_hat ** 2 * popcounts))


# ============================================================================
# Batch Operations (for exhaustive enumeration)
# ============================================================================

def batch_fourier_coefficients(truth_tables: jnp.ndarray, n: int) -> jnp.ndarray:
    """
    Compute Fourier coefficients for a batch of functions simultaneously.

    Args:
        truth_tables: Shape (num_functions, 2^n), values in {-1, +1}
        n: Number of variables

    Returns:
        Shape (num_functions, 2^n) of Fourier coefficients
    """
    return jax.vmap(lambda f: fwht_jax(f, n))(truth_tables)


def batch_all_influences(f_hats: jnp.ndarray, n: int) -> jnp.ndarray:
    """
    Compute all influences for a batch of functions.

    Args:
        f_hats: Shape (num_functions, 2^n) of Fourier coefficients
        n: Number of variables

    Returns:
        Shape (num_functions, n) of coordinate influences
    """
    masks = _build_membership_masks(n)  # (n, N)
    f_hats_sq = f_hats ** 2             # (num_functions, N)
    return f_hats_sq @ masks.T          # (num_functions, n)


def batch_total_influence(f_hats: jnp.ndarray, n: int) -> jnp.ndarray:
    """
    Compute total influence for a batch of functions.

    Args:
        f_hats: Shape (num_functions, 2^n) of Fourier coefficients
        n: Number of variables

    Returns:
        Shape (num_functions,) of total influences
    """
    N = 1 << n
    indices = jnp.arange(N, dtype=jnp.int32)
    popcounts = jnp.zeros(N, dtype=jnp.float32)
    for bit in range(n):
        popcounts = popcounts + ((indices >> bit) & 1).astype(jnp.float32)
    return jnp.sum(f_hats ** 2 * popcounts[None, :], axis=1)


# ============================================================================
# Exact Arithmetic (for Phase E non-algebrization verification)
# ============================================================================

def exact_influences(truth_table_np: np.ndarray, n: int):
    """
    Compute influences using exact rational arithmetic (fractions.Fraction).

    Args:
        truth_table_np: NumPy array of ints in {-1, +1}, length 2^n
        n: Number of variables

    Returns:
        List of fractions.Fraction, one per coordinate
    """
    from fractions import Fraction

    N = 1 << n
    f = [Fraction(int(x)) for x in truth_table_np]

    # Exact FWHT
    x = list(f)
    h = 1
    while h < N:
        for i in range(0, N, h * 2):
            for j in range(h):
                a, b = x[i + j], x[i + j + h]
                x[i + j] = a + b
                x[i + j + h] = a - b
        h *= 2
    f_hat = [c / N for c in x]

    # Compute influences
    influences = []
    for ell in range(n):
        inf_ell = Fraction(0)
        for S in range(N):
            if (S >> ell) & 1:
                inf_ell += f_hat[S] ** 2
        influences.append(inf_ell)

    return influences, f_hat


if __name__ == "__main__":
    # Quick verification
    print("=== BBT Influence Module Verification ===\n")

    # Parity_3: f(x) = x0 * x1 * x2
    n = 3
    N = 1 << n
    inputs = jnp.array([(-1)**((i>>0)&1) * (-1)**((i>>1)&1) * (-1)**((i>>2)&1)
                         for i in range(N)], dtype=jnp.float32)
    f_hat = fourier_coefficients(inputs, n)
    infs = all_influences(f_hat, n)
    total = total_influence(f_hat, n)
    print(f"Parity_3:")
    print(f"  Fourier coefficients: {f_hat}")
    print(f"  Influences: {infs}")
    print(f"  Total influence: {total:.4f} (expected: {n})")
    print(f"  Parseval check: Σf̂² = {float(jnp.sum(f_hat**2)):.4f} (expected: 1.0)")

    # Dictator x0
    inputs_dict = jnp.array([(-1)**((i>>0)&1) for i in range(N)], dtype=jnp.float32)
    f_hat_dict = fourier_coefficients(inputs_dict, n)
    infs_dict = all_influences(f_hat_dict, n)
    print(f"\nDictator x0:")
    print(f"  Fourier coefficients: {f_hat_dict}")
    print(f"  Influences: {infs_dict}")
    print(f"  Total influence: {float(jnp.sum(infs_dict)):.4f} (expected: 1.0)")

    # AND_2
    n2 = 2
    # AND in {-1,+1} encoding: AND(a,b)=1 iff a=1 AND b=1 (both FALSE)
    # Truth table: (-1,-1)->-1, (-1,1)->-1, (1,-1)->-1, (1,1)->1
    # With -1=TRUE, +1=FALSE: AND is TRUE when both TRUE = both -1
    and_tt = jnp.array([-1.0, -1.0, -1.0, 1.0])  # standard ordering
    # Actually let's use the paper convention: sign(-1+a+b+ab)
    and_tt2 = jnp.array([
        jnp.sign(-1 + (-1) + (-1) + (-1)*(-1)),  # a=-1,b=-1 -> sign(-1-1-1+1)=sign(-2)=-1
        jnp.sign(-1 + (-1) + (1) + (-1)*(1)),     # a=-1,b=1 -> sign(-1-1+1-1)=sign(-2)=-1
        jnp.sign(-1 + (1) + (-1) + (1)*(-1)),     # a=1,b=-1 -> sign(-1+1-1-1)=sign(-2)=-1
        jnp.sign(-1 + (1) + (1) + (1)*(1)),       # a=1,b=1 -> sign(-1+1+1+1)=sign(2)=1
    ])
    f_hat_and = fourier_coefficients(and_tt2, n2)
    infs_and = all_influences(f_hat_and, n2)
    print(f"\nAND_2:")
    print(f"  Truth table: {and_tt2}")
    print(f"  Fourier coefficients: {f_hat_and}")
    print(f"  Influences: {infs_and}")
    print(f"  Expected: Inf_0 = Inf_1 = 0.5")

    # Exact arithmetic check
    print(f"\n=== Exact Arithmetic Check (n=2) ===")
    exact_infs, exact_fhat = exact_influences(np.array(and_tt2, dtype=int), n2)
    print(f"  Exact f̂: {exact_fhat}")
    print(f"  Exact influences: {exact_infs}")
