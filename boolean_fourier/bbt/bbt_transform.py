"""
BBT Phase A3: Banach-Butterfly Transform Pipeline
===================================================

Orchestrates the full BBT computation for a Boolean function:
    f → f̂ → {Inf_ℓ} → {p_ℓ} → {λ_ℓ} → {η_ℓ} → margin_product

The BBT profile is a complete description of how f interacts with the
Hadamard structure, layer by layer, in the influence-adapted Banach geometry.
"""

import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax.numpy as jnp
import numpy as np
from typing import Optional

from .influence import (
    fourier_coefficients,
    all_influences,
    total_influence,
    batch_fourier_coefficients,
    batch_all_influences,
    batch_total_influence,
)
from .banach_norms import (
    adaptive_exponents,
    banach_eigenvalues,
    all_contraction_factors,
    margin_product,
    margin_product_log2,
    margin_product_bounds,
    cumulative_margin,
)


# ============================================================================
# Single Function BBT Profile
# ============================================================================

def bbt_profile(truth_table: jnp.ndarray, n: int) -> dict:
    """
    Compute the full BBT profile for a single Boolean function.

    Args:
        truth_table: Values in {-1, +1}, length 2^n
        n: Number of variables

    Returns:
        Dict with all BBT quantities:
            fourier_coefficients: Array of 2^n f̂(S)
            influences: Array of n Inf_ℓ(f)
            total_influence: I(f)
            exponents: Array of n p_ℓ = 1 + Inf_ℓ
            eigenvalues: Array of n λ_ℓ = 2^{1/p_ℓ}
            contraction_factors: Array of n η_ℓ = 2^{-Inf_ℓ/(1+Inf_ℓ)}
            margin_product: μ₀ = ∏ η_ℓ
            margin_log2: log₂(μ₀)
            margin_bounds: (lower, upper) from total influence
            cumulative_margins: Array of n cumulative products
            parseval_check: Σf̂(S)² (should be 1.0 for Boolean functions)
    """
    f_hat = fourier_coefficients(truth_table, n)
    influences = all_influences(f_hat, n)
    total_inf = total_influence(f_hat, n)
    exponents = adaptive_exponents(influences)
    eigenvals = banach_eigenvalues(exponents)
    contractions = all_contraction_factors(influences)
    margin = margin_product(influences)
    margin_log = margin_product_log2(influences)
    bounds = margin_product_bounds(total_inf)
    cum_margins = cumulative_margin(influences)
    parseval = float(jnp.sum(f_hat ** 2))

    return {
        'n': n,
        'fourier_coefficients': f_hat.tolist(),
        'influences': influences.tolist(),
        'total_influence': float(total_inf),
        'exponents': exponents.tolist(),
        'eigenvalues': eigenvals.tolist(),
        'contraction_factors': contractions.tolist(),
        'margin_product': float(margin),
        'margin_log2': float(margin_log),
        'margin_bounds': [float(bounds[0]), float(bounds[1])],
        'cumulative_margins': cum_margins.tolist(),
        'parseval_check': float(parseval),
    }


def bbt_profile_from_coefficients(f_hat: jnp.ndarray, n: int) -> dict:
    """
    Compute BBT profile from pre-computed Fourier coefficients.
    Skips the FWHT step (useful when coefficients already available).
    """
    influences = all_influences(f_hat, n)
    total_inf = total_influence(f_hat, n)
    exponents = adaptive_exponents(influences)
    eigenvals = banach_eigenvalues(exponents)
    contractions = all_contraction_factors(influences)
    margin = margin_product(influences)
    margin_log = margin_product_log2(influences)
    bounds = margin_product_bounds(total_inf)
    cum_margins = cumulative_margin(influences)

    return {
        'n': n,
        'fourier_coefficients': f_hat.tolist(),
        'influences': influences.tolist(),
        'total_influence': float(total_inf),
        'exponents': exponents.tolist(),
        'eigenvalues': eigenvals.tolist(),
        'contraction_factors': contractions.tolist(),
        'margin_product': float(margin),
        'margin_log2': float(margin_log),
        'margin_bounds': [float(bounds[0]), float(bounds[1])],
        'cumulative_margins': cum_margins.tolist(),
        'parseval_check': float(jnp.sum(f_hat ** 2)),
    }


def bbt_layer_detail(f_hat: jnp.ndarray, n: int) -> list:
    """
    Per-layer breakdown of the BBT.

    Returns list of dicts, one per butterfly layer.
    """
    influences = all_influences(f_hat, n)
    layers = []
    for ell in range(n):
        inf = float(influences[ell])
        p = 1.0 + inf
        layers.append({
            'layer': ell,
            'influence': inf,
            'exponent': p,
            'eigenvalue': 2.0 ** (1.0 / p),
            'contraction': 2.0 ** (-inf / p),
            'regime': 'hilbert' if inf > 0.8 else ('banach' if inf < 0.2 else 'mixed'),
        })
    return layers


# ============================================================================
# Batch BBT (for exhaustive enumeration)
# ============================================================================

def batch_bbt_profiles(truth_tables: jnp.ndarray, n: int) -> dict:
    """
    Compute BBT profiles for a batch of functions.

    Args:
        truth_tables: Shape (num_functions, 2^n)
        n: Number of variables

    Returns:
        Dict of arrays:
            fourier_coefficients: (num_functions, 2^n)
            influences: (num_functions, n)
            total_influences: (num_functions,)
            exponents: (num_functions, n)
            margin_log2: (num_functions,)
    """
    f_hats = batch_fourier_coefficients(truth_tables, n)
    influences = batch_all_influences(f_hats, n)
    total_infs = batch_total_influence(f_hats, n)
    exponents = jnp.clip(1.0 + influences, 1.0, 2.0)

    # Vectorized margin product log2
    margin_logs = -jnp.sum(influences / (1.0 + influences), axis=1)

    return {
        'fourier_coefficients': f_hats,
        'influences': influences,
        'total_influences': total_infs,
        'exponents': exponents,
        'margin_log2': margin_logs,
    }


# ============================================================================
# Summary Statistics
# ============================================================================

def bbt_summary(profile: dict) -> str:
    """Human-readable summary of a BBT profile."""
    lines = [
        f"BBT Profile (n={profile['n']})",
        f"  Total influence: I(f) = {profile['total_influence']:.4f}",
        f"  Margin log₂(μ): {profile['margin_log2']:.4f}",
        f"  Margin product: μ = {profile['margin_product']:.6f}",
        f"  Margin bounds: [{profile['margin_bounds'][0]:.4f}, {profile['margin_bounds'][1]:.4f}]",
        f"  Parseval: Σf̂² = {profile['parseval_check']:.6f}",
        f"  Layer details:",
    ]
    for ell, (inf, p, eta) in enumerate(zip(
        profile['influences'], profile['exponents'], profile['contraction_factors']
    )):
        regime = 'H' if inf > 0.8 else ('B' if inf < 0.2 else 'M')
        lines.append(f"    ℓ={ell}: Inf={inf:.4f}, p={p:.4f}, η={eta:.4f} [{regime}]")
    return '\n'.join(lines)


# ============================================================================
# Main: Verification on canonical functions
# ============================================================================

if __name__ == "__main__":
    print("=== BBT Transform Verification ===\n")

    # Parity_3
    n = 3
    N = 1 << n
    parity_tt = jnp.array([
        (-1)**((i>>0)&1) * (-1)**((i>>1)&1) * (-1)**((i>>2)&1)
        for i in range(N)
    ], dtype=jnp.float32)

    profile = bbt_profile(parity_tt, n)
    print(bbt_summary(profile))
    print(f"  Expected: I(f)=3.0, log₂(μ)=-1.5")

    # Dictator x0
    print()
    dict_tt = jnp.array([(-1)**((i>>0)&1) for i in range(N)], dtype=jnp.float32)
    profile_dict = bbt_profile(dict_tt, n)
    print(bbt_summary(profile_dict))
    print(f"  Expected: I(f)=1.0, log₂(μ)=-0.5")

    # Majority_3: sign(x0 + x1 + x2)
    print()
    maj_tt = jnp.array([
        jnp.sign((-1)**((i>>0)&1) + (-1)**((i>>1)&1) + (-1)**((i>>2)&1))
        for i in range(N)
    ], dtype=jnp.float32)
    # sign(0) -> treat as +1
    maj_tt = jnp.where(maj_tt == 0, 1.0, maj_tt)
    profile_maj = bbt_profile(maj_tt, n)
    print(bbt_summary(profile_maj))

    # AND_3: TRUE only when all inputs are -1
    print()
    and3_tt = jnp.array([
        1.0 if all((-1)**((i>>b)&1) == -1 for b in range(3)) else -1.0
        for i in range(N)
    ], dtype=jnp.float32)
    profile_and3 = bbt_profile(and3_tt, n)
    print(bbt_summary(profile_and3))

    # Batch test
    print(f"\n=== Batch BBT Test (all 256 functions at n=3) ===")
    all_tts = jnp.array([
        [1 - 2 * ((func_id >> bit) & 1) for bit in range(N)]
        for func_id in range(1 << N)
    ], dtype=jnp.float32)
    batch_result = batch_bbt_profiles(all_tts, n)
    print(f"  Computed BBT for {len(all_tts)} functions")
    print(f"  Margin log₂ range: [{float(jnp.min(batch_result['margin_log2'])):.4f}, "
          f"{float(jnp.max(batch_result['margin_log2'])):.4f}]")
    print(f"  Total influence range: [{float(jnp.min(batch_result['total_influences'])):.4f}, "
          f"{float(jnp.max(batch_result['total_influences'])):.4f}]")
