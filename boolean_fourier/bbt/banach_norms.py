"""
BBT Phase A2: Banach Space Norms and Operator Theory
=====================================================

Closed-form ℓ_p operator norms for the 2×2 butterfly matrix A = [[1,1],[1,-1]].

Key results (for p ∈ [1,2]):
    ‖A‖_{p→p}   = 2^{1/p}
    ‖A⁻¹‖_{p→p} = 2^{-1+1/p}

The adaptive Banach exponent is p_ℓ = 1 + Inf_ℓ(f), giving:
    - Inf_ℓ ≈ 0 (insensitive): p_ℓ ≈ 1, inverse is isometry (margin preserved)
    - Inf_ℓ ≈ 1 (sensitive):   p_ℓ ≈ 2, inverse contracts by 1/√2 (Hilbert case)
"""

import jax.numpy as jnp
import numpy as np


# ============================================================================
# Adaptive Banach Exponents
# ============================================================================

def adaptive_exponent(influence: float) -> float:
    """
    Compute the Banach exponent p_ℓ = 1 + Inf_ℓ(f), clamped to [1, 2].

    Args:
        influence: Inf_ℓ(f) ∈ [0, 1]

    Returns:
        p_ℓ ∈ [1, 2]
    """
    return float(jnp.clip(1.0 + influence, 1.0, 2.0))


def adaptive_exponents(influences: jnp.ndarray) -> jnp.ndarray:
    """Vectorized: p_ℓ = clip(1 + Inf_ℓ, 1, 2) for all coordinates."""
    return jnp.clip(1.0 + influences, 1.0, 2.0)


# ============================================================================
# 2×2 Butterfly Operator Norms (Analytical)
# ============================================================================

def butterfly_operator_norm(p: float) -> float:
    """
    Operator norm ‖A‖_{p→p} for A = [[1,1],[1,-1]].

    For p ∈ [1, 2]: ‖A‖_{p→p} = 2^{1/p}
    For p ∈ [2, ∞]: ‖A‖_{p→p} = 2^{1-1/p}

    Args:
        p: ℓ_p exponent, p ≥ 1

    Returns:
        ‖A‖_{p→p}
    """
    if p <= 2:
        return 2.0 ** (1.0 / p)
    else:
        return 2.0 ** (1.0 - 1.0 / p)


def butterfly_inverse_contraction(p: float) -> float:
    """
    Operator norm of A⁻¹ = (1/2)A in ℓ_p.

    ‖A⁻¹‖_{p→p} = (1/2) · ‖A‖_{p→p} = 2^{-1+1/p}

    For p ∈ [1, 2]:
        p=1: contraction = 1.0 (isometry, margin fully preserved)
        p=2: contraction = 1/√2 ≈ 0.707 (Hilbert case, margin halves per layer)

    Args:
        p: ℓ_p exponent, p ∈ [1, 2]

    Returns:
        η_ℓ = ‖A⁻¹‖_{p→p}
    """
    return 2.0 ** (-1.0 + 1.0 / p)


def butterfly_inverse_contraction_from_influence(influence: float) -> float:
    """
    Contraction factor directly from influence.

    η_ℓ = 2^{-Inf_ℓ / (1 + Inf_ℓ)}

    Args:
        influence: Inf_ℓ(f) ∈ [0, 1]

    Returns:
        η_ℓ ∈ [1/√2, 1]
    """
    p = 1.0 + influence
    return 2.0 ** (-influence / p)


# ============================================================================
# Banach Eigenvalues
# ============================================================================

def banach_eigenvalue(p: float) -> float:
    """
    Banach eigenvalue: the operator norm of A in ℓ_p.

    λ_ℓ^(B) = ‖A‖_{p→p} = 2^{1/p_ℓ}

    In ℓ_2 (Hilbert): λ = √2 (universal, independent of f)
    In adaptive ℓ_p:  λ = 2^{1/p_ℓ} (depends on f through Inf_ℓ)

    Args:
        p: Banach exponent p_ℓ = 1 + Inf_ℓ(f)

    Returns:
        λ_ℓ^(B) ∈ [√2, 2]
    """
    return 2.0 ** (1.0 / p)


def banach_eigenvalues(exponents: jnp.ndarray) -> jnp.ndarray:
    """Vectorized Banach eigenvalues for all layers."""
    return 2.0 ** (1.0 / exponents)


# ============================================================================
# Margin Product (Core BBT Quantity)
# ============================================================================

def margin_product(influences: jnp.ndarray) -> float:
    """
    Banach margin product: ∏_ℓ η_ℓ = ∏_ℓ 2^{-Inf_ℓ/(1+Inf_ℓ)}

    This is the central BBT quantity. If margin_product is polynomially bounded
    (not exponentially small), sparse ternary PTFs should exist.

    Args:
        influences: Array of shape (n,) with Inf_ℓ(f)

    Returns:
        μ₀ = ∏_ℓ 2^{-Inf_ℓ/(1+Inf_ℓ)}
    """
    return float(2.0 ** margin_product_log2(influences))


def margin_product_log2(influences: jnp.ndarray) -> float:
    """
    Log₂ of the margin product (numerically stable).

    log₂(μ₀) = -Σ_ℓ Inf_ℓ / (1 + Inf_ℓ)

    Bounds (from GPT-5.4 Section 4.2):
        -I(f) ≤ log₂(μ₀) ≤ -I(f)/2

    where I(f) = Σ_ℓ Inf_ℓ(f) is the total influence.

    Args:
        influences: Array of shape (n,) with Inf_ℓ(f)

    Returns:
        log₂(μ₀) ∈ [-n, 0]
    """
    return float(-jnp.sum(influences / (1.0 + influences)))


def margin_product_bounds(total_inf: float) -> tuple:
    """
    Analytical bounds on log₂(margin_product) from total influence.

    Returns (lower_bound, upper_bound) for log₂(μ₀).
    """
    return (-total_inf, -total_inf / 2.0)


# ============================================================================
# Contraction Factor Vectors
# ============================================================================

def all_contraction_factors(influences: jnp.ndarray) -> jnp.ndarray:
    """
    Compute contraction factors for all layers.

    η_ℓ = 2^{-Inf_ℓ/(1+Inf_ℓ)} for each ℓ.

    Args:
        influences: Shape (n,)

    Returns:
        Shape (n,) of contraction factors ∈ [1/√2, 1]
    """
    return 2.0 ** (-influences / (1.0 + influences))


def cumulative_margin(influences: jnp.ndarray) -> jnp.ndarray:
    """
    Cumulative margin product from layer n down to layer ℓ.

    margin[ℓ] = ∏_{k=ℓ}^{n-1} η_k

    Useful for understanding where margin is lost.

    Args:
        influences: Shape (n,)

    Returns:
        Shape (n,) where margin[ℓ] is the margin remaining after layers ℓ..n-1
    """
    contractions = all_contraction_factors(influences)
    # Reverse cumulative product (from last layer backward)
    log_contractions = jnp.log2(contractions)
    cumsum_reverse = jnp.cumsum(log_contractions[::-1])[::-1]
    return 2.0 ** cumsum_reverse


# ============================================================================
# Numerical Verification of Analytical Formulas
# ============================================================================

def verify_operator_norm_numerically(p: float, n_samples: int = 10000) -> dict:
    """
    Numerically verify ‖A‖_{p→p} by sampling unit ℓ_p vectors.

    Args:
        p: ℓ_p exponent
        n_samples: Number of random directions to test

    Returns:
        Dict with analytical and numerical norm estimates
    """
    A = np.array([[1, 1], [1, -1]], dtype=np.float64)

    # Sample on ℓ_p unit sphere: (cos θ, sin θ) normalized in ℓ_p
    thetas = np.linspace(0, 2 * np.pi, n_samples)
    max_ratio = 0.0

    for theta in thetas:
        v = np.array([np.cos(theta), np.sin(theta)])
        # Normalize to ℓ_p unit ball
        v_norm = np.sum(np.abs(v) ** p) ** (1.0 / p)
        if v_norm < 1e-10:
            continue
        v = v / v_norm

        Av = A @ v
        Av_norm = np.sum(np.abs(Av) ** p) ** (1.0 / p)
        ratio = Av_norm  # since ‖v‖_p = 1

        if ratio > max_ratio:
            max_ratio = ratio

    analytical = butterfly_operator_norm(p)
    return {
        'p': p,
        'analytical': analytical,
        'numerical': max_ratio,
        'relative_error': abs(max_ratio - analytical) / analytical,
    }


if __name__ == "__main__":
    print("=== BBT Banach Norms Verification ===\n")

    # Check operator norms at key points
    print("Butterfly operator norms ‖A‖_{p→p}:")
    for p in [1.0, 1.25, 1.5, 1.75, 2.0]:
        result = verify_operator_norm_numerically(p)
        print(f"  p={p:.2f}: analytical={result['analytical']:.6f}, "
              f"numerical={result['numerical']:.6f}, "
              f"error={result['relative_error']:.2e}")

    print(f"\nInverse contraction factors ‖A⁻¹‖_{{p→p}}:")
    for p in [1.0, 1.25, 1.5, 1.75, 2.0]:
        eta = butterfly_inverse_contraction(p)
        print(f"  p={p:.2f}: η = {eta:.6f}")

    print(f"\nMargin products for canonical functions:")
    # Parity_3: all influences = 1
    infs_parity = jnp.array([1.0, 1.0, 1.0])
    m_parity = margin_product_log2(infs_parity)
    print(f"  Parity_3: log₂(μ) = {m_parity:.4f} (expected: -1.5)")

    # Dictator: Inf_0=1, rest=0
    infs_dict = jnp.array([1.0, 0.0, 0.0])
    m_dict = margin_product_log2(infs_dict)
    print(f"  Dictator: log₂(μ) = {m_dict:.4f} (expected: -0.5)")

    # Constant function: all influences = 0
    infs_const = jnp.array([0.0, 0.0, 0.0])
    m_const = margin_product_log2(infs_const)
    print(f"  Constant: log₂(μ) = {m_const:.4f} (expected: 0.0)")

    # Check bounds
    print(f"\nMargin bounds check:")
    for name, infs in [("Parity_3", infs_parity), ("Dictator", infs_dict)]:
        total_inf = float(jnp.sum(infs))
        log2_margin = margin_product_log2(infs)
        lb, ub = margin_product_bounds(total_inf)
        in_bounds = lb <= log2_margin <= ub
        print(f"  {name}: I(f)={total_inf:.2f}, log₂(μ)={log2_margin:.4f}, "
              f"bounds=[{lb:.4f}, {ub:.4f}], valid={in_bounds}")
