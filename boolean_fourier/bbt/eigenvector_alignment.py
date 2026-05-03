"""
BBT Phase C: Eigenvector-Sparsity Alignment Test
==================================================

Tests whether Banach co-eigenvectors predict ternary mask support.

At each butterfly layer ℓ, the 2×2 operator A = [[1,1],[1,-1]] acts in ℓ_{p_ℓ}.
The "co-eigenvector" (norm maximizer) and "annihilator" (orthogonal direction
in the Banach dual) predict which Fourier coefficients should be zero.

Hypothesis: mask zeros align with co-eigenvector annihilators at high-influence layers.
"""

import os
import json
import numpy as np
import jax.numpy as jnp
from typing import List, Tuple, Dict

from .influence import fourier_coefficients, all_influences
from .banach_norms import adaptive_exponent


# ============================================================================
# Butterfly Index Structure
# ============================================================================

def butterfly_pair_indices(n: int, layer: int) -> List[Tuple[int, int]]:
    """
    Get the pairs of Fourier coefficient indices coupled at butterfly layer ℓ.

    At stage ℓ (0-indexed), the butterfly couples indices (i, i + 2^ℓ)
    for all i where bit ℓ is not set.

    Args:
        n: Number of variables
        layer: Butterfly layer (0-indexed)

    Returns:
        List of (i, j) pairs where j = i + 2^layer
    """
    N = 1 << n
    step = 1 << layer
    pairs = []
    for i in range(N):
        if not ((i >> layer) & 1):  # bit ℓ not set in i
            j = i | step  # set bit ℓ → j = i + 2^ℓ
            pairs.append((i, j))
    return pairs


# ============================================================================
# Banach Norm Maximizers and Annihilators for 2×2 Butterfly
# ============================================================================

def banach_norm_maximizer_2x2(p: float, n_grid: int = 10000) -> np.ndarray:
    """
    Find the vector on the ℓ_p unit ball that maximizes ‖Av‖_p
    for A = [[1,1],[1,-1]].

    For p=1: v = (1, 0) or (0, 1)
    For p=2: v = (1/√2, 1/√2)
    For p ∈ (1,2): interpolates smoothly

    Args:
        p: ℓ_p exponent
        n_grid: Grid resolution for optimization

    Returns:
        2D vector on the ℓ_p unit ball
    """
    A = np.array([[1, 1], [1, -1]], dtype=np.float64)

    # Parameterize ℓ_p unit ball in first quadrant (by symmetry, maximizer is there)
    # v = (t^{1/p}, (1-t)^{1/p}) for t ∈ [0, 1] with ‖v‖_p = 1
    best_ratio = 0.0
    best_v = np.array([1.0, 0.0])

    for i in range(n_grid + 1):
        t = i / n_grid
        # v such that |v1|^p + |v2|^p = 1
        v1 = t ** (1.0 / p)
        v2 = (1.0 - t) ** (1.0 / p)
        v = np.array([v1, v2])

        Av = A @ v
        Av_norm = np.sum(np.abs(Av) ** p) ** (1.0 / p)

        if Av_norm > best_ratio:
            best_ratio = Av_norm
            best_v = v.copy()

    return best_v


def banach_annihilator_direction(p: float) -> np.ndarray:
    """
    Direction in the Banach dual that annihilates the co-eigenvector.

    In ℓ_p, the duality map J_p(v)_i = |v_i|^{p-2} * v_i * ‖v‖^{2-p}
    The annihilator of e in X_ℓ is {w : J_p(e)(w) = 0}.

    For a 2D space, if the co-eigenvector is e = (e1, e2),
    the annihilator is spanned by the vector perpendicular to J_p(e).

    Returns:
        2D direction vector (normalized in ℓ_2 for convenience)
    """
    e = banach_norm_maximizer_2x2(p)

    # J_p(e) = (|e1|^{p-2} * e1, |e2|^{p-2} * e2)  (up to scaling)
    J = np.array([
        np.abs(e[0]) ** (p - 2) * e[0] if e[0] != 0 else 0.0,
        np.abs(e[1]) ** (p - 2) * e[1] if e[1] != 0 else 0.0,
    ])

    # Perpendicular to J in 2D
    annihilator = np.array([-J[1], J[0]])
    norm = np.linalg.norm(annihilator)
    if norm > 1e-10:
        annihilator /= norm

    return annihilator


# ============================================================================
# Alignment Scoring
# ============================================================================

def pair_zero_prediction(
    mask: np.ndarray,
    pair: Tuple[int, int],
    p: float,
    threshold: float = 0.3
) -> dict:
    """
    For a single butterfly pair (i,j), predict which coefficient should be zero
    based on the Banach annihilator direction, and check against actual mask.

    Args:
        mask: Ternary mask array
        pair: (i, j) indices
        p: Banach exponent at this layer
        threshold: Annihilator component magnitude threshold for prediction

    Returns:
        Dict with prediction and actual zero pattern
    """
    i, j = pair
    ann = banach_annihilator_direction(p)

    # Annihilator direction tells us the "suppressed" direction in (mask[i], mask[j]) space
    # If ann[0] >> ann[1]: mask[i] should be close to zero
    # If ann[1] >> ann[0]: mask[j] should be close to zero

    predicted_i_zero = abs(ann[0]) > abs(ann[1]) + threshold
    predicted_j_zero = abs(ann[1]) > abs(ann[0]) + threshold

    actual_i_zero = (mask[i] == 0)
    actual_j_zero = (mask[j] == 0)

    return {
        'pair': (i, j),
        'annihilator': ann.tolist(),
        'predicted_i_zero': bool(predicted_i_zero),
        'predicted_j_zero': bool(predicted_j_zero),
        'actual_i_zero': bool(actual_i_zero),
        'actual_j_zero': bool(actual_j_zero),
        'match': (predicted_i_zero == actual_i_zero) and (predicted_j_zero == actual_j_zero),
    }


def layer_alignment_score(
    mask: np.ndarray,
    influences: np.ndarray,
    n: int,
    layer: int
) -> dict:
    """
    Compute alignment score for a single butterfly layer.

    Args:
        mask: Ternary mask, length 2^n
        influences: Coordinate influences, length n
        n: Number of variables
        layer: Butterfly layer index

    Returns:
        Dict with alignment statistics
    """
    p = adaptive_exponent(float(influences[layer]))
    pairs = butterfly_pair_indices(n, layer)

    predictions = [pair_zero_prediction(mask, pair, p) for pair in pairs]
    num_matches = sum(1 for pred in predictions if pred['match'])
    num_pairs = len(pairs)

    return {
        'layer': layer,
        'influence': float(influences[layer]),
        'exponent': p,
        'num_pairs': num_pairs,
        'num_matches': num_matches,
        'alignment_score': num_matches / num_pairs if num_pairs > 0 else 0.0,
    }


def full_alignment_score(
    mask: np.ndarray,
    truth_table: np.ndarray,
    n: int
) -> dict:
    """
    Compute alignment across all butterfly layers for a single function.
    """
    f_hat = np.array(fourier_coefficients(jnp.array(truth_table), n))
    influences = np.array(all_influences(jnp.array(f_hat), n))

    layers = []
    for ell in range(n):
        layer_result = layer_alignment_score(mask, influences, n, ell)
        layers.append(layer_result)

    # Weighted average: weight by influence (high-influence layers matter more)
    total_weight = sum(l['influence'] for l in layers)
    if total_weight > 0:
        weighted_score = sum(
            l['alignment_score'] * l['influence'] for l in layers
        ) / total_weight
    else:
        weighted_score = 0.0

    unweighted_score = np.mean([l['alignment_score'] for l in layers])

    return {
        'layers': layers,
        'weighted_alignment': float(weighted_score),
        'unweighted_alignment': float(unweighted_score),
    }


def run_alignment_analysis(
    n: int,
    masks: List[np.ndarray],
    truth_tables: np.ndarray,
    output_dir: str = None
) -> dict:
    """
    Run alignment analysis across all functions.

    Args:
        n: Number of variables
        masks: List of ternary masks (one per function)
        truth_tables: Shape (num_funcs, 2^n)
        output_dir: Where to save results

    Returns:
        Summary statistics
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== Eigenvector-Sparsity Alignment Analysis (n={n}) ===\n")

    weighted_scores = []
    unweighted_scores = []

    for func_id in range(len(masks)):
        if func_id % 50 == 0:
            print(f"  Function {func_id}/{len(masks)}...")

        mask = np.array(masks[func_id], dtype=np.float32)
        if np.sum(mask != 0) == 0:
            continue

        result = full_alignment_score(mask, truth_tables[func_id], n)
        weighted_scores.append(result['weighted_alignment'])
        unweighted_scores.append(result['unweighted_alignment'])

    results = {
        'n': n,
        'num_functions_analyzed': len(weighted_scores),
        'weighted_alignment_mean': float(np.mean(weighted_scores)),
        'weighted_alignment_std': float(np.std(weighted_scores)),
        'weighted_alignment_median': float(np.median(weighted_scores)),
        'unweighted_alignment_mean': float(np.mean(unweighted_scores)),
        'unweighted_alignment_std': float(np.std(unweighted_scores)),
    }

    out_path = os.path.join(output_dir, f'alignment_n{n}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n=== ALIGNMENT RESULTS (n={n}) ===")
    print(f"  Weighted alignment: {results['weighted_alignment_mean']:.4f} ± {results['weighted_alignment_std']:.4f}")
    print(f"  Unweighted alignment: {results['unweighted_alignment_mean']:.4f} ± {results['unweighted_alignment_std']:.4f}")
    print(f"  Saved to {out_path}")

    return results


if __name__ == "__main__":
    # Run on n=3 using exhaustive results
    import json as json_mod

    results_path = os.path.join(os.path.dirname(__file__), 'results', 'exhaustive_n3.json')
    if os.path.exists(results_path):
        with open(results_path) as f:
            data = json_mod.load(f)

        n = data['n']
        masks = data['per_function']['masks']
        truth_tables = np.array(
            [[1.0 - 2.0 * ((fid >> bit) & 1) for bit in range(1 << n)]
             for fid in range(1 << (1 << n))],
            dtype=np.float32
        )

        results = run_alignment_analysis(n, masks, truth_tables)
    else:
        print("Run exhaustive_validation.py --n 3 first to generate data.")
