"""
BBT Track 1: Per-Layer Butterfly Cancellation Index
=====================================================

Measures destructive interference within butterfly pairs of the WHT.

For each butterfly layer ℓ, pairs (i, j) are coupled by A = [[1,1],[1,-1]].
The cancellation ratio ρ measures how close the pair is to destructive
interference:

    ρ(w_i, w_j) = min(|w_i + w_j|, |w_i - w_j|) / (|w_i| + |w_j| + ε)

    (1, 0) → ρ = 1.0  (clean: one path, no cancellation)
    (1, 1) → ρ = 0.0  (difference cancels)
    (1,-1) → ρ = 0.0  (sum cancels)
    (0, 0) → ρ = 0.0  (both inactive)

Higher ρ = less cancellation = more robust margin propagation.
"""

import os
import json
import numpy as np
from typing import List, Dict, Tuple

from .eigenvector_alignment import butterfly_pair_indices


# ============================================================================
# Pair Cancellation
# ============================================================================

def pair_cancellation(w_i: float, w_j: float, eps: float = 1e-10) -> float:
    """
    Cancellation ratio for a single butterfly pair.

    ρ = min(|w_i + w_j|, |w_i - w_j|) / (|w_i| + |w_j| + ε)
    """
    denom = abs(w_i) + abs(w_j) + eps
    return min(abs(w_i + w_j), abs(w_i - w_j)) / denom


# ============================================================================
# Layer and Full Cancellation Profiles
# ============================================================================

def layer_cancellation_index(mask: np.ndarray, n: int, layer: int) -> dict:
    """
    Compute cancellation index for one butterfly layer.

    Args:
        mask: Ternary mask, length 2^n
        n: Number of variables
        layer: Butterfly layer (0-indexed)

    Returns:
        Dict with rho_median, rho_mean, num_active_pairs, num_cancelling
    """
    pairs = butterfly_pair_indices(n, layer)
    rhos = []
    num_active = 0
    num_cancelling = 0

    for i, j in pairs:
        rho = pair_cancellation(mask[i], mask[j])
        rhos.append(rho)
        if abs(mask[i]) + abs(mask[j]) > 0.5:
            num_active += 1
            if rho < 0.1:
                num_cancelling += 1

    rhos = np.array(rhos)
    return {
        'layer': layer,
        'rho_median': float(np.median(rhos)),
        'rho_mean': float(np.mean(rhos)),
        'num_pairs': len(pairs),
        'num_active_pairs': num_active,
        'num_cancelling_pairs': num_cancelling,
    }


def full_cancellation_profile(mask: np.ndarray, n: int) -> dict:
    """
    Cancellation profile across all butterfly layers.
    """
    layers = [layer_cancellation_index(mask, n, ell) for ell in range(n)]
    layer_medians = [l['rho_median'] for l in layers]
    overall = float(np.mean(layer_medians))
    worst = int(np.argmin(layer_medians))

    return {
        'per_layer': layers,
        'overall_rho': overall,
        'worst_layer': worst,
        'profile_vector': layer_medians,
    }


# ============================================================================
# Vectorized Batch Computation
# ============================================================================

def batch_cancellation_profiles(masks: np.ndarray, n: int, eps: float = 1e-10) -> dict:
    """
    Compute cancellation profiles for a batch of functions. Fully vectorized.

    Args:
        masks: Shape (num_funcs, N) ternary masks
        n: Number of variables
        eps: Small constant to avoid division by zero

    Returns:
        Dict with:
            layer_rho_medians: (num_funcs, n) per-layer median ρ
            overall_rho: (num_funcs,) mean of per-layer medians
            worst_layer: (num_funcs,) index of layer with lowest ρ
    """
    num_funcs, N = masks.shape
    num_layers = n
    layer_rho_medians = np.zeros((num_funcs, num_layers), dtype=np.float64)
    layer_rho_means = np.zeros((num_funcs, num_layers), dtype=np.float64)

    for ell in range(num_layers):
        pairs = butterfly_pair_indices(n, ell)
        i_indices = np.array([p[0] for p in pairs])
        j_indices = np.array([p[1] for p in pairs])

        # Fancy index: extract paired values for all functions at once
        w_i = masks[:, i_indices]  # (num_funcs, num_pairs)
        w_j = masks[:, j_indices]  # (num_funcs, num_pairs)

        sums = np.abs(w_i + w_j)
        diffs = np.abs(w_i - w_j)
        denoms = np.abs(w_i) + np.abs(w_j) + eps

        rho = np.minimum(sums, diffs) / denoms  # (num_funcs, num_pairs)

        layer_rho_medians[:, ell] = np.median(rho, axis=1)
        layer_rho_means[:, ell] = np.mean(rho, axis=1)

    overall_rho = np.mean(layer_rho_medians, axis=1)
    worst_layer = np.argmin(layer_rho_medians, axis=1)

    return {
        'layer_rho_medians': layer_rho_medians,
        'layer_rho_means': layer_rho_means,
        'overall_rho': overall_rho,
        'worst_layer': worst_layer,
    }


# ============================================================================
# Correlation Analysis
# ============================================================================

def cancellation_vs_properties(
    masks: np.ndarray,
    n: int,
    accuracies: np.ndarray = None,
    support_sizes: np.ndarray = None,
    margin_log2: np.ndarray = None,
) -> dict:
    """
    Correlate cancellation index with mask accuracy, support, and margin.
    """
    from scipy import stats

    profiles = batch_cancellation_profiles(masks, n)
    overall_rho = profiles['overall_rho']

    results = {'num_functions': len(masks)}

    if accuracies is not None:
        r, p = stats.pearsonr(overall_rho, accuracies)
        rho_s, p_s = stats.spearmanr(overall_rho, accuracies)
        results['vs_accuracy'] = {
            'pearson_r': float(r), 'pearson_p': float(p),
            'spearman_rho': float(rho_s), 'spearman_p': float(p_s),
        }

    if support_sizes is not None:
        r, p = stats.pearsonr(overall_rho, support_sizes)
        results['vs_support'] = {'pearson_r': float(r), 'pearson_p': float(p)}

    if margin_log2 is not None:
        r, p = stats.pearsonr(overall_rho, margin_log2)
        results['vs_margin'] = {'pearson_r': float(r), 'pearson_p': float(p)}

    results['rho_stats'] = {
        'mean': float(np.mean(overall_rho)),
        'std': float(np.std(overall_rho)),
        'min': float(np.min(overall_rho)),
        'max': float(np.max(overall_rho)),
    }

    return results


if __name__ == "__main__":
    # Validate on n=3 ground truth
    results_path = os.path.join(os.path.dirname(__file__), 'results', 'exhaustive_n3.json')

    if os.path.exists(results_path):
        with open(results_path) as f:
            data = json.load(f)

        n = data['n']
        masks = np.array(data['per_function']['masks'], dtype=np.float32)
        support_sizes = np.array(data['per_function']['support_size'], dtype=np.float32)
        margin_log2 = np.array(data['per_function']['margin_log2'], dtype=np.float64)

        print(f"=== Cancellation Index Validation (n={n}) ===\n")

        # Sanity checks
        print("Pair cancellation sanity:")
        print(f"  (1, 0) → ρ = {pair_cancellation(1, 0):.4f} (expected: 1.0)")
        print(f"  (1, 1) → ρ = {pair_cancellation(1, 1):.4f} (expected: 0.0)")
        print(f"  (1,-1) → ρ = {pair_cancellation(1, -1):.4f} (expected: 0.0)")
        print(f"  (0, 0) → ρ = {pair_cancellation(0, 0):.4f} (expected: 0.0)")

        # Batch computation
        print(f"\nComputing cancellation for {len(masks)} functions...")
        profiles = batch_cancellation_profiles(masks, n)
        overall = profiles['overall_rho']
        print(f"  Overall ρ: mean={np.mean(overall):.4f}, std={np.std(overall):.4f}")
        print(f"  Range: [{np.min(overall):.4f}, {np.max(overall):.4f}]")

        # Correlations
        corr = cancellation_vs_properties(
            masks, n,
            support_sizes=support_sizes,
            margin_log2=margin_log2,
        )
        print(f"\nCorrelation with support: r={corr['vs_support']['pearson_r']:.4f}")
        print(f"Correlation with margin: r={corr['vs_margin']['pearson_r']:.4f}")

        # Save
        out_path = os.path.join(os.path.dirname(__file__), 'results', 'cancellation_n3.json')
        save_data = {
            'n': n,
            'rho_stats': corr['rho_stats'],
            'vs_support': corr.get('vs_support'),
            'vs_margin': corr.get('vs_margin'),
            'per_layer_stats': {
                f'layer_{ell}': {
                    'rho_median_mean': float(np.mean(profiles['layer_rho_medians'][:, ell])),
                    'rho_median_std': float(np.std(profiles['layer_rho_medians'][:, ell])),
                }
                for ell in range(n)
            },
        }
        with open(out_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        print(f"\nSaved to {out_path}")
    else:
        print("Run exhaustive_validation.py --n 3 first.")
