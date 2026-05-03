"""
BBT Phase D: Scaling to n=16 and Beyond
=========================================

Tests BBT on named function families at larger n:
- Parity: I(f) = n, margin = 2^{-n/2} (exponential decay)
- Majority: I(f) = Θ(√n), margin = 2^{-Θ(√n)} (subexponential)
- Dictator: I(f) = 1, margin = 2^{-0.5} (constant)
- Tribes: I(f) = Θ(log n), margin polynomial

Also implements the O(N log N) BBT-guided ternary recovery algorithm.
"""

import os
import sys
import json
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
from math import comb

from .bbt_transform import bbt_profile, bbt_summary
from .influence import fourier_coefficients, all_influences
from .banach_norms import margin_product_log2, adaptive_exponents


# ============================================================================
# Named Function Truth Tables
# ============================================================================

def parity_truth_table(n: int) -> jnp.ndarray:
    """Parity function: f(x) = ∏ x_i. Single Fourier coefficient f̂([n]) = ±1."""
    N = 1 << n
    tt = jnp.ones(N, dtype=jnp.float32)
    for i in range(N):
        val = 1.0
        for bit in range(n):
            val *= (1.0 - 2.0 * ((i >> bit) & 1))
        tt = tt.at[i].set(val)
    return tt


def majority_truth_table(n: int) -> jnp.ndarray:
    """Majority function: f(x) = sign(Σ x_i). Requires odd n."""
    N = 1 << n
    tt = jnp.zeros(N, dtype=jnp.float32)
    for i in range(N):
        total = 0.0
        for bit in range(n):
            total += (1.0 - 2.0 * ((i >> bit) & 1))
        val = jnp.sign(total)
        # sign(0) → +1 (tie-break)
        val = jnp.where(val == 0, 1.0, val)
        tt = tt.at[i].set(val)
    return tt


def dictator_truth_table(n: int, coord: int = 0) -> jnp.ndarray:
    """Dictator function: f(x) = x_{coord}."""
    N = 1 << n
    tt = jnp.array([
        1.0 - 2.0 * ((i >> coord) & 1)
        for i in range(N)
    ], dtype=jnp.float32)
    return tt


def and_truth_table(n: int) -> jnp.ndarray:
    """AND function: f(x) = 1 iff all x_i = -1 (TRUE in our encoding)."""
    N = 1 << n
    tt = jnp.array([
        -1.0 if all(((i >> b) & 1) == 1 for b in range(n)) else 1.0
        for i in range(N)
    ], dtype=jnp.float32)
    # Actually AND is TRUE (=-1) only when all inputs are -1 (TRUE)
    # In {-1,+1}: all -1 means i = 2^n - 1 (all bits set)
    # Correcting: output -1 when i = 0 (all bits 0, meaning all x_i = +1... no)
    # Let's be precise: x_bit = 1 - 2*(bit), so bit=0 → x=+1(FALSE), bit=1 → x=-1(TRUE)
    # AND is TRUE when all x_i = -1 = TRUE, i.e., all bits = 1, i.e., i = 2^n - 1
    tt2 = jnp.ones(N, dtype=jnp.float32)  # default FALSE (+1)
    tt2 = tt2.at[N - 1].set(-1.0)  # only all-TRUE input gives TRUE output
    return tt2


def tribes_truth_table(n: int, w: int = None) -> jnp.ndarray:
    """
    Tribes function: OR of ANDs of disjoint groups of w variables.
    w ≈ log₂(n) gives balanced tribes (I(f) ≈ Θ(log n / n) per variable).

    f(x) = OR(AND(group_1), AND(group_2), ..., AND(group_{n/w}))
    """
    if w is None:
        w = max(1, int(np.log2(max(n, 2))))

    N = 1 << n
    num_tribes = n // w

    tt = jnp.ones(N, dtype=jnp.float32)  # default FALSE
    for i in range(N):
        # Check if any tribe has all TRUE
        for t in range(num_tribes):
            tribe_start = t * w
            tribe_end = min(tribe_start + w, n)
            all_true = all(((i >> b) & 1) == 1 for b in range(tribe_start, tribe_end))
            if all_true:
                tt = tt.at[i].set(-1.0)  # TRUE
                break

    return tt


def threshold_truth_table(n: int, k: int) -> jnp.ndarray:
    """Threshold function: f(x) = 1 iff at least k inputs are TRUE (-1)."""
    N = 1 << n
    tt = jnp.ones(N, dtype=jnp.float32)
    for i in range(N):
        num_true = bin(i).count('1')  # count -1 inputs
        if num_true >= k:
            tt = tt.at[i].set(-1.0)
    return tt


NAMED_FUNCTIONS = {
    'parity': parity_truth_table,
    'majority': majority_truth_table,
    'dictator': dictator_truth_table,
    'and': and_truth_table,
    'tribes': tribes_truth_table,
}


# ============================================================================
# Scaling Experiment
# ============================================================================

def scaling_experiment(
    n_values: list = None,
    function_names: list = None,
    output_dir: str = None
) -> dict:
    """
    Compute BBT profiles for named functions across multiple n values.
    Track how margin_log2, total_influence, and per-layer structure scale.
    """
    if n_values is None:
        n_values = [3, 5, 7, 9, 11, 13, 15]
    if function_names is None:
        function_names = ['parity', 'majority', 'dictator', 'and', 'tribes']
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== BBT Scaling Experiment ===")
    print(f"  Functions: {function_names}")
    print(f"  n values: {n_values}\n")

    results = {}

    for name in function_names:
        results[name] = {'n_values': [], 'margin_log2': [], 'total_influence': [],
                         'mean_influence': [], 'max_influence': [], 'min_influence': []}

        for n in n_values:
            if name == 'majority' and n % 2 == 0:
                n = n + 1  # majority needs odd n

            tt_fn = NAMED_FUNCTIONS[name]
            tt = tt_fn(n)

            t0 = time.time()
            profile = bbt_profile(tt, n)
            dt = time.time() - t0

            results[name]['n_values'].append(n)
            results[name]['margin_log2'].append(profile['margin_log2'])
            results[name]['total_influence'].append(profile['total_influence'])
            results[name]['mean_influence'].append(
                float(np.mean(profile['influences']))
            )
            results[name]['max_influence'].append(float(np.max(profile['influences'])))
            results[name]['min_influence'].append(float(np.min(profile['influences'])))

            print(f"  {name}(n={n}): I(f)={profile['total_influence']:.4f}, "
                  f"log₂(μ)={profile['margin_log2']:.4f}, time={dt:.3f}s")

        print()

    # Save
    out_path = os.path.join(output_dir, 'scaling_experiment.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")

    # Summary
    print(f"\n=== SCALING SUMMARY ===")
    for name in function_names:
        ns = results[name]['n_values']
        ms = results[name]['margin_log2']
        Is = results[name]['total_influence']
        print(f"\n  {name}:")
        for n, m, I in zip(ns, ms, Is):
            print(f"    n={n:2d}: I(f)={I:8.4f}, log₂(μ)={m:8.4f}")

    return results


# ============================================================================
# BBT-Guided Ternary Recovery Algorithm
# ============================================================================

def ternary_recovery_bbt(truth_table: jnp.ndarray, n: int) -> dict:
    """
    O(N log N) BBT-guided ternary mask recovery.

    Algorithm:
    1. Compute Fourier coefficients via FWHT
    2. Compute influences → adaptive exponents
    3. For each layer (backward): use Banach contraction to prune coefficients
    4. Threshold remaining to ternary

    This is the EMPIRICAL test of the conjectured recovery algorithm.
    It may not work for all functions (the proof is open).

    Args:
        truth_table: Values in {-1, +1}, length 2^n
        n: Number of variables

    Returns:
        Dict with recovered mask, accuracy, and diagnostics
    """
    N = 1 << n
    f_hat = np.array(fourier_coefficients(truth_table, n))
    influences = np.array(all_influences(jnp.array(f_hat), n))

    # Approach 1: Adaptive thresholding based on layer structure
    # For each Fourier coefficient S, compute its "Banach weight":
    # the product of contraction factors for layers NOT in S
    # (coefficients in S "pass through" those layers without penalty)

    banach_weights = np.ones(N, dtype=np.float64)
    for ell in range(n):
        eta = 2.0 ** (-influences[ell] / (1.0 + influences[ell]))
        for S in range(N):
            if not ((S >> ell) & 1):
                # Coefficient S does not involve variable ell
                # → it goes through the "contraction" direction at layer ell
                banach_weights[S] *= eta

    # Weighted coefficients: f̂(S) * banach_weight(S)
    weighted = np.abs(f_hat) * banach_weights

    # Adaptive threshold: keep coefficients whose weighted magnitude exceeds threshold
    threshold = np.median(weighted[weighted > 0]) * 0.5 if np.any(weighted > 0) else 0.0

    mask = np.zeros(N, dtype=np.float32)
    for S in range(N):
        if weighted[S] > threshold:
            mask[S] = np.sign(f_hat[S])

    # Verify
    inputs_np = np.zeros((N, n), dtype=np.float32)
    for bit in range(n):
        inputs_np[:, bit] = 1.0 - 2.0 * ((np.arange(N) >> bit) & 1)

    H = np.ones((N, N), dtype=np.float32)
    for S in range(N):
        for bit in range(n):
            if (S >> bit) & 1:
                H[:, S] *= inputs_np[:, bit]

    output = H @ mask
    predicted = np.sign(output)
    predicted[predicted == 0] = 1.0
    tt_np = np.array(truth_table)
    accuracy = float(np.mean(predicted == tt_np))

    support = int(np.sum(mask != 0))

    return {
        'mask': mask.tolist(),
        'accuracy': accuracy,
        'support_size': support,
        'sparsity': 1.0 - support / N,
        'banach_weights': banach_weights.tolist(),
        'threshold_used': float(threshold),
        'n': n,
    }


# ============================================================================
# Recovery Comparison
# ============================================================================

def compare_recovery_methods(n: int, function_name: str) -> dict:
    """
    Compare BBT-guided recovery with naive Fourier thresholding.
    """
    tt = NAMED_FUNCTIONS[function_name](n)

    # Method 1: BBT-guided
    t0 = time.time()
    bbt_result = ternary_recovery_bbt(tt, n)
    bbt_time = time.time() - t0

    # Method 2: Naive Fourier thresholding
    t0 = time.time()
    f_hat = np.array(fourier_coefficients(tt, n))
    naive_mask = np.zeros(1 << n, dtype=np.float32)
    for S in range(1 << n):
        if abs(f_hat[S]) > 0.1:
            naive_mask[S] = np.sign(f_hat[S])

    # Verify naive
    N = 1 << n
    inputs_np = np.zeros((N, n), dtype=np.float32)
    for bit in range(n):
        inputs_np[:, bit] = 1.0 - 2.0 * ((np.arange(N) >> bit) & 1)
    H = np.ones((N, N), dtype=np.float32)
    for S in range(N):
        for bit in range(n):
            if (S >> bit) & 1:
                H[:, S] *= inputs_np[:, bit]
    output_naive = H @ naive_mask
    pred_naive = np.sign(output_naive)
    pred_naive[pred_naive == 0] = 1.0
    naive_acc = float(np.mean(pred_naive == np.array(tt)))
    naive_support = int(np.sum(naive_mask != 0))
    naive_time = time.time() - t0

    return {
        'function': function_name,
        'n': n,
        'bbt': {
            'accuracy': bbt_result['accuracy'],
            'support': bbt_result['support_size'],
            'sparsity': bbt_result['sparsity'],
            'time': bbt_time,
        },
        'naive': {
            'accuracy': naive_acc,
            'support': naive_support,
            'sparsity': 1.0 - naive_support / N,
            'time': naive_time,
        },
    }


# ============================================================================
# Track 2: Cancellation Scaling Experiment
# ============================================================================

def cancellation_scaling_experiment(
    n_values: list = None,
    function_names: list = None,
    output_dir: str = None,
) -> dict:
    """
    Compute BBT profiles + cancellation index + recovery accuracy for named
    functions at increasing n. Tests whether low ρ predicts recovery failure.
    """
    from .cancellation_index import full_cancellation_profile
    from .exhaustive_validation import build_walsh_matrix

    if n_values is None:
        n_values = [3, 5, 7, 9, 11]  # n≤11 keeps H matrix < 16MB
    if function_names is None:
        function_names = ['parity', 'majority', 'dictator', 'tribes']
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== Track 2: Cancellation Scaling Experiment ===\n")

    results = {}

    for name in function_names:
        results[name] = {
            'n_values': [], 'margin_log2': [], 'total_influence': [],
            'heuristic_accuracy': [], 'overall_rho': [],
            'per_layer_rho': [], 'heuristic_support': [],
        }

        for n in n_values:
            if name == 'majority' and n % 2 == 0:
                n = n + 1

            N = 1 << n
            tt_fn = NAMED_FUNCTIONS[name]
            tt = tt_fn(n)
            tt_np = np.array(tt)

            # BBT profile
            profile = bbt_profile(tt, n)

            # Heuristic mask
            f_hat = np.array(profile['fourier_coefficients'])
            mask = (np.sign(f_hat) * (np.abs(f_hat) > 0.05)).astype(np.float32)
            support = int(np.sum(mask != 0))

            # Verify accuracy
            H = build_walsh_matrix(n)
            output = H @ mask
            predicted = np.sign(output)
            predicted[predicted == 0] = 1.0
            acc = float(np.mean(predicted == tt_np))

            # Cancellation profile
            cancel = full_cancellation_profile(mask, n)

            results[name]['n_values'].append(n)
            results[name]['margin_log2'].append(profile['margin_log2'])
            results[name]['total_influence'].append(profile['total_influence'])
            results[name]['heuristic_accuracy'].append(acc)
            results[name]['overall_rho'].append(cancel['overall_rho'])
            results[name]['per_layer_rho'].append(cancel['profile_vector'])
            results[name]['heuristic_support'].append(support)

            print(f"  {name}(n={n}): I={profile['total_influence']:.3f}, "
                  f"log₂μ={profile['margin_log2']:.3f}, "
                  f"acc={acc:.3f}, ρ={cancel['overall_rho']:.3f}, "
                  f"support={support}/{N}")

        print()

    # Save
    out_path = os.path.join(output_dir, 'cancellation_scaling.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")

    # Summary: does low ρ predict low accuracy?
    print(f"\n=== CANCELLATION → ACCURACY PREDICTION ===")
    all_rho = []
    all_acc = []
    for name in function_names:
        for rho, acc in zip(results[name]['overall_rho'],
                            results[name]['heuristic_accuracy']):
            all_rho.append(rho)
            all_acc.append(acc)

    from scipy import stats
    if len(all_rho) > 2:
        r, p = stats.pearsonr(all_rho, all_acc)
        print(f"  Pearson(ρ, accuracy): r={r:.4f}, p={p:.4e}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--track2', action='store_true', help='Run cancellation scaling')
    args = parser.parse_args()

    if args.track2:
        cancellation_scaling_experiment()
    else:
        results = scaling_experiment(
            n_values=[3, 5, 7, 9, 11, 13, 15],
            function_names=['parity', 'majority', 'dictator', 'and', 'tribes'],
        )

        print(f"\n=== BBT Recovery Test (n=3) ===")
        for name in ['parity', 'majority', 'dictator', 'and']:
            comp = compare_recovery_methods(3, name)
            print(f"  {name}: BBT acc={comp['bbt']['accuracy']:.2f} "
                  f"(support={comp['bbt']['support']}), "
                  f"Naive acc={comp['naive']['accuracy']:.2f} "
                  f"(support={comp['naive']['support']})")
