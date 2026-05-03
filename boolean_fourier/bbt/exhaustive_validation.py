"""
BBT Phase B2: Exhaustive Validation — BBT × Sparsity Correlation
=================================================================

For ALL Boolean functions at n=3 (256) and n=4 (65536):
1. Compute BBT profiles (influences, margin product)
2. Find optimal ternary masks (brute-force at n=3, heuristic at n=4)
3. Correlate margin_log2 with mask sparsity

This is the central empirical test of the BBT framework.
"""

import os
import sys
import json
import time
from itertools import product as itertools_product
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats

from .bbt_transform import batch_bbt_profiles, bbt_profile
from .enumerate import all_boolean_functions, compute_npn_classes
from .influence import fourier_coefficients


# ============================================================================
# Shared Walsh Basis Matrix
# ============================================================================

def build_walsh_matrix(n: int) -> np.ndarray:
    """
    Build the N×N Walsh basis matrix H where H[i, S] = χ_S(x_i).

    χ_S(x) = ∏_{j∈S} x_j, with x encoded as x_j = 1 - 2*bit_j(i).

    Args:
        n: Number of variables

    Returns:
        Shape (N, N) float32 array where N = 2^n
    """
    N = 1 << n
    inputs = np.zeros((N, n), dtype=np.float32)
    for bit in range(n):
        inputs[:, bit] = 1.0 - 2.0 * ((np.arange(N) >> bit) & 1)

    H = np.ones((N, N), dtype=np.float32)
    for S in range(N):
        for bit in range(n):
            if (S >> bit) & 1:
                H[:, S] *= inputs[:, bit]
    return H


# ============================================================================
# Brute-Force Optimal Ternary Mask Search
# ============================================================================

def find_optimal_mask_bruteforce(truth_table: np.ndarray, n: int) -> tuple:
    """
    Find the sparsest ternary mask that perfectly represents a Boolean function.

    Tests all 3^{2^n} ternary masks. Only feasible for n ≤ 3.

    Args:
        truth_table: Values in {-1, +1}, length 2^n
        n: Number of variables

    Returns:
        (best_mask, accuracy, support_size, num_tested)
    """
    N = 1 << n
    inputs = np.zeros((N, n), dtype=np.float32)
    for bit in range(n):
        inputs[:, bit] = 1.0 - 2.0 * ((np.arange(N) >> bit) & 1)

    # Build Walsh basis matrix: H[i, S] = χ_S(x_i)
    H = np.ones((N, N), dtype=np.float32)
    for S in range(N):
        for bit in range(n):
            if (S >> bit) & 1:
                H[:, S] *= inputs[:, bit]

    tt = np.array(truth_table, dtype=np.float32)

    best_mask = None
    best_support = N + 1  # minimize support
    best_acc = 0.0
    num_tested = 0
    num_perfect = 0

    for mask_tuple in itertools_product([-1, 0, 1], repeat=N):
        mask = np.array(mask_tuple, dtype=np.float32)
        num_tested += 1

        # Skip all-zero mask
        support = int(np.sum(mask != 0))
        if support == 0:
            continue

        # Compute sign(H @ mask)
        output = H @ mask
        predicted = np.sign(output)
        predicted[predicted == 0] = 1.0  # tie-break to +1

        acc = np.mean(predicted == tt)

        if acc >= 0.9999:
            num_perfect += 1
            if support < best_support:
                best_support = support
                best_mask = mask.copy()
                best_acc = acc

    if best_mask is None:
        # No perfect mask found; return best accuracy
        return None, 0.0, N, num_tested

    return best_mask, best_acc, best_support, num_tested


def find_mask_heuristic(f_hat: np.ndarray, n: int, threshold: float = 0.1) -> tuple:
    """
    Heuristic ternary mask from thresholded Fourier coefficients.

    mask_S = sign(f̂(S)) if |f̂(S)| > threshold, else 0

    Args:
        f_hat: Fourier coefficients
        n: Number of variables
        threshold: Minimum |f̂(S)| to include in mask

    Returns:
        (mask, accuracy, support_size)
    """
    N = 1 << n
    mask = np.zeros(N, dtype=np.float32)

    for S in range(N):
        if abs(f_hat[S]) > threshold:
            mask[S] = np.sign(f_hat[S])

    # Verify accuracy
    inputs = np.zeros((N, n), dtype=np.float32)
    for bit in range(n):
        inputs[:, bit] = 1.0 - 2.0 * ((np.arange(N) >> bit) & 1)

    H = np.ones((N, N), dtype=np.float32)
    for S in range(N):
        for bit in range(n):
            if (S >> bit) & 1:
                H[:, S] *= inputs[:, bit]

    output = H @ mask
    predicted = np.sign(output)
    predicted[predicted == 0] = 1.0

    # Need truth table to compute accuracy
    support = int(np.sum(mask != 0))
    return mask, support


def verify_mask(mask: np.ndarray, truth_table: np.ndarray, n: int) -> float:
    """Verify a ternary mask against a truth table. Returns accuracy."""
    N = 1 << n
    inputs = np.zeros((N, n), dtype=np.float32)
    for bit in range(n):
        inputs[:, bit] = 1.0 - 2.0 * ((np.arange(N) >> bit) & 1)

    H = np.ones((N, N), dtype=np.float32)
    for S in range(N):
        for bit in range(n):
            if (S >> bit) & 1:
                H[:, S] *= inputs[:, bit]

    output = H @ mask
    predicted = np.sign(output)
    predicted[predicted == 0] = 1.0
    return float(np.mean(predicted == truth_table))


# ============================================================================
# Exhaustive Validation Pipeline
# ============================================================================

def run_exhaustive_n3(output_dir: str = None) -> dict:
    """
    Run exhaustive BBT × sparsity analysis for ALL 256 functions at n=3.

    Brute-force finds optimal ternary mask for every function.
    """
    n = 3
    N = 1 << n  # 8
    num_funcs = 1 << N  # 256

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== Exhaustive BBT Validation: n={n}, {num_funcs} functions ===\n")

    # Step 1: Compute all BBT profiles
    print("Step 1: Computing BBT profiles for all 256 functions...")
    t0 = time.time()
    all_tts = all_boolean_functions(n)
    batch_result = batch_bbt_profiles(all_tts, n)
    t1 = time.time()
    print(f"  BBT computed in {t1-t0:.3f}s")

    # Step 2: Find optimal masks via brute force
    print(f"Step 2: Brute-force mask search (3^{N} = {3**N} masks per function)...")
    t0 = time.time()

    all_tts_np = np.array(all_tts)
    masks = []
    support_sizes = []
    sparsities = []
    representable = []

    for func_id in range(num_funcs):
        if func_id % 50 == 0:
            print(f"  Function {func_id}/{num_funcs}...")

        mask, acc, support, num_tested = find_optimal_mask_bruteforce(
            all_tts_np[func_id], n
        )

        if mask is not None:
            masks.append(mask.tolist())
            support_sizes.append(support)
            sparsities.append(1.0 - support / N)
            representable.append(True)
        else:
            masks.append([0] * N)
            support_sizes.append(N)
            sparsities.append(0.0)
            representable.append(False)

    t1 = time.time()
    print(f"  Mask search completed in {t1-t0:.1f}s")
    print(f"  Representable: {sum(representable)}/{num_funcs}")

    # Step 3: Correlation analysis
    print("Step 3: Correlating BBT margin with sparsity...")

    margin_log2 = np.array(batch_result['margin_log2'])
    total_influences = np.array(batch_result['total_influences'])
    sparsities_arr = np.array(sparsities)
    support_arr = np.array(support_sizes)

    # Only correlate representable functions
    rep_mask = np.array(representable)
    if np.sum(rep_mask) > 2:
        r_margin_sparsity, p_margin_sparsity = stats.pearsonr(
            margin_log2[rep_mask], sparsities_arr[rep_mask]
        )
        rho_margin_sparsity, p_rho = stats.spearmanr(
            margin_log2[rep_mask], sparsities_arr[rep_mask]
        )
        r_inf_support, p_inf_support = stats.pearsonr(
            total_influences[rep_mask], support_arr[rep_mask].astype(float)
        )
    else:
        r_margin_sparsity = p_margin_sparsity = 0.0
        rho_margin_sparsity = p_rho = 0.0
        r_inf_support = p_inf_support = 0.0

    # Step 4: NPN class analysis
    print("Step 4: NPN class analysis...")
    classes = compute_npn_classes(n)
    npn_summary = {}
    for cid, members in classes.items():
        member_margins = [float(margin_log2[m]) for m in members]
        member_supports = [support_sizes[m] for m in members]
        member_sparsities = [sparsities[m] for m in members]
        npn_summary[str(cid)] = {
            'class_size': len(members),
            'margin_log2_mean': np.mean(member_margins),
            'margin_log2_std': np.std(member_margins),
            'support_mean': np.mean(member_supports),
            'sparsity_mean': np.mean(member_sparsities),
            'all_representable': all(representable[m] for m in members),
        }

    # Compile results
    results = {
        'n': n,
        'num_functions': num_funcs,
        'num_representable': int(sum(representable)),
        'correlation': {
            'pearson_margin_sparsity': float(r_margin_sparsity),
            'pearson_p_value': float(p_margin_sparsity),
            'spearman_margin_sparsity': float(rho_margin_sparsity),
            'spearman_p_value': float(p_rho),
            'pearson_influence_support': float(r_inf_support),
            'pearson_inf_support_p': float(p_inf_support),
        },
        'statistics': {
            'margin_log2_min': float(np.min(margin_log2)),
            'margin_log2_max': float(np.max(margin_log2)),
            'margin_log2_mean': float(np.mean(margin_log2)),
            'total_influence_min': float(np.min(total_influences)),
            'total_influence_max': float(np.max(total_influences)),
            'support_min': int(np.min(support_arr)),
            'support_max': int(np.max(support_arr)),
            'support_mean': float(np.mean(support_arr)),
            'sparsity_mean': float(np.mean(sparsities_arr)),
        },
        'npn_classes': npn_summary,
        'per_function': {
            'margin_log2': margin_log2.tolist(),
            'total_influence': total_influences.tolist(),
            'support_size': support_sizes,
            'sparsity': sparsities,
            'representable': representable,
            'masks': masks,
        },
    }

    # Save
    out_path = os.path.join(output_dir, f'exhaustive_n{n}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print summary
    print(f"\n=== RESULTS SUMMARY (n={n}) ===")
    print(f"  Functions representable: {results['num_representable']}/{num_funcs}")
    print(f"  Pearson(margin, sparsity): r={r_margin_sparsity:.4f}, p={p_margin_sparsity:.4e}")
    print(f"  Spearman(margin, sparsity): ρ={rho_margin_sparsity:.4f}, p={p_rho:.4e}")
    print(f"  Pearson(influence, support): r={r_inf_support:.4f}, p={p_inf_support:.4e}")
    print(f"  Margin log₂ range: [{results['statistics']['margin_log2_min']:.4f}, "
          f"{results['statistics']['margin_log2_max']:.4f}]")
    print(f"  Support range: [{results['statistics']['support_min']}, "
          f"{results['statistics']['support_max']}]")
    print(f"  Mean sparsity: {results['statistics']['sparsity_mean']:.4f}")

    return results


def run_exhaustive_n4_heuristic(output_dir: str = None) -> dict:
    """
    Run BBT analysis for all 65536 functions at n=4.
    Uses heuristic masks (brute force is infeasible: 3^16 ≈ 43M per function).
    """
    n = 4
    N = 1 << n  # 16
    num_funcs = 1 << N  # 65536

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== BBT Analysis: n={n}, {num_funcs} functions (heuristic masks) ===\n")

    # Step 1: Compute all BBT profiles
    print("Step 1: Computing BBT profiles...")
    t0 = time.time()
    all_tts = all_boolean_functions(n)
    batch_result = batch_bbt_profiles(all_tts, n)
    t1 = time.time()
    print(f"  BBT computed in {t1-t0:.3f}s")

    # Step 2: Heuristic masks via thresholded Fourier coefficients
    print("Step 2: Computing heuristic ternary masks...")
    t0 = time.time()

    f_hats = np.array(batch_result['fourier_coefficients'])
    all_tts_np = np.array(all_tts)

    support_sizes = []
    sparsities = []
    accuracies = []

    for func_id in range(num_funcs):
        if func_id % 10000 == 0:
            print(f"  Function {func_id}/{num_funcs}...")

        # Heuristic mask: sign of large-enough Fourier coefficients
        f_hat = f_hats[func_id]
        mask = np.zeros(N, dtype=np.float32)
        for S in range(N):
            if abs(f_hat[S]) > 0.05:
                mask[S] = np.sign(f_hat[S])

        support = int(np.sum(mask != 0))
        support_sizes.append(support)
        sparsities.append(1.0 - support / N)

        # Verify accuracy
        acc = verify_mask(mask, all_tts_np[func_id], n)
        accuracies.append(acc)

    t1 = time.time()
    print(f"  Heuristic masks computed in {t1-t0:.1f}s")

    # Step 3: Correlation
    margin_log2 = np.array(batch_result['margin_log2'])
    total_influences = np.array(batch_result['total_influences'])
    sparsities_arr = np.array(sparsities)
    support_arr = np.array(support_sizes, dtype=np.float32)
    acc_arr = np.array(accuracies)

    # Correlate only for functions where heuristic achieves 100%
    perfect_mask = acc_arr >= 0.9999
    num_perfect = int(np.sum(perfect_mask))

    if num_perfect > 2:
        r_margin_sparsity, p_val = stats.pearsonr(
            margin_log2[perfect_mask], sparsities_arr[perfect_mask]
        )
        rho, p_rho = stats.spearmanr(
            margin_log2[perfect_mask], sparsities_arr[perfect_mask]
        )
    else:
        r_margin_sparsity = p_val = rho = p_rho = 0.0

    results = {
        'n': n,
        'num_functions': num_funcs,
        'mask_method': 'heuristic_thresholded_fourier',
        'num_perfect_heuristic': num_perfect,
        'heuristic_accuracy_mean': float(np.mean(acc_arr)),
        'correlation': {
            'pearson_margin_sparsity': float(r_margin_sparsity),
            'pearson_p_value': float(p_val),
            'spearman_margin_sparsity': float(rho),
            'spearman_p_value': float(p_rho),
        },
        'statistics': {
            'margin_log2_min': float(np.min(margin_log2)),
            'margin_log2_max': float(np.max(margin_log2)),
            'margin_log2_mean': float(np.mean(margin_log2)),
            'total_influence_min': float(np.min(total_influences)),
            'total_influence_max': float(np.max(total_influences)),
            'support_mean': float(np.mean(support_arr)),
            'sparsity_mean': float(np.mean(sparsities_arr)),
        },
    }

    out_path = os.path.join(output_dir, f'exhaustive_n{n}_heuristic.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print(f"\n=== RESULTS SUMMARY (n={n}) ===")
    print(f"  Perfect heuristic masks: {num_perfect}/{num_funcs} ({100*num_perfect/num_funcs:.1f}%)")
    print(f"  Mean heuristic accuracy: {results['heuristic_accuracy_mean']:.4f}")
    print(f"  Pearson(margin, sparsity): r={r_margin_sparsity:.4f}, p={p_val:.4e}")
    print(f"  Spearman(margin, sparsity): ρ={rho:.4f}, p={p_rho:.4e}")

    return results


def run_exhaustive_n4_with_repair(output_dir: str = None) -> dict:
    """
    Full n=4 pipeline: heuristic masks + repair loop + cancellation + NPN.
    Supersedes run_exhaustive_n4_heuristic.
    """
    from .repair import repair_batch
    from .cancellation_index import batch_cancellation_profiles, cancellation_vs_properties

    n = 4
    N = 1 << n  # 16
    num_funcs = 1 << N  # 65536

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== BBT n={n}: Heuristic + Repair + Cancellation + NPN ===\n")

    # Step 1: BBT profiles
    print("Step 1: Computing BBT profiles for all 65536 functions...")
    t0 = time.time()
    all_tts_jnp = all_boolean_functions(n)
    batch_result = batch_bbt_profiles(all_tts_jnp, n)
    all_tts = np.array(all_tts_jnp)
    f_hats = np.array(batch_result['fourier_coefficients'])
    margin_log2 = np.array(batch_result['margin_log2'])
    total_influences = np.array(batch_result['total_influences'])
    print(f"  Done in {time.time()-t0:.2f}s")

    # Step 2: Vectorized heuristic masks
    print("Step 2: Generating heuristic masks (vectorized)...")
    t0 = time.time()
    masks_heuristic = (np.sign(f_hats) * (np.abs(f_hats) > 0.05)).astype(np.float32)

    # Vectorized accuracy check
    H = build_walsh_matrix(n)
    outputs = (H @ masks_heuristic.T).T
    predicted = np.sign(outputs)
    predicted[predicted == 0] = 1.0
    acc_heuristic = np.mean(predicted == all_tts, axis=1)
    heuristic_perfect = int(np.sum(acc_heuristic >= 0.9999))
    print(f"  Heuristic perfect: {heuristic_perfect}/{num_funcs} ({100*heuristic_perfect/num_funcs:.1f}%)")
    print(f"  Done in {time.time()-t0:.2f}s")

    # Step 3: Repair loop
    print("Step 3: Running repair loop...")
    repair_result = repair_batch(masks_heuristic, all_tts, n, f_hats=f_hats, max_iterations=200, verbose=True)
    masks_repaired = repair_result['repaired_masks']
    acc_repaired = repair_result['accuracies_post']
    post_repair_perfect = repair_result['post_repair_perfect']

    # Support sizes post-repair
    support_sizes = np.sum(masks_repaired != 0, axis=1).astype(np.float32)
    sparsities = 1.0 - support_sizes / N

    # Step 4: Cancellation index (before and after repair)
    print("Step 4: Computing cancellation indices...")
    t0 = time.time()
    cancel_pre = batch_cancellation_profiles(masks_heuristic, n)
    cancel_post = batch_cancellation_profiles(masks_repaired, n)
    print(f"  Pre-repair ρ: mean={np.mean(cancel_pre['overall_rho']):.4f}")
    print(f"  Post-repair ρ: mean={np.mean(cancel_post['overall_rho']):.4f}")
    print(f"  Done in {time.time()-t0:.2f}s")

    # Step 5: Correlations
    print("Step 5: Correlation analysis...")
    perfect_mask = acc_repaired >= 0.9999

    if np.sum(perfect_mask) > 2:
        r_margin_sparsity, p_ms = stats.pearsonr(margin_log2[perfect_mask], sparsities[perfect_mask])
        rho_ms, p_rho_ms = stats.spearmanr(margin_log2[perfect_mask], sparsities[perfect_mask])
        r_cancel_acc, p_ca = stats.pearsonr(cancel_post['overall_rho'], acc_repaired)
    else:
        r_margin_sparsity = p_ms = rho_ms = p_rho_ms = r_cancel_acc = p_ca = 0.0

    print(f"  Pearson(margin, sparsity) [perfect only]: r={r_margin_sparsity:.4f}")
    print(f"  Pearson(cancellation, accuracy): r={r_cancel_acc:.4f}")

    # Step 6: NPN stratification
    print("Step 6: Computing NPN classes (this may take ~60s)...")
    t0 = time.time()
    npn_classes = compute_npn_classes(n)
    print(f"  Found {len(npn_classes)} NPN classes in {time.time()-t0:.1f}s")

    npn_summary = {}
    for cid, members in npn_classes.items():
        m_idx = np.array(members)
        npn_summary[str(cid)] = {
            'class_size': len(members),
            'heuristic_perfect_rate': float(np.mean(acc_heuristic[m_idx] >= 0.9999)),
            'post_repair_perfect_rate': float(np.mean(acc_repaired[m_idx] >= 0.9999)),
            'mean_margin_log2': float(np.mean(margin_log2[m_idx])),
            'mean_support': float(np.mean(support_sizes[m_idx])),
            'mean_cancellation_rho': float(np.mean(cancel_post['overall_rho'][m_idx])),
            'mean_total_influence': float(np.mean(total_influences[m_idx])),
        }

    # Compile results
    results = {
        'n': n,
        'num_functions': num_funcs,
        'heuristic_perfect': heuristic_perfect,
        'post_repair_perfect': post_repair_perfect,
        'repair_statistics': {
            'num_repaired': repair_result['num_repaired'],
            'num_failed': repair_result['num_failed'],
            'total_time_seconds': repair_result['total_time'],
        },
        'cancellation': {
            'pre_repair_mean_rho': float(np.mean(cancel_pre['overall_rho'])),
            'post_repair_mean_rho': float(np.mean(cancel_post['overall_rho'])),
            'correlation_rho_accuracy': float(r_cancel_acc),
        },
        'correlation': {
            'pearson_margin_sparsity': float(r_margin_sparsity),
            'pearson_p_value': float(p_ms),
            'spearman_margin_sparsity': float(rho_ms),
            'spearman_p_value': float(p_rho_ms),
        },
        'statistics': {
            'margin_log2_min': float(np.min(margin_log2)),
            'margin_log2_max': float(np.max(margin_log2)),
            'margin_log2_mean': float(np.mean(margin_log2)),
            'support_mean': float(np.mean(support_sizes)),
            'sparsity_mean': float(np.mean(sparsities)),
        },
        'npn_classes': npn_summary,
    }

    out_path = os.path.join(output_dir, 'exhaustive_n4_repaired.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"=== FINAL RESULTS (n={n}) ===")
    print(f"{'='*60}")
    print(f"  Heuristic perfect: {heuristic_perfect}/{num_funcs} ({100*heuristic_perfect/num_funcs:.1f}%)")
    print(f"  Post-repair perfect: {post_repair_perfect}/{num_funcs} ({100*post_repair_perfect/num_funcs:.1f}%)")
    print(f"  Repair success: {repair_result['num_repaired']}, failures: {repair_result['num_failed']}")
    print(f"  Cancellation ρ: pre={np.mean(cancel_pre['overall_rho']):.4f}, post={np.mean(cancel_post['overall_rho']):.4f}")
    print(f"  Pearson(margin, sparsity): r={r_margin_sparsity:.4f}")
    print(f"  NPN classes: {len(npn_classes)}")
    print(f"  Saved to {out_path}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BBT Exhaustive Validation")
    parser.add_argument('--n', type=int, default=3, choices=[3, 4],
                       help='Number of variables (3 or 4)')
    parser.add_argument('--repair', action='store_true',
                       help='Use repair pipeline for n=4')
    args = parser.parse_args()

    if args.n == 3:
        run_exhaustive_n3()
    elif args.repair:
        run_exhaustive_n4_with_repair()
    else:
        run_exhaustive_n4_heuristic()
