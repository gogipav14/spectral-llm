"""
BBT Track 1: Deterministic Ternary Mask Repair
================================================

Perceptron-style repair loop: given a heuristic ternary mask that doesn't
achieve 100% accuracy, iteratively fix violations by modifying one
coefficient at a time using the violation gradient.

Algorithm:
    1. Compute violations V = {i : f_i * (Hw)_i <= 0}
    2. Violation gradient: delta[S] = f[V] @ H[V, S]
    3. Update: add, flip, or remove the best coefficient
    4. Repeat until perfect or max iterations
"""

import os
import time
import numpy as np
from typing import Optional

from .exhaustive_validation import build_walsh_matrix


# ============================================================================
# Core Repair Functions
# ============================================================================

def compute_violations(H: np.ndarray, mask: np.ndarray, truth_table: np.ndarray) -> np.ndarray:
    """
    Find indices where sign(H @ mask) != truth_table.

    Returns array of violation indices (may be empty if mask is perfect).
    """
    output = H @ mask
    predicted = np.sign(output)
    predicted[predicted == 0] = 1.0
    return np.where(predicted != truth_table)[0]


def violation_gradient(H: np.ndarray, truth_table: np.ndarray, viol_idx: np.ndarray) -> np.ndarray:
    """
    Compute delta[S] = sum_{i in violations} f[i] * H[i, S].

    This is the "vote" of all violated inputs for how each coefficient should change.
    Positive delta[S] means violations want mask[S] to be positive.

    Single matrix operation: delta = truth_table[viol] @ H[viol, :]
    """
    if len(viol_idx) == 0:
        return np.zeros(H.shape[1], dtype=np.float32)
    return truth_table[viol_idx] @ H[viol_idx, :]


def _greedy_repair(
    mask: np.ndarray,
    truth_table: np.ndarray,
    H: np.ndarray,
    max_iterations: int = 100,
) -> tuple:
    """
    Single greedy repair pass: flip the coefficient with highest |delta|
    among those that would actually reduce violations.
    """
    N = len(mask)
    mask = mask.copy()
    best_mask = mask.copy()
    best_viol = N + 1

    for iteration in range(max_iterations):
        viol_idx = compute_violations(H, mask, truth_table)
        if len(viol_idx) == 0:
            return mask, True, iteration
        if len(viol_idx) < best_viol:
            best_viol = len(viol_idx)
            best_mask = mask.copy()

        delta = violation_gradient(H, truth_table, viol_idx)

        # Try all possible single-coordinate ternary updates, pick the one
        # that most reduces violations
        best_update = None
        best_new_viol = len(viol_idx)

        for S in np.argsort(-np.abs(delta))[:8]:  # top 8 candidates
            for new_val in [-1.0, 0.0, 1.0]:
                if new_val == mask[S]:
                    continue
                old_val = mask[S]
                mask[S] = new_val
                new_viol = len(compute_violations(H, mask, truth_table))
                if new_viol < best_new_viol:
                    best_new_viol = new_viol
                    best_update = (S, new_val)
                mask[S] = old_val

        if best_update is None:
            break
        mask[best_update[0]] = best_update[1]

    return best_mask, False, max_iterations


def repair_single(
    mask: np.ndarray,
    truth_table: np.ndarray,
    H: np.ndarray,
    f_hat: np.ndarray = None,
    max_iterations: int = 200,
) -> tuple:
    """
    Multi-start repair: try several initialization strategies and pick
    the one that converges (or has fewest violations).

    Strategies:
    1. Greedy repair from the heuristic mask
    2. Start from sign(f_hat) with various thresholds
    3. Start from all-ones mask (full support)
    4. Start from violation-gradient-derived mask
    """
    N = len(mask)
    best_mask = mask.copy()
    best_viol = len(compute_violations(H, mask, truth_table))

    if best_viol == 0:
        return mask, True, 0, {}

    # Strategy 1: Greedy from heuristic
    m1, conv1, iters1 = _greedy_repair(mask, truth_table, H, max_iterations // 4)
    v1 = len(compute_violations(H, m1, truth_table))
    if v1 == 0:
        return m1, True, iters1, {'strategy': 'greedy_heuristic'}
    if v1 < best_viol:
        best_viol = v1
        best_mask = m1.copy()

    # Strategy 2: Try different thresholds on f_hat
    if f_hat is not None:
        for threshold in [0.01, 0.1, 0.2, 0.3]:
            start = (np.sign(f_hat) * (np.abs(f_hat) > threshold)).astype(np.float32)
            m2, conv2, iters2 = _greedy_repair(start, truth_table, H, max_iterations // 4)
            v2 = len(compute_violations(H, m2, truth_table))
            if v2 == 0:
                return m2, True, iters2, {'strategy': f'threshold_{threshold}'}
            if v2 < best_viol:
                best_viol = v2
                best_mask = m2.copy()

    # Strategy 3: Build mask from violation gradient starting from zero
    delta_full = truth_table @ H  # gradient from ALL inputs as "violations"
    sorted_S = np.argsort(-np.abs(delta_full))
    for k in range(1, N + 1):
        start = np.zeros(N, dtype=np.float32)
        for S in sorted_S[:k]:
            start[S] = np.sign(delta_full[S])
        v = len(compute_violations(H, start, truth_table))
        if v == 0:
            return start, True, 0, {'strategy': f'gradient_top{k}'}

    # Strategy 4: Greedy from best so far
    m4, conv4, iters4 = _greedy_repair(best_mask, truth_table, H, max_iterations // 2)
    v4 = len(compute_violations(H, m4, truth_table))
    if v4 == 0:
        return m4, True, iters4, {'strategy': 'greedy_best'}
    if v4 < best_viol:
        best_mask = m4.copy()

    return best_mask, False, max_iterations, {'strategy': 'failed'}


def repair_batch(
    masks: np.ndarray,
    truth_tables: np.ndarray,
    n: int,
    f_hats: np.ndarray = None,
    max_iterations: int = 200,
    verbose: bool = True,
) -> dict:
    """
    Repair all imperfect masks in a batch.

    Args:
        masks: Shape (num_funcs, N) initial ternary masks
        truth_tables: Shape (num_funcs, N) in {-1, +1}
        n: Number of variables
        max_iterations: Max iterations per function

    Returns:
        Dict with repaired_masks, converged array, iteration counts, timing
    """
    num_funcs, N = masks.shape
    H = build_walsh_matrix(n)

    # Identify imperfect masks
    outputs = (H @ masks.T).T  # (num_funcs, N)
    predicted = np.sign(outputs)
    predicted[predicted == 0] = 1.0
    accuracies = np.mean(predicted == truth_tables, axis=1)
    imperfect_idx = np.where(accuracies < 0.9999)[0]

    if verbose:
        print(f"  Repairing {len(imperfect_idx)} imperfect masks out of {num_funcs}...")

    repaired_masks = masks.copy()
    converged = accuracies >= 0.9999  # already perfect
    iterations = np.zeros(num_funcs, dtype=np.int32)

    t0 = time.time()
    num_repaired = 0
    num_failed = 0

    for count, func_id in enumerate(imperfect_idx):
        if verbose and count % 2000 == 0:
            print(f"    {count}/{len(imperfect_idx)}...")

        fh = f_hats[func_id] if f_hats is not None else None
        mask, conv, iters, diag = repair_single(
            repaired_masks[func_id],
            truth_tables[func_id],
            H,
            f_hat=fh,
            max_iterations=max_iterations,
        )
        repaired_masks[func_id] = mask
        converged[func_id] = conv
        iterations[func_id] = iters

        if conv:
            num_repaired += 1
        else:
            num_failed += 1

    total_time = time.time() - t0

    # Post-repair verification
    outputs_post = (H @ repaired_masks.T).T
    predicted_post = np.sign(outputs_post)
    predicted_post[predicted_post == 0] = 1.0
    accuracies_post = np.mean(predicted_post == truth_tables, axis=1)

    perfect_post = int(np.sum(accuracies_post >= 0.9999))

    if verbose:
        print(f"  Repair complete in {total_time:.1f}s")
        print(f"  Pre-repair perfect: {int(np.sum(accuracies >= 0.9999))}/{num_funcs}")
        print(f"  Post-repair perfect: {perfect_post}/{num_funcs}")
        print(f"  Repaired: {num_repaired}, Failed: {num_failed}")

    return {
        'repaired_masks': repaired_masks,
        'converged': converged,
        'iterations': iterations,
        'accuracies_post': accuracies_post,
        'num_repaired': num_repaired,
        'num_failed': num_failed,
        'pre_repair_perfect': int(np.sum(accuracies >= 0.9999)),
        'post_repair_perfect': perfect_post,
        'total_time': total_time,
    }


if __name__ == "__main__":
    # Quick test at n=3
    from .enumerate import all_boolean_functions
    from .bbt_transform import batch_bbt_profiles

    n = 3
    print(f"=== Repair Module Test (n={n}) ===\n")

    all_tts = np.array(all_boolean_functions(n))
    batch_result = batch_bbt_profiles(all_tts.astype(np.float32), n)
    f_hats = np.array(batch_result['fourier_coefficients'])

    # Generate heuristic masks (should all be perfect at n=3)
    masks = np.sign(f_hats) * (np.abs(f_hats) > 0.05).astype(np.float32)

    result = repair_batch(masks, all_tts, n, verbose=True)
    print(f"\nAll converged: {np.all(result['converged'])}")

    # Test deliberate corruption: corrupt a known mask and repair it
    print(f"\n=== Corruption + Repair Test ===")
    H = build_walsh_matrix(n)

    # Parity_3: correct mask is [0,0,0,0,0,0,0,1]
    parity_tt = all_tts[0]  # func_id 0 may not be parity, find it
    # Build parity truth table
    N = 1 << n
    parity_tt = np.array([
        np.prod([1.0 - 2.0 * ((i >> b) & 1) for b in range(n)])
        for i in range(N)
    ], dtype=np.float32)

    # Start with wrong mask and repair
    wrong_mask = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)  # constant, wrong
    repaired, conv, iters, diag = repair_single(wrong_mask, parity_tt, H)
    print(f"  Started with: {wrong_mask}")
    print(f"  Repaired to:  {repaired}")
    print(f"  Converged: {conv}, Iterations: {iters}")

    # Verify
    acc = np.mean(np.sign(H @ repaired) == parity_tt)
    print(f"  Accuracy: {acc:.4f}")
