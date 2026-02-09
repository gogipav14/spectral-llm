"""
UNIVERSAL TERNARY REPRESENTABILITY: n=4 (65,536 Functions)
AND NPN SYMMETRY ANALYSIS (n=3, n=4)
===========================================================

Part 1: Tests whether EVERY Boolean function f:{-1,+1}^4 -> {-1,+1}
has a ternary PTF via batched mask enumeration.

Strategy: Instead of testing 65,536 functions x 43M masks each,
enumerate all 3^16 = 43,046,721 masks once, compute their truth tables,
and check if all 65,536 possible truth tables are covered.

Part 2: NPN (Negation-Permutation-Negation) equivalence class analysis
for n=3 and n=4. Two functions are NPN-equivalent if one can be obtained
from the other by negating inputs, permuting inputs, and/or negating output.
"""

import numpy as np
from itertools import product, permutations
from collections import Counter
import json
from pathlib import Path
from datetime import datetime


# =============================================================================
# Part 1: Batched Mask Enumeration for n=4
# =============================================================================

def build_fourier_basis(n):
    """Build 2^n x 2^n Boolean Fourier basis matrix.

    Row i = input i (lexicographic over {-1,+1}^n)
    Col j = Fourier character chi_S where S is encoded by bitmask j
    """
    n_inputs = 2 ** n
    inputs = list(product([-1, 1], repeat=n))

    basis = np.zeros((n_inputs, n_inputs), dtype=np.float32)
    for i, x in enumerate(inputs):
        for j in range(n_inputs):
            val = 1.0
            for k in range(n):
                if j & (1 << k):
                    val *= x[k]
            basis[i, j] = val
    return basis


def index_to_masks_batch(start, end, n_positions):
    """Convert mask indices [start, end) to ternary masks via base-3 decomposition."""
    indices = np.arange(start, end, dtype=np.int64)
    masks = np.zeros((len(indices), n_positions), dtype=np.float32)
    remaining = indices.copy()
    for pos in range(n_positions):
        digit = remaining % 3
        masks[:, pos] = digit - 1  # 0->-1, 1->0, 2->+1
        remaining //= 3
    return masks


def index_to_mask_single(idx, n_positions):
    """Convert a single mask index to ternary mask."""
    mask = np.zeros(n_positions, dtype=np.float32)
    remaining = idx
    for pos in range(n_positions):
        digit = remaining % 3
        mask[pos] = digit - 1
        remaining //= 3
    return mask


def test_n4_representability(batch_size=1_000_000):
    """Test all 65,536 four-variable Boolean functions for ternary representability.

    Returns dict with coverage results and witness masks.
    """
    n = 4
    n_inputs = 2 ** n        # 16
    n_functions = 2 ** n_inputs  # 65,536
    n_masks = 3 ** n_inputs     # 43,046,721

    print("=" * 70)
    print("UNIVERSAL TERNARY REPRESENTABILITY TEST (n=4)")
    print(f"  Boolean functions:  {n_functions:,}")
    print(f"  Ternary masks:     {n_masks:,}")
    print(f"  Basis dimension:   {n_inputs}")
    print(f"  Batch size:        {batch_size:,}")
    print(f"  Strategy: enumerate masks, check truth table coverage")
    print("=" * 70)

    basis = build_fourier_basis(n)  # (16, 16)

    # Track which truth tables are achievable
    covered = np.zeros(n_functions, dtype=bool)
    witnesses = np.full(n_functions, -1, dtype=np.int64)

    n_batches = (n_masks + batch_size - 1) // batch_size
    n_covered = 0

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_masks)

        # Generate ternary masks for this batch
        masks = index_to_masks_batch(start, end, n_inputs)  # (batch, 16)

        # Compute truth tables: sign(basis @ masks^T)
        raw = basis @ masks.T  # (16, batch)
        predictions = np.sign(raw)
        predictions[predictions == 0] = 1.0  # sign(0) -> +1

        # Encode truth tables as integers
        # Convention: -1 -> bit 1, +1 -> bit 0
        bits = ((1 - predictions) / 2).astype(np.int32)  # (16, batch)
        powers = (1 << np.arange(n_inputs, dtype=np.int64)).reshape(-1, 1)
        tt_ints = np.sum(bits.astype(np.int64) * powers, axis=0)  # (batch,)

        # Update coverage
        for i in range(len(tt_ints)):
            tt = tt_ints[i]
            if not covered[tt]:
                covered[tt] = True
                witnesses[tt] = start + i

        n_covered = int(np.sum(covered))

        if batch_idx % 5 == 0 or batch_idx == n_batches - 1:
            print(f"  Batch {batch_idx+1:3d}/{n_batches}: masks {start:>12,}-{end:>12,}, "
                  f"covered {n_covered:,}/{n_functions:,} ({100*n_covered/n_functions:.2f}%)")

        if n_covered == n_functions:
            print(f"\n  ALL {n_functions:,} functions covered after {end:,} masks!")
            break

    # Compute support distribution from witnesses
    support_sizes = {}
    for tt_int in range(n_functions):
        if covered[tt_int]:
            mask = index_to_mask_single(int(witnesses[tt_int]), n_inputs)
            support = int(np.sum(mask != 0))
            support_sizes[tt_int] = support

    support_dist = Counter(support_sizes.values())

    print(f"\n{'='*70}")
    print("RESULTS (n=4)")
    print(f"{'='*70}")
    print(f"Representable: {n_covered:,}/{n_functions:,} ({100*n_covered/n_functions:.1f}%)")

    if n_covered == n_functions:
        print("\nTHEOREM CONFIRMED: All 65,536 four-variable Boolean functions")
        print("have ternary PTF representations.")
    else:
        n_missing = n_functions - n_covered
        missing = [i for i in range(n_functions) if not covered[i]]
        print(f"\n{n_missing:,} functions NOT representable.")
        if n_missing <= 20:
            for fid in missing:
                print(f"  Function {fid}: binary = {bin(fid)}")

    if support_sizes:
        supports = list(support_sizes.values())
        print(f"\nSupport distribution (n=4):")
        print(f"  Mean support: {np.mean(supports):.1f}/{n_inputs}")
        print(f"  Min support:  {np.min(supports)}/{n_inputs}")
        print(f"  Max support:  {np.max(supports)}/{n_inputs}")
        for s in sorted(support_dist.keys()):
            pct = 100 * support_dist[s] / len(supports)
            print(f"  support={s:2d}: {support_dist[s]:6,} functions ({pct:.1f}%)")

    return {
        'n': 4,
        'n_functions': n_functions,
        'n_masks': n_masks,
        'n_covered': n_covered,
        'all_representable': n_covered == n_functions,
        'support_distribution': {str(k): v for k, v in sorted(support_dist.items())},
        'mean_support': float(np.mean(supports)) if support_sizes else 0,
        'witnesses': {str(k): int(v) for k, v in enumerate(witnesses) if v >= 0},
    }


# =============================================================================
# Part 2: NPN Equivalence Classes
# =============================================================================

def compute_npn_classes(n):
    """Compute NPN equivalence classes for n-variable Boolean functions.

    NPN = Negation (of inputs) + Permutation (of inputs) + Negation (of output).
    Two functions are NPN-equivalent if one can be obtained from the other
    by any combination of these transformations.

    Returns:
        canonical: array of canonical representative for each function
        classes: dict mapping representative -> list of function indices
    """
    n_inputs = 2 ** n
    n_functions = 2 ** n_inputs
    mask_all = n_functions - 1

    print(f"\nComputing NPN classes for n={n} ({n_functions:,} functions)...")

    # Precompute all input permutations (from variable perm + input negation)
    var_perms = list(permutations(range(n)))
    neg_patterns = list(product([False, True], repeat=n))

    # For each (var_perm, neg_pattern), compute the induced input permutation
    input_perms = []
    for vperm in var_perms:
        for neg in neg_patterns:
            perm = np.zeros(n_inputs, dtype=np.int32)
            for j in range(n_inputs):
                new_j = 0
                for k in range(n):
                    # Variable k is at bit position (n-1-k) in input index
                    # (because product iterates leftmost variable slowest)
                    bit_pos = n - 1 - k
                    bit_val = (j >> bit_pos) & 1

                    # Apply negation
                    if neg[k]:
                        bit_val = 1 - bit_val

                    # Apply permutation: variable k maps to position vperm[k]
                    new_bit_pos = n - 1 - vperm[k]
                    new_j |= (bit_val << new_bit_pos)

                perm[j] = new_j
            input_perms.append(perm)

    print(f"  {len(input_perms)} input permutations (x2 for output negation = {2*len(input_perms)} total transforms)")

    # Decompose all truth tables into bit arrays
    all_T = np.arange(n_functions, dtype=np.int64)
    bits = np.zeros((n_functions, n_inputs), dtype=np.int64)
    for j in range(n_inputs):
        bits[:, j] = (all_T >> j) & 1

    # Track minimum (canonical) representative
    canonical = all_T.copy()
    powers = (np.int64(1) << np.arange(n_inputs, dtype=np.int64))

    for ip_idx, perm in enumerate(input_perms):
        # Apply input permutation: new_bits[:, j] = bits[:, perm[j]]
        new_bits = bits[:, perm]

        # Pack to integers
        new_T = new_bits @ powers

        # Without output negation
        canonical = np.minimum(canonical, new_T)

        # With output negation (flip all bits)
        neg_T = (~new_T) & mask_all
        canonical = np.minimum(canonical, neg_T)

        if (ip_idx + 1) % 100 == 0:
            print(f"  Processed {ip_idx+1}/{len(input_perms)} permutations...")

    # Group into classes
    classes = {}
    for T in range(n_functions):
        c = int(canonical[T])
        if c not in classes:
            classes[c] = []
        classes[c].append(T)

    print(f"  Found {len(classes)} NPN equivalence classes")

    # Class size distribution
    sizes = [len(v) for v in classes.values()]
    size_dist = Counter(sizes)
    print(f"  Class size distribution:")
    for s in sorted(size_dist.keys()):
        print(f"    size {s:5d}: {size_dist[s]:4d} classes")

    return canonical, classes


# =============================================================================
# Main
# =============================================================================

def main():
    results = {}

    # --- Part 1: n=4 Representability ---
    rep_results = test_n4_representability(batch_size=1_000_000)
    results['n4_representability'] = rep_results

    # --- Part 2: NPN Analysis ---
    # n=3
    canonical_3, classes_3 = compute_npn_classes(3)
    results['npn_n3'] = {
        'n': 3,
        'n_functions': 256,
        'n_classes': len(classes_3),
        'class_sizes': {str(k): len(v) for k, v in sorted(classes_3.items())},
    }

    # n=4
    canonical_4, classes_4 = compute_npn_classes(4)
    results['npn_n4'] = {
        'n': 4,
        'n_functions': 65536,
        'n_classes': len(classes_4),
        'class_sizes': {str(k): len(v) for k, v in sorted(classes_4.items())},
    }

    # --- Cross-reference: representability by NPN class ---
    if rep_results['all_representable']:
        print(f"\nSince all n=4 functions are representable, all {len(classes_4)} NPN classes are representable.")
    else:
        # Check which NPN classes have unrepresentable members
        n_covered = rep_results['n_covered']
        print(f"\nNPN class representability analysis:")
        for rep, members in sorted(classes_4.items()):
            all_rep = all(str(m) in rep_results['witnesses'] for m in members)
            if not all_rep:
                n_unrep = sum(1 for m in members if str(m) not in rep_results['witnesses'])
                print(f"  Class {rep}: {n_unrep}/{len(members)} NOT representable")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nn=3: {256} functions, {len(classes_3)} NPN classes")
    print(f"     All 256 representable (Theorem 2)")
    print(f"\nn=4: {65536:,} functions, {len(classes_4)} NPN classes")
    if rep_results['all_representable']:
        print(f"     All {65536:,} representable (extends conjecture)")
    else:
        print(f"     {rep_results['n_covered']:,}/{65536:,} representable")

    # Save
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True, parents=True)

    results['timestamp'] = datetime.now().isoformat()

    output_path = output_dir / 'representability_n4_exhaustive.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {output_path}")


if __name__ == '__main__':
    main()
