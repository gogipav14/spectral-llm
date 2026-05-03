"""
Enumerate NPN-canonical Boolean functions on n=5 variables via the orbit-marking
algorithm: iterate fid = 0..2^32, skip already-seen fids, otherwise emit fid as
the lex-min canonical representative of its orbit and mark all orbit members.

NPN orbit: 2 (output negation) x 2^n (input negations) x n! (input
permutations) = 7680 transformations for n=5. Total NPN classes: 616,126
(OEIS A000370).

The expensive operation is computing one orbit (7680 transforms x 32-bit
truth-table manipulations). With numba @njit this completes in ~3-5 minutes
total for all 616k orbits.

Output: a binary file of uint32 NPN-canonical representatives, length 616,126.
"""
from __future__ import annotations
import itertools
import os
import sys
import time
from pathlib import Path

import numpy as np
from numba import njit

N = 5
DIM = 1 << N           # 32
MASK = (1 << DIM) - 1  # 0xFFFFFFFF


def precompute_perm_lookups() -> np.ndarray:
    """For each permutation pi of [0..N-1], compute the input-permutation
    lookup table p[pi_idx, x] = bit-permuted x. The truth table under pi is
    new_tt[x] = old_tt[p[pi_idx, x]].
    """
    perms = list(itertools.permutations(range(N)))
    n_perms = len(perms)
    lookup = np.zeros((n_perms, DIM), dtype=np.uint8)
    for pi_idx, perm in enumerate(perms):
        for x in range(DIM):
            x_new = 0
            for i in range(N):
                if (x >> i) & 1:
                    x_new |= 1 << perm[i]
            lookup[pi_idx, x] = x_new
    return lookup


@njit(cache=True)
def apply_npn_op(tt: np.uint32, perm_lookup: np.ndarray, pi_idx: int,
                  in_neg_mask: int, out_neg: bool) -> np.uint32:
    """Apply one NPN operator to truth table tt. Returns the resulting
    32-bit truth table integer."""
    new_tt = np.uint32(0)
    for x in range(DIM):
        # First apply input negation: x_inverted = x ^ in_neg_mask
        # Then apply permutation lookup.
        x_inverted = x ^ in_neg_mask
        x_src = perm_lookup[pi_idx, x_inverted]
        if (tt >> x_src) & np.uint32(1):
            new_tt |= np.uint32(1) << x
    if out_neg:
        new_tt = (~new_tt) & np.uint32(MASK)
    return new_tt


@njit(cache=True)
def compute_orbit(tt: np.uint32, perm_lookup: np.ndarray,
                   orbit_buf: np.ndarray) -> int:
    """Compute the full NPN orbit of tt. Writes elements into orbit_buf
    (preallocated, length >= 7680). Returns the number of distinct elements."""
    n_perms = perm_lookup.shape[0]
    count = 0
    for pi_idx in range(n_perms):
        for in_neg_mask in range(DIM):  # 0..31 (each bit = negate var i)
            for out_neg_int in range(2):
                out_neg = (out_neg_int == 1)
                elem = apply_npn_op(tt, perm_lookup, pi_idx, in_neg_mask, out_neg)
                # Naive uniqueness check (could grow buf with hashset, but
                # since orbit size <= 7680 we just store all and dedupe later)
                orbit_buf[count] = elem
                count += 1
    return count


@njit(cache=True)
def enumerate_canonical(perm_lookup: np.ndarray, seen: np.ndarray,
                         out_buf: np.ndarray) -> int:
    """Iterate fid = 0..2^32-1; for each unseen fid, emit it as canonical and
    mark its orbit. Returns count of canonical reps written to out_buf."""
    count = 0
    orbit_buf = np.empty(8192, dtype=np.uint32)
    n_total = np.uint64(1) << np.uint64(DIM)  # 2^32
    fid = np.uint32(0)
    progress_step = np.uint64(1) << np.uint64(28)  # ~268M
    next_progress = progress_step
    while True:
        # is_seen check
        if (seen[fid >> 3] >> (fid & np.uint32(7))) & np.uint8(1):
            if fid == np.uint32(MASK):
                break
            fid = np.uint32(fid + 1)
            continue
        # fid is canonical (lex-min of its unseen orbit)
        out_buf[count] = fid
        count += 1
        # Compute orbit, mark all
        orbit_size = compute_orbit(fid, perm_lookup, orbit_buf)
        for k in range(orbit_size):
            elem = orbit_buf[k]
            seen[elem >> 3] |= np.uint8(1) << np.uint8(elem & 7)
        if fid == np.uint32(MASK):
            break
        fid = np.uint32(fid + 1)
    return count


def main():
    out_path = Path("/mnt/c/Users/gogip/Spectral-LLM/experiments/npn5_canonical.npy")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        existing = np.load(str(out_path))
        print(f"[npn-enum] cached: {len(existing)} canonical reps at {out_path}")
        return existing

    print("[npn-enum] precomputing permutation lookups...")
    perm_lookup = precompute_perm_lookups()
    print(f"[npn-enum] {perm_lookup.shape[0]} permutations x {DIM} inputs")

    print("[npn-enum] allocating seen-bitmap (512 MB)...")
    # 2^32 bits = 2^29 bytes = 512 MB
    seen = np.zeros(1 << (DIM - 3), dtype=np.uint8)

    # Worst case 1M canonical reps; 616,126 expected
    out_buf = np.empty(2 * 616127, dtype=np.uint32)

    print("[npn-enum] warmup numba JIT...")
    _ = compute_orbit(np.uint32(0), perm_lookup, np.empty(8192, dtype=np.uint32))

    print("[npn-enum] enumerating...")
    t0 = time.time()
    count = enumerate_canonical(perm_lookup, seen, out_buf)
    elapsed = time.time() - t0
    print(f"[npn-enum] {count} canonical reps found in {elapsed:.1f}s "
          f"({elapsed/60:.1f} min)")

    canonical = out_buf[:count].copy()
    np.save(str(out_path), canonical)
    print(f"[npn-enum] saved to {out_path}")
    return canonical


if __name__ == "__main__":
    main()
