"""
n=4 minimum-support MILP experiment for the BBT theory paper.

For each of 2^16 = 65,536 Boolean functions on 4 variables, solve

    min  ||w||_0
    s.t. sign(H_4 w) = f,  w ∈ {-1, 0, +1}^16

via a binary IP (w = w+ - w-, mutex w+_j + w-_j ≤ 1, ±1 sign on H rows).

Then for each function compute:
  - I(f)  : total Boolean influence
  - mu(f) : BBT contraction invariant
  - H_inf : entropy of the influence distribution
  - max_inf : largest single-coordinate influence

Output: a CSV with one row per function, suitable for regression.

Runs in batches with persistence so a crash mid-sweep doesn't lose progress.
Solver is scipy.optimize.milp (HiGHS); each instance solves in ~10-100 ms,
so the whole sweep is ~hours single-threaded. Multiprocessed across CPU cores.
"""
from __future__ import annotations
import csv
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds


N = 4
DIM = 1 << N            # 16 inputs
NUM_FUNCTIONS = 1 << DIM   # 65,536 boolean functions


def hadamard(n: int) -> np.ndarray:
    """Sylvester construction. H_n is 2^n x 2^n, ±1, symmetric, H_n^2 = 2^n I."""
    H = np.array([[1]], dtype=np.int8)
    for _ in range(n):
        H = np.block([[H, H], [H, -H]])
    return H


def boolean_function_truth_table(fid: int) -> np.ndarray:
    """Return the truth table of function id `fid` as a (DIM,) int8 in {-1, +1}.
    Bit k of fid encodes the value at input k: 0 -> -1, 1 -> +1."""
    bits = np.array([(fid >> k) & 1 for k in range(DIM)], dtype=np.int8)
    return 2 * bits - 1


def influence_vector(f: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Coordinate influences of the boolean function f (truth table ±1).
    Inf_l(f) = sum_{S ∋ l} f_hat(S)^2 where f_hat = H f / 2^n.
    """
    f_hat = H @ f / DIM   # (DIM,) Walsh-Hadamard transform of f
    influences = np.zeros(N)
    for l in range(N):
        # subsets S containing l: those whose binary index has bit l set
        mask_indices = np.array([i for i in range(DIM) if (i >> l) & 1])
        influences[l] = float(np.sum(f_hat[mask_indices] ** 2))
    return influences


def diagnostics(f: np.ndarray, H: np.ndarray) -> dict:
    """Compute I(f), mu(f), entropy(Inf), max_inf."""
    inf_vec = influence_vector(f, H)
    I = float(inf_vec.sum())
    # mu(f) = prod_l 2^(-Inf_l/(1+Inf_l))
    mu = float(np.prod(np.power(2.0, -inf_vec / (1.0 + inf_vec))))
    # entropy of Inf distribution (normalised)
    if I > 1e-12:
        p = inf_vec / I
        # exclude zero entries to avoid 0*log(0)
        p_pos = p[p > 1e-12]
        H_inf = float(-np.sum(p_pos * np.log2(p_pos)))
    else:
        H_inf = 0.0
    max_inf = float(inf_vec.max())
    return dict(I=I, mu=mu, H_inf=H_inf, max_inf=max_inf,
                log2_mu=float(np.log2(mu)) if mu > 0 else float("-inf"))


def min_support_milp(f: np.ndarray, H: np.ndarray) -> int | None:
    """Solve the minimum-support binary IP for the function f.

    w = w+ - w- with w+, w- ∈ {0,1}^DIM and w+_j + w-_j ≤ 1.
    Sign constraint:  f_i * (H w)_i ≥ 1   (equivalently > 0 for ±1, but
    we use ≥ 1 to enforce a strict-margin ternary representation).
    Objective:        min sum(w+) + sum(w-).

    Returns minimum support (int) or None if infeasible.
    """
    # 2*DIM binary vars: [w+ (DIM), w- (DIM)]
    nvars = 2 * DIM
    # Sign constraints
    A_sign = np.zeros((DIM, nvars))
    for i in range(DIM):
        A_sign[i, :DIM] = f[i] * H[i, :]
        A_sign[i, DIM:] = -f[i] * H[i, :]
    b_lower_sign = np.ones(DIM)
    b_upper_sign = np.full(DIM, np.inf)
    # Mutual exclusion w+_j + w-_j ≤ 1
    A_mutex = np.zeros((DIM, nvars))
    for j in range(DIM):
        A_mutex[j, j] = 1
        A_mutex[j, j + DIM] = 1
    b_lower_mutex = np.zeros(DIM)
    b_upper_mutex = np.ones(DIM)

    A = np.vstack([A_sign, A_mutex])
    bl = np.concatenate([b_lower_sign, b_lower_mutex])
    bu = np.concatenate([b_upper_sign, b_upper_mutex])
    constraints = LinearConstraint(A, bl, bu)
    c = np.ones(nvars)
    integrality = np.ones(nvars)
    bounds = Bounds(np.zeros(nvars), np.ones(nvars))

    res = milp(c, constraints=constraints, integrality=integrality,
               bounds=bounds, options={"disp": False, "time_limit": 30.0})
    if not res.success:
        return None
    return int(round(res.fun))


def _solve_one(fid: int) -> dict:
    """Solve a single function. Top-level so multiprocessing can pickle it."""
    H = hadamard(N).astype(np.float64)
    f = boolean_function_truth_table(fid).astype(np.float64)
    d = diagnostics(f, H)
    try:
        d["min_support"] = min_support_milp(f, H)
    except Exception:
        d["min_support"] = -1
    d["fid"] = fid
    return d


def _solve_chunk(seq):
    """Multiprocessing-friendly worker: solve a list of fids."""
    return [_solve_one(fid) for fid in seq]


def main():
    out_path = Path("/mnt/c/Users/gogip/Spectral-LLM/experiments/n4_min_support_mu.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Resume support: check what's already done
    done_fids: set[int] = set()
    if out_path.exists():
        with open(out_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    done_fids.add(int(row["fid"]))
                except (KeyError, ValueError):
                    pass
        print(f"[n4-milp] resume: {len(done_fids)} functions already computed")
    else:
        with open(out_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=[
                "fid", "min_support", "I", "mu", "log2_mu", "H_inf", "max_inf"
            ]).writeheader()

    todo = [fid for fid in range(NUM_FUNCTIONS) if fid not in done_fids]
    if not todo:
        print(f"[n4-milp] all {NUM_FUNCTIONS} done.")
        return
    print(f"[n4-milp] {len(todo)} functions to solve.")

    n_workers = max(1, mp.cpu_count() - 1)
    chunk_size = 256
    chunks = [todo[i:i + chunk_size] for i in range(0, len(todo), chunk_size)]
    t0 = time.time()
    n_done = 0
    target = len(todo)
    fields = ["fid", "min_support", "I", "mu", "log2_mu", "H_inf", "max_inf"]
    with mp.Pool(n_workers) as pool, open(out_path, "a", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fields)
        for chunk_results in pool.imap_unordered(_solve_chunk, chunks):
            for row in chunk_results:
                writer.writerow({k: row[k] for k in fields})
            fout.flush()
            n_done += len(chunk_results)
            elapsed = time.time() - t0
            rate = n_done / max(1e-6, elapsed)
            eta = (target - n_done) / max(1e-6, rate)
            print(f"[n4-milp] {n_done}/{target} done ({100*n_done/target:.1f}%) "
                  f"rate={rate:.1f}/s eta={eta/60:.1f}min", flush=True)

    print(f"[n4-milp] done. total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
