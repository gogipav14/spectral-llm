"""
n=5 minimum-support MILP experiment for the BBT theory paper.

Universe size at n=5 is 2^32 = 4.3B Boolean functions, infeasible to enumerate
in full. NPN-canonical-representative enumeration (616,126 classes per OEIS
A000370) is feasible but requires either a precomputed list or a fast NPN
canonical-form computation. Both routes are out of scope for the present
experimental round; instead, we run on a UNIFORMLY-SAMPLED set of functions.
This still gives a defensible empirical answer to "does \mu predict
min-support at n=5?" with correlation reported on the sample.

For each sampled function on 5 vars, solve

    min  ||w||_0
    s.t. sign(H_5 w) = f,  w \in {-1, 0, +1}^32

via a binary IP encoded as in n4_min_support_mu.py (w = w+ - w-, mutex on
each pair). Each instance has 64 binary vars and 64 constraints; HiGHS
solves it in under a second on average for n=5.

Output: CSV with one row per sampled function (fid_seed, min_support, I, mu,
log2_mu, H_inf, max_inf). Resumable.
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


N = 5
DIM = 1 << N           # 32 inputs / coefficients
SAMPLE_SIZE = int(os.environ.get("N5_SAMPLE", "5000"))
MILP_TIME_LIMIT = float(os.environ.get("N5_TIME_LIMIT", "20.0"))


def hadamard(n: int) -> np.ndarray:
    H = np.array([[1]], dtype=np.int8)
    for _ in range(n):
        H = np.block([[H, H], [H, -H]])
    return H


def boolean_function_truth_table_from_seed(seed: int) -> np.ndarray:
    """Generate a uniformly random {-1,+1}^DIM truth table from a seed."""
    rng = np.random.default_rng(seed)
    return rng.choice([-1, 1], size=DIM).astype(np.int8)


def influence_vector(f: np.ndarray, H: np.ndarray) -> np.ndarray:
    f_hat = H @ f / DIM
    influences = np.zeros(N)
    for l in range(N):
        mask_indices = np.array([i for i in range(DIM) if (i >> l) & 1])
        influences[l] = float(np.sum(f_hat[mask_indices] ** 2))
    return influences


def diagnostics(f: np.ndarray, H: np.ndarray) -> dict:
    inf_vec = influence_vector(f, H)
    I = float(inf_vec.sum())
    mu = float(np.prod(np.power(2.0, -inf_vec / (1.0 + inf_vec))))
    if I > 1e-12:
        p = inf_vec / I
        p_pos = p[p > 1e-12]
        H_inf = float(-np.sum(p_pos * np.log2(p_pos)))
    else:
        H_inf = 0.0
    max_inf = float(inf_vec.max())
    return dict(I=I, mu=mu, H_inf=H_inf, max_inf=max_inf,
                log2_mu=float(np.log2(mu)) if mu > 0 else float("-inf"))


def min_support_milp(f: np.ndarray, H: np.ndarray) -> int | None:
    nvars = 2 * DIM
    A_sign = np.zeros((DIM, nvars))
    for i in range(DIM):
        A_sign[i, :DIM] = f[i] * H[i, :]
        A_sign[i, DIM:] = -f[i] * H[i, :]
    A_mutex = np.zeros((DIM, nvars))
    for j in range(DIM):
        A_mutex[j, j] = 1
        A_mutex[j, j + DIM] = 1
    A = np.vstack([A_sign, A_mutex])
    bl = np.concatenate([np.ones(DIM), np.zeros(DIM)])
    bu = np.concatenate([np.full(DIM, np.inf), np.ones(DIM)])
    constraints = LinearConstraint(A, bl, bu)
    c = np.ones(nvars)
    integrality = np.ones(nvars)
    bounds = Bounds(np.zeros(nvars), np.ones(nvars))
    res = milp(c, constraints=constraints, integrality=integrality,
               bounds=bounds, options={"disp": False, "time_limit": MILP_TIME_LIMIT})
    if not res.success:
        return None
    return int(round(res.fun))


def _solve_one(seed: int) -> dict:
    H = hadamard(N).astype(np.float64)
    f = boolean_function_truth_table_from_seed(seed).astype(np.float64)
    d = diagnostics(f, H)
    try:
        d["min_support"] = min_support_milp(f, H)
    except Exception:
        d["min_support"] = -1
    d["seed"] = seed
    return d


def _solve_chunk(seeds):
    return [_solve_one(s) for s in seeds]


def main():
    out_path = Path("/mnt/c/Users/gogip/Spectral-LLM/experiments/n5_min_support_mu.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_seeds: set[int] = set()
    if out_path.exists():
        with open(out_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    done_seeds.add(int(row["seed"]))
                except (KeyError, ValueError):
                    pass
        print(f"[n5-milp] resume: {len(done_seeds)} seeds already computed")
    else:
        with open(out_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=[
                "seed", "min_support", "I", "mu", "log2_mu", "H_inf", "max_inf"
            ]).writeheader()

    todo = [s for s in range(SAMPLE_SIZE) if s not in done_seeds]
    if not todo:
        print(f"[n5-milp] all {SAMPLE_SIZE} done.")
        return
    print(f"[n5-milp] {len(todo)} seeds to solve at n={N}, time_limit={MILP_TIME_LIMIT}s")

    n_workers = max(1, mp.cpu_count() - 1)
    chunk_size = 32
    chunks = [todo[i:i + chunk_size] for i in range(0, len(todo), chunk_size)]
    t0 = time.time()
    n_done = 0
    target = len(todo)
    fields = ["seed", "min_support", "I", "mu", "log2_mu", "H_inf", "max_inf"]
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
            print(f"[n5-milp] {n_done}/{target} done ({100*n_done/target:.1f}%) "
                  f"rate={rate:.2f}/s eta={eta/60:.1f}min", flush=True)

    print(f"[n5-milp] done. total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
