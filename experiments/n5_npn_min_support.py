"""
n=5 minimum-support MILP on NPN-canonical representatives.

Consumes /mnt/c/Users/gogip/Spectral-LLM/experiments/npn5_canonical.npy
(produced by experiments/npn_enumerate.py) — a uint32 array of NPN-canonical
truth-table integers — and runs the same minimum-support MILP from
experiments/n5_min_support_mu.py on each canonical rep.

Differences from n5_min_support_mu.py:
  - Functions are looked up by truth-table integer (`tt_int`), not by random seed.
    NPN-canonical reps are deterministic; ``tt_int`` is the unique identifier.
  - Output CSV is keyed by ``tt_int``.
  - Optionally subsample the canonical-rep list (e.g. 10,000 of 616,126) via
    the N5_NPN_SAMPLE env var; defaults to all reps.

Resumable. Same per-instance MILP as the uniform sweep (~0.4 s mean).
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
DIM = 1 << N
NPN_SAMPLE = int(os.environ.get("N5_NPN_SAMPLE", "0"))     # 0 = all reps
MILP_TIME_LIMIT = float(os.environ.get("N5_TIME_LIMIT", "20.0"))


def hadamard(n: int) -> np.ndarray:
    H = np.array([[1]], dtype=np.int8)
    for _ in range(n):
        H = np.block([[H, H], [H, -H]])
    return H


def truth_table_from_int(tt_int: int) -> np.ndarray:
    """Decode 32-bit truth-table int into ±1 vector of length 32."""
    bits = np.array([(tt_int >> k) & 1 for k in range(DIM)], dtype=np.int8)
    return 2 * bits - 1


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


def _solve_one(tt_int: int) -> dict:
    H = hadamard(N).astype(np.float64)
    f = truth_table_from_int(int(tt_int)).astype(np.float64)
    d = diagnostics(f, H)
    try:
        d["min_support"] = min_support_milp(f, H)
    except Exception:
        d["min_support"] = -1
    d["tt_int"] = int(tt_int)
    return d


def _solve_chunk(tt_ints):
    return [_solve_one(t) for t in tt_ints]


def main():
    canon_path = Path("/mnt/c/Users/gogip/Spectral-LLM/experiments/npn5_canonical.npy")
    if not canon_path.exists():
        sys.exit(f"[npn-milp] canonical reps not found at {canon_path} — "
                 f"run experiments/npn_enumerate.py first")
    canonical = np.load(str(canon_path))
    print(f"[npn-milp] {len(canonical)} canonical reps loaded")

    if NPN_SAMPLE > 0 and NPN_SAMPLE < len(canonical):
        rng = np.random.default_rng(0)
        idx = rng.choice(len(canonical), size=NPN_SAMPLE, replace=False)
        idx.sort()
        sample = canonical[idx]
        print(f"[npn-milp] sub-sampling to {len(sample)} reps "
              f"(seed=0 reproducible)")
    else:
        sample = canonical

    out_path = Path("/mnt/c/Users/gogip/Spectral-LLM/experiments/n5_npn_min_support.csv")
    done: set[int] = set()
    if out_path.exists():
        with open(out_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    done.add(int(row["tt_int"]))
                except (KeyError, ValueError):
                    pass
        print(f"[npn-milp] resume: {len(done)} reps already computed")
    else:
        with open(out_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=[
                "tt_int", "min_support", "I", "mu", "log2_mu", "H_inf", "max_inf"
            ]).writeheader()

    todo = [int(t) for t in sample if int(t) not in done]
    if not todo:
        print(f"[npn-milp] all {len(sample)} reps done.")
        return
    print(f"[npn-milp] {len(todo)} reps to solve at n={N}")

    n_workers = max(1, mp.cpu_count() - 1)
    chunk_size = 32
    chunks = [todo[i:i + chunk_size] for i in range(0, len(todo), chunk_size)]
    t0 = time.time()
    n_done = 0
    target = len(todo)
    fields = ["tt_int", "min_support", "I", "mu", "log2_mu", "H_inf", "max_inf"]
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
            if n_done % 100 == 0 or n_done == target:
                print(f"[npn-milp] {n_done}/{target} done ({100*n_done/target:.1f}%) "
                      f"rate={rate:.2f}/s eta={eta/60:.1f}min", flush=True)

    print(f"[npn-milp] done. total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
