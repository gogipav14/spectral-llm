"""
COUNTEREXAMPLE SEARCH: Can we find a Boolean function that RESISTS ternary representation?
===========================================================================================

For n>=5, exhaustive enumeration is infeasible (3^32 ≈ 10^15 masks for n=5).
We use targeted search on "hard" functions + random functions.

Search pipeline (per function):
  1. Fourier rounding (instant, catches ~90%)
  2. Random ternary mask sampling (fast, vectorized)
  3. Multi-start simulated annealing (heavier, only if 1+2 fail)
"""

import numpy as np
from itertools import product
from collections import Counter
import json
from pathlib import Path
from datetime import datetime
import sys


def build_fourier_basis(n):
    """Build 2^n x 2^n Boolean Fourier basis matrix."""
    n_inputs = 2 ** n
    inputs = list(product([-1, 1], repeat=n))
    basis = np.zeros((n_inputs, n_inputs), dtype=np.float64)
    for i, x in enumerate(inputs):
        for j in range(n_inputs):
            val = 1.0
            for k in range(n):
                if j & (1 << k):
                    val *= x[k]
            basis[i, j] = val
    return basis


def check_mask(basis, y, w):
    """Check if ternary mask w represents truth table y."""
    raw = basis @ w
    pred = np.sign(raw)
    pred[pred == 0] = 1.0
    return np.all(pred == y)


def count_correct(basis, y, w):
    """Count how many truth table entries mask w gets correct."""
    raw = basis @ w
    pred = np.sign(raw)
    pred[pred == 0] = 1.0
    return int(np.sum(pred == y))


def fourier_rounding(basis, y, n_thresholds=100):
    """Round Fourier coefficients to {-1,0,+1} at various thresholds."""
    n_dim = basis.shape[1]
    f_hat = basis.T @ y / n_dim

    abs_hat = np.abs(f_hat)
    thresholds = np.unique(np.concatenate([
        np.linspace(0, np.max(abs_hat) + 0.01, n_thresholds),
        abs_hat - 1e-10,  # just below each coefficient
        abs_hat + 1e-10,  # just above each coefficient
    ]))

    for t in thresholds:
        w = np.zeros(n_dim, dtype=np.float64)
        mask = abs_hat > t
        w[mask] = np.sign(f_hat[mask])
        if check_mask(basis, y, w):
            return w
    return None


def random_search(basis, y, n_samples=200_000, rng=None):
    """Vectorized random ternary mask sampling."""
    if rng is None:
        rng = np.random.default_rng(42)
    n_dim = basis.shape[1]
    n_inputs = basis.shape[0]

    batch = 50_000
    for _ in range(n_samples // batch):
        masks = rng.choice([-1.0, 0.0, 1.0], size=(batch, n_dim))
        raw = basis @ masks.T  # (n_inputs, batch)
        pred = np.sign(raw)
        pred[pred == 0] = 1.0
        correct = np.sum(pred == y[:, None], axis=0)

        perfect = np.where(correct == n_inputs)[0]
        if len(perfect) > 0:
            return masks[perfect[0]]

    return None


def simulated_annealing(basis, y, w_init=None, n_steps=50_000, rng=None):
    """Local search from a starting mask."""
    if rng is None:
        rng = np.random.default_rng(42)
    n_dim = basis.shape[1]
    n_inputs = basis.shape[0]

    if w_init is None:
        w = rng.choice([-1.0, 0.0, 1.0], size=n_dim)
    else:
        w = w_init.copy()

    raw = basis @ w
    score = int(np.sum((y * raw) > 0))
    best_w = w.copy()
    best_score = score

    choices = np.array([-1.0, 0.0, 1.0])

    for step in range(n_steps):
        temp = 1.5 * (1.0 - step / n_steps)

        j = rng.integers(0, n_dim)
        old_val = w[j]
        new_val = choices[rng.integers(0, 3)]
        if new_val == old_val:
            continue

        delta_w = new_val - old_val
        new_raw = raw + delta_w * basis[:, j]
        new_score = int(np.sum((y * new_raw) > 0))

        delta = new_score - score
        if delta > 0 or (temp > 0.01 and rng.random() < np.exp(delta / temp)):
            w[j] = new_val
            raw = new_raw
            score = new_score
            if score > best_score:
                best_score = score
                best_w = w.copy()
            if score == n_inputs:
                return best_w

    return best_w if best_score == n_inputs else None


def search_function(basis, y, name="?", verbose=True):
    """Full search pipeline for one function."""
    n_dim = basis.shape[1]

    # Strategy 1: Fourier rounding
    w = fourier_rounding(basis, y)
    if w is not None:
        s = int(np.sum(w != 0))
        if verbose:
            print(f"    {name}: Fourier rounding (support={s}/{n_dim})", flush=True)
        return w, 'fourier', s

    # Strategy 2: Random search
    w = random_search(basis, y, n_samples=200_000, rng=np.random.default_rng(hash(name) % 2**31))
    if w is not None:
        s = int(np.sum(w != 0))
        if verbose:
            print(f"    {name}: random search (support={s}/{n_dim})", flush=True)
        return w, 'random', s

    # Strategy 3: Multi-start SA (5 starts)
    for start_i in range(5):
        w = simulated_annealing(
            basis, y, n_steps=50_000,
            rng=np.random.default_rng(hash(name) % 2**31 + start_i * 1000)
        )
        if w is not None:
            s = int(np.sum(w != 0))
            if verbose:
                print(f"    {name}: SA start {start_i} (support={s}/{n_dim})", flush=True)
            return w, 'sa', s

    if verbose:
        print(f"    {name}: *** NOT FOUND ***", flush=True)
    return None, 'not_found', -1


def generate_named_functions(n):
    """Generate named "hard" Boolean functions for testing."""
    inputs = list(product([-1, 1], repeat=n))
    fns = {}

    # Majority
    fns['majority'] = np.array([-1 if sum(x) < 0 else 1 for x in inputs], dtype=np.float64)

    # Parity
    fns['parity'] = np.array([np.prod(x) for x in inputs], dtype=np.float64)

    # AND / OR
    fns['and'] = np.array([-1 if all(xi == -1 for xi in x) else 1 for x in inputs], dtype=np.float64)
    fns['or'] = np.array([-1 if any(xi == -1 for xi in x) else 1 for x in inputs], dtype=np.float64)

    # Threshold-k for k=2,...,n-1
    for k in range(2, n):
        fns[f'threshold_{k}'] = np.array(
            [-1 if sum(1 for xi in x if xi == -1) >= k else 1 for x in inputs], dtype=np.float64)

    # Weighted threshold: sign(1*x1 + 2*x2 + ... + n*xn)
    fns['weighted_threshold'] = np.array(
        [-1 if sum((i+1)*x[i] for i in range(n)) < 0 else 1 for x in inputs], dtype=np.float64)

    if n >= 5:
        # Tribes: OR(AND(x1,x2), AND(x3,x4), x5)
        fns['tribes'] = np.array(
            [-1 if ((x[0]==-1 and x[1]==-1) or (x[2]==-1 and x[3]==-1) or x[4]==-1) else 1
             for x in inputs], dtype=np.float64)

        # Address/mux
        fns['address_mux'] = np.array([
            x[2 + (2*(x[0]==-1) + (x[1]==-1)) % (n-2)]
            for x in inputs], dtype=np.float64)

        # Recursive majority: maj(maj(x1,x2,x3), x4, x5)
        fns['recursive_majority'] = np.array([
            -1 if ((-1 if (x[0]+x[1]+x[2]) < 0 else 1) + x[3] + x[4]) < 0 else 1
            for x in inputs], dtype=np.float64)

    # Inner product (pairs): x1*x2 * x3*x4 (= XOR of two ANDs in {-1,+1})
    if n >= 4:
        fns['inner_product'] = np.array([x[0]*x[1]*x[2]*x[3] for x in inputs], dtype=np.float64)

    return fns


def main():
    results = {}

    for n in [5, 6]:
        n_dim = 2 ** n
        print(f"\n{'='*70}", flush=True)
        print(f"COUNTEREXAMPLE SEARCH: n={n} (dim={n_dim})", flush=True)
        print(f"  Ternary masks: 3^{n_dim} ≈ {3**n_dim:.1e}", flush=True)
        print(f"{'='*70}", flush=True)

        basis = build_fourier_basis(n)

        # Named functions
        named = generate_named_functions(n)
        print(f"\n--- Named functions ({len(named)}) ---", flush=True)

        fn_results = {}
        for name, y in named.items():
            w, method, support = search_function(basis, y, name=name)
            fn_results[name] = {'found': w is not None, 'method': method, 'support': support}

        # Random functions
        n_random = 500 if n == 5 else 100
        print(f"\n--- Random functions ({n_random}) ---", flush=True)
        rng = np.random.default_rng(54321)
        found_count = 0
        not_found_count = 0
        methods = Counter()
        counterexamples = []

        for i in range(n_random):
            y = rng.choice([-1.0, 1.0], size=n_dim)
            w, method, support = search_function(basis, y, name=f"rand_{i}", verbose=False)
            if w is not None:
                found_count += 1
                methods[method] += 1
            else:
                not_found_count += 1
                counterexamples.append(i)
                # Re-run with verbose for counterexample candidates
                print(f"    rand_{i}: NOT FOUND (checking harder...)", flush=True)
                # Extra SA attempts
                extra_found = False
                for extra in range(10):
                    w2 = simulated_annealing(basis, y, n_steps=100_000,
                                             rng=np.random.default_rng(i * 100 + extra))
                    if w2 is not None:
                        found_count += 1
                        not_found_count -= 1
                        counterexamples.pop()
                        methods['sa_extra'] += 1
                        extra_found = True
                        s = int(np.sum(w2 != 0))
                        print(f"    rand_{i}: found on extra SA attempt {extra} (support={s})", flush=True)
                        break
                if not extra_found:
                    print(f"    rand_{i}: *** CONFIRMED NOT FOUND after extra SA ***", flush=True)

            if (i + 1) % 50 == 0:
                print(f"  Progress: {i+1}/{n_random} tested, {found_count} found, {not_found_count} not found", flush=True)

        print(f"\n  Result: {found_count}/{n_random} representable", flush=True)
        print(f"  Methods: {dict(methods)}", flush=True)
        if not_found_count > 0:
            print(f"  *** {not_found_count} POTENTIAL COUNTEREXAMPLES ***", flush=True)

        results[f'n{n}'] = {
            'n': n, 'dim': n_dim,
            'named': fn_results,
            'random_tested': n_random,
            'random_found': found_count,
            'random_not_found': not_found_count,
            'methods': dict(methods),
            'counterexamples': counterexamples,
        }

    # Summary
    print(f"\n{'='*70}", flush=True)
    print("COUNTEREXAMPLE SEARCH SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)

    for key, res in results.items():
        n = res['n']
        named = res['named']
        nf = sum(1 for v in named.values() if v['found'])
        print(f"\nn={n} (dim={res['dim']}):", flush=True)
        print(f"  Named: {nf}/{len(named)}", flush=True)
        for name, v in named.items():
            status = f"support={v['support']}" if v['found'] else "NOT FOUND"
            print(f"    {name}: {v['method']} ({status})")
        print(f"  Random: {res['random_found']}/{res['random_tested']}")
        if res['random_not_found'] > 0:
            print(f"  *** {res['random_not_found']} potential counterexamples ***")

    has_counterexample = any(
        res['random_not_found'] > 0 or any(not v['found'] for v in res['named'].values())
        for res in results.values()
    )
    if has_counterexample:
        print("\nCONCLUSION: Potential counterexamples found — conjecture may be false for large n.")
    else:
        print("\nCONCLUSION: No counterexamples found — conjecture supported through n=6.")

    # Save
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / 'counterexample_search.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {output_path}", flush=True)


if __name__ == '__main__':
    main()
