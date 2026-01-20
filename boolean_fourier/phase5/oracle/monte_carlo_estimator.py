"""
Phase 5 Track 2: Monte Carlo Coefficient Estimation
====================================================

Estimates individual Fourier coefficients via Monte Carlo sampling
from an oracle (black-box Boolean function).

IMPORTANT: This is a baseline Monte Carlo estimator, NOT a Goldreich-Levin
or KM-style bucket-splitting algorithm. Those algorithms achieve poly(n/ε)
query complexity for finding all ε-heavy coefficients; this baseline uses
more queries but is simpler to implement.

Theoretical Background:
-----------------------
For a Boolean function f: {-1,+1}^n → {-1,+1}, each Fourier coefficient is:

    f̂(S) = E_x[f(x) · χ_S(x)] = (1/2^n) Σ_x f(x) · χ_S(x)

where χ_S(x) = Π_{i∈S} x_i is the parity character.

Monte Carlo estimation: sample m random x ~ Uniform({-1,+1}^n), compute
    f̂(S) ≈ (1/m) Σ_{j=1}^m f(x_j) · χ_S(x_j)

By Hoeffding, with m = O(1/ε² · log(1/δ)) samples we get |estimate - true| ≤ ε
with probability ≥ 1-δ.

Search Strategy:
---------------
We search for heavy coefficients by:
1. Restricting to low-degree subsets (LMN-motivated: Fourier mass concentrates
   on low-degree terms for bounded-depth circuits)
2. Estimating all candidate coefficients via Monte Carlo
3. Returning those above a threshold τ

This is NOT optimal query-wise but is simple and practical for moderate n.
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import jax.random as random
from typing import Callable, Dict, List, Tuple, Optional
from dataclasses import dataclass
import time
import json
from datetime import datetime
from pathlib import Path
import gc


@dataclass
class EstimationResult:
    """Result of coefficient estimation."""
    subset_idx: int
    estimated_value: float
    n_samples: int
    std_error: float


def parity_char(x: jnp.ndarray, S: int, n: int) -> jnp.ndarray:
    """
    Compute χ_S(x) = Π_{i∈S} x_i for batch of inputs.

    Args:
        x: (batch, n) array of {-1,+1} values
        S: integer encoding the subset (bit i set means variable i in S)
        n: number of variables

    Returns:
        (batch,) array of {-1,+1} values
    """
    batch_size = x.shape[0]
    result = jnp.ones(batch_size)

    for i in range(n):
        if (S >> i) & 1:
            result = result * x[:, i]

    return result


def estimate_coefficient(
    oracle: Callable[[jnp.ndarray], jnp.ndarray],
    S: int,
    n: int,
    n_samples: int,
    key: jax.random.PRNGKey
) -> EstimationResult:
    """
    Estimate f̂(S) via Monte Carlo sampling.

    Args:
        oracle: function f: (batch, n) -> (batch,) in {-1,+1}
        S: subset index to estimate
        n: number of variables
        n_samples: number of samples for estimation
        key: PRNG key

    Returns:
        EstimationResult with estimate and uncertainty
    """
    # Generate random inputs
    x = random.choice(key, jnp.array([-1.0, 1.0]), shape=(n_samples, n))

    # Query oracle
    f_vals = oracle(x)

    # Compute parity character
    chi_vals = parity_char(x, S, n)

    # Estimate coefficient
    products = f_vals * chi_vals
    estimate = jnp.mean(products)
    std_error = jnp.std(products) / jnp.sqrt(n_samples)

    return EstimationResult(
        subset_idx=S,
        estimated_value=float(estimate),
        n_samples=n_samples,
        std_error=float(std_error)
    )


def enumerate_low_degree_subsets(n: int, max_degree: int) -> List[int]:
    """
    Enumerate all subsets with at most max_degree elements.

    Returns list of subset indices (as integers).
    """
    subsets = []
    for S in range(2**n):
        if bin(S).count('1') <= max_degree:
            subsets.append(S)
    return subsets


def count_low_degree_subsets(n: int, max_degree: int) -> int:
    """Count subsets with at most max_degree elements: Σ_{k=0}^d C(n,k)."""
    from math import comb
    return sum(comb(n, k) for k in range(max_degree + 1))


class MonteCarloEstimator:
    """
    Monte Carlo estimator for Fourier coefficients.

    This is a BASELINE estimator, not a Goldreich-Levin algorithm.
    Query complexity: O(n^d / ε² · log(1/δ)) for degree-d search.
    """

    def __init__(
        self,
        oracle: Callable[[jnp.ndarray], jnp.ndarray],
        n: int,
        max_degree: int = 3,
        samples_per_coeff: int = 10000,
        threshold: float = 0.1
    ):
        """
        Args:
            oracle: Boolean function f: (batch, n) -> (batch,) in {-1,+1}
            n: number of variables
            max_degree: search subsets up to this degree (LMN-motivated)
            samples_per_coeff: Monte Carlo samples per coefficient
            threshold: report coefficients with |f̂(S)| > threshold
        """
        self.oracle = oracle
        self.n = n
        self.max_degree = max_degree
        self.samples_per_coeff = samples_per_coeff
        self.threshold = threshold

    def find_heavy_coefficients(
        self,
        key: jax.random.PRNGKey,
        verbose: bool = True
    ) -> Dict:
        """
        Find all coefficients above threshold.

        Returns dictionary with heavy coefficients and statistics.
        """
        # Enumerate candidate subsets
        candidates = enumerate_low_degree_subsets(self.n, self.max_degree)
        n_candidates = len(candidates)

        if verbose:
            print(f"Searching {n_candidates} subsets (n={self.n}, max_degree={self.max_degree})")
            print(f"Samples per coefficient: {self.samples_per_coeff}")
            print(f"Total queries: {n_candidates * self.samples_per_coeff:,}")

        # Estimate all candidates
        heavy = []
        all_estimates = []
        total_queries = 0

        start_time = time.perf_counter()

        for i, S in enumerate(candidates):
            key, subkey = random.split(key)
            result = estimate_coefficient(
                self.oracle, S, self.n,
                self.samples_per_coeff, subkey
            )
            total_queries += self.samples_per_coeff

            all_estimates.append(result)

            if abs(result.estimated_value) > self.threshold:
                heavy.append(result)

            if verbose and (i + 1) % max(1, n_candidates // 10) == 0:
                print(f"  Progress: {i+1}/{n_candidates} ({100*(i+1)/n_candidates:.0f}%)")

        elapsed = time.perf_counter() - start_time

        if verbose:
            print(f"\nFound {len(heavy)} heavy coefficients (|f̂(S)| > {self.threshold})")
            print(f"Total time: {elapsed:.2f}s")
            print(f"Queries/sec: {total_queries/elapsed:,.0f}")

        # Sort heavy coefficients by magnitude
        heavy.sort(key=lambda r: -abs(r.estimated_value))

        return {
            'n': self.n,
            'max_degree': self.max_degree,
            'threshold': self.threshold,
            'samples_per_coeff': self.samples_per_coeff,
            'n_candidates': n_candidates,
            'total_queries': total_queries,
            'n_heavy': len(heavy),
            'elapsed_s': elapsed,
            'queries_per_sec': total_queries / elapsed,
            'heavy_coefficients': [
                {
                    'subset': r.subset_idx,
                    'subset_binary': format(r.subset_idx, f'0{self.n}b'),
                    'estimate': r.estimated_value,
                    'std_error': r.std_error,
                }
                for r in heavy
            ],
        }


# =============================================================================
# Test Oracle Functions
# =============================================================================

def make_k_sparse_parity_oracle(
    n: int,
    k: int,
    key: jax.random.PRNGKey
) -> Tuple[Callable, List[int], List[float]]:
    """
    Create a k-sparse parity mixture oracle with UNKNOWN support.

    f(x) = sign(Σ_{j=1}^k a_j · χ_{S_j}(x))

    Returns: (oracle_fn, true_supports, true_coefficients)

    This is a proper test for recovery algorithms since the support
    is not known a priori.
    """
    key1, key2 = random.split(key)

    # Random support: k distinct subsets
    all_subsets = jnp.arange(2**n)
    perm = random.permutation(key1, 2**n)
    supports = perm[:k]

    # Random nonzero coefficients (magnitudes between 0.5 and 1.0)
    key2a, key2b = random.split(key2)
    magnitudes = random.uniform(key2a, shape=(k,), minval=0.3, maxval=1.0)
    signs = random.choice(key2b, jnp.array([-1.0, 1.0]), shape=(k,))
    coefficients = magnitudes * signs

    # Pre-compute for efficiency
    supports_list = [int(s) for s in supports]
    coeffs_array = jnp.array(coefficients)

    def oracle(x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate k-sparse parity mixture."""
        batch_size = x.shape[0]
        total = jnp.zeros(batch_size)

        for j, S in enumerate(supports_list):
            chi = parity_char(x, S, n)
            total = total + coeffs_array[j] * chi

        return jnp.sign(total)

    return oracle, supports_list, [float(c) for c in coefficients]


def make_known_function_oracle(func_name: str, n: int) -> Callable:
    """Create oracle for known Boolean functions."""

    if func_name == 'parity':
        # PARITY_n = χ_{all ones}
        def oracle(x):
            return jnp.prod(x, axis=1)
        return oracle

    elif func_name == 'majority':
        # MAJORITY_n = sign(Σ x_i)
        def oracle(x):
            s = jnp.sum(x, axis=1)
            return jnp.where(s >= 0, 1.0, -1.0)
        return oracle

    elif func_name == 'and':
        # AND_n: true only when all inputs are +1
        def oracle(x):
            all_pos = jnp.all(x == 1, axis=1)
            return jnp.where(all_pos, 1.0, -1.0)
        return oracle

    elif func_name == 'or':
        # OR_n: false only when all inputs are -1
        def oracle(x):
            any_pos = jnp.any(x == 1, axis=1)
            return jnp.where(any_pos, 1.0, -1.0)
        return oracle

    else:
        raise ValueError(f"Unknown function: {func_name}")


# =============================================================================
# Benchmarks
# =============================================================================

def benchmark_sparse_recovery(
    n_values: List[int] = [8, 12, 16, 20],
    k_values: List[int] = [3, 5, 10],
    max_degree: int = 4,
    samples_per_coeff: int = 5000,
    threshold: float = 0.15,
    n_trials: int = 3,
    verbose: bool = True
) -> Dict:
    """
    Benchmark recovery of k-sparse parity mixtures.

    Reports precision, recall, and query efficiency.
    """
    print("=" * 70)
    print("PHASE 5 TRACK 2: Monte Carlo Coefficient Recovery Benchmark")
    print("=" * 70)
    print(f"\nMethod: Monte Carlo estimation (NOT Goldreich-Levin)")
    print(f"Search restriction: degree ≤ {max_degree}")
    print(f"Samples per coefficient: {samples_per_coeff}")
    print(f"Detection threshold: {threshold}")

    results = []

    for n in n_values:
        # Check if search space is tractable
        n_candidates = count_low_degree_subsets(n, max_degree)
        if n_candidates > 1e7:  # > 10M candidates
            print(f"\nn={n}: Skipping (too many candidates: {n_candidates:,})")
            continue

        for k in k_values:
            if k > n_candidates:
                continue

            print(f"\n{'─'*60}")
            print(f"n={n}, k={k} (searching {n_candidates:,} candidates)")

            trial_results = []

            for trial in range(n_trials):
                key = random.PRNGKey(42 + trial * 1000 + n * 100 + k)

                # Create oracle with unknown support
                key1, key2 = random.split(key)
                oracle, true_support, true_coeffs = make_k_sparse_parity_oracle(
                    n, k, key1
                )

                # Restrict true support to low-degree for fair comparison
                true_low_degree = [
                    s for s in true_support
                    if bin(s).count('1') <= max_degree
                ]

                # Run estimator
                estimator = MonteCarloEstimator(
                    oracle=oracle,
                    n=n,
                    max_degree=max_degree,
                    samples_per_coeff=samples_per_coeff,
                    threshold=threshold
                )

                result = estimator.find_heavy_coefficients(key2, verbose=False)

                # Compute precision/recall
                recovered_support = set(r['subset'] for r in result['heavy_coefficients'])
                true_support_set = set(true_low_degree)

                true_positives = len(recovered_support & true_support_set)
                false_positives = len(recovered_support - true_support_set)
                false_negatives = len(true_support_set - recovered_support)

                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                recall = true_positives / len(true_support_set) if len(true_support_set) > 0 else 1.0

                trial_results.append({
                    'precision': precision,
                    'recall': recall,
                    'true_positives': true_positives,
                    'false_positives': false_positives,
                    'false_negatives': false_negatives,
                    'n_true_low_degree': len(true_low_degree),
                    'total_queries': result['total_queries'],
                    'elapsed_s': result['elapsed_s'],
                })

            # Aggregate trial results
            avg_precision = sum(t['precision'] for t in trial_results) / len(trial_results)
            avg_recall = sum(t['recall'] for t in trial_results) / len(trial_results)
            avg_queries = sum(t['total_queries'] for t in trial_results) / len(trial_results)
            avg_time = sum(t['elapsed_s'] for t in trial_results) / len(trial_results)

            result_entry = {
                'n': n,
                'k': k,
                'max_degree': max_degree,
                'n_candidates': n_candidates,
                'n_trials': n_trials,
                'avg_precision': avg_precision,
                'avg_recall': avg_recall,
                'avg_queries': avg_queries,
                'avg_time_s': avg_time,
                'trial_details': trial_results,
            }
            results.append(result_entry)

            if verbose:
                print(f"  Precision: {avg_precision:.2%} ± {jnp.std(jnp.array([t['precision'] for t in trial_results])):.2%}")
                print(f"  Recall: {avg_recall:.2%} ± {jnp.std(jnp.array([t['recall'] for t in trial_results])):.2%}")
                print(f"  Queries: {avg_queries:,.0f}")
                print(f"  Time: {avg_time:.2f}s")

    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    print(f"\n{'n':>4} {'k':>4} {'#cand':>10} {'precision':>10} {'recall':>10} {'queries':>12}")
    print("-" * 60)
    for r in results:
        print(f"{r['n']:>4} {r['k']:>4} {r['n_candidates']:>10,} "
              f"{r['avg_precision']:>10.1%} {r['avg_recall']:>10.1%} "
              f"{r['avg_queries']:>12,.0f}")

    # Save results
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        'benchmark': 'monte_carlo_recovery',
        'method': 'Monte Carlo estimation (NOT Goldreich-Levin)',
        'max_degree': max_degree,
        'samples_per_coeff': samples_per_coeff,
        'threshold': threshold,
        'n_trials': n_trials,
        'results': results,
        'timestamp': datetime.now().isoformat(),
    }

    output_path = output_dir / "oracle_recovery_results.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")

    return output


def verify_known_functions():
    """Verify recovery on known Boolean functions."""
    print("\n" + "=" * 70)
    print("VERIFICATION: Known Boolean Functions")
    print("=" * 70)

    test_cases = [
        ('parity', 4, 4),   # PARITY_4: single coefficient at χ_{1111}
        ('majority', 4, 3), # MAJORITY_4: low-degree concentration
        ('and', 3, 3),      # AND_3: all coefficients nonzero
    ]

    for func_name, n, max_deg in test_cases:
        print(f"\n--- {func_name.upper()}_{n} ---")

        oracle = make_known_function_oracle(func_name, n)
        estimator = MonteCarloEstimator(
            oracle=oracle,
            n=n,
            max_degree=max_deg,
            samples_per_coeff=10000,
            threshold=0.05
        )

        result = estimator.find_heavy_coefficients(random.PRNGKey(42), verbose=False)

        print(f"Found {result['n_heavy']} heavy coefficients:")
        for coeff in result['heavy_coefficients'][:5]:  # Show top 5
            print(f"  S={coeff['subset_binary']}: f̂={coeff['estimate']:.4f} ± {coeff['std_error']:.4f}")


if __name__ == "__main__":
    # First verify on known functions
    verify_known_functions()

    # Then run the sparse recovery benchmark
    results = benchmark_sparse_recovery(
        n_values=[8, 12, 16],
        k_values=[3, 5],
        max_degree=3,
        samples_per_coeff=5000,
        threshold=0.15,
        n_trials=3,
        verbose=True
    )
