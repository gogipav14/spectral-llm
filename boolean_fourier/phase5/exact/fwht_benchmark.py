"""
Phase 5 Track 1: Exact FWHT Benchmark
=====================================

Benchmarks the Fast Walsh-Hadamard Transform for exact spectral coefficient
computation at various scales (n=10 to n=28).

Memory-stable implementation using double-buffering pattern.
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"

# Must disable x64 mode BEFORE importing jax to ensure float32 throughout
from jax import config
config.update("jax_enable_x64", False)

import jax
import jax.numpy as jnp
import jax.lax as lax
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import gc
import functools


# =============================================================================
# Memory-Stable FWHT Implementation
# =============================================================================

@functools.partial(jax.jit, static_argnums=(1,))
def fwht_jax(f: jnp.ndarray, log_n: int) -> jnp.ndarray:
    """
    Fast Walsh-Hadamard Transform using unrolled Python loop.

    Input: f of length 2^n (truth table values in {-1, +1})
    Output: Fourier coefficients (normalized by 2^n)

    Uses Python loop which gets unrolled by JIT at compile time.
    This allows static shapes at each stage while still being efficient.
    """
    n = 2 ** log_n
    x = f.astype(jnp.float32)

    # Python loop gets unrolled by JIT - shapes are static at each stage
    for stage in range(log_n):
        h = 1 << stage  # 2^stage

        # Reshape for butterfly: (n/(2h), 2, h)
        x_reshaped = x.reshape(-1, 2, h)

        # Butterfly operation: compute sums and differences
        left = x_reshaped[:, 0, :]
        right = x_reshaped[:, 1, :]

        # Interleave results back
        new_left = left + right
        new_right = left - right

        # Stack and reshape back to 1D
        x = jnp.stack([new_left, new_right], axis=1).reshape(-1)

    return x / n


@functools.partial(jax.jit, static_argnums=(1,))
def fwht_pingpong(f: jnp.ndarray, log_n: int) -> jnp.ndarray:
    """
    Memory-efficient FWHT using ping-pong buffers via lax.fori_loop.

    This version pre-allocates two buffers and swaps between them,
    minimizing memory allocations during the transform.

    Memory usage: 2 * 2^n * 4 bytes (two float32 buffers)
    """
    n = 2 ** log_n

    # Pre-allocate two buffers
    buf0 = f.astype(jnp.float32)
    buf1 = jnp.zeros_like(buf0)

    def butterfly_stage(carry, stage_idx):
        """Single butterfly stage with explicit buffer swap."""
        buf_in, buf_out = carry
        h = lax.shift_left(1, stage_idx)  # 2^stage

        # Compute butterfly indices
        # For each block of 2h elements, swap pairs at distance h
        n_blocks = n // (2 * h)

        def process_block(block_idx):
            base = block_idx * 2 * h
            # Left half indices: base to base+h-1
            # Right half indices: base+h to base+2h-1
            left_slice = lax.dynamic_slice(buf_in, (base,), (h,))
            right_slice = lax.dynamic_slice(buf_in, (base + h,), (h,))
            return left_slice + right_slice, left_slice - right_slice

        # Process all blocks (vectorized via vmap)
        block_indices = jnp.arange(n_blocks)
        sums, diffs = jax.vmap(process_block)(block_indices)

        # Interleave results into output buffer
        # sums go to even positions, diffs to odd positions within each 2h block
        buf_out = buf_out.at[::2*h].set(0)  # Reset (not strictly needed but clearer)

        # More efficient: reshape and assign
        result = jnp.empty(n, dtype=jnp.float32)
        result = result.reshape(n_blocks, 2, h)
        result = result.at[:, 0, :].set(sums)
        result = result.at[:, 1, :].set(diffs)
        result = result.reshape(n)

        return (result, buf_in), None

    # Use scan for the stages (XLA can optimize buffer reuse)
    stages = jnp.arange(log_n)
    (final_buf, _), _ = lax.scan(butterfly_stage, (buf0, buf1), stages)

    return final_buf / n


def _fwht_stage(x: jnp.ndarray, h: int) -> jnp.ndarray:
    """Single butterfly stage for FWHT."""
    n = x.shape[0]
    n_blocks = n // (2 * h)

    # Reshape to (n_blocks, 2, h) for vectorized butterfly
    y = x.reshape(n_blocks, 2, h)
    a = y[:, 0, :]
    b = y[:, 1, :]

    # Butterfly: (a+b, a-b)
    y0 = a + b
    y1 = a - b

    # Stack and flatten back
    return jnp.stack([y0, y1], axis=1).reshape(-1)


def _make_fwht_donated(log_n: int):
    """
    Create a JIT-compiled FWHT with buffer donation for given log_n.

    Buffer donation allows XLA to reuse the input buffer, reducing peak memory.
    This is critical for pushing to n=28/29 on 8GB VRAM.
    """

    def fwht_inner(x: jnp.ndarray) -> jnp.ndarray:
        """FWHT using lax.fori_loop for memory stability."""

        def body_fn(stage, x):
            h = lax.shift_left(1, stage)  # 2^stage
            n = x.shape[0]
            n_blocks = n // (2 * h)

            # Reshape for butterfly
            y = x.reshape(n_blocks, 2, h)
            a = y[:, 0, :]
            b = y[:, 1, :]

            # Butterfly operation
            y0 = a + b
            y1 = a - b

            return jnp.stack([y0, y1], axis=1).reshape(-1)

        x = x.astype(jnp.float32)
        x = lax.fori_loop(0, log_n, body_fn, x)
        return x / (2 ** log_n)

    # Compile with buffer donation
    return jax.jit(fwht_inner, donate_argnums=(0,))


# Cache compiled functions for each log_n
_FWHT_CACHE = {}


def fwht_donated(f: jnp.ndarray, log_n: int) -> jnp.ndarray:
    """
    Memory-efficient FWHT with buffer donation.

    Uses lax.fori_loop + donate_argnums for minimal peak memory.
    Best for pushing to n=28/29 on limited VRAM.

    IMPORTANT: Do not reference 'f' after this call - it may be invalidated
    by buffer donation.
    """
    if log_n not in _FWHT_CACHE:
        _FWHT_CACHE[log_n] = _make_fwht_donated(log_n)

    # Make a copy since we're donating
    f_copy = f.astype(jnp.float32)
    return _FWHT_CACHE[log_n](f_copy)


@functools.partial(jax.jit, static_argnums=(1,))
def fwht_memory_efficient(f: jnp.ndarray, log_n: int) -> jnp.ndarray:
    """
    Memory-efficient FWHT using Python loop unrolling.

    Slightly less efficient than donated version but works reliably.
    Uses at[].set() which JAX can optimize to avoid copies in many cases.
    """
    n = 2 ** log_n
    x = f.astype(jnp.float32)

    for stage in range(log_n):
        h = 1 << stage

        # Create index arrays for butterfly pairs
        block_size = 2 * h
        n_blocks = n // block_size

        # Reshape to isolate butterfly pairs
        x = x.reshape(n_blocks, 2, h)
        left = x[:, 0, :]
        right = x[:, 1, :]

        # Butterfly operation
        new_vals = jnp.stack([left + right, left - right], axis=1)
        x = new_vals.reshape(n)

    return x / n


def fwht_numpy_fallback(f, log_n: int):
    """NumPy fallback for very large n where JAX may OOM."""
    import numpy as np

    n = 2 ** log_n
    x = np.array(f, dtype=np.float32)

    for stage in range(log_n):
        h = 1 << stage
        x = x.reshape(-1, 2, h)
        left = x[:, 0, :].copy()
        right = x[:, 1, :].copy()
        x[:, 0, :] = left + right
        x[:, 1, :] = left - right
        x = x.reshape(-1)

    return x / n


# =============================================================================
# Test Function Generators
# =============================================================================

def generate_random_function(n: int, key: jax.random.PRNGKey) -> jnp.ndarray:
    """Generate a random Boolean function truth table in {-1, +1}."""
    return jax.random.choice(key, jnp.array([-1.0, 1.0]), shape=(2**n,))


def generate_k_sparse_function(
    n: int,
    k: int,
    key: jax.random.PRNGKey,
    max_degree: Optional[int] = None
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Generate a k-sparse Boolean function with known Fourier support.

    Returns: (truth_table, support_indices, coefficient_signs)

    This is for testing recovery algorithms.
    """
    # Select k random subset indices
    key1, key2 = jax.random.split(key)

    if max_degree is not None:
        # Restrict to subsets with at most max_degree bits set
        valid_subsets = []
        for s in range(2**n):
            if bin(s).count('1') <= max_degree:
                valid_subsets.append(s)
        valid_subsets = jnp.array(valid_subsets)
        k = min(k, len(valid_subsets))
        perm = jax.random.permutation(key1, len(valid_subsets))
        support = valid_subsets[perm[:k]]
    else:
        support = jax.random.choice(key1, 2**n, shape=(k,), replace=False)

    # Random signs for coefficients
    signs = jax.random.choice(key2, jnp.array([-1.0, 1.0]), shape=(k,))

    # Build Fourier representation
    coeffs = jnp.zeros(2**n)
    coeffs = coeffs.at[support].set(signs)

    # Inverse transform (WHT is self-adjoint up to scaling)
    log_n = n
    truth_table = fwht_jax(coeffs, log_n) * (2**n)

    # Quantize to {-1, +1}
    truth_table = jnp.sign(truth_table)
    truth_table = jnp.where(truth_table == 0, 1.0, truth_table)

    return truth_table, support, signs


# =============================================================================
# Benchmark Functions
# =============================================================================

def benchmark_fwht_single(n: int, n_trials: int = 5, use_gpu: bool = True) -> Dict:
    """
    Benchmark FWHT for a single value of n.

    Returns timing statistics and verification info.
    """
    log_n = n
    dim = 2 ** n
    key = jax.random.PRNGKey(42)

    # Choose backend - fwht_jax (unrolled Python loop) works for all sizes
    # The fori_loop/donated versions have tracing issues with dynamic shapes
    if use_gpu and jax.devices()[0].device_kind != 'cpu':
        # Use simple unrolled version for all GPU sizes - proven to work up to n=28
        fwht_fn = lambda f: fwht_jax(f, log_n)
        backend = 'gpu'
    else:
        fwht_fn = lambda f: fwht_numpy_fallback(f, log_n)
        backend = 'cpu'

    # Pre-compile / warm up
    f = generate_random_function(n, key)
    if backend.startswith('gpu'):
        _ = fwht_fn(f).block_until_ready()
    else:
        _ = fwht_fn(f)

    # Benchmark timing
    times = []
    for i in range(n_trials):
        key = jax.random.PRNGKey(i + 100)
        f = generate_random_function(n, key)

        gc.collect()

        if backend.startswith('gpu'):
            start = time.perf_counter()
            coeffs = fwht_fn(f).block_until_ready()
            end = time.perf_counter()
        else:
            start = time.perf_counter()
            coeffs = fwht_fn(f)
            end = time.perf_counter()

        times.append(end - start)

    # Verify on known k-sparse function
    k = min(10, dim // 100)  # Small k relative to dim
    if k >= 1:
        sparse_f, true_support, true_signs = generate_k_sparse_function(
            n, k, jax.random.PRNGKey(999)
        )
        recovered_coeffs = fwht_fn(sparse_f)
        # Count coefficients above threshold
        threshold = 0.1  # Coefficients should be ±1/dim or 0
        large_coeffs = jnp.abs(recovered_coeffs) > threshold
        n_recovered = int(jnp.sum(large_coeffs))
    else:
        n_recovered = -1
        k = 0

    return {
        'n': n,
        'dim': dim,
        'backend': backend,
        'mean_time_s': float(jnp.mean(jnp.array(times))),
        'std_time_s': float(jnp.std(jnp.array(times))),
        'min_time_s': float(min(times)),
        'max_time_s': float(max(times)),
        'throughput_coeffs_per_sec': float(dim / jnp.mean(jnp.array(times))),
        'memory_mb': float(dim * 4 / 1e6),  # float32
        'sparse_test_k': k,
        'sparse_recovered': n_recovered,
    }


def estimate_memory_usage(n: int) -> float:
    """Estimate peak memory usage in GB for FWHT at given n."""
    dim = 2 ** n
    # Input array + working buffers (roughly 3x for butterfly stages)
    bytes_needed = dim * 4 * 3
    return bytes_needed / 1e9


def run_fwht_benchmark(
    n_min: int = 10,
    n_max: int = 26,
    n_trials: int = 5,
    max_memory_gb: float = 6.0,
    verbose: bool = True
) -> Dict:
    """
    Run FWHT benchmark across a range of n values.

    Automatically stops if memory would be prohibitive.
    """
    print("=" * 70)
    print("PHASE 5 TRACK 1: Exact FWHT Benchmark")
    print("=" * 70)

    devices = jax.devices()
    device_info = str(devices[0])
    print(f"\nDevice: {device_info}")
    print(f"Testing n from {n_min} to {n_max}")
    print(f"Trials per n: {n_trials}")
    print(f"Max memory limit: {max_memory_gb:.1f} GB")

    results = []

    for n in range(n_min, n_max + 1):
        dim = 2**n
        mem_est = estimate_memory_usage(n)

        if verbose:
            print(f"\n{'─'*60}")
            print(f"n={n}: dim=2^{n}={dim:,}, memory≈{mem_est:.2f}GB")

        # Skip if memory would exceed limit
        if mem_est > max_memory_gb:
            print(f"  Skipping: estimated memory ({mem_est:.2f}GB) exceeds limit")
            break

        try:
            gc.collect()

            # Try GPU first, fall back to CPU
            try:
                result = benchmark_fwht_single(n, n_trials, use_gpu=True)
            except Exception as e:
                print(f"  GPU failed ({e}), falling back to CPU...")
                result = benchmark_fwht_single(n, n_trials, use_gpu=False)

            results.append(result)

            if verbose:
                print(f"  Backend: {result['backend']}")
                print(f"  Time: {result['mean_time_s']*1000:.2f} ± {result['std_time_s']*1000:.2f} ms")
                print(f"  Throughput: {result['throughput_coeffs_per_sec']/1e6:.2f}M coeffs/sec")
                if result['sparse_test_k'] > 0:
                    print(f"  Sparse verification: found {result['sparse_recovered']} large coeffs (expected ~{result['sparse_test_k']})")

        except Exception as e:
            print(f"  Error at n={n}: {e}")
            break

    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    if results:
        max_n = max(r['n'] for r in results)
        max_dim = max(r['dim'] for r in results)
        max_throughput = max(r['throughput_coeffs_per_sec'] for r in results)

        print(f"\nMax n achieved: {max_n}")
        print(f"Max dimension: {max_dim:,} = 2^{max_n}")
        print(f"Peak throughput: {max_throughput/1e6:.2f}M coeffs/sec")

        # Print table
        print(f"\n{'n':>4} {'dim':>12} {'backend':>8} {'time_ms':>10} {'throughput':>15} {'mem_MB':>10}")
        print("-" * 65)
        for r in results:
            print(f"{r['n']:>4} {r['dim']:>12,} {r['backend']:>8} "
                  f"{r['mean_time_s']*1000:>10.2f} "
                  f"{r['throughput_coeffs_per_sec']/1e6:>12.2f}M/s "
                  f"{r['memory_mb']:>10.1f}")

    # Save results
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        'benchmark': 'exact_fwht',
        'device': device_info,
        'n_min': n_min,
        'n_max_achieved': max(r['n'] for r in results) if results else 0,
        'n_trials': n_trials,
        'max_memory_limit_gb': max_memory_gb,
        'results': results,
        'timestamp': datetime.now().isoformat(),
    }

    output_path = output_dir / "fwht_benchmark_results.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")

    return output


# =============================================================================
# Verification on Known Functions
# =============================================================================

def verify_known_functions():
    """
    Verify FWHT on known Boolean functions with analytical Fourier coefficients.
    """
    print("\n" + "=" * 70)
    print("VERIFICATION: Known Boolean Functions")
    print("=" * 70)

    # n=2: XOR function
    # XOR in {-1,+1} encoding: f(a,b) = a*b
    # Only the χ_{ab} = χ_3 coefficient is nonzero
    print("\n--- XOR (n=2) ---")
    # Truth table: indices are (a,b) in {0,1}^2 mapped to {-1,+1}
    # Index 0=(0,0)->(-1,-1): XOR=0->+1 (FALSE in our encoding)
    # Index 1=(0,1)->(-1,+1): XOR=1->-1 (TRUE)
    # Index 2=(1,0)->(+1,-1): XOR=1->-1 (TRUE)
    # Index 3=(1,1)->(+1,+1): XOR=0->+1 (FALSE)
    xor_truth = jnp.array([1., -1., -1., 1.])
    xor_coeffs = fwht_jax(xor_truth, 2)
    print(f"Truth table: {xor_truth}")
    print(f"Fourier coefficients: {xor_coeffs}")
    print(f"Expected: [0, 0, 0, 1] (only χ_ab nonzero)")
    xor_correct = jnp.allclose(xor_coeffs, jnp.array([0., 0., 0., 1.]))
    print(f"Verification: {'PASS' if xor_correct else 'FAIL'}")

    # n=2: AND function
    # AND in {-1,+1} encoding: PTF = sign(-1 + a + b + ab)
    # Fourier coefficients: [-0.5, 0.5, 0.5, 0.5]
    print("\n--- AND (n=2) ---")
    # Only (+1,+1) -> +1 (TRUE), all others -> -1 (FALSE)
    and_truth = jnp.array([-1., -1., -1., 1.])
    and_coeffs = fwht_jax(and_truth, 2)
    print(f"Truth table: {and_truth}")
    print(f"Fourier coefficients: {and_coeffs}")
    print(f"Expected: [-0.5, 0.5, 0.5, 0.5]")
    and_correct = jnp.allclose(and_coeffs, jnp.array([-0.5, 0.5, 0.5, 0.5]))
    print(f"Verification: {'PASS' if and_correct else 'FAIL'}")

    # n=3: PARITY function
    # PARITY(a,b,c) = a*b*c -> only χ_abc = χ_7 is nonzero
    print("\n--- PARITY (n=3) ---")
    # Parity is TRUE (=-1) when odd number of +1s
    parity_truth = jnp.array([
        1.,   # 000: 0 ones (even) -> +1
        -1.,  # 001: 1 one (odd) -> -1
        -1.,  # 010: 1 one -> -1
        1.,   # 011: 2 ones -> +1
        -1.,  # 100: 1 one -> -1
        1.,   # 101: 2 ones -> +1
        1.,   # 110: 2 ones -> +1
        -1.,  # 111: 3 ones -> -1
    ])
    parity_coeffs = fwht_jax(parity_truth, 3)
    print(f"Truth table: {parity_truth}")
    print(f"Fourier coefficients: {parity_coeffs}")
    print(f"Expected: [0,0,0,0,0,0,0,-1] (only χ_abc)")
    parity_expected = jnp.array([0., 0., 0., 0., 0., 0., 0., -1.])
    parity_correct = jnp.allclose(parity_coeffs, parity_expected)
    print(f"Verification: {'PASS' if parity_correct else 'FAIL'}")

    # Sparse recovery test
    print("\n--- Sparse Recovery Verification ---")
    for n in [4, 6, 8, 10]:
        k = 5
        key = jax.random.PRNGKey(42 + n)
        sparse_f, true_support, true_signs = generate_k_sparse_function(n, k, key)
        coeffs = fwht_jax(sparse_f, n)

        # Find large coefficients
        threshold = 0.1
        recovered_support = jnp.where(jnp.abs(coeffs) > threshold)[0]

        # Check if recovered support matches (approximately - quantization may introduce small errors)
        print(f"n={n}, k={k}: recovered {len(recovered_support)} large coefficients")

    print("\n" + "=" * 70)

    return xor_correct and and_correct and parity_correct


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FWHT Benchmark")
    parser.add_argument("--min_n", type=int, default=10, help="Minimum n to test")
    parser.add_argument("--max_n", type=int, default=27, help="Maximum n to test")
    parser.add_argument("--single_n", type=int, default=None,
                        help="Run only this single n (overrides min/max)")
    parser.add_argument("--n_trials", type=int, default=5, help="Number of trials per n")
    parser.add_argument("--max_memory_gb", type=float, default=7.0,
                        help="Max memory limit in GB")
    parser.add_argument("--out_json", type=str, default=None,
                        help="Output JSON file path (default: auto-generated)")
    parser.add_argument("--skip_verify", action="store_true",
                        help="Skip verification tests")
    args = parser.parse_args()

    # First verify on known functions (unless skipped)
    if not args.skip_verify:
        verification_passed = verify_known_functions()
        if not verification_passed:
            print("\nWARNING: Verification failed! Check FWHT implementation.")

    # Determine n range
    if args.single_n is not None:
        n_min = args.single_n
        n_max = args.single_n
    else:
        n_min = args.min_n
        n_max = args.max_n

    # Run the benchmark
    results = run_fwht_benchmark(
        n_min=n_min,
        n_max=n_max,
        n_trials=args.n_trials,
        max_memory_gb=args.max_memory_gb,
        verbose=True
    )

    # Save to custom path if specified
    if args.out_json:
        with open(args.out_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults also saved to {args.out_json}")
