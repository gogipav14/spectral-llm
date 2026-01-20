#!/usr/bin/env python3
"""Minimal n=28 FWHT test with maximum memory allocation."""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"

# Ensure float32 only - MUST be before importing jax
from jax import config
config.update("jax_enable_x64", False)

import jax
import jax.numpy as jnp
import time
import gc
import functools
import json
from datetime import datetime
from pathlib import Path

print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")
print(f"Memory fraction: 1.0 (maximum)")

@functools.partial(jax.jit, static_argnums=(1,))
def fwht_jax(f: jnp.ndarray, log_n: int) -> jnp.ndarray:
    """FWHT using unrolled Python loop."""
    n = 2 ** log_n
    x = f.astype(jnp.float32)

    for stage in range(log_n):
        h = 1 << stage
        x_reshaped = x.reshape(-1, 2, h)
        left = x_reshaped[:, 0, :]
        right = x_reshaped[:, 1, :]
        new_left = left + right
        new_right = left - right
        x = jnp.stack([new_left, new_right], axis=1).reshape(-1)

    return x / n

def test_n(n: int, n_trials: int = 3):
    """Test FWHT at given n."""
    dim = 2 ** n
    memory_gb = dim * 4 / 1e9
    print(f"\n{'='*60}")
    print(f"Testing n={n}")
    print(f"  Dimension: {dim:,}")
    print(f"  Theoretical memory: {memory_gb:.2f} GB (float32)")

    gc.collect()

    try:
        key = jax.random.PRNGKey(42)
        f = jax.random.choice(key, jnp.array([-1.0, 1.0], dtype=jnp.float32), shape=(dim,))
        print(f"  Input allocated, dtype: {f.dtype}")

        # Warmup/compile
        print("  Compiling...")
        coeffs = fwht_jax(f, n).block_until_ready()
        print(f"  Compiled! Output dtype: {coeffs.dtype}")

        # Timing (multiple trials)
        gc.collect()
        times = []
        for trial in range(n_trials):
            start = time.perf_counter()
            coeffs = fwht_jax(f, n).block_until_ready()
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        mean_time = sum(times) / len(times)
        throughput = dim / mean_time / 1e9

        print(f"  Times: {[f'{t*1000:.1f}ms' for t in times]}")
        print(f"  Mean: {mean_time*1000:.2f} ms")
        print(f"  Throughput: {throughput:.3f} B coeffs/sec")
        print(f"  SUCCESS!")

        return {
            'n': n,
            'dim': dim,
            'backend': 'gpu',
            'mean_time_s': mean_time,
            'times_ms': [t*1000 for t in times],
            'throughput_coeffs_per_sec': throughput * 1e9,
            'memory_gb_theoretical': memory_gb,
            'success': True
        }

    except Exception as e:
        print(f"  FAILED: {e}")
        return {
            'n': n,
            'dim': dim,
            'success': False,
            'error': str(e)
        }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=28, help="n to test")
    parser.add_argument("--warmup_n", type=int, default=27, help="warmup n (smaller)")
    parser.add_argument("--n_trials", type=int, default=3, help="number of timing trials")
    parser.add_argument("--out_json", type=str, default=None, help="output JSON path")
    args = parser.parse_args()

    results = []

    # Warmup with smaller n first
    if args.warmup_n < args.n:
        print(f"Warming up with n={args.warmup_n}...")
        warmup_result = test_n(args.warmup_n, n_trials=1)
        if warmup_result['success']:
            results.append(warmup_result)

    # Test target n
    result = test_n(args.n, n_trials=args.n_trials)
    results.append(result)

    # Save results
    output = {
        'benchmark': 'fwht_minimal_test',
        'results': results,
        'timestamp': datetime.now().isoformat(),
        'settings': {
            'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
            'XLA_PYTHON_CLIENT_MEM_FRACTION': '1.0',
            'jax_enable_x64': False
        }
    }

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(output, indent=2))
        print(f"\nResults saved to {args.out_json}")
    else:
        # Default output path
        out_path = Path(__file__).parent.parent / "results" / f"fwht_n{args.n}_minimal.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2))
        print(f"\nResults saved to {out_path}")
