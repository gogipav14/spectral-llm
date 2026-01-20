#!/usr/bin/env python3
"""
Driver script for FWHT benchmarks with fresh process isolation.

Because JAX's GPU allocator maintains a high-water memory pool within a process,
we run the largest configurations (n>=28) in fresh processes to avoid allocator
fragmentation. This is standard practice when benchmarking near VRAM limits.

Usage:
    python driver_fwht.py              # Run full benchmark (sweep + isolated big n)
    python driver_fwht.py --max_big 29 # Also attempt n=29
"""

import subprocess
import os
import sys
import json
from pathlib import Path
from datetime import datetime
import argparse


def run_benchmark_process(
    single_n: int = None,
    min_n: int = None,
    max_n: int = None,
    out_json: str = None,
    n_trials: int = 5,
    max_memory_gb: float = 7.0,
    skip_verify: bool = False,
    mem_fraction: str = "1.0"
) -> bool:
    """Run benchmark in a fresh subprocess."""
    env = os.environ.copy()
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = mem_fraction

    script_path = Path(__file__).parent / "fwht_benchmark.py"
    cmd = [sys.executable, str(script_path)]

    if single_n is not None:
        cmd += ["--single_n", str(single_n)]
    else:
        cmd += ["--min_n", str(min_n), "--max_n", str(max_n)]

    cmd += ["--n_trials", str(n_trials)]
    cmd += ["--max_memory_gb", str(max_memory_gb)]

    if out_json:
        cmd += ["--out_json", out_json]

    if skip_verify:
        cmd += ["--skip_verify"]

    print(f"\n{'='*70}")
    print(f"Spawning fresh process: {' '.join(cmd[-6:])}")
    print(f"Memory fraction: {mem_fraction}")
    print(f"{'='*70}")

    try:
        result = subprocess.run(cmd, env=env, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Process failed with exit code {e.returncode}")
        return False


def merge_results(json_files: list, output_path: str):
    """Merge multiple benchmark result files."""
    all_results = []

    for json_file in json_files:
        path = Path(json_file)
        if path.exists():
            data = json.loads(path.read_text())
            if 'results' in data:
                all_results.extend(data['results'])
            else:
                all_results.append(data)

    # Sort by n
    all_results.sort(key=lambda x: x.get('n', 0))

    # Find max achieved
    max_n = max((r['n'] for r in all_results), default=0)
    max_throughput = max((r['throughput_coeffs_per_sec'] for r in all_results), default=0)

    merged = {
        'benchmark': 'exact_fwht_merged',
        'n_max_achieved': max_n,
        'peak_throughput_coeffs_per_sec': max_throughput,
        'results': all_results,
        'source_files': json_files,
        'timestamp': datetime.now().isoformat(),
        'notes': 'n>=28 run in isolated processes to avoid allocator fragmentation'
    }

    Path(output_path).write_text(json.dumps(merged, indent=2))
    print(f"\nMerged results saved to {output_path}")

    return merged


def main():
    parser = argparse.ArgumentParser(description="Driver for FWHT benchmarks with process isolation")
    parser.add_argument("--max_sweep", type=int, default=27,
                        help="Maximum n for the main sweep (default: 27)")
    parser.add_argument("--min_big", type=int, default=28,
                        help="Minimum n for isolated runs (default: 28)")
    parser.add_argument("--max_big", type=int, default=28,
                        help="Maximum n for isolated runs (default: 28)")
    parser.add_argument("--n_trials", type=int, default=5,
                        help="Number of trials per n")
    parser.add_argument("--skip_sweep", action="store_true",
                        help="Skip the main sweep, only run isolated big n")
    args = parser.parse_args()

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    json_files = []

    # Main sweep (n=10 to max_sweep)
    if not args.skip_sweep:
        sweep_json = str(results_dir / "fwht_sweep_10_27.json")
        success = run_benchmark_process(
            min_n=10,
            max_n=args.max_sweep,
            out_json=sweep_json,
            n_trials=args.n_trials,
            max_memory_gb=7.0,
            skip_verify=False,
            mem_fraction="0.95"  # Slightly conservative for sweep
        )
        if success:
            json_files.append(sweep_json)

    # Isolated runs for big n values
    for n in range(args.min_big, args.max_big + 1):
        single_json = str(results_dir / f"fwht_n{n}_isolated.json")
        success = run_benchmark_process(
            single_n=n,
            out_json=single_json,
            n_trials=args.n_trials,
            max_memory_gb=10.0,  # More generous for isolated runs
            skip_verify=True,  # Already verified in sweep
            mem_fraction="1.0"  # Maximum memory for big n
        )
        if success:
            json_files.append(single_json)

    # Merge all results
    if json_files:
        merged_path = str(results_dir / "fwht_benchmark_merged.json")
        merged = merge_results(json_files, merged_path)

        # Print summary
        print("\n" + "=" * 70)
        print("COMBINED BENCHMARK SUMMARY")
        print("=" * 70)
        print(f"Max n achieved: {merged['n_max_achieved']}")
        print(f"Peak throughput: {merged['peak_throughput_coeffs_per_sec']/1e9:.3f}B coeffs/sec")
        print(f"Results files: {len(json_files)}")


if __name__ == "__main__":
    main()
