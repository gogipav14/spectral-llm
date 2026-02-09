"""
CFL Stability Experiment: Varying Annealing Rate for Sinkhorn Router
=====================================================================

Tests whether there exists a critical annealing rate beyond which routing
training diverges, analogous to the CFL condition in numerical PDEs.

Annealing rate r = log(tau_start / tau_end) / n_steps.
We fix tau: 1.0 -> 0.1 and vary n_steps to sweep r.

Schedules:
  - Very slow:  10000 steps  (r = 0.00023)
  - Slow:        5000 steps  (r = 0.00046)  [baseline]
  - Medium:      2000 steps  (r = 0.00115)
  - Fast:         500 steps  (r = 0.00461)
  - Instant:      500 steps, tau=0.01 fixed from start
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from demo_router_learning import run_integration_demo
import json
import numpy as np
from datetime import datetime


def run_cfl_experiment():
    """Run CFL stability experiment with 5 annealing schedules."""

    schedules = [
        {
            'name': 'very_slow',
            'label': 'Very Slow (T=10000)',
            'n_steps': 10000,
            'temp_start': 1.0,
            'temp_end': 0.1,
        },
        {
            'name': 'slow',
            'label': 'Slow (T=5000, baseline)',
            'n_steps': 5000,
            'temp_start': 1.0,
            'temp_end': 0.1,
        },
        {
            'name': 'medium',
            'label': 'Medium (T=2000)',
            'n_steps': 2000,
            'temp_start': 1.0,
            'temp_end': 0.1,
        },
        {
            'name': 'fast',
            'label': 'Fast (T=500)',
            'n_steps': 500,
            'temp_start': 1.0,
            'temp_end': 0.1,
        },
        {
            'name': 'instant',
            'label': 'Instant (tau=0.01 fixed)',
            'n_steps': 500,
            'temp_start': 0.01,
            'temp_end': 0.01,
        },
    ]

    all_results = {}

    for i, sched in enumerate(schedules):
        print("\n" + "=" * 60)
        print(f"[{i+1}/{len(schedules)}] {sched['label']}")
        r = np.log(sched['temp_start'] / sched['temp_end']) / sched['n_steps'] if sched['temp_start'] != sched['temp_end'] else float('inf')
        print(f"  Annealing rate r = {r:.6f}")
        print("=" * 60)

        results = run_integration_demo(
            n_train=10000,
            n_test=5000,
            n_steps=sched['n_steps'],
            batch_size=128,
            lr=1e-2,
            temp_start=sched['temp_start'],
            temp_end=sched['temp_end'],
            seed=0,
            router_type='sinkhorn',
        )

        # Extract spectral trajectory stats
        spectral = results.get('spectral_trajectory', [])
        if spectral:
            gaps = [s['spectral_gap'] for s in spectral]
            gap_variance = float(np.var(gaps))
            gap_final = gaps[-1] if gaps else None
            entropies = [s['spectral_entropy'] for s in spectral]
            entropy_final = entropies[-1] if entropies else None
        else:
            gap_variance = None
            gap_final = None
            entropy_final = None

        all_results[sched['name']] = {
            'label': sched['label'],
            'n_steps': sched['n_steps'],
            'temp_start': sched['temp_start'],
            'temp_end': sched['temp_end'],
            'annealing_rate': r,
            'final_accuracy': results['final_accuracy'],
            'final_sparsity': results.get('final_sparsity'),
            'spectral_gap_final': gap_final,
            'spectral_gap_variance': gap_variance,
            'spectral_entropy_final': entropy_final,
            'n_spectral_samples': len(spectral),
        }

        print(f"  -> Accuracy: {results['final_accuracy']:.4f}")
        print(f"  -> Gap variance: {gap_variance:.6f}" if gap_variance is not None else "  -> No spectral data")
        print(f"  -> Final Δ: {gap_final:.4f}" if gap_final is not None else "")

    # Summary
    print("\n" + "=" * 70)
    print("CFL STABILITY EXPERIMENT — SUMMARY")
    print("=" * 70)
    print(f"{'Schedule':<25} {'Rate':>10} {'Acc':>8} {'Δ_final':>10} {'Var(Δ)':>12} {'H_σ':>8}")
    print("-" * 73)
    for name, r in all_results.items():
        rate_str = f"{r['annealing_rate']:.5f}" if r['annealing_rate'] != float('inf') else "inf"
        acc_str = f"{r['final_accuracy']:.4f}"
        gap_str = f"{r['spectral_gap_final']:.4f}" if r['spectral_gap_final'] is not None else "N/A"
        var_str = f"{r['spectral_gap_variance']:.6f}" if r['spectral_gap_variance'] is not None else "N/A"
        ent_str = f"{r['spectral_entropy_final']:.4f}" if r['spectral_entropy_final'] is not None else "N/A"
        print(f"{r['label']:<25} {rate_str:>10} {acc_str:>8} {gap_str:>10} {var_str:>12} {ent_str:>8}")

    # Save
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(exist_ok=True, parents=True)

    output = {
        'experiment': 'CFL Stability',
        'router_type': 'sinkhorn',
        'schedules': all_results,
        'timestamp': datetime.now().isoformat(),
    }

    output_path = output_dir / 'cfl_stability_experiment.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {output_path}")

    return output


if __name__ == '__main__':
    run_cfl_experiment()
