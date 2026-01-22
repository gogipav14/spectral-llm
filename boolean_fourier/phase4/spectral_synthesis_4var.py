"""
Spectral Synthesis for 4-Variable Functions
============================================

Extends the synthesis pipeline to n=4 (16-dim basis).
This demonstrates scalability beyond brute-force.
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import jax.random as random
from typing import Callable, Dict, Tuple
from pathlib import Path
import json
from datetime import datetime

from .spectral_synthesis import (
    synthesize_ternary_mask,
    compute_mask_accuracy,
    boolean_fourier_basis,
)


# =============================================================================
# 4-Variable Boolean Functions
# =============================================================================

def xor_4(x: jnp.ndarray) -> jnp.ndarray:
    """4-way XOR (parity): a ⊕ b ⊕ c ⊕ d"""
    return x[:, 0] * x[:, 1] * x[:, 2] * x[:, 3]


def and_4(x: jnp.ndarray) -> jnp.ndarray:
    """4-way AND: true only when all are +1"""
    # In {-1,+1}: AND_4 requires formula with larger coefficients
    # Direct computation: AND is true only when all inputs are +1
    all_positive = jnp.all(x == 1, axis=1)
    return jnp.where(all_positive, 1.0, -1.0)


def or_4(x: jnp.ndarray) -> jnp.ndarray:
    """4-way OR: true when any is +1"""
    # OR is true when any input is +1, false only when all are -1
    any_positive = jnp.any(x == 1, axis=1)
    return jnp.where(any_positive, 1.0, -1.0)


def majority_4(x: jnp.ndarray) -> jnp.ndarray:
    """4-way majority: sign(a + b + c + d)"""
    result = jnp.sign(jnp.sum(x, axis=1))
    return jnp.where(result == 0, 1.0, result)  # Tie goes to +1


def threshold_3of4(x: jnp.ndarray) -> jnp.ndarray:
    """Threshold function: true when at least 3 of 4 are +1"""
    n_positive = jnp.sum(x == 1, axis=1)
    return jnp.where(n_positive >= 3, 1.0, -1.0)


def exactly_2of4(x: jnp.ndarray) -> jnp.ndarray:
    """Exactly 2 of 4 are +1"""
    n_positive = jnp.sum(x == 1, axis=1)
    return jnp.where(n_positive == 2, 1.0, -1.0)


# Cascade compositions
def xor_ab_and_cd(x: jnp.ndarray) -> jnp.ndarray:
    """(a XOR b) AND (c AND d)"""
    ab_xor = x[:, 0] * x[:, 1]  # XOR
    cd_and = jnp.where((x[:, 2] == 1) & (x[:, 3] == 1), 1.0, -1.0)  # AND
    # AND of ab_xor and cd_and
    return jnp.where((ab_xor == 1) & (cd_and == 1), 1.0, -1.0)


def or_ab_xor_cd(x: jnp.ndarray) -> jnp.ndarray:
    """(a OR b) XOR (c OR d)"""
    ab_or = jnp.where((x[:, 0] == 1) | (x[:, 1] == 1), 1.0, -1.0)
    cd_or = jnp.where((x[:, 2] == 1) | (x[:, 3] == 1), 1.0, -1.0)
    return ab_or * cd_or  # XOR


def nested_xor(x: jnp.ndarray) -> jnp.ndarray:
    """((a XOR b) XOR c) XOR d = a*b*c*d"""
    return x[:, 0] * x[:, 1] * x[:, 2] * x[:, 3]


def implies_chain(x: jnp.ndarray) -> jnp.ndarray:
    """a → (b → (c → d))"""
    # IMPLIES(a,b) in {-1,+1}: true when ¬a ∨ b
    def implies(p, q):
        return jnp.where((p == -1) | (q == 1), 1.0, -1.0)

    c_impl_d = implies(x[:, 2], x[:, 3])
    b_impl_cd = implies(x[:, 1], c_impl_d)
    return implies(x[:, 0], b_impl_cd)


# Operation registry
PHASE4_OPERATIONS = {
    # Pure 4-variable operations
    'xor_4': xor_4,
    'and_4': and_4,
    'or_4': or_4,
    'majority_4': majority_4,
    'threshold_3of4': threshold_3of4,
    'exactly_2of4': exactly_2of4,

    # Cascade compositions
    'xor_ab_and_cd': xor_ab_and_cd,
    'or_ab_xor_cd': or_ab_xor_cd,
    'nested_xor': nested_xor,
    'implies_chain': implies_chain,
}


# =============================================================================
# Subset naming for 4 variables
# =============================================================================

def get_subset_name_4var(subset_idx: int) -> str:
    """
    Get human-readable name for a 4-variable subset.

    Basis ordering: [1, d, c, cd, b, bd, bc, bcd, a, ad, ac, acd, ab, abd, abc, abcd]
    (Gray code pattern with bits ordered as a,b,c,d from MSB to LSB)
    """
    vars = ['a', 'b', 'c', 'd']
    bits = format(subset_idx, '04b')
    name = ''.join([v for v, b in zip(vars, bits) if b == '1'])
    return name if name else '1'


CHAR_NAMES_4VAR = [get_subset_name_4var(i) for i in range(16)]


# =============================================================================
# Run Synthesis for All Operations
# =============================================================================

def run_phase4_synthesis(
    n_estimation_samples: int = 50000,
    n_refinement_samples: int = 2000,
    n_mcmc_steps: int = 500,
    verbose: bool = True
):
    """Run spectral synthesis for all Phase 4 operations."""
    print("=" * 70)
    print("PHASE 4: Spectral Synthesis for 4-Variable Functions")
    print("=" * 70)
    print(f"\nBasis dimension: 16")
    print(f"Characters: {CHAR_NAMES_4VAR}")

    results = {}

    for op_name, op_fn in PHASE4_OPERATIONS.items():
        print(f"\n{'─'*60}")
        print(f"Synthesizing: {op_name}")
        print(f"{'─'*60}")

        mask, info = synthesize_ternary_mask(
            op_fn,
            n_vars=4,
            n_estimation_samples=n_estimation_samples,
            n_refinement_samples=n_refinement_samples,
            n_mcmc_steps=n_mcmc_steps,
            quantization_threshold=0.1,
            use_parallel_tempering=True,
            verbose=verbose
        )

        results[op_name] = info

    # Summary
    print("\n" + "=" * 70)
    print("SYNTHESIS SUMMARY")
    print("=" * 70)

    print(f"\nBasis: {CHAR_NAMES_4VAR}")
    print("\nSynthesized masks:")

    n_perfect = 0
    for op_name, info in results.items():
        mask = info['final_mask']
        mask_str = ''.join(['+' if x > 0 else '-' if x < 0 else '0' for x in mask])
        acc = info['final_accuracy']
        support = info['support']

        status = "✅" if acc > 0.999 else "⚠️"
        if acc > 0.999:
            n_perfect += 1

        # Show non-zero characters
        active_chars = [CHAR_NAMES_4VAR[i] for i, v in enumerate(mask) if v != 0]

        print(f"  {status} {op_name:20s}: [{mask_str}]")
        print(f"       acc={acc:.2%} support={support}/16 chars={active_chars}")

    print(f"\nPerfect operations: {n_perfect}/{len(PHASE4_OPERATIONS)}")

    # Sparsity analysis
    supports = [info['support'] for info in results.values()]
    mean_support = sum(supports) / len(supports)
    mean_sparsity = 1 - mean_support / 16

    print(f"Mean support: {mean_support:.1f}/16")
    print(f"Mean sparsity: {mean_sparsity:.0%}")

    # Save results
    checkpoint_dir = Path("v6/checkpoints/phase4_synthesis")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    results_dict = {
        'n_operations': len(results),
        'n_perfect': n_perfect,
        'mean_support': mean_support,
        'mean_sparsity': mean_sparsity,
        'operations': {
            name: {
                'mask': [int(x) for x in info['final_mask']],
                'accuracy': float(info['final_accuracy']),
                'support': info['support'],
                'raw_coefficients': [float(x) for x in info['raw_coefficients']],
            }
            for name, info in results.items()
        },
        'basis': CHAR_NAMES_4VAR,
        'timestamp': datetime.now().isoformat(),
    }

    results_path = checkpoint_dir / "phase4_synthesis_results.json"
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"\nResults saved to {results_path}")

    return results


if __name__ == "__main__":
    results = run_phase4_synthesis(
        n_estimation_samples=50000,
        n_refinement_samples=2000,
        n_mcmc_steps=500,
        verbose=True
    )
