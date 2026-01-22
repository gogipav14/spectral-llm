"""
Phase 4 Warm-Start Experiment (v2)
==================================

The "killer experiment" that addresses "why not just SAT/MCMC?"

Compares 4 initialization conditions for MCMC refinement:
  (A) Random ternary init
  (B) WHT-threshold init
  (C) GD warm start (2000 steps, random init)
  (D) WHT→GD (WHT init + 2000 GD steps)

Reports MCMC steps to 100% accuracy for each condition.
Logs Jaccard trajectories and eigenspectrum for GD conditions.
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import optax
from functools import partial
from typing import Callable, Dict, Tuple, List
from datetime import datetime
import json

from .spectral_synthesis import (
    boolean_fourier_basis,
    estimate_fourier_coefficients,
    quantize_to_ternary,
    compute_mask_accuracy,
    refine_mask_parallel_tempering,
)
from .spectral_synthesis_4var import PHASE4_OPERATIONS, CHAR_NAMES_4VAR
from ..utils.diagnostics import (
    jaccard_trajectory,
    eigenspectrum_svd,
    spectral_compression_summary,
    DiagnosticsLogger,
    GD_PROTOCOL,
    load_phase4_masks,
)


# =============================================================================
# Gumbel-Softmax GD Training
# =============================================================================

def gumbel_softmax_sample(logits: jnp.ndarray, temperature: float, key: jax.Array) -> jnp.ndarray:
    """Sample from Gumbel-Softmax distribution.

    Args:
        logits: (n_coeffs, 3) logits for each ternary value
        temperature: softmax temperature
        key: random key

    Returns:
        (n_coeffs,) soft ternary values in [-1, 1]
    """
    gumbel_noise = -jnp.log(-jnp.log(random.uniform(key, logits.shape) + 1e-10) + 1e-10)
    soft_samples = jax.nn.softmax((logits + gumbel_noise) / temperature, axis=-1)
    # Map to ternary: [p(-1), p(0), p(+1)] -> weighted sum
    values = jnp.array([-1.0, 0.0, 1.0])
    return jnp.sum(soft_samples * values, axis=-1)


def gumbel_softmax_hard(logits: jnp.ndarray) -> jnp.ndarray:
    """Hard quantization of Gumbel-Softmax logits.

    Args:
        logits: (n_coeffs, 3) logits

    Returns:
        (n_coeffs,) hard ternary values in {-1, 0, +1}
    """
    values = jnp.array([-1.0, 0.0, 1.0])
    idx = jnp.argmax(logits, axis=-1)
    return values[idx]


def compute_gd_loss(
    logits: jnp.ndarray,
    x: jnp.ndarray,
    targets: jnp.ndarray,
    n_vars: int,
    temperature: float,
    key: jax.Array
) -> Tuple[float, jnp.ndarray]:
    """Compute cross-entropy loss for GD training.

    Args:
        logits: (n_coeffs, 3) trainable parameters
        x: (batch, n_vars) inputs in {-1, +1}
        targets: (batch,) target outputs
        n_vars: number of variables
        temperature: Gumbel-Softmax temperature
        key: random key

    Returns:
        loss: scalar loss
        w_soft: (n_coeffs,) soft weights for Jaccard tracking
    """
    w_soft = gumbel_softmax_sample(logits, temperature, key)

    # Compute output
    basis = boolean_fourier_basis(x, n_vars)
    output = jnp.sum(basis * w_soft, axis=-1)

    # Binary cross-entropy style loss (soft)
    # Map targets from {-1, +1} to {0, 1}
    y = (targets + 1) / 2  # {0, 1}
    p = jax.nn.sigmoid(output * 5.0)  # Sharper sigmoid

    loss = -jnp.mean(y * jnp.log(p + 1e-10) + (1 - y) * jnp.log(1 - p + 1e-10))

    return loss, w_soft


@partial(jax.jit, static_argnums=(1,))
def compute_gd_grads(
    logits: jnp.ndarray,
    n_vars: int,
    x: jnp.ndarray,
    targets: jnp.ndarray,
    temperature: float,
    key: jax.Array,
):
    """Compute gradients for GD training step."""
    def loss_fn(logits):
        loss, _ = compute_gd_loss(logits, x, targets, n_vars, temperature, key)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(logits)
    return loss, grads


def gd_step(
    logits: jnp.ndarray,
    opt_state,
    n_vars: int,
    x: jnp.ndarray,
    targets: jnp.ndarray,
    temperature: float,
    key: jax.Array,
    optimizer
):
    """One GD training step (non-jitted wrapper)."""
    loss, grads = compute_gd_grads(logits, n_vars, x, targets, temperature, key)
    updates, opt_state = optimizer.update(grads, opt_state, logits)
    logits = optax.apply_updates(logits, updates)

    return logits, opt_state, loss


def train_gd_warmstart(
    target_fn: Callable,
    n_vars: int,
    initial_logits: jnp.ndarray = None,
    n_steps: int = 2000,
    lr: float = 1e-2,
    batch_size: int = 256,
    log_every: int = 100,
    seed: int = 0,
    verbose: bool = True
) -> Tuple[jnp.ndarray, List[jnp.ndarray], List[float]]:
    """Train via GD with Gumbel-Softmax.

    Args:
        target_fn: Boolean function to learn
        n_vars: Number of variables (4 for Phase 4)
        initial_logits: Optional initial logits (for WHT→GD condition)
        n_steps: Number of training steps
        lr: Learning rate
        batch_size: Batch size
        log_every: How often to log soft weights
        seed: Random seed
        verbose: Print progress

    Returns:
        final_logits: (n_coeffs, 3) trained logits
        W_log: List of (n_coeffs,) soft weights at each log step
        acc_log: List of accuracies at each log step
    """
    key = random.PRNGKey(seed)
    n_coeffs = 2 ** n_vars

    # Initialize logits
    if initial_logits is None:
        key, init_key = random.split(key)
        logits = random.normal(init_key, (n_coeffs, 3)) * 0.1
    else:
        logits = initial_logits

    # Optimizer
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(logits)

    # Temperature annealing
    temp_start, temp_end, _ = GD_PROTOCOL['arch']['temp_anneal']

    # Logging
    W_log = []
    acc_log = []

    # Generate validation data
    key, val_key = random.split(key)
    x_val = random.choice(val_key, jnp.array([-1.0, 1.0]), shape=(1000, n_vars))
    y_val = target_fn(x_val)

    for step in range(n_steps):
        # Anneal temperature
        progress = step / n_steps
        temperature = temp_start * (temp_end / temp_start) ** progress

        # Sample batch
        key, batch_key, step_key = random.split(key, 3)
        x = random.choice(batch_key, jnp.array([-1.0, 1.0]), shape=(batch_size, n_vars))
        targets = target_fn(x)

        # GD step
        logits, opt_state, loss = gd_step(
            logits, opt_state, n_vars, x, targets, temperature, step_key, optimizer
        )

        # Log every N steps
        if step % log_every == 0 or step == n_steps - 1:
            # Get soft weights for logging
            key, log_key = random.split(key)
            _, w_soft = compute_gd_loss(logits, x_val, y_val, n_vars, temperature, log_key)
            W_log.append(np.array(w_soft))

            # Compute hard accuracy
            w_hard = gumbel_softmax_hard(logits)
            acc = compute_mask_accuracy(w_hard, x_val, y_val, n_vars)
            acc_log.append(acc)

            if verbose and step % (log_every * 5) == 0:
                print(f"  Step {step}: loss={loss:.4f} acc={acc:.2%} temp={temperature:.3f}")

    return logits, W_log, acc_log


# =============================================================================
# MCMC with Step Counting
# =============================================================================

def refine_mask_mcmc_counted(
    initial_mask: jnp.ndarray,
    x: jnp.ndarray,
    targets: jnp.ndarray,
    n_vars: int,
    max_steps: int = 5000,
    target_accuracy: float = 0.9999,
    temperature_init: float = 0.1,
    rng_key: jax.Array = None
) -> Tuple[jnp.ndarray, float, int]:
    """Refine mask via MCMC, counting steps to reach target accuracy.

    Returns:
        best_mask: Final ternary mask
        best_acc: Best accuracy achieved
        steps_to_target: Number of steps to reach target (or max_steps if not reached)
    """
    if rng_key is None:
        rng_key = random.PRNGKey(0)

    from .spectral_synthesis import gibbs_step

    mask = initial_mask.copy()
    best_mask = mask
    best_acc = compute_mask_accuracy(mask, x, targets, n_vars)
    steps_to_target = max_steps  # Default if target not reached
    temperature = temperature_init

    for step in range(max_steps):
        rng_key, step_key = random.split(rng_key)
        mask = gibbs_step(step_key, mask, x, targets, n_vars, temperature)

        acc = compute_mask_accuracy(mask, x, targets, n_vars)
        if acc > best_acc:
            best_acc = acc
            best_mask = mask

        # Check if target reached
        if best_acc >= target_accuracy and steps_to_target == max_steps:
            steps_to_target = step + 1

        # Anneal temperature
        if (step + 1) % 100 == 0:
            temperature = max(temperature * 0.9, 0.01)

    return best_mask, best_acc, steps_to_target


# =============================================================================
# Warm-Start Experiment
# =============================================================================

def run_warmstart_experiment(
    op_name: str,
    op_fn: Callable,
    w_star: jnp.ndarray,
    n_seeds: int = 3,
    max_mcmc_steps: int = 2000,
    n_gd_steps: int = 2000,
    verbose: bool = True
) -> Dict:
    """Run the 4-condition warm-start experiment for one operation.

    Args:
        op_name: Name of the operation
        op_fn: Boolean function
        w_star: Ground truth ternary mask
        n_seeds: Number of seeds per condition
        max_mcmc_steps: Maximum MCMC steps
        n_gd_steps: GD warm-up steps for conditions C and D

    Returns:
        Dictionary with results for all conditions
    """
    n_vars = 4
    n_coeffs = 16

    if verbose:
        print(f"\n{'='*60}")
        print(f"Operation: {op_name}")
        print(f"{'='*60}")

    results = {
        'op_name': op_name,
        'conditions': {},
    }

    # Generate shared validation data
    key = random.PRNGKey(42)
    x_val = random.choice(key, jnp.array([-1.0, 1.0]), shape=(2000, n_vars))
    y_val = op_fn(x_val)

    # ==========================================================================
    # Condition A: Random ternary init
    # ==========================================================================
    if verbose:
        print("\n[A] Random ternary init")

    mcmc_steps_A = []
    for seed in range(n_seeds):
        key = random.PRNGKey(seed * 100)
        init_mask = random.choice(key, jnp.array([-1.0, 0.0, 1.0]), shape=(n_coeffs,))

        _, _, steps = refine_mask_mcmc_counted(
            init_mask, x_val, y_val, n_vars,
            max_steps=max_mcmc_steps,
            rng_key=random.PRNGKey(seed)
        )
        mcmc_steps_A.append(steps)
        if verbose:
            print(f"  Seed {seed}: {steps} steps")

    results['conditions']['A_random'] = {
        'mcmc_steps': mcmc_steps_A,
        'mean_steps': float(np.mean(mcmc_steps_A)),
        'std_steps': float(np.std(mcmc_steps_A)),
    }

    # ==========================================================================
    # Condition B: WHT-threshold init
    # ==========================================================================
    if verbose:
        print("\n[B] WHT-threshold init")

    mcmc_steps_B = []
    for seed in range(n_seeds):
        # Estimate WHT coefficients
        key = random.PRNGKey(seed * 200)
        coeffs = estimate_fourier_coefficients(op_fn, n_vars, n_samples=5000, rng_key=key)
        init_mask = quantize_to_ternary(coeffs, threshold=0.1)

        _, _, steps = refine_mask_mcmc_counted(
            init_mask, x_val, y_val, n_vars,
            max_steps=max_mcmc_steps,
            rng_key=random.PRNGKey(seed + 1000)
        )
        mcmc_steps_B.append(steps)
        if verbose:
            print(f"  Seed {seed}: {steps} steps")

    results['conditions']['B_wht'] = {
        'mcmc_steps': mcmc_steps_B,
        'mean_steps': float(np.mean(mcmc_steps_B)),
        'std_steps': float(np.std(mcmc_steps_B)),
    }

    # ==========================================================================
    # Condition C: GD warm-start (random init)
    # ==========================================================================
    if verbose:
        print("\n[C] GD warm-start (random init)")

    mcmc_steps_C = []
    jaccard_C = []
    eigenspectrum_C = []

    for seed in range(n_seeds):
        # Train GD
        logits, W_log, acc_log = train_gd_warmstart(
            op_fn, n_vars,
            initial_logits=None,
            n_steps=n_gd_steps,
            seed=seed,
            verbose=False
        )

        # Compute Jaccard trajectory
        W_log_np = np.array(W_log)
        jac_t, auc_jac = jaccard_trajectory(W_log_np, np.array(w_star))
        jaccard_C.append({
            'trajectory': jac_t.tolist(),
            'auc': float(auc_jac),
            'final': float(jac_t[-1]) if len(jac_t) > 0 else 0.0,
        })

        # Compute eigenspectrum
        if len(W_log_np) > 1:
            s, explained = eigenspectrum_svd(W_log_np)
            summary = spectral_compression_summary(explained)
            eigenspectrum_C.append(summary)
        else:
            eigenspectrum_C.append({})

        # Quantize and run MCMC
        init_mask = gumbel_softmax_hard(logits)
        _, _, steps = refine_mask_mcmc_counted(
            init_mask, x_val, y_val, n_vars,
            max_steps=max_mcmc_steps,
            rng_key=random.PRNGKey(seed + 2000)
        )
        mcmc_steps_C.append(steps)
        if verbose:
            print(f"  Seed {seed}: {steps} steps (AUC_Jaccard={auc_jac:.3f})")

    results['conditions']['C_gd'] = {
        'mcmc_steps': mcmc_steps_C,
        'mean_steps': float(np.mean(mcmc_steps_C)),
        'std_steps': float(np.std(mcmc_steps_C)),
        'jaccard': jaccard_C,
        'eigenspectrum': eigenspectrum_C,
    }

    # ==========================================================================
    # Condition D: WHT → GD (WHT init + GD training)
    # ==========================================================================
    if verbose:
        print("\n[D] WHT → GD (WHT init + GD)")

    mcmc_steps_D = []
    jaccard_D = []
    eigenspectrum_D = []

    for seed in range(n_seeds):
        # Get WHT initialization as logits
        key = random.PRNGKey(seed * 300)
        coeffs = estimate_fourier_coefficients(op_fn, n_vars, n_samples=5000, rng_key=key)
        wht_mask = quantize_to_ternary(coeffs, threshold=0.1)

        # Convert to logits (one-hot style)
        initial_logits = jnp.zeros((n_coeffs, 3))
        for i, v in enumerate(wht_mask):
            if v == -1:
                initial_logits = initial_logits.at[i, 0].set(2.0)
            elif v == 0:
                initial_logits = initial_logits.at[i, 1].set(2.0)
            else:
                initial_logits = initial_logits.at[i, 2].set(2.0)

        # Train GD from WHT init
        logits, W_log, acc_log = train_gd_warmstart(
            op_fn, n_vars,
            initial_logits=initial_logits,
            n_steps=n_gd_steps,
            seed=seed + 5000,
            verbose=False
        )

        # Compute Jaccard trajectory
        W_log_np = np.array(W_log)
        jac_t, auc_jac = jaccard_trajectory(W_log_np, np.array(w_star))
        jaccard_D.append({
            'trajectory': jac_t.tolist(),
            'auc': float(auc_jac),
            'final': float(jac_t[-1]) if len(jac_t) > 0 else 0.0,
        })

        # Compute eigenspectrum
        if len(W_log_np) > 1:
            s, explained = eigenspectrum_svd(W_log_np)
            summary = spectral_compression_summary(explained)
            eigenspectrum_D.append(summary)
        else:
            eigenspectrum_D.append({})

        # Quantize and run MCMC
        init_mask = gumbel_softmax_hard(logits)
        _, _, steps = refine_mask_mcmc_counted(
            init_mask, x_val, y_val, n_vars,
            max_steps=max_mcmc_steps,
            rng_key=random.PRNGKey(seed + 3000)
        )
        mcmc_steps_D.append(steps)
        if verbose:
            print(f"  Seed {seed}: {steps} steps (AUC_Jaccard={auc_jac:.3f})")

    results['conditions']['D_wht_gd'] = {
        'mcmc_steps': mcmc_steps_D,
        'mean_steps': float(np.mean(mcmc_steps_D)),
        'std_steps': float(np.std(mcmc_steps_D)),
        'jaccard': jaccard_D,
        'eigenspectrum': eigenspectrum_D,
    }

    return results


def run_full_experiment(n_seeds: int = 3, verbose: bool = True) -> Dict:
    """Run the full warm-start experiment for all Phase 4 operations."""
    print("=" * 70)
    print("PHASE 4 WARM-START EXPERIMENT")
    print("=" * 70)
    print("\nConditions:")
    print("  (A) Random ternary init → MCMC")
    print("  (B) WHT-threshold init → MCMC")
    print("  (C) GD warm-start (random init, 2000 steps) → MCMC")
    print("  (D) WHT→GD (WHT init + 2000 GD steps) → MCMC")
    print(f"\nSeeds per condition: {n_seeds}")

    # Load ground truth masks
    phase4_masks = load_phase4_masks()

    all_results = {
        'experiment': 'phase4_warmstart_v2',
        'timestamp': datetime.now().isoformat(),
        'n_seeds': n_seeds,
        'operations': {},
        'summary': {},
    }

    for op_name, op_fn in PHASE4_OPERATIONS.items():
        w_star = jnp.array(phase4_masks[op_name])

        op_results = run_warmstart_experiment(
            op_name, op_fn, w_star,
            n_seeds=n_seeds,
            verbose=verbose
        )

        all_results['operations'][op_name] = op_results

    # Compute summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    condition_names = ['A_random', 'B_wht', 'C_gd', 'D_wht_gd']
    summary = {}

    for cond in condition_names:
        all_steps = []
        for op_results in all_results['operations'].values():
            all_steps.extend(op_results['conditions'][cond]['mcmc_steps'])

        summary[cond] = {
            'mean_steps': float(np.mean(all_steps)),
            'std_steps': float(np.std(all_steps)),
        }

        # For GD conditions, also compute mean AUC(Jaccard)
        if cond in ['C_gd', 'D_wht_gd']:
            all_auc = []
            for op_results in all_results['operations'].values():
                for jac_data in op_results['conditions'][cond]['jaccard']:
                    all_auc.append(jac_data['auc'])
            summary[cond]['mean_auc_jaccard'] = float(np.mean(all_auc))

    all_results['summary'] = summary

    print(f"\nMean MCMC steps to 100% accuracy:")
    print(f"  (A) Random:  {summary['A_random']['mean_steps']:.0f} ± {summary['A_random']['std_steps']:.0f}")
    print(f"  (B) WHT:     {summary['B_wht']['mean_steps']:.0f} ± {summary['B_wht']['std_steps']:.0f}")
    print(f"  (C) GD:      {summary['C_gd']['mean_steps']:.0f} ± {summary['C_gd']['std_steps']:.0f} (AUC_Jac={summary['C_gd']['mean_auc_jaccard']:.3f})")
    print(f"  (D) WHT→GD:  {summary['D_wht_gd']['mean_steps']:.0f} ± {summary['D_wht_gd']['std_steps']:.0f} (AUC_Jac={summary['D_wht_gd']['mean_auc_jaccard']:.3f})")

    # Check acceptance criteria
    print("\n" + "-" * 70)
    print("ACCEPTANCE CRITERIA CHECK")
    print("-" * 70)

    # Criterion 1: All conditions produce valid results
    criterion1 = all(
        len(op_results['conditions'][cond]['mcmc_steps']) == n_seeds
        for op_results in all_results['operations'].values()
        for cond in condition_names
    )
    print(f"  [{'PASS' if criterion1 else 'FAIL'}] All conditions produce valid JSON")

    # Criterion 2: At least one GD condition beats random
    criterion2 = (
        summary['C_gd']['mean_steps'] < summary['A_random']['mean_steps'] or
        summary['D_wht_gd']['mean_steps'] < summary['A_random']['mean_steps']
    )
    print(f"  [{'PASS' if criterion2 else 'FAIL'}] GD condition(s) beat random")

    # Criterion 3: AUC(Jaccard) > 0.5 for GD conditions
    criterion3 = (
        summary['C_gd']['mean_auc_jaccard'] > 0.5 and
        summary['D_wht_gd']['mean_auc_jaccard'] > 0.5
    )
    print(f"  [{'PASS' if criterion3 else 'FAIL'}] AUC(Jaccard) > 0.5 for GD conditions")

    all_pass = criterion1 and criterion2 and criterion3
    all_results['acceptance'] = {
        'criterion1_valid_json': criterion1,
        'criterion2_gd_beats_random': criterion2,
        'criterion3_auc_jaccard': criterion3,
        'all_pass': all_pass,
    }

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAIL'}")

    # Save results
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / 'v2_phase4_warmstart.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return all_results


if __name__ == "__main__":
    results = run_full_experiment(n_seeds=3, verbose=True)
