"""
Phase 3 Jaccard + Eigenspectrum Diagnostics (v2)
=================================================

Key tension phase: GD learns topology even when accuracy plateaus at ~76%.

Shows:
1. Jaccard(t) increases even while accuracy plateaus
2. Eigenspectrum collapses to few modes (spectral compression)

Key finding: GD learns *which* coefficients matter, just not exact values.
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

from .logic3_dataset import create_phase3_train_test_split
from .boolean_fourier_3var import boolean_fourier_3var, PHASE3_OPERATIONS, CHAR_NAMES_3VAR
from ..utils.diagnostics import (
    jaccard_trajectory,
    jaccard_final,
    eigenspectrum_svd,
    spectral_compression_summary,
    DiagnosticsLogger,
    GD_PROTOCOL,
    load_phase3_masks,
)


# =============================================================================
# GD Training with Gumbel-Softmax (v2 Protocol)
# =============================================================================

def gumbel_softmax_sample(logits: jnp.ndarray, temperature: float, key: jax.Array) -> jnp.ndarray:
    """Sample from Gumbel-Softmax distribution."""
    gumbel_noise = -jnp.log(-jnp.log(random.uniform(key, logits.shape) + 1e-10) + 1e-10)
    soft_samples = jax.nn.softmax((logits + gumbel_noise) / temperature, axis=-1)
    values = jnp.array([-1.0, 0.0, 1.0])
    return jnp.sum(soft_samples * values, axis=-1)


def gumbel_softmax_hard(logits: jnp.ndarray) -> jnp.ndarray:
    """Hard quantization of Gumbel-Softmax logits."""
    values = jnp.array([-1.0, 0.0, 1.0])
    idx = jnp.argmax(logits, axis=-1)
    return values[idx]


def compute_loss_and_soft_weights(
    logits: jnp.ndarray,
    a: jnp.ndarray,
    b: jnp.ndarray,
    c: jnp.ndarray,
    targets: jnp.ndarray,
    temperature: float,
    key: jax.Array
) -> Tuple[float, jnp.ndarray]:
    """Compute loss and return soft weights for Jaccard tracking.

    Args:
        logits: (8, 3) trainable parameters for 3-var basis
        a, b, c: (batch, n_bits) inputs in {-1, +1}
        targets: (batch, n_bits) target outputs
        temperature: Gumbel-Softmax temperature
        key: random key

    Returns:
        loss: scalar loss
        w_soft: (8,) soft weights
    """
    w_soft = gumbel_softmax_sample(logits, temperature, key)

    # Compute Boolean Fourier features
    features = boolean_fourier_3var(a, b, c)  # [batch, n_bits, 8]

    # Apply mask
    output = jnp.sum(features * w_soft, axis=-1)  # [batch, n_bits]

    # Binary cross-entropy style loss
    y = (targets + 1) / 2  # {0, 1}
    p = jax.nn.sigmoid(output * 5.0)
    loss = -jnp.mean(y * jnp.log(p + 1e-10) + (1 - y) * jnp.log(1 - p + 1e-10))

    return loss, w_soft


def compute_accuracy(logits: jnp.ndarray, a: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray, targets: jnp.ndarray) -> float:
    """Compute accuracy with hard ternary mask."""
    w_hard = gumbel_softmax_hard(logits)
    features = boolean_fourier_3var(a, b, c)
    output = jnp.sum(features * w_hard, axis=-1)
    output = jnp.sign(output)
    output = jnp.where(output == 0, 1.0, output)
    return float(jnp.mean(output == targets))


@jax.jit
def compute_grads(logits, a, b, c, targets, temperature, key):
    """Compute gradients (jitted)."""
    def loss_fn(logits):
        loss, _ = compute_loss_and_soft_weights(logits, a, b, c, targets, temperature, key)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(logits)
    return loss, grads


def train_step(logits, opt_state, a, b, c, targets, temperature, key, optimizer):
    """One training step (non-jitted wrapper)."""
    loss, grads = compute_grads(logits, a, b, c, targets, temperature, key)
    updates, opt_state = optimizer.update(grads, opt_state, logits)
    logits = optax.apply_updates(logits, updates)
    return logits, opt_state, loss


def train_operation_with_diagnostics(
    op_name: str,
    train_data: Dict,
    test_data: Dict,
    w_star: np.ndarray,
    seed: int = 0,
    n_steps: int = 2000,
    log_every: int = 100,
    verbose: bool = True
) -> Dict:
    """Train one operation with full diagnostic logging.

    Args:
        op_name: Operation name
        train_data: Training data dict
        test_data: Test data dict
        w_star: Ground truth ternary mask
        seed: Random seed
        n_steps: Training steps (GD Protocol v2)
        log_every: Logging frequency
        verbose: Print progress

    Returns:
        Dictionary with diagnostics
    """
    n_coeffs = 8
    key = random.PRNGKey(seed)

    # Get data for this operation
    a_train, b_train, c_train, y_train, _ = train_data[op_name]
    a_test, b_test, c_test, y_test, _ = test_data[op_name]

    # Initialize logits
    key, init_key = random.split(key)
    logits = random.normal(init_key, (n_coeffs, 3)) * 0.1

    # GD Protocol v2 hyperparams
    lr = GD_PROTOCOL['hyperparams']['lr']
    batch_size = GD_PROTOCOL['hyperparams']['batch_size']
    temp_start, temp_end, _ = GD_PROTOCOL['arch']['temp_anneal']

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(logits)

    # Logging
    W_log = []
    acc_log = []
    loss_log = []

    n_train = len(a_train)

    for step in range(n_steps):
        # Temperature annealing
        progress = step / n_steps
        temperature = temp_start * (temp_end / temp_start) ** progress

        # Sample batch
        key, batch_key, step_key = random.split(key, 3)
        batch_idx = random.randint(batch_key, (batch_size,), 0, n_train)
        a_batch = a_train[batch_idx]
        b_batch = b_train[batch_idx]
        c_batch = c_train[batch_idx]
        y_batch = y_train[batch_idx]

        # Training step
        logits, opt_state, loss = train_step(
            logits, opt_state, a_batch, b_batch, c_batch, y_batch, temperature, step_key, optimizer
        )

        # Log every N steps
        if step % log_every == 0 or step == n_steps - 1:
            # Get soft weights
            key, log_key = random.split(key)
            _, w_soft = compute_loss_and_soft_weights(
                logits, a_test, b_test, c_test, y_test, temperature, log_key
            )
            W_log.append(np.array(w_soft))

            # Compute accuracy
            acc = compute_accuracy(logits, a_test, b_test, c_test, y_test)
            acc_log.append(acc)
            loss_log.append(float(loss))

            if verbose and step % (log_every * 5) == 0:
                print(f"    Step {step}: loss={loss:.4f} acc={acc:.2%} temp={temperature:.3f}")

    # Convert to arrays
    W_log = np.array(W_log)
    acc_log = np.array(acc_log)

    # Compute Jaccard trajectory
    jac_t, auc_jac = jaccard_trajectory(W_log, w_star)
    final_jac = jaccard_final(W_log[-1], w_star) if len(W_log) > 0 else 0.0

    # Compute eigenspectrum
    if len(W_log) > 1:
        s, explained = eigenspectrum_svd(W_log)
        eigen_summary = spectral_compression_summary(explained)
    else:
        s = np.array([1.0])
        explained = np.array([1.0])
        eigen_summary = {'var_top1': 1.0, 'var_top3': 1.0, 'modes_90': 1, 'modes_95': 1}

    # Final mask
    final_mask = np.array(gumbel_softmax_hard(logits))
    final_acc = acc_log[-1] if len(acc_log) > 0 else 0.0

    return {
        'op_name': op_name,
        'final_accuracy': float(final_acc),
        'final_mask': final_mask.tolist(),
        'jaccard': {
            'trajectory': jac_t.tolist(),
            'auc': float(auc_jac),
            'final': float(final_jac),
        },
        'eigenspectrum': {
            'singular_values': s.tolist()[:8],  # Keep top 8
            'explained_variance': explained.tolist(),
            'summary': eigen_summary,
        },
        'training': {
            'accuracy_log': acc_log.tolist(),
            'loss_log': loss_log,
        },
    }


def run_phase3_diagnostics(n_seeds: int = 3, verbose: bool = True) -> Dict:
    """Run full Phase 3 diagnostics with Jaccard + eigenspectrum."""
    print("=" * 70)
    print("PHASE 3 JACCARD + EIGENSPECTRUM DIAGNOSTICS")
    print("=" * 70)
    print("\nShowing: GD learns topology even when accuracy plateaus at ~76%")
    print("Key finding: Jaccard increases → GD learns *which* coefficients matter")

    # Load ground truth masks
    phase3_masks = load_phase3_masks()

    # Generate dataset
    print("\nGenerating dataset...")
    n_train, n_test, n_bits = 2000, 500, 64
    train_data, test_data = create_phase3_train_test_split(n_train, n_test, n_bits)
    print(f"  Train: {n_train}, Test: {n_test}, Bits: {n_bits}")

    all_results = {
        'experiment': 'phase3_jaccard_eigenspace_v2',
        'timestamp': datetime.now().isoformat(),
        'n_seeds': n_seeds,
        'gd_protocol': GD_PROTOCOL,
        'operations': {},
        'summary': {},
    }

    # Run for each operation and seed
    for op_name in phase3_masks.keys():
        w_star = np.array(phase3_masks[op_name])

        if verbose:
            print(f"\n{'─'*60}")
            print(f"Operation: {op_name}")
            print(f"  Ground truth mask: {w_star.tolist()}")
            print(f"{'─'*60}")

        op_results = {
            'ground_truth_mask': w_star.tolist(),
            'seeds': [],
        }

        for seed in range(n_seeds):
            if verbose:
                print(f"\n  Seed {seed}:")

            seed_results = train_operation_with_diagnostics(
                op_name, train_data, test_data, w_star,
                seed=seed,
                n_steps=GD_PROTOCOL['hyperparams']['steps'],
                log_every=GD_PROTOCOL['hyperparams']['log_every'],
                verbose=verbose
            )
            op_results['seeds'].append(seed_results)

            if verbose:
                print(f"    Final: acc={seed_results['final_accuracy']:.2%} "
                      f"AUC_Jac={seed_results['jaccard']['auc']:.3f} "
                      f"final_Jac={seed_results['jaccard']['final']:.3f}")

        # Aggregate across seeds
        accs = [s['final_accuracy'] for s in op_results['seeds']]
        aucs = [s['jaccard']['auc'] for s in op_results['seeds']]
        final_jacs = [s['jaccard']['final'] for s in op_results['seeds']]

        op_results['aggregate'] = {
            'mean_accuracy': float(np.mean(accs)),
            'std_accuracy': float(np.std(accs)),
            'mean_auc_jaccard': float(np.mean(aucs)),
            'std_auc_jaccard': float(np.std(aucs)),
            'mean_final_jaccard': float(np.mean(final_jacs)),
            'std_final_jaccard': float(np.std(final_jacs)),
        }

        all_results['operations'][op_name] = op_results

    # Compute summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_accs = []
    all_aucs = []
    all_final_jacs = []

    for op_results in all_results['operations'].values():
        for seed_results in op_results['seeds']:
            all_accs.append(seed_results['final_accuracy'])
            all_aucs.append(seed_results['jaccard']['auc'])
            all_final_jacs.append(seed_results['jaccard']['final'])

    summary = {
        'mean_accuracy': float(np.mean(all_accs)),
        'std_accuracy': float(np.std(all_accs)),
        'mean_auc_jaccard': float(np.mean(all_aucs)),
        'std_auc_jaccard': float(np.std(all_aucs)),
        'mean_final_jaccard': float(np.mean(all_final_jacs)),
        'std_final_jaccard': float(np.std(all_final_jacs)),
    }
    all_results['summary'] = summary

    print(f"\nOverall statistics across {len(all_results['operations'])} operations, {n_seeds} seeds:")
    print(f"  Final accuracy: {summary['mean_accuracy']:.2%} ± {summary['std_accuracy']:.2%}")
    print(f"  AUC(Jaccard):   {summary['mean_auc_jaccard']:.3f} ± {summary['std_auc_jaccard']:.3f}")
    print(f"  Final Jaccard:  {summary['mean_final_jaccard']:.3f} ± {summary['std_final_jaccard']:.3f}")

    # Check acceptance criteria
    print("\n" + "-" * 70)
    print("ACCEPTANCE CRITERIA CHECK")
    print("-" * 70)

    # Criterion 1: Jaccard trajectories logged for all ops
    criterion1 = all(
        len(seed_results['jaccard']['trajectory']) > 0
        for op_results in all_results['operations'].values()
        for seed_results in op_results['seeds']
    )
    print(f"  [{'PASS' if criterion1 else 'FAIL'}] Jaccard trajectories logged for all ops")

    # Criterion 2: Mean final Jaccard > 0.6
    criterion2 = summary['mean_final_jaccard'] > 0.6
    print(f"  [{'PASS' if criterion2 else 'FAIL'}] Mean final Jaccard > 0.6 (got {summary['mean_final_jaccard']:.3f})")

    # Criterion 3: AUC(Jaccard) computed and saved
    criterion3 = summary['mean_auc_jaccard'] > 0
    print(f"  [{'PASS' if criterion3 else 'FAIL'}] AUC(Jaccard) computed (got {summary['mean_auc_jaccard']:.3f})")

    all_pass = criterion1 and criterion2 and criterion3
    all_results['acceptance'] = {
        'criterion1_jaccard_logged': criterion1,
        'criterion2_final_jaccard_gt_0.6': criterion2,
        'criterion3_auc_computed': criterion3,
        'all_pass': all_pass,
    }

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAIL'}")

    # Key insight
    print("\n" + "-" * 70)
    print("KEY INSIGHT")
    print("-" * 70)
    print(f"  Accuracy plateaus at ~{summary['mean_accuracy']:.0%} (below 100%)")
    print(f"  But Jaccard reaches {summary['mean_final_jaccard']:.0%} → GD learns the support topology!")
    print("  Discrete refinement can then quantize within this learned subspace.")

    # Save results
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / 'v2_phase3_jaccard_eigenspace.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return all_results


if __name__ == "__main__":
    results = run_phase3_diagnostics(n_seeds=3, verbose=True)
