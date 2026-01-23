"""
Phase 1 v2 Diagnostics (n=2)
============================

The "clean case" - shows diagnostics when GD succeeds.

Demonstrates:
1. Jaccard → 1 quickly (GD learns correct support)
2. Eigenspectrum nearly rank-1 for XOR (spectral compression)
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import optax
from typing import Dict
from datetime import datetime
import json

from .logic_dataset import generate_logic_dataset, create_train_test_split, OP_NAMES
from ..utils.diagnostics import (
    jaccard_trajectory,
    eigenspectrum_svd,
    spectral_compression_summary,
    DiagnosticsLogger,
    GD_PROTOCOL,
    load_phase1_masks,
)


def boolean_fourier_basis_2var(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Compute 2-variable Boolean Fourier basis [1, a, b, ab]."""
    ones = jnp.ones_like(a)
    ab = a * b
    return jnp.stack([ones, a, b, ab], axis=-1)  # [batch, n_bits, 4]


def gumbel_softmax_sample(logits: jnp.ndarray, temperature: float, key: jax.Array) -> jnp.ndarray:
    """Sample from Gumbel-Softmax distribution for ternary values."""
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
    targets: jnp.ndarray,
    temperature: float,
    key: jax.Array
):
    """Compute loss and return soft weights."""
    w_soft = gumbel_softmax_sample(logits, temperature, key)
    features = boolean_fourier_basis_2var(a, b)
    output = jnp.sum(features * w_soft, axis=-1)

    # Hamming loss (more stable than BCE for Boolean functions)
    # Normalize output with tanh for soft approximation
    output = jnp.tanh(output)
    loss = jnp.mean((1 - output * targets) / 2)

    return loss, w_soft


def compute_accuracy(logits: jnp.ndarray, a: jnp.ndarray, b: jnp.ndarray, targets: jnp.ndarray) -> float:
    """Compute accuracy with hard ternary mask."""
    w_hard = gumbel_softmax_hard(logits)
    features = boolean_fourier_basis_2var(a, b)
    output = jnp.sum(features * w_hard, axis=-1)
    output = jnp.sign(output)
    output = jnp.where(output == 0, 1.0, output)
    return float(jnp.mean(output == targets))


@jax.jit
def compute_grads(logits, a, b, targets, temperature, key):
    """Compute gradients (jitted)."""
    def loss_fn(logits):
        loss, _ = compute_loss_and_soft_weights(logits, a, b, targets, temperature, key)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(logits)
    return loss, grads


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
    """Train one operation with full diagnostic logging."""
    n_coeffs = 4
    key = random.PRNGKey(seed)

    # Get data
    a_train, b_train, y_train, _ = train_data[op_name]
    a_test, b_test, y_test, _ = test_data[op_name]

    # Initialize logits
    key, init_key = random.split(key)
    logits = random.normal(init_key, (n_coeffs, 3)) * 0.1

    # GD Protocol v2
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
        y_batch = y_train[batch_idx]

        # Training step
        loss, grads = compute_grads(logits, a_batch, b_batch, y_batch, temperature, step_key)
        updates, opt_state = optimizer.update(grads, opt_state, logits)
        logits = optax.apply_updates(logits, updates)

        # Log every N steps
        if step % log_every == 0 or step == n_steps - 1:
            key, log_key = random.split(key)
            _, w_soft = compute_loss_and_soft_weights(
                logits, a_test, b_test, y_test, temperature, log_key
            )
            W_log.append(np.array(w_soft))

            acc = compute_accuracy(logits, a_test, b_test, y_test)
            acc_log.append(acc)
            loss_log.append(float(loss))

            if verbose and step % (log_every * 5) == 0:
                print(f"    Step {step}: loss={loss:.4f} acc={acc:.2%} temp={temperature:.3f}")

    # Convert to arrays
    W_log = np.array(W_log)
    acc_log = np.array(acc_log)

    # Compute Jaccard trajectory
    jac_t, auc_jac = jaccard_trajectory(W_log, w_star)

    # Compute eigenspectrum
    if len(W_log) > 1:
        s, explained = eigenspectrum_svd(W_log)
        eigen_summary = spectral_compression_summary(explained)
    else:
        s = np.array([1.0])
        explained = np.array([1.0])
        eigen_summary = {}

    final_mask = np.array(gumbel_softmax_hard(logits))
    final_acc = acc_log[-1] if len(acc_log) > 0 else 0.0

    return {
        'op_name': op_name,
        'final_accuracy': float(final_acc),
        'final_mask': final_mask.tolist(),
        'jaccard': {
            'trajectory': jac_t.tolist(),
            'auc': float(auc_jac),
            'final': float(jac_t[-1]) if len(jac_t) > 0 else 0.0,
        },
        'eigenspectrum': {
            'singular_values': s.tolist()[:4],
            'explained_variance': explained.tolist(),
            'summary': eigen_summary,
        },
        'training': {
            'accuracy_log': acc_log.tolist(),
            'loss_log': loss_log,
        },
    }


def run_phase1_diagnostics(n_seeds: int = 3, verbose: bool = True) -> Dict:
    """Run full Phase 1 diagnostics."""
    print("=" * 70)
    print("PHASE 1 JACCARD + EIGENSPECTRUM DIAGNOSTICS (n=2)")
    print("=" * 70)
    print("\nThe 'clean case' - diagnostics when GD succeeds")

    # Load ground truth masks
    phase1_masks = load_phase1_masks()

    # Generate dataset
    print("\nGenerating dataset...")
    n_train, n_test, n_bits = 2000, 500, 64
    train_data, test_data = create_train_test_split(n_train, n_test, n_bits)
    print(f"  Train: {n_train}, Test: {n_test}, Bits: {n_bits}")

    all_results = {
        'experiment': 'phase1_jaccard_eigenspace_v2',
        'timestamp': datetime.now().isoformat(),
        'n_seeds': n_seeds,
        'gd_protocol': GD_PROTOCOL,
        'operations': {},
        'summary': {},
    }

    # Run for each operation and seed
    for op_name in ['xor', 'and', 'or', 'implies']:
        w_star = np.array(phase1_masks[op_name])

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
                      f"AUC_Jac={seed_results['jaccard']['auc']:.3f}")

        # Aggregate
        accs = [s['final_accuracy'] for s in op_results['seeds']]
        aucs = [s['jaccard']['auc'] for s in op_results['seeds']]

        op_results['aggregate'] = {
            'mean_accuracy': float(np.mean(accs)),
            'std_accuracy': float(np.std(accs)),
            'mean_auc_jaccard': float(np.mean(aucs)),
            'std_auc_jaccard': float(np.std(aucs)),
        }

        all_results['operations'][op_name] = op_results

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_accs = []
    all_aucs = []
    for op_results in all_results['operations'].values():
        for seed_results in op_results['seeds']:
            all_accs.append(seed_results['final_accuracy'])
            all_aucs.append(seed_results['jaccard']['auc'])

    summary = {
        'mean_accuracy': float(np.mean(all_accs)),
        'std_accuracy': float(np.std(all_accs)),
        'mean_auc_jaccard': float(np.mean(all_aucs)),
        'std_auc_jaccard': float(np.std(all_aucs)),
    }
    all_results['summary'] = summary

    print(f"\nOverall statistics across 4 operations, {n_seeds} seeds:")
    print(f"  Final accuracy: {summary['mean_accuracy']:.2%} ± {summary['std_accuracy']:.2%}")
    print(f"  AUC(Jaccard):   {summary['mean_auc_jaccard']:.3f} ± {summary['std_auc_jaccard']:.3f}")

    # Save results
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / 'v2_phase1_jaccard_eigenspace.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return all_results


if __name__ == "__main__":
    results = run_phase1_diagnostics(n_seeds=3, verbose=True)
