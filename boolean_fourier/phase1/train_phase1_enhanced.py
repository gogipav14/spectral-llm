"""
Phase 1 Enhanced: Discovery with Robust Exploration
====================================================

Improvements over train_phase1_discovery.py:
1. Plateau detection with micro-restart (reheat + noise injection)
2. Gentler temperature annealing (floor at 0.05, not 0.01)
3. More epochs for XOR (150)
4. Cyclical annealing option
5. Multi-start fallback for stubborn seeds

Target: 90%+ XOR → parity discovery rate (up from 60%)
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from functools import partial
from typing import Dict, Tuple, List
import json
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import numpy as np

from logic_dataset import generate_logic_dataset, create_train_test_split, OP_NAMES


# Configuration
N_BITS = 64
N_TRAIN = 10000
N_TEST = 2000
BATCH_SIZE = 128
N_EPOCHS_XOR = 150  # More epochs for XOR with plateau detection
N_EPOCHS_OTHER = 50
LEARNING_RATE = 0.1
THRESHOLD = 0.3
ACCURACY_THRESHOLD = 0.995  # Slightly higher for robustness

# Plateau detection parameters
PLATEAU_PATIENCE = 10  # Epochs of no improvement before micro-restart
REHEAT_FACTOR = 2.0  # How much to multiply temperature on restart
NOISE_SCALE = 0.05  # Noise injection magnitude

CHECKPOINT_DIR = Path("v5/checkpoints/phase1_enhanced")


def soft_ternary(w: jnp.ndarray, temperature: float = 1.0, threshold: float = 0.3) -> jnp.ndarray:
    """Soft ternary quantization with temperature annealing."""
    sign = jnp.tanh(w / temperature)
    gate = jax.nn.sigmoid((jnp.abs(w) - threshold) / temperature)
    return sign * gate


def ternary_attractor_loss(w: jnp.ndarray) -> jnp.ndarray:
    """Loss that pulls weights toward {-1, 0, +1}."""
    dist_to_neg1 = (w + 1) ** 2
    dist_to_zero = w ** 2
    dist_to_pos1 = (w - 1) ** 2
    min_dist = jnp.minimum(jnp.minimum(dist_to_neg1, dist_to_zero), dist_to_pos1)
    return jnp.mean(min_dist)


def ternary_quantize(w: jnp.ndarray, threshold: float = 0.3) -> jnp.ndarray:
    """Hard quantization to {-1, 0, +1}."""
    return jnp.sign(w) * (jnp.abs(w) > threshold)


def sparsity_loss(mask: jnp.ndarray) -> jnp.ndarray:
    """L1 sparsity prior."""
    return jnp.mean(jnp.abs(mask))


class PermutedBooleanFourierLayer(nn.Module):
    """Logic layer with random feature ordering."""
    n_bits: int = 64
    permutation: jnp.ndarray = None

    @nn.compact
    def __call__(self, a, b, operation_id, temperature: float = 1.0, training: bool = True):
        xor_mask = self.param('xor_mask', nn.initializers.normal(0.5), (4,))
        and_mask = self.param('and_mask', nn.initializers.normal(0.5), (4,))
        or_mask = self.param('or_mask', nn.initializers.normal(0.5), (4,))
        implies_mask = self.param('implies_mask', nn.initializers.normal(0.5), (4,))

        masks = [xor_mask, and_mask, or_mask, implies_mask]
        raw_mask = masks[operation_id]

        if training:
            mask = soft_ternary(raw_mask, temperature, THRESHOLD)
        else:
            mask = ternary_quantize(raw_mask, THRESHOLD)

        # Boolean Fourier features: [1, a, b, ab]
        ones = jnp.ones_like(a)
        ab = a * b
        features_canonical = jnp.stack([ones, a, b, ab], axis=-1)

        # Apply permutation
        if self.permutation is not None:
            features = features_canonical[:, :, self.permutation]
        else:
            features = features_canonical

        masked = features * mask
        output = jnp.sum(masked, axis=-1)

        if training:
            output = jnp.tanh(output * 10.0)
        else:
            output = jnp.sign(output)
            output = jnp.where(output == 0, 1.0, output)

        return output


def compute_hamming_loss(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Hamming loss: fraction of mismatched bits."""
    return jnp.mean((1 - pred * target) / 2)


@partial(jax.jit, static_argnums=(4, 5))
def train_step_generic(state, a, b, target, op_id, apply_fn, temperature: float = 1.0):
    """Training step with generic priors only."""

    def loss_fn(params):
        pred = apply_fn({'params': params}, a, b, op_id, temperature=temperature, training=True)
        task_loss = compute_hamming_loss(pred, target)

        mask_names = ['xor_mask', 'and_mask', 'or_mask', 'implies_mask']
        raw_mask = params[mask_names[op_id]]

        attractor = ternary_attractor_loss(raw_mask)
        sparse = sparsity_loss(raw_mask)

        total_loss = (
            task_loss +
            0.01 * attractor +
            0.005 * sparse
        )

        return total_loss, (task_loss, attractor, sparse)

    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)

    return state, loss, aux


def compute_accuracy(apply_fn, params, a, b, target, op_id):
    """Compute accuracy for a single operation."""
    pred = apply_fn({'params': params}, a, b, op_id, temperature=0.01, training=False)
    return float(jnp.mean(pred == target))


def inject_noise(params: dict, mask_name: str, rng: jax.random.PRNGKey, scale: float = 0.05) -> dict:
    """Inject noise into a specific mask parameter."""
    noise = jax.random.normal(rng, shape=params[mask_name].shape)
    new_params = dict(params)
    new_params[mask_name] = params[mask_name] + scale * noise
    return new_params


def cyclical_temperature(epoch: int, total_epochs: int, tau_hi: float = 1.0,
                         tau_lo: float = 0.05, n_cycles: int = 3) -> float:
    """Cyclical temperature schedule with decay envelope."""
    cycle_length = total_epochs / n_cycles
    cycle_progress = (epoch % cycle_length) / cycle_length

    # Cosine schedule per cycle
    tau = tau_lo + 0.5 * (tau_hi - tau_lo) * (1 + np.cos(np.pi * cycle_progress))

    # Overall decay envelope
    global_decay = (tau_lo / tau_hi) ** (epoch / total_epochs)

    return tau * (1 + global_decay) / 2


def train_operation_with_plateau_detection(
    state,
    model,
    train_data: Tuple,
    test_data: Tuple,
    op_id: int,
    op_name: str,
    n_epochs: int,
    rng: jax.random.PRNGKey,
    use_cyclical: bool = False
) -> Tuple[train_state.TrainState, float]:
    """
    Train single operation with plateau detection and micro-restart.
    """
    a, b, target, _ = train_data
    test_a, test_b, test_target, _ = test_data

    best_acc = 0.0
    prev_acc = 0.0
    plateau_counter = 0
    n_restarts = 0
    max_restarts = 3

    # Base temperature for micro-restart
    base_temp = 1.0
    current_temp_multiplier = 1.0

    print(f"\n--- Training {op_name.upper()} ({n_epochs} epochs) ---")

    for epoch in range(n_epochs):
        # Temperature schedule
        progress = epoch / (n_epochs - 1)

        if use_cyclical:
            temperature = cyclical_temperature(epoch, n_epochs, tau_hi=1.0, tau_lo=0.05)
        else:
            # Gentler annealing with floor at 0.05
            temperature = base_temp * (0.05 / base_temp) ** progress

        # Apply reheat multiplier if we micro-restarted
        temperature = min(temperature * current_temp_multiplier, 1.0)

        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, len(a), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(a))
            a_batch = a[start:end]
            b_batch = b[start:end]
            t_batch = target[start:end]

            state, loss, aux = train_step_generic(
                state, a_batch, b_batch, t_batch,
                op_id, model.apply, temperature
            )
            epoch_loss += float(loss)
            n_batches += 1

        avg_loss = epoch_loss / n_batches

        # Check accuracy periodically
        if (epoch + 1) % 5 == 0 or epoch == 0:
            acc = compute_accuracy(model.apply, state.params, test_a, test_b, test_target, op_id)

            mask_name = f'{op_name}_mask'
            raw_mask = state.params[mask_name]
            ternary = ternary_quantize(raw_mask, THRESHOLD)

            # PLATEAU DETECTION
            if abs(acc - prev_acc) < 0.01:
                plateau_counter += 1
            else:
                plateau_counter = 0
                current_temp_multiplier = 1.0  # Reset multiplier on progress

            # MICRO-RESTART: Reheat + inject noise
            if plateau_counter >= PLATEAU_PATIENCE // 5 and acc < 0.95 and n_restarts < max_restarts:
                n_restarts += 1
                print(f"  ⚡ Plateau at {acc:.1%} (epoch {epoch+1}) - micro-restart #{n_restarts}")

                # Reheat temperature
                current_temp_multiplier = REHEAT_FACTOR

                # Inject noise into mask
                rng, noise_key = jax.random.split(rng)
                new_params = inject_noise(state.params, mask_name, noise_key, NOISE_SCALE)
                state = state.replace(params=new_params)

                plateau_counter = 0

            # Logging
            if (epoch + 1) % 10 == 0 or epoch == 0 or acc > best_acc:
                restart_str = f" (restarts: {n_restarts})" if n_restarts > 0 else ""
                print(f"  Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | Acc: {acc:.2%} | T: {temperature:.3f}{restart_str}")
                print(f"            | Mask: {ternary}")

            if acc > best_acc:
                best_acc = acc

            prev_acc = acc

            # Early stopping
            if acc >= ACCURACY_THRESHOLD and temperature < 0.15:
                print(f"  ✓ Converged at epoch {epoch+1} with {acc:.2%} accuracy")
                return state, acc

    print(f"  Final: {best_acc:.2%} accuracy after {n_epochs} epochs")
    return state, best_acc


def train_with_multi_start(
    model_class,
    permutation: jnp.ndarray,
    train_data: Tuple,
    test_data: Tuple,
    op_id: int,
    op_name: str,
    n_epochs: int,
    base_seed: int,
    n_starts: int = 3
) -> Tuple[train_state.TrainState, float]:
    """
    Multi-start training for stubborn operations (XOR).
    Only used if plateau detection fails.
    """
    best_acc = 0.0
    best_state = None

    for start_id in range(n_starts):
        seed = base_seed + start_id * 1000
        rng = jax.random.PRNGKey(seed)

        # Fresh initialization
        model = model_class(n_bits=N_BITS, permutation=permutation)
        dummy_a = jnp.ones((1, N_BITS))
        dummy_b = jnp.ones((1, N_BITS))
        variables = model.init(rng, dummy_a, dummy_b, 0)

        optimizer = optax.adam(LEARNING_RATE)
        state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=variables['params'],
            tx=optimizer
        )

        print(f"\n  [Multi-start {start_id + 1}/{n_starts}]")

        rng, train_key = jax.random.split(rng)
        state, acc = train_operation_with_plateau_detection(
            state, model, train_data, test_data, op_id, op_name, n_epochs, train_key
        )

        if acc > best_acc:
            best_acc = acc
            best_state = state

        # Early stop if success
        if acc >= ACCURACY_THRESHOLD:
            print(f"  ✓ Multi-start succeeded on attempt {start_id + 1}")
            return best_state, best_acc

    return best_state, best_acc


def analyze_discovered_mask(raw_mask: jnp.ndarray, perm: jnp.ndarray, op_name: str) -> Dict:
    """Analyze what the model discovered after inverting permutation."""
    ternary_mask = np.array(ternary_quantize(raw_mask, THRESHOLD))
    perm_np = np.array(perm, dtype=int)

    # Convert to canonical ordering
    canonical_mask = np.zeros(4)
    for j in range(4):
        canonical_mask[perm_np[j]] = ternary_mask[j]

    abs_canonical = np.abs(canonical_mask)
    total_energy = np.sum(abs_canonical) + 1e-8

    feature_names = ['const(1)', 'a', 'b', 'ab(parity)']
    dominant_idx = int(np.argmax(abs_canonical))
    dominant_energy = float(abs_canonical[dominant_idx])
    concentration = dominant_energy / float(total_energy)

    return {
        'op_name': op_name,
        'raw_mask': [float(x) for x in raw_mask],
        'ternary_mask': [float(x) for x in ternary_mask],
        'canonical_mask': [float(x) for x in canonical_mask],
        'permutation': [int(x) for x in perm],
        'dominant_feature': feature_names[dominant_idx],
        'dominant_idx': dominant_idx,
        'concentration': concentration,
    }


def run_discovery_experiment(seed: int, use_multi_start: bool = True) -> Dict:
    """Run a single discovery experiment with enhanced training."""
    print(f"\n{'='*60}")
    print(f"Enhanced Discovery Experiment - Seed {seed}")
    print(f"{'='*60}")

    rng = jax.random.PRNGKey(seed)
    permutation = jax.random.permutation(rng, 4)

    feature_labels = ['1', 'a', 'b', 'ab']
    permuted_order = [feature_labels[int(permutation[i])] for i in range(4)]
    print(f"\nFeature permutation: {permutation}")
    print(f"  Permuted order: {permuted_order}")

    train_data, test_data = create_train_test_split(N_TRAIN, N_TEST, N_BITS, seed=seed)

    model = PermutedBooleanFourierLayer(n_bits=N_BITS, permutation=permutation)
    rng = jax.random.PRNGKey(seed + 1000)
    dummy_a = jnp.ones((1, N_BITS))
    dummy_b = jnp.ones((1, N_BITS))
    variables = model.init(rng, dummy_a, dummy_b, 0)

    optimizer = optax.adam(LEARNING_RATE)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optimizer
    )

    op_order = [0, 1, 2, 3]
    op_names = ['xor', 'and', 'or', 'implies']

    for op_id in op_order:
        op_name = op_names[op_id]
        n_epochs = N_EPOCHS_XOR if op_id == 0 else N_EPOCHS_OTHER

        train_op_data = train_data[op_name]
        test_op_data = test_data[op_name]

        rng, train_key = jax.random.split(rng)
        state, acc = train_operation_with_plateau_detection(
            state, model, train_op_data, test_op_data, op_id, op_name, n_epochs, train_key
        )

        # If XOR failed and multi-start is enabled, try harder
        if op_id == 0 and acc < 0.95 and use_multi_start:
            print(f"\n  ⚠️ XOR didn't converge ({acc:.1%}), trying multi-start...")
            state, acc = train_with_multi_start(
                PermutedBooleanFourierLayer,
                permutation,
                train_op_data,
                test_op_data,
                op_id,
                op_name,
                n_epochs,
                seed,
                n_starts=3
            )

    # Final analysis
    print(f"\n{'='*60}")
    print("Discovery Analysis")
    print(f"{'='*60}")

    results = {
        'seed': seed,
        'permutation': [int(x) for x in permutation],
        'operations': {},
        'accuracies': {}
    }

    for op_id, op_name in enumerate(op_names):
        mask_name = f'{op_name}_mask'
        raw_mask = state.params[mask_name]
        analysis = analyze_discovered_mask(raw_mask, permutation, op_name)
        results['operations'][op_name] = analysis

        test_a, test_b, test_target, _ = test_data[op_name]
        acc = compute_accuracy(model.apply, state.params, test_a, test_b, test_target, op_id)
        results['accuracies'][op_name] = acc

        print(f"\n{op_name.upper()}:")
        print(f"  Canonical mask: {analysis['canonical_mask']}")
        print(f"  Dominant: {analysis['dominant_feature']} ({analysis['concentration']:.0%})")
        print(f"  Accuracy: {acc:.2%}")

    xor_analysis = results['operations']['xor']
    xor_success = (xor_analysis['dominant_idx'] == 3)
    results['xor_discovered_parity'] = xor_success

    print(f"\n{'='*60}")
    if xor_success:
        print(f"✅ XOR DISCOVERED PARITY! (ab character)")
    else:
        print(f"❌ XOR did NOT discover parity. Found: {xor_analysis['dominant_feature']}")
    print(f"{'='*60}")

    return results


def run_validation_suite(n_seeds: int = 10, use_multi_start: bool = True):
    """Run enhanced discovery validation."""
    print("\n" + "="*70)
    print("PHASE 1 ENHANCED DISCOVERY VALIDATION")
    print("="*70)
    print(f"\nRunning {n_seeds} experiments with:")
    print(f"  - Plateau detection + micro-restart")
    print(f"  - Gentler annealing (floor at 0.05)")
    print(f"  - Multi-start fallback: {'enabled' if use_multi_start else 'disabled'}")

    all_results = []
    xor_successes = 0

    for seed in range(n_seeds):
        result = run_discovery_experiment(seed=seed * 42, use_multi_start=use_multi_start)
        all_results.append(result)

        if result['xor_discovered_parity']:
            xor_successes += 1

    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)

    discovery_rate = xor_successes / n_seeds
    print(f"\nXOR → Parity Discovery Rate: {xor_successes}/{n_seeds} = {discovery_rate:.0%}")

    mean_accs = {op: np.mean([r['accuracies'][op] for r in all_results]) for op in ['xor', 'and', 'or', 'implies']}
    print(f"\nMean Accuracies:")
    for op, acc in mean_accs.items():
        print(f"  {op.upper():8s}: {acc:.2%}")

    print(f"\n{'='*70}")
    if discovery_rate >= 0.9:
        print(f"✅ VALIDATION PASSED: {discovery_rate:.0%} discovery rate (≥90% required)")
    elif discovery_rate >= 0.8:
        print(f"⚠️  VALIDATION MARGINAL: {discovery_rate:.0%} discovery rate (target: 90%)")
    else:
        print(f"❌ VALIDATION FAILED: {discovery_rate:.0%} discovery rate (<80%)")
    print(f"{'='*70}")

    # Save results
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = CHECKPOINT_DIR / "validation_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'n_seeds': n_seeds,
            'discovery_rate': discovery_rate,
            'xor_successes': xor_successes,
            'mean_accuracies': mean_accs,
            'all_results': all_results
        }, f, indent=2)

    print(f"\n✓ Results saved to {results_path}")
    return all_results, discovery_rate


def main():
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--validate':
        n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        run_validation_suite(n_seeds=n_seeds)
    else:
        run_discovery_experiment(seed=42)


if __name__ == "__main__":
    main()
