"""
Phase 1 Discovery Training: Proving Spectral Emergence Without Supervision
===========================================================================

This script validates that the spectral structure (XOR → parity character)
emerges NATURALLY from generic priors, NOT from explicit supervision.

Key differences from train_phase1_fixed.py:
- REMOVED: Operation-specific mask regularization
- REMOVED: Explicit purity loss targeting index 3
- REMOVED: ANY reference to expected mask values [0,0,0,1]

KEPT:
- Soft-ternary annealing
- Sequential training
- Generic priors only

ADDED:
- Random feature permutation per run
- Post-hoc permutation inversion
- Statistical validation over multiple seeds
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
N_EPOCHS_XOR = 100  # More epochs for XOR (hardest to discover)
N_EPOCHS_OTHER = 50  # Other operations are easier
LEARNING_RATE = 0.1
THRESHOLD = 0.3
ACCURACY_THRESHOLD = 0.99  # Early stopping threshold

CHECKPOINT_DIR = Path("v5/checkpoints/phase1_discovery")


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


# =============================================================================
# GENERIC PRIORS (No operation-specific supervision!)
# =============================================================================

def sparsity_loss(mask: jnp.ndarray) -> jnp.ndarray:
    """
    L1 sparsity prior: encourage most mask values to be near zero.
    This is generic - doesn't specify WHICH positions should be zero.
    """
    return jnp.mean(jnp.abs(mask))


def single_peak_entropy_loss(mask: jnp.ndarray, temperature: float = 0.1) -> jnp.ndarray:
    """
    Entropy-based prior: encourage mask energy to concentrate on ONE position.
    Uses softmax of absolute values to compute "attention" distribution.
    Low entropy = energy concentrated on single peak.
    """
    # Softmax over absolute values (higher abs = more attention)
    abs_mask = jnp.abs(mask)
    probs = jax.nn.softmax(abs_mask / temperature)

    # Entropy: lower is better (more concentrated)
    entropy = -jnp.sum(probs * jnp.log(probs + 1e-8))

    # Normalize by max entropy (uniform = log(4))
    max_entropy = jnp.log(len(mask))
    return entropy / max_entropy


def orthogonality_loss(masks: List[jnp.ndarray]) -> jnp.ndarray:
    """
    Encourage masks to be orthogonal to each other.
    This is generic - doesn't specify what the masks should be.
    """
    masks_stack = jnp.stack(masks)  # [4, 4]
    norms = jnp.linalg.norm(masks_stack, axis=1, keepdims=True) + 1e-8
    normalized = masks_stack / norms

    # Gram matrix
    gram = normalized @ normalized.T  # [4, 4]

    # Off-diagonal should be zero
    eye = jnp.eye(len(masks))
    off_diag = gram * (1 - eye)

    return jnp.mean(off_diag ** 2)


# =============================================================================
# PERMUTED BOOLEAN FOURIER LAYER
# =============================================================================

class PermutedBooleanFourierLayer(nn.Module):
    """
    Logic layer with RANDOM feature ordering.

    The true feature order is [1, a, b, ab] but we apply a random permutation
    so the model doesn't know which index corresponds to which character.

    After training, we invert the permutation to check if the model
    "discovered" the correct spectral structure.
    """
    n_bits: int = 64
    permutation: jnp.ndarray = None  # [4] permutation of [0,1,2,3]

    @nn.compact
    def __call__(self, a, b, operation_id, temperature: float = 1.0, training: bool = True):
        # Initialize 4 continuous masks
        xor_mask = self.param('xor_mask', nn.initializers.normal(0.5), (4,))
        and_mask = self.param('and_mask', nn.initializers.normal(0.5), (4,))
        or_mask = self.param('or_mask', nn.initializers.normal(0.5), (4,))
        implies_mask = self.param('implies_mask', nn.initializers.normal(0.5), (4,))

        masks = [xor_mask, and_mask, or_mask, implies_mask]
        raw_mask = masks[operation_id]

        # Apply soft ternary for differentiable training
        if training:
            mask = soft_ternary(raw_mask, temperature, THRESHOLD)
        else:
            mask = ternary_quantize(raw_mask, THRESHOLD)

        # Boolean Fourier features: [1, a, b, ab]
        ones = jnp.ones_like(a)
        ab = a * b
        features_canonical = jnp.stack([ones, a, b, ab], axis=-1)  # [batch, n_bits, 4]

        # Apply permutation to features (model sees permuted order)
        if self.permutation is not None:
            features = features_canonical[:, :, self.permutation]
        else:
            features = features_canonical

        # Apply mask
        masked = features * mask  # [batch, n_bits, 4]

        # Sum over Fourier characters
        output = jnp.sum(masked, axis=-1)  # [batch, n_bits]

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
    """
    Training step with ONLY generic priors.
    NO operation-specific supervision!

    Key insight: Task loss is the PRIMARY signal for discovery.
    Priors should be LIGHT to not interfere with task learning.
    """

    def loss_fn(params):
        # Forward pass
        pred = apply_fn({'params': params}, a, b, op_id, temperature=temperature, training=True)
        task_loss = compute_hamming_loss(pred, target)

        # Generic priors - ONLY apply to current operation's mask
        # This prevents gradient interference between operations
        mask_names = ['xor_mask', 'and_mask', 'or_mask', 'implies_mask']
        raw_mask = params[mask_names[op_id]]

        # 1. Ternary attractor (pull toward {-1, 0, +1})
        attractor = ternary_attractor_loss(raw_mask)

        # 2. Light sparsity (encourage some zeros, but don't force it)
        sparse = sparsity_loss(raw_mask)

        # Weighted combination - TASK LOSS IS PRIMARY
        # NO orthogonality loss - it causes gradient interference!
        total_loss = (
            task_loss +              # Primary signal
            0.01 * attractor +       # Very light ternary push
            0.005 * sparse           # Very light sparsity
        )

        return total_loss, (task_loss, attractor, sparse)

    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)

    return state, loss, aux


def compute_accuracy(apply_fn, params, a, b, target, op_id, permutation=None):
    """Compute accuracy for a single operation."""
    # Create model with permutation for inference
    pred = apply_fn({'params': params}, a, b, op_id, temperature=0.01, training=False)
    return float(jnp.mean(pred == target))


def invert_permutation(perm: jnp.ndarray) -> np.ndarray:
    """Compute inverse permutation."""
    perm_np = np.array(perm, dtype=int)
    inv = np.zeros(len(perm_np), dtype=int)
    for i, p in enumerate(perm_np):
        inv[p] = i
    return inv


def analyze_discovered_mask(raw_mask: jnp.ndarray, perm: jnp.ndarray, op_name: str) -> Dict:
    """
    Analyze what the model discovered after inverting permutation.
    Returns metrics about which canonical feature got the most weight.

    Key insight:
    - Model sees features in permuted order: features_permuted[j] = features_canonical[P[j]]
    - If model learns mask with weight at index j, it selected canonical feature P[j]
    - To get canonical mask: canonical[P[j]] = permuted[j]
    """
    # Quantize mask (in permuted/model space)
    ternary_mask = np.array(ternary_quantize(raw_mask, THRESHOLD))
    perm_np = np.array(perm, dtype=int)

    # Convert to canonical ordering: canonical[P[j]] = permuted[j]
    canonical_mask = np.zeros(4)
    for j in range(4):
        canonical_mask[perm_np[j]] = ternary_mask[j]

    # Compute energy per canonical feature
    abs_canonical = np.abs(canonical_mask)
    total_energy = np.sum(abs_canonical) + 1e-8

    # Feature names in canonical order: [1, a, b, ab]
    feature_names = ['const(1)', 'a', 'b', 'ab(parity)']

    # Find dominant feature
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
        'feature_energies': {
            name: float(abs_canonical[i])
            for i, name in enumerate(feature_names)
        }
    }


def run_discovery_experiment(seed: int) -> Dict:
    """
    Run a single discovery experiment with random permutation.
    Returns whether XOR correctly discovered parity (ab).
    """
    print(f"\n{'='*60}")
    print(f"Discovery Experiment - Seed {seed}")
    print(f"{'='*60}")

    # Generate random permutation
    rng = jax.random.PRNGKey(seed)
    permutation = jax.random.permutation(rng, 4)
    print(f"\nFeature permutation: {permutation}")
    print(f"  Original order: [1, a, b, ab]")
    feature_labels = ['1', 'a', 'b', 'ab']
    permuted_order = [feature_labels[int(permutation[i])] for i in range(4)]
    print(f"  Permuted order: {permuted_order}")

    # Create dataset
    train_data, test_data = create_train_test_split(N_TRAIN, N_TEST, N_BITS, seed=seed)

    # Initialize model with permutation
    model = PermutedBooleanFourierLayer(n_bits=N_BITS, permutation=permutation)

    rng = jax.random.PRNGKey(seed + 1000)
    dummy_a = jnp.ones((1, N_BITS))
    dummy_b = jnp.ones((1, N_BITS))
    variables = model.init(rng, dummy_a, dummy_b, 0)

    # Create optimizer
    optimizer = optax.adam(LEARNING_RATE)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optimizer
    )

    # Sequential training: XOR → AND → OR → IMPLIES
    op_order = [0, 1, 2, 3]  # xor, and, or, implies
    op_names = ['xor', 'and', 'or', 'implies']

    for op_id in op_order:
        op_name = op_names[op_id]
        n_epochs = N_EPOCHS_XOR if op_id == 0 else N_EPOCHS_OTHER

        print(f"\n--- Training {op_name.upper()} ({n_epochs} epochs) ---")

        a, b, target, _ = train_data[op_name]
        test_a, test_b, test_target, _ = test_data[op_name]

        best_acc = 0.0
        converged = False

        # Temperature annealing schedule (slower for XOR)
        for epoch in range(n_epochs):
            # Slower annealing: 1.0 → 0.05 (not all the way to 0.01)
            progress = epoch / (n_epochs - 1)
            temperature = 1.0 * (0.05 / 1.0) ** progress

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

            # Check accuracy every 5 epochs
            if (epoch + 1) % 5 == 0 or epoch == 0:
                acc = compute_accuracy(model.apply, state.params, test_a, test_b, test_target, op_id)

                mask_name = f'{op_name}_mask'
                raw_mask = state.params[mask_name]
                ternary = ternary_quantize(raw_mask, THRESHOLD)

                if (epoch + 1) % 10 == 0 or epoch == 0 or acc > best_acc:
                    print(f"  Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | Acc: {acc:.2%} | T: {temperature:.3f}")
                    print(f"            | Mask: {ternary}")

                if acc > best_acc:
                    best_acc = acc

                # Early stopping if converged
                if acc >= ACCURACY_THRESHOLD and temperature < 0.2:
                    print(f"  ✓ Converged at epoch {epoch+1} with {acc:.2%} accuracy")
                    converged = True
                    break

        if not converged:
            print(f"  Final: {best_acc:.2%} accuracy after {n_epochs} epochs")

    # Final evaluation and analysis
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

        # Analyze discovered structure
        analysis = analyze_discovered_mask(raw_mask, permutation, op_name)
        results['operations'][op_name] = analysis

        # Test accuracy
        test_a, test_b, test_target, _ = test_data[op_name]
        acc = compute_accuracy(model.apply, state.params, test_a, test_b, test_target, op_id)
        results['accuracies'][op_name] = acc

        print(f"\n{op_name.upper()}:")
        print(f"  Permuted mask:  {analysis['ternary_mask']}")
        print(f"  Canonical mask: {analysis['canonical_mask']}")
        print(f"  Dominant feature: {analysis['dominant_feature']} ({analysis['concentration']:.1%})")
        print(f"  Accuracy: {acc:.2%}")

    # Check XOR discovery success
    xor_analysis = results['operations']['xor']
    xor_success = (xor_analysis['dominant_idx'] == 3)  # Index 3 = ab (parity)
    results['xor_discovered_parity'] = xor_success

    print(f"\n{'='*60}")
    if xor_success:
        print(f"✅ XOR DISCOVERED PARITY! (ab character)")
    else:
        print(f"❌ XOR did NOT discover parity. Found: {xor_analysis['dominant_feature']}")
    print(f"{'='*60}")

    return results


def run_validation_suite(n_seeds: int = 10):
    """
    Run discovery experiments over multiple seeds.
    Reports success rate for XOR → parity discovery.
    """
    print("\n" + "="*70)
    print("PHASE 1 DISCOVERY VALIDATION SUITE")
    print("="*70)
    print(f"\nRunning {n_seeds} experiments with random feature permutations...")
    print("Goal: Verify XOR naturally discovers parity (ab) without supervision")

    all_results = []
    xor_successes = 0

    for seed in range(n_seeds):
        result = run_discovery_experiment(seed=seed * 42)
        all_results.append(result)

        if result['xor_discovered_parity']:
            xor_successes += 1

    # Summary statistics
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)

    discovery_rate = xor_successes / n_seeds
    print(f"\nXOR → Parity Discovery Rate: {xor_successes}/{n_seeds} = {discovery_rate:.1%}")

    # Accuracy statistics
    mean_accs = {op: np.mean([r['accuracies'][op] for r in all_results]) for op in ['xor', 'and', 'or', 'implies']}
    print(f"\nMean Accuracies:")
    for op, acc in mean_accs.items():
        print(f"  {op.upper():8s}: {acc:.2%}")

    # Success criterion
    print(f"\n{'='*70}")
    if discovery_rate >= 0.8:
        print(f"✅ VALIDATION PASSED: {discovery_rate:.0%} discovery rate (≥80% required)")
        print("   Spectral structure emerges naturally from generic priors!")
    else:
        print(f"❌ VALIDATION FAILED: {discovery_rate:.0%} discovery rate (<80%)")
        print("   Need stronger priors or architecture changes.")
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
    """Run single experiment or full validation suite."""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--validate':
        n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        run_validation_suite(n_seeds=n_seeds)
    else:
        # Single experiment for quick testing
        result = run_discovery_experiment(seed=42)

        # Save single result
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        with open(CHECKPOINT_DIR / "single_experiment.json", 'w') as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
