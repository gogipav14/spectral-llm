"""
Validation Tests for Phase 1 Logic Layer
=========================================

Comprehensive validation suite for the trained Binary Logic Layer.

Tests:
1. XOR Spectral Spike - Parity character concentration
2. Mask Sparsity - Fraction of zeros
3. Mask Orthogonality - Cosine similarity between masks
4. Operation Accuracy - Per-operation test accuracy
5. Binary Inference - Verify ternary quantization

Success Criteria:
- XOR mask has >90% energy on parity (ab) coordinate
- Masks have <0.3 cosine similarity
- All operations achieve >99% accuracy
- All inference values are in {-1, 0, +1}
"""

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path

from binary_logic_layer import BinaryLogicLayer, GROUND_TRUTH_OPS, EXPECTED_MASKS
from ternary_ops import ternary_quantize
from logic_dataset import generate_logic_dataset, OP_NAMES


THRESHOLD = 0.3


def validate_spectral_spike(
    params: dict,
    threshold: float = 0.3,
    required_concentration: float = 0.9
) -> dict:
    """
    XOR mask should have ALL weight on coordinate 3 (ab character).

    Args:
        params: Model parameters
        threshold: Ternary quantization threshold
        required_concentration: Minimum energy on parity coordinate

    Returns:
        Dict with validation results
    """
    print("\n" + "="*60)
    print("Test 1: XOR Spectral Spike")
    print("="*60)

    results = {'passed': True, 'details': {}}

    for mask_name in ['xor_mask', 'and_mask', 'or_mask', 'implies_mask']:
        if mask_name not in params:
            continue

        w = params[mask_name]['w']
        w_ternary = ternary_quantize(w, threshold)

        # Energy distribution
        abs_ternary = jnp.abs(w_ternary)
        total_energy = jnp.sum(abs_ternary)

        if total_energy > 0:
            energy_dist = abs_ternary / total_energy
        else:
            energy_dist = jnp.zeros(4)

        parity_concentration = float(energy_dist[3])

        results['details'][mask_name] = {
            'continuous': np.array(w),
            'ternary': np.array(w_ternary),
            'energy_distribution': np.array(energy_dist),
            'parity_concentration': parity_concentration
        }

        op_name = mask_name.replace('_mask', '').upper()
        print(f"\n{op_name}:")
        print(f"  Continuous:  {w}")
        print(f"  Ternary:     {w_ternary}")
        print(f"  Energy dist: {energy_dist}")
        print(f"  Parity (ab): {parity_concentration:.1%}")

        # Check XOR specifically
        if mask_name == 'xor_mask':
            passed = parity_concentration >= required_concentration
            results[mask_name + '_passed'] = passed
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  Status: {status} (required: {required_concentration:.0%})")
            if not passed:
                results['passed'] = False

    return results


def validate_sparsity(
    params: dict,
    threshold: float = 0.3,
    min_xor_sparsity: float = 0.5
) -> dict:
    """
    Validate mask sparsity.

    XOR should be mostly sparse (only parity coordinate active).
    """
    print("\n" + "="*60)
    print("Test 2: Mask Sparsity")
    print("="*60)

    results = {'passed': True, 'details': {}}

    for mask_name in ['xor_mask', 'and_mask', 'or_mask', 'implies_mask']:
        if mask_name not in params:
            continue

        w = ternary_quantize(params[mask_name]['w'], threshold)
        sparsity = float(jnp.mean(w == 0))

        op_name = mask_name.replace('_mask', '').upper()
        print(f"  {op_name:8s}: {sparsity:.0%} sparse")

        results['details'][mask_name] = sparsity

        # XOR should be sparse
        if mask_name == 'xor_mask':
            passed = sparsity >= min_xor_sparsity
            results['xor_passed'] = passed
            if not passed:
                print(f"    ✗ XOR sparsity below {min_xor_sparsity:.0%}")
                results['passed'] = False
            else:
                print(f"    ✓ XOR sparsity meets requirement")

    return results


def validate_orthogonality(
    params: dict,
    threshold: float = 0.3,
    max_cosine: float = 0.3
) -> dict:
    """
    Validate that masks are nearly orthogonal.

    Uses scale-invariant cosine similarity.
    """
    print("\n" + "="*60)
    print("Test 3: Mask Orthogonality")
    print("="*60)

    masks = []
    mask_names = []

    for mask_name in ['xor_mask', 'and_mask', 'or_mask', 'implies_mask']:
        if mask_name in params:
            w = ternary_quantize(params[mask_name]['w'], threshold)
            masks.append(w)
            mask_names.append(mask_name.replace('_mask', '').upper())

    if len(masks) < 2:
        print("  Not enough masks for orthogonality check")
        return {'passed': True, 'details': {}}

    masks = jnp.stack(masks)
    norms = jnp.linalg.norm(masks, axis=1, keepdims=True) + 1e-8
    normalized = masks / norms

    cosine_sim = np.array(normalized @ normalized.T)

    print("\nCosine Similarity Matrix:")
    print("          " + " ".join(f"{n:8s}" for n in mask_names))
    for i, name in enumerate(mask_names):
        row = " ".join(f"{cosine_sim[i,j]:8.3f}" for j in range(len(mask_names)))
        print(f"  {name:8s} {row}")

    # Check off-diagonal
    off_diag = cosine_sim - np.diag(np.diag(cosine_sim))
    max_overlap = float(np.abs(off_diag).max())

    print(f"\nMax off-diagonal: {max_overlap:.3f}")

    passed = max_overlap < max_cosine
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"Status: {status} (required: < {max_cosine})")

    return {
        'passed': passed,
        'cosine_matrix': cosine_sim,
        'max_overlap': max_overlap
    }


def validate_accuracy(
    model: BinaryLogicLayer,
    params: dict,
    n_test: int = 2000,
    min_accuracy: float = 0.99
) -> dict:
    """
    Validate per-operation accuracy on test set.
    """
    print("\n" + "="*60)
    print("Test 4: Operation Accuracy")
    print("="*60)

    test_data = generate_logic_dataset(n_test, n_bits=model.n_bits, seed=9999)

    results = {'passed': True, 'accuracies': {}}

    for op_id, op_name in OP_NAMES.items():
        a, b, target, _ = test_data[op_name]
        pred = model.apply({'params': params}, a, b, op_id)
        acc = float(jnp.mean(pred == target))

        results['accuracies'][op_name] = acc

        passed = acc >= min_accuracy
        status = "✓" if passed else "✗"
        print(f"  {status} {op_name.upper():8s}: {acc:.2%}")

        if not passed:
            results['passed'] = False

    mean_acc = np.mean(list(results['accuracies'].values()))
    print(f"\n  Mean accuracy: {mean_acc:.2%}")

    return results


def validate_binary_inference(
    params: dict,
    threshold: float = 0.3
) -> dict:
    """
    Verify inference uses only ternary values.
    """
    print("\n" + "="*60)
    print("Test 5: Binary Inference (Ternary Check)")
    print("="*60)

    results = {'passed': True, 'details': {}}

    for mask_name in ['xor_mask', 'and_mask', 'or_mask', 'implies_mask']:
        if mask_name not in params:
            continue

        w = ternary_quantize(params[mask_name]['w'], threshold)
        valid = bool(jnp.all((w == -1) | (w == 0) | (w == 1)))

        op_name = mask_name.replace('_mask', '').upper()
        status = "✓" if valid else "✗"
        print(f"  {status} {op_name:8s}: values = {set(map(float, w))}")

        results['details'][mask_name] = {
            'valid': valid,
            'values': set(map(float, w))
        }

        if not valid:
            results['passed'] = False

    overall = "✓ All masks are ternary" if results['passed'] else "✗ Some masks have invalid values"
    print(f"\n{overall}")

    return results


def compare_to_expected(params: dict, threshold: float = 0.3) -> dict:
    """
    Compare learned masks to theoretical expected values.
    """
    print("\n" + "="*60)
    print("Comparison to Expected Masks")
    print("="*60)

    results = {}

    expected = {
        'xor': jnp.array([0., 0., 0., 1.]),
        'and': jnp.array([1., 1., 1., 1.]),
        'or': jnp.array([1., 1., 1., -1.]),
        'implies': jnp.array([1., -1., 1., 1.])
    }

    for op_name, exp_mask in expected.items():
        mask_name = f'{op_name}_mask'
        if mask_name not in params:
            continue

        learned = ternary_quantize(params[mask_name]['w'], threshold)

        # Cosine similarity to expected
        norm_l = jnp.linalg.norm(learned) + 1e-8
        norm_e = jnp.linalg.norm(exp_mask) + 1e-8
        cosine = float(jnp.dot(learned, exp_mask) / (norm_l * norm_e))

        # Exact match
        exact = bool(jnp.all(learned == exp_mask))

        print(f"\n{op_name.upper()}:")
        print(f"  Expected: {exp_mask}")
        print(f"  Learned:  {learned}")
        print(f"  Cosine:   {cosine:.3f}")
        print(f"  Match:    {'✓' if exact else '✗'}")

        results[op_name] = {
            'expected': np.array(exp_mask),
            'learned': np.array(learned),
            'cosine': cosine,
            'exact_match': exact
        }

    return results


def run_all_validations(
    model: BinaryLogicLayer,
    params: dict,
    threshold: float = 0.3
) -> dict:
    """
    Run complete validation suite.
    """
    print("\n" + "="*60)
    print("PHASE 1 VALIDATION SUITE")
    print("="*60)

    results = {}

    # Run all tests
    results['spectral_spike'] = validate_spectral_spike(params, threshold)
    results['sparsity'] = validate_sparsity(params, threshold)
    results['orthogonality'] = validate_orthogonality(params, threshold)
    results['accuracy'] = validate_accuracy(model, params)
    results['binary_inference'] = validate_binary_inference(params, threshold)
    results['comparison'] = compare_to_expected(params, threshold)

    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)

    all_passed = True
    tests = [
        ('Spectral Spike (XOR)', results['spectral_spike']['passed']),
        ('Sparsity', results['sparsity']['passed']),
        ('Orthogonality', results['orthogonality']['passed']),
        ('Accuracy', results['accuracy']['passed']),
        ('Binary Inference', results['binary_inference']['passed'])
    ]

    for name, passed in tests:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL TESTS PASSED - Phase 1 Proof Successful!")
    else:
        print("⚠️  SOME TESTS FAILED - Review results above")
    print("="*60)

    results['all_passed'] = all_passed
    return results


if __name__ == "__main__":
    # Load checkpoint if exists
    CHECKPOINT_PATH = Path("v5/checkpoints/phase1_logic/final_params.npy")

    if CHECKPOINT_PATH.exists():
        print(f"Loading checkpoint from {CHECKPOINT_PATH}")
        params = np.load(CHECKPOINT_PATH, allow_pickle=True).item()
    else:
        print("No checkpoint found. Running with random initialization.")
        model = BinaryLogicLayer(n_bits=64, threshold=THRESHOLD)
        rng = jax.random.PRNGKey(42)
        dummy_a = jnp.ones((1, 64))
        dummy_b = jnp.ones((1, 64))
        variables = model.init(rng, dummy_a, dummy_b, 0)
        params = variables['params']

    # Create model
    model = BinaryLogicLayer(n_bits=64, threshold=THRESHOLD)

    # Run validations
    results = run_all_validations(model, params, THRESHOLD)
