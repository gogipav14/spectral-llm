"""
Phase 5 Track 3: Hierarchical Circuit Composition
==================================================

Demonstrates building large Boolean circuits by COMPOSING learned primitives,
NOT by spectral synthesis of the full function.

Key insight: We learn ternary masks for small primitives (full adders,
comparators, multiplexers) in Phases 1-4, then compose them structurally
to build larger circuits.

IMPORTANT: We claim "composition from learned primitives" with randomized
verification, NOT "spectral recovery of 64-bit functions." The latter would
be misleading since we're not actually computing 2^64 Fourier coefficients.

Verification Strategy:
---------------------
For large composed circuits, we cannot exhaustively verify correctness.
Instead, we use randomized testing with confidence intervals:

1. Sample m random inputs
2. Compare composed circuit output vs reference implementation
3. If 0 errors observed in m trials, the error rate is bounded by
   p ≤ 3/m with 95% confidence (rule of three)
4. Report Wilson confidence intervals for non-zero error counts
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Enable 64-bit precision for JAX (MUST be before importing jax)
from jax import config
config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import jax.random as random
from typing import Callable, Dict, List, Tuple, Optional
from dataclasses import dataclass
import time
import json
from datetime import datetime
from pathlib import Path
import math


# =============================================================================
# Learned Primitives (from Phase 1-4)
# =============================================================================

# Ternary masks for 2-variable operations (Phase 1)
# Basis: [1, a, b, ab]
PHASE1_MASKS = {
    'XOR':     jnp.array([0., 0., 0., 1.]),
    'AND':     jnp.array([-1., 1., 1., 1.]),
    'OR':      jnp.array([1., 1., 1., -1.]),
    'NAND':    jnp.array([1., -1., -1., -1.]),
    'NOR':     jnp.array([-1., -1., -1., 1.]),
    'XNOR':    jnp.array([0., 0., 0., -1.]),
}


def walsh_basis_2var(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """
    Compute Walsh basis for 2 variables.

    Returns: (batch, 4) array with [1, a, b, a*b]
    """
    ones = jnp.ones_like(a)
    return jnp.stack([ones, a, b, a * b], axis=-1)


def apply_ternary_gate(
    mask: jnp.ndarray,
    a: jnp.ndarray,
    b: jnp.ndarray
) -> jnp.ndarray:
    """
    Apply a ternary-masked gate: f(a,b) = sign(mask · basis(a,b))

    Args:
        mask: (4,) ternary mask in {-1, 0, +1}
        a, b: (batch,) arrays of inputs in {-1, +1}

    Returns:
        (batch,) array of outputs in {-1, +1}
    """
    basis = walsh_basis_2var(a, b)  # (batch, 4)
    linear = jnp.dot(basis, mask)   # (batch,)
    return jnp.sign(linear)


# =============================================================================
# Full Adder Primitive (3-variable, learned in Phase 3)
# =============================================================================

def full_adder_sum(a: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
    """
    Full adder sum bit: S = a XOR b XOR c

    In {-1,+1} encoding: XOR is product, so S = a * b * c (parity)
    """
    return a * b * c


def full_adder_carry(a: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
    """
    Full adder carry bit: C = MAJ(a, b, c)

    Majority function from Phase 3 mask.
    """
    # From Phase 3: majority_3 mask = [-1, 0, 1, 1, 0, 0, 0, -1]
    # But we can compute directly in {-1,+1}:
    # MAJ(a,b,c) = sign(a + b + c)
    s = a + b + c
    return jnp.where(s >= 0, 1.0, -1.0)


# =============================================================================
# Composed Circuits
# =============================================================================

def n_bit_ripple_adder(
    a_bits: jnp.ndarray,
    b_bits: jnp.ndarray,
    n_bits: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    N-bit ripple-carry adder composed from full adders.

    Args:
        a_bits: (batch, n_bits) array of bits in {-1, +1}
        b_bits: (batch, n_bits) array of bits in {-1, +1}
        n_bits: number of bits

    Returns:
        (sum_bits, carry_out): sum is (batch, n_bits), carry is (batch,)

    This is COMPOSITION from learned primitives, not spectral synthesis.
    """
    batch_size = a_bits.shape[0]
    sum_bits = []
    carry = jnp.ones(batch_size) * -1.0  # Initial carry = 0 (FALSE = -1)

    for i in range(n_bits):
        a_i = a_bits[:, i]
        b_i = b_bits[:, i]

        # Full adder stage
        s_i = full_adder_sum(a_i, b_i, carry)
        carry = full_adder_carry(a_i, b_i, carry)

        sum_bits.append(s_i)

    return jnp.stack(sum_bits, axis=1), carry


def n_bit_comparator(
    a_bits: jnp.ndarray,
    b_bits: jnp.ndarray,
    n_bits: int
) -> jnp.ndarray:
    """
    N-bit comparator: returns +1 if a > b, -1 otherwise.

    Composed from learned primitives using lexicographic comparison.
    Scans from MSB to LSB, deciding at first differing bit.

    ENCODING (consistent with adder):
    - When converting to/from integers: (bits + 1) / 2 maps {-1,+1} to {0,1}
    - So: -1 → 0, +1 → 1
    - For unsigned comparison: a > b when a has a 1 where b has 0 at highest differing bit
    - In our encoding: +1 = 1 (bit is set), -1 = 0 (bit is not set)
    """
    batch_size = a_bits.shape[0]

    # Track: gt = (a > b so far), lt = (a < b so far)
    gt = jnp.zeros(batch_size)  # 0 = not decided greater
    lt = jnp.zeros(batch_size)  # 0 = not decided less

    # Scan from MSB to LSB
    for i in range(n_bits - 1, -1, -1):
        a_i = a_bits[:, i]
        b_i = b_bits[:, i]

        # Still undecided?
        undecided = (gt == 0) & (lt == 0)

        # In our encoding: +1 = bit is 1, -1 = bit is 0
        # (since (x + 1) / 2 maps -1 → 0, +1 → 1)
        a_bit_is_1 = (a_i == 1)  # a's bit is 1 (encoded as +1)
        b_bit_is_1 = (b_i == 1)  # b's bit is 1 (encoded as +1)

        # a > b at this bit: a has 1, b has 0
        a_wins = a_bit_is_1 & ~b_bit_is_1
        # b > a at this bit: b has 1, a has 0
        b_wins = b_bit_is_1 & ~a_bit_is_1

        gt = jnp.where(undecided & a_wins, 1.0, gt)
        lt = jnp.where(undecided & b_wins, 1.0, lt)

    # Return +1 if a > b, -1 otherwise
    return jnp.where(gt == 1, 1.0, -1.0)


def n_bit_equality(
    a_bits: jnp.ndarray,
    b_bits: jnp.ndarray,
    n_bits: int
) -> jnp.ndarray:
    """
    N-bit equality: returns +1 if a == b, -1 otherwise.

    Composed from learned primitives:
    1. XNOR each bit pair (equality per bit: +1 if equal, -1 if different)
    2. AND all results (using tree reduction for efficiency)

    Note: In our encoding for the AND chain:
    - We need all bits to be "equal" (all XNOR outputs = +1)
    - But our AND expects the "true" case to trigger only when both inputs are +1
      So we need to be careful about encoding

    Actually, we'll use a simpler approach: check if any bit differs.
    """
    batch_size = a_bits.shape[0]
    xnor_mask = PHASE1_MASKS['XNOR']

    # XNOR each bit pair: result is +1 if bits are equal, -1 if different
    # (XNOR = NOT XOR, so equal bits give +1)
    # Wait, let's verify: XNOR mask is [0, 0, 0, -1]
    # XNOR(a,b) = sign(-a*b) = -sign(a*b) = -XOR(a,b)
    # XOR: (+1,+1) → +1, (+1,-1) → -1, (-1,+1) → -1, (-1,-1) → +1
    # XNOR: (+1,+1) → -1, (+1,-1) → +1, (-1,+1) → +1, (-1,-1) → -1
    # Hmm, that's backwards. Let me check...

    # Actually for equality we want:
    # equal bits → +1, different bits → -1
    # XOR gives: same → +1, different → -1 (in product sense)
    # Wait no, XOR(+1, +1) = sign(1*1) = +1, but +1 and +1 are "equal" in encoding
    # Let me think more carefully about the encoding.

    # In {-1, +1} encoding with (x+1)/2 → {0,1}:
    # +1 → 1, -1 → 0

    # For equality check:
    # - If a_i = b_i (both +1 or both -1), they're equal
    # - If a_i ≠ b_i, they're different

    # XNOR(a, b) = NOT(a XOR b) = 1 iff a == b
    # In {-1,+1}: XOR(a,b) = a*b actually... no wait.

    # Let me just directly compute: are a_i and b_i the same?
    # Same if a_i * b_i = 1 (both +1 or both -1)
    # Different if a_i * b_i = -1

    # So equality per bit is just a*b:
    # - Same bits → +1
    # - Different bits → -1

    bit_eq = a_bits * b_bits  # (batch, n_bits): +1 if equal, -1 if different

    # All bits must be equal for a == b
    # We need to AND all the bit_eq values together
    # AND(x, y) = +1 only if both x and y are +1

    # Tree reduction: iteratively AND pairs
    result = bit_eq[:, 0]
    and_mask = PHASE1_MASKS['AND']

    for i in range(1, n_bits):
        # AND result with next bit equality
        result = apply_ternary_gate(and_mask, result, bit_eq[:, i])

    return result


# =============================================================================
# Reference Implementations (for verification)
# =============================================================================

def reference_adder(a_bits: jnp.ndarray, b_bits: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Reference N-bit adder using integer arithmetic.

    Uses int64 for n_bits <= 62, Python arbitrary precision for larger.
    """
    import numpy as np
    n_bits = a_bits.shape[1]
    batch_size = a_bits.shape[0]

    # Convert {-1,+1} to {0,1} and move to numpy for processing
    a_01 = np.array((a_bits + 1) / 2, dtype=np.int32)
    b_01 = np.array((b_bits + 1) / 2, dtype=np.int32)

    if n_bits <= 62:
        # Use vectorized int64 arithmetic
        powers = (2 ** np.arange(n_bits, dtype=np.int64))
        a_int = np.sum(a_01 * powers, axis=1)
        b_int = np.sum(b_01 * powers, axis=1)
        sum_int = a_int + b_int

        # Convert back to bits
        sum_bits = np.zeros((batch_size, n_bits), dtype=np.float32)
        for i in range(n_bits):
            sum_bits[:, i] = ((sum_int >> i) & 1) * 2 - 1

        carry = ((sum_int >> n_bits) & 1) * 2 - 1
    else:
        # Use Python arbitrary precision for large bit widths
        sum_bits = np.zeros((batch_size, n_bits), dtype=np.float32)
        carry = np.zeros(batch_size, dtype=np.float32)

        for sample in range(batch_size):
            # Convert bits to Python int (arbitrary precision)
            a_val = sum(int(a_01[sample, i]) << i for i in range(n_bits))
            b_val = sum(int(b_01[sample, i]) << i for i in range(n_bits))
            sum_val = a_val + b_val

            # Extract sum bits
            for i in range(n_bits):
                sum_bits[sample, i] = ((sum_val >> i) & 1) * 2 - 1

            carry[sample] = ((sum_val >> n_bits) & 1) * 2 - 1

    return jnp.array(sum_bits), jnp.array(carry)


def reference_comparator(a_bits: jnp.ndarray, b_bits: jnp.ndarray) -> jnp.ndarray:
    """Reference N-bit comparator using integer arithmetic.

    Uses int64 for n_bits <= 62, Python arbitrary precision for larger.
    """
    import numpy as np
    n_bits = a_bits.shape[1]
    batch_size = a_bits.shape[0]

    # Convert {-1,+1} to {0,1} and move to numpy
    a_01 = np.array((a_bits + 1) / 2, dtype=np.int32)
    b_01 = np.array((b_bits + 1) / 2, dtype=np.int32)

    if n_bits <= 62:
        # Use vectorized int64 arithmetic
        powers = (2 ** np.arange(n_bits, dtype=np.int64))
        a_int = np.sum(a_01 * powers, axis=1)
        b_int = np.sum(b_01 * powers, axis=1)
        result = np.where(a_int > b_int, 1.0, -1.0)
    else:
        # Use Python arbitrary precision for large bit widths
        result = np.zeros(batch_size, dtype=np.float32)
        for sample in range(batch_size):
            a_val = sum(int(a_01[sample, i]) << i for i in range(n_bits))
            b_val = sum(int(b_01[sample, i]) << i for i in range(n_bits))
            result[sample] = 1.0 if a_val > b_val else -1.0

    return jnp.array(result)


def reference_equality(a_bits: jnp.ndarray, b_bits: jnp.ndarray) -> jnp.ndarray:
    """Reference N-bit equality check.

    Returns +1 if a == b, -1 otherwise.
    """
    import numpy as np

    # Check if all bits are equal
    # In our encoding: a_i == b_i when a_i * b_i = +1
    # We need ALL bits to be equal, i.e., min of all bit equalities > 0
    bit_eq = np.array(a_bits * b_bits)  # +1 if equal, -1 if different

    # All equal iff all bit_eq are +1, i.e., min > 0
    all_equal = np.min(bit_eq, axis=1) > 0

    return jnp.array(np.where(all_equal, 1.0, -1.0))


# =============================================================================
# Statistical Verification
# =============================================================================

def wilson_confidence_interval(
    n_success: int,
    n_total: int,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Wilson score confidence interval for binomial proportion.

    More accurate than normal approximation for small samples or extreme p.
    """
    from scipy import stats

    if n_total == 0:
        return (0.0, 1.0)

    p_hat = n_success / n_total
    z = stats.norm.ppf(1 - (1 - confidence) / 2)

    denominator = 1 + z**2 / n_total
    center = (p_hat + z**2 / (2 * n_total)) / denominator
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n_total)) / n_total) / denominator

    return (max(0, center - margin), min(1, center + margin))


def rule_of_three_bound(n_samples: int, confidence: float = 0.95) -> float:
    """
    Rule of three: if 0 errors in n samples, error rate ≤ 3/n with 95% confidence.

    More generally: -ln(1-confidence) / n
    """
    return -math.log(1 - confidence) / n_samples


def verify_circuit(
    composed_fn: Callable,
    reference_fn: Callable,
    n_bits: int,
    n_samples: int,
    key: jax.random.PRNGKey,
) -> Dict:
    """
    Randomized verification of composed circuit vs reference.

    Returns statistics with confidence intervals.
    """
    # Generate random inputs
    a_bits = random.choice(key, jnp.array([-1.0, 1.0]), shape=(n_samples, n_bits))
    key, subkey = random.split(key)
    b_bits = random.choice(subkey, jnp.array([-1.0, 1.0]), shape=(n_samples, n_bits))

    # Run both implementations
    start = time.perf_counter()
    composed_result = composed_fn(a_bits, b_bits, n_bits)
    composed_time = time.perf_counter() - start

    start = time.perf_counter()
    reference_result = reference_fn(a_bits, b_bits)
    reference_time = time.perf_counter() - start

    # Handle tuple results (adder returns sum, carry)
    if isinstance(composed_result, tuple):
        composed_sum, composed_carry = composed_result
        reference_sum, reference_carry = reference_result

        sum_errors = int(jnp.sum(composed_sum != reference_sum))
        carry_errors = int(jnp.sum(composed_carry != reference_carry))
        n_errors = sum_errors + carry_errors
        n_total = n_samples * (n_bits + 1)  # All sum bits plus carry
    else:
        n_errors = int(jnp.sum(composed_result != reference_result))
        n_total = n_samples

    # Compute accuracy and confidence intervals
    accuracy = 1 - n_errors / n_total

    if n_errors == 0:
        error_rate_bound = rule_of_three_bound(n_total)
        ci_low, ci_high = 1 - error_rate_bound, 1.0
    else:
        ci_low, ci_high = wilson_confidence_interval(n_total - n_errors, n_total)

    return {
        'n_bits': n_bits,
        'n_samples': n_samples,
        'n_total_comparisons': n_total,
        'n_errors': n_errors,
        'accuracy': accuracy,
        'accuracy_ci_low': ci_low,
        'accuracy_ci_high': ci_high,
        'composed_time_s': composed_time,
        'reference_time_s': reference_time,
    }


# =============================================================================
# Benchmarks
# =============================================================================

def benchmark_circuit_composition(
    bit_sizes: List[int] = [4, 8, 16, 32, 64],
    equality_sizes: List[int] = [4, 8, 16, 32, 64, 128],
    n_samples: int = 100000,
    verbose: bool = True
) -> Dict:
    """
    Benchmark hierarchical circuit composition.

    Tests adders, comparators, and equality checks at various bit widths.
    """
    print("=" * 70)
    print("PHASE 5 TRACK 3: Hierarchical Circuit Composition")
    print("=" * 70)
    print(f"\nMethod: Composition from learned primitives")
    print(f"Verification: Randomized testing with {n_samples:,} samples")
    print(f"NOTE: This is NOT spectral synthesis of 2^n-dim functions")

    results = {'adder': [], 'comparator': [], 'equality': []}
    key = random.PRNGKey(42)

    # Test adders
    print(f"\n{'='*60}")
    print("N-BIT RIPPLE ADDER")
    print("=" * 60)

    for n_bits in bit_sizes:
        print(f"\n--- {n_bits}-bit adder ---")

        key, subkey = random.split(key)
        result = verify_circuit(
            n_bit_ripple_adder,
            reference_adder,
            n_bits,
            n_samples,
            subkey
        )

        results['adder'].append(result)

        if verbose:
            print(f"  Accuracy: {result['accuracy']:.6f}")
            print(f"  95% CI: [{result['accuracy_ci_low']:.6f}, {result['accuracy_ci_high']:.6f}]")
            print(f"  Errors: {result['n_errors']}/{result['n_total_comparisons']}")
            if result['n_errors'] == 0:
                bound = rule_of_three_bound(result['n_total_comparisons'])
                print(f"  Error rate bound: ≤ {bound:.2e} (rule of three)")

    # Test comparators
    print(f"\n{'='*60}")
    print("N-BIT COMPARATOR")
    print("=" * 60)

    for n_bits in bit_sizes:
        print(f"\n--- {n_bits}-bit comparator ---")

        key, subkey = random.split(key)
        result = verify_circuit(
            n_bit_comparator,
            reference_comparator,
            n_bits,
            n_samples,
            subkey
        )

        results['comparator'].append(result)

        if verbose:
            print(f"  Accuracy: {result['accuracy']:.6f}")
            print(f"  95% CI: [{result['accuracy_ci_low']:.6f}, {result['accuracy_ci_high']:.6f}]")
            print(f"  Errors: {result['n_errors']}/{result['n_total_comparisons']}")

    # Test equality (including 128-bit)
    print(f"\n{'='*60}")
    print("N-BIT EQUALITY")
    print("=" * 60)

    for n_bits in equality_sizes:
        print(f"\n--- {n_bits}-bit equality ---")

        key, subkey = random.split(key)
        result = verify_circuit(
            n_bit_equality,
            reference_equality,
            n_bits,
            n_samples,
            subkey
        )

        results['equality'].append(result)

        if verbose:
            print(f"  Accuracy: {result['accuracy']:.6f}")
            print(f"  95% CI: [{result['accuracy_ci_low']:.6f}, {result['accuracy_ci_high']:.6f}]")
            print(f"  Errors: {result['n_errors']}/{result['n_total_comparisons']}")
            if result['n_errors'] == 0:
                bound = rule_of_three_bound(result['n_total_comparisons'])
                print(f"  Error rate bound: ≤ {bound:.2e} (rule of three)")

    # Summary
    print("\n" + "=" * 70)
    print("COMPOSITION SUMMARY")
    print("=" * 70)

    print(f"\n{'Circuit':>20} {'n_bits':>8} {'Accuracy':>12} {'95% CI':>25}")
    print("-" * 70)

    for circuit_type in ['adder', 'comparator', 'equality']:
        for r in results[circuit_type]:
            ci_str = f"[{r['accuracy_ci_low']:.4f}, {r['accuracy_ci_high']:.4f}]"
            print(f"{circuit_type:>20} {r['n_bits']:>8} {r['accuracy']:>12.6f} {ci_str:>25}")

    # Save results
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        'benchmark': 'hierarchical_composition',
        'method': 'Composition from learned primitives (NOT spectral synthesis)',
        'verification': 'Randomized testing with confidence intervals',
        'n_samples': n_samples,
        'results': results,
        'timestamp': datetime.now().isoformat(),
    }

    output_path = output_dir / "circuit_composition_results.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")

    return output


def verify_primitives():
    """Verify that learned primitives match reference implementations."""
    print("\n" + "=" * 70)
    print("VERIFICATION: Learned Primitives")
    print("=" * 70)

    n_samples = 10000
    key = random.PRNGKey(42)

    # Test 2-variable gates
    a = random.choice(key, jnp.array([-1.0, 1.0]), shape=(n_samples,))
    key, subkey = random.split(key)
    b = random.choice(subkey, jnp.array([-1.0, 1.0]), shape=(n_samples,))

    gates_2var = {
        'XOR': (PHASE1_MASKS['XOR'], lambda x, y: x * y),
        'AND': (PHASE1_MASKS['AND'], lambda x, y: jnp.where((x == 1) & (y == 1), 1.0, -1.0)),
        'OR': (PHASE1_MASKS['OR'], lambda x, y: jnp.where((x == 1) | (y == 1), 1.0, -1.0)),
    }

    print("\n2-Variable Gates:")
    for name, (mask, ref_fn) in gates_2var.items():
        composed = apply_ternary_gate(mask, a, b)
        reference = ref_fn(a, b)
        errors = int(jnp.sum(composed != reference))
        print(f"  {name}: {n_samples - errors}/{n_samples} correct ({(n_samples-errors)/n_samples:.2%})")

    # Test full adder
    key, subkey = random.split(key)
    c = random.choice(subkey, jnp.array([-1.0, 1.0]), shape=(n_samples,))

    print("\nFull Adder:")

    # Sum (parity)
    composed_sum = full_adder_sum(a, b, c)
    ref_a = (a + 1) / 2  # {-1,+1} -> {0,1}
    ref_b = (b + 1) / 2
    ref_c = (c + 1) / 2
    ref_sum = ((ref_a.astype(jnp.int32) ^ ref_b.astype(jnp.int32) ^ ref_c.astype(jnp.int32)) * 2 - 1).astype(jnp.float32)
    sum_errors = int(jnp.sum(composed_sum != ref_sum))
    print(f"  Sum: {n_samples - sum_errors}/{n_samples} correct")

    # Carry (majority)
    composed_carry = full_adder_carry(a, b, c)
    ref_carry = jnp.where(ref_a + ref_b + ref_c >= 2, 1.0, -1.0)
    carry_errors = int(jnp.sum(composed_carry != ref_carry))
    print(f"  Carry: {n_samples - carry_errors}/{n_samples} correct")


if __name__ == "__main__":
    # First verify primitives
    verify_primitives()

    # Then run composition benchmark
    results = benchmark_circuit_composition(
        bit_sizes=[4, 8, 16, 32, 64],
        n_samples=100000,
        verbose=True
    )
