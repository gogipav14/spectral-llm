"""
Temporal Dataset for Phase 2 Training
======================================

Temporal patterns that build on Phase 1 logic primitives.
Each temporal operation composes multiple logic operations over time.

Temporal concepts:
- BEFORE: a precedes b in sequence
- AFTER: a follows b in sequence
- DURING: a and b overlap temporally
- UNTIL: a holds until b becomes true
- SINCE: a has been true since b
- ALWAYS: a is always true
- EVENTUALLY: a becomes true at some point
- NEXT: a is true in next step

For Phase 2, we represent temporal sequences as pairs of bit vectors
where each position represents a time step or state.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Tuple


# Operation IDs for temporal tasks
TEMPORAL_OP_NAMES = {
    0: 'rising_edge',      # a transitions from -1 to +1
    1: 'falling_edge',     # a transitions from +1 to -1
    2: 'stable_high',      # a stays at +1
    3: 'stable_low',       # a stays at -1
    4: 'toggle',           # a changes state
    5: 'sync_rise',        # a and b both rise
    6: 'sync_fall',        # a and b both fall
    7: 'a_before_b',       # a rises before b rises
    8: 'a_after_b',        # a rises after b rises
    9: 'concurrent',       # a and b change together
    10: 'majority',        # majority of sequence is +1
    11: 'parity_seq',      # parity over sequence
    12: 'first_high',      # first element is +1
    13: 'last_high',       # last element is +1
    14: 'alternating',     # sequence alternates
    15: 'monotonic_up',    # sequence is monotonically increasing
}


def generate_temporal_sequences(
    n_samples: int,
    seq_len: int = 8,
    seed: int = 42
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate pairs of temporal sequences.

    Args:
        n_samples: Number of sequence pairs to generate
        seq_len: Length of each sequence (default 8 for 8-bit patterns)
        seed: Random seed

    Returns:
        a_seq: [n_samples, seq_len] in {-1, +1}
        b_seq: [n_samples, seq_len] in {-1, +1}
    """
    rng = np.random.default_rng(seed)

    # Generate binary sequences in {-1, +1}
    a_seq = 2 * rng.integers(0, 2, (n_samples, seq_len)) - 1
    b_seq = 2 * rng.integers(0, 2, (n_samples, seq_len)) - 1

    return jnp.array(a_seq, dtype=jnp.float32), jnp.array(b_seq, dtype=jnp.float32)


# ============================================================================
# Temporal Ground Truth Functions
# ============================================================================

def ground_truth_rising_edge(a_seq: jnp.ndarray, b_seq: jnp.ndarray) -> jnp.ndarray:
    """
    Detect rising edges in sequence a.
    Output[i] = +1 if a[i-1] = -1 and a[i] = +1, else -1
    First position is -1 (no previous state).
    """
    # a_prev: shift right, first element is 0 (neutral)
    a_prev = jnp.concatenate([jnp.zeros_like(a_seq[:, :1]), a_seq[:, :-1]], axis=1)
    # Rising edge: prev was -1, current is +1
    rising = (a_prev == -1) & (a_seq == 1)
    return jnp.where(rising, 1.0, -1.0)


def ground_truth_falling_edge(a_seq: jnp.ndarray, b_seq: jnp.ndarray) -> jnp.ndarray:
    """
    Detect falling edges in sequence a.
    Output[i] = +1 if a[i-1] = +1 and a[i] = -1, else -1
    """
    a_prev = jnp.concatenate([jnp.zeros_like(a_seq[:, :1]), a_seq[:, :-1]], axis=1)
    falling = (a_prev == 1) & (a_seq == -1)
    return jnp.where(falling, 1.0, -1.0)


def ground_truth_stable_high(a_seq: jnp.ndarray, b_seq: jnp.ndarray) -> jnp.ndarray:
    """
    Detect stable high regions: a[i-1] = +1 and a[i] = +1
    """
    a_prev = jnp.concatenate([jnp.ones_like(a_seq[:, :1]), a_seq[:, :-1]], axis=1)
    stable = (a_prev == 1) & (a_seq == 1)
    return jnp.where(stable, 1.0, -1.0)


def ground_truth_stable_low(a_seq: jnp.ndarray, b_seq: jnp.ndarray) -> jnp.ndarray:
    """
    Detect stable low regions: a[i-1] = -1 and a[i] = -1
    """
    a_prev = jnp.concatenate([-jnp.ones_like(a_seq[:, :1]), a_seq[:, :-1]], axis=1)
    stable = (a_prev == -1) & (a_seq == -1)
    return jnp.where(stable, 1.0, -1.0)


def ground_truth_toggle(a_seq: jnp.ndarray, b_seq: jnp.ndarray) -> jnp.ndarray:
    """
    Detect state changes: a[i] != a[i-1]
    This is XOR of adjacent elements.
    """
    a_prev = jnp.concatenate([a_seq[:, :1], a_seq[:, :-1]], axis=1)
    # XOR in {-1, +1}: different = -1, same = +1
    # So toggle is when product is -1
    product = a_prev * a_seq
    return -product  # Invert: toggle=+1 when product=-1


def ground_truth_sync_rise(a_seq: jnp.ndarray, b_seq: jnp.ndarray) -> jnp.ndarray:
    """
    Both a and b have rising edge at same position.
    """
    a_rising = ground_truth_rising_edge(a_seq, b_seq)
    b_rising = ground_truth_rising_edge(b_seq, a_seq)
    # AND of two rising edges
    return jnp.sign(a_rising + b_rising - 0.5 * a_rising * b_rising + 0.5)


def ground_truth_sync_fall(a_seq: jnp.ndarray, b_seq: jnp.ndarray) -> jnp.ndarray:
    """
    Both a and b have falling edge at same position.
    """
    a_falling = ground_truth_falling_edge(a_seq, b_seq)
    b_falling = ground_truth_falling_edge(b_seq, a_seq)
    return jnp.sign(a_falling + b_falling - 0.5 * a_falling * b_falling + 0.5)


def ground_truth_a_before_b(a_seq: jnp.ndarray, b_seq: jnp.ndarray) -> jnp.ndarray:
    """
    At each position i, was there a rising edge in a before position i,
    and is there a rising edge in b at or after position i?

    Simplified: cumulative OR of a_rising up to i, AND b_rising at i.
    """
    a_rising = ground_truth_rising_edge(a_seq, b_seq) == 1
    b_rising = ground_truth_rising_edge(b_seq, a_seq) == 1

    # Cumulative OR of a_rising (any rising edge before)
    a_cumulative = jnp.cumsum(a_rising.astype(jnp.float32), axis=1) > 0
    a_before = jnp.concatenate([jnp.zeros_like(a_seq[:, :1], dtype=bool), a_cumulative[:, :-1]], axis=1)

    # a rose before this position AND b rises at this position
    result = a_before & b_rising
    return jnp.where(result, 1.0, -1.0)


def ground_truth_a_after_b(a_seq: jnp.ndarray, b_seq: jnp.ndarray) -> jnp.ndarray:
    """
    a rises after b has risen.
    """
    return ground_truth_a_before_b(b_seq, a_seq)


def ground_truth_concurrent(a_seq: jnp.ndarray, b_seq: jnp.ndarray) -> jnp.ndarray:
    """
    Both a and b change state at the same position.
    """
    a_toggle = ground_truth_toggle(a_seq, b_seq)
    b_toggle = ground_truth_toggle(b_seq, a_seq)
    # AND of toggles
    return jnp.sign(a_toggle + b_toggle - 0.5 * a_toggle * b_toggle + 0.5)


def ground_truth_majority(a_seq: jnp.ndarray, b_seq: jnp.ndarray) -> jnp.ndarray:
    """
    Output +1 at each position if majority of sequence so far is +1.
    """
    # Cumulative sum: positive means more +1s
    cum_sum = jnp.cumsum((a_seq + 1) / 2, axis=1)  # Count of +1s
    positions = jnp.arange(1, a_seq.shape[1] + 1)
    majority = cum_sum > (positions / 2)
    return jnp.where(majority, 1.0, -1.0)


def ground_truth_parity_seq(a_seq: jnp.ndarray, b_seq: jnp.ndarray) -> jnp.ndarray:
    """
    Cumulative parity: XOR of all elements up to position i.
    """
    # In {-1, +1}, parity is cumulative product
    cum_parity = jnp.cumprod(a_seq, axis=1)
    return cum_parity


def ground_truth_first_high(a_seq: jnp.ndarray, b_seq: jnp.ndarray) -> jnp.ndarray:
    """
    All outputs are +1 if first element is +1, else all -1.
    """
    first = a_seq[:, 0:1]
    return jnp.broadcast_to(first, a_seq.shape)


def ground_truth_last_high(a_seq: jnp.ndarray, b_seq: jnp.ndarray) -> jnp.ndarray:
    """
    All outputs are +1 if last element is +1, else all -1.
    """
    last = a_seq[:, -1:]
    return jnp.broadcast_to(last, a_seq.shape)


def ground_truth_alternating(a_seq: jnp.ndarray, b_seq: jnp.ndarray) -> jnp.ndarray:
    """
    Output +1 at position i if sequence alternates up to position i.
    """
    # Check if adjacent elements are different
    a_prev = jnp.concatenate([a_seq[:, :1], a_seq[:, :-1]], axis=1)
    is_alternating = a_prev * a_seq == -1  # Different = alternating

    # Cumulative AND (all positions up to i alternate)
    # Use product since alternating positions give -1, we want all -1s
    # Convert to {0, 1}: is_alternating = True -> 1
    cum_all = jnp.cumprod(is_alternating.astype(jnp.float32), axis=1)

    # First position is always "alternating" (vacuously true)
    cum_all = cum_all.at[:, 0].set(1.0)

    return jnp.where(cum_all > 0, 1.0, -1.0)


def ground_truth_monotonic_up(a_seq: jnp.ndarray, b_seq: jnp.ndarray) -> jnp.ndarray:
    """
    Output +1 if sequence has been monotonically non-decreasing.
    In {-1, +1}: no +1 followed by -1.
    """
    a_prev = jnp.concatenate([jnp.ones_like(a_seq[:, :1]) * -1, a_seq[:, :-1]], axis=1)

    # Violation: prev = +1 and current = -1
    violation = (a_prev == 1) & (a_seq == -1)

    # Cumulative OR of violations
    cum_violation = jnp.cumsum(violation.astype(jnp.float32), axis=1) > 0

    return jnp.where(cum_violation, -1.0, 1.0)


# Map operation ID to ground truth function
TEMPORAL_GROUND_TRUTH = {
    0: ground_truth_rising_edge,
    1: ground_truth_falling_edge,
    2: ground_truth_stable_high,
    3: ground_truth_stable_low,
    4: ground_truth_toggle,
    5: ground_truth_sync_rise,
    6: ground_truth_sync_fall,
    7: ground_truth_a_before_b,
    8: ground_truth_a_after_b,
    9: ground_truth_concurrent,
    10: ground_truth_majority,
    11: ground_truth_parity_seq,
    12: ground_truth_first_high,
    13: ground_truth_last_high,
    14: ground_truth_alternating,
    15: ground_truth_monotonic_up,
}


def generate_temporal_dataset(
    n_samples: int = 10000,
    seq_len: int = 8,
    seed: int = 42
) -> Dict[str, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]]:
    """
    Generate temporal dataset for all operations.

    Returns:
        Dictionary mapping operation name to (a_seq, b_seq, target, op_id)
    """
    a_seq, b_seq = generate_temporal_sequences(n_samples, seq_len, seed)

    dataset = {}
    for op_id, op_name in TEMPORAL_OP_NAMES.items():
        ground_truth_fn = TEMPORAL_GROUND_TRUTH[op_id]
        target = ground_truth_fn(a_seq, b_seq)
        dataset[op_name] = (a_seq, b_seq, target, op_id)

    return dataset


def create_temporal_train_test_split(
    n_train: int = 10000,
    n_test: int = 2000,
    seq_len: int = 8,
    seed: int = 42
) -> Tuple[Dict, Dict]:
    """
    Create train/test split for temporal dataset.
    """
    train_data = generate_temporal_dataset(n_train, seq_len, seed)
    test_data = generate_temporal_dataset(n_test, seq_len, seed + 10000)

    return train_data, test_data


def validate_temporal_operations(seq_len: int = 8, n_test: int = 100):
    """
    Validate that temporal ground truth functions are correct.
    """
    print("="*60)
    print("Temporal Operation Validation")
    print("="*60)

    # Generate small test set for manual inspection
    a_seq, b_seq = generate_temporal_sequences(n_test, seq_len, seed=42)

    print(f"\nSequence length: {seq_len}")
    print(f"Number of samples: {n_test}")

    # Show first sample for each operation
    print("\n--- Sample Sequences ---")
    print(f"a[0]: {a_seq[0].astype(int)}")
    print(f"b[0]: {b_seq[0].astype(int)}")

    print("\n--- Ground Truth Outputs (sample 0) ---")
    for op_id, op_name in TEMPORAL_OP_NAMES.items():
        fn = TEMPORAL_GROUND_TRUTH[op_id]
        target = fn(a_seq, b_seq)
        print(f"{op_id:2d}. {op_name:15s}: {target[0].astype(int)}")

    # Verify outputs are in {-1, +1}
    print("\n--- Output Range Check ---")
    all_valid = True
    for op_id, op_name in TEMPORAL_OP_NAMES.items():
        fn = TEMPORAL_GROUND_TRUTH[op_id]
        target = fn(a_seq, b_seq)
        valid = jnp.all((target == -1) | (target == 1))
        status = "OK" if valid else "FAIL"
        if not valid:
            all_valid = False
            unique = jnp.unique(target)
            print(f"  {op_name}: {status} (unique values: {unique})")

    if all_valid:
        print("  All operations output valid {-1, +1} values")

    print("\n" + "="*60)


if __name__ == "__main__":
    # Run validation
    validate_temporal_operations()

    # Show dataset sizes
    print("\nGenerating full dataset...")
    train, test = create_temporal_train_test_split(n_train=1000, n_test=200)

    print(f"\nTrain size: {len(train)} operations")
    print(f"Test size: {len(test)} operations")

    for op_name, (a, b, t, oid) in list(train.items())[:3]:
        print(f"  {op_name}: a={a.shape}, b={b.shape}, target={t.shape}")
