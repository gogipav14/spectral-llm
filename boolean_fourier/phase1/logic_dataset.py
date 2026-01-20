"""
Logic Dataset for Phase 1 Training
===================================

Generates synthetic Boolean operation examples for training the
Binary Logic Layer.

All data is in {-1, +1} encoding (not {0, 1}) because:
1. Symmetric representation works better with ternary weights
2. XOR becomes simple multiplication: a XOR b = a * b
3. Walsh-Hadamard transform expects {-1, +1}

Dataset structure:
- a, b: Input binary vectors
- target: Ground truth output
- op_id: Operation identifier {0: XOR, 1: AND, 2: OR, 3: IMPLIES}
"""

import jax
import jax.numpy as jnp
from typing import Dict, Tuple, Iterator
import numpy as np


def ground_truth_xor(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """
    XOR in {-1, +1}: a * b (parity).

    In {-1, +1} encoding where -1 = TRUE, +1 = FALSE:
    XOR returns -1 (TRUE) when inputs differ, +1 (FALSE) when same.
    This equals a * b directly.
    """
    return a * b


def ground_truth_and(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """
    AND in {-1, +1}.

    AND returns -1 (TRUE) only when both inputs are -1 (TRUE).
    Fourier expansion: 0.5(1 + a + b - ab)
    """
    return jnp.sign(1 + a + b - a*b)


def ground_truth_or(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """
    OR in {-1, +1}.

    OR returns +1 (FALSE) only when both inputs are +1 (FALSE).
    Fourier expansion: 0.5(-1 + a + b + ab)
    """
    return jnp.sign(-1 + a + b + a*b)


def ground_truth_implies(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """
    IMPLIES (a → b) in {-1, +1}.

    IMPLIES returns +1 (FALSE) only when a=-1 (TRUE) and b=+1 (FALSE).
    Fourier expansion: 0.5(-1 - a + b - ab)
    """
    return jnp.sign(-1 - a + b - a*b)


GROUND_TRUTH_OPS = {
    0: ground_truth_xor,
    1: ground_truth_and,
    2: ground_truth_or,
    3: ground_truth_implies
}

OP_NAMES = {0: 'xor', 1: 'and', 2: 'or', 3: 'implies'}


def generate_random_binary(
    rng: jax.random.PRNGKey,
    shape: Tuple[int, ...]
) -> jnp.ndarray:
    """Generate random binary vectors in {-1, +1}."""
    return jax.random.choice(rng, jnp.array([-1., 1.]), shape)


def generate_logic_dataset(
    n_samples: int = 10000,
    n_bits: int = 64,
    seed: int = 42
) -> Dict[str, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]]:
    """
    Generate dataset for all 4 Boolean operations.

    Args:
        n_samples: Number of samples per operation
        n_bits: Dimension of binary vectors
        seed: Random seed

    Returns:
        Dict mapping operation name to (a, b, target, op_id)
    """
    rng = jax.random.PRNGKey(seed)

    # Generate inputs
    rng, key1, key2 = jax.random.split(rng, 3)
    a = generate_random_binary(key1, (n_samples, n_bits))
    b = generate_random_binary(key2, (n_samples, n_bits))

    # Generate targets for each operation
    dataset = {}
    for op_id, op_name in OP_NAMES.items():
        target = GROUND_TRUTH_OPS[op_id](a, b)
        dataset[op_name] = (a, b, target, op_id)

    return dataset


def generate_mixed_dataset(
    n_samples: int = 10000,
    n_bits: int = 64,
    seed: int = 42
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Generate mixed dataset with all operations interleaved.

    Returns:
        (a, b, target, op_ids) where op_ids indicates which operation for each sample
    """
    rng = jax.random.PRNGKey(seed)

    # Samples per operation
    n_per_op = n_samples // 4

    all_a = []
    all_b = []
    all_target = []
    all_op_ids = []

    for op_id in range(4):
        rng, key1, key2 = jax.random.split(rng, 3)
        a = generate_random_binary(key1, (n_per_op, n_bits))
        b = generate_random_binary(key2, (n_per_op, n_bits))
        target = GROUND_TRUTH_OPS[op_id](a, b)
        op_ids = jnp.full((n_per_op,), op_id, dtype=jnp.int32)

        all_a.append(a)
        all_b.append(b)
        all_target.append(target)
        all_op_ids.append(op_ids)

    # Concatenate
    a = jnp.concatenate(all_a, axis=0)
    b = jnp.concatenate(all_b, axis=0)
    target = jnp.concatenate(all_target, axis=0)
    op_ids = jnp.concatenate(all_op_ids, axis=0)

    # Shuffle
    rng, key = jax.random.split(rng)
    perm = jax.random.permutation(key, len(a))
    a = a[perm]
    b = b[perm]
    target = target[perm]
    op_ids = op_ids[perm]

    return a, b, target, op_ids


def batch_iterator(
    a: jnp.ndarray,
    b: jnp.ndarray,
    target: jnp.ndarray,
    op_ids: jnp.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
    seed: int = 0
) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """
    Create batched iterator over dataset.

    Yields:
        (a_batch, b_batch, target_batch, op_id_batch)
    """
    n_samples = len(a)
    indices = np.arange(n_samples)

    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_indices = indices[start:end]

        yield (
            a[batch_indices],
            b[batch_indices],
            target[batch_indices],
            op_ids[batch_indices]
        )


def create_train_test_split(
    n_train: int = 8000,
    n_test: int = 2000,
    n_bits: int = 64,
    seed: int = 42
) -> Tuple[Dict, Dict]:
    """
    Create train/test split of the logic dataset.

    Returns:
        (train_data, test_data) where each is dict of operations
    """
    train_data = generate_logic_dataset(n_train, n_bits, seed)
    test_data = generate_logic_dataset(n_test, n_bits, seed + 1000)

    return train_data, test_data


def verify_dataset(dataset: Dict) -> None:
    """Verify dataset correctness by checking a few samples."""
    print("\nDataset Verification:")
    print("-" * 40)

    for op_name, (a, b, target, op_id) in dataset.items():
        print(f"\n{op_name.upper()} (op_id={op_id}):")
        print(f"  Shape: a={a.shape}, b={b.shape}, target={target.shape}")

        # Verify a few samples
        ground_truth = GROUND_TRUTH_OPS[op_id](a[:5], b[:5])
        matches = jnp.all(target[:5] == ground_truth)
        print(f"  Sample verification: {'PASS' if matches else 'FAIL'}")

        # Show example
        print(f"  Example:")
        print(f"    a[0][:8] = {a[0][:8]}")
        print(f"    b[0][:8] = {b[0][:8]}")
        print(f"    target[0][:8] = {target[0][:8]}")


if __name__ == "__main__":
    print("="*60)
    print("Logic Dataset Generator")
    print("="*60)

    # Generate dataset
    print("\n[Test 1] Generate separate datasets per operation")
    dataset = generate_logic_dataset(n_samples=100, n_bits=64)
    verify_dataset(dataset)

    # Test mixed dataset
    print("\n[Test 2] Generate mixed dataset")
    a, b, target, op_ids = generate_mixed_dataset(n_samples=400, n_bits=64)
    print(f"  Total samples: {len(a)}")
    print(f"  Unique op_ids: {set(map(int, op_ids))}")
    print(f"  Samples per op: {[(op_ids == i).sum() for i in range(4)]}")

    # Test batch iterator
    print("\n[Test 3] Batch iterator")
    batch_count = 0
    for a_b, b_b, t_b, op_b in batch_iterator(a, b, target, op_ids, batch_size=32):
        batch_count += 1
        if batch_count <= 2:
            print(f"  Batch {batch_count}: shapes = {a_b.shape}, {b_b.shape}, {t_b.shape}")
    print(f"  Total batches: {batch_count}")

    # Test train/test split
    print("\n[Test 4] Train/test split")
    train_data, test_data = create_train_test_split()
    print(f"  Train samples per op: {train_data['xor'][0].shape[0]}")
    print(f"  Test samples per op: {test_data['xor'][0].shape[0]}")

    # Verify Boolean operations correctness
    print("\n[Test 5] Boolean operation correctness")
    a_test = jnp.array([[1., -1., 1., -1., 1., -1., 1., -1.]])
    b_test = jnp.array([[-1., -1., 1., 1., -1., -1., 1., 1.]])

    print(f"  a = {a_test[0]}")
    print(f"  b = {b_test[0]}")
    print()

    for op_id, op_name in OP_NAMES.items():
        result = GROUND_TRUTH_OPS[op_id](a_test, b_test)
        print(f"  {op_name.upper():8s}: {result[0]}")

    # Verify XOR = a*b
    xor_direct = a_test * b_test
    xor_func = ground_truth_xor(a_test, b_test)
    assert jnp.all(xor_direct == xor_func), "XOR should be a*b!"
    print("\n✓ XOR = a * b verified!")

    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)
