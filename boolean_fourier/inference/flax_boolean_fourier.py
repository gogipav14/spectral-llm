"""
Boolean Fourier Model in Flax
=============================

Flax-based implementation for better export compatibility.
"""

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np


class BooleanFourierPhase1(nnx.Module):
    """
    Boolean Fourier inference for Phase 1 operations.

    Computes all 4 operations (XOR, AND, OR, IMPLIES) in parallel.
    Uses fixed ternary masks stored as model parameters.
    """

    def __init__(self, rngs: nnx.Rngs):
        # Ternary mask ROM - fixed weights, not trainable
        # Shape: [4, 4] - 4 operations, 4 basis coefficients each
        # Derived from Boolean Fourier analysis (see derivation below)
        masks = jnp.array([
            [0, 0, 0, 1],      # XOR: pure parity (ab)
            [-1, 1, 1, 1],     # AND: -1 + a + b + ab (scaled)
            [1, 1, 1, -1],     # OR: 1 + a + b - ab (scaled)
            [1, -1, 1, 1],     # IMPLIES: 1 - a + b + ab (scaled)
        ], dtype=jnp.float32)

        # Store as non-trainable parameter
        self.masks = nnx.Variable(masks)

    def __call__(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass.

        Args:
            a: [batch, n_bits] in {-1, +1}
            b: [batch, n_bits] in {-1, +1}

        Returns:
            [batch, n_bits, 4] - output for all 4 operations
        """
        # Compute Boolean Fourier basis: [1, a, b, ab]
        ones = jnp.ones_like(a)
        ab = a * b
        basis = jnp.stack([ones, a, b, ab], axis=-1)  # [batch, n_bits, 4]

        # Apply all masks via einsum
        # basis: [batch, n_bits, 4], masks: [4, 4]
        # output: [batch, n_bits, 4]
        outputs = jnp.einsum('bni,oi->bno', basis, self.masks.value)

        # Sign with tie-breaker (0 -> +1)
        outputs = jnp.sign(outputs)
        outputs = jnp.where(outputs == 0, 1.0, outputs)

        return outputs


class BooleanFourierPhase3(nnx.Module):
    """
    Boolean Fourier inference for Phase 3 (3-variable) operations.

    10 operations over 8-dim basis.
    """

    def __init__(self, rngs: nnx.Rngs):
        # Phase 3 masks: [10, 8]
        masks = jnp.array([
            [-1, 0, 0, 0, 0, 0, 0, 1],      # parity_3
            [-1, 0, 1, 1, 0, 0, 0, -1],     # majority_3
            [-1, 0, 0, 1, 0, 1, 1, 1],      # and_3
            [-1, 1, 1, 1, -1, -1, -1, 1],   # or_3
            [-1, 0, 0, 0, 0, 0, 0, 1],      # xor_ab_xor_c
            [-1, 0, 0, 1, -1, 1, 1, 0],     # and_ab_or_c
            [-1, 0, 1, 1, 1, 0, -1, -1],    # or_ab_and_c
            [-1, 0, -1, 1, 0, 1, 0, 1],     # implies_ab_c
            [-1, -1, 0, 1, 0, 1, 1, -1],    # xor_and_ab_c
            [-1, 0, 0, 1, 1, 0, 0, -1],     # and_xor_ab_c
        ], dtype=jnp.float32)

        self.masks = nnx.Variable(masks)

    def __call__(self, a: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass for 3-variable operations.

        Args:
            a, b, c: [batch, n_bits] in {-1, +1}

        Returns:
            [batch, n_bits, 10] - output for all 10 operations
        """
        # 8-dim Boolean Fourier basis: [1, a, b, c, ab, ac, bc, abc]
        ones = jnp.ones_like(a)
        ab, ac, bc = a * b, a * c, b * c
        abc = a * b * c
        basis = jnp.stack([ones, a, b, c, ab, ac, bc, abc], axis=-1)  # [batch, n_bits, 8]

        # Apply masks
        outputs = jnp.einsum('bni,oi->bno', basis, self.masks.value)

        # Sign with tie-breaker
        outputs = jnp.sign(outputs)
        outputs = jnp.where(outputs == 0, 1.0, outputs)

        return outputs


def export_to_onnx(batch_size: int = 1000, n_bits: int = 64):
    """Export Flax model to ONNX."""
    from jax2onnx import to_onnx
    import onnx

    # Create model
    model = BooleanFourierPhase1(rngs=nnx.Rngs(0))

    # Create example inputs
    example_a = jnp.ones((batch_size, n_bits), dtype=jnp.float32)
    example_b = jnp.ones((batch_size, n_bits), dtype=jnp.float32)

    # Export
    onnx_model = to_onnx(
        model,
        [example_a, example_b],
        model_name='BooleanFourierPhase1',
        return_mode='proto'
    )

    path = f'phase1_flax_b{batch_size}_n{n_bits}.onnx'
    onnx.save(onnx_model, path)
    print(f'Exported: {path}')

    return path


def test_model():
    """Test the Flax model."""
    print("Testing BooleanFourierPhase1...")

    model = BooleanFourierPhase1(rngs=nnx.Rngs(0))

    # Test data
    rng = np.random.default_rng(42)
    batch_size, n_bits = 10, 8
    a = jnp.array((2 * rng.integers(0, 2, (batch_size, n_bits)) - 1), dtype=jnp.float32)
    b = jnp.array((2 * rng.integers(0, 2, (batch_size, n_bits)) - 1), dtype=jnp.float32)

    # Forward pass
    outputs = model(a, b)
    print(f"Input shapes: a={a.shape}, b={b.shape}")
    print(f"Output shape: {outputs.shape}")

    # Verify XOR (op 0)
    xor_expected = a * b
    xor_actual = outputs[:, :, 0]
    xor_match = jnp.allclose(xor_expected, xor_actual)
    print(f"XOR correct: {xor_match}")

    # Verify AND (op 1)
    and_expected = jnp.where((a == 1) & (b == 1), 1.0, -1.0)
    and_actual = outputs[:, :, 1]
    and_match = jnp.allclose(and_expected, and_actual)
    print(f"AND correct: {and_match}")

    # Verify OR (op 2)
    or_expected = jnp.where((a == 1) | (b == 1), 1.0, -1.0)
    or_actual = outputs[:, :, 2]
    or_match = jnp.allclose(or_expected, or_actual)
    print(f"OR correct: {or_match}")

    # Verify IMPLIES (op 3): a → b = ¬a ∨ b
    implies_expected = jnp.where((a == -1) | (b == 1), 1.0, -1.0)
    implies_actual = outputs[:, :, 3]
    implies_match = jnp.allclose(implies_expected, implies_actual)
    print(f"IMPLIES correct: {implies_match}")

    all_correct = xor_match and and_match and or_match and implies_match
    print(f"\nAll operations correct: {all_correct}")

    return all_correct


def benchmark_flax(batch_size: int = 10000, n_bits: int = 64, n_runs: int = 100):
    """Benchmark Flax model on JAX."""
    import time

    model = BooleanFourierPhase1(rngs=nnx.Rngs(0))

    # JIT compile
    @jax.jit
    def forward(a, b):
        return model(a, b)

    # Generate data
    rng = np.random.default_rng(42)
    a = jnp.array((2 * rng.integers(0, 2, (batch_size, n_bits)) - 1), dtype=jnp.float32)
    b = jnp.array((2 * rng.integers(0, 2, (batch_size, n_bits)) - 1), dtype=jnp.float32)

    # Warmup
    for _ in range(10):
        _ = forward(a, b).block_until_ready()

    # Benchmark
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = forward(a, b).block_until_ready()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    avg_time = np.mean(times)
    total_ops = batch_size * n_bits * 4
    throughput = (total_ops / avg_time) / 1000

    print(f"\nFlax Benchmark (batch={batch_size}, bits={n_bits}):")
    print(f"  Avg time: {avg_time:.3f} ms")
    print(f"  Throughput: {throughput:.2f} MOps/s")

    return throughput


if __name__ == "__main__":
    # Test correctness
    test_model()

    # Benchmark
    print("\n" + "=" * 50)
    benchmark_flax(batch_size=1000, n_bits=64)
    benchmark_flax(batch_size=10000, n_bits=64)
    benchmark_flax(batch_size=100000, n_bits=64)

    # Export to ONNX
    print("\n" + "=" * 50)
    print("Exporting to ONNX...")
    for batch_size in [100, 500, 1000, 5000, 10000]:
        try:
            export_to_onnx(batch_size=batch_size, n_bits=64)
        except Exception as e:
            print(f"  Failed for batch={batch_size}: {e}")
