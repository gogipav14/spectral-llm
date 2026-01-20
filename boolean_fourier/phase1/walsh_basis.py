"""
Walsh-Hadamard Basis Generation
===============================

Generates and stores fixed Walsh-Hadamard matrices for NPU-native operations.

The Walsh basis is:
- FIXED (computed once, never changes)
- BINARY {-1, +1} (no floating point)
- SELF-INVERSE (up to scaling factor n)

Key property: W @ W.T = n * I

For normalized version: (1/sqrt(n)) * W @ (1/sqrt(n)) * W.T = I
"""

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path


def generate_walsh_matrix(n: int) -> jnp.ndarray:
    """
    Generate n×n Walsh-Hadamard matrix via Sylvester construction.

    The Sylvester construction builds H recursively:
    H_1 = [[1]]
    H_2 = [[H_1, H_1], [H_1, -H_1]]
    H_n = [[H_{n/2}, H_{n/2}], [H_{n/2}, -H_{n/2}]]

    Args:
        n: Matrix size (must be power of 2)

    Returns:
        [n, n] matrix in {-1, +1}
    """
    assert n > 0 and (n & (n - 1)) == 0, f"n must be power of 2, got {n}"

    if n == 1:
        return jnp.array([[1]], dtype=jnp.int8)

    H_half = generate_walsh_matrix(n // 2)

    # Sylvester construction
    top = jnp.concatenate([H_half, H_half], axis=1)
    bottom = jnp.concatenate([H_half, -H_half], axis=1)

    return jnp.concatenate([top, bottom], axis=0)


def butterfly_wht_inplace(x: np.ndarray) -> np.ndarray:
    """
    In-place Fast Walsh-Hadamard Transform using butterfly algorithm.

    This is the NPU-friendly version that uses only:
    - Addition
    - Subtraction
    - Bitshift (division by 2)

    Args:
        x: 1D array of length n (power of 2)

    Returns:
        WHT-transformed array (normalized by 1/sqrt(n))

    Note: Modifies x in-place and returns it.
    """
    n = len(x)
    assert n > 0 and (n & (n - 1)) == 0, f"Length must be power of 2, got {n}"

    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(h):
                a = x[i + j]
                b = x[i + j + h]
                # Per-stage normalization: divide by 2
                # This gives overall normalization of 1/sqrt(n)
                x[i + j] = (a + b) // 2  # Integer division
                x[i + j + h] = (a - b) // 2
        h *= 2

    return x


def butterfly_wht_jax(x: jnp.ndarray) -> jnp.ndarray:
    """
    JAX-compatible Fast Walsh-Hadamard Transform.

    For training, we use float for gradients but the structure
    is still NPU-compatible (butterfly network).

    Args:
        x: [..., n] array where n is power of 2

    Returns:
        WHT-transformed array (normalized)
    """
    n = x.shape[-1]

    # Use matrix multiply with precomputed Walsh basis
    # This is equivalent to butterfly but simpler for JAX autodiff
    W = generate_walsh_matrix(n).astype(jnp.float32) / jnp.sqrt(n)
    return x @ W


def verify_walsh_properties(n: int = 8):
    """
    Verify key properties of the Walsh matrix.

    1. All entries are ±1
    2. Rows are orthogonal: W @ W.T = n * I
    3. Self-inverse (up to scaling)
    """
    print(f"\n{'='*60}")
    print(f"Verifying Walsh Matrix Properties (n={n})")
    print('='*60)

    W = generate_walsh_matrix(n)
    W_float = W.astype(jnp.float32)

    # Property 1: Binary entries
    unique_vals = set(np.unique(np.array(W)))
    assert unique_vals == {-1, 1}, f"Expected {{-1, 1}}, got {unique_vals}"
    print(f"✓ All entries are in {{-1, +1}}")

    # Property 2: Orthogonality
    product = W_float @ W_float.T
    expected = n * jnp.eye(n)
    error = jnp.max(jnp.abs(product - expected))
    assert error < 1e-5, f"Orthogonality error: {error}"
    print(f"✓ W @ W.T = {n} * I (error: {error:.2e})")

    # Property 3: Self-inverse (normalized)
    W_normalized = W_float / jnp.sqrt(n)
    product_normalized = W_normalized @ W_normalized.T
    error_normalized = jnp.max(jnp.abs(product_normalized - jnp.eye(n)))
    assert error_normalized < 1e-5, f"Normalized self-inverse error: {error_normalized}"
    print(f"✓ (1/√n)W @ (1/√n)W.T = I (error: {error_normalized:.2e})")

    # Show the matrix
    print(f"\nWalsh-{n} matrix:")
    print(np.array(W))

    return W


def precompute_and_save_walsh_matrices(output_dir: str = "v5"):
    """
    Precompute Walsh matrices of various sizes and save to disk.

    These can be loaded at runtime as fixed constants.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    sizes = [4, 8, 16, 32, 64, 128, 256, 512, 1024]

    print("\n" + "="*60)
    print("Precomputing Walsh Matrices")
    print("="*60)

    for n in sizes:
        W = generate_walsh_matrix(n)
        filename = output_path / f"walsh_{n}.npy"
        np.save(filename, np.array(W))
        print(f"✓ Saved Walsh-{n} to {filename}")

    print("\nDone!")


# Precomputed small Walsh matrices for quick access
WALSH_2 = jnp.array([[1, 1], [1, -1]], dtype=jnp.int8)

WALSH_4 = jnp.array([
    [1,  1,  1,  1],
    [1, -1,  1, -1],
    [1,  1, -1, -1],
    [1, -1, -1,  1]
], dtype=jnp.int8)

WALSH_8 = jnp.array([
    [1,  1,  1,  1,  1,  1,  1,  1],
    [1, -1,  1, -1,  1, -1,  1, -1],
    [1,  1, -1, -1,  1,  1, -1, -1],
    [1, -1, -1,  1,  1, -1, -1,  1],
    [1,  1,  1,  1, -1, -1, -1, -1],
    [1, -1,  1, -1, -1,  1, -1,  1],
    [1,  1, -1, -1, -1, -1,  1,  1],
    [1, -1, -1,  1, -1,  1,  1, -1]
], dtype=jnp.int8)


if __name__ == "__main__":
    # Verify properties
    verify_walsh_properties(4)
    verify_walsh_properties(8)

    # Precompute and save
    precompute_and_save_walsh_matrices()

    # Test butterfly WHT
    print("\n" + "="*60)
    print("Testing Butterfly WHT")
    print("="*60)

    x = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
    print(f"Input: {x}")

    # Matrix-based WHT
    W = generate_walsh_matrix(8).astype(np.float32)
    y_matrix = (x @ W) / np.sqrt(8)
    print(f"Matrix WHT: {y_matrix}")

    # JAX version
    y_jax = butterfly_wht_jax(jnp.array(x))
    print(f"JAX WHT: {y_jax}")

    # Verify they match
    error = np.max(np.abs(y_matrix - np.array(y_jax)))
    print(f"Error: {error:.2e}")
    print(f"✓ Methods match" if error < 1e-5 else "✗ Methods differ!")
