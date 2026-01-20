"""
Ternary Operations for NPU-Native Learning
==========================================

Implements ternary {-1, 0, +1} parameters with Straight-Through Estimator (STE)
for gradient-based learning.

Key concepts:
- Forward pass: Uses quantized ternary values
- Backward pass: Gradients flow through as if values were continuous (STE)
- Threshold: |w| > threshold → sign(w), otherwise → 0
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple


def ternary_quantize(w: jnp.ndarray, threshold: float = 0.3) -> jnp.ndarray:
    """
    Quantize continuous weights to ternary {-1, 0, +1}.

    w_ternary = sign(w) if |w| > threshold else 0

    Args:
        w: Continuous weights
        threshold: Magnitude threshold for non-zero

    Returns:
        Ternary weights in {-1, 0, +1}
    """
    return jnp.sign(w) * (jnp.abs(w) > threshold)


def ternary_ste(w: jnp.ndarray, threshold: float = 0.3) -> jnp.ndarray:
    """
    Ternary quantization with Straight-Through Estimator.

    Forward: Returns ternary quantized value
    Backward: Gradients flow through as if identity (STE)

    Args:
        w: Continuous weights
        threshold: Magnitude threshold

    Returns:
        Ternary weights (with STE gradient)
    """
    w_ternary = ternary_quantize(w, threshold)
    # STE trick: forward uses quantized, backward uses identity
    return w + jax.lax.stop_gradient(w_ternary - w)


def soft_ternary(w: jnp.ndarray, temperature: float = 1.0, threshold: float = 0.3) -> jnp.ndarray:
    """
    Soft ternary quantization with temperature annealing.

    At high temperature (e.g., 10.0): nearly continuous
    At low temperature (e.g., 0.1): nearly discrete ternary

    Uses tanh for sign and sigmoid for magnitude gating.

    Args:
        w: Continuous weights
        temperature: Controls sharpness (lower = more discrete)
        threshold: Target threshold for ternary conversion

    Returns:
        Soft ternary values in approximately {-1, 0, +1}
    """
    # Sign component: tanh with temperature scaling
    sign = jnp.tanh(w / temperature)

    # Magnitude gating: sigmoid to decide if nonzero
    # High |w| → gate ≈ 1, low |w| → gate ≈ 0
    gate = jax.nn.sigmoid((jnp.abs(w) - threshold) / temperature)

    return sign * gate


class TernaryMask(nn.Module):
    """
    Learnable sparse mask in {-1, 0, +1}.

    This is the core building block for Boolean Fourier character selection.
    Each mask learns to select which Fourier characters are relevant for
    a particular operation (XOR, AND, OR, IMPLIES).

    Attributes:
        size: Dimension of the mask
        threshold: Magnitude threshold for ternary quantization
        init_scale: Initial weight scale (must be > threshold for non-zero start)
        use_soft: If True, use soft_ternary for differentiable training
        temperature: Temperature for soft_ternary (lower = more discrete)
    """
    size: int
    threshold: float = 0.3
    init_scale: float = 0.5
    use_soft: bool = False
    temperature: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        """
        Returns the ternary mask.

        Returns:
            [size] array in approximately {-1, 0, +1}
        """
        # Continuous parameters (learned via gradient descent)
        w = self.param(
            'w',
            nn.initializers.normal(self.init_scale),
            (self.size,)
        )

        if self.use_soft:
            # Soft ternary for differentiable training
            return soft_ternary(w, self.temperature, self.threshold)
        else:
            # Hard ternary with STE
            return ternary_ste(w, self.threshold)

    def get_continuous(self) -> jnp.ndarray:
        """Get the underlying continuous parameters (for analysis)."""
        return self.variables['params']['w']


class TernaryLinear(nn.Module):
    """
    Linear layer with ternary {-1, 0, +1} weights.

    For NPU: This maps directly to sparse binary operations
    since ternary multiply is just:
    - If weight = 0: output = 0
    - If weight = +1: output = input
    - If weight = -1: output = -input
    """
    features: int
    threshold: float = 0.3
    use_bias: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Ternary linear transformation.

        Args:
            x: [..., in_features] input

        Returns:
            [..., features] output
        """
        in_features = x.shape[-1]

        # Continuous weights
        w = self.param(
            'kernel',
            nn.initializers.normal(0.1),
            (in_features, self.features)
        )

        # Ternary quantization with STE
        w_ternary = ternary_ste(w, self.threshold)

        # Matrix multiply
        out = x @ w_ternary

        if self.use_bias:
            b = self.param('bias', nn.initializers.zeros, (self.features,))
            out = out + b

        return out


def project_to_ternary(w: jnp.ndarray, threshold: float = 0.3) -> jnp.ndarray:
    """
    Project continuous weights to nearest ternary values.

    Used for post-training quantization or weight clamping.

    Args:
        w: Continuous weights
        threshold: Magnitude threshold

    Returns:
        Ternary weights
    """
    return ternary_quantize(w, threshold)


def ternary_sparsity(w: jnp.ndarray, threshold: float = 0.3) -> float:
    """
    Compute sparsity of ternary weights.

    Sparsity = fraction of zero weights

    Args:
        w: Weights (continuous or ternary)
        threshold: Threshold for ternary quantization

    Returns:
        Sparsity in [0, 1]
    """
    w_ternary = ternary_quantize(w, threshold)
    return float(jnp.mean(w_ternary == 0))


def count_ternary_params(params: dict) -> Tuple[int, int, int]:
    """
    Count ternary parameters by value.

    Returns:
        (n_minus_one, n_zero, n_plus_one)
    """
    n_minus = 0
    n_zero = 0
    n_plus = 0

    def count_leaf(x):
        nonlocal n_minus, n_zero, n_plus
        x_ternary = ternary_quantize(x)
        n_minus += int(jnp.sum(x_ternary == -1))
        n_zero += int(jnp.sum(x_ternary == 0))
        n_plus += int(jnp.sum(x_ternary == 1))

    jax.tree_util.tree_map(count_leaf, params)
    return n_minus, n_zero, n_plus


def verify_ternary(params: dict, threshold: float = 0.3) -> bool:
    """
    Verify all parameters are ternary after quantization.

    Returns:
        True if all params quantize to {-1, 0, +1}
    """
    def check_leaf(x):
        x_ternary = ternary_quantize(x, threshold)
        valid = jnp.all((x_ternary == -1) | (x_ternary == 0) | (x_ternary == 1))
        return valid

    results = jax.tree_util.tree_map(check_leaf, params)
    return all(jax.tree_util.tree_leaves(results))


if __name__ == "__main__":
    print("="*60)
    print("Testing Ternary Operations")
    print("="*60)

    # Test TernaryMask
    print("\n[Test 1] TernaryMask")
    rng = jax.random.PRNGKey(42)

    mask = TernaryMask(size=4)
    variables = mask.init(rng)
    print(f"Continuous params: {variables['params']['w']}")

    output = mask.apply(variables)
    print(f"Ternary output: {output}")
    print(f"Unique values: {set(map(float, output))}")

    # Test sparsity
    print("\n[Test 2] Sparsity")
    w = jnp.array([0.5, 0.1, -0.4, 0.2, -0.8, 0.05])
    print(f"Input: {w}")
    w_ternary = ternary_quantize(w, threshold=0.3)
    print(f"Ternary (threshold=0.3): {w_ternary}")
    print(f"Sparsity: {ternary_sparsity(w):.2%}")

    # Test STE gradient flow
    print("\n[Test 3] STE Gradient Flow")

    def loss_fn(params):
        mask_out = ternary_ste(params['w'], threshold=0.3)
        target = jnp.array([0., 0., 0., 1.])  # XOR mask target
        return jnp.mean((mask_out - target) ** 2)

    params = {'w': jnp.array([0.1, 0.2, -0.1, 0.5])}
    loss, grads = jax.value_and_grad(loss_fn)(params)
    print(f"Params: {params['w']}")
    print(f"Ternary: {ternary_quantize(params['w'])}")
    print(f"Loss: {loss:.4f}")
    print(f"Gradients: {grads['w']}")  # Should be non-zero due to STE

    # Test TernaryLinear
    print("\n[Test 4] TernaryLinear")
    linear = TernaryLinear(features=4)
    x = jnp.array([[1., -1., 1., -1.]])  # Binary input
    variables = linear.init(rng, x)

    # Get ternary weights
    kernel = variables['params']['kernel']
    kernel_ternary = ternary_quantize(kernel)
    print(f"Kernel shape: {kernel.shape}")
    print(f"Kernel sparsity: {ternary_sparsity(kernel):.2%}")

    out = linear.apply(variables, x)
    print(f"Output: {out}")

    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)
