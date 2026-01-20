"""
Binary Logic Layer with Boolean Fourier Basis
==============================================

This is the Phase 1 minimal architecture that proves the spectral claim:
- XOR should concentrate ALL weight on the parity character (a*b)
- Each operation has a distinct spectral signature
- Pure binary/ternary, no hidden layers needed

Boolean Fourier Basis:
For each bit position, we construct: φ(a, b) = [1, a, b, a*b]
This is the complete Walsh-Fourier basis for 2-variable Boolean functions.

In this basis:
- XOR(a,b) = a*b (parity character only)
- AND(a,b) = (1 + a + b + a*b) / 4
- OR(a,b) = (1 + a + b - a*b) / 4
- IMPLIES(a,b) = (1 - a + b + a*b) / 4

The ternary masks learn to select the appropriate Fourier characters.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple, Dict

from ternary_ops import TernaryMask, ternary_ste


def boolean_fourier_features(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """
    Construct Boolean Fourier features for each bit.

    For each bit position i:
    φ(a_i, b_i) = [1, a_i, b_i, a_i * b_i]

    This is the 4-character Walsh-Fourier basis of f(a,b).

    Args:
        a: [batch, n_bits] in {-1, +1}
        b: [batch, n_bits] in {-1, +1}

    Returns:
        [batch, n_bits, 4] features
    """
    ones = jnp.ones_like(a)
    ab = a * b  # Parity / XOR character

    # Stack along last dimension: [1, a, b, ab]
    return jnp.stack([ones, a, b, ab], axis=-1)


class BinaryLogicLayer(nn.Module):
    """
    Pure NPU-native logic layer with Boolean Fourier basis.

    This is the MINIMAL architecture for proving spectral spikes:
    - 4 ternary masks (one per operation)
    - Each mask is 4-dimensional (over Boolean Fourier basis)
    - Total: 16 learnable parameters

    Operations:
    - op=0: XOR  → expected mask: [0, 0, 0, 1]
    - op=1: AND  → expected mask: [1, 1, 1, 1]
    - op=2: OR   → expected mask: [1, 1, 1, -1]
    - op=3: IMPLIES → expected mask: [1, -1, 1, 1]
    """
    n_bits: int = 64
    threshold: float = 0.3

    def setup(self):
        # 4 ternary masks over 4-dim Boolean Fourier basis [1, a, b, ab]
        self.xor_mask = TernaryMask(size=4, threshold=self.threshold)
        self.and_mask = TernaryMask(size=4, threshold=self.threshold)
        self.or_mask = TernaryMask(size=4, threshold=self.threshold)
        self.implies_mask = TernaryMask(size=4, threshold=self.threshold)

    def get_mask(self, operation_id: int) -> jnp.ndarray:
        """Get the mask for a specific operation."""
        masks = [
            self.xor_mask(),
            self.and_mask(),
            self.or_mask(),
            self.implies_mask()
        ]
        return masks[operation_id]

    def get_all_masks(self) -> jnp.ndarray:
        """Get all masks as [4, 4] array."""
        return jnp.stack([
            self.xor_mask(),
            self.and_mask(),
            self.or_mask(),
            self.implies_mask()
        ], axis=0)

    def __call__(
        self,
        a: jnp.ndarray,
        b: jnp.ndarray,
        operation_id: int
    ) -> jnp.ndarray:
        """
        Apply Boolean operation using learned spectral mask.

        Args:
            a: [batch, n_bits] in {-1, +1}
            b: [batch, n_bits] in {-1, +1}
            operation_id: int in {0, 1, 2, 3} for {XOR, AND, OR, IMPLIES}

        Returns:
            [batch, n_bits] in {-1, +1}
        """
        # 1. Construct Boolean Fourier features per bit
        # φ(a_i, b_i) = [1, a_i, b_i, a_i*b_i]
        features = boolean_fourier_features(a, b)  # [batch, n_bits, 4]

        # 2. Get operation-specific ternary mask
        mask = self.get_mask(operation_id)  # [4] in {-1, 0, +1}

        # 3. Apply mask (element-wise multiply)
        masked = features * mask  # [batch, n_bits, 4]

        # 4. Sum over Fourier characters
        output = jnp.sum(masked, axis=-1)  # [batch, n_bits]

        # 5. Threshold to binary
        output = jnp.sign(output)
        # Handle zeros (shouldn't happen for valid Boolean ops)
        output = jnp.where(output == 0, 1.0, output)

        return output

    def forward_all(
        self,
        a: jnp.ndarray,
        b: jnp.ndarray
    ) -> Dict[str, jnp.ndarray]:
        """
        Apply all operations and return results.

        Useful for analysis and comparison.
        """
        return {
            'xor': self(a, b, 0),
            'and': self(a, b, 1),
            'or': self(a, b, 2),
            'implies': self(a, b, 3)
        }


class BinaryLogicLayerSoft(nn.Module):
    """
    Soft-routed version for end-to-end training.

    Instead of hard operation_id selection, learns to route
    based on input characteristics.
    """
    n_bits: int = 64
    threshold: float = 0.3

    def setup(self):
        self.logic_layer = BinaryLogicLayer(
            n_bits=self.n_bits,
            threshold=self.threshold
        )

    def __call__(
        self,
        a: jnp.ndarray,
        b: jnp.ndarray,
        operation_weights: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Soft-routed Boolean operation.

        Args:
            a, b: [batch, n_bits] in {-1, +1}
            operation_weights: [batch, 4] softmax weights over operations

        Returns:
            [batch, n_bits] weighted combination of operation outputs
        """
        outputs = []
        for op_id in range(4):
            out = self.logic_layer(a, b, op_id)
            outputs.append(out)

        # Stack: [batch, 4, n_bits]
        outputs = jnp.stack(outputs, axis=1)

        # Weight and sum: [batch, n_bits]
        weights = operation_weights[:, :, None]  # [batch, 4, 1]
        return jnp.sum(outputs * weights, axis=1)


# Ground truth Boolean operations in {-1, +1} encoding
# Convention: -1 = TRUE, +1 = FALSE
def ground_truth_xor(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """XOR in {-1, +1}: returns -1 when inputs differ, +1 when same."""
    return a * b


def ground_truth_and(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """AND in {-1, +1}: returns -1 only when both are -1 (TRUE)."""
    return jnp.sign(1 + a + b - a*b)


def ground_truth_or(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """OR in {-1, +1}: returns +1 only when both are +1 (FALSE)."""
    return jnp.sign(-1 + a + b + a*b)


def ground_truth_implies(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """IMPLIES in {-1, +1}: returns +1 only when a=-1 and b=+1."""
    return jnp.sign(-1 - a + b - a*b)


GROUND_TRUTH_OPS = {
    0: ground_truth_xor,
    1: ground_truth_and,
    2: ground_truth_or,
    3: ground_truth_implies
}

OP_NAMES = {0: 'XOR', 1: 'AND', 2: 'OR', 3: 'IMPLIES'}


# Expected masks (theoretical, for validation)
# Based on Boolean Fourier expansion with sign threshold
# Mask order: [constant, a, b, ab]
EXPECTED_MASKS = {
    'xor': jnp.array([0., 0., 0., 1.]),      # Only parity (ab)
    'and': jnp.array([1., 1., 1., -1.]),     # sign(1 + a + b - ab)
    'or': jnp.array([-1., 1., 1., 1.]),      # sign(-1 + a + b + ab)
    'implies': jnp.array([-1., -1., 1., -1.])  # sign(-1 - a + b - ab)
}


if __name__ == "__main__":
    print("="*60)
    print("Testing Binary Logic Layer")
    print("="*60)

    rng = jax.random.PRNGKey(42)

    # Create model
    model = BinaryLogicLayer(n_bits=8)

    # Initialize
    a = jax.random.choice(rng, jnp.array([-1., 1.]), (4, 8))
    b = jax.random.choice(jax.random.PRNGKey(123), jnp.array([-1., 1.]), (4, 8))

    variables = model.init(rng, a, b, 0)

    print("\n[Test 1] Model Structure")
    print(f"Parameters: {jax.tree_util.tree_map(lambda x: x.shape, variables['params'])}")

    print("\n[Test 2] Initial Masks (before training)")
    masks = model.apply(variables, method=model.get_all_masks)
    for i, name in enumerate(['XOR', 'AND', 'OR', 'IMPLIES']):
        print(f"  {name}: {masks[i]}")

    print("\n[Test 3] Boolean Fourier Features")
    a_test = jnp.array([[1., -1., 1., -1.]])
    b_test = jnp.array([[-1., 1., 1., -1.]])
    features = boolean_fourier_features(a_test, b_test)
    print(f"  a: {a_test}")
    print(f"  b: {b_test}")
    print(f"  Features [1, a, b, ab]:")
    print(f"    {features}")

    print("\n[Test 4] Ground Truth Operations")
    for op_id, op_name in OP_NAMES.items():
        gt = GROUND_TRUTH_OPS[op_id](a_test, b_test)
        print(f"  {op_name}: {gt}")

    print("\n[Test 5] Model Output (random init)")
    for op_id, op_name in OP_NAMES.items():
        pred = model.apply(variables, a_test, b_test, op_id)
        gt = GROUND_TRUTH_OPS[op_id](a_test, b_test)
        match = jnp.all(pred == gt)
        print(f"  {op_name}: pred={pred}, gt={gt}, match={match}")

    print("\n[Test 6] Expected Masks (theoretical)")
    for name, expected in EXPECTED_MASKS.items():
        print(f"  {name.upper()}: {expected}")

    print("\n" + "="*60)
    print("Tests complete! Model ready for training.")
    print("="*60)
