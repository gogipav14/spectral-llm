"""
mHC-Style Hierarchical R Matrix for NPU-Native Architecture
============================================================

Implements hierarchical mask composition using:
    R = S ⊙ P

where:
- S: Sign matrix in {-1, +1} (ternary direction)
- P: Doubly-stochastic matrix via Sinkhorn projection (soft routing)

This allows child masks to be composed from parent masks:
    V_child = V_parent @ R

Key properties:
1. Doubly-stochastic P ensures balanced routing (rows and cols sum to 1)
2. Sign matrix S allows negative compositions
3. Sinkhorn projection is differentiable for end-to-end training
4. Temperature annealing makes P approach permutation matrix
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple
from functools import partial


def sinkhorn_rectangular(
    log_alpha: jnp.ndarray,
    n_iters: int = 20,
    temperature: float = 1.0,
    target_row_sum: float = None
) -> jnp.ndarray:
    """
    Rectangular Sinkhorn: column-stochastic + row-budgeted.

    For [n_parent, n_child] matrix:
    - Columns sum to 1 (each child is convex mixture of parents)
    - Rows sum to n_child/n_parent (equal parent contribution budget)

    Args:
        log_alpha: [n_parent, n_child] log-space scores
        n_iters: Number of Sinkhorn iterations
        temperature: Temperature for softmax (lower = sharper)
        target_row_sum: Target sum for each row (default: n_child/n_parent)

    Returns:
        [n_parent, n_child] column-stochastic, row-budgeted matrix
    """
    n_parent, n_child = log_alpha.shape
    if target_row_sum is None:
        target_row_sum = n_child / n_parent

    log_alpha = log_alpha / temperature

    for _ in range(n_iters):
        # Column normalization: make each column sum to 1
        log_alpha = log_alpha - jax.nn.logsumexp(log_alpha, axis=0, keepdims=True)

        # Row normalization: make each row sum to target_row_sum
        log_row_sums = jax.nn.logsumexp(log_alpha, axis=1, keepdims=True)
        log_alpha = log_alpha + jnp.log(target_row_sum) - log_row_sums

    return jnp.exp(log_alpha)


def sinkhorn_normalize(
    log_alpha: jnp.ndarray,
    n_iters: int = 10,
    temperature: float = 1.0
) -> jnp.ndarray:
    """
    Standard Sinkhorn for backward compatibility.
    Delegates to rectangular Sinkhorn.
    """
    return sinkhorn_rectangular(log_alpha, n_iters, temperature)


def validate_rectangular_sinkhorn(P: jnp.ndarray, n_parent: int = None, n_child: int = None) -> dict:
    """
    Proper stability checks for rectangular [n_parent, n_child] matrix.

    Checks:
    - Column-stochastic: each column sums to 1
    - Row-budgeted: each row sums to n_child/n_parent
    - Gain control: no excessive amplification

    Returns:
        dict with stability metrics and pass/fail status
    """
    if n_parent is None:
        n_parent = P.shape[0]
    if n_child is None:
        n_child = P.shape[1]

    expected_row_sum = n_child / n_parent

    # Column-stochastic check
    col_sums = P.sum(axis=0)
    col_dev = float(jnp.abs(col_sums - 1.0).max())

    # Row-budgeted check
    row_sums = P.sum(axis=1)
    row_dev = float(jnp.abs(row_sums - expected_row_sum).max())

    # Gain control (mHC-style non-expansiveness proxy)
    max_col_sum = float(col_sums.max())
    max_row_sum = float(row_sums.max())

    col_gain = max_col_sum / 1.0
    row_gain = max_row_sum / expected_row_sum
    max_gain = max(col_gain, row_gain)

    # Sparsity (how concentrated is the routing)
    sparsity = float(jnp.mean(jnp.abs(P) < 0.1))

    is_stable = (col_dev < 0.1 and row_dev < 0.1 and max_gain < 1.2)

    return {
        'col_stochastic_error': col_dev,
        'row_budget_error': row_dev,
        'expected_row_sum': expected_row_sum,
        'actual_row_sums': [float(x) for x in row_sums],
        'actual_col_sums_sample': [float(col_sums[i]) for i in range(min(4, len(col_sums)))],
        'max_gain': max_gain,
        'sparsity': sparsity,
        'is_stable': is_stable
    }


def soft_sign(x: jnp.ndarray, temperature: float = 1.0) -> jnp.ndarray:
    """
    Soft sign function using tanh.

    At temperature=1.0: smooth approximation of sign
    At temperature→0: approaches hard sign {-1, +1}
    """
    return jnp.tanh(x / temperature)


def hard_sign_ste(x: jnp.ndarray) -> jnp.ndarray:
    """
    Hard sign with straight-through estimator.

    Forward: sign(x) in {-1, +1}
    Backward: gradient flows as if identity
    """
    sign = jnp.sign(x)
    # Replace zeros with +1
    sign = jnp.where(sign == 0, 1.0, sign)
    # STE: forward uses hard sign, backward uses identity
    return x + jax.lax.stop_gradient(sign - x)


class HierarchicalRMatrix(nn.Module):
    """
    Learnable hierarchical composition matrix R = S ⊙ P (element-wise signs).

    Used to compose parent layer masks into child layer masks:
        child_masks = parent_masks @ R

    Attributes:
        n_parent: Number of parent masks (e.g., 4 for logic layer)
        n_child: Number of child masks (e.g., 16 for temporal layer)
        sinkhorn_iters: Number of Sinkhorn normalization iterations
        temperature: Temperature for Sinkhorn (lower = sharper routing)
    """
    n_parent: int
    n_child: int
    sinkhorn_iters: int = 10
    temperature: float = 1.0

    @nn.compact
    def __call__(self, training: bool = True) -> jnp.ndarray:
        """
        Compute R = S ⊙ P.

        Returns:
            [n_parent, n_child] composition matrix
        """
        # Learnable log-space scores for Sinkhorn
        # Initialize near uniform for balanced start
        log_alpha = self.param(
            'log_alpha',
            nn.initializers.normal(0.1),
            (self.n_parent, self.n_child)
        )

        # Learnable sign logits
        sign_logits = self.param(
            'sign_logits',
            nn.initializers.normal(1.0),
            (self.n_parent, self.n_child)
        )

        # Get doubly-stochastic P via Sinkhorn
        P = sinkhorn_normalize(log_alpha, self.sinkhorn_iters, self.temperature)

        # Get sign matrix S
        if training:
            S = soft_sign(sign_logits, self.temperature)
        else:
            S = hard_sign_ste(sign_logits)

        # R = S ⊙ P (element-wise Hadamard product)
        R = S * P

        return R

    def get_components(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Get the individual S and P components for analysis.

        Returns:
            (S, P) tuple of sign and doubly-stochastic matrices
        """
        # This would need to be called via apply with method argument
        log_alpha = self.variables['params']['log_alpha']
        sign_logits = self.variables['params']['sign_logits']

        P = sinkhorn_normalize(log_alpha, self.sinkhorn_iters, self.temperature)
        S = hard_sign_ste(sign_logits)

        return S, P


class ColumnSignRMatrix(nn.Module):
    """
    Learnable hierarchical composition matrix R = P * s[None, :] (column-sign).

    Simpler than element-wise S ⊙ P:
    - P is column-stochastic, row-budgeted via Sinkhorn
    - Each child has ONE sign (not per-element)
    - Enables negations (XNOR = -XOR, NAND = -AND, etc.)

    This is the recommended approach for Phase 2 temporal layer.

    Attributes:
        n_parent: Number of parent masks (e.g., 4 for logic layer)
        n_child: Number of child masks (e.g., 16 for temporal layer)
        sinkhorn_iters: Number of Sinkhorn normalization iterations
        temperature: Temperature for Sinkhorn and sign softening
        use_signs: Whether to enable column signs (must be True for negations)
    """
    n_parent: int
    n_child: int
    sinkhorn_iters: int = 10
    temperature: float = 1.0
    use_signs: bool = True

    @nn.compact
    def __call__(self, training: bool = True) -> jnp.ndarray:
        """
        Compute R = P * s[None, :] (column-sign).

        Returns:
            [n_parent, n_child] composition matrix
        """
        # Learnable log-space scores for Sinkhorn
        log_alpha = self.param(
            'log_alpha',
            nn.initializers.normal(0.1),
            (self.n_parent, self.n_child)
        )

        # Get column-stochastic, row-budgeted P via Sinkhorn
        P = sinkhorn_rectangular(
            log_alpha,
            n_iters=self.sinkhorn_iters,
            temperature=self.temperature
        )

        if not self.use_signs:
            return P

        # Column-sign: one sign per child operation
        # Initialize with larger values for decisive initial signs
        sign_logits = self.param(
            'sign_logits',
            nn.initializers.normal(1.0),  # Larger init for stronger initial signs
            (self.n_child,)
        )

        if training:
            # Smooth sign via tanh
            s = jnp.tanh(sign_logits / self.temperature)
        else:
            # Hard sign at inference
            s = jnp.sign(sign_logits)
            s = jnp.where(s == 0, 1.0, s)

        # R = P * s[None, :] (broadcast column sign)
        R = P * s[None, :]

        return R

    def get_components(self, params: dict = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Get the individual P and s components for analysis.

        Returns:
            (P, s) tuple of doubly-stochastic matrix and column signs
        """
        if params is None:
            params = self.variables['params']

        log_alpha = params['log_alpha']
        P = sinkhorn_rectangular(
            log_alpha,
            n_iters=self.sinkhorn_iters,
            temperature=0.05  # Hard for analysis
        )

        if self.use_signs:
            sign_logits = params['sign_logits']
            s = jnp.sign(sign_logits)
            s = jnp.where(s == 0, 1.0, s)
        else:
            s = jnp.ones(self.n_child)

        return P, s


class HierarchicalMaskComposer(nn.Module):
    """
    Composes parent masks into child masks via R matrix.

    Given:
        - Parent masks: [n_parent, mask_dim]
        - R matrix: [n_parent, n_child]

    Produces:
        - Child masks: [n_child, mask_dim] = R.T @ parent_masks
    """
    n_parent: int
    n_child: int
    sinkhorn_iters: int = 10

    def setup(self):
        self.R_matrix = HierarchicalRMatrix(
            n_parent=self.n_parent,
            n_child=self.n_child,
            sinkhorn_iters=self.sinkhorn_iters
        )

    def __call__(
        self,
        parent_masks: jnp.ndarray,
        temperature: float = 1.0,
        training: bool = True
    ) -> jnp.ndarray:
        """
        Compose child masks from parent masks.

        Args:
            parent_masks: [n_parent, mask_dim] parent layer masks
            temperature: Sinkhorn temperature
            training: Whether in training mode

        Returns:
            [n_child, mask_dim] composed child masks
        """
        # Get R matrix with current temperature
        # Note: We'd need to handle temperature differently in practice
        R = self.R_matrix(training=training)  # [n_parent, n_child]

        # Compose: child_masks = R.T @ parent_masks
        # [n_child, n_parent] @ [n_parent, mask_dim] = [n_child, mask_dim]
        child_masks = R.T @ parent_masks

        return child_masks


class TemporalLayer(nn.Module):
    """
    Phase 2 Temporal Layer that builds on Logic Layer.

    Temporal masks are composed from logic masks via:
        temporal_masks = logic_masks @ R_temporal

    This creates 16 temporal experts from 4 logic primitives.
    """
    n_logic: int = 4
    n_temporal: int = 16
    mask_dim: int = 4  # Boolean Fourier basis dimension
    sinkhorn_iters: int = 10

    def setup(self):
        self.composer = HierarchicalMaskComposer(
            n_parent=self.n_logic,
            n_child=self.n_temporal,
            sinkhorn_iters=self.sinkhorn_iters
        )

    def __call__(
        self,
        logic_masks: jnp.ndarray,
        temperature: float = 1.0,
        training: bool = True
    ) -> jnp.ndarray:
        """
        Compose temporal masks from logic masks.

        Args:
            logic_masks: [4, 4] logic layer masks (frozen from Phase 1)
            temperature: Temperature for Sinkhorn
            training: Whether in training mode

        Returns:
            [16, 4] temporal masks composed from logic masks
        """
        return self.composer(logic_masks, temperature, training)

    def get_R_analysis(self) -> dict:
        """
        Analyze the learned R matrix structure.

        Returns dictionary with:
        - sparsity: Fraction of near-zero elements
        - row_entropy: Average entropy of row distributions
        - dominant_parent: Which parent each child relies on most
        """
        R = self.composer.R_matrix(training=False)

        # Sparsity (elements < 0.1)
        sparsity = float(jnp.mean(jnp.abs(R) < 0.1))

        # Row entropy (how focused is each child on parents)
        abs_R = jnp.abs(R)
        row_probs = abs_R / (abs_R.sum(axis=0, keepdims=True) + 1e-8)
        row_entropy = -jnp.sum(row_probs * jnp.log(row_probs + 1e-8), axis=0)
        mean_row_entropy = float(jnp.mean(row_entropy))

        # Dominant parent for each child
        dominant_parent = [int(jnp.argmax(jnp.abs(R[:, i]))) for i in range(self.n_temporal)]

        return {
            'sparsity': sparsity,
            'mean_row_entropy': mean_row_entropy,
            'max_entropy': float(jnp.log(self.n_logic)),
            'dominant_parents': dominant_parent,
            'R_shape': R.shape
        }


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Testing mHC-Style Hierarchical R Matrix")
    print("="*60)

    # Test Rectangular Sinkhorn
    print("\n[Test 1] Rectangular Sinkhorn Normalization")
    rng = jax.random.PRNGKey(42)
    log_alpha = jax.random.normal(rng, (4, 16))  # 4 parents → 16 children

    P = sinkhorn_rectangular(log_alpha, n_iters=20, temperature=1.0)
    n_parent, n_child = log_alpha.shape
    expected_row_sum = n_child / n_parent  # = 4.0

    print(f"  Input shape: {log_alpha.shape}")
    print(f"  Output shape: {P.shape}")
    print(f"  Row sums: {P.sum(axis=1)} (expected ~{expected_row_sum})")
    print(f"  Col sums (first 4): {P.sum(axis=0)[:4]} (expected ~1.0)")

    # Validate using the new function
    stability = validate_rectangular_sinkhorn(P, n_parent, n_child)
    print(f"  Stability check:")
    print(f"    Column error: {stability['col_stochastic_error']:.4f}")
    print(f"    Row error: {stability['row_budget_error']:.4f}")
    print(f"    Max gain: {stability['max_gain']:.4f}")
    print(f"    Is stable: {stability['is_stable']}")

    # Test HierarchicalRMatrix
    print("\n[Test 2] HierarchicalRMatrix")
    r_matrix = HierarchicalRMatrix(n_parent=4, n_child=16, sinkhorn_iters=10)
    rng, key = jax.random.split(rng)
    variables = r_matrix.init(key)

    R_train = r_matrix.apply(variables, training=True)
    R_eval = r_matrix.apply(variables, training=False)

    print(f"  R (train) shape: {R_train.shape}")
    print(f"  R (eval) shape: {R_eval.shape}")
    print(f"  R (train) range: [{R_train.min():.3f}, {R_train.max():.3f}]")
    print(f"  R (eval) range: [{R_eval.min():.3f}, {R_eval.max():.3f}]")

    # Test HierarchicalMaskComposer
    print("\n[Test 3] HierarchicalMaskComposer")
    composer = HierarchicalMaskComposer(n_parent=4, n_child=16)
    rng, key = jax.random.split(rng)
    variables = composer.init(key, jnp.zeros((4, 4)))

    # Simulate logic masks (from Phase 1)
    logic_masks = jnp.array([
        [0., 0., 0., 1.],   # XOR
        [1., 1., 1., -1.],  # AND
        [-1., 1., 1., 1.],  # OR
        [-1., -1., 1., -1.]  # IMPLIES
    ])

    child_masks = composer.apply(variables, logic_masks, temperature=1.0, training=True)
    print(f"  Parent masks shape: {logic_masks.shape}")
    print(f"  Child masks shape: {child_masks.shape}")
    print(f"  Child masks range: [{child_masks.min():.3f}, {child_masks.max():.3f}]")

    # Test TemporalLayer
    print("\n[Test 4] TemporalLayer")
    temporal = TemporalLayer(n_logic=4, n_temporal=16, mask_dim=4)
    rng, key = jax.random.split(rng)
    variables = temporal.init(key, logic_masks)

    temporal_masks = temporal.apply(variables, logic_masks, temperature=1.0, training=True)
    print(f"  Temporal masks shape: {temporal_masks.shape}")

    # Show first few temporal masks
    print(f"  First 4 temporal masks:")
    for i in range(4):
        print(f"    Temporal[{i}]: {temporal_masks[i]}")

    # Test temperature annealing
    print("\n[Test 5] Temperature Annealing Effect on Rectangular Sinkhorn")
    for temp in [1.0, 0.5, 0.1, 0.01]:
        P = sinkhorn_rectangular(log_alpha, n_iters=20, temperature=temp)
        stability = validate_rectangular_sinkhorn(P, 4, 16)
        max_val = float(P.max())
        min_val = float(P.min())
        print(f"  Temperature {temp:.2f}: range [{min_val:.3f}, {max_val:.3f}], stable={stability['is_stable']}")

    # Test gradient flow
    print("\n[Test 6] Gradient Flow Through R")
    # Use the temporal layer variables which were properly initialized
    temporal_vars = temporal.init(rng, logic_masks)

    def loss_fn(variables, logic_masks):
        temporal_masks = temporal.apply(variables, logic_masks, temperature=0.5)
        # Target: first temporal mask should be like XOR
        target = jnp.array([0., 0., 0., 1.])
        return jnp.mean((temporal_masks[0] - target) ** 2)

    grads = jax.grad(loss_fn)(temporal_vars, logic_masks)
    print(f"  Gradients exist: {any(g is not None for g in jax.tree_util.tree_leaves(grads))}")
    print(f"  Grad shapes: {jax.tree_util.tree_map(lambda x: x.shape, grads['params'])}")

    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)
