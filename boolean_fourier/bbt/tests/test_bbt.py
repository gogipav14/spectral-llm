"""
Unit tests for BBT modules.

Tests core mathematical properties:
- Fourier coefficient correctness and Parseval identity
- Influence computation for known functions
- Operator norm analytical formulas vs numerical verification
- Margin product bounds
- Walsh matrix properties (H² = NI)
- Cancellation index values for known pairs
- Repair convergence on corrupted masks
"""

import os
import sys
import unittest

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import jax.numpy as jnp


class TestInfluence(unittest.TestCase):
    """Test influence computation."""

    def setUp(self):
        from bbt.influence import fourier_coefficients, all_influences, total_influence
        self.fourier_coefficients = fourier_coefficients
        self.all_influences = all_influences
        self.total_influence = total_influence

    def test_parity3_influences(self):
        """Parity_3: all influences = 1, total = 3."""
        n = 3
        N = 1 << n
        tt = jnp.array([
            np.prod([1.0 - 2.0 * ((i >> b) & 1) for b in range(n)])
            for i in range(N)
        ], dtype=jnp.float32)
        f_hat = self.fourier_coefficients(tt, n)
        infs = self.all_influences(f_hat, n)
        total = self.total_influence(f_hat, n)

        np.testing.assert_allclose(infs, [1.0, 1.0, 1.0], atol=1e-5)
        self.assertAlmostEqual(total, 3.0, places=4)

    def test_dictator_influences(self):
        """Dictator x_0: Inf_0=1, rest=0."""
        n = 3
        N = 1 << n
        tt = jnp.array([1.0 - 2.0 * ((i >> 0) & 1) for i in range(N)], dtype=jnp.float32)
        f_hat = self.fourier_coefficients(tt, n)
        infs = self.all_influences(f_hat, n)

        self.assertAlmostEqual(float(infs[0]), 1.0, places=5)
        self.assertAlmostEqual(float(infs[1]), 0.0, places=5)
        self.assertAlmostEqual(float(infs[2]), 0.0, places=5)

    def test_parseval(self):
        """Parseval identity: sum of squared Fourier coefficients = 1 for Boolean f."""
        n = 3
        N = 1 << n
        tt = jnp.array([1.0 - 2.0 * ((i >> 0) & 1) for i in range(N)], dtype=jnp.float32)
        f_hat = self.fourier_coefficients(tt, n)
        self.assertAlmostEqual(float(jnp.sum(f_hat ** 2)), 1.0, places=5)


class TestBanachNorms(unittest.TestCase):
    """Test operator norm formulas."""

    def setUp(self):
        from bbt.banach_norms import (
            butterfly_operator_norm, butterfly_inverse_contraction,
            margin_product_log2, verify_operator_norm_numerically,
        )
        self.butterfly_operator_norm = butterfly_operator_norm
        self.butterfly_inverse_contraction = butterfly_inverse_contraction
        self.margin_product_log2 = margin_product_log2
        self.verify_operator_norm_numerically = verify_operator_norm_numerically

    def test_norm_p1(self):
        """p=1: ||A||=2, ||A^-1||=1."""
        self.assertAlmostEqual(self.butterfly_operator_norm(1.0), 2.0, places=10)
        self.assertAlmostEqual(self.butterfly_inverse_contraction(1.0), 1.0, places=10)

    def test_norm_p2(self):
        """p=2: ||A||=sqrt(2), ||A^-1||=1/sqrt(2)."""
        self.assertAlmostEqual(self.butterfly_operator_norm(2.0), 2 ** 0.5, places=10)
        self.assertAlmostEqual(self.butterfly_inverse_contraction(2.0), 2 ** -0.5, places=10)

    def test_norm_numerical_agreement(self):
        """Analytical norm matches numerical grid search."""
        for p in [1.0, 1.25, 1.5, 1.75, 2.0]:
            result = self.verify_operator_norm_numerically(p, n_samples=50000)
            self.assertLess(result['relative_error'], 1e-4,
                           f"Norm mismatch at p={p}: analytical={result['analytical']}, "
                           f"numerical={result['numerical']}")

    def test_margin_parity(self):
        """Parity: log2(mu) = -n/2."""
        infs = jnp.array([1.0, 1.0, 1.0])
        self.assertAlmostEqual(self.margin_product_log2(infs), -1.5, places=5)

    def test_margin_dictator(self):
        """Dictator: log2(mu) = -0.5."""
        infs = jnp.array([1.0, 0.0, 0.0])
        self.assertAlmostEqual(self.margin_product_log2(infs), -0.5, places=5)

    def test_margin_constant(self):
        """Constant function: log2(mu) = 0."""
        infs = jnp.array([0.0, 0.0, 0.0])
        self.assertAlmostEqual(self.margin_product_log2(infs), 0.0, places=5)


class TestWalshMatrix(unittest.TestCase):
    """Test Walsh-Hadamard matrix properties."""

    def test_h_squared_equals_ni(self):
        """H² = N·I for n=2,3,4."""
        from bbt.exhaustive_validation import build_walsh_matrix
        for n in [2, 3, 4]:
            H = build_walsh_matrix(n)
            N = 1 << n
            np.testing.assert_allclose(H @ H, N * np.eye(N), atol=1e-5)

    def test_h_integer_entries(self):
        """H has entries in {-1, +1}."""
        from bbt.exhaustive_validation import build_walsh_matrix
        H = build_walsh_matrix(3)
        self.assertTrue(np.all(np.isin(H.astype(int), [-1, 1])))


class TestCancellation(unittest.TestCase):
    """Test cancellation index."""

    def test_pair_values(self):
        """Known pair cancellation ratios."""
        from bbt.cancellation_index import pair_cancellation
        self.assertAlmostEqual(pair_cancellation(1, 0), 1.0, places=5)
        self.assertAlmostEqual(pair_cancellation(1, 1), 0.0, places=5)
        self.assertAlmostEqual(pair_cancellation(1, -1), 0.0, places=5)
        self.assertAlmostEqual(pair_cancellation(0, 0), 0.0, places=5)


class TestRepair(unittest.TestCase):
    """Test repair algorithm."""

    def test_repair_corrupted_parity(self):
        """Repair a deliberately wrong mask for parity_3."""
        from bbt.repair import repair_single
        from bbt.exhaustive_validation import build_walsh_matrix

        n = 3
        N = 1 << n
        H = build_walsh_matrix(n)

        # Parity truth table
        tt = np.array([
            np.prod([1.0 - 2.0 * ((i >> b) & 1) for b in range(n)])
            for i in range(N)
        ], dtype=np.float32)

        # Wrong mask (constant)
        wrong = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)

        mask, converged, iters, diag = repair_single(wrong, tt, H)
        self.assertTrue(converged, "Repair should converge for parity_3")

        # Verify the repaired mask
        predicted = np.sign(H @ mask)
        predicted[predicted == 0] = 1.0
        self.assertTrue(np.array_equal(predicted, tt))

    def test_repair_preserves_ternary(self):
        """Repaired mask values must be in {-1, 0, +1}."""
        from bbt.repair import repair_single
        from bbt.exhaustive_validation import build_walsh_matrix

        n = 3
        H = build_walsh_matrix(n)
        tt = np.array([1, -1, -1, 1, -1, 1, 1, -1], dtype=np.float32)
        wrong = np.zeros(8, dtype=np.float32)

        mask, _, _, _ = repair_single(wrong, tt, H)
        self.assertTrue(np.all(np.isin(mask.astype(int), [-1, 0, 1])),
                       f"Mask contains non-ternary values: {mask}")


class TestEnumerate(unittest.TestCase):
    """Test Boolean function enumeration."""

    def test_npn_classes_n3(self):
        """n=3 has exactly 14 NPN classes."""
        from bbt.enumerate import compute_npn_classes
        classes = compute_npn_classes(3)
        self.assertEqual(len(classes), 14)

    def test_all_functions_count(self):
        """2^{2^n} functions for n=2,3."""
        from bbt.enumerate import all_boolean_functions
        self.assertEqual(len(all_boolean_functions(2)), 16)
        self.assertEqual(len(all_boolean_functions(3)), 256)


if __name__ == '__main__':
    unittest.main()
