"""
Spectral Synthesis via BlackJAX
===============================

Synthesis pipeline for ternary masks using:
1. Fourier coefficient estimation (Monte Carlo)
2. Ternary quantization (thresholding)
3. BlackJAX MCMC refinement (discrete optimization)

This avoids SGD drift for discrete mask problems.
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import jax.random as random
import blackjax
from functools import partial
from typing import Callable, Tuple, Dict, NamedTuple
import numpy as np


# =============================================================================
# Boolean Fourier Basis Generation
# =============================================================================

def generate_all_subsets(n: int) -> jnp.ndarray:
    """
    Generate all 2^n subsets as binary vectors.

    Returns: [2^n, n] array where each row is a subset indicator.
    """
    return jnp.array([[int(b) for b in format(i, f'0{n}b')] for i in range(2**n)])


def compute_character(x: jnp.ndarray, subset: jnp.ndarray) -> jnp.ndarray:
    """
    Compute Walsh character χ_S(x) = ∏_{i∈S} x_i

    x: [..., n] in {-1, +1}
    subset: [n] binary indicator of which variables to include

    Returns: [...] in {-1, +1}
    """
    # For each position in subset, either take x_i or 1
    # χ_S(x) = ∏_i x_i^{S_i} = ∏_{i: S_i=1} x_i
    masked = jnp.where(subset, x, 1.0)
    return jnp.prod(masked, axis=-1)


def boolean_fourier_basis(x: jnp.ndarray, n_vars: int) -> jnp.ndarray:
    """
    Compute full Boolean Fourier basis for n variables.

    x: [batch, n_vars] in {-1, +1}

    Returns: [batch, 2^n_vars] basis values
    """
    subsets = generate_all_subsets(n_vars)  # [2^n, n]

    # Compute each character
    characters = []
    for s in range(2**n_vars):
        chi_s = compute_character(x, subsets[s])  # [batch]
        characters.append(chi_s)

    return jnp.stack(characters, axis=-1)  # [batch, 2^n]


# =============================================================================
# Fourier Coefficient Estimation (Monte Carlo)
# =============================================================================

def estimate_fourier_coefficients(
    target_fn: Callable,
    n_vars: int,
    n_samples: int = 10000,
    rng_key: jax.Array = None
) -> jnp.ndarray:
    """
    Estimate Fourier coefficients via Monte Carlo:

    f̂(S) = E[f(x) · χ_S(x)]

    target_fn: function mapping [batch, n_vars] -> [batch] in {-1, +1}
    n_vars: number of input variables
    n_samples: Monte Carlo samples

    Returns: [2^n_vars] estimated coefficients
    """
    if rng_key is None:
        rng_key = random.PRNGKey(42)

    # Sample uniform from {-1, +1}^n
    x = random.choice(rng_key, jnp.array([-1.0, 1.0]), shape=(n_samples, n_vars))

    # Compute target function
    f_x = target_fn(x)  # [n_samples]

    # Compute all characters
    basis = boolean_fourier_basis(x, n_vars)  # [n_samples, 2^n]

    # Estimate coefficients: f̂(S) = E[f(x) · χ_S(x)]
    coefficients = jnp.mean(f_x[:, None] * basis, axis=0)  # [2^n]

    return coefficients


def quantize_to_ternary(
    coefficients: jnp.ndarray,
    threshold: float = 0.1
) -> jnp.ndarray:
    """
    Quantize real-valued coefficients to ternary {-1, 0, +1}.

    coefficients: [d] real values
    threshold: values below this magnitude become 0

    Returns: [d] in {-1, 0, +1}
    """
    return jnp.where(
        jnp.abs(coefficients) < threshold,
        0.0,
        jnp.sign(coefficients)
    )


# =============================================================================
# BlackJAX MCMC for Mask Refinement
# =============================================================================

class MaskState(NamedTuple):
    """State for mask MCMC."""
    mask: jnp.ndarray  # [d] in {-1, 0, +1}
    accuracy: float


def compute_mask_accuracy(
    mask: jnp.ndarray,
    x: jnp.ndarray,
    targets: jnp.ndarray,
    n_vars: int
) -> float:
    """
    Compute accuracy of a ternary mask.

    mask: [2^n_vars] ternary coefficients
    x: [n_samples, n_vars] inputs
    targets: [n_samples] target outputs
    """
    basis = boolean_fourier_basis(x, n_vars)  # [n_samples, 2^n]
    output = jnp.sum(basis * mask, axis=-1)  # [n_samples]
    output = jnp.sign(output)
    output = jnp.where(output == 0, 1.0, output)
    return float(jnp.mean(output == targets))


def mask_log_likelihood(
    mask: jnp.ndarray,
    x: jnp.ndarray,
    targets: jnp.ndarray,
    n_vars: int,
    temperature: float = 1.0
) -> float:
    """
    Log-likelihood for mask based on accuracy.

    Higher accuracy → higher likelihood.
    Temperature controls sharpness.
    """
    acc = compute_mask_accuracy(mask, x, targets, n_vars)
    # Use accuracy as energy, lower temperature → sharper peak at optimal
    return acc / temperature


def gibbs_step(
    rng_key: jax.Array,
    mask: jnp.ndarray,
    x: jnp.ndarray,
    targets: jnp.ndarray,
    n_vars: int,
    temperature: float = 0.1
) -> jnp.ndarray:
    """
    One Gibbs sampling step: update one random coordinate.
    """
    d = mask.shape[0]

    # Choose random coordinate to update
    key1, key2 = random.split(rng_key)
    coord = random.randint(key1, (), 0, d)

    # Compute likelihood for each possible value at this coordinate
    values = jnp.array([-1.0, 0.0, 1.0])
    log_probs = []

    for v in values:
        new_mask = mask.at[coord].set(v)
        ll = mask_log_likelihood(new_mask, x, targets, n_vars, temperature)
        log_probs.append(ll)

    log_probs = jnp.array(log_probs)

    # Sample from categorical
    probs = jax.nn.softmax(log_probs)
    new_value = random.choice(key2, values, p=probs)

    return mask.at[coord].set(new_value)


def refine_mask_mcmc(
    initial_mask: jnp.ndarray,
    x: jnp.ndarray,
    targets: jnp.ndarray,
    n_vars: int,
    n_steps: int = 1000,
    temperature: float = 0.1,
    rng_key: jax.Array = None
) -> Tuple[jnp.ndarray, float]:
    """
    Refine ternary mask using Gibbs sampling MCMC.

    Returns: (best_mask, best_accuracy)
    """
    if rng_key is None:
        rng_key = random.PRNGKey(0)

    mask = initial_mask
    best_mask = mask
    best_acc = compute_mask_accuracy(mask, x, targets, n_vars)

    for step in range(n_steps):
        rng_key, step_key = random.split(rng_key)
        mask = gibbs_step(step_key, mask, x, targets, n_vars, temperature)

        acc = compute_mask_accuracy(mask, x, targets, n_vars)
        if acc > best_acc:
            best_acc = acc
            best_mask = mask

        # Anneal temperature
        if (step + 1) % 100 == 0:
            temperature = max(temperature * 0.9, 0.01)

    return best_mask, best_acc


# =============================================================================
# Parallel Tempering for Better Exploration
# =============================================================================

def parallel_tempering_step(
    rng_key: jax.Array,
    masks: jnp.ndarray,  # [n_chains, d]
    temperatures: jnp.ndarray,  # [n_chains]
    x: jnp.ndarray,
    targets: jnp.ndarray,
    n_vars: int
) -> jnp.ndarray:
    """
    One step of parallel tempering:
    1. Gibbs step for each chain
    2. Propose swaps between adjacent temperatures
    """
    n_chains, d = masks.shape

    # Gibbs step for each chain
    keys = random.split(rng_key, n_chains + 1)
    rng_key = keys[0]

    new_masks = []
    for i in range(n_chains):
        new_mask = gibbs_step(keys[i+1], masks[i], x, targets, n_vars, temperatures[i])
        new_masks.append(new_mask)
    masks = jnp.stack(new_masks)

    # Propose swaps between adjacent chains
    for i in range(n_chains - 1):
        rng_key, swap_key = random.split(rng_key)

        # Compute log-likelihoods
        ll_i = mask_log_likelihood(masks[i], x, targets, n_vars, temperatures[i])
        ll_j = mask_log_likelihood(masks[i+1], x, targets, n_vars, temperatures[i+1])
        ll_i_at_j = mask_log_likelihood(masks[i], x, targets, n_vars, temperatures[i+1])
        ll_j_at_i = mask_log_likelihood(masks[i+1], x, targets, n_vars, temperatures[i])

        # Acceptance ratio
        log_alpha = (ll_i_at_j + ll_j_at_i) - (ll_i + ll_j)
        accept = jnp.log(random.uniform(swap_key)) < log_alpha

        if accept:
            masks = masks.at[i].set(masks[i+1])
            masks = masks.at[i+1].set(masks[i])

    return masks


def refine_mask_parallel_tempering(
    initial_mask: jnp.ndarray,
    x: jnp.ndarray,
    targets: jnp.ndarray,
    n_vars: int,
    n_chains: int = 4,
    n_steps: int = 500,
    rng_key: jax.Array = None
) -> Tuple[jnp.ndarray, float]:
    """
    Refine mask using parallel tempering MCMC.

    Uses multiple chains at different temperatures for better exploration.
    """
    if rng_key is None:
        rng_key = random.PRNGKey(0)

    d = initial_mask.shape[0]

    # Initialize chains with small perturbations
    rng_key, init_key = random.split(rng_key)
    noise = random.normal(init_key, (n_chains, d)) * 0.1
    masks = jnp.clip(jnp.round(initial_mask + noise), -1, 1)
    masks = masks.at[0].set(initial_mask)  # Keep one at initial

    # Temperature ladder (geometric)
    temperatures = jnp.array([0.01 * (3.0 ** i) for i in range(n_chains)])

    best_mask = initial_mask
    best_acc = compute_mask_accuracy(initial_mask, x, targets, n_vars)

    for step in range(n_steps):
        rng_key, step_key = random.split(rng_key)
        masks = parallel_tempering_step(step_key, masks, temperatures, x, targets, n_vars)

        # Check cold chain for best
        acc = compute_mask_accuracy(masks[0], x, targets, n_vars)
        if acc > best_acc:
            best_acc = acc
            best_mask = masks[0]

        if (step + 1) % 100 == 0:
            print(f"  Step {step+1}: best_acc={best_acc:.2%}")

    return best_mask, best_acc


# =============================================================================
# Full Synthesis Pipeline
# =============================================================================

def synthesize_ternary_mask(
    target_fn: Callable,
    n_vars: int,
    n_estimation_samples: int = 10000,
    n_refinement_samples: int = 1000,
    n_mcmc_steps: int = 500,
    quantization_threshold: float = 0.1,
    use_parallel_tempering: bool = True,
    rng_key: jax.Array = None,
    verbose: bool = True
) -> Tuple[jnp.ndarray, Dict]:
    """
    Full spectral synthesis pipeline:

    1. Estimate Fourier coefficients via Monte Carlo
    2. Quantize to ternary
    3. Refine via MCMC

    Args:
        target_fn: Boolean function [batch, n_vars] -> [batch] in {-1, +1}
        n_vars: Number of input variables
        n_estimation_samples: Samples for coefficient estimation
        n_refinement_samples: Samples for MCMC refinement
        n_mcmc_steps: MCMC iterations
        quantization_threshold: Threshold for ternary quantization
        use_parallel_tempering: Use parallel tempering vs simple Gibbs

    Returns:
        (ternary_mask, info_dict)
    """
    if rng_key is None:
        rng_key = random.PRNGKey(42)

    if verbose:
        print(f"Synthesizing mask for {n_vars}-variable function...")
        print(f"  Basis dimension: {2**n_vars}")

    # Step 1: Estimate Fourier coefficients
    rng_key, est_key = random.split(rng_key)
    if verbose:
        print(f"\n1. Estimating Fourier coefficients ({n_estimation_samples} samples)...")

    coefficients = estimate_fourier_coefficients(
        target_fn, n_vars, n_estimation_samples, est_key
    )

    if verbose:
        print(f"   Raw coefficients: {coefficients.round(3)}")

    # Step 2: Quantize to ternary
    if verbose:
        print(f"\n2. Quantizing to ternary (threshold={quantization_threshold})...")

    initial_mask = quantize_to_ternary(coefficients, quantization_threshold)

    if verbose:
        mask_str = ''.join(['+' if x > 0 else '-' if x < 0 else '0' for x in initial_mask])
        support = int(jnp.sum(jnp.abs(initial_mask) > 0))
        print(f"   Initial mask: [{mask_str}] (support={support})")

    # Generate refinement samples
    rng_key, sample_key = random.split(rng_key)
    x_refine = random.choice(sample_key, jnp.array([-1.0, 1.0]),
                             shape=(n_refinement_samples, n_vars))
    targets_refine = target_fn(x_refine)

    # Check initial accuracy
    initial_acc = compute_mask_accuracy(initial_mask, x_refine, targets_refine, n_vars)
    if verbose:
        print(f"   Initial accuracy: {initial_acc:.2%}")

    # Step 3: MCMC refinement
    if initial_acc < 0.9999:
        if verbose:
            method = "parallel tempering" if use_parallel_tempering else "Gibbs"
            print(f"\n3. Refining via MCMC ({method}, {n_mcmc_steps} steps)...")

        rng_key, mcmc_key = random.split(rng_key)

        if use_parallel_tempering:
            final_mask, final_acc = refine_mask_parallel_tempering(
                initial_mask, x_refine, targets_refine, n_vars,
                n_chains=4, n_steps=n_mcmc_steps, rng_key=mcmc_key
            )
        else:
            final_mask, final_acc = refine_mask_mcmc(
                initial_mask, x_refine, targets_refine, n_vars,
                n_steps=n_mcmc_steps, rng_key=mcmc_key
            )
    else:
        if verbose:
            print("\n3. Skipping MCMC (already optimal)")
        final_mask = initial_mask
        final_acc = initial_acc

    if verbose:
        mask_str = ''.join(['+' if x > 0 else '-' if x < 0 else '0' for x in final_mask])
        support = int(jnp.sum(jnp.abs(final_mask) > 0))
        print(f"\nFinal mask: [{mask_str}] (support={support})")
        print(f"Final accuracy: {final_acc:.2%}")

    info = {
        'raw_coefficients': coefficients,
        'initial_mask': initial_mask,
        'initial_accuracy': initial_acc,
        'final_mask': final_mask,
        'final_accuracy': final_acc,
        'n_vars': n_vars,
        'support': int(jnp.sum(jnp.abs(final_mask) > 0)),
    }

    return final_mask, info


# =============================================================================
# Test with 3-variable functions
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Spectral Synthesis via BlackJAX MCMC")
    print("=" * 70)

    # Test functions
    def xor_3(x):
        """3-way XOR: a ⊕ b ⊕ c"""
        return x[:, 0] * x[:, 1] * x[:, 2]

    def majority_3(x):
        """Majority vote"""
        result = jnp.sign(x[:, 0] + x[:, 1] + x[:, 2])
        return jnp.where(result == 0, 1.0, result)

    def and_3(x):
        """3-way AND"""
        result = jnp.sign(-3 + x[:, 0] + x[:, 1] + x[:, 2] +
                         x[:, 0]*x[:, 1] + x[:, 0]*x[:, 2] + x[:, 1]*x[:, 2] +
                         x[:, 0]*x[:, 1]*x[:, 2])
        return jnp.where(result == 0, 1.0, result)

    def or_3(x):
        """3-way OR"""
        result = jnp.sign(3 + x[:, 0] + x[:, 1] + x[:, 2] -
                         x[:, 0]*x[:, 1] - x[:, 0]*x[:, 2] - x[:, 1]*x[:, 2] -
                         x[:, 0]*x[:, 1]*x[:, 2])
        return jnp.where(result == 0, 1.0, result)

    test_fns = [
        ('XOR_3', xor_3),
        ('MAJORITY_3', majority_3),
        ('AND_3', and_3),
        ('OR_3', or_3),
    ]

    results = {}

    for name, fn in test_fns:
        print(f"\n{'─'*60}")
        print(f"Synthesizing: {name}")
        print(f"{'─'*60}")

        mask, info = synthesize_ternary_mask(
            fn, n_vars=3,
            n_estimation_samples=10000,
            n_refinement_samples=1000,
            n_mcmc_steps=300,
            use_parallel_tempering=True,
            verbose=True
        )

        results[name] = info

    # Summary
    print("\n" + "=" * 70)
    print("SYNTHESIS SUMMARY")
    print("=" * 70)

    print("\nBasis: [1, a, b, c, ab, ac, bc, abc]")
    for name, info in results.items():
        mask_str = ''.join(['+' if x > 0 else '-' if x < 0 else '0' for x in info['final_mask']])
        print(f"  {name:15s}: [{mask_str}] acc={info['final_accuracy']:.2%} support={info['support']}")
