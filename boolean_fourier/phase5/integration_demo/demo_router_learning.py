"""
Integration Demo: Differentiable Router + Frozen Primitives

Task: Mode-switching binary logic
- Input: (a, b, mode) where a,b ∈ {-1,+1}, mode ∈ {-1,+1}
- Target: if mode=-1 then XOR(a,b) else AND(a,b)

Architecture:
- Frozen primitives: [XOR, AND, OR, IMPLIES] from Phase 1
- Trainable router: Sinkhorn-constrained matrix + sign vector
- Router selects primitive based on mode feature

Demonstrates:
1. End-to-end gradient flow through Sinkhorn + sign modulation
2. Frozen spectral primitives as building blocks
3. Quantization evaluation (hard routing k=1 + ternary signs)
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
import numpy as np
import json
from pathlib import Path
from datetime import datetime

# Import Sinkhorn from Phase 2
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / 'phase2'))
from hierarchical_r import sinkhorn_rectangular


# Phase 1 canonical masks (from train_phase1_fixed.py)
PHASE1_MASKS = {
    'xor': jnp.array([0, 0, 0, 1], dtype=jnp.float32),
    'and': jnp.array([1, 1, 1, -1], dtype=jnp.float32),
    'or': jnp.array([-1, 1, 1, 1], dtype=jnp.float32),
    'implies': jnp.array([1, -1, 1, 1], dtype=jnp.float32),
}


def boolean_fourier_basis_2var(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """
    Compute 2-variable Boolean Fourier basis: [1, a, b, ab]

    Args:
        a, b: (batch,) in {-1, +1}

    Returns:
        phi: (batch, 4) basis features
    """
    return jnp.stack([jnp.ones_like(a), a, b, a * b], axis=-1)


class DifferentiableRouter(nn.Module):
    """
    Trainable router with Sinkhorn constraint + sign modulation.

    Learns to route input to correct frozen primitive based on mode.
    """
    n_primitives: int = 4
    n_features: int = 3  # [a, b, mode]

    def setup(self):
        # Routing logits: features → primitives
        self.P_logits = self.param('P_logits',
                                    nn.initializers.normal(stddev=0.1),
                                    (self.n_features, self.n_primitives))

        # Sign modulation logits
        self.s_logits = self.param('s_logits',
                                    nn.initializers.zeros,
                                    (self.n_primitives,))

    def __call__(self, x: jnp.ndarray, temperature: float = 1.0):
        """
        Route input x to primitives with soft selection.

        Args:
            x: (batch, n_features) features
            temperature: Sinkhorn temperature

        Returns:
            P: (n_features, n_primitives) routing matrix (column-stochastic)
            s: (n_primitives,) sign vector in [-1, +1]
        """
        # Sinkhorn projection (column-stochastic)
        P = sinkhorn_rectangular(self.P_logits,
                                 n_iters=20,
                                 temperature=temperature)

        # Sign modulation via tanh
        s = jnp.tanh(self.s_logits)

        return P, s


class GumbelSTERouter(nn.Module):
    """
    Gumbel-Softmax with Straight-Through Estimator (from Mind the Gap paper).

    Forward: Hard routing via argmax (one-hot)
    Backward: Soft Gumbel-Softmax gradient (differentiable)

    Expected: Faster convergence than soft Sinkhorn, sharper routing decisions.
    """
    n_primitives: int = 4
    n_features: int = 3  # [a, b, mode]

    def setup(self):
        # Routing logits: features → primitives
        self.P_logits = self.param('P_logits',
                                    nn.initializers.normal(stddev=0.1),
                                    (self.n_features, self.n_primitives))

        # Sign modulation logits
        self.s_logits = self.param('s_logits',
                                    nn.initializers.zeros,
                                    (self.n_primitives,))

    def __call__(self, x: jnp.ndarray, temperature: float = 1.0):
        """
        Gumbel-Softmax routing with straight-through estimator.

        Args:
            x: (batch, n_features) features
            temperature: Gumbel-Softmax temperature

        Returns:
            P: (n_features, n_primitives) routing matrix (approximately one-hot via STE)
            s: (n_primitives,) sign vector in [-1, +1]
        """
        # Gumbel-Softmax: add Gumbel noise for exploration
        key = self.make_rng('gumbel')
        gumbel_noise = jax.random.gumbel(key, shape=self.P_logits.shape)
        logits_with_noise = (self.P_logits + gumbel_noise) / temperature

        # Soft routing (Gumbel-Softmax)
        P_soft = jax.nn.softmax(logits_with_noise, axis=-1)  # (n_features, n_primitives)

        # Hard routing (argmax) - used in forward pass
        P_hard = jax.nn.one_hot(jnp.argmax(P_soft, axis=-1), self.n_primitives)

        # Straight-through: forward uses hard, backward uses soft
        P = jax.lax.stop_gradient(P_hard - P_soft) + P_soft

        # Sign modulation via tanh
        s = jnp.tanh(self.s_logits)

        return P, s


class ReinMaxRouter(nn.Module):
    """
    Sinkhorn with entropic regularization (inspired by CardNN/ReinMax).

    Uses standard Sinkhorn projection in the forward pass.
    Entropy regularization is applied in the loss function (train_step_reinmax),
    NOT in the logits, because Sinkhorn is shift-invariant in log-space.
    """
    n_primitives: int = 4
    n_features: int = 3  # [a, b, mode]

    def setup(self):
        self.P_logits = self.param('P_logits',
                                    nn.initializers.normal(stddev=0.1),
                                    (self.n_features, self.n_primitives))

        self.s_logits = self.param('s_logits',
                                    nn.initializers.zeros,
                                    (self.n_primitives,))

    def __call__(self, x: jnp.ndarray, temperature: float = 1.0):
        """
        Forward pass: standard Sinkhorn projection.
        Entropy regularization is handled externally in the loss.
        """
        P = sinkhorn_rectangular(self.P_logits,
                                 n_iters=20,
                                 temperature=temperature)

        s = jnp.tanh(self.s_logits)

        return P, s


def compute_gini(P: jnp.ndarray) -> float:
    """
    Compute Gini coefficient for routing sparsity.

    Gini = 0 → uniform distribution (all weights equal)
    Gini = 1 → one-hot distribution (perfect sparsity)

    Args:
        P: (n_features, n_primitives) routing matrix

    Returns:
        gini: scalar sparsity measure
    """
    # Flatten to 1D and sort
    p_flat = jnp.abs(P.flatten())
    p_sorted = jnp.sort(p_flat)

    n = len(p_sorted)
    index = jnp.arange(1, n + 1)
    gini = (2 * jnp.sum(index * p_sorted)) / (n * jnp.sum(p_sorted)) - (n + 1) / n

    return float(gini)


def compute_interpretability_metrics(
    P: jnp.ndarray,
    s: jnp.ndarray,
    mode: jnp.ndarray,
    features: jnp.ndarray,
    prev_P: jnp.ndarray = None
) -> dict:
    """
    Compute transparent-by-design interpretability metrics.

    Metrics:
    1. Routing Sparsity (Gini): How concentrated are routing weights?
    2. Expert Specialization: Do experts focus on specific modes?
    3. Decision Stability: Consistency with previous routing state
    4. Sign Coherence: How close are signs to ternary {-1, 0, +1}?

    Args:
        P: (n_features, n_primitives) routing matrix
        s: (n_primitives,) sign vector
        mode: (batch,) mode indicators
        features: (batch, n_features) input features [a, b, mode]
        prev_P: Previous routing matrix for stability computation

    Returns:
        metrics: Dict with interpretability scores
    """
    # 1. Routing Sparsity (Gini coefficient)
    sparsity = compute_gini(P)

    # 2. Expert Specialization: mode-conditional routing entropy
    # Compute per-sample routing weights and stratify by mode
    routing_weights = jnp.abs(features @ P)  # (batch, n_prim)
    # Normalize to distribution per sample
    routing_dist = routing_weights / (routing_weights.sum(axis=-1, keepdims=True) + 1e-10)

    mode_mask_neg1 = (mode == -1)
    mode_mask_pos1 = (mode == +1)

    n_prims = P.shape[1]
    log_n = jnp.log(n_prims)

    # Average routing distribution for each mode
    if mode_mask_neg1.sum() > 0:
        avg_dist_neg1 = routing_dist[mode_mask_neg1].mean(axis=0)
        entropy_neg1 = -jnp.sum(avg_dist_neg1 * jnp.log(avg_dist_neg1 + 1e-10)) / log_n
    else:
        entropy_neg1 = 1.0

    if mode_mask_pos1.sum() > 0:
        avg_dist_pos1 = routing_dist[mode_mask_pos1].mean(axis=0)
        entropy_pos1 = -jnp.sum(avg_dist_pos1 * jnp.log(avg_dist_pos1 + 1e-10)) / log_n
    else:
        entropy_pos1 = 1.0

    # Specialization = 1 - average normalized entropy (lower entropy = more specialized)
    specialization = 1.0 - (float(entropy_neg1) + float(entropy_pos1)) / 2

    # 3. Decision Stability: consistency with previous routing
    if prev_P is not None:
        stability = 1.0 - float(jnp.linalg.norm(P - prev_P) / jnp.sqrt(P.size))
        stability = max(0.0, min(1.0, stability))
    else:
        stability = 1.0

    # 4. Sign Coherence: How close are signs to ternary {-1, 0, +1}?
    s_ternary = jnp.round(s)
    sign_coherence = 1.0 - float(jnp.mean(jnp.abs(s - s_ternary)))

    return {
        'routing_sparsity': float(sparsity),
        'expert_specialization': float(specialization),
        'decision_stability': float(stability),
        'sign_coherence': float(sign_coherence),
    }


class IntegrationDemoModel(nn.Module):
    """
    Full model: Router + Frozen Primitives
    """
    frozen_masks: dict  # {name: (4,) mask}
    router_cls: type = DifferentiableRouter  # Router class to use

    def setup(self):
        self.router = self.router_cls(
            n_primitives=len(self.frozen_masks),
            n_features=3
        )

    def __call__(self, a: jnp.ndarray, b: jnp.ndarray, mode: jnp.ndarray,
                 temperature: float = 1.0):
        """
        Forward pass.

        Args:
            a, b: (batch,) binary features in {-1, +1}
            mode: (batch,) mode indicator in {-1, +1}
            temperature: Routing temperature

        Returns:
            output: (batch,) soft logits
        """
        # Compute Fourier basis
        phi = boolean_fourier_basis_2var(a, b)  # (batch, 4)

        # Stack frozen masks
        mask_matrix = jnp.stack(list(self.frozen_masks.values()), axis=0)  # (n_prim, 4)

        # Compute primitive outputs
        prim_outputs = phi @ mask_matrix.T  # (batch, n_prim)

        # Route based on features [a, b, mode]
        features = jnp.stack([a, b, mode], axis=-1)  # (batch, 3)
        P, s = self.router(features, temperature)

        # Weighted combination with sign modulation
        routing_weights = features @ P  # (batch, n_prim)
        output = jnp.sum(routing_weights * s * prim_outputs, axis=-1)  # (batch,)

        return output


def create_mode_switching_dataset(n_samples: int, key: jax.Array):
    """
    Create mode-switching dataset:
    - mode = -1 → XOR(a, b)
    - mode = +1 → AND(a, b)

    Convention: +1 = False, -1 = True (standard Boolean Fourier encoding).
    XOR mask [0,0,0,1]: f(a,b) = a*b → True(-1) when inputs differ
    AND mask [1,1,1,-1]: f(a,b) = sign(1+a+b-ab) → True(-1) only when both -1(True)
    """
    keys = jax.random.split(key, 3)

    a = jax.random.choice(keys[0], jnp.array([-1, 1]), shape=(n_samples,))
    b = jax.random.choice(keys[1], jnp.array([-1, 1]), shape=(n_samples,))
    mode = jax.random.choice(keys[2], jnp.array([-1, 1]), shape=(n_samples,))

    # Target logic (convention: -1 = True, +1 = False)
    xor_target = a * b  # XOR: True(-1) when inputs differ
    and_target = jnp.where((a == -1) & (b == -1), -1, 1)  # AND: True(-1) only when both True(-1)
    target = jnp.where(mode == -1, xor_target, and_target)

    return a, b, mode, target


@jax.jit
def train_step_standard(state, a, b, mode, target, temperature):
    """JIT-compiled training step for Sinkhorn/ReinMax (no RNG needed)."""
    def loss_fn(params):
        output = state.apply_fn({'params': params}, a, b, mode, temperature)
        loss = jnp.mean((1 - jnp.tanh(output) * target) / 2)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


@jax.jit
def train_step_gumbel(state, a, b, mode, target, temperature, rng_key):
    """JIT-compiled training step for Gumbel-STE (rng_key is traced, not static)."""
    def loss_fn(params):
        output = state.apply_fn({'params': params}, a, b, mode, temperature,
                               rngs={'gumbel': rng_key})
        loss = jnp.mean((1 - jnp.tanh(output) * target) / 2)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


@jax.jit
def train_step_reinmax(state, a, b, mode, target, temperature, entropy_reg):
    """JIT-compiled training step for ReinMax (adds entropy regularization to loss)."""
    def loss_fn(params):
        output = state.apply_fn({'params': params}, a, b, mode, temperature)
        task_loss = jnp.mean((1 - jnp.tanh(output) * target) / 2)

        # Entropy regularization: encourage exploration in routing matrix
        P_logits = params['router']['P_logits']
        P = sinkhorn_rectangular(P_logits, n_iters=20, temperature=temperature)
        routing_entropy = -jnp.sum(P * jnp.log(P + 1e-10))
        # Subtract entropy penalty (maximize entropy = minimize negative entropy)
        loss = task_loss - entropy_reg * routing_entropy
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def run_integration_demo(
    n_train: int = 10000,
    n_test: int = 5000,
    n_steps: int = 5000,
    batch_size: int = 128,  # Match Phase 3 to avoid OOM
    lr: float = 1e-2,
    temp_start: float = 1.0,
    temp_end: float = 0.1,
    seed: int = 0,
    router_type: str = 'sinkhorn',  # 'sinkhorn', 'gumbel-ste', 'reinmax'
):
    """
    Run integration demo experiment.

    Args:
        router_type: Type of router to use ('sinkhorn', 'gumbel-ste', 'reinmax')

    Returns:
        results: Dict with training curve, final accuracy, and interpretability metrics
    """
    # Select router class
    router_cls_map = {
        'sinkhorn': DifferentiableRouter,
        'gumbel-ste': GumbelSTERouter,
        'reinmax': ReinMaxRouter,
    }
    if router_type not in router_cls_map:
        raise ValueError(f"Unknown router_type: {router_type}. "
                        f"Choose from {list(router_cls_map.keys())}")
    router_cls = router_cls_map[router_type]

    print(f"\n[{router_type}] Initializing with seed={seed}")
    key = jax.random.PRNGKey(seed)

    # Create datasets
    print(f"[{router_type}] Creating datasets (train={n_train}, test={n_test})")
    key, train_key, test_key = jax.random.split(key, 3)
    a_train, b_train, mode_train, target_train = create_mode_switching_dataset(n_train, train_key)
    a_test, b_test, mode_test, target_test = create_mode_switching_dataset(n_test, test_key)

    # Initialize model with selected router
    print(f"[{router_type}] Initializing model with {router_cls.__name__}")
    model = IntegrationDemoModel(frozen_masks=PHASE1_MASKS, router_cls=router_cls)
    key, init_key = jax.random.split(key)

    # Dummy input for initialization
    a_dummy = jnp.array([1.0])
    b_dummy = jnp.array([1.0])
    mode_dummy = jnp.array([1.0])

    # Initialize with RNG for Gumbel-STE router
    print(f"[{router_type}] Initializing parameters...")
    if router_type == 'gumbel-ste':
        params = model.init({'params': init_key, 'gumbel': jax.random.PRNGKey(42)},
                           a_dummy, b_dummy, mode_dummy)['params']
    else:
        params = model.init(init_key, a_dummy, b_dummy, mode_dummy)['params']

    # Optimizer
    print(f"[{router_type}] Setting up optimizer (lr={lr})")
    tx = optax.adam(lr)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    print(f"[{router_type}] Compiling training step (this may take 1-2 minutes)...")

    # Training loop
    loss_log = []
    acc_log = []
    interp_log = []
    spectral_log = []
    P_trajectory = []
    prev_P = None

    print(f"[{router_type}] Starting training: {n_steps} steps", flush=True)
    print(f"[{router_type}] Progress will be logged every 100 steps", flush=True)
    for step in range(n_steps):
        # Temperature annealing
        progress = step / n_steps
        temperature = temp_start * (temp_end / temp_start) ** progress

        # Sample batch
        key, batch_key = jax.random.split(key)
        idx = jax.random.randint(batch_key, (batch_size,), 0, n_train)

        a_batch = a_train[idx]
        b_batch = b_train[idx]
        mode_batch = mode_train[idx]
        target_batch = target_train[idx]

        # Update (each router type uses its own jitted train_step)
        if router_type == 'gumbel-ste':
            key, step_key = jax.random.split(key)
            state, loss = train_step_gumbel(state, a_batch, b_batch, mode_batch, target_batch, temperature, step_key)
        elif router_type == 'reinmax':
            # Anneal entropy_reg from 0.1 → 0.0 so exploration fades and routing sharpens
            entropy_reg = 0.1 * (1.0 - progress)
            state, loss = train_step_reinmax(state, a_batch, b_batch, mode_batch, target_batch, temperature, entropy_reg)
        else:
            state, loss = train_step_standard(state, a_batch, b_batch, mode_batch, target_batch, temperature)

        # Log
        if step % 100 == 0:
            # Evaluate on test set (use current training temperature for consistency)
            if router_type == 'gumbel-ste':
                key, eval_key = jax.random.split(key)
                output_test = model.apply({'params': state.params},
                                        a_test, b_test, mode_test,
                                        temperature=temperature,
                                        rngs={'gumbel': eval_key})
            else:
                output_test = model.apply({'params': state.params},
                                        a_test, b_test, mode_test,
                                        temperature=temperature)
            pred_test = jnp.sign(output_test)
            acc = (pred_test == target_test).mean()

            loss_log.append(float(loss))
            acc_log.append(float(acc))

            # Extract routing matrix and sign vector for interpretability
            # Get router logits directly from parameters
            P_logits = state.params['router']['P_logits']
            s_logits = state.params['router']['s_logits']

            # Compute routing matrix for interpretability analysis
            if router_type in ('sinkhorn', 'reinmax'):
                P = sinkhorn_rectangular(P_logits, n_iters=20, temperature=temperature)
            elif router_type == 'gumbel-ste':
                # For Gumbel-STE, use softmax without noise for evaluation
                P = jax.nn.softmax(P_logits / temperature, axis=-1)

            s = jnp.tanh(s_logits)

            # Compute interpretability metrics (pass features for mode-conditional analysis)
            features_test = jnp.stack([a_test[:100], b_test[:100], mode_test[:100]], axis=-1)
            interp_metrics = compute_interpretability_metrics(
                P, s, mode_test[:100], features_test, prev_P
            )
            interp_log.append(interp_metrics)
            # Keep P in JAX arrays - no numpy conversion
            P_trajectory.append(P)
            prev_P = P

            # Spectral analysis of routing matrix (SVD)
            svd_vals = jnp.linalg.svd(P, compute_uv=False)  # 3 singular values for 3x4 P
            svd_sorted = jnp.sort(svd_vals)[::-1]  # descending
            spectral_gap = float(1.0 - svd_sorted[1] / (svd_sorted[0] + 1e-10))
            # Spectral entropy from normalized squared singular values
            sv_sq = svd_sorted ** 2
            sv_dist = sv_sq / (sv_sq.sum() + 1e-10)
            spectral_entropy = -float(jnp.sum(sv_dist * jnp.log(sv_dist + 1e-10)))
            spectral_log.append({
                'singular_values': [float(v) for v in svd_sorted],
                'spectral_gap': spectral_gap,
                'spectral_entropy': spectral_entropy,
            })

            # Print progress
            print(f"[{router_type}] Step {step:4d}/{n_steps}: loss = {loss:.4f}, acc = {acc:.4f}, "
                  f"sparsity = {interp_metrics['routing_sparsity']:.3f}, "
                  f"Δ = {spectral_gap:.3f}, H_σ = {spectral_entropy:.3f}, "
                  f"temp = {temperature:.3f}", flush=True)

    # Final evaluation with quantization
    # Hard routing: k=1 (argmax)
    print("Running final evaluation...")
    if router_type == 'gumbel-ste':
        key, final_key = jax.random.split(key)
        final_output = model.apply({'params': state.params},
                                    a_test, b_test, mode_test,
                                    temperature=0.01,
                                    rngs={'gumbel': final_key})
    else:
        final_output = model.apply({'params': state.params},
                                    a_test, b_test, mode_test,
                                    temperature=0.01)  # Very low temp → hard routing
    final_pred = jnp.sign(final_output)
    final_acc = (final_pred == target_test).mean()

    print(f"\n[{router_type}] ===== FINAL RESULTS =====", flush=True)
    print(f"[{router_type}] Final Test Accuracy: {final_acc:.4f}", flush=True)
    if interp_log:
        final_interp = interp_log[-1]
        print(f"[{router_type}] Final Routing Sparsity: {final_interp['routing_sparsity']:.4f}", flush=True)
        print(f"[{router_type}] Final Expert Specialization: {final_interp['expert_specialization']:.4f}", flush=True)
        print(f"[{router_type}] Final Decision Stability: {final_interp['decision_stability']:.4f}", flush=True)
        print(f"[{router_type}] Final Sign Coherence: {final_interp['sign_coherence']:.4f}", flush=True)
    if spectral_log:
        final_spectral = spectral_log[-1]
        print(f"[{router_type}] Final Singular Values: {final_spectral['singular_values']}", flush=True)
        print(f"[{router_type}] Final Spectral Gap (Δ): {final_spectral['spectral_gap']:.4f}", flush=True)
        print(f"[{router_type}] Final Spectral Entropy (H_σ): {final_spectral['spectral_entropy']:.4f}", flush=True)

    return {
        'router_type': router_type,
        'n_train': n_train,
        'n_test': n_test,
        'n_steps': n_steps,
        'lr': lr,
        'seed': seed,
        'loss_trajectory': loss_log,
        'accuracy_trajectory': acc_log,
        'final_accuracy': float(final_acc),
        'interpretability_trajectory': interp_log,
        'spectral_trajectory': spectral_log,
        'P_trajectory': [jnp.array(p).tolist() for p in P_trajectory],
        'timestamp': datetime.now().isoformat(),
    }


def compute_convergence_speed(results: dict, threshold: float = 0.95) -> int:
    """
    Compute number of steps to reach threshold accuracy.

    Args:
        results: Results dict with 'accuracy_trajectory'
        threshold: Accuracy threshold (default 0.95 = 95%)

    Returns:
        steps: Number of steps to reach threshold (or n_steps if not reached)
    """
    acc_trajectory = results['accuracy_trajectory']
    for i, acc in enumerate(acc_trajectory):
        if acc >= threshold:
            return i * 100  # Each log point is 100 steps
    return results['n_steps']  # Never reached threshold


def run_routing_comparison(
    n_train: int = 10000,
    n_test: int = 5000,
    n_steps: int = 5000,
    batch_size: int = 128,  # Match Phase 3 to avoid OOM
    lr: float = 1e-2,
    seed: int = 0,
):
    """
    Run all 3 routing types and compare performance + interpretability.

    Returns:
        all_results: Dict mapping router_type → results
        comparison: Summary comparison dict
    """
    router_types = ['sinkhorn', 'gumbel-ste', 'reinmax']
    all_results = {}

    for router_type in router_types:
        print(f"\n{'='*60}")
        print(f"Running: {router_type.upper()}")
        print(f"{'='*60}")

        results = run_integration_demo(
            n_train=n_train,
            n_test=n_test,
            n_steps=n_steps,
            batch_size=batch_size,
            lr=lr,
            seed=seed,
            router_type=router_type
        )
        all_results[router_type] = results

    # Compute comparative statistics
    comparison = {
        'router_types': router_types,
        'final_accuracy': {rt: all_results[rt]['final_accuracy'] for rt in router_types},
        'convergence_speed': {rt: compute_convergence_speed(all_results[rt]) for rt in router_types},
        'final_interpretability': {
            rt: all_results[rt]['interpretability_trajectory'][-1]
            for rt in router_types
        },
    }

    return all_results, comparison


def main():
    print("Integration Demo: Routing Mechanism Comparison")
    print("=" * 60)
    print("Comparing: Sinkhorn, Gumbel-STE, ReinMax")
    print("=" * 60)

    # Run all 3 routing types
    all_results, comparison = run_routing_comparison(
        n_train=10000,
        n_test=5000,
        n_steps=5000,
        batch_size=128,  # Match Phase 3 to avoid OOM
        lr=1e-2,
        seed=0,
    )

    # Save results
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save individual results
    for router_type, results in all_results.items():
        output_path = output_dir / f'integration_demo_{router_type}.json'
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved: {output_path}")

    # Save comparison summary
    comparison_path = output_dir / 'routing_comparison_summary.json'
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"Saved: {comparison_path}")

    # Print summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    for router_type in ['sinkhorn', 'gumbel-ste', 'reinmax']:
        acc = comparison['final_accuracy'][router_type]
        speed = comparison['convergence_speed'][router_type]
        interp = comparison['final_interpretability'][router_type]
        print(f"\n{router_type.upper()}:")
        print(f"  Final Accuracy: {acc:.4f}")
        print(f"  Steps to 95%: {speed}")
        print(f"  Routing Sparsity: {interp['routing_sparsity']:.4f}")
        print(f"  Expert Specialization: {interp['expert_specialization']:.4f}")
        print(f"  Sign Coherence: {interp['sign_coherence']:.4f}")


if __name__ == '__main__':
    main()
