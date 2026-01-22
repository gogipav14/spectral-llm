#!/usr/bin/env python3
"""
Integration test for Phase 1 training (n=2 binary logic).

This test runs a minimal training loop to verify the full pipeline works.
Runs quickly (50 steps) to verify reproducibility without requiring GPU.
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax


def configure_device(device_type='cpu'):
    """Configure JAX to use specified device."""
    if device_type == 'cpu':
        jax.config.update('jax_platform_name', 'cpu')
        print(f"Using device: CPU")
    elif device_type == 'gpu':
        devices = jax.devices('gpu')
        if devices:
            print(f"Using device: GPU - {devices[0]}")
        else:
            print("GPU requested but not available, using CPU")
            jax.config.update('jax_platform_name', 'cpu')
    else:  # auto
        devices = jax.devices()
        cpu_devices = [d for d in devices if d.platform == 'cpu']
        if cpu_devices:
            jax.config.update('jax_platform_name', 'cpu')
            print(f"Using device: CPU (auto)")
        else:
            print(f"Using device: {devices[0]} (auto)")


class BooleanFourierLayer(nn.Module):
    """Simple Boolean Fourier layer for n=2."""
    training: bool = True

    @nn.compact
    def __call__(self, x):
        """
        x: (batch, 2) inputs in {-1, +1}
        returns: (batch,) outputs in {-1, +1} (or soft approximation during training)
        """
        # Boolean Fourier features: [1, a, b, ab]
        a, b = x[:, 0:1], x[:, 1:2]
        ones = jnp.ones_like(a)
        ab = a * b
        features = jnp.concatenate([ones, a, b, ab], axis=-1)  # (batch, 4)

        # Learnable ternary mask (initialized continuous, will quantize)
        w = self.param('w', nn.initializers.normal(0.1), (4,))

        # Compute output
        logits = features @ w  # (batch,)

        if self.training:
            # Use tanh for differentiable approximation during training
            return jnp.tanh(logits * 5.0)  # Scale to make it sharper
        else:
            # Use sign for evaluation
            return jnp.sign(logits)  # (batch,)


def create_xor_dataset():
    """Create XOR truth table dataset."""
    inputs = jnp.array([
        [-1, -1],
        [-1, +1],
        [+1, -1],
        [+1, +1],
    ], dtype=jnp.float32)

    # XOR: sign(ab) in {-1, +1} encoding
    outputs = jnp.array([+1, -1, -1, +1], dtype=jnp.float32)

    return inputs, outputs


def loss_fn(params, model_train, model_eval, inputs, targets):
    """MSE loss in {-1, +1} encoding."""
    # During training, use soft predictions
    predictions = model_train.apply(params, inputs)
    # MSE loss (predictions and targets both in range [-1, +1])
    loss = jnp.mean((predictions - targets) ** 2)

    # Accuracy (always use hard predictions for this)
    hard_predictions = model_eval.apply(params, inputs)
    correct = (hard_predictions == targets).sum()
    accuracy = correct / len(targets)

    return loss, accuracy


def quantize_to_ternary(w, threshold=0.5):
    """Quantize weights to {-1, 0, +1}."""
    w_ternary = jnp.where(jnp.abs(w) < threshold, 0, jnp.sign(w))
    return w_ternary


def train_xor(steps=50, learning_rate=1e-2, seed=42):
    """Train a simple XOR learner."""
    print(f"\nTraining XOR for {steps} steps...")

    # Create dataset
    inputs, targets = create_xor_dataset()
    print(f"  Dataset: {len(inputs)} examples")

    # Create models (one for training with soft outputs, one for eval with hard outputs)
    model_train = BooleanFourierLayer(training=True)
    model_eval = BooleanFourierLayer(training=False)

    key = jax.random.PRNGKey(seed)
    params = model_train.init(key, inputs)

    print(f"  Initial parameters: {params['params']['w']}")

    # Create optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    # Training loop
    @jax.jit
    def train_step(params, opt_state, inputs, targets):
        (loss, acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, model_train, model_eval, inputs, targets
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, acc

    for step in range(steps):
        params, opt_state, loss, acc = train_step(params, opt_state, inputs, targets)

        if step % 10 == 0 or step == steps - 1:
            print(f"  Step {step:3d}: loss={loss:.4f}, acc={acc:.2%}")

    # Final evaluation
    print(f"\n  Final parameters: {params['params']['w']}")

    # Quantize to ternary
    w_continuous = params['params']['w']
    w_ternary = quantize_to_ternary(w_continuous, threshold=0.5)
    print(f"  Quantized (ternary): {w_ternary}")

    # Test with ternary mask
    params_ternary = {'params': {'w': w_ternary}}
    final_loss, final_acc = loss_fn(params_ternary, model, inputs, targets, training=False)

    print(f"\n  Final accuracy (ternary): {final_acc:.2%}")

    return final_acc >= 1.0  # Should achieve 100% accuracy


def run_test(device_type='cpu'):
    """Run integration test."""
    print("=" * 60)
    print(f"Phase 1 Training Integration Test (device: {device_type.upper()})")
    print("=" * 60)

    configure_device(device_type)

    try:
        success = train_xor(steps=50)

        print("\n" + "=" * 60)
        if success:
            print("✓ Integration test passed!")
            print("\nPhase 1 training pipeline works correctly.")
            print("The model learned XOR with 100% accuracy.")
            return 0
        else:
            print("✗ Integration test failed!")
            print("Model did not achieve 100% accuracy on XOR.")
            return 1

    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Integration test for Phase 1 Boolean Fourier training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument('--cpu', action='store_true',
                              help='Force CPU execution')
    device_group.add_argument('--gpu', action='store_true',
                              help='Force GPU execution (if available)')

    args = parser.parse_args()

    if args.cpu:
        device_type = 'cpu'
    elif args.gpu:
        device_type = 'gpu'
    else:
        device_type = 'auto'

    return run_test(device_type)


if __name__ == '__main__':
    sys.exit(main())
