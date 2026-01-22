# Paper Reproduction Guide

This guide explains how to reproduce all experimental results from the paper "Differentiable Logic Synthesis: Spectral Coefficient Selection via Sinkhorn-Constrained Composition" ([arXiv:2601.13953](https://arxiv.org/abs/2601.13953)).

## Quick Start

**Minimal CPU test** (verify installation):
```bash
python3 tests/test_phase1_smoke.py --cpu
# Should complete in <10 seconds and show: "✓ All smoke tests passed!"
```

**Full Phase 1 training** (reproduce Table 2):
```bash
python3 boolean_fourier/phase1/train_phase1_fixed.py
# Runtime: ~30 minutes on CPU, ~5 minutes on GPU
# Expected: 100% accuracy on all 4 operations (XOR, AND, OR, IMPLIES)
```

## System Requirements

### Minimum (CPU-only):
- Python 3.9+
- 8GB RAM
- 30 minutes runtime for Phase 1

### Recommended (with GPU):
- Python 3.9+
- NVIDIA GPU with 8GB+ VRAM (e.g., RTX 3060)
- CUDA 11.8+
- 16GB RAM
- 5-40 minutes runtime per phase

### Installation

```bash
# Clone repository
git clone https://github.com/gogipav14/spectral-llm.git
cd spectral-llm

# Install dependencies
pip install -r requirements.txt

# Verify installation
python3 tests/test_phase1_smoke.py --cpu
```

## Paper Results Reproduction

### Table 2: Phase 1 Ternary Masks (n=2, 4 binary operations)

**Command:**
```bash
python3 boolean_fourier/phase1/train_phase1_fixed.py
```

**Expected Output:**
- XOR mask: `[0, 0, 0, ±1]` (75% sparse)
- AND mask: `[±1, ±1, ±1, ∓1]` (0% sparse)
- OR mask: `[∓1, ±1, ±1, ±1]` (0% sparse)
- IMPLIES mask: `[∓1, ∓1, ±1, ∓1]` (0% sparse)
- Test accuracy: 100% on all operations

**Implementation Details:**
- Architecture: `SoftLogicLayer` with 4 learnable ternary masks
- Training: Soft-ternary annealing with temperature decay (1.0 → 0.01)
- Loss: Hamming loss + ternary attractor regularization + mask-specific regularization
- Steps: 5000 per operation, sequential training
- Key file: `boolean_fourier/phase1/train_phase1_fixed.py`

**Result Location:** Results are printed to stdout (no JSON saved for Phase 1)

**Troubleshooting:**
- If accuracy < 100%, check that temperature annealing reaches 0.01
- XOR purity should be >0.95 (all weight in `ab` coefficient)
- Ternary distance should decrease to <0.01

---

### Table 3: Phase 2 Masks (n=2, 16 operations with routing)

**Command:**
```bash
python3 boolean_fourier/phase2/train_phase2_all16.py
```

**Expected Output:**
- All 16 Boolean operations learned with 100% accuracy
- Routing matrices learn near-identity (drift ≈ 0.0)
- Average sparsity: ~50%

**Result Location:** `boolean_fourier/phase2/results/phase2_all16_results.json`

**Key Implementation:**
- Architecture: Manifold-constrained routing with sign modulation
- Training: Sinkhorn iteration for doubly stochastic constraints
- Identity initialization for routing matrix P

---

### Table 5: Phase 3 GD Training (n=3, gradient descent baseline)

**Expected Accuracy:** ~76% (gradient descent alone cannot find optimal)

**Pre-computed Results:** `boolean_fourier/checkpoints/phase3_full_basis/phase3_full_basis_results.json`

**To re-run (optional):**
```bash
python3 boolean_fourier/phase3/train_phase3_full_basis.py
```

**Note:** This demonstrates the limitation of gradient descent - enumeration (Table 6) is needed for 100% accuracy.

---

### Table 6: Phase 3 Optimal Masks (n=3, exhaustive enumeration)

**Expected Accuracy:** 100% on all 10 three-variable operations

**Pre-computed Results:** `boolean_fourier/checkpoints/phase3_final/phase3_final_results.json`

**Mask Summary:**
- Basis dimension: 8 (Walsh-Fourier for 3 variables)
- Operations: parity_3, majority_3, and_3, or_3, etc.
- Average sparsity: 41.25%
- Method: Exhaustive search over 3^8 = 6561 ternary configurations

**To verify results:**
```bash
python3 boolean_fourier/phase3/validate_phase3_final.py
```

**Result File:**
```json
{
  "operation": "parity_3",
  "mask": [-1, 0, 0, 0, 0, 0, 0, 1],
  "accuracy": 1.0,
  "sparsity": 0.75
}
```

---

### Table 7: Phase 4 Spectral Synthesis (n=4, Walsh-Hadamard + MCMC)

**Expected Accuracy:** 100% on 10 four-variable operations

**Pre-computed Results:** `boolean_fourier/phase4/checkpoints/phase4_synthesis/phase4_synthesis_results.json`

**To re-run:**
```bash
python3 boolean_fourier/phase4/spectral_synthesis_4var.py
```

**Implementation:**
- Basis dimension: 16 (Walsh-Fourier for 4 variables)
- Method: Monte Carlo coefficient estimation → Ternary quantization → BlackJAX MCMC refinement
- Operations: xor_4, and_4, majority_4, threshold_3of4, etc.
- Average sparsity: 36%
- Runtime: ~40 minutes (uses probabilistic search, not exhaustive)

**Verification:**
```bash
python3 boolean_fourier/phase4/validate_phase4_final.py
```

---

### Table 8: Phase 5 Throughput Benchmarks

**Pre-computed Results:** `boolean_fourier/phase5/results/fwht_benchmark_results.json`

**To re-run:**
```bash
python3 boolean_fourier/phase5/benchmark_fwht.py
```

**Expected Results:**
- FWHT (n=28): ~1.4B coefficients/s on GPU
- Boolean ops: ~10,959 MOps/s on GPU
- Ternary inference: ~68 MOps/s on CPU (INT8)

**Result File:**
```json
{
  "device": "GPU",
  "operation": "FWHT",
  "n": 28,
  "throughput_coeffs_per_sec": 1400000000,
  "batch_size": 64
}
```

---

## Figure Reproduction

### Figure 3: XOR Learning Dynamics

**Generated during:** Phase 1 training

**To reproduce:**
```bash
python3 boolean_fourier/phase1/train_phase1_fixed.py
```

**Key Metrics to Plot:**
- XOR purity over training steps (should reach >0.95)
- Mask coefficients over time (should converge to `[0, 0, 0, 1]`)
- Temperature annealing curve (1.0 → 0.01)

**Data:** Printed to stdout during training (see "XOR purity" metric)

---

## Diagnostics (Optional v2 Experiments)

### Phase 1 v2: Jaccard + Eigenspectrum Diagnostics

**Command:**
```bash
python3 boolean_fourier/phase1/phase1_v2_diagnostics.py
```

**Purpose:** Show that GD learns the correct support topology early

**Expected Output:**
- Jaccard trajectories → 1.0 quickly for all operations
- Eigenspectrum shows learning collapses to rank-1 subspace for XOR
- Result: `boolean_fourier/phase1/results/v2_phase1_jaccard_eigenspace.json`

---

### Phase 4 v2: Warm-Start Experiment

**Command:**
```bash
python3 boolean_fourier/phase4/phase4_v2_warmstart_jaccard.py
```

**Purpose:** Prove GD warm-start reduces MCMC steps

**Expected Output:**
- Random init → MCMC: ~X steps to 100%
- WHT init → MCMC: ~Y steps (Y < X)
- GD warm-start → MCMC: ~Z steps (Z < Y)
- Result: `boolean_fourier/phase4/results/v2_phase4_warmstart_jaccard.json`

---

## Hierarchical Composition (Phase 5)

### 64-bit Adder Verification

**Pre-computed Results:** `boolean_fourier/phase5/results/circuit_composition_results.json`

**To re-run:**
```bash
python3 boolean_fourier/phase5/test_hierarchical_composition.py
```

**Expected:**
- 64-bit full adder composed from learned primitives
- Verified on 1M random samples
- 100% accuracy

---

## Troubleshooting

### JAX GPU not detected

```bash
# Check JAX installation
python3 -c "import jax; print(jax.devices())"

# Should show: [GpuDevice(id=0)] or [CpuDevice(id=0)]

# If CPU only, install JAX with GPU support:
pip install --upgrade "jax[cuda12]"
```

### Out of memory (GPU)

Reduce batch size in configuration:
```python
# In train_phase1_fixed.py, line 48:
BATCH_SIZE = 64  # Default: 128
```

### Slow training (CPU)

Expected runtimes on CPU:
- Phase 1: ~30 minutes
- Phase 3: ~1 hour (GD baseline)
- Phase 4: ~2 hours (MCMC search)

Use GPU for faster training (5-10x speedup).

### Results don't match paper exactly

Small numerical differences (<1%) are expected due to:
- Random initialization seeds
- JAX version differences
- GPU vs CPU execution

Accuracy should always be 100% for optimal masks.

---

## Citation

If you use this code or reproduce results, please cite:

```bibtex
@article{pavlov2026differentiable,
  title={Differentiable Logic Synthesis: Spectral Coefficient Selection via Sinkhorn-Constrained Composition},
  author={Pavlov, Gorgi},
  journal={arXiv preprint arXiv:2601.13953},
  year={2026}
}
```

---

## Additional Resources

- **Full experiment guide:** [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)
- **Architecture overview:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **Installation help:** [docs/SETUP.md](docs/SETUP.md)
- **Paper:** [arXiv:2601.13953](https://arxiv.org/abs/2601.13953)
- **Issues:** [GitHub Issues](https://github.com/gogipav14/spectral-llm/issues)

---

## Contact

For questions about reproduction:
- Open an issue: https://github.com/gogipav14/spectral-llm/issues
- Email: gorgipavlov@gmail.com
