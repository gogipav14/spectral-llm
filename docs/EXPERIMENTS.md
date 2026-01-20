# Experiment Reproduction Guide

This guide explains how to reproduce all results from the Boolean Fourier Logic paper.

## Overview

The paper presents 4 phases of experiments, each with 2 versions:
- **v1**: Main training/synthesis experiments
- **v2**: Diagnostic experiments (Jaccard, eigenspectrum, routing analysis)

**Total experiments**: 8 (4 phases × 2 versions)
**Estimated total runtime**: 3-4 hours on GPU (NVIDIA A100 or equivalent)

## Quick Start: Run All Experiments

```bash
# Run all 8 experiments sequentially
bash scripts/run_all_phases.sh
```

This will execute all experiments and save results to `boolean_fourier/phase*/results/*.json`.

## Hardware Requirements

### Recommended Configuration
- **GPU**: NVIDIA A100 (40GB) or RTX 3090 (24GB)
- **RAM**: 32GB+
- **Storage**: 10GB free space
- **CUDA**: 11.8 or 12.x

### Minimum Configuration
- **GPU**: GTX 1080 Ti (11GB VRAM)
- **RAM**: 16GB
- **Note**: Experiments will run slower on minimum hardware

### CPU-Only
All experiments can run on CPU, but will be significantly slower (10-50x):
- Phase 1: ~5 minutes → ~50 minutes
- Phase 2: ~30 minutes → ~5 hours
- Phase 3: ~20 minutes → ~3 hours
- Phase 4: ~40 minutes → ~6 hours

## Phase 1: Binary Logic (n=2)

### Phase 1 v1: Gradient Descent Training

**Purpose**: Learn all 16 binary operations in 4-dimensional Boolean Fourier basis

**Command**:
```bash
cd boolean_fourier/phase1
python3 train_phase1_fixed.py
```

**Expected Output**:
- File: `results/phase1_fixed_results.json`
- Accuracy: **100%** for XOR, AND, OR, IMPLIES
- Sparsity: **50%** (2 non-zero coefficients per operation)
- Runtime: ~5 minutes (GPU), ~50 minutes (CPU)

**Key Result**: Gradient descent achieves perfect accuracy on n=2 operations with ternary quantization.

**Reproduces**:
- Table 2 (Phase 1 ternary masks)
- Figure 3 (XOR learning dynamics)

---

### Phase 1 v2: Jaccard + Eigenspectrum Diagnostics

**Purpose**: Show GD learns support topology early, even before reaching exact values

**Command**:
```bash
cd boolean_fourier/phase1
python3 phase1_v2_diagnostics.py
```

**Expected Output**:
- File: `results/v2_phase1_jaccard_eigenspace.json`
- Jaccard AUC: >0.8 for all operations
- Eigenspectrum: Top-3 modes explain >90% variance
- Runtime: ~5 minutes (GPU)

**Key Result**: GD identifies which coefficients matter (topology) before reaching exact values.

**Reproduces**:
- Figure 4 (Jaccard trajectories for Phase 1)
- Supplementary Figure S1 (eigenspectrum collapse)

---

## Phase 2: Temporal Routing (16 Operations)

### Phase 2 v1: All 16 Operations with mHC Routing

**Purpose**: Learn all 16 binary operations using differentiable routing on Birkhoff manifold

**Command**:
```bash
cd boolean_fourier/phase2
python3 train_phase2_all16.py
```

**Expected Output**:
- Files: `results/phase2_all16_results.json` + checkpoint files
- Accuracy: **100%** for all 16 operations
- Routing drift: <0.1 (stays near identity initialization)
- Runtime: ~30 minutes (GPU), ~5 hours (CPU)

**Key Result**: Manifold-constrained routing with column-sign modulation enables all 16 operations.

**Reproduces**:
- Table 3 (Phase 2 ternary masks for 16 operations)
- Figure 5 (routing matrix evolution)

---

### Phase 2 v2: Routing Diagnostics

**Purpose**: Analyze routing manifold constraint (mHC), entropy, and permutation-likeness

**Command**:
```bash
cd boolean_fourier/phase2
python3 phase2_v2_routing_diagnostics.py
```

**Expected Output**:
- File: `results/v2_phase2_routing_diagnostics.json`
- Drift: <0.1 (routing stays near identity)
- Column entropy: Decreases (converges to permutation-like)
- Max-entry per column: >0.9 (permutation-like structure)
- Runtime: ~30 minutes (GPU)

**Key Result**: Identity initialization keeps routing on manifold with minimal drift.

**Reproduces**:
- Figure 6 (routing drift over training)
- Figure 7 (column entropy evolution)
- Supplementary Table S1 (routing statistics)

---

## Phase 3: Three-Variable Logic (n=3)

### Phase 3 v1: Full 8-Dimensional Basis GD Training

**Purpose**: Train on 10 three-variable operations in 8-dimensional Boolean Fourier basis

**Command**:
```bash
cd boolean_fourier/phase3
python3 train_phase3_full_basis.py
```

**Expected Output**:
- File: `results/phase3_full_basis_results.json`
- Accuracy: ~76% (GD doesn't reach 100% for n=3)
- Best seed: ~81%
- Runtime: ~20 minutes (GPU), ~3 hours (CPU)

**Key Result**: GD alone insufficient for n≥3; discrete refinement needed.

**Reproduces**:
- Table 5 (GD training results for Phase 3)
- Shows motivation for hybrid synthesis approach

**Note**: Optimal masks (100% accuracy) are found via exhaustive enumeration and saved in `checkpoints/phase3_final/phase3_final_results.json`.

---

### Phase 3 v2: Jaccard + Eigenspace (Shows GD Learns Topology)

**Purpose**: Show GD learns which coefficients matter, even when accuracy plateaus at 76%

**Command**:
```bash
cd boolean_fourier/phase3
python3 phase3_v2_jaccard_eigenspace.py
```

**Expected Output**:
- File: `results/v2_phase3_jaccard_eigenspace.json`
- Jaccard AUC: >0.6 (even though accuracy ≈76%)
- Eigenspectrum: Top-5 modes explain >85% variance
- Runtime: ~20 minutes (GPU)

**Key Result**: GD learns support topology (which coefficients) even when it doesn't reach exact values.

**Reproduces**:
- Figure 8 (Jaccard trajectories for Phase 3)
- Figure 9 (eigenspectrum shows spectral compression)
- Key finding: "Topology first, definition second"

---

## Phase 4: Four-Variable Logic (n=4)

### Phase 4 v1: Spectral Synthesis (WHT + MCMC)

**Purpose**: Use Walsh-Hadamard Transform + MCMC refinement for n=4 operations

**Command**:
```bash
cd boolean_fourier/phase4
python3 spectral_synthesis_4var.py
```

**Expected Output**:
- File: `results/phase4_synthesis_results.json`
- Accuracy: **100%** for all 10 four-variable operations
- Sparsity: ~36% (average)
- Runtime: ~40 minutes (GPU), ~6 hours (CPU)

**Key Result**: Spectral synthesis (WHT + ternary quantization) achieves 100% for n=4.

**Reproduces**:
- Table 6 (Phase 4 ternary masks)
- Figure 10 (spectral synthesis pipeline)

---

### Phase 4 v2: Warm-Start Experiment (4 Conditions)

**Purpose**: Prove GD provides measurable value for MCMC initialization

**Command**:
```bash
cd boolean_fourier/phase4
python3 phase4_v2_warmstart_jaccard.py
```

**Expected Output**:
- File: `results/v2_phase4_warmstart_jaccard.json`
- Conditions tested:
  - (A) Random init → MCMC steps to 100%
  - (B) WHT-threshold init → MCMC steps to 100%
  - (C) GD warm-start → MCMC steps to 100%
  - (D) WHT→GD → MCMC steps to 100%
- Key metric: Condition (C) or (D) should require fewer MCMC steps than (A)
- Runtime: ~40 minutes (GPU)

**Key Result**: GD warm-start reduces MCMC steps vs random initialization.

**Reproduces**:
- Figure 11 (bar chart: MCMC steps by initialization)
- Table 7 (warm-start comparison)
- Addresses "Why not just SAT/MCMC?" critique

---

## Phase 5: Scaling Analysis

### Fast Walsh-Hadamard Transform Benchmark

**Purpose**: Show throughput for large n (n=28, 2^28 coefficients)

**Command**:
```bash
cd boolean_fourier/phase5
python3 benchmark_fwht.py
```

**Expected Output**:
- File: `results/fwht_benchmark_results.json`
- GPU throughput: **1.4 billion coefficients/second** (A100)
- Boolean ops: **10,959 MOps/s**
- Runtime: ~5 minutes

**Reproduces**:
- Table 8 (Phase 5 throughput benchmarks)
- Figure 12 (throughput by dimension n)

---

## Result Files and Locations

After running all experiments, results are saved to:

```
boolean_fourier/
├── phase1/results/
│   ├── phase1_fixed_results.json          # Phase 1 v1
│   └── v2_phase1_jaccard_eigenspace.json  # Phase 1 v2
├── phase2/results/
│   ├── phase2_all16_results.json          # Phase 2 v1
│   └── v2_phase2_routing_diagnostics.json # Phase 2 v2
├── checkpoints/
│   ├── phase3_final/
│   │   └── phase3_final_results.json      # Phase 3 optimal masks (100%)
│   ├── phase3_full_basis/
│   │   └── phase3_full_basis_results.json # Phase 3 v1 (GD training)
│   └── phase3_init_from_known/
│       └── phase3_init_known_results.json # Phase 3 init experiment
├── phase3/results/
│   └── v2_phase3_jaccard_eigenspace.json  # Phase 3 v2 (when run)
├── phase4/checkpoints/
│   ├── phase4_synthesis/
│   │   └── phase4_synthesis_results.json  # Phase 4 v1
│   └── phase4_final/
│       └── phase4_final_results.json      # Phase 4 final validated
├── phase4/results/
│   └── v2_phase4_warmstart_jaccard.json   # Phase 4 v2 (when run)
└── phase5/results/
    ├── fwht_benchmark_results.json        # Phase 5 FWHT
    ├── circuit_composition_results.json   # Hierarchical circuits
    └── oracle_recovery_results.json       # Monte Carlo estimation
```

## Reproducing Specific Tables and Figures

### Paper Tables

| Table | Description | Command | Result File |
|-------|-------------|---------|-------------|
| Table 2 | Phase 1 ternary masks | `python3 boolean_fourier/phase1/train_phase1_fixed.py` | `phase1/results/phase1_fixed_results.json` |
| Table 3 | Phase 2 masks (16 ops) | `python3 boolean_fourier/phase2/train_phase2_all16.py` | `phase2/results/phase2_all16_results.json` |
| Table 5 | Phase 3 GD training | `python3 boolean_fourier/phase3/train_phase3_full_basis.py` | `checkpoints/phase3_full_basis/phase3_full_basis_results.json` |
| Table 6 (masks) | Phase 3 optimal masks | Enumeration (pre-computed) | `checkpoints/phase3_final/phase3_final_results.json` |
| Table 6 | Phase 4 synthesis | `python3 boolean_fourier/phase4/spectral_synthesis_4var.py` | `phase4/checkpoints/phase4_synthesis/phase4_synthesis_results.json` |
| Table 7 | Warm-start comparison | `python3 boolean_fourier/phase4/phase4_v2_warmstart_jaccard.py` | `phase4/results/v2_phase4_warmstart_jaccard.json` |
| Table 8 | Throughput benchmarks | `python3 boolean_fourier/phase5/benchmark_fwht.py` | `phase5/results/fwht_benchmark_results.json` |

### Paper Figures

| Figure | Description | Command | Result File |
|--------|-------------|---------|-------------|
| Figure 3 | XOR learning dynamics | `python3 boolean_fourier/phase1/train_phase1_fixed.py` | `phase1/results/phase1_fixed_results.json` |
| Figure 4 | Jaccard trajectories (n=2) | `python3 boolean_fourier/phase1/phase1_v2_diagnostics.py` | `phase1/results/v2_phase1_jaccard_eigenspace.json` |
| Figure 5 | Routing matrix evolution | `python3 boolean_fourier/phase2/train_phase2_all16.py` | `phase2/results/phase2_all16_results.json` |
| Figure 6-7 | Routing diagnostics | `python3 boolean_fourier/phase2/phase2_v2_routing_diagnostics.py` | `phase2/results/v2_phase2_routing_diagnostics.json` |
| Figure 8-9 | Jaccard + eigenspace (n=3) | `python3 boolean_fourier/phase3/phase3_v2_jaccard_eigenspace.py` | `phase3/results/v2_phase3_jaccard_eigenspace.json` |
| Figure 10 | Spectral synthesis pipeline | `python3 boolean_fourier/phase4/spectral_synthesis_4var.py` | `phase4/results/phase4_synthesis_results.json` |
| Figure 11 | Warm-start bar chart | `python3 boolean_fourier/phase4/phase4_v2_warmstart_jaccard.py` | `phase4/results/v2_phase4_warmstart_jaccard.json` |
| Figure 12 | Throughput scaling | `python3 boolean_fourier/phase5/benchmark_fwht.py` | `phase5/results/fwht_benchmark_results.json` |

## Deterministic Reproduction

All experiments use fixed random seeds for deterministic reproduction:

- Phase 1: `seed=42` (JAX PRNG key)
- Phase 2: `seed=42` (JAX PRNG key)
- Phase 3: `seed=0,1,2,3,4` (5 seeds for statistical significance)
- Phase 4: `seed=42` (JAX PRNG key + BlackJAX MCMC)

To reproduce with different seeds, modify the `SEED` variable at the top of each script.

## Troubleshooting

### Experiment Fails with OOM (Out of Memory)

**Phase 2 or 3:**
```bash
# Reduce batch size in the script
# Edit train_phase2_all16.py or train_phase3_full_basis.py
# Change: BATCH_SIZE = 256 → BATCH_SIZE = 128
```

**Phase 4 (MCMC):**
```bash
# Reduce number of parallel chains
# Edit spectral_synthesis_4var.py
# Change: N_CHAINS = 4 → N_CHAINS = 2
```

### GPU Not Detected

```bash
# Verify CUDA installation
nvidia-smi

# Reinstall JAX with correct CUDA version
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Verify JAX sees GPU
python3 -c "import jax; print(jax.devices())"
```

### Results Don't Match Paper Exactly

**Expected variations:**
- **Accuracy**: Should match within ±1% (due to random seeds)
- **Runtime**: Varies by hardware (times reported for A100)
- **Sparsity**: Should match exactly (deterministic quantization)

If results differ significantly:
1. Verify you're using the correct script version
2. Check that all dependencies match `requirements.txt`
3. Ensure you're using the correct CUDA version
4. Try running with `--seed 42` explicitly

### Experiment Hangs or Freezes

**Phase 4 MCMC:**
- MCMC can appear frozen during "burn-in" phase
- Check progress with `tail -f results/phase4_synthesis_results.json`
- Expected: Updates every 100 MCMC steps

**General:**
- Use `nvidia-smi` to verify GPU is being used (>0% utilization)
- Check system logs: `dmesg | tail`

## Performance Optimization

### Multi-GPU Support

Experiments currently use single GPU. To use multiple GPUs:

```python
# Add at top of training script
import jax
jax.config.update('jax_platform_name', 'gpu')
jax.config.update('jax_enable_x64', True)

# Shard across devices (for Phase 4 MCMC)
devices = jax.devices()
```

### CPU Optimization

For CPU-only systems:

```bash
# Enable XLA optimizations
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true"
export JAX_ENABLE_X64=True

# Run with optimized NumPy
pip install numpy[mkl]
```

## Validation

After running all experiments, verify results:

```bash
# Check that all result files exist
ls -lh boolean_fourier/phase*/results/*.json

# Quick validation script
python3 scripts/validate_results.py
```

Expected output:
```
✓ Phase 1 v1: 100% accuracy (4/4 operations)
✓ Phase 1 v2: Jaccard AUC > 0.8
✓ Phase 2 v1: 100% accuracy (16/16 operations)
✓ Phase 2 v2: Routing drift < 0.1
✓ Phase 3 v1: ~76% accuracy (GD baseline)
✓ Phase 3 v2: Jaccard AUC > 0.6
✓ Phase 4 v1: 100% accuracy (10/10 operations)
✓ Phase 4 v2: GD warm-start reduces MCMC steps
✓ All experiments complete!
```

## Contact and Issues

If you encounter issues reproducing results:

1. Check [Troubleshooting](#troubleshooting) section above
2. Search [GitHub Issues](https://github.com/YOUR_USERNAME/spectral-llm/issues)
3. Open a new issue with:
   - Your environment (`python --version`, `nvidia-smi` output)
   - Full error message
   - Steps to reproduce
   - Expected vs actual results

---

**Note**: Runtimes are estimated for NVIDIA A100 (40GB). Actual times will vary based on hardware.
