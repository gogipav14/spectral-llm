# Boolean Fourier Logic

Differentiable logic synthesis via spectral coefficient selection with Sinkhorn-constrained routing.

## Overview

This project demonstrates that exact Boolean logic can be learned via gradient descent by:
1. Grounding the architecture in Boolean Fourier analysis
2. Learning ternary spectral coefficients {-1, 0, +1}
3. Using Sinkhorn-constrained routing with column-sign modulation

## Directory Structure

```
boolean_fourier/
├── phase1/                     # 2-variable logic (n=2, 4-dim basis)
│   ├── train_phase1_*.py       # Training scripts
│   ├── binary_logic_layer.py   # Core layer implementation
│   ├── logic_dataset.py        # Data generation
│   ├── ternary_ops.py          # Ternary operations
│   ├── walsh_basis.py          # Walsh-Hadamard basis
│   └── validate_logic.py       # Validation utilities
├── phase2/                     # 16 temporal operations
│   ├── train_phase2_*.py       # Training scripts
│   ├── temporal_dataset*.py    # Temporal data generation
│   └── hierarchical_r.py       # Hierarchical routing
├── phase3/                     # 3-variable operations (n=3, 8-dim basis)
│   ├── train_phase3_*.py       # Training scripts
│   ├── boolean_fourier_3var.py # 3-var implementation
│   ├── logic3_dataset.py       # 3-var data generation
│   └── validate_phase3_final.py
├── phase4/                     # 4-variable operations (n=4, 16-dim basis)
│   ├── spectral_synthesis.py   # MCMC-based synthesis
│   └── spectral_synthesis_4var.py
├── inference/                  # Inference implementations
│   ├── npu_inference.py        # INT8 inference (corrected masks)
│   └── flax_boolean_fourier.py # Flax/JAX implementation
├── checkpoints/                # Saved results
│   ├── phase3_final/           # 100% accuracy results
│   ├── phase3_full_basis/
│   └── benchmark/              # Throughput benchmarks
└── README.md
```

## Key Results

### Phase 1: Two-Variable Operations (n=2)

**Basis**: `[1, a, b, ab]` (4-dimensional)

| Operation | Mask | Formula |
|-----------|------|---------|
| XOR | `[0, 0, 0, 1]` | sign(ab) |
| AND | `[-1, 1, 1, 1]` | sign(-1 + a + b + ab) |
| OR | `[1, 1, 1, -1]` | sign(1 + a + b - ab) |
| IMPLIES | `[1, -1, 1, 1]` | sign(1 - a + b + ab) |

**Accuracy**: 100% on all 16 operations (10/10 seeds)

### Phase 3: Three-Variable Operations (n=3)

**Basis**: `[1, a, b, c, ab, ac, bc, abc]` (8-dimensional)

| Operation | Mask |
|-----------|------|
| parity_3 | `[-1, 0, 0, 0, 0, 0, 0, 1]` |
| majority_3 | `[-1, 0, 1, 1, 0, 0, 0, -1]` |
| and_3 | `[-1, 0, 0, 1, 0, 1, 1, 1]` |
| or_3 | `[-1, 1, 1, 1, -1, -1, -1, 1]` |
| xor_ab_xor_c | `[-1, 0, 0, 0, 0, 0, 0, 1]` |
| and_ab_or_c | `[-1, 0, 0, 1, -1, 1, 1, 0]` |
| or_ab_and_c | `[-1, 0, 1, 1, 1, 0, -1, -1]` |
| implies_ab_c | `[-1, 0, -1, 1, 0, 1, 0, 1]` |
| xor_and_ab_c | `[-1, -1, 0, 1, 0, 1, 1, -1]` |
| and_xor_ab_c | `[-1, 0, 0, 1, 1, 0, 0, -1]` |

**Accuracy**: 100.0% on all 10 operations (5 seeds)
**Sparsity**: 41.25% mean zero coefficients

### Benchmark Performance

| Backend | Throughput (MOps/s) | Notes |
|---------|---------------------|-------|
| JAX/GPU (RTX 3080) | 10,959.40 | Phase 3, batch=100k |
| OpenVINO/CPU | 202.53 | Phase 3, batch=100k |
| NumPy/CPU (INT8) | 36.93 | Phase 3, batch=100k |

## Usage

### Running Phase 1
```bash
cd boolean_fourier/phase1
python train_phase1_logic.py
```

### Running Phase 3
```bash
cd boolean_fourier/phase3
python train_phase3_full_basis.py
```

### Validating Masks
```bash
cd boolean_fourier/inference
python -c "from npu_inference import verify_phase1_correctness; verify_phase1_correctness()"
```

## Mathematical Foundation

All Boolean functions f: {-1, +1}^n → {-1, +1} have a unique Fourier expansion:

```
f(x) = sign(Σ_S ĉ(S) χ_S(x))
```

where χ_S(x) = ∏_{i∈S} x_i are the Walsh characters.

For n=2: basis = [1, a, b, ab]
For n=3: basis = [1, a, b, c, ab, ac, bc, abc]
For n=4: basis = 16 characters

The key insight: ternary coefficients {-1, 0, +1} suffice for exact Boolean logic,
enabling zero-loss quantization and single-cycle hardware inference.

## Citation

```bibtex
@article{pavlov2026boolean,
  title={Differentiable Logic Synthesis: Spectral Coefficient Selection via Sinkhorn-Constrained Composition},
  author={Pavlov, Gorgi},
  journal={arXiv preprint},
  year={2026}
}
```
