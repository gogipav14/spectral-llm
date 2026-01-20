# Spectral-LLM: Boolean Fourier Logic with Differentiable Routing

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
<!-- Add DOI badge after Zenodo registration:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
-->

Research code for the paper **"[Paper Title Here]"** by [Author Names].

## Overview

Learning precise Boolean logic via gradient descent remains challenging: neural networks converge to fuzzy approximations that degrade under quantization. We introduce **Hybrid Spectral Composition**, a pipeline combining differentiable optimization with discrete synthesis in the Walsh-Hadamard basis.

### Key Contributions

1. **Differentiable Boolean Logic** (n=2): Gradient descent achieves 100% accuracy on all 16 binary operations
2. **Hybrid Synthesis** (n=3): Gradient descent reaches 76%; exhaustive enumeration finds optimal ternary masks achieving 100% accuracy (39% sparsity)
3. **Spectral Synthesis** (n=4): Exact Walsh-Hadamard Transform with ternary quantization achieves 100% (36% sparsity)
4. **Manifold-Constrained Routing**: Column-sign modulation enables Boolean negation on doubly stochastic matrices

### Performance

- GPU (JAX): **10,959 MOps/s** peak throughput
- Ternary masks: **No floating-point arithmetic** required at inference
- Hierarchical composition: 64-bit adders, 128-bit comparators verified on millions of samples

## Installation

### Requirements

- Python 3.9+
- JAX 0.4.1+
- Flax 0.6.0+
- NumPy, SciPy
- BlackJAX (for Phase 4 MCMC)

### Quick Start

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/spectral-llm.git
cd spectral-llm

# Install dependencies
pip install -r requirements.txt

# Run Phase 1 example (XOR learning)
python boolean_fourier/phase1/train_phase1_fixed.py
```

## Repository Structure

```
spectral-llm/
├── boolean_fourier/          # Core research code
│   ├── phase1/               # n=2 binary logic (4-dim basis)
│   ├── phase2/               # Temporal routing (16 operations)
│   ├── phase3/               # n=3 three-variable logic (8-dim basis)
│   ├── phase4/               # n=4 four-variable logic (16-dim basis)
│   ├── phase5/               # Scaling analysis
│   ├── inference/            # NPU deployment
│   └── utils/                # Shared diagnostic utilities
├── paper/                    # LaTeX paper and figures
├── docs/                     # Documentation
│   ├── SETUP.md             # Installation guide
│   ├── EXPERIMENTS.md       # How to reproduce experiments
│   └── ARCHITECTURE.md      # System design overview
└── scripts/                  # Helper scripts
    └── run_all_phases.sh    # Run all 8 experiments (v1 + v2)
```

## Reproducing Experiments

See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for detailed instructions on reproducing all results.

### Quick Overview

**Phase 1 (n=2):**
```bash
# v1: Gradient descent training
python boolean_fourier/phase1/train_phase1_fixed.py

# v2: Jaccard + eigenspectrum diagnostics
python boolean_fourier/phase1/phase1_v2_diagnostics.py
```

**Phase 2 (Routing):**
```bash
# v1: All 16 operations with identity initialization
python boolean_fourier/phase2/train_phase2_all16.py

# v2: Routing diagnostics (mHC manifold constraint)
python boolean_fourier/phase2/phase2_v2_routing_diagnostics.py
```

**Phase 3 (n=3):**
```bash
# v1: Full 8-dim basis GD training
python boolean_fourier/phase3/train_phase3_full_basis.py

# v2: Jaccard + eigenspectrum (shows GD learns topology)
python boolean_fourier/phase3/phase3_v2_jaccard_eigenspace.py
```

**Phase 4 (n=4):**
```bash
# v1: Spectral synthesis (WHT + MCMC)
python boolean_fourier/phase4/spectral_synthesis_4var.py

# v2: Warm-start experiment (4 conditions)
python boolean_fourier/phase4/phase4_v2_warmstart_jaccard.py
```

**Run all 8 experiments:**
```bash
bash scripts/run_all_phases.sh
```

## Key Results

| Phase | n | Method | Accuracy | Sparsity | Time |
|-------|---|--------|----------|----------|------|
| 1 | 2 | Gradient Descent | 100% | 50% | ~5 min |
| 2 | 2 (16 ops) | mHC Routing | 100% | 50% | ~30 min |
| 3 | 3 | GD + Enumeration | 100% | 41% | ~20 min |
| 4 | 4 | WHT + MCMC | 100% | 36% | ~40 min |
| 5 | 28 | Fast WHT | - | - | 1.4B coeffs/s |

See the paper for detailed analysis and ablation studies.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{yourname2025spectral,
  title={[Paper Title]},
  author={[Author Names]},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## Related Projects

- **[spectralbit](https://github.com/YOUR_USERNAME/spectralbit)** - Production-ready Python package (`pip install spectralbit`)
- **[spectral-llm-experiments](https://github.com/YOUR_USERNAME/spectral-llm-experiments)** - Historical development iterations (archived)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Research supported by [Institution/Grant]
- Thanks to [Collaborators/Contributors]

## Contact

For questions or collaboration inquiries, please open an issue or contact [your email].

---

**Note:** This is research code. For production use, see the [spectralbit](https://github.com/YOUR_USERNAME/spectralbit) package.
