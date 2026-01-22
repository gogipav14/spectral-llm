# Contributing to Spectral-LLM

Thank you for your interest in contributing to this research project!

## Overview

This repository contains the research code for the paper "Differentiable Logic Synthesis: Spectral Coefficient Selection via Sinkhorn-Constrained Composition" ([arXiv:2601.13953](https://arxiv.org/abs/2601.13953)).

The primary purpose of this repository is **scientific reproducibility**. Contributions that improve reproducibility, fix bugs, or clarify documentation are welcome.

## How to Contribute

### Reporting Issues

If you encounter problems reproducing experiments:

1. Check [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for detailed instructions
2. Verify your environment matches the requirements (Python 3.9+, JAX 0.4.1+)
3. Open an issue with:
   - Which experiment/script failed
   - Your environment details (Python version, JAX version, GPU/CPU)
   - Full error message and stack trace
   - Steps you took to reproduce

### Bug Fixes

If you find a bug in the research code:

1. Open an issue describing the bug
2. Fork the repository
3. Create a branch: `git checkout -b fix/description-of-bug`
4. Fix the bug with minimal changes
5. Test that the fix doesn't break existing experiments
6. Submit a pull request referencing the issue

### Documentation Improvements

Improvements to documentation are always welcome:

- Clarify installation instructions
- Add missing dependencies
- Fix typos or broken links
- Improve code comments
- Add usage examples

### Running Tests

Before submitting changes:

```bash
# Run a quick smoke test (Phase 1 - fastest)
python boolean_fourier/phase1/train_phase1_fixed.py

# Verify the change doesn't break other phases
bash scripts/run_all_phases.sh
```

### Code Style

- Follow existing code style (we use standard Python conventions)
- Keep changes focused and minimal
- Add comments for non-obvious logic
- Update documentation if behavior changes

## What We're NOT Looking For

This is research code frozen to match the published paper. We generally **do not accept**:

- New features or experiments beyond the paper scope
- Major refactoring or architectural changes
- Performance optimizations that change results
- Changes to hyperparameters or training procedures

For such contributions, consider:
- Forking the repository for your own experiments
- Contributing to the [spectralbit](https://github.com/gogipav14/spectralbit) package (production-ready library)
- Opening a discussion to explore collaboration

## Questions?

- **General questions**: Open a GitHub Discussion
- **Bug reports**: Open an issue
- **Research collaboration**: Contact gorgipavlov@gmail.com

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for helping improve the reproducibility of this research!
