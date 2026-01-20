# Installation Guide

This guide covers installation and setup for the Spectral-LLM research code.

## Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA support (recommended for full experiments)
  - Minimum: GTX 1080 Ti (11GB VRAM)
  - Recommended: RTX 3090 / A100 (24GB+ VRAM)
- **CPU**: Multi-core CPU with AVX2 support
- **RAM**: 16GB minimum, 32GB+ recommended
- **Storage**: 10GB free space

### Software
- **Python**: 3.9 or higher
- **CUDA**: 11.8 or 12.x (for GPU acceleration)
- **Git**: For cloning the repository

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/spectral-llm.git
cd spectral-llm
```

### 2. Create Virtual Environment (Recommended)

Using `venv`:
```bash
python3 -m venv spectral_env
source spectral_env/bin/activate  # On Windows: spectral_env\Scripts\activate
```

Or using `conda`:
```bash
conda create -n spectral python=3.9
conda activate spectral
```

### 3. Install Dependencies

#### GPU Installation (Recommended)

```bash
# Install JAX with CUDA support
pip install --upgrade pip
pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install remaining dependencies
pip install -r requirements.txt
```

#### CPU-Only Installation

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note**: CPU-only installation will work but will be significantly slower for training experiments.

### 4. Verify Installation

```bash
python3 -c "import jax; print(f'JAX version: {jax.__version__}'); print(f'Devices: {jax.devices()}')"
```

Expected output (GPU):
```
JAX version: 0.4.20
Devices: [CudaDevice(id=0), ...]
```

Expected output (CPU):
```
JAX version: 0.4.20
Devices: [CpuDevice(id=0)]
```

## Quick Test

Run a quick Phase 1 test to verify everything works:

```bash
cd boolean_fourier/phase1
python3 train_phase1_fixed.py --test
```

This should complete in ~1 minute and verify that:
- JAX is working
- All imports are available
- Basic functionality is operational

## Troubleshooting

### JAX GPU Not Detected

If JAX doesn't detect your GPU:

```bash
# Check CUDA installation
nvidia-smi

# Reinstall JAX with explicit CUDA version
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### CUDA Version Mismatch

If you get CUDA version errors:

```bash
# For CUDA 11.x
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# For CUDA 12.x
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Out of Memory Errors

If you get OOM errors during training:

1. **Reduce batch size**: Edit the training script and reduce `BATCH_SIZE`
2. **Use gradient checkpointing**: Enable in config if available
3. **Use mixed precision**: JAX will do this automatically

### Import Errors

If you get import errors:

```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Check for conflicts
pip list | grep jax
```

## Optional: NPU Deployment (Phase 5)

For Intel NPU deployment experiments:

```bash
pip install openvino>=2023.2.0
pip install openvino-dev>=2023.2.0
```

## Development Setup

For development and running tests:

```bash
# Install development dependencies
pip install pytest black flake8 mypy

# Run tests
pytest boolean_fourier/tests/

# Format code
black boolean_fourier/

# Lint
flake8 boolean_fourier/
```

## Docker (Alternative)

A Dockerfile is provided for containerized execution:

```bash
# Build image
docker build -t spectral-llm .

# Run container
docker run --gpus all -it spectral-llm
```

## Next Steps

After successful installation:

1. Review [EXPERIMENTS.md](EXPERIMENTS.md) for reproduction instructions
2. Run the quick start example in the main README
3. Explore individual phase experiments

## Support

If you encounter issues:

1. Check the [troubleshooting section](#troubleshooting) above
2. Search [GitHub Issues](https://github.com/YOUR_USERNAME/spectral-llm/issues)
3. Open a new issue with:
   - Your environment details (`python --version`, `nvidia-smi`)
   - Full error message
   - Steps to reproduce
