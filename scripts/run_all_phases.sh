#!/bin/bash

# Run All Phase Experiments (v1 + v2)
# =====================================
#
# This script reproduces all results from the Boolean Fourier Logic paper.
# Runs 8 experiments total: 4 phases × 2 versions (v1: baseline, v2: diagnostics)
#
# Estimated total runtime: 3-4 hours (GPU recommended)
# Output: JSON files in boolean_fourier/phase*/results/

set -e  # Exit on error

echo "=========================================="
echo "Running All Phase Experiments"
echo "=========================================="
echo ""
echo "This will run 8 experiments (4 phases × 2 versions)"
echo "Estimated total time: 3-4 hours"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -d "boolean_fourier" ]; then
    echo "Error: Must run from spectral-llm root directory"
    exit 1
fi

# Function to run and time an experiment
run_experiment() {
    local phase=$1
    local script=$2
    local desc=$3

    echo ""
    echo -e "${BLUE}=========================================="
    echo -e "Phase ${phase}: ${desc}"
    echo -e "==========================================${NC}"
    echo "Running: python3 ${script}"
    echo ""

    start_time=$(date +%s)
    python3 "${script}"
    end_time=$(date +%s)

    elapsed=$((end_time - start_time))
    echo ""
    echo -e "${GREEN}✓ Completed in ${elapsed}s${NC}"
}

# ====================
# Phase 1: n=2 Binary Logic
# ====================

run_experiment "1 v1" \
    "boolean_fourier/phase1/train_phase1_fixed.py" \
    "GD Training (n=2, 4-dim basis)"

run_experiment "1 v2" \
    "boolean_fourier/phase1/phase1_v2_diagnostics.py" \
    "Jaccard + Eigenspectrum Diagnostics (n=2)"

# ====================
# Phase 2: Temporal Routing (16 ops)
# ====================

run_experiment "2 v1" \
    "boolean_fourier/phase2/train_phase2_all16.py" \
    "All 16 Operations with mHC Routing"

run_experiment "2 v2" \
    "boolean_fourier/phase2/phase2_v2_routing_diagnostics.py" \
    "Routing Diagnostics (drift, entropy, permutation)"

# ====================
# Phase 3: n=3 Three-Variable Logic
# ====================

run_experiment "3 v1" \
    "boolean_fourier/phase3/train_phase3_full_basis.py" \
    "Full 8-dim Basis GD Training (n=3)"

run_experiment "3 v2" \
    "boolean_fourier/phase3/phase3_v2_jaccard_eigenspace.py" \
    "Jaccard + Eigenspectrum (shows GD learns topology)"

# ====================
# Phase 4: n=4 Four-Variable Logic
# ====================

run_experiment "4 v1" \
    "boolean_fourier/phase4/spectral_synthesis_4var.py" \
    "Spectral Synthesis (WHT + MCMC)"

run_experiment "4 v2" \
    "boolean_fourier/phase4/phase4_v2_warmstart_jaccard.py" \
    "Warm-Start Experiment (4 conditions)"

# ====================
# Summary
# ====================

echo ""
echo "=========================================="
echo "All Experiments Complete!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - boolean_fourier/phase1/results/"
echo "  - boolean_fourier/phase2/results/"
echo "  - boolean_fourier/phase3/results/"
echo "  - boolean_fourier/phase4/results/"
echo ""
echo "Key result files:"
echo "  Phase 1 v1: phase1_fixed_results.json"
echo "  Phase 1 v2: v2_phase1_jaccard_eigenspace.json"
echo "  Phase 2 v1: phase2_all16_results.json (checkpoints)"
echo "  Phase 2 v2: v2_phase2_routing_diagnostics.json"
echo "  Phase 3 v1: phase3_full_basis_results.json"
echo "  Phase 3 v2: v2_phase3_jaccard_eigenspace.json"
echo "  Phase 4 v1: phase4_synthesis_results.json"
echo "  Phase 4 v2: v2_phase4_warmstart_jaccard.json"
echo ""
echo -e "${GREEN}Done!${NC}"
