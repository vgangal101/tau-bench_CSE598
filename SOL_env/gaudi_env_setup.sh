#!/bin/bash
# ========================================
# Gaudi Environment Setup Script for SOL
# ========================================
# Run this script on a SOL login node to create the tau-gaudi
# conda environment with all required packages.
#
# Usage: bash SOL_env/gaudi_env_setup.sh
#
# This is a ONE-TIME setup. After this, you can run experiments
# with: sbatch SOL_env/gaudi_experiment.sh
# ========================================

set -e  # Exit on error

echo "========================================"
echo "=== Gaudi Environment Setup for SOL ==="
echo "========================================"
echo ""

# Check we're on a login node (not compute)
if [[ $(hostname) == gaudi* ]]; then
    echo "ERROR: Run this script on a login node, not a Gaudi compute node"
    echo "Exit the interactive session and run from login node"
    exit 1
fi

# ========================================
# Step 1: Load Mamba
# ========================================
echo "=== Step 1: Loading Mamba ==="
module load mamba/latest
echo "Mamba loaded: $(which conda)"
echo ""

# ========================================
# Step 2: Create Conda Environment
# ========================================
ENV_NAME="tau-gaudi"
echo "=== Step 2: Creating Conda Environment: $ENV_NAME ==="

# Check if environment already exists
if conda env list | grep -q "^$ENV_NAME "; then
    echo "Environment $ENV_NAME already exists."
    read -p "Do you want to remove and recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n $ENV_NAME -y
    else
        echo "Using existing environment"
    fi
fi

# Create environment with Python 3.10 (required for Habana)
if ! conda env list | grep -q "^$ENV_NAME "; then
    echo "Creating new conda environment with Python 3.10..."
    mamba create -n $ENV_NAME python=3.10 -y
fi

# Activate
source activate $ENV_NAME
echo "Activated: $CONDA_DEFAULT_ENV"
echo "Python: $(which python) - $(python --version)"
echo ""

# ========================================
# Step 3: Set Up Working Directory
# ========================================
WORK_DIR="/scratch/$USER/gaudi-vllm-setup"
echo "=== Step 3: Setting Up Working Directory ==="
echo "Work directory: $WORK_DIR"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"
echo ""

# ========================================
# Step 4: Install Habana PyTorch
# ========================================
echo "=== Step 4: Installing Habana PyTorch ==="

# Habana provides pre-built wheels for their PyTorch fork
# Check https://vault.habana.ai/ui/native/gaudi-installer/ for latest

# For SynapseAI 1.23.0 (matching the driver version on SOL)
SYNAPSE_VERSION="1.23.0"
echo "Installing for SynapseAI version: $SYNAPSE_VERSION"

# Install PyTorch for Gaudi
pip install torch==2.6.0 --index-url https://vault.habana.ai/artifactory/api/pypi/gaudi-pypi/simple

# Install habana_frameworks
pip install habana_frameworks --index-url https://vault.habana.ai/artifactory/api/pypi/gaudi-pypi/simple

echo ""
echo "Verifying habana_frameworks installation..."
python -c "import habana_frameworks; print('habana_frameworks: OK')" || {
    echo "ERROR: habana_frameworks failed to import"
    echo "This may require running on a Gaudi node. Continuing anyway..."
}
echo ""

# ========================================
# Step 5: Install vLLM for Gaudi
# ========================================
echo "=== Step 5: Installing vLLM for Gaudi ==="

# Clone vllm-gaudi
if [ ! -d "vllm-gaudi" ]; then
    echo "Cloning vllm-gaudi..."
    git clone https://github.com/vllm-project/vllm-gaudi
else
    echo "vllm-gaudi already cloned, updating..."
    cd vllm-gaudi && git pull && cd ..
fi

cd vllm-gaudi

# Get the verified vLLM commit
echo "Getting verified vLLM commit..."
VLLM_COMMIT=$(git show "origin/vllm/last-good-commit-for-vllm-gaudi:VLLM_STABLE_COMMIT" 2>/dev/null || echo "")

if [ -z "$VLLM_COMMIT" ]; then
    echo "Could not get verified commit, using latest"
    VLLM_COMMIT="main"
fi
echo "vLLM commit: $VLLM_COMMIT"

cd ..

# Clone vLLM
if [ ! -d "vllm" ]; then
    echo "Cloning vLLM..."
    git clone https://github.com/vllm-project/vllm
else
    echo "vLLM already cloned"
fi

cd vllm
git fetch --all
git checkout $VLLM_COMMIT 2>/dev/null || git checkout main
echo "Checked out vLLM at: $(git rev-parse HEAD)"

# Install vLLM for empty platform (no CUDA dependency)
echo "Installing vLLM for empty platform..."
pip install -r <(sed '/^torch/d' requirements/build.txt) 2>/dev/null || pip install build wheel
VLLM_TARGET_DEVICE=empty pip install --no-build-isolation -e .

cd ..

# Install Gaudi plugin
echo "Installing vllm-gaudi plugin..."
cd vllm-gaudi
pip install -e .
cd ..

echo ""

# ========================================
# Step 6: Install tau-bench
# ========================================
echo "=== Step 6: Installing tau-bench ==="
TAU_BENCH_DIR="/scratch/$USER/tau-bench-project/tau-bench_CSE598"

if [ -d "$TAU_BENCH_DIR" ]; then
    cd "$TAU_BENCH_DIR"
    pip install -e .
    echo "tau-bench installed from: $TAU_BENCH_DIR"
else
    echo "WARNING: tau-bench not found at $TAU_BENCH_DIR"
    echo "You may need to install it manually"
fi
echo ""

# ========================================
# Step 7: Verify Installation
# ========================================
echo "=== Step 7: Verification ==="

echo "Python packages:"
pip list | grep -i -E 'torch|vllm|habana' | head -10

echo ""
echo "Testing imports..."
python -c "
import sys
print(f'Python: {sys.version}')

try:
    import torch
    print(f'PyTorch: {torch.__version__}')
except Exception as e:
    print(f'PyTorch: FAILED - {e}')

try:
    import vllm
    print(f'vLLM: {vllm.__version__}')
except Exception as e:
    print(f'vLLM: FAILED - {e}')

try:
    import habana_frameworks
    print('habana_frameworks: OK')
except Exception as e:
    print(f'habana_frameworks: FAILED - {e}')
    print('  (This may work on Gaudi compute nodes)')
"

echo ""
echo "========================================"
echo "=== Setup Complete ==="
echo "========================================"
echo ""
echo "Environment created: $ENV_NAME"
echo ""
echo "To use this environment:"
echo "  module load mamba/latest"
echo "  source activate $ENV_NAME"
echo ""
echo "To run experiments:"
echo "  sbatch /scratch/\$USER/tau-bench-project/tau-bench_CSE598/SOL_env/gaudi_experiment.sh"
echo ""
echo "Note: Some packages (habana_frameworks) may only work on Gaudi compute nodes."
echo ""
