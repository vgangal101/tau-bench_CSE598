#!/bin/bash
#SBATCH --job-name=gaudi-check
#SBATCH --partition=gaudi
#SBATCH --qos=class_gaudi
#SBATCH --gres=gpu:hl225:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=00:15:00
#SBATCH --output=gaudi-check_%j.out
#SBATCH --error=gaudi-check_%j.err

# ========================================
# Gaudi Environment Diagnostic Script
# ========================================
# Run this FIRST to check what Gaudi software is available
# before attempting full experiments.
#
# Usage: sbatch SOL_env/gaudi_setup_check.sh
# ========================================

# Get script directory for log organization
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$SCRIPT_DIR/logs"

echo "========================================"
echo "=== Gaudi Environment Diagnostic ==="
echo "========================================"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Script Dir: $SCRIPT_DIR"
echo ""

# ========================================
# Step 1: Check Hardware / Devices
# ========================================
echo "=== Step 1: Hardware Detection ==="

echo "Checking for hl-smi in PATH..."
if command -v hl-smi &> /dev/null; then
    echo "hl-smi found at: $(which hl-smi)"
    echo ""
    hl-smi
else
    echo "hl-smi: NOT IN PATH"
fi
echo ""

echo "Searching for hl-smi on system..."
find /opt /usr/local /usr -name "hl-smi" 2>/dev/null | head -5 || echo "Not found"
echo ""

echo "Checking device files..."
echo "/dev/accel*:"
ls -la /dev/accel* 2>/dev/null || echo "  No /dev/accel devices found"
echo "/dev/hl*:"
ls -la /dev/hl* 2>/dev/null || echo "  No /dev/hl devices found"
echo ""

echo "Checking lspci for accelerators..."
lspci 2>/dev/null | grep -i -E 'habana|gaudi|accel|co-processor' | head -5 || echo "No accelerators found in lspci"
echo ""

# ========================================
# Step 2: Check /opt Directory
# ========================================
echo "=== Step 2: /opt Directory Contents ==="
echo "Looking for Habana installation..."
ls -la /opt/ 2>/dev/null
echo ""

if [ -d "/opt/habanalabs" ]; then
    echo "Found /opt/habanalabs:"
    ls -la /opt/habanalabs/
fi

if [ -d "/opt/habana" ]; then
    echo "Found /opt/habana:"
    ls -la /opt/habana/
fi
echo ""

# ========================================
# Step 3: Check Environment Variables
# ========================================
echo "=== Step 3: Environment Variables ==="
echo "HABANA_VISIBLE_DEVICES: ${HABANA_VISIBLE_DEVICES:-<not set>}"
echo "HABANA_LOGS: ${HABANA_LOGS:-<not set>}"
echo "PT_HPU_LAZY_MODE: ${PT_HPU_LAZY_MODE:-<not set>}"
echo "LD_LIBRARY_PATH:"
echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -i habana || echo "  No habana paths in LD_LIBRARY_PATH"
echo ""

echo "All HABANA env vars:"
env | grep -i habana || echo "  None set"
echo ""

# ========================================
# Step 4: Check Modules
# ========================================
echo "=== Step 4: Available Modules ==="
echo "Searching for Habana/Gaudi/Intel modules..."
module avail 2>&1 | grep -i -E "habana|gaudi|hpu" || echo "No Habana/Gaudi modules found"
echo ""

echo "Currently loaded modules:"
module list 2>&1
echo ""

# ========================================
# Step 5: Check Python
# ========================================
echo "=== Step 5: System Python ==="

echo "Python3 location and version:"
which python3 2>&1
python3 --version 2>&1
echo ""

echo "Checking for habana_frameworks (system python):"
python3 -c "import habana_frameworks; print('habana_frameworks: FOUND')" 2>&1 || echo "habana_frameworks: NOT INSTALLED"
echo ""

echo "Checking pip packages for habana:"
pip3 list 2>/dev/null | grep -i -E 'habana|hpu|synapse' || echo "No habana packages found"
echo ""

# ========================================
# Step 6: Check Conda Environments
# ========================================
echo "=== Step 6: Conda Environments ==="

# Try to find conda
if command -v conda &> /dev/null; then
    echo "Conda found: $(which conda)"
    conda env list 2>&1 | head -20
else
    # Try loading mamba module
    module load mamba/latest 2>/dev/null
    if command -v conda &> /dev/null; then
        echo "Conda found after loading mamba: $(which conda)"
        conda env list 2>&1 | head -20
    else
        echo "Conda not available"
    fi
fi
echo ""

# ========================================
# Step 7: Container/Singularity Check
# ========================================
echo "=== Step 7: Container Options ==="

echo "Checking for Singularity/Apptainer..."
which singularity 2>&1 || which apptainer 2>&1 || echo "No container runtime found in PATH"
echo ""

echo "Singularity module:"
module avail 2>&1 | grep -i singularity || echo "No singularity module"
echo ""

# ========================================
# Summary
# ========================================
echo "========================================"
echo "=== Diagnostic Summary ==="
echo "========================================"

# Check what we found
HAS_HLSMI=false
HAS_DEVICES=false
HAS_HABANA_PY=false

command -v hl-smi &> /dev/null && HAS_HLSMI=true
[ -e /dev/accel0 ] || [ -e /dev/hl0 ] && HAS_DEVICES=true
python3 -c "import habana_frameworks" 2>/dev/null && HAS_HABANA_PY=true

echo "hl-smi available: $HAS_HLSMI"
echo "Device files exist: $HAS_DEVICES"
echo "habana_frameworks (Python): $HAS_HABANA_PY"
echo ""

if [ "$HAS_HLSMI" = true ] && [ "$HAS_DEVICES" = true ]; then
    echo "STATUS: Gaudi hardware appears to be available!"
    echo ""
    echo "Next step: Set up Python environment with vllm-gaudi"
    echo "See README_gaudi.md for instructions"
elif [ "$HAS_DEVICES" = true ]; then
    echo "STATUS: Gaudi devices exist but tools not in PATH"
    echo ""
    echo "Look for Habana installation in /opt and add to PATH"
else
    echo "STATUS: Gaudi software may not be installed on this cluster"
    echo ""
    echo "Contact cluster admins about Gaudi/Habana software availability"
fi

echo ""
echo "========================================"
echo "=== Diagnostic Complete ==="
echo "========================================"

# Move output files to logs directory
if [ -n "$SLURM_SUBMIT_DIR" ] && [ -n "$SLURM_JOB_ID" ]; then
    mv "$SLURM_SUBMIT_DIR/gaudi-check_${SLURM_JOB_ID}.out" "$SCRIPT_DIR/logs/" 2>/dev/null || true
    mv "$SLURM_SUBMIT_DIR/gaudi-check_${SLURM_JOB_ID}.err" "$SCRIPT_DIR/logs/" 2>/dev/null || true
fi
