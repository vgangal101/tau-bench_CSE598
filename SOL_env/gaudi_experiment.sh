#!/bin/bash
#SBATCH --job-name=tau-gaudi
#SBATCH --partition=gaudi
#SBATCH --qos=class_gaudi
#SBATCH --nodes=1
#SBATCH --gres=gpu:hl225:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=tau-gaudi_%j.out
#SBATCH --error=tau-gaudi_%j.err

# ========================================
# Intel Gaudi (HPU) Experiment Script
# ========================================
# This script runs tau-bench experiments on Intel Gaudi accelerators
# using the vllm-gaudi plugin.
#
# PREREQUISITES:
# 1. Install vllm-gaudi plugin (see setup instructions below)
# 2. Habana SynapseAI SDK must be installed on the cluster
# 3. Python environment with vLLM built for Gaudi
#
# SETUP INSTRUCTIONS (one-time):
#   # Clone and install vllm-gaudi
#   git clone https://github.com/vllm-project/vllm-gaudi
#   cd vllm-gaudi
#   export VLLM_COMMIT_HASH=$(git show "origin/vllm/last-good-commit-for-vllm-gaudi:VLLM_STABLE_COMMIT" 2>/dev/null)
#   cd ..
#
#   # Install vLLM for empty platform
#   git clone https://github.com/vllm-project/vllm
#   cd vllm && git checkout $VLLM_COMMIT_HASH
#   pip install -r <(sed '/^torch/d' requirements/build.txt)
#   VLLM_TARGET_DEVICE=empty pip install --no-build-isolation -e .
#   cd ..
#
#   # Install Gaudi plugin
#   cd vllm-gaudi && pip install -e . && cd ..
# ========================================

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Create logs directory
mkdir -p "$SCRIPT_DIR/logs"

# Redirect all output to log files
exec > >(tee -a "$SCRIPT_DIR/logs/tau-gaudi_${SLURM_JOB_ID}.out") 2>&1
exec 2> >(tee -a "$SCRIPT_DIR/logs/tau-gaudi_${SLURM_JOB_ID}.err" >&2)

echo "========================================"
echo "=== Intel Gaudi Experiment ==="
echo "========================================"
echo "Started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
echo "Script directory: $SCRIPT_DIR"
echo "Repository root: $REPO_ROOT"
echo ""

# ========================================
# Environment Setup for Gaudi on SOL
# ========================================
echo "=== Setting up Gaudi Environment ==="

# Load mamba for conda
module load mamba/latest

# Activate the tau-gaudi conda environment
source activate tau-gaudi || {
    echo "ERROR: tau-gaudi environment not found!"
    echo "Run this first: bash SOL_env/gaudi_env_setup.sh"
    exit 1
}

echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Python: $(which python)"

# ========================================
# Habana Environment Variables for SOL
# ========================================
# Habana software is installed at /opt/habanalabs on SOL Gaudi nodes

# Add Habana binaries to PATH
export PATH="/opt/habanalabs/bin:$PATH"

# Add Habana libraries to LD_LIBRARY_PATH
export LD_LIBRARY_PATH="/opt/habanalabs/lib:$LD_LIBRARY_PATH"

# OpenMPI for multi-card
export PATH="/opt/habanalabs/openmpi-5.0.8/bin:$PATH"
export LD_LIBRARY_PATH="/opt/habanalabs/openmpi-5.0.8/lib:$LD_LIBRARY_PATH"

# libfabric for networking
export LD_LIBRARY_PATH="/opt/habanalabs/libfabric-1.20.0/lib:$LD_LIBRARY_PATH"

# RDMA core
export LD_LIBRARY_PATH="/opt/habanalabs/rdma-core/lib64:$LD_LIBRARY_PATH"

# ========================================
# Gaudi-specific Environment Variables
# ========================================
echo "=== Setting Gaudi Environment Variables ==="

# Core Habana settings
export HABANA_VISIBLE_DEVICES=all
export PT_HPU_LAZY_MODE=1                    # Enable HPU Graphs for best performance
export PT_HPU_ENABLE_LAZY_COLLECTIVES=true   # For tensor parallelism

# vLLM Gaudi settings
export VLLM_SKIP_WARMUP=false               # Warmup for production
export VLLM_GRAPH_RESERVED_MEM=0.1          # 10% memory for graph capture
export VLLM_GRAPH_PROMPT_RATIO=0.3          # Memory split prefill/decode

# Debugging (uncomment if needed)
# export VLLM_LOGGING_LEVEL=DEBUG
# export VLLM_HPU_LOG_STEP_GRAPH_COMPILATION=1

# HuggingFace cache
export HF_HOME=/scratch/$USER/hf_cache
mkdir -p $HF_HOME

echo "HABANA_VISIBLE_DEVICES=$HABANA_VISIBLE_DEVICES"
echo "PT_HPU_LAZY_MODE=$PT_HPU_LAZY_MODE"
echo "HF_HOME=$HF_HOME"
echo ""

# Verify Gaudi is available
echo "=== Checking Gaudi Hardware ==="
if command -v hl-smi &> /dev/null; then
    hl-smi
else
    echo "WARNING: hl-smi not found. Gaudi drivers may not be loaded."
    echo "Attempting to continue anyway..."
fi
echo ""

# ========================================
# Model Configuration
# ========================================
# Using smaller model for initial testing - adjust as needed
# Gaudi 2 (HL-225) has ~96GB HBM, can fit larger models
USER_MODEL="Qwen/Qwen3-8B"
AGENT_MODEL="Qwen/Qwen3-8B"

# Ports for local servers
USER_PORT=8000
AGENT_PORT=8001

# Context length - Gaudi can handle longer contexts with block-size 128
MAX_MODEL_LEN=16384
BLOCK_SIZE=128  # Critical for BF16 performance on Gaudi

echo "=== Configuration ==="
echo "User Model: $USER_MODEL"
echo "Agent Model: $AGENT_MODEL"
echo "Max Model Length: $MAX_MODEL_LEN"
echo "Block Size: $BLOCK_SIZE"
echo "User Port: $USER_PORT"
echo "Agent Port: $AGENT_PORT"
echo ""

# ========================================
# Cleanup Function
# ========================================
cleanup() {
    echo ""
    echo "=== Cleaning up ==="
    pkill -f "vllm serve" 2>/dev/null || true

    # Kill by port
    fuser -k $USER_PORT/tcp 2>/dev/null || true
    fuser -k $AGENT_PORT/tcp 2>/dev/null || true

    echo "Cleanup complete"
}

trap cleanup EXIT INT TERM

# ========================================
# Step 1: Start User Simulator
# ========================================
echo "=== Step 1: Starting User Simulator ==="

USER_LOG="$SCRIPT_DIR/logs/gaudi_user_${SLURM_JOB_ID}.log"

# Install tau-bench
cd "$REPO_ROOT"
pip install -q -e .

echo "Starting User vLLM server..."
echo "Log file: $USER_LOG"

# Start vLLM with Gaudi device
# Key differences from CUDA:
#   --device hpu (explicitly specify Habana Processing Unit)
#   --block-size 128 (optimal for BF16 on Gaudi)
#   No --enforce-eager (HPU Graphs are preferred)
vllm serve $USER_MODEL \
    --device hpu \
    --host 0.0.0.0 \
    --port $USER_PORT \
    --block-size $BLOCK_SIZE \
    --gpu-memory-utilization 0.90 \
    --max-model-len $MAX_MODEL_LEN \
    --trust-remote-code \
    --disable-log-requests \
    > "$USER_LOG" 2>&1 &

USER_PID=$!
echo "User server PID: $USER_PID"

# ========================================
# Step 2: Start Agent (if using different model)
# ========================================
echo "=== Step 2: Starting Agent ==="

AGENT_LOG="$SCRIPT_DIR/logs/gaudi_agent_${SLURM_JOB_ID}.log"
echo "Log file: $AGENT_LOG"

# For tool-calling, we need --enable-auto-tool-choice
vllm serve $AGENT_MODEL \
    --device hpu \
    --host 0.0.0.0 \
    --port $AGENT_PORT \
    --block-size $BLOCK_SIZE \
    --gpu-memory-utilization 0.90 \
    --max-model-len $MAX_MODEL_LEN \
    --trust-remote-code \
    --disable-log-requests \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    > "$AGENT_LOG" 2>&1 &

AGENT_PID=$!
echo "Agent server PID: $AGENT_PID"
echo ""

# ========================================
# Step 3: Wait for Servers to be Ready
# ========================================
echo "=== Step 3: Waiting for servers to be ready ==="

check_server() {
    curl -s --connect-timeout 5 "http://localhost:${1}/health" > /dev/null 2>&1
    return $?
}

# Wait for User Simulator
echo -n "Waiting for User Simulator (port $USER_PORT)..."
USER_READY=0
for i in {1..60}; do
    if check_server "$USER_PORT"; then
        USER_READY=1
        echo " Ready! (${i}0s)"
        break
    fi
    echo -n "."
    sleep 10
done

if [ $USER_READY -eq 0 ]; then
    echo " FAILED!"
    echo "User Simulator did not start. Last 50 lines of log:"
    tail -50 "$USER_LOG"
    exit 1
fi

# Wait for Agent
echo -n "Waiting for Agent (port $AGENT_PORT)..."
AGENT_READY=0
for i in {1..60}; do
    if check_server "$AGENT_PORT"; then
        AGENT_READY=1
        echo " Ready! (${i}0s)"
        break
    fi
    echo -n "."
    sleep 10
done

if [ $AGENT_READY -eq 0 ]; then
    echo " FAILED!"
    echo "Agent did not start. Last 50 lines of log:"
    tail -50 "$AGENT_LOG"
    exit 1
fi

echo ""
echo "Both servers are ready!"
echo ""

# ========================================
# Step 4: Run Experiments
# ========================================
echo "=== Step 4: Running Experiments ==="

export OPENAI_API_KEY="dummy"

cd "$REPO_ROOT"

echo "Working directory: $(pwd)"
echo ""

# Run a small test first
TOTAL_EXPERIMENTS=0
SUCCESSFUL_EXPERIMENTS=0

for ENV in retail airline; do
    echo ""
    echo ">>> Running Environment: $ENV"

    for STRATEGY in tool-calling act react; do
        echo "  > Strategy: $STRATEGY"

        LOG_DIR="$SCRIPT_DIR/results_gaudi/${ENV}/${STRATEGY}"
        mkdir -p "$LOG_DIR"

        CMD="python run.py \
            --env ${ENV} \
            --agent-strategy ${STRATEGY} \
            --model ${AGENT_MODEL} \
            --model-provider openai \
            --model-base-url http://localhost:${AGENT_PORT}/v1 \
            --user-model ${USER_MODEL} \
            --user-model-provider openai \
            --user-model-base-url http://localhost:${USER_PORT}/v1 \
            --log-dir ${LOG_DIR} \
            --max-concurrency 3 \
            --num-trials 1 \
            --end-index 3 \
            --shuffle 0"

        echo "    Executing..."
        TOTAL_EXPERIMENTS=$((TOTAL_EXPERIMENTS + 1))

        if eval $CMD; then
            echo "    Completed $STRATEGY for $ENV"
            SUCCESSFUL_EXPERIMENTS=$((SUCCESSFUL_EXPERIMENTS + 1))
        else
            echo "    FAILED: $STRATEGY for $ENV"
        fi
    done
done

echo ""
echo "========================================"
echo "=== Experiments Summary ==="
echo "========================================"
echo "Total experiments: $TOTAL_EXPERIMENTS"
echo "Successful: $SUCCESSFUL_EXPERIMENTS"
echo "Failed: $((TOTAL_EXPERIMENTS - SUCCESSFUL_EXPERIMENTS))"
echo ""
echo "Results saved to: $SCRIPT_DIR/results_gaudi/"
echo ""

echo "========================================"
echo "=== Gaudi Experiment Complete ==="
echo "========================================"
echo "Finished at: $(date)"
