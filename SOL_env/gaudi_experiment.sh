#!/bin/bash
#SBATCH --job-name=tau-gaudi
#SBATCH --partition=gaudi
#SBATCH --qos=public
#SBATCH --nodes=1
#SBATCH --gres=gpu:hl225:1
#SBATCH --cpus-per-task=30
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=tau-gaudi_%j.out
#SBATCH --error=tau-gaudi_%j.err

# ========================================
# Intel Gaudi Experiment Script for SOL
# ========================================
# This script runs tau-bench experiments on Intel Gaudi2 accelerators
# using the pre-configured gaudi-pytorch-vllm environment on SOL.
#
# Based on ASU RC Workshop "Introducing the Gaudi2"
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
echo "Script directory: $SCRIPT_DIR"
echo "Repository root: $REPO_ROOT"
echo ""

# ========================================
# Check Gaudi Hardware
# ========================================
echo "=== Checking Gaudi Hardware ==="
hl-smi
echo ""

# ========================================
# Environment Setup - Use SOL's Pre-configured Environment
# ========================================
echo "=== Setting up Environment ==="

# Check for guides and containers
echo "Available Gaudi resources on SOL:"
ls -la /data/sse/gaudi/ 2>/dev/null || echo "Note: /data/sse/gaudi not accessible"
echo ""

# Source the gaudi-pytorch-vllm environment
# This is pre-configured by SOL admins with all Habana packages
if [ -f "/data/sse/gaudi/activate_vllm.sh" ]; then
    echo "Sourcing /data/sse/gaudi/activate_vllm.sh"
    source /data/sse/gaudi/activate_vllm.sh
elif [ -f "/data/sse/gaudi/env/vllm/bin/activate" ]; then
    echo "Activating vLLM virtual environment"
    source /data/sse/gaudi/env/vllm/bin/activate
else
    # Fall back to using Apptainer/Singularity container
    echo "Looking for Apptainer container..."
    CONTAINER=$(find /data/sse/gaudi -name "*.sif" 2>/dev/null | head -1)
    if [ -n "$CONTAINER" ]; then
        echo "Found container: $CONTAINER"
        USE_CONTAINER=true
    else
        echo "WARNING: No pre-configured environment found"
        echo "Check /data/sse/gaudi/guides/ for setup instructions"

        # Try loading conda environment if available
        module load mamba/latest 2>/dev/null
        source activate tau-gaudi 2>/dev/null || {
            echo "ERROR: No Gaudi environment available"
            echo "Please check /data/sse/gaudi/ for setup instructions"
            exit 1
        }
    fi
fi

echo "Python: $(which python 2>/dev/null || echo 'not found')"
echo "vLLM: $(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo 'not found')"
echo ""

# ========================================
# Habana Environment Variables
# ========================================
echo "=== Setting Habana Environment Variables ==="

# Core Habana settings (may already be set by environment)
export HABANA_VISIBLE_DEVICES=${HABANA_VISIBLE_DEVICES:-all}
export PT_HPU_LAZY_MODE=${PT_HPU_LAZY_MODE:-1}
export PT_HPU_ENABLE_LAZY_COLLECTIVES=${PT_HPU_ENABLE_LAZY_COLLECTIVES:-true}

# vLLM settings
export VLLM_SKIP_WARMUP=${VLLM_SKIP_WARMUP:-false}

# HuggingFace cache
export HF_HOME=${HF_HOME:-/scratch/$USER/hf_cache}
mkdir -p $HF_HOME

echo "HABANA_VISIBLE_DEVICES=$HABANA_VISIBLE_DEVICES"
echo "PT_HPU_LAZY_MODE=$PT_HPU_LAZY_MODE"
echo "HF_HOME=$HF_HOME"
echo ""

# ========================================
# Model Configuration
# ========================================
# Qwen models are supported on Gaudi (per slide 11)
USER_MODEL="Qwen/Qwen2.5-7B-Instruct"
AGENT_MODEL="Qwen/Qwen2.5-7B-Instruct"

# Ports for local servers
USER_PORT=8000
AGENT_PORT=8001

# Context length - Gaudi2 has 96GB HBM
MAX_MODEL_LEN=32768
BLOCK_SIZE=128  # Optimal for BF16 on Gaudi

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
pip install -q -e . 2>/dev/null || pip install -e .

echo "Starting User vLLM server..."
echo "Log file: $USER_LOG"

# Start vLLM with Gaudi device
# Key flags for Gaudi: --device hpu, --block-size 128
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
# Step 2: Start Agent
# ========================================
echo "=== Step 2: Starting Agent ==="

AGENT_LOG="$SCRIPT_DIR/logs/gaudi_agent_${SLURM_JOB_ID}.log"
echo "Log file: $AGENT_LOG"

# For tool-calling, enable auto tool choice
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

# Run experiments
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
