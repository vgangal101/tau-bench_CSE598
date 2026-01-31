#!/bin/bash
#SBATCH --job-name=tau-multi-node
#SBATCH --partition=public
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH --output=tau-multi-node_%j.out
#SBATCH --error=tau-multi-node_%j.err

# ========================================
# Multi-Node Experiment: User on Node 1, Agent on Node 2
# ========================================
# This script requests 2 nodes, each with 1 A100 GPU:
#   - Node 1: User Simulator (Qwen3-32B-INT8) on port 8000
#   - Node 2: Agent Model (Qwen3-8B) on port 8001
#
# Communication happens over the cluster network (not localhost)
# ========================================

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Create logs directory
mkdir -p "$SCRIPT_DIR/logs"
SUBMIT_DIR="$(pwd)"

echo "========================================"
echo "=== Multi-Node Experiment ==="
echo "========================================"
echo "Started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Script directory: $SCRIPT_DIR"
echo "Repository root: $REPO_ROOT"
echo ""

# ========================================
# Get Node Hostnames
# ========================================
# SLURM_JOB_NODELIST contains all allocated nodes
# We need to expand it to get individual hostnames

echo "=== Node Information ==="
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_NNODES: $SLURM_NNODES"

# Get list of nodes as an array
NODES=($(scontrol show hostnames $SLURM_JOB_NODELIST))
USER_NODE=${NODES[0]}
AGENT_NODE=${NODES[1]}

echo "User Simulator Node: $USER_NODE"
echo "Agent Model Node: $AGENT_NODE"
echo ""

# Define ports
USER_PORT=8000
AGENT_PORT=8000  # Same port since different nodes

# URLs for cross-node communication
USER_URL="http://${USER_NODE}:${USER_PORT}"
AGENT_URL="http://${AGENT_NODE}:${AGENT_PORT}"

echo "User URL: $USER_URL"
echo "Agent URL: $AGENT_URL"
echo ""

# ========================================
# Model Configuration
# ========================================
# Both models use the FP16 32B model
# Each node has 1 A100 80GB which can fit the 32B FP16 model (~64GB)
USER_MODEL="Qwen/Qwen3-32B"
AGENT_MODEL="Qwen/Qwen3-32B"

echo "=== Configuration ==="
echo "User Model: $USER_MODEL (on $USER_NODE)"
echo "Agent Model: $AGENT_MODEL (on $AGENT_NODE)"
echo ""

# ========================================
# Cleanup Function
# ========================================
cleanup() {
    echo ""
    echo "=== Cleaning up ==="

    # Kill vLLM on both nodes
    srun --nodes=1 --ntasks=1 -w $USER_NODE pkill -f "vllm serve" 2>/dev/null || true
    srun --nodes=1 --ntasks=1 -w $AGENT_NODE pkill -f "vllm serve" 2>/dev/null || true

    echo "Cleanup complete"

    # Move SLURM logs
    mv "$SUBMIT_DIR/tau-multi-node_${SLURM_JOB_ID}.out" "$SCRIPT_DIR/logs/" 2>/dev/null || true
    mv "$SUBMIT_DIR/tau-multi-node_${SLURM_JOB_ID}.err" "$SCRIPT_DIR/logs/" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

# ========================================
# Step 1: Start User Simulator on Node 1
# ========================================
echo "=== Step 1: Starting User Simulator on $USER_NODE ==="
$QWEN3_MAX_TOK_LEN=32768

# Use srun to execute on specific node
srun --nodes=1 --ntasks=1 -w $USER_NODE bash -c "
    # Load modules on the remote node
    module load mamba/latest
    module load cuda-12.1.1-gcc-12.1.0
    source activate tau-bench

    export HF_HOME=/scratch/\$USER/hf_cache
    export VLLM_USE_V1=0
    mkdir -p \$HF_HOME

    echo 'Starting User Simulator on \$(hostname)...'

    vllm serve $USER_MODEL \
        --host 0.0.0.0 \
        --port $USER_PORT \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.90 \
        --max-model-len $QWEN3_MAX_TOK_LEN \
        --trust-remote-code \
        --enforce-eager \
        --disable-log-requests \
        > $SCRIPT_DIR/logs/multi_node_user_${SLURM_JOB_ID}.log 2>&1 &

    echo 'User Simulator started in background'
" &

USER_SRUN_PID=$!
echo "User srun PID: $USER_SRUN_PID"

# ========================================
# Step 2: Start Agent on Node 2
# ========================================
echo "=== Step 2: Starting Agent on $AGENT_NODE ==="

srun --nodes=1 --ntasks=1 -w $AGENT_NODE bash -c "
    # Load modules on the remote node
    module load mamba/latest
    module load cuda-12.1.1-gcc-12.1.0
    source activate tau-bench

    export HF_HOME=/scratch/\$USER/hf_cache
    export VLLM_USE_V1=0
    mkdir -p \$HF_HOME

    echo 'Starting Agent on \$(hostname)...'

    vllm serve $AGENT_MODEL \
        --host 0.0.0.0 \
        --port $AGENT_PORT \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.90 \
        --max-model-len $QWEN3_MAX_TOK_LEN \
        --trust-remote-code \
        --enforce-eager \
        --disable-log-requests \
        --enable-auto-tool-choice \
        --tool-call-parser hermes \
        > $SCRIPT_DIR/logs/multi_node_agent_${SLURM_JOB_ID}.log 2>&1 &

    echo 'Agent started in background'
" &

AGENT_SRUN_PID=$!
echo "Agent srun PID: $AGENT_SRUN_PID"
echo ""

# ========================================
# Step 3: Wait for Servers to be Ready
# ========================================
echo "=== Step 3: Waiting for servers to be ready ==="
echo ""

# Function to check server health
check_server() {
    curl -s --connect-timeout 5 "${1}/health" > /dev/null 2>&1
    return $?
}

# Wait for User Simulator
echo -n "Waiting for User Simulator (${USER_URL})..."
USER_READY=0
for i in {1..60}; do
    if check_server "$USER_URL"; then
        USER_READY=1
        echo " Ready! (${i}0s)"
        break
    fi
    echo -n "."
    sleep 10
done

if [ $USER_READY -eq 0 ]; then
    echo " FAILED!"
    echo "User Simulator did not start. Check $SCRIPT_DIR/logs/multi_node_user_${SLURM_JOB_ID}.log"
    cat "$SCRIPT_DIR/logs/multi_node_user_${SLURM_JOB_ID}.log" 2>/dev/null | tail -50
    exit 1
fi

# Wait for Agent
echo -n "Waiting for Agent (${AGENT_URL})..."
AGENT_READY=0
for i in {1..60}; do
    if check_server "$AGENT_URL"; then
        AGENT_READY=1
        echo " Ready! (${i}0s)"
        break
    fi
    echo -n "."
    sleep 10
done

if [ $AGENT_READY -eq 0 ]; then
    echo " FAILED!"
    echo "Agent did not start. Check $SCRIPT_DIR/logs/multi_node_agent_${SLURM_JOB_ID}.log"
    cat "$SCRIPT_DIR/logs/multi_node_agent_${SLURM_JOB_ID}.log" 2>/dev/null | tail -50
    exit 1
fi

echo ""
echo "Both servers are ready!"
echo ""

# ========================================
# Step 4: Install Dependencies and Run Experiments
# ========================================
echo "=== Step 4: Running Experiments ==="
echo ""

# Load modules on the current node for running experiments
module load mamba/latest
module load cuda-12.1.1-gcc-12.1.0
source activate tau-bench

# Install dependencies
cd "$REPO_ROOT"
pip install -q -e .

export OPENAI_API_KEY="dummy"

echo "Working directory: $(pwd)"
echo "User URL: $USER_URL"
echo "Agent URL: $AGENT_URL"
echo ""

# Run experiments
TOTAL_EXPERIMENTS=0
SUCCESSFUL_EXPERIMENTS=0

for ENV in retail airline; do
    echo ""
    echo ">>> Running Environment: $ENV"

    for STRATEGY in tool-calling act react; do
        echo "  > Strategy: $STRATEGY"

        LOG_DIR="$SCRIPT_DIR/results_multi_node/${ENV}/${STRATEGY}"
        mkdir -p "$LOG_DIR"

        CMD="python run.py \
            --env ${ENV} \
            --agent-strategy ${STRATEGY} \
            --model ${AGENT_MODEL} \
            --model-provider openai \
            --model-base-url ${AGENT_URL}/v1 \
            --user-model ${USER_MODEL} \
            --user-model-provider openai \
            --user-model-base-url ${USER_URL}/v1 \
            --log-dir ${LOG_DIR} \
            --max-concurrency 5 \
            --num-trials 1 \
            --end-index 3 \
            --shuffle 0"

        echo "    Executing..."
        TOTAL_EXPERIMENTS=$((TOTAL_EXPERIMENTS + 1))

        eval $CMD
        echo "    Completed $STRATEGY for $ENV"
        SUCCESSFUL_EXPERIMENTS=$((SUCCESSFUL_EXPERIMENTS + 1))
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
echo "Results saved to: $SCRIPT_DIR/results_multi_node/"
echo ""

echo "========================================"
echo "=== Multi-Node Experiment Complete ==="
echo "========================================"
echo "Finished at: $(date)"
