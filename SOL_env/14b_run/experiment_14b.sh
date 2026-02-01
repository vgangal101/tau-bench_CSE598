#!/bin/bash
#SBATCH --job-name=tau-14b-exp
#SBATCH --partition=public
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --time=6:00:00
#SBATCH --output=tau-14b-exp_%j.out
#SBATCH --error=tau-14b-exp_%j.err
#SBATCH --account=class_cse59827694spring2026

# ========================================
# 14B Experiment: User (32B) on Node 1, Agent (14B) on Node 2
# ========================================
# Configuration:
#   - Time: 6 hours
#   - GPUs: 1 per node (A100)
#   - Nodes: 2
#   - CPUs: 16 per task
#   - Memory: 96G per node
#   - Environments: retail, airline
#   - Trials: 5
#   - Tasks: all
#   - Max Concurrency: 20
# ========================================

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Create logs directory
mkdir -p "$SCRIPT_DIR/logs"

echo "========================================"
echo "=== 14B Experiment ==="
echo "========================================"
echo "Started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
echo "Script directory: $SCRIPT_DIR"
echo "Repository root: $REPO_ROOT"
echo "Logs directory: $SCRIPT_DIR/logs"
echo ""

# Verify the paths are valid
if [ ! -d "$SCRIPT_DIR" ]; then
    echo "ERROR: Script directory does not exist: $SCRIPT_DIR"
    exit 1
fi
if [ ! -d "$REPO_ROOT" ]; then
    echo "ERROR: Repository root does not exist: $REPO_ROOT"
    exit 1
fi

# ========================================
# Get Node Hostnames
# ========================================
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
USER_MODEL="Qwen/Qwen3-32B"
AGENT_MODEL="Qwen/Qwen3-14B"

echo "=== Configuration ==="
echo "User Model: $USER_MODEL (on $USER_NODE)"
echo "Agent Model: $AGENT_MODEL (on $AGENT_NODE)"
echo "Max Concurrency: 20"
echo "Trials: 5"
echo "Environments: retail, airline"
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

    # Move SLURM output files to logs directory
    if [ -n "$SLURM_SUBMIT_DIR" ] && [ -n "$SLURM_JOB_ID" ]; then
        mv "$SLURM_SUBMIT_DIR/tau-14b-exp_${SLURM_JOB_ID}.out" "$SCRIPT_DIR/logs/" 2>/dev/null || true
        mv "$SLURM_SUBMIT_DIR/tau-14b-exp_${SLURM_JOB_ID}.err" "$SCRIPT_DIR/logs/" 2>/dev/null || true
    fi
}

trap cleanup EXIT INT TERM

# ========================================
# Step 1: Start User Simulator on Node 1 (32B)
# ========================================
echo "=== Step 1: Starting User Simulator (32B) on $USER_NODE ==="
QWEN3_MAX_TOK_LEN=32768

# Define log file paths
USER_LOG="$SCRIPT_DIR/logs/14b_user_${SLURM_JOB_ID}.log"
AGENT_LOG="$SCRIPT_DIR/logs/14b_agent_${SLURM_JOB_ID}.log"

srun --nodes=1 --ntasks=1 -w $USER_NODE bash -c "
    # Load modules on the remote node
    module load mamba/latest
    module load cuda-12.1.1-gcc-12.1.0
    source activate tau-bench

    export HF_HOME=/scratch/\$USER/hf_cache
    export VLLM_USE_V1=0
    mkdir -p \$HF_HOME

    # Ensure log directory exists on this compute node
    mkdir -p $SCRIPT_DIR/logs

    echo 'Starting User Simulator (32B) on '\$(hostname)'...' >> $USER_LOG 2>&1
    echo 'Log file: $USER_LOG' >> $USER_LOG 2>&1
    echo 'Python: '\$(which python) >> $USER_LOG 2>&1

    # Run vLLM in foreground (srun itself is backgrounded)
    vllm serve $USER_MODEL \
        --host 0.0.0.0 \
        --port $USER_PORT \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.90 \
        --max-model-len $QWEN3_MAX_TOK_LEN \
        --trust-remote-code \
        --enforce-eager \
        --disable-log-requests \
        >> $USER_LOG 2>&1
" &

USER_SRUN_PID=$!
echo "User srun PID: $USER_SRUN_PID"
echo "User log: $USER_LOG"

# Small delay to let first srun establish
sleep 2

# ========================================
# Step 2: Start Agent on Node 2 (14B)
# ========================================
echo "=== Step 2: Starting Agent (14B) on $AGENT_NODE ==="

srun --nodes=1 --ntasks=1 -w $AGENT_NODE bash -c "
    # Load modules on the remote node
    module load mamba/latest
    module load cuda-12.1.1-gcc-12.1.0
    source activate tau-bench

    export HF_HOME=/scratch/\$USER/hf_cache
    export VLLM_USE_V1=0
    mkdir -p \$HF_HOME

    # Ensure log directory exists on this compute node
    mkdir -p $SCRIPT_DIR/logs

    echo 'Starting Agent (14B) on '\$(hostname)'...' >> $AGENT_LOG 2>&1
    echo 'Log file: $AGENT_LOG' >> $AGENT_LOG 2>&1
    echo 'Python: '\$(which python) >> $AGENT_LOG 2>&1

    # Run vLLM in foreground (srun itself is backgrounded)
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
        >> $AGENT_LOG 2>&1
" &

AGENT_SRUN_PID=$!
echo "Agent srun PID: $AGENT_SRUN_PID"
echo "Agent log: $AGENT_LOG"
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
    echo "User Simulator did not start. Check $USER_LOG"
    cat "$USER_LOG" 2>/dev/null | tail -50
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
    echo "Agent did not start. Check $AGENT_LOG"
    cat "$AGENT_LOG" 2>/dev/null | tail -50
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

# Environments: retail, airline
for ENV in retail airline; do
    echo ""
    echo ">>> Running Environment: $ENV"

    for STRATEGY in tool-calling act react; do
        echo "  > Strategy: $STRATEGY"

        LOG_DIR="$SCRIPT_DIR/results/${ENV}/${STRATEGY}"
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
            --max-concurrency 20 \
            --num-trials 5"

        echo "    Executing with max-concurrency=20, num-trials=5, all tasks..."
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
echo "Results saved to: $SCRIPT_DIR/results/"
echo ""

echo "========================================"
echo "=== 14B Experiment Complete ==="
echo "========================================"
echo "Finished at: $(date)"
