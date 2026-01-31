#!/bin/bash
#SBATCH --job-name=tau-day3-14b
#SBATCH --partition=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=tau-day3-14b_%j.out
#SBATCH --error=tau-day3-14b_%j.err

# Get the directory where this script is located (robust to submission directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Create logs directory
mkdir -p "$SCRIPT_DIR/logs"
SUBMIT_DIR="$(pwd)"

echo "========================================"
echo "=== Day 3: User 32B + Agent 14B ==="
echo "========================================"
echo "Started at: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Script directory: $SCRIPT_DIR"
echo "Repository root: $REPO_ROOT"
echo "Submit directory: $SUBMIT_DIR"
echo ""

# Load modules
module load mamba/latest
module load cuda-12.1.1-gcc-12.1.0
source activate tau-bench

# Set up HuggingFace cache
export HF_HOME=/scratch/$USER/hf_cache
# Disable V1 engine to avoid process-based init issues with multi-server setup
export VLLM_USE_V1=0
mkdir -p $HF_HOME

# Install tau-bench package with dependencies
echo "=== Installing tau-bench dependencies ==="
cd "$REPO_ROOT"
pip install -q -e .
echo "Dependencies installed"
echo ""

# Set API key for experiments
export OPENAI_API_KEY="dummy"

# ========================================
# Model Configuration
# ========================================
USER_MODEL="Qwen/Qwen3-32B"
AGENT_MODEL="Qwen/Qwen3-14B"

echo "=== Configuration ==="
echo "User Model:  $USER_MODEL"
echo "Agent Model: $AGENT_MODEL"
echo ""

# Display GPU info
echo "=== GPU Information ==="
nvidia-smi
echo ""

# Cleanup function to kill background processes
cleanup() {
    echo ""
    echo "=== Cleaning up background processes ==="
    if [ ! -z "$GPU_MONITOR_PID" ]; then
        echo "Stopping GPU Monitor (PID: $GPU_MONITOR_PID)"
        kill $GPU_MONITOR_PID 2>/dev/null || true
    fi
    if [ ! -z "$USER_PID" ]; then
        echo "Stopping User Simulator (PID: $USER_PID)"
        kill $USER_PID 2>/dev/null || true
    fi
    if [ ! -z "$AGENT_PID" ]; then
        echo "Stopping Agent Server (PID: $AGENT_PID)"
        kill $AGENT_PID 2>/dev/null || true
    fi
    # Final GPU snapshot
    echo ""
    echo "=== Final GPU Usage ==="
    nvidia-smi
    # Wait a moment for graceful shutdown
    sleep 5
    # Force kill if still running
    pkill -f "vllm serve" 2>/dev/null || true
    echo "Cleanup complete"

    # Move SLURM output files to logs directory
    echo "Moving SLURM logs to $SCRIPT_DIR/logs/"
    mv "$SUBMIT_DIR/tau-day3-14b_${SLURM_JOB_ID}.out" "$SCRIPT_DIR/logs/" 2>/dev/null || true
    mv "$SUBMIT_DIR/tau-day3-14b_${SLURM_JOB_ID}.err" "$SCRIPT_DIR/logs/" 2>/dev/null || true
}

# Set trap to cleanup on exit
trap cleanup EXIT INT TERM

# ========================================
# Step 1: Start vLLM Servers
# ========================================
echo "=== Step 1: Starting vLLM Servers (1 GPU per 32B model) ==="
echo ""

# NOTE on memory: Qwen3-32B needs ~64GB in fp16.
# A100 80GB at 90% = 72GB available. That leaves ~8GB for KV cache.
# max-model-len 36000 is near the max (~36224) for 32B on A100 80GB at 90% util.
# If you hit OOM, reduce max-model-len or request 4 GPUs
# and use tensor-parallel-size 2 per model.

# Start User Simulator on GPU 0, port 8000
echo "[1/2] Starting User Simulator ($USER_MODEL) on GPU 0, port 8000..."
CUDA_VISIBLE_DEVICES=0 VLLM_USE_V1=0 vllm serve $USER_MODEL \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 36000 \
    --trust-remote-code \
    --enforce-eager \
    --disable-log-requests \
    > "$SCRIPT_DIR/logs/combined_experiment_14b_user_${SLURM_JOB_ID}.log" 2>&1 &

USER_PID=$!
echo "   User Simulator started with PID: $USER_PID"

# Wait for first server to load before starting second (32B takes longer)
echo "   Waiting for User Simulator to load (checking health endpoint)..."
for i in {1..60}; do
    if curl -s "http://localhost:8000/health" > /dev/null 2>&1; then
        echo "   User Simulator loaded successfully after ${i}0s"
        break
    fi
    sleep 10
done

echo ""
echo "=== GPU Usage After User Simulator Loaded ==="
nvidia-smi
echo ""

# Start Agent on GPU 1, port 8001
# NOTE: Qwen3-14B needs ~28GB, leaving ~44GB for KV cache on A100 80GB
# This allows longer context than the 32B user model
echo "[2/2] Starting Agent ($AGENT_MODEL) on GPU 1, port 8001..."
CUDA_VISIBLE_DEVICES=1 VLLM_USE_V1=0 vllm serve $AGENT_MODEL \
    --host 0.0.0.0 \
    --port 8001 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 65536 \
    --trust-remote-code \
    --enforce-eager \
    --disable-log-requests \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    > "$SCRIPT_DIR/logs/combined_experiment_14b_agent_${SLURM_JOB_ID}.log" 2>&1 &

AGENT_PID=$!
echo "   Agent started with PID: $AGENT_PID"
echo ""

# ========================================
# Step 2: Wait for servers to be ready
# ========================================
echo "=== Step 2: Waiting for servers to be ready ==="
echo ""

USER_URL="http://localhost:8000"
AGENT_URL="http://localhost:8001"

# Function to check if server is ready
check_server() {
    local url=$1
    curl -s "${url}/health" > /dev/null 2>&1
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
    echo "User Simulator did not start. Check $SCRIPT_DIR/logs/combined_experiment_14b_user_${SLURM_JOB_ID}.log"
    cat "$SCRIPT_DIR/logs/combined_experiment_14b_user_${SLURM_JOB_ID}.log" 2>/dev/null | tail -50
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
    echo "Agent did not start. Check $SCRIPT_DIR/logs/combined_experiment_14b_agent_${SLURM_JOB_ID}.log"
    cat "$SCRIPT_DIR/logs/combined_experiment_14b_agent_${SLURM_JOB_ID}.log" 2>/dev/null | tail -50
    exit 1
fi

echo ""
echo "Both servers are ready!"
echo ""
echo "=== GPU Usage After Both Servers Loaded ==="
nvidia-smi
echo ""

# ========================================
# Step 3: Run Experiments
# ========================================
echo "========================================"
echo "=== Step 3: Running Experiments ==="
echo "========================================"
echo "Experiment: User ($USER_MODEL) + Agent ($AGENT_MODEL)"
echo "Strategies: tool-calling, act, react"
echo "Envs: retail, airline"
echo ""

# Navigate to repository root
cd "$REPO_ROOT"
echo "Working directory: $(pwd)"
echo ""

# Start GPU monitoring in background (every 60s)
GPU_LOG="$SCRIPT_DIR/logs/combined_experiment_14b_gpu_usage_${SLURM_JOB_ID}.log"
echo "=== Starting GPU monitoring (logs every 60s to ${GPU_LOG}) ==="
(while true; do
    echo "=== GPU Usage at $(date) ===" >> ${GPU_LOG}
    nvidia-smi >> ${GPU_LOG} 2>&1
    echo "" >> ${GPU_LOG}
    sleep 60
done) &
GPU_MONITOR_PID=$!
echo ""

# Loop through environments and strategies
TOTAL_EXPERIMENTS=0
SUCCESSFUL_EXPERIMENTS=0

for ENV in retail airline; do
    echo ""
    echo ">>> Running Environment: $ENV"

    for STRATEGY in tool-calling act react; do
        echo "  > Strategy: $STRATEGY"

        # Create log directory
        LOG_DIR="$SCRIPT_DIR/results/${ENV}/${STRATEGY}"
        mkdir -p "$LOG_DIR"

        # Run experiment
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

# ========================================
# Step 4: Display Results Info
# ========================================
echo "=== Results Location ==="
echo "Results saved to: $SCRIPT_DIR/results/"
echo ""
echo "Server logs:"
echo "  User: $SCRIPT_DIR/logs/combined_experiment_14b_user_${SLURM_JOB_ID}.log"
echo "  Agent: $SCRIPT_DIR/logs/combined_experiment_14b_agent_${SLURM_JOB_ID}.log"
echo "  GPU Usage: ${GPU_LOG}"
echo ""

# ========================================
# Done
# ========================================
echo "========================================"
echo "=== All Complete ==="
echo "========================================"
echo "Finished at: $(date)"
echo ""
echo "Servers will be stopped automatically..."
