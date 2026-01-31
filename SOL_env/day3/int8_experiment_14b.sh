#!/bin/bash
#SBATCH --job-name=tau-int8-14b
#SBATCH --partition=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=tau-int8-14b_%j.out
#SBATCH --error=tau-int8-14b_%j.err

# Get the directory where this script is located (robust to submission directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Create logs directory
mkdir -p "$SCRIPT_DIR/logs"
SUBMIT_DIR="$(pwd)"

echo "========================================"
echo "=== INT8 Experiment: User 32B (INT8) + Agent 14B ==="
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
export VLLM_USE_V1=0
mkdir -p $HF_HOME

# Install dependencies
echo "=== Installing dependencies ==="
cd "$REPO_ROOT"
pip install -q -e .
echo "Dependencies installed"
echo ""

# Set API key for experiments
export OPENAI_API_KEY="dummy"

# ========================================
# Model Configuration
# ========================================
# Using pre-quantized GPTQ INT8 models from HuggingFace
# User: https://huggingface.co/zankich/Qwen3-32B-INT8
# Agent: https://huggingface.co/JunHowie/Qwen3-14B-GPTQ-Int8
USER_MODEL="zankich/Qwen3-32B-INT8"
AGENT_MODEL="JunHowie/Qwen3-14B-GPTQ-Int8"

echo "=== Configuration ==="
echo "User Model:  $USER_MODEL (GPTQ INT8)"
echo "Agent Model: $AGENT_MODEL (GPTQ INT8)"
echo ""

# Display GPU info
echo "=== GPU Information ==="
nvidia-smi
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo "=== Cleaning up background processes ==="
    if [ ! -z "$GPU_MONITOR_PID" ]; then
        kill $GPU_MONITOR_PID 2>/dev/null || true
    fi
    if [ ! -z "$USER_PID" ]; then
        kill $USER_PID 2>/dev/null || true
    fi
    if [ ! -z "$AGENT_PID" ]; then
        kill $AGENT_PID 2>/dev/null || true
    fi
    echo ""
    echo "=== Final GPU Usage ==="
    nvidia-smi
    sleep 5
    pkill -f "vllm serve" 2>/dev/null || true
    echo "Cleanup complete"

    # Move SLURM output files to logs directory
    echo "Moving SLURM logs to $SCRIPT_DIR/logs/"
    mv "$SUBMIT_DIR/tau-int8-14b_${SLURM_JOB_ID}.out" "$SCRIPT_DIR/logs/" 2>/dev/null || true
    mv "$SUBMIT_DIR/tau-int8-14b_${SLURM_JOB_ID}.err" "$SCRIPT_DIR/logs/" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

# ========================================
# Step 1: Start vLLM Servers
# ========================================
echo "=== Step 1: Starting vLLM Servers ==="
echo ""

# NOTE on INT8 memory: Qwen3-32B in GPTQ INT8 needs ~32GB (vs ~64GB in FP16)
# A100 80GB at 90% = 72GB available. That leaves ~40GB for KV cache.
# This allows max-model-len up to 65536 tokens.

# Start User Simulator (GPTQ INT8) on GPU 0, port 8000
echo "[1/2] Starting User Simulator ($USER_MODEL) on GPU 0, port 8000..."
CUDA_VISIBLE_DEVICES=0 VLLM_USE_V1=0 vllm serve $USER_MODEL \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 50000 \
    --quantization gptq \
    --trust-remote-code \
    --enforce-eager \
    --disable-log-requests \
    > "$SCRIPT_DIR/logs/int8_experiment_14b_user_${SLURM_JOB_ID}.log" 2>&1 &

USER_PID=$!
echo "   User Simulator started with PID: $USER_PID"

# Wait for user model to load
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

# Start Agent (GPTQ INT8) on GPU 1, port 8001
echo "[2/2] Starting Agent ($AGENT_MODEL) on GPU 1, port 8001..."
CUDA_VISIBLE_DEVICES=1 VLLM_USE_V1=0 vllm serve $AGENT_MODEL \
    --host 0.0.0.0 \
    --port 8001 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 50000 \
    --quantization gptq \
    --trust-remote-code \
    --enforce-eager \
    --disable-log-requests \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    > "$SCRIPT_DIR/logs/int8_experiment_14b_agent_${SLURM_JOB_ID}.log" 2>&1 &

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

check_server() {
    curl -s "${1}/health" > /dev/null 2>&1
    return $?
}

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
    echo "User Simulator did not start. Check $SCRIPT_DIR/logs/int8_experiment_14b_user_${SLURM_JOB_ID}.log"
    cat "$SCRIPT_DIR/logs/int8_experiment_14b_user_${SLURM_JOB_ID}.log" 2>/dev/null | tail -50
    exit 1
fi

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
    echo "Agent did not start. Check $SCRIPT_DIR/logs/int8_experiment_14b_agent_${SLURM_JOB_ID}.log"
    cat "$SCRIPT_DIR/logs/int8_experiment_14b_agent_${SLURM_JOB_ID}.log" 2>/dev/null | tail -50
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
echo "Experiment: User ($USER_MODEL GPTQ INT8) + Agent ($AGENT_MODEL GPTQ INT8)"
echo "Strategies: tool-calling, act, react"
echo "Envs: retail, airline"
echo ""

cd "$REPO_ROOT"
echo "Working directory: $(pwd)"
echo ""

# Start GPU monitoring
GPU_LOG="$SCRIPT_DIR/logs/int8_experiment_14b_gpu_usage_${SLURM_JOB_ID}.log"
(while true; do
    echo "=== GPU Usage at $(date) ===" >> ${GPU_LOG}
    nvidia-smi >> ${GPU_LOG} 2>&1
    echo "" >> ${GPU_LOG}
    sleep 60
done) &
GPU_MONITOR_PID=$!

TOTAL_EXPERIMENTS=0
SUCCESSFUL_EXPERIMENTS=0

for ENV in retail airline; do
    echo ""
    echo ">>> Running Environment: $ENV"

    for STRATEGY in tool-calling act react; do
        echo "  > Strategy: $STRATEGY"

        LOG_DIR="$SCRIPT_DIR/results_int8/${ENV}/${STRATEGY}"
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

echo "=== Results Location ==="
echo "Results saved to: $SCRIPT_DIR/results_int8/"
echo ""
echo "Server logs:"
echo "  User: $SCRIPT_DIR/logs/int8_experiment_14b_user_${SLURM_JOB_ID}.log"
echo "  Agent: $SCRIPT_DIR/logs/int8_experiment_14b_agent_${SLURM_JOB_ID}.log"
echo "  GPU Usage: ${GPU_LOG}"
echo ""

echo "========================================"
echo "=== All Complete ==="
echo "========================================"
echo "Finished at: $(date)"
echo ""
echo "Servers will be stopped automatically..."
