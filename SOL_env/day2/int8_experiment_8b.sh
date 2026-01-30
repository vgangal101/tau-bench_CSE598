#!/bin/bash
#SBATCH --job-name=tau-int8-8b
#SBATCH --partition=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH --output=logs/int8_experiment_8b_%j.out
#SBATCH --error=logs/int8_experiment_8b_%j.err

echo "========================================"
echo "=== INT8 Experiment: User 32B (INT8) + Agent 8B ==="
echo "========================================"
echo "Started at: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
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
cd ../../
pip install -q -e .
cd SOL_env/day2
echo "Dependencies installed"
echo ""

# Create logs directory
mkdir -p logs

# Set API key for experiments
export OPENAI_API_KEY="dummy"

# ========================================
# Model Configuration
# ========================================
# Using pre-quantized GPTQ INT8 model from HuggingFace
# Source: https://huggingface.co/zankich/Qwen3-32B-INT8
USER_MODEL="zankich/Qwen3-32B-INT8"
AGENT_MODEL="Qwen/Qwen3-8B"

echo "=== Configuration ==="
echo "User Model:  $USER_MODEL (GPTQ INT8)"
echo "Agent Model: $AGENT_MODEL (FP16)"
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
    > logs/int8_experiment_8b_user_${SLURM_JOB_ID}.log 2>&1 &

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

# Start Agent on GPU 1, port 8001
echo "[2/2] Starting Agent ($AGENT_MODEL) on GPU 1, port 8001..."
CUDA_VISIBLE_DEVICES=1 VLLM_USE_V1=0 vllm serve $AGENT_MODEL \
    --host 0.0.0.0 \
    --port 8001 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 50000 \
    --trust-remote-code \
    --enforce-eager \
    --disable-log-requests \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    > logs/int8_experiment_8b_agent_${SLURM_JOB_ID}.log 2>&1 &

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
    echo "User Simulator did not start. Check logs/int8_experiment_8b_user_${SLURM_JOB_ID}.log"
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
    echo "Agent did not start. Check logs/int8_experiment_8b_agent_${SLURM_JOB_ID}.log"
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
echo "Experiment: User ($USER_MODEL GPTQ INT8) + Agent ($AGENT_MODEL)"
echo "Strategies: tool-calling, act, react"
echo "Envs: retail, airline"
echo ""

cd ../../
echo "Working directory: $(pwd)"
echo ""

# Start GPU monitoring
GPU_LOG="SOL_env/day2/logs/int8_experiment_8b_gpu_usage_${SLURM_JOB_ID}.log"
mkdir -p SOL_env/day2/logs
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

        LOG_DIR="SOL_env/day2/results_int8/${ENV}/${STRATEGY}"
        mkdir -p $LOG_DIR

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
echo "Results saved to: SOL_env/day2/results_int8/"
echo ""
echo "Server logs:"
echo "  User: SOL_env/day2/logs/int8_experiment_8b_user_${SLURM_JOB_ID}.log"
echo "  Agent: SOL_env/day2/logs/int8_experiment_8b_agent_${SLURM_JOB_ID}.log"
echo "  GPU Usage: ${GPU_LOG}"
echo ""

echo "========================================"
echo "=== All Complete ==="
echo "========================================"
echo "Finished at: $(date)"
echo ""
echo "Servers will be stopped automatically..."
