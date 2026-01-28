#!/bin/bash
#SBATCH --job-name=tau-test-4b
#SBATCH --partition=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/test_4b_%j.out
#SBATCH --error=logs/test_4b_%j.err

echo "========================================"
echo "=== QUICK TEST: Both 4B Models ==="
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

# Install tau-bench package with dependencies (ensures litellm is available)
echo "=== Installing tau-bench dependencies ==="
cd ../../
pip install -q -e .
cd SOL_env/day1
echo "Dependencies installed"
echo ""

# Set API key for experiments
export OPENAI_API_KEY="dummy"

# Display GPU info
echo "=== GPU Information ==="
nvidia-smi
echo ""

# Cleanup function to kill background processes
cleanup() {
    echo ""
    echo "=== Cleaning up background processes ==="
    if [ ! -z "$USER_PID" ]; then
        echo "Stopping User Simulator (PID: $USER_PID)"
        kill $USER_PID 2>/dev/null || true
    fi
    if [ ! -z "$AGENT_PID" ]; then
        echo "Stopping Agent Server (PID: $AGENT_PID)"
        kill $AGENT_PID 2>/dev/null || true
    fi
    # Wait a moment for graceful shutdown
    sleep 5
    # Force kill if still running
    pkill -f "vllm serve" 2>/dev/null || true
    echo "Cleanup complete"
}

# Set trap to cleanup on exit
trap cleanup EXIT INT TERM

# ========================================
# Step 1: Start vLLM Servers
# ========================================
echo "=== Step 1: Starting vLLM Servers (Both on 1 GPU) ==="
echo ""

# Start User Simulator (Qwen3-4B) on port 8000
echo "[1/2] Starting User Simulator (Qwen3-4B) on port 8000..."
echo "   Allocating 35% GPU memory (~28GB) for user simulator..."
vllm serve Qwen/Qwen3-4B \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.35 \
    --max-model-len 2048 \
    --trust-remote-code \
    --disable-log-requests \
    > logs/user_4b_${SLURM_JOB_ID}.log 2>&1 &

USER_PID=$!
echo "   User Simulator started with PID: $USER_PID"

# Wait for first server to fully load model before starting second
echo "   Waiting for first model to fully load (60s)..."
sleep 60

# Start Agent (Qwen3-4B) on port 8001 with lower allocation
echo "[2/2] Starting Agent (Qwen3-4B) on port 8001..."
echo "   Allocating 30% GPU memory (~24GB) for agent..."
vllm serve Qwen/Qwen3-4B \
    --host 0.0.0.0 \
    --port 8001 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.30 \
    --max-model-len 2048 \
    --trust-remote-code \
    --disable-log-requests \
    > logs/agent_4b_${SLURM_JOB_ID}.log 2>&1 &

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
    echo "User Simulator did not start. Check logs/user_4b_${SLURM_JOB_ID}.log"
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
    echo "Agent did not start. Check logs/agent_4b_${SLURM_JOB_ID}.log"
    exit 1
fi

echo ""
echo "Both servers are ready!"
echo ""

# ========================================
# Step 3: Run Quick Test
# ========================================
echo "========================================"
echo "=== Step 3: Running Quick Test ==="
echo "========================================"
echo "Test: User (Qwen3-4B) + Agent (Qwen3-4B)"
echo "Strategy: tool-calling only"
echo "Env: retail only"
echo "Tasks: First 2 tasks only"
echo ""

# Navigate to repository root
cd ../../
echo "Working directory: $(pwd)"
echo ""

# Create log directory
LOG_DIR="SOL_env/day1/test_results/retail/tool-calling"
mkdir -p $LOG_DIR

# Run single quick experiment
echo ">>> Running TEST: retail + tool-calling (2 tasks)"

CMD="python run.py \
    --env retail \
    --agent-strategy tool-calling \
    --model Qwen/Qwen3-4B \
    --model-provider openai \
    --model-base-url ${AGENT_URL}/v1 \
    --user-model Qwen/Qwen3-4B \
    --user-model-provider openai \
    --user-model-base-url ${USER_URL}/v1 \
    --log-dir ${LOG_DIR} \
    --max-concurrency 2 \
    --num-trials 1 \
    --end-index 2 \
    --shuffle 0"

echo "Executing..."

if eval $CMD; then
    echo "✓ TEST PASSED!"
    TEST_STATUS="SUCCESS"
else
    echo "✗ TEST FAILED!"
    TEST_STATUS="FAILED"
fi

echo ""
echo "========================================"
echo "=== Test Summary ==="
echo "========================================"
echo "Status: $TEST_STATUS"
echo "Results saved to: $LOG_DIR"
echo ""
echo "Server logs:"
echo "  User: logs/user_4b_${SLURM_JOB_ID}.log"
echo "  Agent: logs/agent_4b_${SLURM_JOB_ID}.log"
echo ""

# ========================================
# Done
# ========================================
echo "========================================"
echo "=== Test Complete ==="
echo "========================================"
echo "Finished at: $(date)"
echo ""
echo "If this test succeeded, the combined approach works!"
echo "You can now run the full experiment with larger models."
echo ""
echo "Servers will be stopped automatically..."
