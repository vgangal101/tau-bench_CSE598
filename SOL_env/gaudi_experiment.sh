#!/bin/bash
#SBATCH --job-name=tau-gaudi
#SBATCH --partition=gaudi
#SBATCH --qos=public
#SBATCH --nodes=1
#SBATCH --gres=gpu:hl225:2
#SBATCH --cpus-per-task=60
#SBATCH --mem=192G
#SBATCH --time=06:00:00
#SBATCH --output=tau-gaudi_%j.out
#SBATCH --error=tau-gaudi_%j.err

# ========================================
# Intel Gaudi Experiment Script for SOL
# ========================================
# Uses Apptainer container-based vLLM for Gaudi
# Based on /data/sse/gaudi/guides documentation
# ========================================

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Create logs directory
mkdir -p "$SCRIPT_DIR/logs"

# Redirect all output to log files
exec > >(tee -a "$SCRIPT_DIR/logs/tau-gaudi_${SLURM_JOB_ID}.out") 2>&1
exec 2> >(tee -a "$SCRIPT_DIR/logs/tau-gaudi_${SLURM_JOB_ID}.err" >&2)

echo "========================================"
echo "=== Intel Gaudi Experiment (Apptainer) ==="
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
hl-smi || echo "hl-smi not available yet"
echo ""

# ========================================
# Environment Setup - Cache Redirects
# ========================================
echo "=== Setting up Cache Directories ==="

# Source cache redirects from SOL's gaudi scripts
source /data/sse/gaudi/scripts/cache-redirects.sh /scratch/$USER

# Additional cache directories
export APPTAINER_CACHEDIR="/scratch/$USER/apptainer_cache"
export APPTAINER_TMPDIR="/scratch/$USER/apptainer_tmp"
export HF_HOME="/scratch/$USER/hf_cache"

mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$HF_HOME"

echo "HF_HOME=$HF_HOME"
echo "APPTAINER_CACHEDIR=$APPTAINER_CACHEDIR"
echo ""

# ========================================
# Model Configuration
# ========================================
# Using Qwen models (supported on Gaudi per documentation)
USER_MODEL="Qwen/Qwen2.5-7B-Instruct"
AGENT_MODEL="Qwen/Qwen2.5-7B-Instruct"

# Ports for local servers
USER_PORT=8000
AGENT_PORT=8001

# Context length settings
MAX_MODEL_LEN=4096
MAX_NUM_SEQS=16

echo "=== Configuration ==="
echo "User Model: $USER_MODEL"
echo "Agent Model: $AGENT_MODEL"
echo "Max Model Length: $MAX_MODEL_LEN"
echo "Max Num Seqs: $MAX_NUM_SEQS"
echo "User Port: $USER_PORT"
echo "Agent Port: $AGENT_PORT"
echo ""

# ========================================
# Gaudi Container and Paths
# ========================================
GAUDI_BASE="/data/sse/gaudi"
CONTAINER="$GAUDI_BASE/containers/vllm-gaudi.sif"
VLLM_CD="$GAUDI_BASE/vllm-fork/.cd"

echo "=== Gaudi Paths ==="
echo "Container: $CONTAINER"
echo "vLLM .cd dir: $VLLM_CD"

if [ ! -f "$CONTAINER" ]; then
    echo "ERROR: Container not found at $CONTAINER"
    ls -la "$GAUDI_BASE/containers/" 2>/dev/null || echo "Cannot list containers directory"
    exit 1
fi

if [ ! -d "$VLLM_CD" ]; then
    echo "ERROR: vLLM .cd directory not found at $VLLM_CD"
    exit 1
fi
echo ""

# ========================================
# Create Working Directories
# ========================================
WORK_DIR="/scratch/$USER/gaudi_tau_bench_${SLURM_JOB_ID}"
mkdir -p "$WORK_DIR/user_logs" "$WORK_DIR/agent_logs"

echo "Working directory: $WORK_DIR"
echo ""

# ========================================
# Cleanup Function
# ========================================
cleanup() {
    echo ""
    echo "=== Cleaning up ==="
    # Kill any vLLM processes
    pkill -f "entrypoint_main" 2>/dev/null || true
    fuser -k $USER_PORT/tcp 2>/dev/null || true
    fuser -k $AGENT_PORT/tcp 2>/dev/null || true
    echo "Cleanup complete"
}

trap cleanup EXIT INT TERM

# ========================================
# Function to Start vLLM Server via Apptainer
# ========================================
start_vllm_server() {
    local MODEL=$1
    local PORT=$2
    local LOG_DIR=$3
    local SERVER_NAME=$4
    local EXTRA_ARGS=${5:-""}

    echo "Starting $SERVER_NAME server..."
    echo "  Model: $MODEL"
    echo "  Port: $PORT"
    echo "  Log dir: $LOG_DIR"

    # Set environment variables for this server
    export APPTAINERENV_MODEL="$MODEL"
    export APPTAINERENV_HF_HOME=/mnt/hf_cache
    export APPTAINERENV_HABANA_VISIBLE_DEVICES=all
    export APPTAINERENV_PT_HPU_LAZY_MODE=0
    export APPTAINERENV_PT_HPU_ENABLE_LAZY_COLLECTIVES=0
    export APPTAINERENV_MAX_MODEL_LEN=$MAX_MODEL_LEN
    export APPTAINERENV_MAX_NUM_SEQS=$MAX_NUM_SEQS
    export APPTAINERENV_GPU_MEMORY_UTILIZATION=0.85
    export APPTAINERENV_VLLM_SKIP_WARMUP=True
    export APPTAINERENV_PYTHONUNBUFFERED=1
    export APPTAINERENV_PORT=$PORT
    export APPTAINERENV_HOST="0.0.0.0"

    # Run the container
    cd "$VLLM_CD"

    apptainer exec \
        --bind /usr/lib64:/host-lib64 \
        --bind /usr/lib/habanalabs:/usr/lib/habanalabs \
        --bind /opt/habanalabs:/opt/habanalabs \
        --bind /usr/bin/shim_ctl:/usr/bin/shim_ctl \
        --bind "$HF_HOME:/mnt/hf_cache" \
        --bind "$(pwd):/workspace/.cd" \
        --bind "$LOG_DIR:/var/log/habana_logs" \
        --pwd /workspace/.cd \
        --writable-tmpfs \
        "$CONTAINER" \
        python3 -m entrypoints.entrypoint_main server $EXTRA_ARGS \
        > "$SCRIPT_DIR/logs/gaudi_${SERVER_NAME}_${SLURM_JOB_ID}.log" 2>&1 &

    echo $!
}

# ========================================
# Step 1: Start User Simulator Server
# ========================================
echo "=== Step 1: Starting User Simulator ==="

USER_PID=$(start_vllm_server "$USER_MODEL" "$USER_PORT" "$WORK_DIR/user_logs" "user")
echo "User server PID: $USER_PID"
echo ""

# Wait a bit before starting second server
sleep 30

# ========================================
# Step 2: Start Agent Server
# ========================================
echo "=== Step 2: Starting Agent Server ==="

# For tool-calling, we may need additional flags
AGENT_PID=$(start_vllm_server "$AGENT_MODEL" "$AGENT_PORT" "$WORK_DIR/agent_logs" "agent")
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
for i in {1..90}; do
    if check_server "$USER_PORT"; then
        USER_READY=1
        echo " Ready! (${i}0s)"
        break
    fi
    # Check if process is still running
    if ! kill -0 $USER_PID 2>/dev/null; then
        echo " FAILED! (process died)"
        echo "Last 100 lines of user server log:"
        tail -100 "$SCRIPT_DIR/logs/gaudi_user_${SLURM_JOB_ID}.log"
        exit 1
    fi
    echo -n "."
    sleep 10
done

if [ $USER_READY -eq 0 ]; then
    echo " TIMEOUT!"
    echo "User Simulator did not start. Last 100 lines of log:"
    tail -100 "$SCRIPT_DIR/logs/gaudi_user_${SLURM_JOB_ID}.log"
    exit 1
fi

# Wait for Agent
echo -n "Waiting for Agent (port $AGENT_PORT)..."
AGENT_READY=0
for i in {1..90}; do
    if check_server "$AGENT_PORT"; then
        AGENT_READY=1
        echo " Ready! (${i}0s)"
        break
    fi
    # Check if process is still running
    if ! kill -0 $AGENT_PID 2>/dev/null; then
        echo " FAILED! (process died)"
        echo "Last 100 lines of agent server log:"
        tail -100 "$SCRIPT_DIR/logs/gaudi_agent_${SLURM_JOB_ID}.log"
        exit 1
    fi
    echo -n "."
    sleep 10
done

if [ $AGENT_READY -eq 0 ]; then
    echo " TIMEOUT!"
    echo "Agent did not start. Last 100 lines of log:"
    tail -100 "$SCRIPT_DIR/logs/gaudi_agent_${SLURM_JOB_ID}.log"
    exit 1
fi

echo ""
echo "Both servers are ready!"
echo ""

# Test the servers
echo "=== Testing Servers ==="
echo "User server models:"
curl -s "http://localhost:$USER_PORT/v1/models" | head -20
echo ""
echo "Agent server models:"
curl -s "http://localhost:$AGENT_PORT/v1/models" | head -20
echo ""

# ========================================
# Step 4: Install tau-bench
# ========================================
echo "=== Step 4: Installing tau-bench ==="

# Load mamba and activate tau-bench environment
module load mamba/latest
source activate tau-bench 2>/dev/null || {
    echo "Creating tau-bench environment..."
    mamba create -n tau-bench -c conda-forge python=3.11 -y
    source activate tau-bench
    cd "$REPO_ROOT"
    pip install -e .
}

cd "$REPO_ROOT"
pip install -q -e . 2>/dev/null || pip install -e .
echo ""

# ========================================
# Step 5: Run Experiments
# ========================================
echo "=== Step 5: Running Experiments ==="

export OPENAI_API_KEY="dummy"

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
            --max-concurrency 2 \
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
