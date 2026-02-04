#!/bin/bash
#SBATCH --job-name=8b-airline-act-tau-gaudi
#SBATCH --partition=gaudi
#SBATCH --qos=class_gaudi
#SBATCH --account=class_cse59827694spring2026
#SBATCH --nodes=1
#SBATCH --gres=gpu:hl225:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=8b-airline-act-tau-gaudi_%j.out
#SBATCH --error=8b-airline-act-tau-gaudi_%j.err

# ========================================
# Gaudi 8B Airline Act Experiment
# ========================================
# Configuration:
#   - HPUs: 1 x HL-225 (Gaudi2, 96GB HBM)
#   - Model: Qwen3-8B (same for user and agent)
#   - Environment: airline
#   - Strategy: act
#   - Trials: 5
#   - Max Concurrency: 3
# ========================================

set -e

# Use SLURM_SUBMIT_DIR for reliable path resolution
SCRIPT_DIR="${SLURM_SUBMIT_DIR}/SOL_env/8b_run"
REPO_ROOT="${SLURM_SUBMIT_DIR}"

# Create logs directory
mkdir -p "$SCRIPT_DIR/logs"

# Redirect all output to log files
exec > >(tee -a "$SCRIPT_DIR/logs/tau-gaudi-8b-airline-act_${SLURM_JOB_ID}.out") 2>&1
exec 2> >(tee -a "$SCRIPT_DIR/logs/tau-gaudi-8b-airline-act_${SLURM_JOB_ID}.err" >&2)

echo "========================================"
echo "=== Gaudi 8B Airline Act Experiment ==="
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
MODEL="Qwen/Qwen3-8B"

# Port for vLLM server (8B airline uses 8101 to avoid conflicts)
PORT=8101

# Context length settings - match regular A100 scripts
MAX_MODEL_LEN=32768
MAX_NUM_SEQS=16

# Experiment settings
ENV="airline"
STRATEGY="act"
NUM_TRIALS=5
MAX_CONCURRENCY=5

echo "=== Configuration ==="
echo "Model: $MODEL"
echo "Max Model Length: $MAX_MODEL_LEN"
echo "Port: $PORT"
echo "Environment: $ENV"
echo "Strategy: $STRATEGY"
echo "Trials: $NUM_TRIALS"
echo "Max Concurrency: $MAX_CONCURRENCY"
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
mkdir -p "$WORK_DIR/logs"

echo "Working directory: $WORK_DIR"
echo ""

# ========================================
# Cleanup Function
# ========================================
cleanup() {
    echo ""
    echo "=== Cleaning up ==="
    pkill -f "vllm serve" 2>/dev/null || true
    fuser -k $PORT/tcp 2>/dev/null || true
    echo "Cleanup complete"
}

trap cleanup EXIT INT TERM

# ========================================
# Step 1: Start vLLM Server via Apptainer
# ========================================
echo "=== Step 1: Starting vLLM Server ==="

SERVER_LOG="$SCRIPT_DIR/logs/gaudi_vllm_8b_airline_${SLURM_JOB_ID}.log"

echo "Starting vLLM server..."
echo "  Model: $MODEL"
echo "  Port: $PORT"
echo "  Log: $SERVER_LOG"

# Set environment variables
export APPTAINERENV_MODEL="$MODEL"
export APPTAINERENV_HF_HOME=/mnt/hf_cache
export APPTAINERENV_HABANA_VISIBLE_DEVICES=all
export APPTAINERENV_PT_HPU_LAZY_MODE=0
export APPTAINERENV_PT_HPU_ENABLE_LAZY_COLLECTIVES=True
export APPTAINERENV_VLLM_SKIP_WARMUP=True
export APPTAINERENV_VLLM_DELAYED_SAMPLING=True
export APPTAINERENV_PYTHONUNBUFFERED=1

# Bucketing configuration
export APPTAINERENV_VLLM_PROMPT_BS_BUCKET_MIN=1
export APPTAINERENV_VLLM_PROMPT_BS_BUCKET_STEP=32
export APPTAINERENV_VLLM_DECODE_BS_BUCKET_MIN=1
export APPTAINERENV_VLLM_DECODE_BS_BUCKET_STEP=32
export APPTAINERENV_VLLM_PROMPT_SEQ_BUCKET_MIN=128
export APPTAINERENV_VLLM_PROMPT_SEQ_BUCKET_STEP=256
export APPTAINERENV_VLLM_DECODE_BLOCK_BUCKET_MIN=128
export APPTAINERENV_VLLM_DECODE_BLOCK_BUCKET_STEP=256

# Run the container with vllm serve
cd "$VLLM_CD"
mkdir -p "$WORK_DIR/logs"

apptainer exec \
    --bind /usr/lib64:/host-lib64 \
    --bind /usr/lib/habanalabs:/usr/lib/habanalabs \
    --bind /opt/habanalabs:/opt/habanalabs \
    --bind /usr/bin/shim_ctl:/usr/bin/shim_ctl \
    --bind "$HF_HOME:/mnt/hf_cache" \
    --bind "$(pwd):/workspace/.cd" \
    --bind "$WORK_DIR/logs:/var/log/habana_logs" \
    --pwd /workspace/.cd \
    --writable-tmpfs \
    "$CONTAINER" \
    vllm serve "$MODEL" \
        --host 0.0.0.0 \
        --port $PORT \
        --block-size 128 \
        --dtype bfloat16 \
        --tensor-parallel-size 1 \
        --download-dir /mnt/hf_cache \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization 0.90 \
        --use-padding-aware-scheduling \
        --max-num-seqs $MAX_NUM_SEQS \
        --max-num-prefill-seqs 8 \
        --num-scheduler-steps 1 \
        --disable-log-requests \
        --enable-auto-tool-choice \
        --tool-call-parser hermes \
    > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!
echo "vLLM server PID: $SERVER_PID"
echo ""

# ========================================
# Step 2: Wait for Server to be Ready
# ========================================
echo "=== Step 2: Waiting for server to be ready ==="

check_server() {
    curl -s --connect-timeout 5 "http://localhost:${1}/health" > /dev/null 2>&1
    return $?
}

echo -n "Waiting for vLLM server (port $PORT)..."
SERVER_READY=0
for i in {1..90}; do
    if check_server "$PORT"; then
        SERVER_READY=1
        echo " Ready! (${i}0s)"
        break
    fi
    # Check if process is still running
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo " FAILED! (process died)"
        echo "Last 100 lines of server log:"
        tail -100 "$SERVER_LOG"
        exit 1
    fi
    echo -n "."
    sleep 10
done

if [ $SERVER_READY -eq 0 ]; then
    echo " TIMEOUT!"
    echo "vLLM server did not start. Last 100 lines of log:"
    tail -100 "$SERVER_LOG"
    exit 1
fi

echo ""
echo "Server is ready!"
echo ""

# Test the server
echo "=== Testing Server ==="
echo "Available models:"
curl -s "http://localhost:$PORT/v1/models" | head -20
echo ""

# ========================================
# Step 3: Install tau-bench
# ========================================
echo "=== Step 3: Installing tau-bench ==="

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
# Step 4: Run Experiment
# ========================================
echo "=== Step 4: Running Experiment ==="

export OPENAI_API_KEY="dummy"

echo "Working directory: $(pwd)"
echo ""

# Results directory
LOG_DIR="$SCRIPT_DIR/results_gaudi_8b/${ENV}/${STRATEGY}"
mkdir -p "$LOG_DIR"

SERVER_URL="http://localhost:${PORT}/v1"

echo ">>> Running Environment: $ENV, Strategy: $STRATEGY"
echo "    Trials: $NUM_TRIALS"
echo "    Max Concurrency: $MAX_CONCURRENCY"
echo ""

CMD="python run.py \
    --env ${ENV} \
    --agent-strategy ${STRATEGY} \
    --model ${MODEL} \
    --model-provider openai \
    --model-base-url ${SERVER_URL} \
    --user-model ${MODEL} \
    --user-model-provider openai \
    --user-model-base-url ${SERVER_URL} \
    --log-dir ${LOG_DIR} \
    --max-concurrency ${MAX_CONCURRENCY} \
    --num-trials ${NUM_TRIALS}"

echo "Executing..."
echo "$CMD"
echo ""

if eval $CMD; then
    echo ""
    echo "Experiment completed successfully!"
else
    echo ""
    echo "Experiment failed!"
fi

echo ""
echo "========================================"
echo "=== Experiment Summary ==="
echo "========================================"
echo "Environment: $ENV"
echo "Strategy: $STRATEGY"
echo "Model: $MODEL"
echo "Trials: $NUM_TRIALS"
echo ""
echo "Results saved to: $LOG_DIR"
echo ""

echo "========================================"
echo "=== Gaudi 8B Airline Act Complete ==="
echo "========================================"
echo "Finished at: $(date)"
