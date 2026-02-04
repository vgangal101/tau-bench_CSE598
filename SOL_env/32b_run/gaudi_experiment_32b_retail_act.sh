#!/bin/bash
#SBATCH --job-name=tau-gaudi-32b-retail-act
#SBATCH --partition=gaudi
#SBATCH --qos=class_gaudi
#SBATCH --account=class_cse59827694spring2026
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:hl225:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=tau-gaudi-32b-retail-act_%j.out
#SBATCH --error=tau-gaudi-32b-retail-act_%j.err

# ========================================
# Gaudi 32B Retail Act Experiment (Multi-Node)
# ========================================
# Configuration:
#   - Nodes: 2 (User 32B on Node 1, Agent 32B on Node 2)
#   - HPUs: 1 x HL-225 per node (Gaudi2)
#   - Model: Qwen3-32B
#   - Environment: retail
#   - Strategy: act
#   - Trials: 5
#   - Max Concurrency: 5
# ========================================

set -e

# Use SLURM_SUBMIT_DIR for reliable path resolution
SCRIPT_DIR="${SLURM_SUBMIT_DIR}/SOL_env/32b_run"
REPO_ROOT="${SLURM_SUBMIT_DIR}"

# Create logs directory
mkdir -p "$SCRIPT_DIR/logs"

# Redirect all output to log files
exec > >(tee -a "$SCRIPT_DIR/logs/tau-gaudi-32b-retail-act_${SLURM_JOB_ID}.out") 2>&1
exec 2> >(tee -a "$SCRIPT_DIR/logs/tau-gaudi-32b-retail-act_${SLURM_JOB_ID}.err" >&2)

echo "========================================"
echo "=== Gaudi 32B Retail Act (Multi-Node) ==="
echo "========================================"
echo "Started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Script directory: $SCRIPT_DIR"
echo "Repository root: $REPO_ROOT"
echo ""

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
# Environment Setup
# ========================================
echo "=== Setting up Cache Directories ==="

export APPTAINER_CACHEDIR="/scratch/$USER/apptainer_cache"
export APPTAINER_TMPDIR="/scratch/$USER/apptainer_tmp"
export HF_HOME="/scratch/$USER/hf_cache"

mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$HF_HOME"

echo "HF_HOME=$HF_HOME"
echo ""

# ========================================
# Model Configuration
# ========================================
USER_MODEL="Qwen/Qwen3-32B"
AGENT_MODEL="Qwen/Qwen3-32B"

# Context length - 32B on single HPU (96GB) can handle ~16K
MAX_MODEL_LEN=16384
MAX_NUM_SEQS=8

# Experiment settings
ENV="retail"
STRATEGY="act"
NUM_TRIALS=5
MAX_CONCURRENCY=5

echo "=== Configuration ==="
echo "User Model: $USER_MODEL (on $USER_NODE)"
echo "Agent Model: $AGENT_MODEL (on $AGENT_NODE)"
echo "Max Model Length: $MAX_MODEL_LEN"
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
    # Kill vLLM on both nodes
    srun --nodes=1 --ntasks=1 -w $USER_NODE pkill -f "vllm serve" 2>/dev/null || true
    srun --nodes=1 --ntasks=1 -w $AGENT_NODE pkill -f "vllm serve" 2>/dev/null || true
    echo "Cleanup complete"
}

trap cleanup EXIT INT TERM

# ========================================
# Step 1: Start User Simulator on Node 1
# ========================================
echo "=== Step 1: Starting User Simulator (32B) on $USER_NODE ==="

USER_LOG="$SCRIPT_DIR/logs/gaudi_user_32b_${SLURM_JOB_ID}.log"

srun --nodes=1 --ntasks=1 -w $USER_NODE bash -c "
    # Source cache redirects
    source /data/sse/gaudi/scripts/cache-redirects.sh /scratch/\$USER 2>/dev/null || true

    export APPTAINER_CACHEDIR='/scratch/\$USER/apptainer_cache'
    export APPTAINER_TMPDIR='/scratch/\$USER/apptainer_tmp'
    export HF_HOME='/scratch/\$USER/hf_cache'
    export HABANA_VISIBLE_DEVICES=all
    export PT_HPU_LAZY_MODE=0
    export PT_HPU_ENABLE_LAZY_COLLECTIVES=True
    export VLLM_SKIP_WARMUP=True

    mkdir -p \$HF_HOME $WORK_DIR/user_logs

    echo 'Starting User Simulator (32B) on '\$(hostname)'...'
    echo 'Log file: $USER_LOG'

    cd $VLLM_CD

    apptainer exec \\
        --bind /usr/lib64:/host-lib64 \\
        --bind /usr/lib/habanalabs:/usr/lib/habanalabs \\
        --bind /opt/habanalabs:/opt/habanalabs \\
        --bind /usr/bin/shim_ctl:/usr/bin/shim_ctl \\
        --bind \$HF_HOME:/mnt/hf_cache \\
        --bind \$(pwd):/workspace/.cd \\
        --bind $WORK_DIR/user_logs:/var/log/habana_logs \\
        --pwd /workspace/.cd \\
        --writable-tmpfs \\
        $CONTAINER \\
        vllm serve $USER_MODEL \\
            --host 0.0.0.0 \\
            --port $USER_PORT \\
            --block-size 128 \\
            --dtype bfloat16 \\
            --tensor-parallel-size 1 \\
            --download-dir /mnt/hf_cache \\
            --max-model-len $MAX_MODEL_LEN \\
            --gpu-memory-utilization 0.90 \\
            --use-padding-aware-scheduling \\
            --max-num-seqs $MAX_NUM_SEQS \\
            --max-num-prefill-seqs 4 \\
            --num-scheduler-steps 1 \\
            --disable-log-requests
" > "$USER_LOG" 2>&1 &

USER_SRUN_PID=$!
echo "User srun PID: $USER_SRUN_PID"
echo "User log: $USER_LOG"

# Small delay
sleep 5

# ========================================
# Step 2: Start Agent on Node 2
# ========================================
echo "=== Step 2: Starting Agent (32B) on $AGENT_NODE ==="

AGENT_LOG="$SCRIPT_DIR/logs/gaudi_agent_32b_${SLURM_JOB_ID}.log"

srun --nodes=1 --ntasks=1 -w $AGENT_NODE bash -c "
    # Source cache redirects
    source /data/sse/gaudi/scripts/cache-redirects.sh /scratch/\$USER 2>/dev/null || true

    export APPTAINER_CACHEDIR='/scratch/\$USER/apptainer_cache'
    export APPTAINER_TMPDIR='/scratch/\$USER/apptainer_tmp'
    export HF_HOME='/scratch/\$USER/hf_cache'
    export HABANA_VISIBLE_DEVICES=all
    export PT_HPU_LAZY_MODE=0
    export PT_HPU_ENABLE_LAZY_COLLECTIVES=True
    export VLLM_SKIP_WARMUP=True

    mkdir -p \$HF_HOME $WORK_DIR/agent_logs

    echo 'Starting Agent (32B) on '\$(hostname)'...'
    echo 'Log file: $AGENT_LOG'

    cd $VLLM_CD

    apptainer exec \\
        --bind /usr/lib64:/host-lib64 \\
        --bind /usr/lib/habanalabs:/usr/lib/habanalabs \\
        --bind /opt/habanalabs:/opt/habanalabs \\
        --bind /usr/bin/shim_ctl:/usr/bin/shim_ctl \\
        --bind \$HF_HOME:/mnt/hf_cache \\
        --bind \$(pwd):/workspace/.cd \\
        --bind $WORK_DIR/agent_logs:/var/log/habana_logs \\
        --pwd /workspace/.cd \\
        --writable-tmpfs \\
        $CONTAINER \\
        vllm serve $AGENT_MODEL \\
            --host 0.0.0.0 \\
            --port $AGENT_PORT \\
            --block-size 128 \\
            --dtype bfloat16 \\
            --tensor-parallel-size 1 \\
            --download-dir /mnt/hf_cache \\
            --max-model-len $MAX_MODEL_LEN \\
            --gpu-memory-utilization 0.90 \\
            --use-padding-aware-scheduling \\
            --max-num-seqs $MAX_NUM_SEQS \\
            --max-num-prefill-seqs 4 \\
            --num-scheduler-steps 1 \\
            --disable-log-requests \\
            --enable-auto-tool-choice \\
            --tool-call-parser hermes
" > "$AGENT_LOG" 2>&1 &

AGENT_SRUN_PID=$!
echo "Agent srun PID: $AGENT_SRUN_PID"
echo "Agent log: $AGENT_LOG"
echo ""

# ========================================
# Step 3: Wait for Servers to be Ready
# ========================================
echo "=== Step 3: Waiting for servers to be ready ==="

check_server() {
    curl -s --connect-timeout 5 "${1}/health" > /dev/null 2>&1
    return $?
}

# Wait for User Simulator
echo -n "Waiting for User Simulator (${USER_URL})..."
USER_READY=0
for i in {1..120}; do
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
    echo "User Simulator did not start. Last 50 lines of log:"
    tail -50 "$USER_LOG"
    exit 1
fi

# Wait for Agent
echo -n "Waiting for Agent (${AGENT_URL})..."
AGENT_READY=0
for i in {1..120}; do
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
    echo "Agent did not start. Last 50 lines of log:"
    tail -50 "$AGENT_LOG"
    exit 1
fi

echo ""
echo "Both servers are ready!"
echo ""

# ========================================
# Step 4: Install tau-bench and Run Experiment
# ========================================
echo "=== Step 4: Running Experiment ==="

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

export OPENAI_API_KEY="dummy"

echo "Working directory: $(pwd)"
echo "User URL: $USER_URL"
echo "Agent URL: $AGENT_URL"
echo ""

# Results directory
LOG_DIR="$SCRIPT_DIR/results_gaudi/${ENV}/${STRATEGY}"
mkdir -p "$LOG_DIR"

echo ">>> Running Environment: $ENV, Strategy: $STRATEGY"
echo "    Trials: $NUM_TRIALS"
echo "    Max Concurrency: $MAX_CONCURRENCY"
echo ""

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
echo "User Model: $USER_MODEL"
echo "Agent Model: $AGENT_MODEL"
echo "Trials: $NUM_TRIALS"
echo ""
echo "Results saved to: $LOG_DIR"
echo ""

echo "========================================"
echo "=== Gaudi 32B Retail Act Complete ==="
echo "========================================"
echo "Finished at: $(date)"
