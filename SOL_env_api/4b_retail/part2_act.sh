#!/bin/bash
#SBATCH --job-name=4b-retail-act-api-p2-tau-gaudi
#SBATCH --partition=gaudi
#SBATCH --qos=class_gaudi
#SBATCH --account=class_cse59827694spring2026
#SBATCH --nodes=1
#SBATCH --gres=gpu:hl225:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=10:00:00
#SBATCH --output=4b-retail-act-api-p2-tau-gaudi_%j.out
#SBATCH --error=4b-retail-act-api-p2-tau-gaudi_%j.err
#SBATCH --exclusive

set -e
SCRIPT_DIR="${SLURM_SUBMIT_DIR}/SOL_env_api/4b_retail"
REPO_ROOT="${SLURM_SUBMIT_DIR}"
mkdir -p "$SCRIPT_DIR/logs"
exec > >(tee -a "$SCRIPT_DIR/logs/tau-gaudi-4b-retail-act-api-p2_${SLURM_JOB_ID}.out") 2>&1
exec 2> >(tee -a "$SCRIPT_DIR/logs/tau-gaudi-4b-retail-act-api-p2_${SLURM_JOB_ID}.err" >&2)

echo "========================================"; echo "=== Gaudi 4B Retail Act (API User) Part 2/3 (Tasks 40-79) ==="; echo "========================================"
echo "Started at: $(date)"; echo "Job ID: $SLURM_JOB_ID"; echo "Node: $(hostname)"

hl-smi || echo "hl-smi not available yet"
source /data/sse/gaudi/scripts/cache-redirects.sh /scratch/$USER
export APPTAINER_CACHEDIR="/scratch/$USER/apptainer_cache"
export APPTAINER_TMPDIR="/scratch/$USER/apptainer_tmp"
export HF_HOME="/scratch/$USER/hf_cache"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$HF_HOME"

AGENT_MODEL="Qwen/Qwen3-4B"
USER_MODEL="qwen3-235b-a22b-instruct-2507"
API_BASE_URL="https://openai.rc.asu.edu/v1"
# Dynamic port for agent vLLM server
AGENT_PORT=$((20000 + (SLURM_JOB_ID % 10000)))
MAX_MODEL_LEN=40960
ENV="retail"
STRATEGY="act"
NUM_TRIALS=5
MAX_CONCURRENCY=2
PART_NUM=2
TASK_RANGE="40-79"

# Batch configuration for Part 2 (tasks 40-79)
BATCHES=("40 59" "60 79")

# === Check SOL API Key ===
if [ -z "$SOL_API_KEY" ]; then
    echo "ERROR: SOL_API_KEY environment variable not set"
    echo ""
    echo "To get an API key:"
    echo "1. Go to https://voyager.rc.asu.edu/"
    echo "2. Navigate to 'LLM Access' tab"
    echo "3. Click 'Create Key'"
    echo "4. Add to your ~/.bashrc: export SOL_API_KEY='your-key'"
    echo ""
    exit 1
fi

# Use SOL_API_KEY for all OpenAI-compatible endpoints
# (local vLLM accepts any key; SOL API requires the real key)
export OPENAI_API_KEY="$SOL_API_KEY"

GAUDI_BASE="/data/sse/gaudi"
CONTAINER="$GAUDI_BASE/containers/vllm-gaudi.sif"
VLLM_CD="$GAUDI_BASE/vllm-fork/.cd"
WORK_DIR="/scratch/$USER/gaudi_tau_bench_${SLURM_JOB_ID}"
mkdir -p "$WORK_DIR/logs"

cleanup() {
    # Graceful shutdown: SIGTERM first, wait, then SIGKILL as fallback
    [ -n "$AGENT_PID" ] && kill $AGENT_PID 2>/dev/null || true
    sleep 10  # Give processes time to shut down gracefully
    [ -n "$AGENT_PID" ] && kill -0 $AGENT_PID 2>/dev/null && kill -9 $AGENT_PID 2>/dev/null || true
    fuser -k $AGENT_PORT/tcp 2>/dev/null || true
    pkill -9 -f "vllm serve.*${AGENT_PORT}" 2>/dev/null || true
    pkill -9 -f "apptainer exec.*vllm-gaudi" 2>/dev/null || true
    sleep 5
    # Final sweep: kill any python3 processes on HPU devices
    for pid in $(hl-smi 2>/dev/null | grep -oP 'pid=\K[0-9]+' | sort -u); do
        kill -9 "$pid" 2>/dev/null || true
    done
}

trap cleanup EXIT INT TERM

export APPTAINERENV_HF_HOME=/mnt/hf_cache
export APPTAINERENV_PT_HPU_LAZY_MODE=0
export APPTAINERENV_PT_HPU_ENABLE_LAZY_COLLECTIVES=True
export APPTAINERENV_VLLM_SKIP_WARMUP=True
export APPTAINERENV_VLLM_DELAYED_SAMPLING=True
export APPTAINERENV_PYTHONUNBUFFERED=1
export APPTAINERENV_VLLM_PROMPT_BS_BUCKET_MIN=1
export APPTAINERENV_VLLM_PROMPT_BS_BUCKET_STEP=32
export APPTAINERENV_VLLM_DECODE_BS_BUCKET_MIN=1
export APPTAINERENV_VLLM_DECODE_BS_BUCKET_STEP=32
export APPTAINERENV_VLLM_PROMPT_SEQ_BUCKET_MIN=128
export APPTAINERENV_VLLM_PROMPT_SEQ_BUCKET_STEP=256
export APPTAINERENV_VLLM_DECODE_BLOCK_BUCKET_MIN=128
export APPTAINERENV_VLLM_DECODE_BLOCK_BUCKET_STEP=256

check_server() { curl -s --connect-timeout 5 "http://localhost:${1}/health" > /dev/null 2>&1; return $?; }

# Setup conda environment once before batch loop
module load mamba/latest
source activate tau-bench 2>/dev/null || { mamba create -n tau-bench -c conda-forge python=3.11 -y; source activate tau-bench; }
cd "$REPO_ROOT"; pip uninstall tau_bench -y 2>/dev/null || true; pip install -e .

LOG_DIR="$SCRIPT_DIR/results_gaudi_api/${ENV}/${STRATEGY}"
mkdir -p "$LOG_DIR"

AGENT_URL="http://localhost:${AGENT_PORT}/v1"

# Test SOL API connection
echo "=== Testing SOL API Connection ==="
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" \
    -H "Authorization: Bearer $SOL_API_KEY" \
    "$API_BASE_URL/models" 2>/dev/null)
if [ "$RESPONSE" = "200" ]; then
    echo "SOL API connection successful!"
else
    echo "WARNING: SOL API connection test returned HTTP $RESPONSE"
    echo "Continuing anyway (API may still work for completions)"
fi
echo ""

# Run experiments in batches with server restart between batches
# Fail-fast: abort if consecutive batches fail (HPU devices likely stuck)
CONSECUTIVE_FAILURES=0
MAX_CONSECUTIVE_FAILURES=2
SUCCESSFUL_BATCHES=0
BATCH_NUM=0
for BATCH in "${BATCHES[@]}"; do
    read START_IDX END_IDX <<< "$BATCH"
    BATCH_NUM=$((BATCH_NUM + 1))
    echo ""
    echo "=============================================="
    echo "=== BATCH ${BATCH_NUM}/${#BATCHES[@]}: Tasks ${START_IDX} to ${END_IDX} ==="
    echo "=============================================="
    echo "Batch started at: $(date)"

    cd "$VLLM_CD"

    # Only start the Agent server locally; User model is served by SOL API
    echo "=== Starting Agent Model Server (4B) ==="
    AGENT_LOG="$SCRIPT_DIR/logs/gaudi_vllm_agent_4b_${SLURM_JOB_ID}_batch${BATCH_NUM}.log"
    export APPTAINERENV_HABANA_VISIBLE_DEVICES=2
    apptainer exec --bind /usr/lib64:/host-lib64 --bind /usr/lib/habanalabs:/usr/lib/habanalabs --bind /opt/habanalabs:/opt/habanalabs --bind /usr/bin/shim_ctl:/usr/bin/shim_ctl --bind "$HF_HOME:/mnt/hf_cache" --bind "$(pwd):/workspace/.cd" --bind "$WORK_DIR/logs:/var/log/habana_logs" --pwd /workspace/.cd --writable-tmpfs "$CONTAINER" vllm serve "$AGENT_MODEL" --host 0.0.0.0 --port $AGENT_PORT --block-size 128 --dtype bfloat16 --tensor-parallel-size 1 --download-dir /mnt/hf_cache --max-model-len $MAX_MODEL_LEN --gpu-memory-utilization 0.85 --use-padding-aware-scheduling --max-num-seqs 16 --max-num-prefill-seqs 8 --num-scheduler-steps 1 --disable-log-requests --enable-auto-tool-choice --tool-call-parser hermes --swap-space 16 > "$AGENT_LOG" 2>&1 &
    AGENT_PID=$!

    # Wait for agent server with skip-on-failure instead of exit 1
    BATCH_SKIP=false

    echo -n "Waiting for Agent server (4B)..."
    for i in {1..90}; do
        if check_server "$AGENT_PORT"; then echo " Ready! (${i}0s)"; break; fi
        if ! kill -0 $AGENT_PID 2>/dev/null; then
            echo " FAILED!"
            tail -100 "$AGENT_LOG"
            echo "WARNING: Agent server failed to start for batch ${BATCH_NUM}, skipping..."
            BATCH_SKIP=true
            break
        fi
        echo -n "."; sleep 10
    done
    if [ "$BATCH_SKIP" = false ] && ! check_server "$AGENT_PORT"; then
        echo " TIMEOUT!"
        tail -100 "$AGENT_LOG"
        echo "WARNING: Agent server timed out for batch ${BATCH_NUM}, skipping..."
        BATCH_SKIP=true
    fi

    if [ "$BATCH_SKIP" = false ]; then
        echo "Agent server ready! User model via SOL API: ${USER_MODEL}"

        cd "$REPO_ROOT"
        python run.py --env ${ENV} --agent-strategy ${STRATEGY} \
            --model ${AGENT_MODEL} --model-provider openai --model-base-url ${AGENT_URL} \
            --user-model ${USER_MODEL} --user-model-provider openai --user-model-base-url ${API_BASE_URL} \
            --log-dir ${LOG_DIR} --max-concurrency ${MAX_CONCURRENCY} --num-trials ${NUM_TRIALS} \
            --start-index ${START_IDX} --end-index $((END_IDX + 1)) || echo "WARNING: Batch ${BATCH_NUM} run.py exited with non-zero status, continuing to next batch..."

        # Wait for vLLM to finish processing any queued requests
        echo "Batch ${BATCH_NUM} execution complete, waiting for vLLM to finish processing..."
        sleep 30

        # Check if server is still responsive
        echo "Checking server status before cleanup..."
        check_server "$AGENT_PORT" && echo "Agent server still responsive" || echo "Agent server not responding"

        # Rename output file to include batch info for clean merging
        EXPECTED_RANGE="range_${START_IDX}-$((END_IDX + 1))"
        LATEST=$(ls -t "$LOG_DIR"/*.json 2>/dev/null | grep "$EXPECTED_RANGE" | grep -v "_batch" | grep -v "_merged" | grep -v "_part" | head -1)
        if [ -n "$LATEST" ]; then
            BATCH_FILE="${LATEST%.json}_part${PART_NUM}_batch${BATCH_NUM}_job${SLURM_JOB_ID}.json"
            mv "$LATEST" "$BATCH_FILE"
            echo "Batch ${BATCH_NUM} results saved to: $BATCH_FILE"
        fi

        echo "Batch ${BATCH_NUM} completed at: $(date)"
        CONSECUTIVE_FAILURES=0
        SUCCESSFUL_BATCHES=$((SUCCESSFUL_BATCHES + 1))
    else
        echo "SKIPPING batch ${BATCH_NUM} due to server startup failure"
        CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))
        echo "Consecutive failures: ${CONSECUTIVE_FAILURES}/${MAX_CONSECUTIVE_FAILURES}"
        if [ "$CONSECUTIVE_FAILURES" -ge "$MAX_CONSECUTIVE_FAILURES" ]; then
            echo ""
            echo "ABORTING: ${MAX_CONSECUTIVE_FAILURES} consecutive batch failures detected."
            echo "HPU devices are likely stuck. Remaining batches would also fail."
            echo "Completed ${SUCCESSFUL_BATCHES}/${#BATCHES[@]} batches successfully."
            break
        fi
    fi

    # Kill server before next batch (cleanup clears memory fragmentation)
    echo "Stopping server for memory cleanup..."
    echo "Giving server 5 seconds to finish any final requests..."
    sleep 5
    cleanup
    AGENT_PID=""
    # Kill any remaining vLLM worker processes owned by this user
    pkill -9 -u $USER -f "vllm.entrypoints" 2>/dev/null || true
    # Wait for HPU devices to release memory (poll every 15s, max 180s)
    echo "Waiting for HPU devices to release memory..."
    for hpu_wait in $(seq 1 12); do
        sleep 15
        if ! pgrep -u $USER -f "vllm" > /dev/null 2>&1; then
            echo "All vLLM processes exited after $((hpu_wait * 15))s"
            break
        fi
        pkill -9 -u $USER -f "vllm" 2>/dev/null || true
        echo -n "."
    done
    # Force-kill any orphaned python3 processes still holding HPU memory
    for hpu_pid in $(hl-smi 2>/dev/null | awk '/python3/{print $3}'); do
        echo "Killing orphaned HPU process: $hpu_pid"
        kill -9 "$hpu_pid" 2>/dev/null || true
    done
    # Allow HPU memory to fully deallocate
    sleep 30
    echo "Checking HPU device status..."
    hl-smi || echo "WARNING: hl-smi not available, continuing anyway"
done

echo ""
if [ "$SUCCESSFUL_BATCHES" -eq 0 ]; then
    echo "FAILED: No batches completed successfully. This part produced no usable results."
    echo "Experiment failed at: $(date)"
    exit 1
else
    echo "Part ${PART_NUM} (tasks ${TASK_RANGE}) complete: ${SUCCESSFUL_BATCHES}/${#BATCHES[@]} batches succeeded."
    echo "Use merge_results.py to combine all parts."
    echo "All part results saved to: $LOG_DIR"
    echo "Experiment finished at: $(date)"
fi
