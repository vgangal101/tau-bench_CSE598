#!/bin/bash
#SBATCH --job-name=14b-airline-tc-tau-gaudi
#SBATCH --partition=gaudi
#SBATCH --qos=class_gaudi
#SBATCH --account=class_cse59827694spring2026
#SBATCH --nodes=1
#SBATCH --gres=gpu:hl225:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=14b-airline-tc-tau-gaudi_%j.out
#SBATCH --error=14b-airline-tc-tau-gaudi_%j.err
#SBATCH --exclusive

set -e
SCRIPT_DIR="${SLURM_SUBMIT_DIR}/SOL_env/14b_run"
REPO_ROOT="${SLURM_SUBMIT_DIR}"
mkdir -p "$SCRIPT_DIR/logs"
exec > >(tee -a "$SCRIPT_DIR/logs/tau-gaudi-14b-airline-tool-calling_${SLURM_JOB_ID}.out") 2>&1
exec 2> >(tee -a "$SCRIPT_DIR/logs/tau-gaudi-14b-airline-tool-calling_${SLURM_JOB_ID}.err" >&2)

echo "========================================"; echo "=== Gaudi 14B Airline Tool-Calling ==="; echo "========================================"
echo "Started at: $(date)"; echo "Job ID: $SLURM_JOB_ID"; echo "Node: $(hostname)"

hl-smi || echo "hl-smi not available yet"
source /data/sse/gaudi/scripts/cache-redirects.sh /scratch/$USER
export APPTAINER_CACHEDIR="/scratch/$USER/apptainer_cache"
export APPTAINER_TMPDIR="/scratch/$USER/apptainer_tmp"
export HF_HOME="/scratch/$USER/hf_cache"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$HF_HOME"

USER_MODEL="Qwen/Qwen3-32B"
AGENT_MODEL="Qwen/Qwen3-14B"
# Dynamic ports based on SLURM job ID to avoid conflicts on shared nodes
USER_PORT=$((10000 + (SLURM_JOB_ID % 10000)))
AGENT_PORT=$((20000 + (SLURM_JOB_ID % 10000)))
MAX_MODEL_LEN=40000
ENV="airline"
STRATEGY="tool-calling"
NUM_TRIALS=5
MAX_CONCURRENCY=2

# Batch configuration for airline (50 tasks)
BATCHES=("0 12" "13 24" "25 37" "38 49")

GAUDI_BASE="/data/sse/gaudi"
CONTAINER="$GAUDI_BASE/containers/vllm-gaudi.sif"
VLLM_CD="$GAUDI_BASE/vllm-fork/.cd"
WORK_DIR="/scratch/$USER/gaudi_tau_bench_${SLURM_JOB_ID}"
mkdir -p "$WORK_DIR/logs"

cleanup() {
    # Graceful shutdown: SIGTERM first, wait, then SIGKILL as fallback
    [ -n "$USER_PID" ] && kill $USER_PID 2>/dev/null || true
    [ -n "$AGENT_PID" ] && kill $AGENT_PID 2>/dev/null || true
    sleep 10  # Give processes time to shut down gracefully
    [ -n "$USER_PID" ] && kill -0 $USER_PID 2>/dev/null && kill -9 $USER_PID 2>/dev/null || true
    [ -n "$AGENT_PID" ] && kill -0 $AGENT_PID 2>/dev/null && kill -9 $AGENT_PID 2>/dev/null || true
    fuser -k $USER_PORT/tcp 2>/dev/null || true
    fuser -k $AGENT_PORT/tcp 2>/dev/null || true
    # Kill ALL python3/vllm processes holding HPU devices (apptainer children)
    pkill -9 -f "vllm serve.*${USER_PORT}" 2>/dev/null || true
    pkill -9 -f "vllm serve.*${AGENT_PORT}" 2>/dev/null || true
    # Kill any remaining apptainer/python3 processes from this job
    pkill -9 -f "apptainer exec.*vllm-gaudi" 2>/dev/null || true
    sleep 5
    # Final sweep: kill any python3 processes on HPU devices
    for pid in $(hl-smi 2>/dev/null | grep -oP 'pid=\K[0-9]+' | sort -u); do
        kill -9 "$pid" 2>/dev/null || true
    done
}

wait_for_hpu_release() {
    # Poll hl-smi until all HPUs show base memory (~768MiB) or timeout
    local max_wait=300  # 5 minutes max
    local interval=15
    local elapsed=0
    echo "Polling HPU device status until memory is released (max ${max_wait}s)..."
    while [ $elapsed -lt $max_wait ]; do
        # Check if any HPU still has high memory usage (>2000 MiB = still occupied)
        local busy_hpus=$(hl-smi 2>/dev/null | grep "MiB" | awk '{print $5}' | sed 's/MiB//' | awk '$1 > 2000 {count++} END {print count+0}')
        if [ "$busy_hpus" -eq 0 ]; then
            echo "All HPU devices released after ${elapsed}s"
            hl-smi 2>/dev/null || true
            return 0
        fi
        echo "  ${busy_hpus} HPU(s) still occupied after ${elapsed}s, waiting..."
        sleep $interval
        elapsed=$((elapsed + interval))
    done
    echo "WARNING: HPU devices still not fully released after ${max_wait}s"
    hl-smi 2>/dev/null || true
    # Last resort: kill any remaining python3 processes on HPUs
    for pid in $(hl-smi 2>/dev/null | grep -oP 'pid=\K[0-9]+' | sort -u); do
        echo "Force-killing HPU process pid=$pid"
        kill -9 "$pid" 2>/dev/null || true
    done
    sleep 30
    return 1
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

export OPENAI_API_KEY="dummy"
LOG_DIR="$SCRIPT_DIR/results_gaudi/${ENV}/${STRATEGY}"
mkdir -p "$LOG_DIR"

USER_URL="http://localhost:${USER_PORT}/v1"
AGENT_URL="http://localhost:${AGENT_PORT}/v1"

# Run experiments in batches with server restart between batches
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

    echo "=== Starting User Model Server (32B) ==="
    USER_LOG="$SCRIPT_DIR/logs/gaudi_vllm_user_32b_${SLURM_JOB_ID}_batch${BATCH_NUM}.log"
    export APPTAINERENV_HABANA_VISIBLE_DEVICES=0,1
    apptainer exec --bind /usr/lib64:/host-lib64 --bind /usr/lib/habanalabs:/usr/lib/habanalabs --bind /opt/habanalabs:/opt/habanalabs --bind /usr/bin/shim_ctl:/usr/bin/shim_ctl --bind "$HF_HOME:/mnt/hf_cache" --bind "$(pwd):/workspace/.cd" --bind "$WORK_DIR/logs:/var/log/habana_logs" --pwd /workspace/.cd --writable-tmpfs "$CONTAINER" vllm serve "$USER_MODEL" --host 0.0.0.0 --port $USER_PORT --block-size 128 --dtype bfloat16 --tensor-parallel-size 2 --download-dir /mnt/hf_cache --max-model-len $MAX_MODEL_LEN --gpu-memory-utilization 0.85 --use-padding-aware-scheduling --max-num-seqs 2 --max-num-prefill-seqs 1 --num-scheduler-steps 1 --disable-log-requests --enable-auto-tool-choice --tool-call-parser hermes --swap-space 16 > "$USER_LOG" 2>&1 &
    USER_PID=$!

    echo "=== Starting Agent Model Server (14B) ==="
    AGENT_LOG="$SCRIPT_DIR/logs/gaudi_vllm_agent_14b_${SLURM_JOB_ID}_batch${BATCH_NUM}.log"
    export APPTAINERENV_HABANA_VISIBLE_DEVICES=2
    apptainer exec --bind /usr/lib64:/host-lib64 --bind /usr/lib/habanalabs:/usr/lib/habanalabs --bind /opt/habanalabs:/opt/habanalabs --bind /usr/bin/shim_ctl:/usr/bin/shim_ctl --bind "$HF_HOME:/mnt/hf_cache" --bind "$(pwd):/workspace/.cd" --bind "$WORK_DIR/logs:/var/log/habana_logs" --pwd /workspace/.cd --writable-tmpfs "$CONTAINER" vllm serve "$AGENT_MODEL" --host 0.0.0.0 --port $AGENT_PORT --block-size 128 --dtype bfloat16 --tensor-parallel-size 1 --download-dir /mnt/hf_cache --max-model-len $MAX_MODEL_LEN --gpu-memory-utilization 0.85 --use-padding-aware-scheduling --max-num-seqs 12 --max-num-prefill-seqs 6 --num-scheduler-steps 1 --disable-log-requests --enable-auto-tool-choice --tool-call-parser hermes --swap-space 16 > "$AGENT_LOG" 2>&1 &
    AGENT_PID=$!

    BATCH_SKIP=false
    echo -n "Waiting for User server (32B)..."
    for i in {1..180}; do
        if check_server "$USER_PORT"; then echo " Ready! (${i}0s)"; break; fi
        if ! kill -0 $USER_PID 2>/dev/null; then echo " FAILED!"; tail -100 "$USER_LOG"; BATCH_SKIP=true; break; fi
        echo -n "."; sleep 10
    done
    if [ "$BATCH_SKIP" = false ] && ! check_server "$USER_PORT"; then echo " TIMEOUT!"; tail -100 "$USER_LOG"; BATCH_SKIP=true; fi

    if [ "$BATCH_SKIP" = false ]; then
    echo -n "Waiting for Agent server (14B)..."
    for i in {1..120}; do
        if check_server "$AGENT_PORT"; then echo " Ready! (${i}0s)"; break; fi
        if ! kill -0 $AGENT_PID 2>/dev/null; then echo " FAILED!"; tail -100 "$AGENT_LOG"; BATCH_SKIP=true; break; fi
        echo -n "."; sleep 10
    done
    if [ "$BATCH_SKIP" = false ] && ! check_server "$AGENT_PORT"; then echo " TIMEOUT!"; tail -100 "$AGENT_LOG"; BATCH_SKIP=true; fi
    fi

    if [ "$BATCH_SKIP" = false ]; then
    echo "Both servers ready!"

    cd "$REPO_ROOT"
    python run.py --env ${ENV} --agent-strategy ${STRATEGY} \
        --model ${AGENT_MODEL} --model-provider openai --model-base-url ${AGENT_URL} \
        --user-model ${USER_MODEL} --user-model-provider openai --user-model-base-url ${USER_URL} \
        --log-dir ${LOG_DIR} --max-concurrency ${MAX_CONCURRENCY} --num-trials ${NUM_TRIALS} \
        --start-index ${START_IDX} --end-index $((END_IDX + 1)) || echo "WARNING: Batch ${BATCH_NUM} run.py exited with non-zero status, continuing to next batch..."

    # Wait for vLLM to finish processing any queued requests
    echo "Batch ${BATCH_NUM} execution complete, waiting for vLLM to finish processing..."
    sleep 30

    # Check if servers are still responsive
    echo "Checking server status before cleanup..."
    check_server "$USER_PORT" && echo "User server still responsive" || echo "User server not responding"
    check_server "$AGENT_PORT" && echo "Agent server still responsive" || echo "Agent server not responding"

    # Rename output file to include batch info for clean merging
    LATEST=$(ls -t "$LOG_DIR"/*.json 2>/dev/null | grep -v "_batch" | grep -v "_merged" | head -1)
    if [ -n "$LATEST" ]; then
        BATCH_FILE="${LATEST%.json}_batch${BATCH_NUM}_job${SLURM_JOB_ID}.json"
        mv "$LATEST" "$BATCH_FILE"
        echo "Batch ${BATCH_NUM} results saved to: $BATCH_FILE"
    fi

    echo "Batch ${BATCH_NUM} completed at: $(date)"
    else
    echo "SKIPPING batch ${BATCH_NUM} due to server startup failure"
    fi

    # Kill servers before next batch (cleanup clears memory fragmentation)
    echo "Stopping servers for memory cleanup..."
    echo "Giving servers 5 seconds to finish any final requests..."
    sleep 5
    cleanup
    USER_PID=""
    AGENT_PID=""
    # Poll for HPU device release instead of fixed sleep
    wait_for_hpu_release
done

# Merge all batch results from this job
echo ""
echo "=============================================="
echo "=== Merging batch results for Job $SLURM_JOB_ID ==="
echo "=============================================="
TOTAL_TASKS=50
FINAL_RESULTS="$LOG_DIR/${STRATEGY}-${AGENT_MODEL##*/}-0.0_range_0-${TOTAL_TASKS}_job${SLURM_JOB_ID}_merged.json"
python3 -c "
import json, glob, sys
# Only merge files from THIS job
files = sorted(glob.glob('$LOG_DIR/*_job${SLURM_JOB_ID}.json'))
if not files:
    print('No batch files found to merge')
    sys.exit(0)
all_results = []
for f in files:
    with open(f) as fp:
        data = json.load(fp)
        if isinstance(data, list):
            all_results.extend(data)
        else:
            all_results.append(data)
with open('$FINAL_RESULTS', 'w') as fp:
    json.dump(all_results, fp, indent=2)
print(f'Merged {len(files)} batch files ({len(all_results)} results) into: $FINAL_RESULTS')
"

echo "All results saved to: $LOG_DIR"
echo "Experiment finished at: $(date)"
