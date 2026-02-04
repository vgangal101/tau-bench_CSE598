#!/bin/bash
#SBATCH --job-name=8b-airline-act-tau-gaudi
#SBATCH --partition=gaudi
#SBATCH --qos=class_gaudi
#SBATCH --account=class_cse59827694spring2026
#SBATCH --nodes=1
#SBATCH --gres=gpu:hl225:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --output=8b-airline-act-tau-gaudi_%j.out
#SBATCH --error=8b-airline-act-tau-gaudi_%j.err
#SBATCH --exclusive

set -e
SCRIPT_DIR="${SLURM_SUBMIT_DIR}/SOL_env/8b_run"
REPO_ROOT="${SLURM_SUBMIT_DIR}"
mkdir -p "$SCRIPT_DIR/logs"
exec > >(tee -a "$SCRIPT_DIR/logs/tau-gaudi-8b-airline-act_${SLURM_JOB_ID}.out") 2>&1
exec 2> >(tee -a "$SCRIPT_DIR/logs/tau-gaudi-8b-airline-act_${SLURM_JOB_ID}.err" >&2)

echo "========================================"; echo "=== Gaudi 8B Airline Act Experiment ==="; echo "========================================"
echo "Started at: $(date)"; echo "Job ID: $SLURM_JOB_ID"; echo "Node: $(hostname)"

hl-smi || echo "hl-smi not available yet"
source /data/sse/gaudi/scripts/cache-redirects.sh /scratch/$USER
export APPTAINER_CACHEDIR="/scratch/$USER/apptainer_cache"
export APPTAINER_TMPDIR="/scratch/$USER/apptainer_tmp"
export HF_HOME="/scratch/$USER/hf_cache"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$HF_HOME"

USER_MODEL="Qwen/Qwen3-32B"
AGENT_MODEL="Qwen/Qwen3-8B"
USER_PORT=8301
AGENT_PORT=8101
MAX_MODEL_LEN=32768
ENV="airline"
STRATEGY="act"
NUM_TRIALS=5
MAX_CONCURRENCY=3

GAUDI_BASE="/data/sse/gaudi"
CONTAINER="$GAUDI_BASE/containers/vllm-gaudi.sif"
VLLM_CD="$GAUDI_BASE/vllm-fork/.cd"
WORK_DIR="/scratch/$USER/gaudi_tau_bench_${SLURM_JOB_ID}"
mkdir -p "$WORK_DIR/logs"

cleanup() { [ -n "$USER_PID" ] && kill $USER_PID 2>/dev/null; [ -n "$AGENT_PID" ] && kill $AGENT_PID 2>/dev/null; fuser -k $USER_PORT/tcp 2>/dev/null || true; fuser -k $AGENT_PORT/tcp 2>/dev/null || true; }
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

cd "$VLLM_CD"

echo "=== Starting User Model Server (32B) ==="
USER_LOG="$SCRIPT_DIR/logs/gaudi_vllm_user_32b_${SLURM_JOB_ID}.log"
export APPTAINERENV_HABANA_VISIBLE_DEVICES=0,1
apptainer exec --bind /usr/lib64:/host-lib64 --bind /usr/lib/habanalabs:/usr/lib/habanalabs --bind /opt/habanalabs:/opt/habanalabs --bind /usr/bin/shim_ctl:/usr/bin/shim_ctl --bind "$HF_HOME:/mnt/hf_cache" --bind "$(pwd):/workspace/.cd" --bind "$WORK_DIR/logs:/var/log/habana_logs" --pwd /workspace/.cd --writable-tmpfs "$CONTAINER" vllm serve "$USER_MODEL" --host 0.0.0.0 --port $USER_PORT --block-size 128 --dtype bfloat16 --tensor-parallel-size 2 --download-dir /mnt/hf_cache --max-model-len $MAX_MODEL_LEN --gpu-memory-utilization 0.95 --use-padding-aware-scheduling --max-num-seqs 8 --max-num-prefill-seqs 2 --num-scheduler-steps 1 --disable-log-requests --enable-auto-tool-choice --tool-call-parser hermes > "$USER_LOG" 2>&1 &
USER_PID=$!

echo "=== Starting Agent Model Server (8B) ==="
AGENT_LOG="$SCRIPT_DIR/logs/gaudi_vllm_agent_8b_${SLURM_JOB_ID}.log"
export APPTAINERENV_HABANA_VISIBLE_DEVICES=2
apptainer exec --bind /usr/lib64:/host-lib64 --bind /usr/lib/habanalabs:/usr/lib/habanalabs --bind /opt/habanalabs:/opt/habanalabs --bind /usr/bin/shim_ctl:/usr/bin/shim_ctl --bind "$HF_HOME:/mnt/hf_cache" --bind "$(pwd):/workspace/.cd" --bind "$WORK_DIR/logs:/var/log/habana_logs" --pwd /workspace/.cd --writable-tmpfs "$CONTAINER" vllm serve "$AGENT_MODEL" --host 0.0.0.0 --port $AGENT_PORT --block-size 128 --dtype bfloat16 --tensor-parallel-size 1 --download-dir /mnt/hf_cache --max-model-len $MAX_MODEL_LEN --gpu-memory-utilization 0.90 --use-padding-aware-scheduling --max-num-seqs 16 --max-num-prefill-seqs 8 --num-scheduler-steps 1 --disable-log-requests --enable-auto-tool-choice --tool-call-parser hermes > "$AGENT_LOG" 2>&1 &
AGENT_PID=$!

check_server() { curl -s --connect-timeout 5 "http://localhost:${1}/health" > /dev/null 2>&1; return $?; }

echo -n "Waiting for User server (32B)..."
for i in {1..180}; do
    if check_server "$USER_PORT"; then echo " Ready! (${i}0s)"; break; fi
    if ! kill -0 $USER_PID 2>/dev/null; then echo " FAILED!"; tail -100 "$USER_LOG"; exit 1; fi
    echo -n "."; sleep 10
done
if ! check_server "$USER_PORT"; then echo " TIMEOUT!"; tail -100 "$USER_LOG"; exit 1; fi

echo -n "Waiting for Agent server (8B)..."
for i in {1..90}; do
    if check_server "$AGENT_PORT"; then echo " Ready! (${i}0s)"; break; fi
    if ! kill -0 $AGENT_PID 2>/dev/null; then echo " FAILED!"; tail -100 "$AGENT_LOG"; exit 1; fi
    echo -n "."; sleep 10
done
if ! check_server "$AGENT_PORT"; then echo " TIMEOUT!"; tail -100 "$AGENT_LOG"; exit 1; fi

echo "Both servers ready!"

module load mamba/latest
source activate tau-bench 2>/dev/null || { mamba create -n tau-bench -c conda-forge python=3.11 -y; source activate tau-bench; }
cd "$REPO_ROOT"; pip uninstall tau_bench -y 2>/dev/null || true; pip install -e .

export OPENAI_API_KEY="dummy"
LOG_DIR="$SCRIPT_DIR/results_gaudi/${ENV}/${STRATEGY}"
mkdir -p "$LOG_DIR"

USER_URL="http://localhost:${USER_PORT}/v1"
AGENT_URL="http://localhost:${AGENT_PORT}/v1"

python run.py --env ${ENV} --agent-strategy ${STRATEGY} --model ${AGENT_MODEL} --model-provider openai --model-base-url ${AGENT_URL} --user-model ${USER_MODEL} --user-model-provider openai --user-model-base-url ${USER_URL} --log-dir ${LOG_DIR} --max-concurrency ${MAX_CONCURRENCY} --num-trials ${NUM_TRIALS}

echo "Results saved to: $LOG_DIR"; echo "Finished at: $(date)"
