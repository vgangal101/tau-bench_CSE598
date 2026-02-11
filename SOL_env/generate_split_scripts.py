#!/usr/bin/env python3
"""Generate split experiment scripts for all model sizes and environments.

Each experiment is split into independent SLURM parts so that if one part fails,
the others still succeed. This replaces the old monolithic scripts in *_run/ dirs.

Usage:
    python generate_split_scripts.py          # Generate all 7 directories
    python generate_split_scripts.py --dry-run # Preview what would be created
"""

import argparse
import os
import stat
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Configuration Tables ───────────────────────────────────────────────────

STRATEGIES = ["act", "react", "tool-calling"]

# Hardware configs per model size
HARDWARE = {
    "4b": {
        "agent_model": "Qwen/Qwen3-4B",
        "hpus": 3,
        "gres": "gpu:hl225:3",
        "cpus": 24,
        "mem": "160G",
        "user_tp": 2,
        "agent_tp": 1,
        "user_hpus": "0,1",
        "agent_hpus": "2",
        "agent_max_num_seqs": 16,
        "agent_max_num_prefill_seqs": 8,
        "agent_wait_iters": 90,
        "gpu_mem_util": "0.85",
    },
    "8b": {
        "agent_model": "Qwen/Qwen3-8B",
        "hpus": 3,
        "gres": "gpu:hl225:3",
        "cpus": 24,
        "mem": "160G",
        "user_tp": 2,
        "agent_tp": 1,
        "user_hpus": "0,1",
        "agent_hpus": "2",
        "agent_max_num_seqs": 16,
        "agent_max_num_prefill_seqs": 8,
        "agent_wait_iters": 90,
        "gpu_mem_util": "0.85",
    },
    "14b": {
        "agent_model": "Qwen/Qwen3-14B",
        "hpus": 3,
        "gres": "gpu:hl225:3",
        "cpus": 24,
        "mem": "160G",
        "user_tp": 2,
        "agent_tp": 1,
        "user_hpus": "0,1",
        "agent_hpus": "2",
        "agent_max_num_seqs": 16,
        "agent_max_num_prefill_seqs": 8,
        "agent_wait_iters": 90,
        "gpu_mem_util": "0.85",
    },
    "32b": {
        "agent_model": "Qwen/Qwen3-32B",
        "hpus": 8,
        "gres": "gpu:hl225:8",
        "cpus": 60,
        "mem": "384G",
        "user_tp": 4,
        "agent_tp": 4,
        "user_hpus": "0,1,2,3",
        "agent_hpus": "4,5,6,7",
        "agent_max_num_seqs": 2,
        "agent_max_num_prefill_seqs": 1,
        "agent_wait_iters": 180,
        "gpu_mem_util": "0.80",
        "max_concurrency": 1,
    },
}

# Time limits per (model_size, env) → per-part time
TIME_LIMITS = {
    ("4b", "airline"): "8:00:00",
    ("4b", "retail"): "10:00:00",
    ("8b", "airline"): "8:00:00",
    ("8b", "retail"): "10:00:00",
    ("14b", "airline"): "10:00:00",
    ("14b", "retail"): "12:00:00",
    ("32b", "airline"): "6:00:00",
    ("32b", "retail"): "8:00:00",
}

# Batch splits per environment
# Airline: 50 tasks → 2 parts, 2 batches each
AIRLINE_PARTS = {
    1: {"task_range": "0-24", "batches": [("0", "12"), ("13", "24")]},
    2: {"task_range": "25-49", "batches": [("25", "37"), ("38", "49")]},
}

# Airline 32B: 50 tasks → 8 parts, 1 batch each (~6-7 tasks per part)
# HPU devices can't be reacquired after vLLM shutdown within same SLURM job
# (synStatus=8 [Device not found]), so each part runs exactly 1 batch and exits.
# Smaller parts reduce exposure to vLLM server degradation over long runs.
AIRLINE_32B_PARTS = {
    1: {"task_range": "0-6", "batches": [("0", "6")]},
    2: {"task_range": "7-12", "batches": [("7", "12")]},
    3: {"task_range": "13-18", "batches": [("13", "18")]},
    4: {"task_range": "19-24", "batches": [("19", "24")]},
    5: {"task_range": "25-31", "batches": [("25", "31")]},
    6: {"task_range": "32-37", "batches": [("32", "37")]},
    7: {"task_range": "38-43", "batches": [("38", "43")]},
    8: {"task_range": "44-49", "batches": [("44", "49")]},
}

# Retail (4B/8B/14B): 115 tasks → 3 parts, 2 batches each
RETAIL_PARTS = {
    1: {"task_range": "0-39", "batches": [("0", "19"), ("20", "39")]},
    2: {"task_range": "40-79", "batches": [("40", "59"), ("60", "79")]},
    3: {"task_range": "80-114", "batches": [("80", "99"), ("100", "114")]},
}

# Retail 32B: 115 tasks → 12 parts, 1 batch each (~10 tasks per part)
# Same HPU device reacquisition issue as airline 32B.
# Smaller parts reduce exposure to vLLM server degradation over long runs.
RETAIL_32B_PARTS = {
    1: {"task_range": "0-9", "batches": [("0", "9")]},
    2: {"task_range": "10-19", "batches": [("10", "19")]},
    3: {"task_range": "20-29", "batches": [("20", "29")]},
    4: {"task_range": "30-39", "batches": [("30", "39")]},
    5: {"task_range": "40-49", "batches": [("40", "49")]},
    6: {"task_range": "50-59", "batches": [("50", "59")]},
    7: {"task_range": "60-69", "batches": [("60", "69")]},
    8: {"task_range": "70-79", "batches": [("70", "79")]},
    9: {"task_range": "80-89", "batches": [("80", "89")]},
    10: {"task_range": "90-99", "batches": [("90", "99")]},
    11: {"task_range": "100-109", "batches": [("100", "109")]},
    12: {"task_range": "110-114", "batches": [("110", "114")]},
}

# What to generate
EXPERIMENTS = []
for size in ["4b", "8b", "14b", "32b"]:
    for env in ["airline", "retail"]:
        EXPERIMENTS.append((size, env))


# ─── Template: Part Script ──────────────────────────────────────────────────


def generate_part_script(
    model_size: str,
    env: str,
    strategy: str,
    part_num: int,
    parts_config: dict,
) -> str:
    """Generate a part script from the proven 32b_retail template."""
    hw = HARDWARE[model_size]
    time_limit = TIME_LIMITS[(model_size, env)]
    part = parts_config[part_num]
    task_range = part["task_range"]
    batches = part["batches"]
    total_parts = len(parts_config)
    total_tasks = 50 if env == "airline" else 115
    dir_name = f"{model_size}_{env}"

    # Format batches array for bash
    batches_str = " ".join(f'"{s} {e}"' for s, e in batches)

    job_name = f"{model_size}-{env}-{strategy}-p{part_num}-tau-gaudi"
    short_model = model_size.upper()

    # Agent server description includes HPU info for 32B (multi-HPU)
    if model_size == "32b":
        user_desc = f"32B on HPUs {hw['user_hpus']}"
        agent_desc = f"32B on HPUs {hw['agent_hpus']}"
    else:
        user_desc = "32B"
        agent_desc = short_model

    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition=gaudi
#SBATCH --qos=public
#SBATCH --account=grp_jzou22
#SBATCH --nodes=1
#SBATCH --gres={hw['gres']}
#SBATCH --cpus-per-task={hw['cpus']}
#SBATCH --mem={hw['mem']}
#SBATCH --time={time_limit}
#SBATCH --output={job_name}_%j.out
#SBATCH --error={job_name}_%j.err
#SBATCH --exclusive

set -e
SCRIPT_DIR="${{SLURM_SUBMIT_DIR}}/SOL_env/{dir_name}"
REPO_ROOT="${{SLURM_SUBMIT_DIR}}"
mkdir -p "$SCRIPT_DIR/logs"
exec > >(tee -a "$SCRIPT_DIR/logs/tau-gaudi-{model_size}-{env}-{strategy}-p{part_num}_${{SLURM_JOB_ID}}.out") 2>&1
exec 2> >(tee -a "$SCRIPT_DIR/logs/tau-gaudi-{model_size}-{env}-{strategy}-p{part_num}_${{SLURM_JOB_ID}}.err" >&2)

echo "========================================"; echo "=== Gaudi {short_model} {env.title()} {strategy.title()} Part {part_num}/{total_parts} (Tasks {task_range}) ==="; echo "========================================"
echo "Started at: $(date)"; echo "Job ID: $SLURM_JOB_ID"; echo "Node: $(hostname)"

hl-smi || echo "hl-smi not available yet"
source /data/sse/gaudi/scripts/cache-redirects.sh /scratch/$USER
export APPTAINER_CACHEDIR="/scratch/$USER/apptainer_cache"
export APPTAINER_TMPDIR="/scratch/$USER/apptainer_tmp"
export HF_HOME="/scratch/$USER/hf_cache"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$HF_HOME"

USER_MODEL="Qwen/Qwen3-32B"
AGENT_MODEL="{hw['agent_model']}"
# Dynamic ports based on SLURM job ID to avoid conflicts on shared nodes
USER_PORT=$((10000 + (SLURM_JOB_ID % 10000)))
AGENT_PORT=$((20000 + (SLURM_JOB_ID % 10000)))
MAX_MODEL_LEN=40960
ENV="{env}"
STRATEGY="{strategy}"
NUM_TRIALS=5
MAX_CONCURRENCY={hw.get('max_concurrency', 2)}
PART_NUM={part_num}
TASK_RANGE="{task_range}"

# Batch configuration for Part {part_num} (tasks {task_range})
BATCHES=({batches_str})

GAUDI_BASE="/data/sse/gaudi"
CONTAINER="$GAUDI_BASE/containers/vllm-gaudi.sif"
VLLM_CD="$GAUDI_BASE/vllm-fork/.cd"
WORK_DIR="/scratch/$USER/gaudi_tau_bench_${{SLURM_JOB_ID}}"
mkdir -p "$WORK_DIR/logs"

cleanup() {{
    # Graceful shutdown: SIGTERM first, wait, then SIGKILL as fallback
    [ -n "$USER_PID" ] && kill $USER_PID 2>/dev/null || true
    [ -n "$AGENT_PID" ] && kill $AGENT_PID 2>/dev/null || true
    sleep 10  # Give processes time to shut down gracefully
    [ -n "$USER_PID" ] && kill -0 $USER_PID 2>/dev/null && kill -9 $USER_PID 2>/dev/null || true
    [ -n "$AGENT_PID" ] && kill -0 $AGENT_PID 2>/dev/null && kill -9 $AGENT_PID 2>/dev/null || true
    fuser -k $USER_PORT/tcp 2>/dev/null || true
    fuser -k $AGENT_PORT/tcp 2>/dev/null || true
    # Kill ALL python3/vllm processes holding HPU devices (apptainer children)
    pkill -9 -f "vllm serve.*${{USER_PORT}}" 2>/dev/null || true
    pkill -9 -f "vllm serve.*${{AGENT_PORT}}" 2>/dev/null || true
    # Kill any remaining apptainer/python3 processes from this job
    pkill -9 -f "apptainer exec.*vllm-gaudi" 2>/dev/null || true
    sleep 5
    # Final sweep: kill any python3 processes on HPU devices
    for pid in $(hl-smi 2>/dev/null | grep -oP 'pid=\\K[0-9]+' | sort -u); do
        kill -9 "$pid" 2>/dev/null || true
    done
}}

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

check_server() {{ curl -s --connect-timeout 5 "http://localhost:${{1}}/health" > /dev/null 2>&1; return $?; }}

# Setup conda environment once before batch loop
module load mamba/latest
source activate tau-bench 2>/dev/null || {{ mamba create -n tau-bench -c conda-forge python=3.11 -y; source activate tau-bench; }}
cd "$REPO_ROOT"; pip uninstall tau_bench -y 2>/dev/null || true; pip install -e .

export OPENAI_API_KEY="dummy"
LOG_DIR="$SCRIPT_DIR/results_gaudi/${{ENV}}/${{STRATEGY}}"
mkdir -p "$LOG_DIR"

USER_URL="http://localhost:${{USER_PORT}}/v1"
AGENT_URL="http://localhost:${{AGENT_PORT}}/v1"

# Run experiments in batches with server restart between batches
# Fail-fast: abort if consecutive batches fail (HPU devices likely stuck)
CONSECUTIVE_FAILURES=0
MAX_CONSECUTIVE_FAILURES=2
SUCCESSFUL_BATCHES=0
BATCH_NUM=0
for BATCH in "${{BATCHES[@]}}"; do
    read START_IDX END_IDX <<< "$BATCH"
    BATCH_NUM=$((BATCH_NUM + 1))
    echo ""
    echo "=============================================="
    echo "=== BATCH ${{BATCH_NUM}}/${{#BATCHES[@]}}: Tasks ${{START_IDX}} to ${{END_IDX}} ==="
    echo "=============================================="
    echo "Batch started at: $(date)"

    cd "$VLLM_CD"

    echo "=== Starting User Model Server ({user_desc}) ==="
    USER_LOG="$SCRIPT_DIR/logs/gaudi_vllm_user_32b_${{SLURM_JOB_ID}}_batch${{BATCH_NUM}}.log"
    export APPTAINERENV_HABANA_VISIBLE_DEVICES={hw['user_hpus']}
    apptainer exec --bind /usr/lib64:/host-lib64 --bind /usr/lib/habanalabs:/usr/lib/habanalabs --bind /opt/habanalabs:/opt/habanalabs --bind /usr/bin/shim_ctl:/usr/bin/shim_ctl --bind "$HF_HOME:/mnt/hf_cache" --bind "$(pwd):/workspace/.cd" --bind "$WORK_DIR/logs:/var/log/habana_logs" --pwd /workspace/.cd --writable-tmpfs "$CONTAINER" vllm serve "$USER_MODEL" --host 0.0.0.0 --port $USER_PORT --block-size 128 --dtype bfloat16 --tensor-parallel-size {hw['user_tp']} --download-dir /mnt/hf_cache --max-model-len $MAX_MODEL_LEN --gpu-memory-utilization {hw['gpu_mem_util']} --use-padding-aware-scheduling --max-num-seqs 2 --max-num-prefill-seqs 1 --num-scheduler-steps 1 --disable-log-requests --enable-auto-tool-choice --tool-call-parser hermes --swap-space 16 > "$USER_LOG" 2>&1 &
    USER_PID=$!

    echo "=== Starting Agent Model Server ({agent_desc}) ==="
    AGENT_LOG="$SCRIPT_DIR/logs/gaudi_vllm_agent_{model_size}_${{SLURM_JOB_ID}}_batch${{BATCH_NUM}}.log"
    export APPTAINERENV_HABANA_VISIBLE_DEVICES={hw['agent_hpus']}
    apptainer exec --bind /usr/lib64:/host-lib64 --bind /usr/lib/habanalabs:/usr/lib/habanalabs --bind /opt/habanalabs:/opt/habanalabs --bind /usr/bin/shim_ctl:/usr/bin/shim_ctl --bind "$HF_HOME:/mnt/hf_cache" --bind "$(pwd):/workspace/.cd" --bind "$WORK_DIR/logs:/var/log/habana_logs" --pwd /workspace/.cd --writable-tmpfs "$CONTAINER" vllm serve "$AGENT_MODEL" --host 0.0.0.0 --port $AGENT_PORT --block-size 128 --dtype bfloat16 --tensor-parallel-size {hw['agent_tp']} --download-dir /mnt/hf_cache --max-model-len $MAX_MODEL_LEN --gpu-memory-utilization {hw['gpu_mem_util']} --use-padding-aware-scheduling --max-num-seqs {hw['agent_max_num_seqs']} --max-num-prefill-seqs {hw['agent_max_num_prefill_seqs']} --num-scheduler-steps 1 --disable-log-requests --enable-auto-tool-choice --tool-call-parser hermes --swap-space 16 > "$AGENT_LOG" 2>&1 &
    AGENT_PID=$!

    # Wait for servers with skip-on-failure instead of exit 1
    BATCH_SKIP=false

    echo -n "Waiting for User server (32B)..."
    for i in {{1..180}}; do
        if check_server "$USER_PORT"; then echo " Ready! (${{i}}0s)"; break; fi
        if ! kill -0 $USER_PID 2>/dev/null; then
            echo " FAILED!"
            tail -100 "$USER_LOG"
            echo "WARNING: User server failed to start for batch ${{BATCH_NUM}}, skipping..."
            BATCH_SKIP=true
            break
        fi
        echo -n "."; sleep 10
    done
    if [ "$BATCH_SKIP" = false ] && ! check_server "$USER_PORT"; then
        echo " TIMEOUT!"
        tail -100 "$USER_LOG"
        echo "WARNING: User server timed out for batch ${{BATCH_NUM}}, skipping..."
        BATCH_SKIP=true
    fi

    if [ "$BATCH_SKIP" = false ]; then
        echo -n "Waiting for Agent server ({agent_desc})..."
        for i in {{1..{hw['agent_wait_iters']}}}; do
            if check_server "$AGENT_PORT"; then echo " Ready! (${{i}}0s)"; break; fi
            if ! kill -0 $AGENT_PID 2>/dev/null; then
                echo " FAILED!"
                tail -100 "$AGENT_LOG"
                echo "WARNING: Agent server failed to start for batch ${{BATCH_NUM}}, skipping..."
                BATCH_SKIP=true
                break
            fi
            echo -n "."; sleep 10
        done
        if [ "$BATCH_SKIP" = false ] && ! check_server "$AGENT_PORT"; then
            echo " TIMEOUT!"
            tail -100 "$AGENT_LOG"
            echo "WARNING: Agent server timed out for batch ${{BATCH_NUM}}, skipping..."
            BATCH_SKIP=true
        fi
    fi

    if [ "$BATCH_SKIP" = false ]; then
        echo "Both servers ready!"

        cd "$REPO_ROOT"
        python run.py --env ${{ENV}} --agent-strategy ${{STRATEGY}} \\
            --model ${{AGENT_MODEL}} --model-provider openai --model-base-url ${{AGENT_URL}} \\
            --user-model ${{USER_MODEL}} --user-model-provider openai --user-model-base-url ${{USER_URL}} \\
            --log-dir ${{LOG_DIR}} --max-concurrency ${{MAX_CONCURRENCY}} --num-trials ${{NUM_TRIALS}} \\
            --start-index ${{START_IDX}} --end-index $((END_IDX + 1)) || echo "WARNING: Batch ${{BATCH_NUM}} run.py exited with non-zero status, continuing to next batch..."

        # Wait for vLLM to finish processing any queued requests
        echo "Batch ${{BATCH_NUM}} execution complete, waiting for vLLM to finish processing..."
        sleep 30

        # Check if servers are still responsive
        echo "Checking server status before cleanup..."
        check_server "$USER_PORT" && echo "User server still responsive" || echo "User server not responding"
        check_server "$AGENT_PORT" && echo "Agent server still responsive" || echo "Agent server not responding"

        # Rename output file to include batch info for clean merging
        # Match on expected range to avoid race condition with concurrent parts sharing LOG_DIR
        EXPECTED_RANGE="range_${{START_IDX}}-$((END_IDX + 1))"
        LATEST=$(ls -t "$LOG_DIR"/*.json 2>/dev/null | grep "$EXPECTED_RANGE" | grep -v "_batch" | grep -v "_merged" | grep -v "_part" | head -1)
        if [ -n "$LATEST" ]; then
            BATCH_FILE="${{LATEST%.json}}_part${{PART_NUM}}_batch${{BATCH_NUM}}_job${{SLURM_JOB_ID}}.json"
            mv "$LATEST" "$BATCH_FILE"
            echo "Batch ${{BATCH_NUM}} results saved to: $BATCH_FILE"
        fi

        echo "Batch ${{BATCH_NUM}} completed at: $(date)"
        CONSECUTIVE_FAILURES=0
        SUCCESSFUL_BATCHES=$((SUCCESSFUL_BATCHES + 1))
    else
        echo "SKIPPING batch ${{BATCH_NUM}} due to server startup failure"
        CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))
        echo "Consecutive failures: ${{CONSECUTIVE_FAILURES}}/${{MAX_CONSECUTIVE_FAILURES}}"
        if [ "$CONSECUTIVE_FAILURES" -ge "$MAX_CONSECUTIVE_FAILURES" ]; then
            echo ""
            echo "ABORTING: ${{MAX_CONSECUTIVE_FAILURES}} consecutive batch failures detected."
            echo "HPU devices are likely stuck. Remaining batches would also fail."
            echo "Completed ${{SUCCESSFUL_BATCHES}}/${{#BATCHES[@]}} batches successfully."
            break
        fi
    fi

    # Kill servers before next batch (cleanup clears memory fragmentation)
    echo "Stopping servers for memory cleanup..."
    echo "Giving servers 5 seconds to finish any final requests..."
    sleep 5
    cleanup
    USER_PID=""
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
    # These are child workers from vLLM containers that survive parent process kill
    for hpu_pid in $(hl-smi 2>/dev/null | awk '/python3/{{print $3}}'); do
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
    echo "Part ${{PART_NUM}} (tasks ${{TASK_RANGE}}) complete: ${{SUCCESSFUL_BATCHES}}/${{#BATCHES[@]}} batches succeeded."
    echo "Use merge_results.py to combine all parts."
    echo "All part results saved to: $LOG_DIR"
    echo "Experiment finished at: $(date)"
fi
"""


# ─── Template: submit_all.sh ────────────────────────────────────────────────


def generate_submit_all(
    model_size: str, env: str, num_parts: int, parts_config: dict
) -> str:
    """Generate submit_all.sh for a directory."""
    dir_name = f"{model_size}_{env}"

    # Build the parts summary for display
    parts_display = []
    for p in range(1, num_parts + 1):
        tr = parts_config[p]["task_range"]
        parts_display.append(f'echo "  Part {p} (tasks {tr}):   ${{JOBS[{p - 1}]}}"')
    parts_echo = "\n".join(parts_display)

    return f"""#!/bin/bash
# Submit all {num_parts} parts of a {model_size.upper()} {env} experiment for a given strategy.
# Usage: ./submit_all.sh <strategy>
#   strategy: act, react, or tool-calling
#
# Example:
#   ./submit_all.sh react        # Submit parts 1-{num_parts} for react
#   ./submit_all.sh act           # Submit parts 1-{num_parts} for act
#   ./submit_all.sh tool-calling  # Submit parts 1-{num_parts} for tool-calling

set -e

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"

STRATEGY="$1"

if [ -z "$STRATEGY" ]; then
    echo "Usage: $0 <strategy>"
    echo "  strategy: act, react, or tool-calling"
    exit 1
fi

if [ "$STRATEGY" != "act" ] && [ "$STRATEGY" != "react" ] && [ "$STRATEGY" != "tool-calling" ]; then
    echo "ERROR: Invalid strategy '$STRATEGY'"
    echo "  Valid strategies: act, react, tool-calling"
    exit 1
fi

echo "=========================================="
echo "=== Submitting {model_size.upper()} {env.title()} ${{STRATEGY}} ==="
echo "=========================================="
echo ""

JOBS=()
for PART in $(seq 1 {num_parts}); do
    SCRIPT="${{SCRIPT_DIR}}/part${{PART}}_${{STRATEGY}}.sh"
    if [ ! -f "$SCRIPT" ]; then
        echo "ERROR: Script not found: $SCRIPT"
        exit 1
    fi
    echo "Submitting Part ${{PART}}: ${{SCRIPT}}"
    JOB_OUTPUT=$(sbatch "$SCRIPT")
    JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP '\\d+$')
    JOBS+=("$JOB_ID")
    echo "  -> Job ID: $JOB_ID"
done

echo ""
echo "=========================================="
echo "All {num_parts} parts submitted successfully!"
echo "=========================================="
echo ""
echo "Strategy: ${{STRATEGY}}"
echo "Job IDs:"
{parts_echo}
echo ""
echo "Monitor with: squeue -u \\$USER"
echo "After all jobs complete, merge results with:"
echo "  python ${{SCRIPT_DIR}}/merge_results.py --strategy ${{STRATEGY}}"
"""


# ─── Template: merge_results.py ─────────────────────────────────────────────


def generate_merge_results(
    model_size: str, env: str, num_parts: int, total_tasks: int,
    parts_config: dict,
) -> str:
    """Generate merge_results.py for a directory."""
    agent_model_short = HARDWARE[model_size]["agent_model"].split("/")[-1]
    dir_name = f"{model_size}_{env}"

    # Build parts task range mapping for the generated script
    parts_ranges = {}
    for pn, pcfg in parts_config.items():
        # Get the full task range from batches (min start, max end)
        all_starts = [int(s) for s, e in pcfg["batches"]]
        all_ends = [int(e) for s, e in pcfg["batches"]]
        parts_ranges[pn] = (min(all_starts), max(all_ends))
    parts_ranges_str = repr(parts_ranges)

    return f'''#!/usr/bin/env python3
"""Merge split {model_size.upper()} {env} experiment results from multiple parts into a single file.

Protections against failed jobs:
  - --job-ids: Only merge results from specific SLURM job IDs (whitelist)
  - --exclude-jobs: Exclude results from specific failed job IDs (blacklist)
  - Corrupted/truncated JSON files are skipped with warnings
  - Empty result files (0 results) are skipped with warnings
  - Duplicate task_id+trial combos from retried jobs: keeps latest job's results
  - --dry-run: Preview everything before writing

Usage:
    python merge_results.py --strategy react
    python merge_results.py --strategy react --job-ids 46800001 46800002
    python merge_results.py --strategy react --exclude-jobs 46703271
    python merge_results.py --strategy react --dry-run
"""

import argparse
import glob
import json
import os
import re
import sys


def find_part_files(results_dir: str) -> list[str]:
    """Find all part result files in the given directory."""
    pattern = os.path.join(results_dir, "*_part*_batch*_job*.json")
    files = sorted(glob.glob(pattern))
    return files


def extract_part_info(filepath: str) -> dict | None:
    """Extract part number, batch number, and job ID from filename."""
    basename = os.path.basename(filepath)
    match = re.search(r"_part(\\d+)_batch(\\d+)_job(\\d+)\\.json$", basename)
    if match:
        return {{
            "part": int(match.group(1)),
            "batch": int(match.group(2)),
            "job_id": match.group(3),
        }}
    return None


def load_results_safe(filepath: str) -> tuple[list, str | None]:
    """Load a result file with error handling.

    Returns (results_list, error_message).
    error_message is None on success.
    """
    try:
        with open(filepath) as fp:
            data = json.load(fp)
    except json.JSONDecodeError as e:
        return [], f"Corrupted JSON: {{e}}"
    except OSError as e:
        return [], f"Read error: {{e}}"

    if isinstance(data, list):
        return data, None
    elif isinstance(data, dict):
        return [data], None
    else:
        return [], f"Unexpected data type: {{type(data).__name__}}"


def deduplicate_results(results: list) -> tuple[list, int]:
    """Remove duplicate task_id+trial entries, keeping the last occurrence.

    When a job is retried, the newer job's results appear later in the file list
    (higher job ID = later submission). By keeping the last occurrence, we
    automatically prefer the retry over the original failed run.

    Returns (deduplicated_results, num_duplicates_removed).
    """
    seen = {{}}
    for r in results:
        if not isinstance(r, dict):
            continue
        task_id = r.get("task_id")
        trial = r.get("trial", 0)
        if task_id is not None:
            key = (task_id, trial)
            seen[key] = r  # Last write wins
        else:
            # No task_id, keep as-is with a unique key
            seen[id(r)] = r

    deduped = list(seen.values())
    num_removed = len(results) - len(deduped)
    return deduped, num_removed


def validate_coverage(results: list, total_tasks: int = {total_tasks}, num_trials: int = 5) -> dict:
    """Check which task indices are covered and trial completeness."""
    from collections import defaultdict

    covered = set()
    trials_per_task = defaultdict(set)
    for r in results:
        if isinstance(r, dict):
            task_id = None
            for key in ("task_id", "task_index", "index"):
                if key in r:
                    try:
                        task_id = int(r[key])
                        covered.add(task_id)
                    except (ValueError, TypeError):
                        pass
                    break
            if task_id is not None:
                trial = r.get("trial", 0)
                trials_per_task[task_id].add(trial)

    expected = set(range(total_tasks))
    missing = expected - covered
    extra = covered - expected

    # Trial completeness
    expected_trials = set(range(num_trials))
    complete_tasks = sorted(t for t in range(total_tasks)
                            if trials_per_task.get(t, set()) >= expected_trials)
    incomplete_tasks = {{}}
    for t in range(total_tasks):
        got = trials_per_task.get(t, set())
        missing_t = expected_trials - got
        if missing_t:
            incomplete_tasks[t] = sorted(missing_t)

    total_trajectories = sum(len(v) for v in trials_per_task.values())
    expected_trajectories = total_tasks * num_trials

    return {{
        "covered": sorted(covered),
        "missing": sorted(missing),
        "extra": sorted(extra),
        "total_covered": len(covered),
        "total_expected": total_tasks,
        "complete_tasks": len(complete_tasks),
        "incomplete_tasks": incomplete_tasks,
        "total_trajectories": total_trajectories,
        "expected_trajectories": expected_trajectories,
    }}


def main():
    parser = argparse.ArgumentParser(
        description="Merge split {model_size.upper()} {env} experiment part results"
    )
    parser.add_argument(
        "--strategy",
        required=True,
        choices=["act", "react", "tool-calling"],
        help="Strategy to merge results for",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Custom results directory (default: auto-detect from script location)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Custom output filename (default: auto-generated)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be merged without writing",
    )
    parser.add_argument(
        "--total-tasks",
        type=int,
        default={total_tasks},
        help="Total expected tasks (default: {total_tasks})",
    )
    parser.add_argument(
        "--job-ids",
        nargs="+",
        default=None,
        help="Only include results from these SLURM job IDs (whitelist)",
    )
    parser.add_argument(
        "--exclude-jobs",
        nargs="+",
        default=None,
        help="Exclude results from these SLURM job IDs (blacklist)",
    )
    args = parser.parse_args()

    if args.job_ids and args.exclude_jobs:
        print("ERROR: Cannot use both --job-ids and --exclude-jobs")
        sys.exit(1)

    # Determine results directory
    if args.results_dir:
        results_dir = args.results_dir
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(
            script_dir, "results_gaudi", "{env}", args.strategy
        )

    if not os.path.isdir(results_dir):
        print(f"ERROR: Results directory not found: {{results_dir}}")
        sys.exit(1)

    # Find part files
    all_files = find_part_files(results_dir)
    if not all_files:
        print(f"No part files found in: {{results_dir}}")
        print(f"  Pattern: *_part*_batch*_job*.json")
        sys.exit(1)

    # Apply job ID filters
    job_whitelist = set(args.job_ids) if args.job_ids else None
    job_blacklist = set(args.exclude_jobs) if args.exclude_jobs else None

    files = []
    skipped_by_filter = []
    for f in all_files:
        info = extract_part_info(f)
        if info is None:
            files.append(f)
            continue
        job_id = info["job_id"]
        if job_whitelist and job_id not in job_whitelist:
            skipped_by_filter.append((f, job_id, "not in --job-ids"))
            continue
        if job_blacklist and job_id in job_blacklist:
            skipped_by_filter.append((f, job_id, "excluded by --exclude-jobs"))
            continue
        files.append(f)

    # Display header
    print(f"Strategy: {{args.strategy}}")
    print(f"Results dir: {{results_dir}}")
    print(f"Found {{len(all_files)}} total part files, {{len(files)}} after filtering")
    if job_whitelist:
        print(f"Job ID whitelist: {{sorted(job_whitelist)}}")
    if job_blacklist:
        print(f"Job ID blacklist: {{sorted(job_blacklist)}}")
    print()

    # Show filtered-out files
    if skipped_by_filter:
        print(f"FILTERED OUT ({{len(skipped_by_filter)}} files):")
        for f, job_id, reason in skipped_by_filter:
            print(f"  SKIP job {{job_id}}: {{os.path.basename(f)}} ({{reason}})")
        print()

    if not files:
        print("ERROR: No files remaining after filtering")
        sys.exit(1)

    # Load and validate each file
    print(f"Loading {{len(files)}} files:")
    parts_seen = {{}}
    all_results = []
    load_errors = []

    for f in files:
        info = extract_part_info(f)
        basename = os.path.basename(f)
        size_kb = os.path.getsize(f) / 1024

        results, error = load_results_safe(f)

        if error:
            load_errors.append((f, error))
            print(f"  SKIP {{basename}} ({{size_kb:.0f}}KB) - {{error}}")
            continue

        if len(results) == 0:
            load_errors.append((f, "Empty file (0 results)"))
            print(f"  SKIP {{basename}} ({{size_kb:.0f}}KB) - Empty file (0 results)")
            continue

        if info:
            part = info["part"]
            if part not in parts_seen:
                parts_seen[part] = []
            parts_seen[part].append(f)
            print(
                f"  OK   Part {{info[\'part\']}}, Batch {{info[\'batch\']}}, "
                f"Job {{info[\'job_id\']}}: {{len(results)}} results ({{size_kb:.0f}}KB)"
            )
        else:
            print(f"  OK   {{basename}}: {{len(results)}} results ({{size_kb:.0f}}KB)")

        all_results.extend(results)

    print()

    # Report load errors
    if load_errors:
        print(f"WARNING: {{len(load_errors)}} files skipped due to errors:")
        for f, err in load_errors:
            print(f"  {{os.path.basename(f)}}: {{err}}")
        print()

    # Check parts coverage
    print(f"Parts found: {{sorted(parts_seen.keys())}}")
    expected_parts = set(range(1, {num_parts + 1}))
    missing_parts = expected_parts - set(parts_seen.keys())
    if missing_parts:
        print(f"WARNING: Missing parts: {{sorted(missing_parts)}}")
        print("  Some jobs may still be running or failed.")
    print()

    # Deduplicate (handles retried jobs)
    all_results, num_dupes = deduplicate_results(all_results)
    if num_dupes > 0:
        print(
            f"Deduplicated: removed {{num_dupes}} duplicate task_id+trial entries "
            f"(kept latest job's results)"
        )
        print()

    print(f"Total results after validation: {{len(all_results)}}")

    # Validate task coverage
    coverage = validate_coverage(all_results, args.total_tasks)
    if coverage["total_covered"] > 0:
        print(
            f"Task coverage: {{coverage[\'total_covered\']}}/{{coverage[\'total_expected\']}}"
        )
        if coverage["missing"]:
            print(f"  Missing tasks: {{coverage[\'missing\']}}")
        if coverage["extra"]:
            print(f"  Extra tasks: {{coverage[\'extra\']}}")
        print(
            f"Trial coverage: {{coverage[\'total_trajectories\']}}/{{coverage[\'expected_trajectories\']}} "
            f"({{coverage[\'complete_tasks\']}}/{{coverage[\'total_expected\']}} tasks have all 5 trials)"
        )
        if coverage["incomplete_tasks"]:
            # Show summary of incomplete tasks
            incomplete = coverage["incomplete_tasks"]
            num_incomplete = len(incomplete)
            total_missing_trials = sum(len(v) for v in incomplete.values())
            print(f"  {{num_incomplete}} tasks missing {{total_missing_trials}} trials total")
            if num_incomplete <= 20:
                for task_id, missing_trials in sorted(incomplete.items()):
                    print(f"    Task {{task_id}}: missing trials {{missing_trials}}")
    else:
        print("  (Could not determine task coverage from result format)")

    # Map parts to task ranges for rerun suggestions
    PARTS_TASK_RANGES = {parts_ranges_str}

    def find_parts_for_tasks(missing_tasks):
        """Find which parts need rerunning based on missing task IDs."""
        failed_parts = set()
        for task in missing_tasks:
            for part_num, (start, end) in PARTS_TASK_RANGES.items():
                if start <= task <= end:
                    failed_parts.add(part_num)
                    break
        return sorted(failed_parts)

    # Determine pass/fail status (requires ALL tasks AND ALL trials)
    tasks_complete = coverage["total_covered"] == coverage["total_expected"]
    trials_complete = coverage["total_trajectories"] == coverage["expected_trajectories"]
    is_complete = tasks_complete and trials_complete
    failed_parts = find_parts_for_tasks(coverage["missing"]) if coverage["missing"] else []
    passed_parts = sorted(set(PARTS_TASK_RANGES.keys()) - set(failed_parts))

    if failed_parts:
        print()
        if passed_parts:
            print(f"Passed parts: {{passed_parts}}")
        print(f"Failed parts (missing tasks): {{failed_parts}}")
        print()
        print("Rerun commands:")
        for p in failed_parts:
            start, end = PARTS_TASK_RANGES[p]
            print(f"  sbatch SOL_env/{dir_name}/part{{p}}_{{args.strategy}}.sh"
                  f"    # tasks {{start}}-{{end}}")
        print()
        print("After rerunning, re-merge:")
        print(f"  python SOL_env/{dir_name}/merge_results.py "
              f"--strategy {{args.strategy}} --dry-run")
    elif not trials_complete:
        # All tasks have at least 1 trial but not all 5
        print()
        print(f"WARNING: All tasks present but only "
              f"{{coverage[\'total_trajectories\']}}/{{coverage[\'expected_trajectories\']}} "
              f"trials complete.")
        # Find which parts have incomplete trials
        incomplete_parts = set()
        for task_id in coverage["incomplete_tasks"]:
            for part_num, (start, end) in PARTS_TASK_RANGES.items():
                if start <= task_id <= end:
                    incomplete_parts.add(part_num)
                    break
        if incomplete_parts:
            print(f"Parts with incomplete trials: {{sorted(incomplete_parts)}}")
            print()
            print("Rerun commands to fill missing trials:")
            for p in sorted(incomplete_parts):
                start, end = PARTS_TASK_RANGES[p]
                print(f"  sbatch SOL_env/{dir_name}/part{{p}}_{{args.strategy}}.sh"
                      f"    # tasks {{start}}-{{end}}")

    if args.dry_run:
        print()
        if is_complete:
            print("[DRY RUN] All tasks and trials complete! After merging, download with:")
            merged_name = (
                f"{{args.strategy}}-{agent_model_short}-0.0"
                f"_range_0-{{args.total_tasks}}_merged.json"
            )
            username = os.getenv("USER", "your_username")
            print(f"  scp {{username}}@sol.asu.edu:$(pwd)/SOL_env/{dir_name}"
                  f"/results_gaudi/{env}/{{args.strategy}}/{{merged_name}}"
                  f" ~/Downloads/")
        else:
            print("[DRY RUN] Would merge the above results. No output written.")
            if tasks_complete:
                print(f"NOTE: All {{coverage[\'total_expected\']}} tasks present but "
                      f"only {{coverage[\'complete_tasks\']}} have all 5 trials.")
        sys.exit(0)

    # Write output
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(
            results_dir,
            f"{{args.strategy}}-{agent_model_short}-0.0_range_0-{{args.total_tasks}}_merged.json",
        )

    with open(output_path, "w") as fp:
        json.dump(all_results, fp, indent=2)

    print()
    print(f"Merged output written to: {{output_path}}")
    print(f"  {{len(files) - len(load_errors)}} valid files -> {{len(all_results)}} results")

    if is_complete:
        username = os.getenv("USER", "your_username")
        print()
        print("All tasks and trials complete! Download with:")
        print(f"  scp {{username}}@sol.asu.edu:{{output_path}} ~/Downloads/")
    elif tasks_complete:
        username = os.getenv("USER", "your_username")
        print()
        print(f"NOTE: All {{coverage[\'total_expected\']}} tasks present but "
              f"only {{coverage[\'complete_tasks\']}} have all 5 trials "
              f"({{coverage[\'total_trajectories\']}}/{{coverage[\'expected_trajectories\']}} trajectories).")
        print("Merging anyway. Download with:")
        print(f"  scp {{username}}@sol.asu.edu:{{output_path}} ~/Downloads/")


if __name__ == "__main__":
    main()
'''


# ─── Main Generator ─────────────────────────────────────────────────────────


def get_parts_config(model_size: str, env: str) -> dict:
    """Get the parts configuration for a given model/env combo."""
    if env == "airline":
        return AIRLINE_32B_PARTS if model_size == "32b" else AIRLINE_PARTS
    elif model_size == "32b":
        return RETAIL_32B_PARTS
    else:
        return RETAIL_PARTS


def main():
    parser = argparse.ArgumentParser(
        description="Generate split experiment scripts for all model sizes and environments"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be created without writing files",
    )
    args = parser.parse_args()

    total_files = 0
    total_dirs = 0

    for model_size, env in EXPERIMENTS:
        dir_name = f"{model_size}_{env}"
        dir_path = os.path.join(SCRIPT_DIR, dir_name)
        parts_config = get_parts_config(model_size, env)
        num_parts = len(parts_config)
        total_tasks = 50 if env == "airline" else 115

        print(f"\n{'='*60}")
        print(f"=== {dir_name}: {num_parts} parts × 3 strategies ===")
        print(f"{'='*60}")

        if args.dry_run:
            print(f"  Would create: {dir_path}/")
        else:
            os.makedirs(dir_path, exist_ok=True)
            total_dirs += 1

        files_in_dir = 0

        # Generate part scripts for each strategy × part
        for strategy in STRATEGIES:
            for part_num in range(1, num_parts + 1):
                filename = f"part{part_num}_{strategy}.sh"
                filepath = os.path.join(dir_path, filename)
                content = generate_part_script(
                    model_size, env, strategy, part_num, parts_config
                )
                if args.dry_run:
                    print(f"  Would create: {filename}")
                else:
                    with open(filepath, "w") as f:
                        f.write(content)
                    os.chmod(filepath, os.stat(filepath).st_mode | stat.S_IEXEC)
                    files_in_dir += 1

        # Generate submit_all.sh
        submit_path = os.path.join(dir_path, "submit_all.sh")
        submit_content = generate_submit_all(model_size, env, num_parts, parts_config)
        if args.dry_run:
            print(f"  Would create: submit_all.sh")
        else:
            with open(submit_path, "w") as f:
                f.write(submit_content)
            os.chmod(submit_path, os.stat(submit_path).st_mode | stat.S_IEXEC)
            files_in_dir += 1

        # Generate merge_results.py
        merge_path = os.path.join(dir_path, "merge_results.py")
        merge_content = generate_merge_results(model_size, env, num_parts, total_tasks, parts_config)
        if args.dry_run:
            print(f"  Would create: merge_results.py")
        else:
            with open(merge_path, "w") as f:
                f.write(merge_content)
            os.chmod(merge_path, os.stat(merge_path).st_mode | stat.S_IEXEC)
            files_in_dir += 1

        if not args.dry_run:
            print(f"  Created {files_in_dir} files in {dir_path}/")
        total_files += files_in_dir

    # Summary
    print(f"\n{'='*60}")
    if args.dry_run:
        print(f"DRY RUN: Would create {total_files} files in {len(EXPERIMENTS)} directories")
        print(f"  (includes 32b splits with 1 batch per part for HPU compatibility)")
    else:
        print(f"Generated {total_files} files in {total_dirs} directories")
        print("\nNext steps:")
        print("  1. Review generated scripts")
        print("  2. Delete old dirs: rm -rf SOL_env/{4b,8b,14b,32b}_run/")
        print("  3. Submit experiments: ./SOL_env/8b_retail/submit_all.sh act")


if __name__ == "__main__":
    main()
