#!/bin/bash
set -euo pipefail

# ========================================
# tau-bench Experiment Runner
# ========================================
# Launches two vLLM servers (user-model + agent-model) on localhost
# with different ports and runs tau-bench experiments across
# configurable environments and agent strategies.
# ========================================

# ----------------------------------------
# Defaults
# ----------------------------------------
AGENT_MODEL=""
USER_MODEL="Qwen/Qwen3-32B"
ENVS="retail airline"
STRATEGIES="tool-calling act react"
MAX_CONCURRENCY=10
NUM_TRIALS=5
USER_PORT=8000
AGENT_PORT=8001
GPU_MEM_UTIL_USER=0.90
GPU_MEM_UTIL_AGENT=0.90
MAX_MODEL_LEN=32768
TENSOR_PARALLEL_SIZE=1
RESULTS_DIR="results"
END_INDEX=-1
START_INDEX=0
CONDA_ENV="tau-bench"
NO_INSTALL=0
REPO_ROOT=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ----------------------------------------
# Usage
# ----------------------------------------
usage() {
    cat <<'EOF'
Usage: run_experiment.sh --agent-model MODEL [OPTIONS]

Required:
  --agent-model MODEL        Agent model name (e.g. Qwen/Qwen3-4B)

Options:
  --user-model MODEL         User simulator model        [Qwen/Qwen3-32B]
  --envs "ENV1 ENV2"         Environments to run         [retail airline]
  --strategies "S1 S2"       Agent strategies to run      [tool-calling act react]
  --max-concurrency N        Parallel tasks               [10]
  --num-trials N             Number of trial runs          [5]
  --user-port PORT           Port for user vLLM server    [8000]
  --agent-port PORT          Port for agent vLLM server   [8001]
  --gpu-mem-util-user F      GPU mem utilization (user)   [0.90]
  --gpu-mem-util-agent F     GPU mem utilization (agent)  [0.90]
  --max-model-len N          Max token context length     [32768]
  --tensor-parallel-size N   Tensor parallel GPUs         [1]
  --results-dir DIR          Base directory for results   [results]
  --start-index N            Task start index             [0]
  --end-index N              Task end index (-1 = all)    [-1]
  --conda-env NAME           Conda environment name       [tau-bench]
  --no-install               Skip pip install -e .
  --repo-root DIR            Path to repo root            [auto-detected]
  --help                     Show this help message
EOF
    exit 0
}

# ----------------------------------------
# Parse CLI Arguments
# ----------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --agent-model)      AGENT_MODEL="$2"; shift 2 ;;
        --user-model)       USER_MODEL="$2"; shift 2 ;;
        --envs)             ENVS="$2"; shift 2 ;;
        --strategies)       STRATEGIES="$2"; shift 2 ;;
        --max-concurrency)  MAX_CONCURRENCY="$2"; shift 2 ;;
        --num-trials)       NUM_TRIALS="$2"; shift 2 ;;
        --user-port)        USER_PORT="$2"; shift 2 ;;
        --agent-port)       AGENT_PORT="$2"; shift 2 ;;
        --gpu-mem-util-user)  GPU_MEM_UTIL_USER="$2"; shift 2 ;;
        --gpu-mem-util-agent) GPU_MEM_UTIL_AGENT="$2"; shift 2 ;;
        --max-model-len)    MAX_MODEL_LEN="$2"; shift 2 ;;
        --tensor-parallel-size) TENSOR_PARALLEL_SIZE="$2"; shift 2 ;;
        --results-dir)      RESULTS_DIR="$2"; shift 2 ;;
        --start-index)      START_INDEX="$2"; shift 2 ;;
        --end-index)        END_INDEX="$2"; shift 2 ;;
        --conda-env)        CONDA_ENV="$2"; shift 2 ;;
        --no-install)       NO_INSTALL=1; shift ;;
        --repo-root)        REPO_ROOT="$2"; shift 2 ;;
        --help)             usage ;;
        *)
            echo "ERROR: Unknown argument: $1"
            echo "Run with --help for usage."
            exit 1
            ;;
    esac
done

# ----------------------------------------
# Validate
# ----------------------------------------
if [[ -z "$AGENT_MODEL" ]]; then
    echo "ERROR: --agent-model is required."
    echo "Run with --help for usage."
    exit 1
fi

if [[ -z "$REPO_ROOT" ]]; then
    REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

if [[ ! -d "$REPO_ROOT" ]]; then
    echo "ERROR: Repository root does not exist: $REPO_ROOT"
    exit 1
fi

if [[ ! -f "$REPO_ROOT/run.py" ]]; then
    echo "ERROR: run.py not found at $REPO_ROOT/run.py — is --repo-root correct?"
    exit 1
fi

# ----------------------------------------
# Setup
# ----------------------------------------
RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

USER_LOG="$LOG_DIR/user_server_${RUN_ID}.log"
AGENT_LOG="$LOG_DIR/agent_server_${RUN_ID}.log"

USER_URL="http://localhost:${USER_PORT}"
AGENT_URL="http://localhost:${AGENT_PORT}"

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export VLLM_USE_V1=0
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"

# ----------------------------------------
# Activate Conda Environment
# ----------------------------------------
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
echo "Activated conda environment: $CONDA_ENV"

USER_PID=""
AGENT_PID=""

# ----------------------------------------
# Cleanup
# ----------------------------------------
cleanup() {
    echo ""
    echo "=== Cleaning up ==="
    if [[ -n "$USER_PID" ]] && kill -0 "$USER_PID" 2>/dev/null; then
        kill "$USER_PID" 2>/dev/null || true
        wait "$USER_PID" 2>/dev/null || true
        echo "Stopped user vLLM server (PID $USER_PID)"
    fi
    if [[ -n "$AGENT_PID" ]] && kill -0 "$AGENT_PID" 2>/dev/null; then
        kill "$AGENT_PID" 2>/dev/null || true
        wait "$AGENT_PID" 2>/dev/null || true
        echo "Stopped agent vLLM server (PID $AGENT_PID)"
    fi
    echo "Cleanup complete"
}

trap cleanup EXIT INT TERM

# ----------------------------------------
# Print Configuration
# ----------------------------------------
echo "========================================"
echo "=== tau-bench Experiment Runner ==="
echo "========================================"
echo "Started at: $(date)"
echo "Run ID: $RUN_ID"
echo ""
echo "=== Configuration ==="
echo "Agent Model:       $AGENT_MODEL"
echo "User Model:        $USER_MODEL"
echo "Environments:      $ENVS"
echo "Strategies:        $STRATEGIES"
echo "Max Concurrency:   $MAX_CONCURRENCY"
echo "Num Trials:        $NUM_TRIALS"
echo "Start Index:       $START_INDEX"
echo "End Index:         $END_INDEX"
echo "User Server:       $USER_URL"
echo "Agent Server:      $AGENT_URL"
echo "GPU Mem (user):    $GPU_MEM_UTIL_USER"
echo "GPU Mem (agent):   $GPU_MEM_UTIL_AGENT"
echo "Max Model Len:     $MAX_MODEL_LEN"
echo "Tensor Parallel:   $TENSOR_PARALLEL_SIZE"
echo "Results Dir:       $RESULTS_DIR"
echo "Repo Root:         $REPO_ROOT"
echo "Conda Env:         $CONDA_ENV"
echo "Log Dir:           $LOG_DIR"
echo ""

# ----------------------------------------
# Step 1: Start User Simulator vLLM Server
# ----------------------------------------
echo "=== Step 1: Starting User Simulator vLLM server ==="
echo "  Model: $USER_MODEL"
echo "  Port:  $USER_PORT"
echo "  Log:   $USER_LOG"

vllm serve "$USER_MODEL" \
    --host 0.0.0.0 \
    --port "$USER_PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --gpu-memory-utilization "$GPU_MEM_UTIL_USER" \
    --max-model-len "$MAX_MODEL_LEN" \
    --trust-remote-code \
    --enforce-eager \
    --disable-log-requests \
    >> "$USER_LOG" 2>&1 &

USER_PID=$!
echo "  PID:   $USER_PID"
echo ""

sleep 2

# ----------------------------------------
# Step 2: Start Agent vLLM Server
# ----------------------------------------
echo "=== Step 2: Starting Agent vLLM server ==="
echo "  Model: $AGENT_MODEL"
echo "  Port:  $AGENT_PORT"
echo "  Log:   $AGENT_LOG"

vllm serve "$AGENT_MODEL" \
    --host 0.0.0.0 \
    --port "$AGENT_PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --gpu-memory-utilization "$GPU_MEM_UTIL_AGENT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --trust-remote-code \
    --enforce-eager \
    --disable-log-requests \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    >> "$AGENT_LOG" 2>&1 &

AGENT_PID=$!
echo "  PID:   $AGENT_PID"
echo ""

# ----------------------------------------
# Step 3: Wait for Servers
# ----------------------------------------
echo "=== Step 3: Waiting for servers to be ready ==="

check_server() {
    curl -s --connect-timeout 5 "${1}/health" > /dev/null 2>&1
}

wait_for_server() {
    local url="$1"
    local name="$2"
    local log="$3"
    local pid="$4"

    echo -n "Waiting for $name ($url)..."
    for i in $(seq 1 60); do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo " FAILED! (process exited)"
            echo "Server process died. Last 50 lines of $log:"
            tail -50 "$log" 2>/dev/null || true
            exit 1
        fi
        if check_server "$url"; then
            echo " Ready! (~${i}0s)"
            return 0
        fi
        echo -n "."
        sleep 10
    done

    echo " FAILED! (timeout)"
    echo "Server did not start within 600s. Last 50 lines of $log:"
    tail -50 "$log" 2>/dev/null || true
    exit 1
}

wait_for_server "$USER_URL" "User Simulator" "$USER_LOG" "$USER_PID"
wait_for_server "$AGENT_URL" "Agent" "$AGENT_LOG" "$AGENT_PID"

echo ""
echo "Both servers are ready!"
echo ""

# ----------------------------------------
# Step 4: Install Dependencies
# ----------------------------------------
echo "=== Step 4: Preparing environment ==="

if [[ "$NO_INSTALL" -eq 0 ]]; then
    echo "Installing dependencies from $REPO_ROOT..."
    pip install -q -e "$REPO_ROOT"
fi

echo "Working directory: $REPO_ROOT"
echo ""

# ----------------------------------------
# Step 5: Run Experiments
# ----------------------------------------
echo "=== Step 5: Running Experiments ==="
echo ""

TOTAL_EXPERIMENTS=0
SUCCESSFUL_EXPERIMENTS=0

for ENV in $ENVS; do
    echo ">>> Environment: $ENV"

    for STRATEGY in $STRATEGIES; do
        echo "  > Strategy: $STRATEGY"

        EXP_LOG_DIR="$SCRIPT_DIR/${RESULTS_DIR}/${ENV}/${STRATEGY}"
        mkdir -p "$EXP_LOG_DIR"

        TOTAL_EXPERIMENTS=$((TOTAL_EXPERIMENTS + 1))

        RUN_CMD=(
            python "$REPO_ROOT/run.py"
            --env "$ENV"
            --agent-strategy "$STRATEGY"
            --model "$AGENT_MODEL"
            --model-provider openai
            --model-base-url "${AGENT_URL}/v1"
            --user-model "$USER_MODEL"
            --user-model-provider openai
            --user-model-base-url "${USER_URL}/v1"
            --log-dir "$EXP_LOG_DIR"
            --max-concurrency "$MAX_CONCURRENCY"
            --num-trials "$NUM_TRIALS"
            --start-index "$START_INDEX"
            --end-index "$END_INDEX"
            --shuffle 0
        )

        echo "    Running: ${RUN_CMD[*]}"
        if "${RUN_CMD[@]}"; then
            echo "    Completed $STRATEGY for $ENV"
            SUCCESSFUL_EXPERIMENTS=$((SUCCESSFUL_EXPERIMENTS + 1))
        else
            echo "    FAILED: $STRATEGY for $ENV (exit code $?)"
        fi
        echo ""
    done
done

# ----------------------------------------
# Summary
# ----------------------------------------
echo "========================================"
echo "=== Experiments Summary ==="
echo "========================================"
echo "Total experiments: $TOTAL_EXPERIMENTS"
echo "Successful:        $SUCCESSFUL_EXPERIMENTS"
echo "Failed:            $((TOTAL_EXPERIMENTS - SUCCESSFUL_EXPERIMENTS))"
echo ""
echo "Results saved to: $SCRIPT_DIR/$RESULTS_DIR/"
echo "Server logs:      $LOG_DIR/"
echo ""
echo "Finished at: $(date)"
