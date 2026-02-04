#!/bin/bash
# ========================================
# Gaudi Experiment via SOL's OpenAI-Compatible API
# ========================================
# This script uses SOL's pre-hosted LLM API running on Gaudi2
# No local vLLM setup required!
#
# Prerequisites:
# 1. Get API key from: https://voyager.rc.asu.edu/ → LLM Access tab
# 2. Set environment variable: export SOL_API_KEY="your-key"
#
# Available models (as of RC Workshop):
#   - llama4-scout-17b (Context: 66K)
#   - qwen3-coder-30b-a3b-instruct (Context: 131K)
#   - qwen3-30b-a3b-thinking-2507 (Context: 131K)
#   - qwen3-30b-a3b-instruct-2507 (Context: 131K)
#   - qwen3-235b-a22b-instruct-2507 (Context: 262K)
#   - qwen3-235b-a22b-thinking-2507 (Context: 262K)
#
# Usage:
#   export SOL_API_KEY="your-api-key"
#   bash SOL_env/gaudi_api_experiment.sh
# ========================================

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "========================================"
echo "=== Gaudi API Experiment ==="
echo "========================================"
echo "Started at: $(date)"
echo ""

# ========================================
# Check API Key
# ========================================
if [ -z "$SOL_API_KEY" ]; then
    echo "ERROR: SOL_API_KEY environment variable not set"
    echo ""
    echo "To get an API key:"
    echo "1. Go to https://voyager.rc.asu.edu/"
    echo "2. Navigate to 'LLM Access' tab"
    echo "3. Click 'Create Key'"
    echo "4. Run: export SOL_API_KEY='your-key'"
    echo ""
    exit 1
fi

# ========================================
# Configuration
# ========================================
API_BASE_URL="https://openai.rc.asu.edu/v1"

# Models running on Gaudi2 at SOL
# Choose based on your needs (context length, speed, etc.)
AGENT_MODEL="qwen3-30b-a3b-instruct-2507"  # 131K context
USER_MODEL="qwen3-30b-a3b-instruct-2507"   # 131K context

echo "=== Configuration ==="
echo "API Base URL: $API_BASE_URL"
echo "Agent Model: $AGENT_MODEL"
echo "User Model: $USER_MODEL"
echo ""

# ========================================
# Environment Setup
# ========================================
echo "=== Setting up Environment ==="

module load mamba/latest 2>/dev/null || true
source activate tau-bench 2>/dev/null || {
    echo "Creating tau-bench environment..."
    mamba create -n tau-bench python=3.11 -y
    source activate tau-bench
}

cd "$REPO_ROOT"
pip install -q -e .

echo "Python: $(which python)"
echo ""

# ========================================
# Test API Connection
# ========================================
echo "=== Testing API Connection ==="

# Test the API
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" \
    -H "Authorization: Bearer $SOL_API_KEY" \
    "$API_BASE_URL/models" 2>/dev/null)

if [ "$RESPONSE" = "200" ]; then
    echo "API connection successful!"
    echo ""
    echo "Available models:"
    curl -s -H "Authorization: Bearer $SOL_API_KEY" "$API_BASE_URL/models" 2>/dev/null | \
        python -c "import sys,json; models=json.load(sys.stdin).get('data',[]); print('\n'.join(['  - '+m['id'] for m in models]))" 2>/dev/null || \
        echo "  (Could not parse model list)"
else
    echo "ERROR: API connection failed (HTTP $RESPONSE)"
    echo "Check your API key and try again"
    exit 1
fi
echo ""

# ========================================
# Run Experiments
# ========================================
echo "=== Running Experiments ==="

# Set the API key for OpenAI client
export OPENAI_API_KEY="$SOL_API_KEY"

TOTAL_EXPERIMENTS=0
SUCCESSFUL_EXPERIMENTS=0

for ENV in retail airline; do
    echo ""
    echo ">>> Running Environment: $ENV"

    for STRATEGY in tool-calling act react; do
        echo "  > Strategy: $STRATEGY"

        LOG_DIR="$SCRIPT_DIR/results_gaudi_api/${ENV}/${STRATEGY}"
        mkdir -p "$LOG_DIR"

        CMD="python run.py \
            --env ${ENV} \
            --agent-strategy ${STRATEGY} \
            --model ${AGENT_MODEL} \
            --model-provider openai \
            --model-base-url ${API_BASE_URL} \
            --user-model ${USER_MODEL} \
            --user-model-provider openai \
            --user-model-base-url ${API_BASE_URL} \
            --log-dir ${LOG_DIR} \
            --max-concurrency 5 \
            --num-trials 1 \
            --end-index 5 \
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
echo "Results saved to: $SCRIPT_DIR/results_gaudi_api/"
echo ""

echo "========================================"
echo "=== Gaudi API Experiment Complete ==="
echo "========================================"
echo "Finished at: $(date)"
