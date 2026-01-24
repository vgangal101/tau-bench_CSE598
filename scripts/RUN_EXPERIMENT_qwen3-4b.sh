#!/bin/bash
# Run τ-bench experiment with qwen3-4b model via vLLM
# Prerequisites: vLLM server must be running (use START_VLLM_qwen3-4b.sh)
# Usage: ./RUN_EXPERIMENT_qwen3-4b.sh

echo "🧪 Running τ-bench experiment with qwen3-4b"
echo "============================================"
echo ""

# Activate conda environment
conda activate Tau1

# Set vLLM server endpoint
export OPENAI_API_BASE="http://localhost:8000/v1"
export OPENAI_API_KEY="dummy"

# Verify vLLM server is running
echo "Checking if vLLM server is accessible..."
if curl -s http://localhost:8000/v1/models > /dev/null; then
    echo "✅ vLLM server is running"
    echo ""
else
    echo "❌ ERROR: vLLM server is not accessible at http://localhost:8000"
    echo "Please start the server first using: ./START_VLLM_qwen3-4b.sh"
    exit 1
fi

# Test connection
echo "Testing LiteLLM connection..."
python test_vllm_connection.py
echo ""

# Run experiment
echo "Starting τ-bench experiment..."
echo "Strategy: tool-calling"
echo "Environment: retail"
echo "Model: qwen3-4b"
echo "User Model: gpt-4o"
echo "Trials: 5"
echo ""

python run.py \
  --agent-strategy tool-calling \
  --env retail \
  --model qwen3-4b \
  --model-provider openai \
  --user-model gpt-4o \
  --user-model-provider openai \
  --num-trials 5 \
  --max-concurrency 3 \
  --task-ids 0 1 2

echo ""
echo "✅ Experiment complete! Check results/ directory for output"
