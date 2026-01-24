#!/bin/bash
# Start vLLM server with qwen3-4b model
# Usage: ./START_VLLM_qwen3-4b.sh

echo "🚀 Starting vLLM server with qwen3-4b..."
echo "========================================"
echo ""
echo "Server will be accessible at: http://localhost:8000"
echo "Keep this terminal open while running experiments"
echo ""

# Activate conda environment
conda activate Tau1

# Start vLLM server
# Note: Model will auto-download from HuggingFace on first run
vllm serve qwen3-4b \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --max-model-len 8192

# If you get "CUDA out of memory" error, add:
# --gpu-memory-utilization 0.9
