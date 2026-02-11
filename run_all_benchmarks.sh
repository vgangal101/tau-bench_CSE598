#!/bin/bash

# Run all tau-bench benchmarks with OpenRouter Qwen models
# Covers Tool calling, ACT, and ReAct strategies for both retail and airline environments

set -e  # Exit on error

echo "=========================================="
echo "Starting tau-bench benchmark runs"
echo "=========================================="

# ======================
# RETAIL ENVIRONMENT
# ======================

echo ""
echo "=========================================="
echo "RETAIL: Tool Calling Strategy"
echo "=========================================="

echo "Running qwen/qwen3-8b..."
python run.py --model qwen/qwen3-8b --model-provider openrouter --env retail --max-concurrency 1 --env-file .env

echo "Running qwen/qwen3-14b..."
python run.py --model qwen/qwen3-14b --model-provider openrouter --env retail --max-concurrency 1 --env-file .env

echo "Running qwen/qwen3-32b..."
python run.py --model qwen/qwen3-32b --model-provider openrouter --env retail --max-concurrency 1 --env-file .env

echo ""
echo "=========================================="
echo "RETAIL: ACT Strategy"
echo "=========================================="

echo "Running qwen/qwen3-8b with ACT..."
python run.py --model qwen/qwen3-8b --model-provider openrouter --env retail --max-concurrency 1 --agent-strategy act --env-file .env

echo "Running qwen/qwen3-14b with ACT..."
python run.py --model qwen/qwen3-14b --model-provider openrouter --env retail --max-concurrency 1 --agent-strategy act --env-file .env

echo "Running qwen/qwen3-32b with ACT..."
python run.py --model qwen/qwen3-32b --model-provider openrouter --env retail --max-concurrency 1 --agent-strategy act --env-file .env

echo ""
echo "=========================================="
echo "RETAIL: ReAct Strategy"
echo "=========================================="

echo "Running qwen/qwen3-8b with ReAct..."
python run.py --model qwen/qwen3-8b --model-provider openrouter --env retail --max-concurrency 1 --agent-strategy react --env-file .env

echo "Running qwen/qwen3-14b with ReAct..."
python run.py --model qwen/qwen3-14b --model-provider openrouter --env retail --max-concurrency 1 --agent-strategy react --env-file .env

echo "Running qwen/qwen3-32b with ReAct..."
python run.py --model qwen/qwen3-32b --model-provider openrouter --env retail --max-concurrency 1 --agent-strategy react --env-file .env

# ======================
# AIRLINE ENVIRONMENT
# ======================

echo ""
echo "=========================================="
echo "AIRLINE: Tool Calling Strategy"
echo "=========================================="

echo "Running qwen/qwen3-8b..."
python run.py --model qwen/qwen3-8b --model-provider openrouter --env airline --max-concurrency 1 --env-file .env

echo "Running qwen/qwen3-14b..."
python run.py --model qwen/qwen3-14b --model-provider openrouter --env airline --max-concurrency 1 --env-file .env

echo "Running qwen/qwen3-32b..."
python run.py --model qwen/qwen3-32b --model-provider openrouter --env airline --max-concurrency 1 --env-file .env

echo ""
echo "=========================================="
echo "AIRLINE: ACT Strategy"
echo "=========================================="

echo "Running qwen/qwen3-8b with ACT..."
python run.py --model qwen/qwen3-8b --model-provider openrouter --env airline --max-concurrency 1 --agent-strategy act --env-file .env

echo "Running qwen/qwen3-14b with ACT..."
python run.py --model qwen/qwen3-14b --model-provider openrouter --env airline --max-concurrency 1 --agent-strategy act --env-file .env

echo "Running qwen/qwen3-32b with ACT..."
python run.py --model qwen/qwen3-32b --model-provider openrouter --env airline --max-concurrency 1 --agent-strategy act --env-file .env

echo ""
echo "=========================================="
echo "AIRLINE: ReAct Strategy"
echo "=========================================="

echo "Running qwen/qwen3-8b with ReAct..."
python run.py --model qwen/qwen3-8b --model-provider openrouter --env airline --max-concurrency 1 --agent-strategy react --env-file .env

echo "Running qwen/qwen3-14b with ReAct..."
python run.py --model qwen/qwen3-14b --model-provider openrouter --env airline --max-concurrency 1 --agent-strategy react --env-file .env

echo "Running qwen/qwen3-32b with ReAct..."
python run.py --model qwen/qwen3-32b --model-provider openrouter --env airline --max-concurrency 1 --agent-strategy react --env-file .env

echo ""
echo "=========================================="
echo "All benchmarks completed!"
echo "=========================================="
