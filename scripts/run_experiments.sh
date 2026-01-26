#!/bin/bash

# Configuration
# Replace this with your actual LiteLLM proxy URL
LITELLM_API_BASE="http://localhost:4000/v1" 

# User Simulator Model (fixed across experiments)
USER_MODEL="qwen-32b" # Ensure this name matches your LiteLLM config

# Common Arguments
COMMON_ARGS="--env retail --model-provider openai --user-model-provider openai --user-model $USER_MODEL --api-base $LITELLM_API_BASE --num-trials 5"

# --- Experiment 1: Qwen 4B Agent ---
echo "Running Experiment: Qwen 4B"
python3 run.py \
    $COMMON_ARGS \
    --model "qwen-4b" \
    --agent-strategy tool-calling \
    --log-dir results/qwen-4b

# --- Experiment 2: Qwen 8B Agent ---
echo "Running Experiment: Qwen 8B"
python3 run.py \
    $COMMON_ARGS \
    --model "qwen-8b" \
    --agent-strategy tool-calling \
    --log-dir results/qwen-8b

# --- Experiment 3: Qwen 14B Agent ---
echo "Running Experiment: Qwen 14B"
python3 run.py \
    $COMMON_ARGS \
    --model "qwen-14b" \
    --agent-strategy tool-calling \
    --log-dir results/qwen-14b

# --- Experiment 4: Qwen 32B Agent ---
echo "Running Experiment: Qwen 32B"
python3 run.py \
    $COMMON_ARGS \
    --model "qwen-32b" \
    --agent-strategy tool-calling \
    --log-dir results/qwen-32b

echo "All experiments completed."
