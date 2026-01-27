#!/bin/bash
#SBATCH --job-name=day1_exp
#SBATCH --partition=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/day1_exp_%j.out
#SBATCH --error=logs/day1_exp_%j.err

echo "=== Day 1 Experiments Started at $(date) ==="
echo "Experiment: User (GPT-OSS-20B) + Agent (Qwen-4B)"
echo "Strategies: tool-calling, act, react"
echo "Envs: retail, airline"

# Configuration
# Run `python test_vllm_servers.py <args>` first to get these IP addresses if not known
# OR, replace with the specific node names/IPs if you know them.
# For automation, we assume this script is run where specific node vars are set
# BUT for now, we will assume this is run AFTER setting up the vars or editing this file.

# Check if node vars are passed or set, otherwise warn
if [ -z "$USER_NODE" ] || [ -z "$AGENT_NODE_4B" ]; then
    echo "⚠️  WARNING: USER_NODE or AGENT_NODE_4B variables are not set."
    echo "    Please export them before running this script, or edit the script."
    echo "    Example: export USER_NODE=sg001; export AGENT_NODE_4B=sg002; sbatch day1_experiments.sh"
    # We will exit to prevent running with bad URLs, but comment out 'exit 1' if you want to hardcode
    exit 1
fi

USER_URL="http://${USER_NODE}:8000/v1"
AGENT_URL="http://${AGENT_NODE_4B}:8001/v1"

echo "User URL: $USER_URL"
echo "Agent URL: $AGENT_URL"

module load mamba/latest
module load cuda-12.1.1-gcc-12.1.0
source activate tau-bench

export OPENAI_API_KEY="dummy"

# Loop through environments
for ENV in retail airline; do
    echo ""
    echo ">>> Running Environment: $ENV"
    
    # Loop through strategies
    for STRATEGY in tool-calling act react; do
        echo "  > Strategy: $STRATEGY"
        
        # Construct log dir
        LOG_DIR="results/day1/${ENV}/${STRATEGY}"
        mkdir -p $LOG_DIR
        
        # Run command
        # Note: --task-ids or --end-index can be adjusted. Defaulting to first 10 for "day 1" trial if not specified.
        # Or run all? Let's run a subset for validity check first: start-index 0, end-index 10
        
        CMD="python run.py \
            --env ${ENV} \
            --agent-strategy ${STRATEGY} \
            --model qwen \
            --model-provider openai \
            --model-base-url ${AGENT_URL} \
            --user-model gpt-oss \
            --user-model-provider openai \
            --user-model-base-url ${USER_URL} \
            --len-short-traj-displays 2 \
            --few-shot-displays-path few_shot_data/${ENV}_few_shot_data.jsonl \
            --log-dir ${LOG_DIR} \
            --max-concurrency 5 \
            --num-trials 1 \
            --end-index 3 \
            --shuffle 0"
            
        echo "    Executing..."
        # echo $CMD
        eval $CMD
        
        echo "    Done $STRATEGY for $ENV"
    done
done

echo "=== Experiments Completed at $(date) ==="
