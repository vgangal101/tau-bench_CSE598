#!/bin/bash
#SBATCH --job-name=vllm-32b
#SBATCH --partition=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/vllm_32b_%j.out
#SBATCH --error=logs/vllm_32b_%j.err

echo "=== vLLM Server Started at $(date) ==="
echo "Model: Qwen3-32B (Agent)"
echo "Port: 8004"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

module load mamba/latest
module load cuda-12.1.1-gcc-12.1.0
source activate tau-bench

export HF_HOME=/scratch/$USER/hf_cache
mkdir -p $HF_HOME

nvidia-smi

echo "=== Starting vLLM Server ==="

vllm serve Qwen/Qwen3-32B \
    --host 0.0.0.0 \
    --port 8004 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192 \
    --trust-remote-code

echo "=== Server Stopped at $(date) ==="
