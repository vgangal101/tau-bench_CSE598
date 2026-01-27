#!/bin/bash
#SBATCH --job-name=vllm-qwen-32b
#SBATCH --partition=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/vllm_qwen_32b_%j.out
#SBATCH --error=logs/vllm_qwen_32b_%j.err

echo "=== vLLM Server Started at $(date) ==="
echo "Model: Qwen2.5-32B-Instruct (User Simulator)"
echo "Port: 8000"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

module load mamba/latest
module load cuda-12.1.1-gcc-12.1.0
source activate tau-bench

export HF_HOME=/scratch/$USER/hf_cache
export VLLM_USE_V1=0
mkdir -p $HF_HOME

nvidia-smi

echo "=== Starting vLLM Server ==="

vllm serve Qwen/Qwen2.5-32B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096 \
    --trust-remote-code \
    --enforce-eager

echo "=== Server Stopped at $(date) ==="
