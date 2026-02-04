# Intel Gaudi Experiment Setup

This guide explains how to run tau-bench experiments on Intel Gaudi accelerators (HPU) instead of NVIDIA GPUs.

## Why Standard vLLM Fails on Gaudi

If you see this error:

```
INFO: No platform detected, vLLM is running on UnspecifiedPlatform
WARNING: Failed to import from vllm._C with ImportError('libcuda.so.1: cannot open shared object file')
RuntimeError: Failed to infer device type
```

This happens because:

1. **Standard vLLM is CUDA-only** - It tries to load NVIDIA CUDA libraries
2. **Gaudi requires a special plugin** - The `vllm-gaudi` hardware plugin
3. **No Habana libraries detected** - vLLM couldn't find any supported accelerator

## Prerequisites

- Access to Gaudi partition on SOL cluster
- Habana SynapseAI SDK installed on cluster (contact admin if missing)
- Python 3.10 environment

## One-Time Environment Setup

Run these commands on a SOL login node to create the Gaudi-compatible environment:

```bash
# Load modules (adjust based on your cluster)
module load mamba/latest
module load habana  # or habanalabs, intel-gaudi - depends on cluster

# Create conda environment
mamba create -n tau-gaudi python=3.10 -y
source activate tau-gaudi

# Clone vllm-gaudi and get verified vLLM commit
cd /scratch/$USER
git clone https://github.com/vllm-project/vllm-gaudi
cd vllm-gaudi
export VLLM_COMMIT=$(git show "origin/vllm/last-good-commit-for-vllm-gaudi:VLLM_STABLE_COMMIT" 2>/dev/null)
echo "Using vLLM commit: $VLLM_COMMIT"
cd ..

# Install vLLM for empty platform (bypasses CUDA dependency)
git clone https://github.com/vllm-project/vllm
cd vllm
git checkout $VLLM_COMMIT
pip install -r <(sed '/^torch/d' requirements/build.txt)
VLLM_TARGET_DEVICE=empty pip install --no-build-isolation -e .
cd ..

# Install Gaudi plugin
cd vllm-gaudi
pip install -e .
cd ..

# Install tau-bench
cd /scratch/$USER/tau-bench-project/tau-bench_CSE598
pip install -e .
```

## Available Scripts

| Script | Purpose |
|--------|---------|
| `gaudi_setup_check.sh` | Diagnostic script - run first to verify environment |
| `gaudi_experiment.sh` | Main experiment script for Gaudi |

## Running the Diagnostic

Before running experiments, verify your Gaudi environment:

```bash
sbatch /scratch/$USER/tau-bench-project/tau-bench_CSE598/SOL_env/gaudi_setup_check.sh
```

Check the output:

```bash
cat gaudi-check_<job_id>.out
```

The diagnostic checks:
- Gaudi hardware detection (`hl-smi`)
- Available Habana modules
- vLLM and vllm-gaudi plugin installation
- Required environment variables

## Running Experiments

Once the diagnostic passes:

```bash
sbatch /scratch/$USER/tau-bench-project/tau-bench_CSE598/SOL_env/gaudi_experiment.sh
```

Monitor progress:

```bash
# Check job status
squeue -u $USER

# View logs
tail -f SOL_env/logs/tau-gaudi_<job_id>.out
```

## SLURM Configuration

The Gaudi scripts use these SLURM settings:

```bash
#SBATCH --partition=gaudi
#SBATCH --qos=class_gaudi
#SBATCH --gres=gpu:hl225:1    # HL-225 = Gaudi 2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
```

## Key Differences from A100 Scripts

| Setting | A100 (CUDA) | Gaudi (HPU) |
|---------|-------------|-------------|
| Device flag | (none, auto-detect) | `--device hpu` |
| Block size | 16 (default) | `--block-size 128` (optimal for BF16) |
| Eager mode | `--enforce-eager` | Not used (HPU Graphs preferred) |
| Memory | `--gpu-memory-utilization 0.90` | Same |
| Modules | `cuda-12.1.1-gcc-12.1.0` | `habana` |

## Environment Variables

These are set automatically in the scripts:

```bash
# Core Habana settings
export HABANA_VISIBLE_DEVICES=all
export PT_HPU_LAZY_MODE=1                    # Enable HPU Graphs
export PT_HPU_ENABLE_LAZY_COLLECTIVES=true   # For tensor parallelism

# vLLM Gaudi settings
export VLLM_SKIP_WARMUP=false
export VLLM_GRAPH_RESERVED_MEM=0.1
export VLLM_GRAPH_PROMPT_RATIO=0.3

# Debug (uncomment if needed)
# export VLLM_LOGGING_LEVEL=DEBUG
```

## vLLM Serve Command

For Gaudi, the vLLM serve command looks like:

```bash
vllm serve Qwen/Qwen3-8B \
    --device hpu \
    --host 0.0.0.0 \
    --port 8000 \
    --block-size 128 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 16384 \
    --trust-remote-code \
    --disable-log-requests \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
```

## Results Location

Results are saved to:

```
SOL_env/results_gaudi/{env}/{strategy}/
```

## Troubleshooting

### "hl-smi not found"

Habana drivers not loaded. Try:

```bash
module load habana
# or
module load habanalabs
```

If no module exists, contact cluster admin to install Habana SynapseAI SDK.

### "vllm-gaudi plugin: NOT INSTALLED"

Re-run the setup steps above. Make sure you're in the `tau-gaudi` conda environment.

### "Failed to infer device type"

1. Check `hl-smi` works
2. Verify `habana_frameworks` is installed: `python -c "import habana_frameworks"`
3. Set debug logging: `export VLLM_LOGGING_LEVEL=DEBUG`

### Model loading fails

Gaudi may have different model support. Check [vllm-gaudi supported models](https://github.com/vllm-project/vllm-gaudi#supported-models).

## References

- [vLLM Gaudi Installation](https://docs.vllm.ai/en/v0.9.2/getting_started/installation/intel_gaudi.html)
- [vllm-gaudi GitHub](https://github.com/vllm-project/vllm-gaudi)
- [Habana Documentation](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/vLLM_Inference/vLLM_FAQs.html)
- [Gaudi 2 Specifications](https://habana.ai/products/gaudi2/)
