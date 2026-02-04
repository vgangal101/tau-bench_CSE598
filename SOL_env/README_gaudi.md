# Intel Gaudi Experiments on SOL

This guide explains how to run tau-bench experiments on Intel Gaudi2 accelerators on the ASU SOL cluster.

Based on the **ASU RC Workshop: "Introducing the Gaudi2 and Demystifying AI Processors"**

## SOL Gaudi2 Hardware

| Component | Details |
|-----------|---------|
| Nodes | 10 (gaudi001-gaudi010), 200+ more coming |
| Accelerator | HL-225 (Gaudi2) |
| HPUs per node | 8 |
| Memory per HPU | 96GB HBM |
| CPUs per node | 152 |
| Driver | SynapseAI 1.23.0 |

## Quick Start Options

### Option 1: Use SOL's Pre-hosted API (Easiest)

SOL already has LLMs running on Gaudi2 with an OpenAI-compatible API. No local setup needed!

**Step 1: Get API Key**
1. Go to https://voyager.rc.asu.edu/
2. Navigate to "LLM Access" tab
3. Click "Create Key"

**Step 2: Run Experiments**
```bash
export SOL_API_KEY="your-api-key"
bash SOL_env/gaudi_api_experiment.sh
```

**Available Models:**
| Model | Context Length |
|-------|----------------|
| llama4-scout-17b | 66K |
| qwen3-30b-a3b-instruct-2507 | 131K |
| qwen3-235b-a22b-instruct-2507 | 262K |

### Option 2: Run Local vLLM on Gaudi Node

If you need custom models or configurations:

```bash
sbatch SOL_env/gaudi_experiment.sh
```

### Option 3: Interactive Gaudi Shell

For development and testing:

```bash
interactive -p gaudi -c 30 --mem=30G -G 3 -t 0-6
```

## Pre-configured Environments on SOL

SOL provides ready-to-use Jupyter kernels:

| Kernel | Purpose |
|--------|---------|
| `gaudi-pytorch` | General PyTorch on Gaudi |
| `gaudi-pytorch-vllm` | vLLM inference on Gaudi |
| `gaudi-pytorch-diffusion` | Stable Diffusion on Gaudi |

### Using Jupyter

1. Go to SOL JupyterHub
2. Select partition: `gaudi`
3. QOS: `public`
4. GPU Resources: `gpu:hl225:1`
5. Select kernel: `gaudi-pytorch-vllm`

## Gaudi Resources on SOL

Pre-configured containers and guides are available at:

```
/data/sse/gaudi/
/data/sse/gaudi/guides/
/data/sse/gaudi/notebooks/
```

Example notebooks demonstrating MNIST training:
- `/data/sse/gaudi/notebooks/mnist-training-lazy.ipynb`
- `/data/sse/gaudi/notebooks/mnist-training-eager.ipynb`

## SLURM Configuration

### Interactive Session
```bash
interactive -p gaudi -c 30 --mem=30G -G 3 -t 0-6
```

### Batch Job Header
```bash
#SBATCH --partition=gaudi
#SBATCH --qos=public
#SBATCH --gres=gpu:hl225:1
#SBATCH --cpus-per-task=30
#SBATCH --mem=96G
```

## vLLM on Gaudi

### Supported Models (from RC Workshop)
- DeepSeek-R1
- Llama 3.x
- Qwen 2.5

### vLLM Command for Gaudi
```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --device hpu \
    --block-size 128 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 32768 \
    --trust-remote-code
```

Key differences from CUDA:
- `--device hpu` (explicitly specify Habana Processing Unit)
- `--block-size 128` (optimal for BF16 on Gaudi)
- No `--enforce-eager` (HPU Graphs preferred)

## PyTorch on Gaudi

### Code Modifications

```python
# Import Habana PyTorch core
import habana_frameworks.torch.core as htcore

# Target Gaudi device
device = torch.device("hpu")

# In lazy mode, call mark_step() after backward/optimizer
loss.backward()
htcore.mark_step()
optimizer.step()
htcore.mark_step()
```

### Execution Modes

| Mode | Description |
|------|-------------|
| Eager | Standard PyTorch, immediate execution |
| Lazy | Graph-based, requires `mark_step()` |
| torch.compile | Modern replacement for lazy mode |

## Available Scripts

| Script | Purpose |
|--------|---------|
| `gaudi_api_experiment.sh` | Use SOL's hosted API (no local setup) |
| `gaudi_experiment.sh` | Run local vLLM on Gaudi node |
| `gaudi_setup_check.sh` | Diagnostic to verify environment |

## Checking Gaudi Status

On a Gaudi node, use:
```bash
hl-smi
```
(equivalent to `nvidia-smi` for GPUs)

## Troubleshooting

### "RuntimeError: Failed to infer device type"

Standard vLLM tries to use CUDA. Solutions:
1. Use SOL's pre-configured `gaudi-pytorch-vllm` environment
2. Check `/data/sse/gaudi/guides/` for setup instructions
3. Use the hosted API instead

### Environment Not Found

Check available resources:
```bash
ls -la /data/sse/gaudi/
cat /data/sse/gaudi/guides/*.md
```

### API Connection Failed

1. Verify API key at https://voyager.rc.asu.edu/
2. Check model availability on Voyager dashboard
3. Test connection: `curl -H "Authorization: Bearer $SOL_API_KEY" https://openai.rc.asu.edu/v1/models`

## References

- [Habana Documentation](https://docs.habana.ai/en/latest/index.html)
- [vLLM Gaudi Plugin](https://github.com/vllm-project/vllm-gaudi)
- [Intel Gaudi2 White Paper](https://www.intel.com/content/www/us/en/content-details/839363/intel-gaudi-2-ai-accelerators-white-paper.html)
- SOL Voyager: https://voyager.rc.asu.edu/
