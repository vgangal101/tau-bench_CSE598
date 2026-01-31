# Multi-Node Experiment

This script runs tau-bench experiments using **2 separate nodes** on the SOL cluster, with each node hosting one vLLM server:

- **Node 1**: User Simulator (Qwen/Qwen3-32B)
- **Node 2**: Agent Model (Qwen/Qwen3-32B)

## Prerequisites

1. **tau-bench conda environment**: Must have `tau-bench` environment created
   ```bash
   mamba create -n tau-bench python=3.10
   source activate tau-bench
   pip install -e .
   ```

2. **HuggingFace cache**: Models will be downloaded to `/scratch/$USER/hf_cache`

## How to Run

### Option 1: Submit from repo root (recommended)
```bash
cd /scratch/$USER/tau-bench-project/tau-bench
sbatch SOL_env/multi_node_experiment.sh
```

### Option 2: Submit from SOL_env directory
```bash
cd /scratch/$USER/tau-bench-project/tau-bench/SOL_env
sbatch multi_node_experiment.sh
```

## Monitor Job

```bash
# Check job status
squeue -u $USER

# Watch output in real-time (logs are in SOL_env/logs/)
tail -f SOL_env/logs/tau-multi-node_<job_id>.out

# Check for errors
cat SOL_env/logs/tau-multi-node_<job_id>.err
```

## Log Files

All logs are written **directly** to `SOL_env/logs/` from the start (not moved after completion):

| File | Description |
|------|-------------|
| `tau-multi-node_<job_id>.out` | SLURM stdout (main experiment output) |
| `tau-multi-node_<job_id>.err` | SLURM stderr (errors and warnings) |
| `multi_node_user_<job_id>.log` | vLLM user simulator server log |
| `multi_node_agent_<job_id>.log` | vLLM agent model server log |

## Results

Experiment results are saved to:
```
SOL_env/results_multi_node/
├── retail/
│   ├── tool-calling/
│   ├── act/
│   └── react/
└── airline/
    ├── tool-calling/
    ├── act/
    └── react/
```

## Configuration

| Parameter | Value |
|-----------|-------|
| Nodes | 2 |
| GPUs per node | 1x A100 80GB |
| Memory per node | 96GB |
| Time limit | 2 hours |
| User Model | Qwen/Qwen3-32B |
| Agent Model | Qwen/Qwen3-32B |
| Max context length | 32768 tokens |
| GPU memory utilization | 90% |

## Troubleshooting

### No output files / "Permission denied"
**Cause**: `SCRIPT_DIR` resolved incorrectly if submitted from an unrelated directory.
**Fix**: Submit from repo root or SOL_env directory:
```bash
cd /scratch/$USER/tau-bench-project/tau-bench
sbatch SOL_env/multi_node_experiment.sh
```

### vLLM server fails to start
Check the vLLM logs:
```bash
cat SOL_env/logs/multi_node_user_<job_id>.log
cat SOL_env/logs/multi_node_agent_<job_id>.log
```

Common issues:
- Out of GPU memory: Reduce `--gpu-memory-utilization` or `--max-model-len`
- Model not found: Check HuggingFace cache and network access

### Server health check fails
The script waits up to 10 minutes for each server. If timeout occurs:
1. Check if model download is still in progress
2. Verify GPU allocation with `squeue -u $USER`
3. Check CUDA errors in the `.err` file

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Head Node (srun)                        │
│                                                              │
│  ┌─────────────────────┐    ┌─────────────────────┐         │
│  │  Node 1 (USER_NODE) │    │  Node 2 (AGENT_NODE)│         │
│  │  ┌───────────────┐  │    │  ┌───────────────┐  │         │
│  │  │ vLLM Server   │  │    │  │ vLLM Server   │  │         │
│  │  │ Qwen3-32B     │  │    │  │ Qwen3-32B     │  │         │
│  │  │ Port 8000     │  │    │  │ Port 8000     │  │         │
│  │  │ (User Sim)    │  │    │  │ (Agent)       │  │         │
│  │  └───────────────┘  │    │  └───────────────┘  │         │
│  │        ▲            │    │        ▲            │         │
│  └────────│────────────┘    └────────│────────────┘         │
│           │                          │                       │
│           └──────────┬───────────────┘                       │
│                      │                                       │
│              ┌───────▼───────┐                              │
│              │   run.py      │                              │
│              │ (experiments) │                              │
│              └───────────────┘                              │
└─────────────────────────────────────────────────────────────┘
```

Communication happens over the cluster network using hostnames (e.g., `http://node1:8000`).
