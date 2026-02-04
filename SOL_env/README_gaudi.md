# Intel Gaudi Experiments for tau-bench

This guide explains how to run tau-bench experiments on Intel Gaudi2 (HPU) accelerators on the ASU SOL cluster.

## SOL Gaudi2 Hardware

| Component | Details |
|-----------|---------|
| Nodes | 10 (gaudi001-gaudi010), 200+ more coming |
| Accelerator | HL-225 (Gaudi2) |
| HPUs per node | 8 |
| Memory per HPU | 96GB HBM |
| CPUs per node | 152 |
| Driver | SynapseAI 1.23.0 |

## Prerequisites

1. **SOL Cluster Access**: You need an account on the ASU SOL cluster
2. **SLURM Account**: `class_cse59827694spring2026`
3. **tau-bench Environment**: Created automatically if missing

## Quick Start

### Submit a Single Experiment

```bash
# From the repository root directory
cd /scratch/$USER/tau-bench-project/tau-bench_CSE598

# Submit a specific experiment
sbatch SOL_env/4b_run/gaudi_experiment_4b_retail_act.sh
```

### Submit All Experiments

```bash
# Submit all 24 experiments at once
for script in SOL_env/*/gaudi_experiment_*.sh; do sbatch $script; done
```

### Monitor Jobs

```bash
# Check job status
squeue -u $USER

# View job logs (replace JOB_ID with actual job ID)
tail -f SOL_env/4b_run/logs/tau-gaudi-4b-retail-act_JOB_ID.out

# Cancel a job
scancel JOB_ID

# Cancel all your jobs
scancel -u $USER
```

## Available Scripts (24 Total)

### Script Naming Convention
```
gaudi_experiment_{MODEL_SIZE}_{ENVIRONMENT}_{STRATEGY}.sh
```

### 4B Model Scripts
| Environment | Strategy | Script |
|-------------|----------|--------|
| retail | act | `4b_run/gaudi_experiment_4b_retail_act.sh` |
| retail | react | `4b_run/gaudi_experiment_4b_retail_react.sh` |
| retail | tool-calling | `4b_run/gaudi_experiment_4b_retail_tool-calling.sh` |
| airline | act | `4b_run/gaudi_experiment_4b_airline_act.sh` |
| airline | react | `4b_run/gaudi_experiment_4b_airline_react.sh` |
| airline | tool-calling | `4b_run/gaudi_experiment_4b_airline_tool-calling.sh` |

### 8B Model Scripts
| Environment | Strategy | Script |
|-------------|----------|--------|
| retail | act | `8b_run/gaudi_experiment_8b_retail_act.sh` |
| retail | react | `8b_run/gaudi_experiment_8b_retail_react.sh` |
| retail | tool-calling | `8b_run/gaudi_experiment_8b_retail_tool-calling.sh` |
| airline | act | `8b_run/gaudi_experiment_8b_airline_act.sh` |
| airline | react | `8b_run/gaudi_experiment_8b_airline_react.sh` |
| airline | tool-calling | `8b_run/gaudi_experiment_8b_airline_tool-calling.sh` |

### 14B Model Scripts
| Environment | Strategy | Script |
|-------------|----------|--------|
| retail | act | `14b_run/gaudi_experiment_14b_retail_act.sh` |
| retail | react | `14b_run/gaudi_experiment_14b_retail_react.sh` |
| retail | tool-calling | `14b_run/gaudi_experiment_14b_retail_tool-calling.sh` |
| airline | act | `14b_run/gaudi_experiment_14b_airline_act.sh` |
| airline | react | `14b_run/gaudi_experiment_14b_airline_react.sh` |
| airline | tool-calling | `14b_run/gaudi_experiment_14b_airline_tool-calling.sh` |

### 32B Model Scripts
| Environment | Strategy | Script |
|-------------|----------|--------|
| retail | act | `32b_run/gaudi_experiment_32b_retail_act.sh` |
| retail | react | `32b_run/gaudi_experiment_32b_retail_react.sh` |
| retail | tool-calling | `32b_run/gaudi_experiment_32b_retail_tool-calling.sh` |
| airline | act | `32b_run/gaudi_experiment_32b_airline_act.sh` |
| airline | react | `32b_run/gaudi_experiment_32b_airline_react.sh` |
| airline | tool-calling | `32b_run/gaudi_experiment_32b_airline_tool-calling.sh` |

## Configuration

### Common Settings (All Scripts)

| Setting | Value |
|---------|-------|
| MAX_MODEL_LEN | 32768 |
| MAX_CONCURRENCY | 3 |
| NUM_TRIALS | 5 |
| Time Limit | 6 hours |
| Partition | gaudi |
| QOS | class_gaudi |

### Dual-Server Architecture

All experiments use a **consistent 32B user simulator** (Qwen3-32B) to ensure fair comparison across agent model sizes. The 4B, 8B, and 14B scripts run two separate vLLM servers:

| Server | Model | HPUs | Tensor Parallel | Purpose |
|--------|-------|------|-----------------|---------|
| User Server | Qwen3-32B | 0,1 | 2 | User simulator |
| Agent Server | Qwen3-{4B,8B,14B} | 2 | 1 | Agent model |

### Resource Allocation by Agent Model

| Agent Model | Total HPUs | CPUs | Memory | Architecture |
|-------------|------------|------|--------|--------------|
| Qwen3-4B | 3 | 24 | 160G | 32B (TP=2) + 4B (TP=1) |
| Qwen3-8B | 3 | 24 | 160G | 32B (TP=2) + 8B (TP=1) |
| Qwen3-14B | 3 | 24 | 160G | 32B (TP=2) + 14B (TP=1) |
| Qwen3-32B | 2 | 16 | 128G | Single 32B (TP=2) |

### Server Configuration Details

| Server | MAX_NUM_SEQS | MAX_PREFILL | GPU Util |
|--------|--------------|-------------|----------|
| User (32B) | 8 | 2 | 0.95 |
| Agent (4B) | 16 | 8 | 0.90 |
| Agent (8B) | 16 | 8 | 0.90 |
| Agent (14B) | 12 | 6 | 0.90 |
| Agent (32B) | 8 | 2 | 0.95 |

### Port Assignments (to avoid conflicts)

Each script uses two ports for the dual-server setup:

| Agent Model | Retail User | Retail Agent | Airline User | Airline Agent |
|-------------|-------------|--------------|--------------|---------------|
| 4B | 8200 | 8000 | 8300 | 8100 |
| 8B | 8201 | 8001 | 8301 | 8101 |
| 14B | 8202 | 8002 | 8302 | 8102 |
| 32B | - | 8003 | - | 8103 |

Note: 32B scripts use a single server (same model for user and agent).

## Results Location

Results are saved to:
```
SOL_env/{MODEL}_run/results_gaudi/{ENVIRONMENT}/{STRATEGY}/
```

Example:
```
SOL_env/4b_run/results_gaudi/retail/act/
SOL_env/8b_run/results_gaudi/airline/react/
```

## Log Files

Logs are saved to:
```
SOL_env/{MODEL}_run/logs/
```

Each job creates:
- `tau-gaudi-{MODEL}-{ENV}-{STRATEGY}_{JOB_ID}.out` - Main output
- `tau-gaudi-{MODEL}-{ENV}-{STRATEGY}_{JOB_ID}.err` - Error log
- `gaudi_vllm_user_32b_{JOB_ID}.log` - User model (32B) vLLM server log
- `gaudi_vllm_agent_{MODEL}_{JOB_ID}.log` - Agent model vLLM server log

Note: 32B agent scripts have a single vLLM log since user and agent share the same server.

## Alternative: SOL's Hosted API

SOL provides LLMs on Gaudi2 with an OpenAI-compatible API (no local setup needed).

**Step 1: Get API Key**
1. Go to https://voyager.rc.asu.edu/
2. Navigate to "LLM Access" tab
3. Click "Create Key"

**Step 2: Run Experiments**
```bash
export SOL_API_KEY="your-api-key"
bash SOL_env/gaudi_api_experiment.sh
```

**Available Hosted Models:**
| Model | Context Length |
|-------|----------------|
| qwen3-30b-a3b-instruct-2507 | 131K |
| qwen3-235b-a22b-instruct-2507 | 262K |

## How It Works

1. **Job Submission**: Script submitted to SLURM gaudi partition
2. **Environment Setup**: Cache directories and Gaudi env vars configured
3. **User Server Start**: Apptainer container launches vLLM with Qwen3-32B (TP=2 on HPUs 0,1)
4. **Agent Server Start**: Second Apptainer container launches vLLM with agent model (TP=1 on HPU 2)
5. **Health Check**: Waits for both servers (up to 30 min for 32B, 20 min for agent)
6. **tau-bench Run**: `python run.py` with `--user-model-base-url` and `--model-base-url` pointing to separate servers
7. **Cleanup**: Both vLLM servers terminated

Note: 32B agent scripts use a single server for both user and agent models.

## First-Time Setup

Scripts auto-create the environment, but you can do it manually:

```bash
module load mamba/latest
mamba create -n tau-bench -c conda-forge python=3.11 -y
source activate tau-bench
cd /scratch/$USER/tau-bench-project/tau-bench_CSE598
pip install -e .
```

## Troubleshooting

### Job Stuck in Queue
```bash
squeue -p gaudi
sacctmgr show user $USER withassoc
```

### vLLM Server Failed
Check the vLLM logs (both user and agent servers):
```bash
# User model (32B) log
cat SOL_env/4b_run/logs/gaudi_vllm_user_32b_JOB_ID.log

# Agent model log
cat SOL_env/4b_run/logs/gaudi_vllm_agent_4b_JOB_ID.log
```

### Context Window Exceeded
If you see `ContextWindowExceededError`, increase `MAX_MODEL_LEN` in the script.

### Model Download Slow
First run downloads weights to `/scratch/$USER/hf_cache`. Subsequent runs use cache.

## Useful Commands

```bash
# Check Gaudi hardware (on compute node)
hl-smi

# Interactive Gaudi session (3 HPUs for dual-server testing)
interactive -p gaudi -c 24 --mem=160G -G 3 -t 0-6

# View Gaudi nodes
sinfo -p gaudi

# Check available resources
ls -la /data/sse/gaudi/
```

## Utility Scripts

| Script | Purpose |
|--------|---------|
| `gaudi_api_experiment.sh` | Use SOL's hosted API |
| `gaudi_setup_check.sh` | Diagnostic to verify environment |
| `gaudi_env_setup.sh` | Environment setup helper |

## References

- [Habana Documentation](https://docs.habana.ai/en/latest/index.html)
- [vLLM Gaudi Plugin](https://github.com/vllm-project/vllm-gaudi)
- SOL Voyager: https://voyager.rc.asu.edu/
