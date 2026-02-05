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
| MAX_MODEL_LEN | 40000 |
| MAX_CONCURRENCY | 2 |
| NUM_TRIALS | 5 |
| Time Limit | 10 hours |
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
| Qwen3-4B | 3 | 24 | 160G | 32B (TP=2, HPU 0,1) + 4B (TP=1, HPU 2) |
| Qwen3-8B | 3 | 24 | 160G | 32B (TP=2, HPU 0,1) + 8B (TP=1, HPU 2) |
| Qwen3-14B | 3 | 24 | 160G | 32B (TP=2, HPU 0,1) + 14B (TP=1, HPU 2) |
| Qwen3-32B | 4 | 32 | 200G | 32B (TP=2, HPU 0,1) + 32B (TP=2, HPU 2,3) |

### Server Configuration Details

| Server | MAX_NUM_SEQS | MAX_PREFILL | GPU Util | Notes |
|--------|--------------|-------------|----------|-------|
| User (32B) | 4 | 1 | 0.90 | Stability-optimized for long runs |
| Agent (4B) | 16 | 8 | 0.85 | |
| Agent (8B) | 16 | 8 | 0.90 | |
| Agent (14B) | 12 | 6 | 0.90 | |
| Agent (32B) | 4 | 1 | 0.90 | Same as User (dual 32B servers) |

### Stability Settings Rationale

The User server (32B) settings were optimized for long-running experiments (115 retail tasks × 5 trials = 575 runs):

| Setting | Value | Reason |
|---------|-------|--------|
| `--gpu-memory-utilization` | 0.90 | Prevents OOM during extended runs (was 0.95) |
| `--max-num-seqs` | 4 | Only need 2× MAX_CONCURRENCY headroom (was 6-8) |
| `--max-num-prefill-seqs` | 1 | Reduces concurrent prefill memory pressure (was 2) |
| `--max-model-len` | 40000 | Safe within Qwen3's 40960 max_position_embeddings |

**Why these matter for retail experiments:**
- Retail has 115 tasks vs airline's 50 tasks
- With 5 trials each, that's 575 server requests over many hours
- Memory fragmentation accumulates, causing "client has been closed" errors
- Lower `max-num-seqs` drastically reduces KV cache memory requirements

### Port Assignments (to avoid conflicts)

Each script uses two ports for the dual-server setup:

| Agent Model | Retail User | Retail Agent | Airline User | Airline Agent |
|-------------|-------------|--------------|--------------|---------------|
| 4B | 8200 | 8000 | 8300 | 8100 |
| 8B | 8201 | 8001 | 8301 | 8101 |
| 14B | 8202 | 8002 | 8302 | 8102 |
| 32B | 8200 | 8000 | 8300 | 8100 |

Note: All experiments now use dual-server architecture with separate User and Agent vLLM servers.

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
4. **Agent Server Start**: Second Apptainer container launches vLLM with agent model
   - 4B/8B/14B: TP=1 on HPU 2
   - 32B: TP=2 on HPUs 2,3
5. **Health Check**: Waits for both servers (up to 30 min for 32B user, 15-20 min for smaller agents)
6. **tau-bench Run**: `python run.py` with `--user-model-base-url` and `--model-base-url` pointing to separate servers
7. **Cleanup**: Both vLLM servers terminated

All experiments use dual-server architecture for isolation and stability.

## Script Robustness Features

All 24 Gaudi scripts include robustness features to prevent common issues:

### 1. Exclusive Node Access (`--exclusive`)

Each job gets exclusive access to a Gaudi node to prevent:
- **Port conflicts**: Multiple jobs using the same ports
- **HPU conflicts**: Hardcoded `HABANA_VISIBLE_DEVICES` causing resource contention

```bash
#SBATCH --exclusive  # One job per node
```

**Trade-off**: Jobs may wait longer in queue, but run reliably.

### 2. PID-Based Process Cleanup

The cleanup function uses process IDs instead of pattern matching:

```bash
# Safe - only kills this job's processes:
cleanup() {
    [ -n "$USER_PID" ] && kill $USER_PID 2>/dev/null
    [ -n "$AGENT_PID" ] && kill $AGENT_PID 2>/dev/null
    ...
}

# DANGEROUS - would kill ALL vLLM processes on node (including other jobs):
# cleanup() { pkill -f "vllm serve" ... }  # DON'T DO THIS
```

**Why it matters**: If multiple jobs ran on the same node and one failed, `pkill -f` would kill the other job's vLLM servers.

### 3. Stale Editable Install Fix

Scripts uninstall before reinstalling tau-bench:

```bash
pip uninstall tau_bench -y 2>/dev/null || true
pip install -e .
```

**Why it's needed**: Python editable installs create `.pth` files that point to the source directory. If the repo was moved or a previous install was corrupted, pip fails with:
```
OSError: [Errno 2] No such file or directory: '.../__editable__.tau_bench-0.1.0.pth'
```

**This is a user-level issue, NOT a node-level issue**:
- The conda environment (`~/.conda/envs/tau-bench/`) is on shared filesystem
- Same environment is used regardless of which node runs the job
- Uninstall clears stale `.pth` files before fresh install

### Issue Summary

| Issue | Scope | Cause | Fix |
|-------|-------|-------|-----|
| Port conflicts | Node-level | Multiple jobs, same ports | `--exclusive` |
| HPU conflicts | Node-level | Hardcoded `HABANA_VISIBLE_DEVICES` | `--exclusive` |
| Process kill race | Node-level | `pkill -f` kills wrong processes | PID-based cleanup |
| Stale `.pth` files | User-level | Moved repo or corrupted install | `pip uninstall` first |

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

### Stale Editable Install Error
If you see:
```
OSError: [Errno 2] No such file or directory: '.../__editable__.tau_bench-0.1.0.pth'
```
This is handled automatically by the scripts (uninstall before install). If it persists, manually clean up:
```bash
source activate tau-bench
pip uninstall tau_bench -y
rm -f ~/.conda/envs/tau-bench/lib/python3.11/site-packages/__editable__.tau_bench*.pth
```

### vLLM Server Killed Unexpectedly
If your vLLM server dies shortly after starting (e.g., "Shutdown complete" after 30-60 seconds), check if another job on the same node ran a cleanup. The `--exclusive` flag prevents this.

### "Cannot send a request, as the client has been closed" Error
This error means the vLLM server crashed during the experiment. Common causes:
1. **Memory pressure**: Reduce `--gpu-memory-utilization` or `--max-num-seqs`
2. **Long-running fragmentation**: Server memory fragments over hundreds of requests
3. **Context too long**: Some tasks exceed `--max-model-len`

The current settings (gpu-mem 0.90, max-num-seqs 4) are tuned to prevent this.

### Jobs Interfering with Each Other
If multiple jobs seem to affect each other:
1. Verify `--exclusive` is in the SBATCH directives
2. Check cleanup function uses PIDs, not `pkill -f`
3. Check port assignments are unique per model/environment

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
