# Intel Gaudi Experiments for tau-bench

This guide explains how to run tau-bench experiments on Intel Gaudi2 (HPU) accelerators on the ASU SOL cluster.

## Prerequisites

1. **SOL Cluster Access**: You need an account on the ASU SOL cluster
2. **SLURM Account**: `class_cse59827694spring2026`
3. **tau-bench Environment**: The `tau-bench` conda environment (created automatically if missing)

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

## Available Scripts

### Script Naming Convention
```
gaudi_experiment_{MODEL_SIZE}_{ENVIRONMENT}_{STRATEGY}.sh
```

### All 24 Scripts

| Model | Environment | Strategy | Script Path |
|-------|-------------|----------|-------------|
| 4B | retail | act | `4b_run/gaudi_experiment_4b_retail_act.sh` |
| 4B | retail | react | `4b_run/gaudi_experiment_4b_retail_react.sh` |
| 4B | retail | tool-calling | `4b_run/gaudi_experiment_4b_retail_tool-calling.sh` |
| 4B | airline | act | `4b_run/gaudi_experiment_4b_airline_act.sh` |
| 4B | airline | react | `4b_run/gaudi_experiment_4b_airline_react.sh` |
| 4B | airline | tool-calling | `4b_run/gaudi_experiment_4b_airline_tool-calling.sh` |
| 8B | retail | act | `8b_run/gaudi_experiment_8b_retail_act.sh` |
| 8B | retail | react | `8b_run/gaudi_experiment_8b_retail_react.sh` |
| 8B | retail | tool-calling | `8b_run/gaudi_experiment_8b_retail_tool-calling.sh` |
| 8B | airline | act | `8b_run/gaudi_experiment_8b_airline_act.sh` |
| 8B | airline | react | `8b_run/gaudi_experiment_8b_airline_react.sh` |
| 8B | airline | tool-calling | `8b_run/gaudi_experiment_8b_airline_tool-calling.sh` |
| 14B | retail | act | `14b_run/gaudi_experiment_14b_retail_act.sh` |
| 14B | retail | react | `14b_run/gaudi_experiment_14b_retail_react.sh` |
| 14B | retail | tool-calling | `14b_run/gaudi_experiment_14b_retail_tool-calling.sh` |
| 14B | airline | act | `14b_run/gaudi_experiment_14b_airline_act.sh` |
| 14B | airline | react | `14b_run/gaudi_experiment_14b_airline_react.sh` |
| 14B | airline | tool-calling | `14b_run/gaudi_experiment_14b_airline_tool-calling.sh` |
| 32B | retail | act | `32b_run/gaudi_experiment_32b_retail_act.sh` |
| 32B | retail | react | `32b_run/gaudi_experiment_32b_retail_react.sh` |
| 32B | retail | tool-calling | `32b_run/gaudi_experiment_32b_retail_tool-calling.sh` |
| 32B | airline | act | `32b_run/gaudi_experiment_32b_airline_act.sh` |
| 32B | airline | react | `32b_run/gaudi_experiment_32b_airline_react.sh` |
| 32B | airline | tool-calling | `32b_run/gaudi_experiment_32b_airline_tool-calling.sh` |

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
| HPU | 1 x HL-225 (96GB HBM) |

### Model-Specific Settings

| Model | Memory | MAX_NUM_SEQS | GPU Util |
|-------|--------|--------------|----------|
| Qwen3-4B | 32G | 16 | 0.90 |
| Qwen3-8B | 32G | 16 | 0.90 |
| Qwen3-14B | 48G | 12 | 0.90 |
| Qwen3-32B | 64G | 8 | 0.95 |

### Port Assignments (to avoid conflicts)

Scripts use different ports so multiple jobs can run on the same node:

| Model | Retail Port | Airline Port |
|-------|-------------|--------------|
| 4B | 8000 | 8100 |
| 8B | 8001 | 8101 |
| 14B | 8002 | 8102 |
| 32B | 8003 | 8103 |

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

Each job creates multiple log files:
- `tau-gaudi-{MODEL}-{ENV}-{STRATEGY}_{JOB_ID}.out` - Main job output
- `tau-gaudi-{MODEL}-{ENV}-{STRATEGY}_{JOB_ID}.err` - Error log
- `gaudi_vllm_{MODEL}_{ENV}_{STRATEGY}_{JOB_ID}.log` - vLLM server log

## How It Works

1. **Job Submission**: Script is submitted to SLURM gaudi partition
2. **Environment Setup**: Cache directories and Gaudi environment variables are configured
3. **vLLM Server Start**: Apptainer container launches vLLM with the Qwen model
4. **Health Check**: Script waits for vLLM server to be ready (up to 15-30 min for large models)
5. **tau-bench Activation**: Conda environment is loaded
6. **Experiment Run**: `python run.py` executes with specified parameters
7. **Cleanup**: vLLM server is terminated when done

## Troubleshooting

### Job Stuck in Queue
```bash
# Check queue status
squeue -p gaudi

# Check your account limits
sacctmgr show user $USER withassoc
```

### vLLM Server Failed to Start
Check the vLLM log:
```bash
cat SOL_env/4b_run/logs/gaudi_vllm_4b_retail_act_JOB_ID.log
```

Common issues:
- **OOM**: Model too large for available memory
- **Port conflict**: Another job using the same port (should be rare with port scheme)

### Context Window Exceeded Error
If you see `ContextWindowExceededError`, the `MAX_MODEL_LEN` may need to be increased. Current setting is 32768 tokens.

### Model Download Slow
First run for each model size downloads weights from HuggingFace to `/scratch/$USER/hf_cache`. Subsequent runs use cached weights.

## First-Time Setup (If tau-bench env doesn't exist)

The scripts automatically create the environment, but you can do it manually:

```bash
module load mamba/latest
mamba create -n tau-bench -c conda-forge python=3.11 -y
source activate tau-bench
cd /scratch/$USER/tau-bench-project/tau-bench_CSE598
pip install -e .
```

## Architecture Notes

- **Single Node**: Each script runs on 1 Gaudi node with 1 HPU
- **Same Model for User & Agent**: Due to QOS limits, the same model serves both the user simulator and the agent
- **Apptainer Container**: Uses SOL's pre-built vLLM-Gaudi container at `/data/sse/gaudi/containers/vllm-gaudi.sif`

## Useful Commands

```bash
# Check Gaudi hardware status (on compute node)
hl-smi

# Interactive Gaudi session
interactive -p gaudi -c 30 --mem=30G -G 1 -t 0-2

# View all available Gaudi nodes
sinfo -p gaudi
```
