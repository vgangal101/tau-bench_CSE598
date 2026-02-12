# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

τ-bench (tau-bench) is a benchmark for evaluating Tool-Agent-User interactions in real-world domains. It simulates dynamic conversations between a user (simulated by LLMs) and a language agent provided with domain-specific API tools and policy guidelines.

**Important**: τ²-bench has been released as an extension of this benchmark with code fixes and an additional telecom domain. This repository represents the original τ-bench implementation.

## Installation and Setup

Install from source:
```bash
pip install -e .
```

Set up required API keys as environment variables:
```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...
export MISTRAL_API_KEY=...
```

## Running the Benchmark

Basic command to run the benchmark:
```bash
python run.py --agent-strategy tool-calling --env retail --model gpt-4o --model-provider openai --user-model gpt-4o --user-model-provider openai --user-strategy llm --max-concurrency 10
```

Run specific tasks using `--task-ids`:
```bash
python run.py --agent-strategy tool-calling --env retail --model gpt-4o --model-provider openai --user-model gpt-4o --user-model-provider openai --user-strategy llm --max-concurrency 10 --task-ids 2 4 6
```

Run auto error identification:
```bash
python auto_error_identification.py --env <airline/retail> --platform openai --results-path <path_to_results> --max-concurrency 16 --output-path test-auto-error-identification --max-num-failed-results 10
```

## Architecture

### Core Components

1. **Environments** (`tau_bench/envs/`)
   - Base class: `Env` in [base.py](tau_bench/envs/base.py)
   - Two domains: `airline` and `retail`
   - Each environment has:
     - Tools: Domain-specific actions agents can take
     - Tasks: Evaluation scenarios with expected outputs/actions
     - Rules: Policy constraints that must be followed
     - Wiki: Domain knowledge provided to agents
     - Data: JSON files with domain state (users, orders/reservations, products/flights)

2. **Agents** (`tau_bench/agents/`)
   - Base class: `Agent` in [base.py](tau_bench/agents/base.py)
   - Implementations:
     - `ToolCallingAgent`: Native function calling (default)
     - `ChatReActAgent`: ReAct/Act patterns from the paper
     - `FewShotToolCallingAgent`: Few-shot learning approach
   - All agents implement `solve(env, task_index, max_num_steps)` → `SolveResult`

3. **Tools** (`tau_bench/envs/{domain}/tools/`)
   - Base class: `Tool` in [tool.py](tau_bench/envs/tool.py)
   - Each tool must implement:
     - `invoke(data, **kwargs)`: Execute the tool action
     - `get_info()`: Return OpenAI function calling format specification
   - Tools modify environment state (data) and return observations
   - Special tools:
     - `think`: Internal reasoning without observation
     - `transfer_to_human_agents`: Terminates episode in airline domain
     - `respond`: Agent response to user (not a tool, handled specially)

4. **Tasks** (`tau_bench/envs/{domain}/tasks*.py`)
   - Task splits: `train`, `dev`, `test` (retail only has splits)
   - Each task contains:
     - `user_id`: User identity in the simulation
     - `instruction`: Initial user request
     - `actions`: Ground truth sequence of actions
     - `outputs`: Expected information to communicate to user
   - Reward calculation checks:
     - Database state matches expected state (via hash comparison)
     - Required outputs appear in agent responses

5. **User Simulator** (`tau_bench/envs/user.py`)
   - Strategies: `llm`, `react`, `verify`, `reflection`
   - Simulates natural user responses to agent messages
   - Tracks conversation history and validates agent requests
   - Returns `###STOP###` when satisfied with resolution

6. **Model Utilities** (`tau_bench/model_utils/`)
   - Abstractions for different LLM providers via litellm
   - API routing, caching, token tracking, cost calculation
   - Supports: OpenAI, Anthropic, Google, Mistral, Anyscale

### Key Architectural Patterns

**Environment Step Loop**:
1. Agent receives observation (user message or tool result)
2. Agent calls tool or responds to user
3. Environment executes action and returns next observation
4. Loop continues until done (user satisfied or timeout)
5. Reward calculated by comparing final state to ground truth

**Reward Calculation** ([base.py:124](tau_bench/envs/base.py#L124)):
- Replays ground truth actions to compute expected data hash
- Compares actual data hash to expected hash
- Checks if all required outputs were communicated
- Returns 1.0 only if both checks pass, 0.0 otherwise

**Action Types**:
- Tool calls: Modify environment state, return tool observations
- User responses: Trigger user simulator, get user reply
- Terminate actions: End episode (e.g., transfer_to_human_agents)

**Data Isolation**:
- Each task run gets fresh data via `data_load_func`
- Prevents state leakage between tasks
- Allows parallel execution with `--max-concurrency`

## Domain-Specific Notes

### Airline Domain
- Data: flights, reservations, users
- Key tools: search flights, book/cancel/update reservations, send certificates
- Terminate tool: `transfer_to_human_agents`
- Only has `test` split

### Retail Domain
- Data: products, orders, users
- Key tools: find users, get order details, cancel/modify orders, returns/exchanges
- Task splits: `train`, `dev`, `test`
- More complex policy rules around order modifications

## Results and Metrics

Results are saved to `results/` directory with naming pattern:
```
{agent_strategy}-{model}-{temperature}_range_{start}-{end}_user-{user_model}-{user_strategy}_{timestamp}.json
```

Metrics calculated:
- Average reward
- Pass^k: Probability of success in best of k trials (from paper)

Historical trajectories available in `historical_trajectories/` directory.

## Common Patterns When Modifying

**Adding a new tool**:
1. Create tool file in `tau_bench/envs/{domain}/tools/`
2. Inherit from `Tool`, implement `invoke()` and `get_info()`
3. Add to `ALL_TOOLS` in `tau_bench/envs/{domain}/tools/__init__.py`

**Adding a new task**:
1. Add `Task` object to appropriate tasks file
2. Define `user_id`, `instruction`, `actions`, `outputs`
3. Test that ground truth actions produce expected state

**Adding a new agent strategy**:
1. Create agent class inheriting from `Agent`
2. Implement `solve(env, task_index, max_num_steps)` method
3. Add to `agent_factory()` in [run.py](tau_bench/run.py#L124)
4. Add command line argument choice

**Debugging failed tasks**:
- Check `results/*.json` for trajectory and error info
- Use `auto_error_identification.py` to classify failure types
- Verify tool outputs match expected format
- Check if required outputs were communicated to user

## SOL Cluster Experiments (vLLM)

Scripts for running experiments on ASU SOL cluster with local vLLM inference.

### First-Time Environment Setup (Required for New Users)

Each user must create the `tau-bench` conda environment **once** before running any jobs. Run these commands on a Sol login node:

```bash
module load mamba/latest
module load cuda-12.1.1-gcc-12.1.0
mamba create -n tau-bench -c conda-forge python=3.11 -y
source activate tau-bench
cd /scratch/$USER/tau-bench-project/tau-bench_CSE598
pip install -e .
pip install vllm
```

After this one-time setup, all experiment jobs will work automatically.

### Directory Structure

```
SOL_env/
├── generate_split_scripts.py          # Generator for all split experiment directories
├── multi_node_experiment.sh           # Multi-node: User + Agent on separate nodes
├── 4b_airline/                        # 4B agent, airline env (2 parts)
├── 4b_retail/                         # 4B agent, retail env (3 parts)
├── 8b_airline/                        # 8B agent, airline env (2 parts)
├── 8b_retail/                         # 8B agent, retail env (3 parts)
├── 14b_airline/                       # 14B agent, airline env (2 parts)
├── 14b_retail/                        # 14B agent, retail env (3 parts)
├── 32b_airline/                       # 32B agent, airline env (2 parts)
└── 32b_retail/                        # 32B agent, retail env (3 parts)
```

### FP16 Scripts (Default Precision)

| Script | User Model | Agent Model | Context Limit |
|--------|-----------|-------------|---------------|
| `day1/combined_experiment_4b.sh` | Qwen/Qwen3-32B | Qwen/Qwen3-4B | 36000 / 50000 |
| `day2/combined_experiment_8b.sh` | Qwen/Qwen3-32B | Qwen/Qwen3-8B | 36000 / 50000 |
| `day3/combined_experiment_14b.sh` | Qwen/Qwen3-32B | Qwen/Qwen3-14B | 36000 / 50000 |
| `day4/combined_experiment_32b.sh` | Qwen/Qwen3-32B | Qwen/Qwen3-32B | 36000 / 36000 |

### INT8 GPTQ Scripts (Quantized)

Uses pre-quantized GPTQ INT8 models from HuggingFace for reduced memory and longer context.

| Script | User Model | Agent Model | Context Limit |
|--------|-----------|-------------|---------------|
| `day1/int8_experiment_4b.sh` | zankich/Qwen3-32B-INT8 | Qwen/Qwen3-4B | 50000 / 50000 |
| `day2/int8_experiment_8b.sh` | zankich/Qwen3-32B-INT8 | Qwen/Qwen3-8B | 50000 / 50000 |
| `day3/int8_experiment_14b.sh` | zankich/Qwen3-32B-INT8 | JunHowie/Qwen3-14B-GPTQ-Int8 | 50000 / 50000 |
| `day4/int8_experiment_32b.sh` | zankich/Qwen3-32B-INT8 | zankich/Qwen3-32B-INT8 | 50000 / 50000 |

### Memory Requirements (A100 80GB)

| Precision | 32B Model | Memory for Weights | KV Cache Available |
|-----------|-----------|-------------------|-------------------|
| FP16 | Qwen3-32B | ~64GB | ~8GB (max-model-len ~36000) |
| INT8 GPTQ | Qwen3-32B-INT8 | ~32GB | ~40GB (max-model-len ~50000) |

### Running on SOL Cluster

Scripts are **directory-independent** - you can submit from any location:

```bash
# Submit from anywhere (recommended)
sbatch /scratch/$USER/tau-bench-project/tau-bench_CSE598/SOL_env/day2/int8_experiment_8b.sh

# Or from the script's directory
cd /scratch/$USER/tau-bench-project/tau-bench_CSE598/SOL_env/day2
sbatch int8_experiment_8b.sh

# Monitor job
squeue -u $USER

# Check logs (all logs are in the script's directory)
tail -f SOL_env/day2/logs/int8_experiment_8b_<job_id>.out
cat SOL_env/day2/logs/int8_experiment_8b_<job_id>.err
```

### Script Robustness Features

All experiment scripts include the following robustness features:

1. **Directory-independent execution**: Scripts auto-detect their location using `SCRIPT_DIR` and `REPO_ROOT` variables, so they work regardless of which directory you submit from.

2. **Automatic path resolution using BASH_SOURCE**:
   ```bash
   # BASH_SOURCE always gives the actual script location
   # Works regardless of where sbatch is called from
   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
   REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"  # or ../.. for scripts in subdirectories
   ```

3. **All paths are absolute**: Log files, results, and dependencies use `$SCRIPT_DIR` for consistent file locations.

4. **Enhanced error handling**: If vLLM fails to start, the last 50 lines of the server log are printed to help diagnose issues.

5. **Direct log output**: All stdout/stderr is redirected directly to `$SCRIPT_DIR/logs/` at job start using `exec` redirection.

6. **Diagnostic output**: Scripts print `SCRIPT_DIR`, `REPO_ROOT`, and `SUBMIT_DIR` at startup for debugging.

### Log File Locations

After a job completes, all logs are consolidated in the script's `logs/` directory:

```
SOL_env/day2/logs/
├── tau-int8-8b_<job_id>.out          # SLURM stdout
├── tau-int8-8b_<job_id>.err          # SLURM stderr
├── int8_experiment_8b_user_<job_id>.log    # vLLM user server log
├── int8_experiment_8b_agent_<job_id>.log   # vLLM agent server log
└── int8_experiment_8b_gpu_usage_<job_id>.log  # GPU monitoring log
```

### vLLM Quantization Flags

```bash
# FP16 (default) - no flag needed
vllm serve Qwen/Qwen3-32B --max-model-len 36000

# GPTQ INT8 - requires pre-quantized model
vllm serve zankich/Qwen3-32B-INT8 --quantization gptq --max-model-len 50000
```

### Pre-quantized INT8 Models

- [zankich/Qwen3-32B-INT8](https://huggingface.co/zankich/Qwen3-32B-INT8) - GPTQ INT8 W8A8
- [JunHowie/Qwen3-14B-GPTQ-Int8](https://huggingface.co/JunHowie/Qwen3-14B-GPTQ-Int8) - GPTQ INT8
- [JunHowie/Qwen3-8B-GPTQ-Int8](https://huggingface.co/JunHowie/Qwen3-8B-GPTQ-Int8) - GPTQ INT8

### Results Location

- FP16 results: `SOL_env/dayX/results/{env}/{strategy}/`
- INT8 results: `SOL_env/dayX/results_int8/{env}/{strategy}/`

### Multi-Node Experiment

The `multi_node_experiment.sh` script runs experiments across **2 separate nodes**, each with its own vLLM server:

- **Node 1**: User Simulator (Qwen/Qwen3-32B) on port 8000
- **Node 2**: Agent Model (Qwen/Qwen3-32B) on port 8000

#### Running Multi-Node Experiment

```bash
# From repo root (recommended)
cd /scratch/$USER/tau-bench-project/tau-bench_CSE598
sbatch SOL_env/multi_node_experiment.sh

# Or from SOL_env directory
cd /scratch/$USER/tau-bench-project/tau-bench_CSE598/SOL_env
sbatch multi_node_experiment.sh
```

#### Multi-Node Log Files

All logs are written **directly** to `SOL_env/logs/` from job start:

```
SOL_env/logs/
├── tau-multi-node_<job_id>.out      # SLURM stdout
├── tau-multi-node_<job_id>.err      # SLURM stderr
├── multi_node_user_<job_id>.log     # vLLM user server log
└── multi_node_agent_<job_id>.log    # vLLM agent server log
```

#### Multi-Node Results

Results saved to `SOL_env/results_multi_node/{env}/{strategy}/`

#### Multi-Node Configuration

| Parameter | Value |
|-----------|-------|
| Nodes | 2 |
| GPUs per node | 1x A100 80GB |
| User Model | Qwen/Qwen3-32B |
| Agent Model | Qwen/Qwen3-32B |
| Max context | 32768 tokens |

See `SOL_env/README_multi_node.md` for detailed documentation.

### Intel Gaudi Experiments

Scripts for running experiments on Intel Gaudi2 accelerators (HPU) on SOL cluster.

Based on **ASU RC Workshop: "Introducing the Gaudi2 and Demystifying AI Processors"**

#### Hardware Available on SOL

| Component | Details |
|-----------|---------|
| Nodes | 10 (gaudi001-gaudi010), 200+ coming |
| Accelerator | HL-225 (Gaudi2) |
| HPUs per node | 8 |
| Memory per HPU | 96GB HBM |
| CPUs per node | 152 |
| Driver | SynapseAI 1.23.0 |

#### Quick Start Options

**Option 1: Use SOL's Pre-hosted API (Easiest)**
```bash
# Get API key from https://voyager.rc.asu.edu/ → LLM Access tab
export SOL_API_KEY="your-api-key"
bash SOL_env/gaudi_api_experiment.sh
```

Available models: `qwen3-30b-a3b-instruct-2507` (131K context), `qwen3-235b-a22b-instruct-2507` (262K context)

**Option 2: Run Local vLLM on Gaudi Node**
```bash
sbatch SOL_env/gaudi_experiment.sh
```

**Option 3: Interactive Shell**
```bash
interactive -p gaudi -c 30 --mem=30G -G 3 -t 0-6
```

#### Pre-configured Resources on SOL

| Resource | Location/Name |
|----------|---------------|
| Jupyter kernels | `gaudi-pytorch`, `gaudi-pytorch-vllm` |
| Containers/guides | `/data/sse/gaudi/`, `/data/sse/gaudi/guides/` |
| Example notebooks | `/data/sse/gaudi/notebooks/` |
| OpenAI-compatible API | `https://openai.rc.asu.edu/v1` |

#### SLURM Configuration for Gaudi

```bash
#SBATCH --partition=gaudi
#SBATCH --qos=public
#SBATCH --gres=gpu:hl225:1
#SBATCH --cpus-per-task=30
#SBATCH --mem=96G
```

#### Gaudi Scripts

| Script | Purpose |
|--------|---------|
| `SOL_env/gaudi_api_experiment.sh` | Use SOL's hosted API (no local setup) |
| `SOL_env/gaudi_experiment.sh` | Run local vLLM on Gaudi node |
| `SOL_env/gaudi_setup_check.sh` | Diagnostic script (optional) |
| `SOL_env/README_gaudi.md` | Full documentation |

#### Key Differences from A100 Scripts

| Setting | A100 (CUDA) | Gaudi (HPU) |
|---------|-------------|-------------|
| Device flag | (auto-detect) | `--device hpu` |
| Block size | 16 (default) | `--block-size 128` |
| Eager mode | `--enforce-eager` | Not used (HPU Graphs preferred) |
| Partition | `general` or `public` | `gaudi` |
| GPU resource | `gpu:a100:1` | `gpu:hl225:1` |

#### Checking Gaudi Status

```bash
hl-smi   # equivalent to nvidia-smi
```

See `SOL_env/README_gaudi.md` for detailed setup instructions.

#### Gaudi Experiment Scripts Structure

All experiments are split into **independent SLURM parts** so that if one part fails, the others still succeed. Each directory contains part scripts, a submit helper, and a merge tool.

```
SOL_env/
├── generate_split_scripts.py            # Generator for all split directories
├── 4b_airline/                          # 2 parts × 3 strategies = 6 scripts
├── 4b_retail/                           # 3 parts × 3 strategies = 9 scripts
├── 8b_airline/                          # 2 parts × 3 strategies = 6 scripts
├── 8b_retail/                           # 3 parts × 3 strategies = 9 scripts
├── 14b_airline/                         # 2 parts × 3 strategies = 6 scripts
├── 14b_retail/                          # 3 parts × 3 strategies = 9 scripts
├── 32b_airline/                         # 4 parts × 3 strategies = 12 scripts (1 batch/part)
└── 32b_retail/                          # 6 parts × 3 strategies = 18 scripts (1 batch/part)
```

Each directory contains:
- `part{N}_{strategy}.sh` — self-contained SLURM job for a subset of tasks
- `submit_all.sh` — submits all parts for a given strategy
- `merge_results.py` — merges part results into a single file

#### Gaudi Experiment Configuration

| Agent Size | HPUs | User Model | Agent Model | Architecture |
|------------|------|------------|-------------|--------------|
| 4B | 3 | Qwen3-32B (TP=2, HPU 0,1) | Qwen3-4B (TP=1, HPU 2) | Dual-server |
| 8B | 3 | Qwen3-32B (TP=2, HPU 0,1) | Qwen3-8B (TP=1, HPU 2) | Dual-server |
| 14B | 3 | Qwen3-32B (TP=2, HPU 0,1) | Qwen3-14B (TP=1, HPU 2) | Dual-server |
| 32B | 8 | Qwen3-32B (TP=4, HPU 0-3) | Qwen3-32B (TP=4, HPU 4-7) | Dual-server |

#### Standardized vLLM Settings (Stability-Optimized)

All Gaudi scripts use these settings for the **User server (32B)**:

```bash
--gpu-memory-utilization 0.90 \
--max-num-seqs 4 \
--max-num-prefill-seqs 1 \
--max-model-len 40000
```

**Rationale:**
- `gpu-memory-utilization 0.90`: Prevents OOM during long runs (was 0.95)
- `max-num-seqs 4`: Only need 2× MAX_CONCURRENCY for safety (was 6-8)
- `max-num-prefill-seqs 1`: Reduces concurrent prefill memory pressure (was 2)
- `max-model-len 40000`: Qwen3 supports up to 40960 (max_position_embeddings)

#### Experiment Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| MAX_CONCURRENCY | 2 | Parallel task workers |
| NUM_TRIALS | 5 | Trials per task |
| MAX_MODEL_LEN | 40960 | Context window tokens |
| Environments | retail (115 tasks), airline (50 tasks) | |
| Strategies | act, react, tool-calling | |

#### SLURM Time Limits (Per Part)

| Experiment | Parts | Time per Part |
|------------|-------|---------------|
| 4B airline | 2 | 8:00:00 |
| 4B retail | 3 | 10:00:00 |
| 8B airline | 2 | 14:00:00 |
| 8B retail | 3 | 14:00:00 |
| 14B airline | 2 | 10:00:00 |
| 14B retail | 3 | 12:00:00 |
| 32B airline | 4 | 10:00:00 |
| 32B retail | 12 | 20:00:00 |

#### Split Parts Configuration

All experiments are split into independent SLURM parts. Parts run in parallel as separate SLURM jobs.

For 4B/8B/14B, each part runs 2 batches with server restart between them. For 32B, each part runs **1 batch only** — the Habana driver cannot reacquire HPU devices after vLLM shutdown within the same SLURM job (`synStatus=8 [Device not found]`).

**Airline 4B/8B/14B (50 tasks) → 2 parts, 2 batches each:**

| Part | Tasks | Batches |
|------|-------|---------|
| Part 1 | 0-24 | `("0 12" "13 24")` |
| Part 2 | 25-49 | `("25 37" "38 49")` |

**Airline 32B (50 tasks) → 4 parts, 1 batch each:**

| Part | Tasks |
|------|-------|
| Part 1 | 0-12 |
| Part 2 | 13-24 |
| Part 3 | 25-37 |
| Part 4 | 38-49 |

**Retail 4B/8B/14B (115 tasks) → 3 parts, 2 batches each:**

| Part | Tasks | Batches |
|------|-------|---------|
| Part 1 | 0-39 | `("0 19" "20 39")` |
| Part 2 | 40-79 | `("40 59" "60 79")` |
| Part 3 | 80-114 | `("80 99" "100 114")` |

**Retail 32B (115 tasks) → 6 parts, 1 batch each:**

| Part | Tasks |
|------|-------|
| Part 1 | 0-19 |
| Part 2 | 20-39 |
| Part 3 | 40-59 |
| Part 4 | 60-79 |
| Part 5 | 80-99 |
| Part 6 | 100-114 |

#### Running Split Experiments

```bash
# 1. Submit all parts for a strategy (runs 2 or 3 SLURM jobs in parallel)
./SOL_env/8b_retail/submit_all.sh act

# 2. Monitor
squeue -u $USER

# 3. After all parts complete, merge results
python SOL_env/8b_retail/merge_results.py --strategy act --dry-run   # preview first
python SOL_env/8b_retail/merge_results.py --strategy act              # write merged file

# 4. If a part failed, just resubmit it
sbatch SOL_env/8b_retail/part2_act.sh

# 5. Re-merge (deduplication handles overlapping results automatically)
python SOL_env/8b_retail/merge_results.py --strategy act
```

#### How Part Scripts Work

Each `part{N}_{strategy}.sh` is a self-contained SLURM job that:
1. Requests HPUs, sets up environment
2. Loops through its assigned batches (2 per part for 4B/8B/14B, 1 for 32B):
   - Starts User (32B) and Agent vLLM servers
   - Waits for both servers to be healthy
   - Runs `run.py` for that batch's task range
   - Saves results with `_part{N}_batch{M}_job{SLURM_JOB_ID}.json` naming
   - Kills servers, waits for HPU memory release
3. Reports success/failure count
4. Fail-fast: aborts if 2 consecutive batches fail (HPU devices likely stuck)

#### How merge_results.py Works

The merge tool scans `results_gaudi/{env}/{strategy}/` for files matching `*_part*_batch*_job*.json` and combines them into a single result file. Protections:

1. **Job ID filtering** (`--job-ids` / `--exclude-jobs`): Whitelist/blacklist SLURM job IDs
2. **Corrupted file detection**: Skips truncated JSON with warnings
3. **Empty file skipping**: Skips 0-result files
4. **Deduplication**: Keeps latest result for each `(task_id, trial)` pair

```bash
python merge_results.py --strategy react --dry-run                    # preview
python merge_results.py --strategy react --job-ids 46800001 46800002  # whitelist
python merge_results.py --strategy react --exclude-jobs 46703271      # blacklist
python merge_results.py --strategy react                              # merge all
```

#### Regenerating Scripts

If you need to modify the template or configuration, edit `SOL_env/generate_split_scripts.py` and re-run:

```bash
python SOL_env/generate_split_scripts.py          # regenerate all 8 directories
python SOL_env/generate_split_scripts.py --dry-run # preview only
```

#### Team Assignments

| Model | Assigned To |
|-------|------------|
| 4B | Harish |
| 8B | Sai |
| 14B | Smit |
| 32B | Vardaan / hehernan |

#### Log File Locations (Gaudi)

After a job completes, logs are in the directory's `logs/` folder:

```
SOL_env/8b_retail/logs/
├── tau-gaudi-8b-retail-act-p1_<job_id>.out     # SLURM stdout
├── tau-gaudi-8b-retail-act-p1_<job_id>.err     # SLURM stderr
├── gaudi_vllm_user_32b_<job_id>_batch1.log     # vLLM user server log
└── gaudi_vllm_agent_8b_<job_id>_batch1.log     # vLLM agent server log
```

#### Results Location (Gaudi)

Results are saved to `SOL_env/{size}_{env}/results_gaudi/{env}/{strategy}/`
