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
2. **SLURM Account**: `grp_jzou22`
3. **tau-bench Environment**: Created automatically if missing

## Step-by-Step: Running an Experiment

This walks through the full workflow using 32B retail as an example. Replace `32b_retail` with your directory (e.g., `8b_airline`, `14b_retail`).

### Step 1: Submit Jobs

```bash
# Always start from repo root
cd /scratch/$USER/tau-bench-project/tau-bench_CSE598

# Submit all parts for a strategy (runs independent SLURM jobs in parallel)
./SOL_env/32b_retail/submit_all.sh act
./SOL_env/32b_retail/submit_all.sh react
./SOL_env/32b_retail/submit_all.sh tool-calling

# Output for each (32B retail = 6 parts):
#   Part 1 (tasks 0-19):   Job 46900001
#   Part 2 (tasks 20-39):  Job 46900002
#   ...
#   Part 6 (tasks 100-114): Job 46900006
```

Or submit a single part individually:
```bash
sbatch SOL_env/32b_retail/part1_act.sh
```

### Step 2: Monitor Running Jobs

```bash
# Quick status of all your jobs
squeue -u $USER

# Detailed status with time elapsed and node info
squeue -u $USER -o "%.10i %.30j %.8T %.10M %.6D %R"

# Detailed info for a specific job (state, node, time used, etc.)
scontrol show job <JOB_ID>

# Watch live output of a running job
tail -f SOL_env/32b_retail/logs/tau-gaudi-32b-retail-act-p1_<JOB_ID>.out

# Watch vLLM server logs if debugging server issues
tail -f SOL_env/32b_retail/logs/gaudi_vllm_user_32b_<JOB_ID>_batch1.log
tail -f SOL_env/32b_retail/logs/gaudi_vllm_agent_32b_<JOB_ID>_batch1.log

# Cancel a job
scancel <JOB_ID>

# Cancel all your jobs
scancel -u $USER
```

### Step 3: Check if Jobs Succeeded

After a job finishes, verify it completed successfully:

```bash
# Check exit status (look for COMPLETED vs FAILED/TIMEOUT)
sacct -j <JOB_ID> --format=JobID,JobName,State,ExitCode,Elapsed

# Check the log tail — successful part ends with:
#   "Part 1 (tasks 0-39) complete: 2/2 batches succeeded."
# Failed part ends with:
#   "FAILED: No batches completed successfully."
tail -20 SOL_env/32b_retail/logs/tau-gaudi-32b-retail-act-p1_<JOB_ID>.out

# Count result files (each successful batch produces one file)
ls SOL_env/32b_retail/results_gaudi/retail/act/*_part*_job*.json
```

### Step 4: Merge Results

The merge tool is the best way to see overall completeness:

```bash
# ALWAYS dry-run first to see what you have
python SOL_env/32b_retail/merge_results.py --strategy act --dry-run
```

Successful output looks like:
```
Parts found: [1, 2, 3]           ← all 3 present = good
Task coverage: 115/115            ← full coverage = good
Total results: 575                ← 115 × 5 trials = complete
[DRY RUN] Would merge the above results. No output written.
```

If a part is missing:
```
Parts found: [1, 3]
WARNING: Missing parts: [2]      ← Part 2 failed — resubmit it
Task coverage: 75/115
```

Once satisfied, write the merged file:
```bash
python SOL_env/32b_retail/merge_results.py --strategy act
```

### Step 5: Handle Failures

If a part failed, just resubmit that one part:

```bash
# Only part 2 failed — resubmit it
sbatch SOL_env/32b_retail/part2_act.sh

# After it completes, re-merge (deduplication handles overlaps automatically)
python SOL_env/32b_retail/merge_results.py --strategy act --dry-run
python SOL_env/32b_retail/merge_results.py --strategy act
```

If a resubmitted job also produced bad results alongside the new good ones:
```bash
# Exclude the failed job's results by ID
python SOL_env/32b_retail/merge_results.py --strategy act --exclude-jobs <FAILED_JOB_ID>

# Or whitelist only the good jobs
python SOL_env/32b_retail/merge_results.py --strategy act --job-ids <GOOD_JOB_1> <GOOD_JOB_2> <GOOD_JOB_3>
```

### Quick Reference: All Commands for 32B

```bash
# ── Airline (4 parts × 3 strategies = 12 jobs) ──
./SOL_env/32b_airline/submit_all.sh act
./SOL_env/32b_airline/submit_all.sh react
./SOL_env/32b_airline/submit_all.sh tool-calling

# After all airline jobs complete:
python SOL_env/32b_airline/merge_results.py --strategy act --dry-run
python SOL_env/32b_airline/merge_results.py --strategy act
python SOL_env/32b_airline/merge_results.py --strategy react --dry-run
python SOL_env/32b_airline/merge_results.py --strategy react
python SOL_env/32b_airline/merge_results.py --strategy tool-calling --dry-run
python SOL_env/32b_airline/merge_results.py --strategy tool-calling

# ── Retail (6 parts × 3 strategies = 18 jobs) ──
./SOL_env/32b_retail/submit_all.sh act
./SOL_env/32b_retail/submit_all.sh react
./SOL_env/32b_retail/submit_all.sh tool-calling

# After all retail jobs complete:
python SOL_env/32b_retail/merge_results.py --strategy act --dry-run
python SOL_env/32b_retail/merge_results.py --strategy act
python SOL_env/32b_retail/merge_results.py --strategy react --dry-run
python SOL_env/32b_retail/merge_results.py --strategy react
python SOL_env/32b_retail/merge_results.py --strategy tool-calling --dry-run
python SOL_env/32b_retail/merge_results.py --strategy tool-calling
```

## Directory Structure

All experiments are split into **independent SLURM parts** so that if one part fails, the others still succeed. HPU memory leaks cause vLLM to crash after a few batches, so splitting prevents losing an entire job.

```
SOL_env/
├── generate_split_scripts.py    # Generator for all split directories
├── 4b_airline/                  # 2 parts × 3 strategies = 6 scripts
├── 4b_retail/                   # 3 parts × 3 strategies = 9 scripts
├── 8b_airline/                  # 6 scripts
├── 8b_retail/                   # 9 scripts
├── 14b_airline/                 # 6 scripts
├── 14b_retail/                  # 9 scripts
├── 32b_airline/                 # 4 parts × 3 strategies = 12 scripts (1 batch/part)
└── 32b_retail/                  # 6 parts × 3 strategies = 18 scripts (1 batch/part)
```

Each directory contains:
- `part{N}_{strategy}.sh` — self-contained SLURM job for a subset of tasks
- `submit_all.sh` — submits all parts for a given strategy
- `merge_results.py` — merges part results into a single file

### Script Naming Convention

```
SOL_env/{MODEL}_{ENV}/part{PART}_{STRATEGY}.sh
```

Examples:
- `SOL_env/8b_retail/part1_act.sh` — 8B agent, retail env, act strategy, part 1
- `SOL_env/32b_airline/part2_tool-calling.sh` — 32B agent, airline env, tool-calling, part 2

## Available Scripts

### 4B Model (assigned to Harish)

| Environment | Strategy | Parts | Scripts |
|-------------|----------|-------|---------|
| airline | act, react, tool-calling | 2 each | `4b_airline/part{1,2}_{strategy}.sh` |
| retail | act, react, tool-calling | 3 each | `4b_retail/part{1,2,3}_{strategy}.sh` |

### 8B Model (assigned to Sai)

| Environment | Strategy | Parts | Scripts |
|-------------|----------|-------|---------|
| airline | act, react, tool-calling | 2 each | `8b_airline/part{1,2}_{strategy}.sh` |
| retail | act, react, tool-calling | 3 each | `8b_retail/part{1,2,3}_{strategy}.sh` |

### 14B Model (assigned to Smit)

| Environment | Strategy | Parts | Scripts |
|-------------|----------|-------|---------|
| airline | act, react, tool-calling | 2 each | `14b_airline/part{1,2}_{strategy}.sh` |
| retail | act, react, tool-calling | 3 each | `14b_retail/part{1,2,3}_{strategy}.sh` |

### 32B Model (assigned to Vardaan / hehernan)

| Environment | Strategy | Parts | Scripts |
|-------------|----------|-------|---------|
| airline | act, react, tool-calling | 4 each | `32b_airline/part{1..4}_{strategy}.sh` |
| retail | act, react, tool-calling | 6 each | `32b_retail/part{1..6}_{strategy}.sh` |

**Note**: 32B uses 1 batch per part because the Habana driver cannot reacquire HPU devices after vLLM shutdown within the same SLURM job (`synStatus=8 [Device not found]`).

**Total: 72 part scripts** (12 for 4B/8B/14B each + 30 for 32B)

## Split Configuration

### Why Split?

A single SLURM job runs all batches sequentially. If batch 3 of 6 crashes due to HPU memory leaks, batches 4-6 never run. With splits, each part is independent — if Part 2 fails, Parts 1 and 3 still succeed.

For 32B specifically, the Habana driver cannot reacquire HPU devices after vLLM shutdown within the same SLURM job (`synStatus=8 [Device not found]`), so 32B parts run exactly **1 batch each**.

### Airline 4B/8B/14B (50 tasks) → 2 parts, 2 batches each

| Part | Tasks | Batches | Expected Results |
|------|-------|---------|-----------------|
| Part 1 | 0-24 | `("0 12" "13 24")` | 125 (25 tasks × 5 trials) |
| Part 2 | 25-49 | `("25 37" "38 49")` | 125 (25 tasks × 5 trials) |

### Airline 32B (50 tasks) → 4 parts, 1 batch each

| Part | Tasks | Expected Results |
|------|-------|-----------------|
| Part 1 | 0-12 | 65 (13 tasks × 5 trials) |
| Part 2 | 13-24 | 60 (12 tasks × 5 trials) |
| Part 3 | 25-37 | 65 (13 tasks × 5 trials) |
| Part 4 | 38-49 | 60 (12 tasks × 5 trials) |

### Retail 4B/8B/14B (115 tasks) → 3 parts, 2 batches each

| Part | Tasks | Batches | Expected Results |
|------|-------|---------|-----------------|
| Part 1 | 0-39 | `("0 19" "20 39")` | 200 (40 tasks × 5 trials) |
| Part 2 | 40-79 | `("40 59" "60 79")` | 200 (40 tasks × 5 trials) |
| Part 3 | 80-114 | `("80 99" "100 114")` | 175 (35 tasks × 5 trials) |

### Retail 32B (115 tasks) → 6 parts, 1 batch each

| Part | Tasks | Expected Results |
|------|-------|-----------------|
| Part 1 | 0-19 | 100 (20 tasks × 5 trials) |
| Part 2 | 20-39 | 100 (20 tasks × 5 trials) |
| Part 3 | 40-59 | 100 (20 tasks × 5 trials) |
| Part 4 | 60-79 | 100 (20 tasks × 5 trials) |
| Part 5 | 80-99 | 100 (20 tasks × 5 trials) |
| Part 6 | 100-114 | 75 (15 tasks × 5 trials) |

## Configuration

### Common Settings (All Scripts)

| Setting | Value |
|---------|-------|
| MAX_MODEL_LEN | 40960 |
| MAX_CONCURRENCY | 2 |
| NUM_TRIALS | 5 |
| Partition | gaudi |
| QOS | public |

### SLURM Time Limits (Per Part)

| Experiment | Parts | Time per Part |
|------------|-------|---------------|
| 4B airline | 2 | 8:00:00 |
| 4B retail | 3 | 10:00:00 |
| 8B airline | 2 | 8:00:00 |
| 8B retail | 3 | 10:00:00 |
| 14B airline | 2 | 10:00:00 |
| 14B retail | 3 | 12:00:00 |
| 32B airline | 4 | 10:00:00 |
| 32B retail | 6 | 14:00:00 |

### Dual-Server Architecture

All experiments use a **consistent 32B user simulator** (Qwen3-32B) to ensure fair comparison across agent model sizes. Each part script runs two separate vLLM servers:

**4B/8B/14B:**

| Server | Model | HPUs | Tensor Parallel | Purpose |
|--------|-------|------|-----------------|---------|
| User Server | Qwen3-32B | 0,1 | 2 | User simulator |
| Agent Server | Qwen3-{4B,8B,14B} | 2 | 1 | Agent model |

**32B:**

| Server | Model | HPUs | Tensor Parallel | Purpose |
|--------|-------|------|-----------------|---------|
| User Server | Qwen3-32B | 0,1,2,3 | 4 | User simulator |
| Agent Server | Qwen3-32B | 4,5,6,7 | 4 | Agent model |

### Resource Allocation by Agent Model

| Agent Model | Total HPUs | CPUs | Memory | SLURM gres |
|-------------|------------|------|--------|------------|
| Qwen3-4B | 3 | 24 | 160G | `gpu:hl225:3` |
| Qwen3-8B | 3 | 24 | 160G | `gpu:hl225:3` |
| Qwen3-14B | 3 | 24 | 160G | `gpu:hl225:3` |
| Qwen3-32B | 8 | 60 | 384G | `gpu:hl225:8` |

### Server Configuration Details

| Server | MAX_NUM_SEQS | MAX_PREFILL | GPU Util | Notes |
|--------|--------------|-------------|----------|-------|
| User (32B, all sizes) | 2 | 1 | 0.85 | Stability-optimized |
| Agent (4B/8B/14B) | 16 | 8 | 0.85 | Single HPU, more headroom |
| Agent (32B) | 2 | 1 | 0.85 | Same config as User |

### Stability Settings Rationale

| Setting | Value | Reason |
|---------|-------|--------|
| `--gpu-memory-utilization` | 0.85 | Prevents OOM during extended runs |
| `--max-num-seqs` | 2 (user/32B agent) | Only need MAX_CONCURRENCY headroom |
| `--max-num-prefill-seqs` | 1 | Reduces concurrent prefill memory pressure |
| `--max-model-len` | 40960 | Qwen3's max_position_embeddings |
| `--swap-space` | 16 | Extra swap for memory overflow |

### Port Assignments

Ports are dynamically assigned based on SLURM job ID to avoid conflicts:

```bash
USER_PORT=$((10000 + (SLURM_JOB_ID % 10000)))
AGENT_PORT=$((20000 + (SLURM_JOB_ID % 10000)))
```

## How Part Scripts Work

Each `part{N}_{strategy}.sh` is a self-contained SLURM job that:

1. **Setup**: Requests HPUs, sets up conda environment, configures Gaudi env vars
2. **Batch Loop** (2 batches per part for 4B/8B/14B, 1 batch for 32B):
   - Starts User (32B) and Agent vLLM servers via Apptainer
   - Waits for both servers to be healthy (up to 30 min)
   - Runs `python run.py` for that batch's task range
   - Saves results with `_part{N}_batch{M}_job{SLURM_JOB_ID}.json` naming
   - Kills servers, waits for HPU memory release (poll 15s × 12 + sleep 30)
3. **Fail-fast**: Aborts if 2 consecutive batches fail (HPU devices likely stuck)
4. **Reports** success/failure count

### Output File Naming

```
{strategy}-Qwen3-{size}-0.0_range_{start}-{end}_..._part{N}_batch{M}_job{JOB_ID}.json
```

The `_part{N}_batch{M}_job{JOB_ID}` suffix enables the merge tool to identify and combine files.

## How merge_results.py Works

The merge tool scans `results_gaudi/{env}/{strategy}/` for files matching `*_part*_batch*_job*.json` and combines them. Protections:

1. **Job ID filtering** (`--job-ids` / `--exclude-jobs`): Whitelist/blacklist specific SLURM job IDs
2. **Corrupted file detection**: Skips truncated JSON with warnings
3. **Empty file skipping**: Skips 0-result files
4. **Deduplication**: Keeps latest result for each `(task_id, trial)` pair

```bash
# Preview what will be merged (always do this first)
python merge_results.py --strategy react --dry-run

# Merge only results from specific successful jobs
python merge_results.py --strategy react --job-ids 46800001 46800002

# Merge everything except a known-failed job
python merge_results.py --strategy react --exclude-jobs 46703271

# Default: merge all found files (with dedup + corruption handling)
python merge_results.py --strategy react
```

## Regenerating Scripts

All split scripts are generated by `generate_split_scripts.py`. If you need to modify the template or configuration:

```bash
# Edit the generator
vi SOL_env/generate_split_scripts.py

# Preview what would be created
python SOL_env/generate_split_scripts.py --dry-run

# Regenerate all 8 directories
python SOL_env/generate_split_scripts.py
```

## Results Location

Results are saved to:
```
SOL_env/{MODEL}_{ENV}/results_gaudi/{ENV}/{STRATEGY}/
```

Example:
```
SOL_env/8b_retail/results_gaudi/retail/act/
SOL_env/32b_airline/results_gaudi/airline/react/
```

## Log Files

Logs are saved to:
```
SOL_env/{MODEL}_{ENV}/logs/
```

Each part job creates:
- `tau-gaudi-{MODEL}-{ENV}-{STRATEGY}-p{PART}_{JOB_ID}.out` — Main output
- `tau-gaudi-{MODEL}-{ENV}-{STRATEGY}-p{PART}_{JOB_ID}.err` — Error log
- `gaudi_vllm_user_32b_{JOB_ID}_batch{N}.log` — User model vLLM server log
- `gaudi_vllm_agent_{MODEL}_{JOB_ID}_batch{N}.log` — Agent model vLLM server log

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

## Script Robustness Features

All part scripts include these robustness features:

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
```

### 3. Fail-Fast on Consecutive Failures

If 2 consecutive batches fail, the part aborts rather than wasting time on batches that will also fail:

```bash
CONSECUTIVE_FAILURES=0
MAX_CONSECUTIVE_FAILURES=2
# ... if batch fails, increment counter; if it succeeds, reset to 0
```

### 4. Stale Editable Install Fix

Scripts uninstall before reinstalling tau-bench:

```bash
pip uninstall tau_bench -y 2>/dev/null || true
pip install -e .
```

### Issue Summary

| Issue | Scope | Cause | Fix |
|-------|-------|-------|-----|
| Port conflicts | Node-level | Multiple jobs, same ports | `--exclusive` |
| HPU conflicts | Node-level | Hardcoded `HABANA_VISIBLE_DEVICES` | `--exclusive` |
| Process kill race | Node-level | `pkill -f` kills wrong processes | PID-based cleanup |
| Stale `.pth` files | User-level | Moved repo or corrupted install | `pip uninstall` first |
| HPU memory leaks | Job-level | vLLM crashes after few batches | Split into independent parts |

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
cat SOL_env/8b_retail/logs/gaudi_vllm_user_32b_JOB_ID_batch1.log

# Agent model log
cat SOL_env/8b_retail/logs/gaudi_vllm_agent_8b_JOB_ID_batch1.log
```

### A Part Failed — What Do I Do?
1. Check which part failed:
   ```bash
   sacct -j <JOB_ID> --format=JobID,JobName,State,ExitCode,Elapsed
   tail -20 SOL_env/8b_retail/logs/tau-gaudi-8b-retail-act-p2_<JOB_ID>.out
   ```
2. Resubmit just that part:
   ```bash
   sbatch SOL_env/8b_retail/part2_act.sh
   ```
3. After it completes, dry-run merge to verify, then merge:
   ```bash
   python SOL_env/8b_retail/merge_results.py --strategy act --dry-run
   python SOL_env/8b_retail/merge_results.py --strategy act
   ```
4. Deduplication automatically keeps the latest results

### How Do I Know if Everything Succeeded?
Run the dry-run merge — it tells you exactly what's complete and what's missing:
```bash
python SOL_env/8b_retail/merge_results.py --strategy act --dry-run
```
- `Parts found: [1, 2, 3]` — all parts present
- `Task coverage: 115/115` — all tasks covered
- `Total results: 575` — all 115 tasks × 5 trials
- `WARNING: Missing parts: [2]` — Part 2 failed, needs resubmission

### Context Window Exceeded
If you see `ContextWindowExceededError`, increase `MAX_MODEL_LEN` in the script.

### Model Download Slow
First run downloads weights to `/scratch/$USER/hf_cache`. Subsequent runs use cache.

### "Cannot send a request, as the client has been closed" Error
This error means the vLLM server crashed during the experiment. The split architecture limits damage — only the current batch's results are lost. Common causes:
1. **Memory pressure**: Reduce `--gpu-memory-utilization` or `--max-num-seqs`
2. **Long-running fragmentation**: Server memory fragments over hundreds of requests
3. **Context too long**: Some tasks exceed `--max-model-len`

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
| `generate_split_scripts.py` | Generate/regenerate all split directories |
| `gaudi_api_experiment.sh` | Use SOL's hosted API |
| `gaudi_setup_check.sh` | Diagnostic to verify environment |

## References

- [Habana Documentation](https://docs.habana.ai/en/latest/index.html)
- [vLLM Gaudi Plugin](https://github.com/vllm-project/vllm-gaudi)
- SOL Voyager: https://voyager.rc.asu.edu/
