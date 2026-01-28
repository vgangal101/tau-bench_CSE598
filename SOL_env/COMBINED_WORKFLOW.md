# τ-Bench SOL Environment Experiment Workflow

This workflow runs experiments across **4 days** with different Qwen3 agent model sizes, all using the Qwen3-32B user simulator.

## Experiment Schedule

| Day | Agent Model | User Model | Script |
|-----|-------------|------------|--------|
| Day 1 | Qwen3-4B | Qwen3-32B | `day1/combined_experiment_4b.sh` |
| Day 2 | Qwen3-8B | Qwen3-32B | `day2/combined_experiment_8b.sh` |
| Day 3 | Qwen3-14B | Qwen3-32B | `day3/combined_experiment_14b.sh` |
| Day 4 | Qwen3-32B | Qwen3-32B | `day4/combined_experiment_32b.sh` |

---

## Quick Start

### Submit a Day's Experiments
```bash
# From the SOL_env directory
cd SOL_env/day1  # or day2, day3, day4
sbatch combined_experiment_4b.sh  # adjust filename per day

# Monitor job status
watch -n 10 'squeue -u $USER'
```

### Monitor Progress
```bash
# Watch output (replace JOBID)
tail -f logs/combined_experiment_*_JOBID.out

# Check for errors
tail -f logs/combined_experiment_*_JOBID.err
```

---

## What Each Job Does

1. **Allocates 2 A100 GPUs** on a single node
2. **Starts vLLM servers**:
   - GPU 0: Qwen3-32B (User Simulator) on port 8000
   - GPU 1: Agent model (4B/8B/14B/32B) on port 8001
3. **Waits for health checks** (up to 10 minutes)
4. **Runs 6 experiments**: 2 envs (retail, airline) × 3 strategies (tool-calling, act, react)
5. **Cleans up**: Servers stopped automatically

---

## Resource Requirements

| Day | Memory | GPUs | Time |
|-----|--------|------|------|
| Day 1 (4B) | 96GB | 2x A100 | 8 hours |
| Day 2 (8B) | 96GB | 2x A100 | 2 hours |
| Day 3 (14B) | 128GB | 2x A100 | 2 hours |
| Day 4 (32B) | 128GB | 2x A100 | 2 hours |

---

## Results Location

Results are saved per day:
```bash
SOL_env/day1/results/{retail,airline}/{tool-calling,act,react}/
SOL_env/day2/results/{retail,airline}/{tool-calling,act,react}/
SOL_env/day3/results/{retail,airline}/{tool-calling,act,react}/
SOL_env/day4/results/{retail,airline}/{tool-calling,act,react}/
```

---

## Troubleshooting

### Server Fails to Start
```bash
# Check server logs (replace dayX and JOBID)
tail -100 SOL_env/dayX/logs/combined_experiment_*_user_JOBID.log
tail -100 SOL_env/dayX/logs/combined_experiment_*_agent_JOBID.log
```

### Cancel a Job
```bash
scancel JOBID
# Or cancel all your jobs
scancel -u $USER
```

---

## Customization

Edit the experiment script to modify:
- **Task range**: `--end-index 3` (currently runs 3 tasks per env/strategy)
- **Concurrency**: `--max-concurrency 5`
- **Time limit**: `#SBATCH --time=...`
