# Combined Experiment Workflow (Single Job)

This is a simplified workflow that runs everything in a single SLURM job - both vLLM servers and all experiments.

## Advantages
- **Simple**: Just one `sbatch` command
- **No manual steps**: No need to get node names or set environment variables
- **Automatic cleanup**: Servers are stopped when experiments finish
- **Guaranteed locality**: Everything runs on the same node

## Disadvantages
- **Higher queue wait**: Requesting 2 GPUs may take longer
- **Less flexible**: Can't reuse servers for multiple experiment runs
- **All-or-nothing**: If one component fails, you need to restart everything

---

## Quick Start

### Step 1: Submit the Job
```bash
cd SOL_env/day1
sbatch combined_experiment.sh

# Monitor status
watch -n 10 'squeue -u $USER'
```

### Step 2: Monitor Progress
```bash
# Watch the main output in real-time
tail -f logs/combined_exp_*.out

# Check for errors
tail -f logs/combined_exp_*.err
```

### Step 3: Check Server Logs (if needed)
If experiments fail, check the individual server logs:
```bash
# User simulator logs
tail -f logs/user_server_*.log

# Agent logs
tail -f logs/agent_server_*.log
```

### Step 4: View Results
After completion, results are in the same location as the original workflow:
```bash
# View structure
ls -R SOL_env/day1/results/

# Count completed tasks
find SOL_env/day1/results/ -name "*.json" | wc -l

# View a sample result
cat SOL_env/day1/results/retail/tool-calling/*.json | head -30
```

---

## What Happens Inside

1. **Job starts** → Allocates 2 A100 GPUs on a single node
2. **Servers start** → Both vLLM servers launch in background:
   - GPU 0: Qwen2.5-32B (User) on port 8000
   - GPU 1: Qwen3-4B (Agent) on port 8001
3. **Health checks** → Script waits up to 10 minutes for servers to be ready
4. **Experiments run** → All 6 experiments (2 envs × 3 strategies) execute sequentially
5. **Cleanup** → Servers are automatically stopped when done

---

## Resource Requirements

- **GPUs**: 2x A100
- **Memory**: 96GB
- **CPUs**: 16
- **Time**: 8 hours (generous buffer)
- **Actual runtime**: ~1.5-2 hours typically

---

## Troubleshooting

### Job Stays in Queue (PD status)
```bash
# Check why it's pending
squeue -u $USER --start

# Check GPU availability
squeue -p public | grep gpu

# If waiting too long (>1 hour), consider using the original 3-job workflow instead
```

### Server Fails to Start
Check the server logs:
```bash
# Find your job ID
squeue -u $USER

# Check server logs (replace JOBID)
tail -100 logs/user_server_JOBID.log
tail -100 logs/agent_server_JOBID.log

# Common issues:
# - Model download timeout (network issues)
# - CUDA out of memory (check GPU allocation)
# - Port conflict (shouldn't happen on fresh node)
```

### Experiments Fail
```bash
# Check main experiment log
tail -100 logs/combined_exp_*.err

# Verify servers are responding (run this on the compute node if job is still running)
# Note: You'll need to ssh to the node first
curl http://localhost:8000/health
curl http://localhost:8001/health
```

### Need to Cancel
```bash
# Cancel the job (this will trigger cleanup automatically)
scancel JOBID

# Or cancel all your jobs
scancel -u $USER
```

---

## Customization

Edit [combined_experiment.sh](combined_experiment.sh) to modify:

- **Task range**: Change `--end-index 3` to run more/fewer tasks
- **Concurrency**: Adjust `--max-concurrency 5`
- **Time limit**: Modify `#SBATCH --time=08:00:00`
- **Memory**: Adjust `#SBATCH --mem=96G` if needed
- **Environments/Strategies**: Modify the loops at lines 162-185

---

## Comparison with Original Workflow

| Aspect | Combined (This) | Original (3 Jobs) |
|--------|----------------|-------------------|
| Setup complexity | Low (1 command) | Medium (3 commands + exports) |
| Queue wait | Longer (2 GPUs) | Shorter (1 GPU each) |
| Server reuse | No | Yes |
| Debugging | Harder (all in one) | Easier (separate logs) |
| Resource efficiency | Lower | Higher |
| Best for | Quick tests | Production runs |

**Recommendation**: Use combined workflow for initial testing and quick iterations. Use original workflow for longer production runs or when GPU queue is busy.
