# Day 1 Experiments Workflow

## Step 1: Start vLLM Servers (Separate Jobs)
Submit the SLURM jobs to start the User Simulator (Qwen2.5-32B) on port 8000 and the Agent (Qwen-4B) on port 8001.
```bash
sbatch vllm_qwen3_32b.sh      # User on port 8000
sbatch test_vllm_qwen4b.sh    # Agent on port 8001

# Monitor status
watch -n 10 'squeue -u $USER'
```

## Step 2: Get Node Names
Wait for the jobs to start (status changes from `PD` to `R`), then run this script to find out which nodes they were assigned to.
```bash
chmod +x get_server_nodes.sh
./get_server_nodes.sh
```

*Example Output:*
```text
USER server is on node: sg032 (Job 12345)
AGENT_4B server is on node: sg014 (Job 12346)

To export these variables, run:
export USER_NODE=sg032
export AGENT_NODE_4B=sg014
```

## Step 3: Set Environment Variables
Copy and paste the export commands from the previous step output into your terminal.
```bash
export USER_NODE=<your_user_node>
export AGENT_NODE_4B=<your_agent_node>

# Verify they're set
echo "User: $USER_NODE, Agent: $AGENT_NODE_4B"
```

## Step 4: Submit Experiment Job
Once the variables are set, submit the experiment script.
```bash
sbatch day1_experiments.sh

# Monitor experiments
tail -f logs/day1_exp_*.out
```

## Step 5: View Results
There are two places to look for output:

1.  **Job Logs (Did it run?)**
    Check the `logs/` folder to see the direct output of the script (errors, progress prints).
```bash
    cat logs/day1_exp_*.out
    # or monitor in real-time
    tail -f logs/day1_exp_*.out
```

2.  **Experiment Data (Scores & Trajectories)**
    The actual data is saved in the `results/` folder, organized by environment and strategy.
```bash
    # View structure
    ls -R results/
    
    # Example structure:
    # results/retail/act/
    # results/retail/tool-calling/
    # results/retail/react/
    # results/airline/act/
    # results/airline/tool-calling/
    # results/airline/react/
```

## Step 6: Quick Results Check
After experiments complete, check key metrics:
```bash
# Count completed tasks
find results/ -name "*.json" | wc -l

# View a sample result
ls results/retail/tool-calling/
cat results/retail/tool-calling/<filename>.json | head -30

# Check for errors
grep -i "error\|exception" logs/day1_exp_*.out
```

---

## Troubleshooting

### Jobs Stay in PD Status
```bash
# Check your GPU allocation
squeue -u $USER

# If stuck for >30 min, check queue
squeue -p public | grep gpu
```

### Servers Don't Start
```bash
# Check server logs for errors
tail -50 vllm_qwen_32b_*.err
tail -50 vllm_4b_*.err

# Common issues:
# - Model download failed (network issue)
# - CUDA out of memory (wrong GPU count)
# - Port already in use (previous job still running)
```

### Experiments Fail
```bash
# Check experiment log
cat logs/day1_exp_*.err

# Verify servers are responding
curl http://$USER_NODE:8000/health
curl http://$AGENT_NODE_4B:8001/health

# Test connectivity
python -c "from openai import OpenAI; c = OpenAI(base_url='http://$AGENT_NODE_4B:8001/v1', api_key='dummy'); print(c.models.list())"
```

### Environment Variables Not Set
```bash
# Check if they exist
echo "USER_NODE=$USER_NODE"
echo "AGENT_NODE_4B=$AGENT_NODE_4B"

# If empty, re-run get_server_nodes.sh and export again
./get_server_nodes.sh
```

---

## Expected Timeline
- **Step 1 → Step 2:** 10-20 minutes (GPU allocation + model download)
- **Step 2 → Step 4:** 2-3 minutes (get nodes, set vars, submit)
- **Step 4 → Step 5:** 30-60 minutes (run 18 experiments)
- **Total:** ~1-1.5 hours from start to finish

---

## Cleanup (After Experiments Complete)
```bash
# Cancel running servers (to free GPU hours)
scancel 46126816 46126817

# Or cancel all your jobs
scancel -u $USER

# Archive results
tar -czf day1_results_$(date +%Y%m%d).tar.gz results/ logs/
```
