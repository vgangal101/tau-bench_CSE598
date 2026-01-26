# How to Log Back Into SOL and Restart Jobs

## 1. SSH Back In
Replace `your_asurite` with your actual username.
```bash
ssh your_asurite@sol.asu.edu
```

## 2. Navigate to Project
```bash
cd /scratch/$USER/tau-bench-project
```

## 3. Activate Environment
```bash
module load mamba/latest
source activate tau-bench
```

## 4. Check Status
```bash
# See what jobs are running
squeue -u $USER

# Check recent output logs
ls -ltr *.out
```

## 5. Submit Jobs
Submit the user simulator and the agents.

```bash
# Submit User Simulator (GPT-OSS-20B)
sbatch vllm_gpt_oss_20b.sh

# Submit Agents
sbatch test_vllm_qwen4b.sh
sbatch test_vllm_qwen8b.sh
sbatch test_vllm_qwen14b.sh
sbatch test_vllm_qwen32b.sh
```

## 6. Monitor Logs
Watch for "Application startup complete" to know when the server is ready.
```bash
# Replace <job_id> with the actual job ID
tail -f vllm_gpt_oss_20b_*.out
```
*Press `Ctrl+C` to stop watching the log.*

## 7. Run Experiments (Day 1)
**CRITICAL:** You must find the NODE NAMES where your models are running before you can start experiments.

### Step 7a: Find Node Names
Run `squeue -u $USER` and look at the **NODELIST(REASON)** column.

Example output:
```
JOBID   PARTITIO NAME     USER      ST  TIME  NODES NODELIST(REASON)
12345   public   vllm-gpt hehernan  R   0:05  1     sg001
12346   public   vllm-4b  hehernan  R   0:05  1     sg002
```
In this example:
- **User Node (GPT-OSS-20B):** `sg001`
- **Agent Node (Qwen-4B):** `sg002`

### Step 7b: Export Variables
Run these commands in your terminal (replace with YOUR actual node names):

```bash
export USER_NODE=sg001
export AGENT_NODE_4B=sg002
```

### Step 7c: Run Day 1 Script
```bash
sbatch day1_experiments.sh
```

To see the results later:
```bash
tail -f day1_exp_*.out
```
