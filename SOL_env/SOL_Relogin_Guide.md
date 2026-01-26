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
