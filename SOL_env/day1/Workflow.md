# Day 1 Experiments Workflow

## Step 1: Start vLLM Servers (Separate Jobs)
Submit the SLURM jobs to start the User Simulator (GPT-OSS-20B) on port 8000 and the Agent (Qwen-4B) on port 8001.

```bash
sbatch vllm_gpt_oss_20b.sh    # User on port 8000
sbatch test_vllm_qwen4b.sh    # Agent on port 8001
```

## Step 2: Get Node Names
Wait for the jobs to start (check `squeue --me`), then run this script to find out which nodes they were assigned to.

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
```

## Step 4: Submit Experiment Job
Once the variables are set, submit the experiment script.

```bash
sbatch day1_experiments.sh
```

You can monitor the output in the `logs/` directory.
