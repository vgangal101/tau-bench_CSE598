# Agent.md - SOL Environment Findings

## Domain Companies

### 1. Airline Domain (tau_bench/envs/airline/)
**Purpose**: Flight booking, modification, and reservation management

**Key Features**:
- Flight search, booking, cancellation, modification
- Membership tiers: Regular, Silver, Gold
- Cabin classes: Basic Economy, Economy, Business
- Baggage allowance policies (varies by membership tier)
- Travel insurance options ($30/passenger)
- Certificate compensation for delays/cancellations ($50-$100 × passengers)

**Data**: Users, flights, reservations

**Policies**:
- 24-hour free cancellation window
- Strict modification rules (basic economy cannot be modified)
- No refunds without insurance for economy tickets
- Business tickets can always be cancelled
- Membership tier affects baggage allowance:
  - Regular: 0-2 free bags (depending on cabin)
  - Silver: 1-3 free bags
  - Gold: 2-3 free bags

### 2. Retail Domain (tau_bench/envs/retail/)
**Purpose**: E-commerce order management

**Key Features**:
- 50 product types with variants (color, size, material, style)
- Order states: pending, processed, delivered, cancelled
- Operations: cancel, modify, return, exchange orders
- Payment methods: Gift cards, PayPal, Credit cards
- User authentication via email or name+zip code

**Data**: Products, orders, users

**Policies**:
- Pending orders can be modified (address, payment, items - once only)
- Delivered orders can be returned or exchanged
- Returns go to original payment method or existing gift card
- Refund timeline: immediate for gift cards, 5-7 business days otherwise
- Item exchanges limited to same product type (different variants allowed)

---

## SOL Environment Configuration

**Cluster**: Arizona State University SOL Cluster
- **Hardware**: A100 80GB GPUs
- **Partition**: public
- **Account**: class_cse59827694spring2026

**Software Stack**:
```bash
module load mamba/latest
module load cuda-12.1.1-gcc-12.1.0
source activate tau-bench
```

**Environment Variables**:
- `HF_HOME`: `/scratch/$USER/hf_cache`
- `VLLM_USE_V1`: 0
- `OPENAI_API_KEY`: dummy (required placeholder)

---

## Experiment Scripts Structure

```
SOL_env/
├── multi_node_experiment.sh    # 2 nodes: User(32B) + Agent(32B)
├── 4b_run/experiment_4b.sh     # User(32B) + Agent(4B)
├── 8b_run/experiment_8b.sh     # User(32B) + Agent(8B)
├── 14b_run/experiment_14b.sh   # User(32B) + Agent(14B)
└── 32b_run/experiment_32b.sh   # User(32B) + Agent(32B)
```

**Each script configures**:
- **Nodes**: 2 (Node 1 = User Simulator, Node 2 = Agent Model)
- **GPUs**: 1x A100 80GB per node
- **CPUs**: 16 per task
- **Memory**: 96GB per node
- **Time**: 6 hours (multi_node uses 2 hours)
- **Ports**: 8000 (both nodes, same port due to separate hosts)

---

## Model Configurations

| Directory | User Model | Agent Model | Context | Concurrency | Time |
|-----------|-----------|-------------|---------|-------------|------|
| 4b_run | Qwen/Qwen3-32B | Qwen/Qwen3-4B | 32768 | 20 | 6 hours |
| 8b_run | Qwen/Qwen3-32B | Qwen/Qwen3-8B | 32768 | 20 | 6 hours |
| 14b_run | Qwen/Qwen3-32B | Qwen/Qwen3-14B | 32768 | 20 | 6 hours |
| 32b_run | Qwen/Qwen3-32B | Qwen/Qwen3-32B | 32768 | 20 | 6 hours |

**vLLM Server Flags**:
```bash
# Common to all models
--host 0.0.0.0
--port 8000
--tensor-parallel-size 1
--gpu-memory-utilization 0.90-0.95
--max-model-len 32768
--trust-remote-code
--enforce-eager
--disable-log-requests

# Agent model only
--enable-auto-tool-choice
--tool-call-parser hermes
```

---

## Experiment Loop

For each combination of:
- **Environments**: retail, airline
- **Strategies**: tool-calling, act, react

Run:
```bash
python run.py \
    --env ${ENV} \
    --agent-strategy ${STRATEGY} \
    --model ${AGENT_MODEL} \
    --model-provider openai \
    --model-base-url ${AGENT_URL}/v1 \
    --user-model ${USER_MODEL} \
    --user-model-provider openai \
    --user-model-base-url ${USER_URL}/v1 \
    --log-dir ${LOG_DIR} \
    --max-concurrency 20 \
    --num-trials 5 \
    --end-index -1
```

**Parameters**:
- `--end-index -1`: Runs all tasks (no upper limit)
- `--num-trials 5`: Each task is run 5 times
- `--max-concurrency 20`: 20 tasks run in parallel

---

## Log File Locations

All logs are consolidated in the script's directory:
```
SOL_env/{X}b_run/logs/
├── tau-{X}b-exp_<job_id>.out      # SLURM stdout
├── tau-{X}b-exp_<job_id>.err      # SLURM stderr
├── {X}b_user_<job_id>.log         # vLLM user server log
├── {X}b_agent_<job_id>.log        # vLLM agent server log
```

**Results Directory**:
```
SOL_env/{X}b_run/results/{env}/{strategy}/
```

---

## Running Experiments

**Submit from anywhere** (scripts auto-detect location):
```bash
# From repo root (recommended)
cd /scratch/$USER/tau-bench-project/tau-bench
sbatch SOL_env/4b_run/experiment_4b.sh

# Or from SOL_env directory
cd SOL_env/4b_run
sbatch experiment_4b.sh

# Or use absolute path
sbatch /scratch/$USER/tau-bench-project/tau-bench/SOL_env/4b_run/experiment_4b.sh
```

**Monitor job**:
```bash
squeue -u $USER
tail -f SOL_env/4b_run/logs/4b_user_<job_id>.log
cat SOL_env/4b_run/logs/tau-4b-exp_<job_id>.out
```

---

## Key Observations

1. **Directory Independence**: All scripts use `SCRIPT_DIR` and `REPO_ROOT` variables for path resolution, allowing submission from any directory

2. **BASH_SOURCE Path Resolution**: Scripts use `BASH_SOURCE` for reliable path detection regardless of where you submit from:
   ```bash
   # BASH_SOURCE always gives the actual script location
   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
   REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
   ```

3. **Cross-Node Communication**: User and Agent models run on separate nodes, communicate via `http://${NODE}:8000/v1`

4. **Robust Error Handling**: Scripts include:
   - Server health checks via `/health` endpoint
   - Timeout handling (10 minute intervals, up to 60 attempts)
   - Automatic log consolidation on cleanup
   - Tail error output on server startup failure

5. **User Simulator**: Always uses Qwen/Qwen3-32B regardless of agent model size (prioritizing user simulation quality)

6. **Resource Allocation**: 
   - 32B FP16 model: ~64GB GPU memory (A100 80GB fits comfortably)
   - Smaller models (4B, 8B, 14B) use higher `gpu-memory-utilization` (0.95)

---

## Task Splits

- **Airline**: test split only
- **Retail**: train, dev, test splits (default: test)

---

## Multi-Node Experiment (multi_node_experiment.sh)

Special configuration for multi-node runs:
- **Nodes**: 2 (separate nodes for user and agent, both using 32B)
- **Time**: 2 hours (shorter duration)
- **Concurrency**: 5 (lower than main experiments)
- **Max Context**: 32768 tokens
- **Purpose**: Validation/testing configuration

---

## Quick Reference: Command Line Options

```bash
python run.py \
    --env <retail|airline> \
    --agent-strategy <tool-calling|act|react|few-shot> \
    --model <model_name> \
    --model-provider <openai|anthropic|google|mistral> \
    --model-base-url <url>/v1 \
    --user-model <model_name> \
    --user-model-provider <provider> \
    --user-model-base-url <url>/v1 \
    --log-dir <directory> \
    --max-concurrency <int> \
    --num-trials <int> \
    --task-split <train|dev|test> \
    --start-index <int> \
    --end-index <int> \
    --task-ids <int1 int2 ...> \
    --seed <int> \
    --shuffle <0|1> \
    --user-strategy <llm|react|verify|reflection> \
    --few-shot-displays-path <path> \
    --temperature <float>
```

---

## Results Analysis

Results saved as JSON files in the specified log directory, containing:
- Task trajectories (conversation history, tool calls, responses)
- Rewards (0.0 or 1.0 per task)
- Pass^k metrics (success rate in best of k trials)
- Error information (if applicable)

---

## Troubleshooting

**Server fails to start**:
- Check corresponding log file in `{X}b_run/logs/`
- Verify HF cache path: `/scratch/$USER/hf_cache`
- Ensure model is downloaded or accessible

**Job submission fails**:
- Verify account: `class_cse59827694spring2026`
- Check partition: `public`
- Ensure scripts are executable: `chmod +x *.sh`

**Experiment runs but shows errors**:
- Check SLURM output files in `logs/` directory
- Verify vLLM server health via curl: `curl http://${NODE}:8000/health`
- Review agent strategy implementation for syntax errors

---

## Contact & Support

For issues specific to this experiment setup:
- Review logs in `SOL_env/{X}b_run/logs/`
- Check results in `SOL_env/{X}b_run/results/`
- Refer to CLAUDE.md for general project documentation
