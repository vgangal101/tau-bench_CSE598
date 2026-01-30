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

### Directory Structure

```
SOL_env/
├── day1/
│   ├── combined_experiment_4b.sh      # FP16: User 32B + Agent 4B
│   └── int8_experiment_4b.sh          # INT8: User 32B + Agent 4B
├── day2/
│   ├── combined_experiment_8b.sh      # FP16: User 32B + Agent 8B
│   └── int8_experiment_8b.sh          # INT8: User 32B + Agent 8B
├── day3/
│   ├── combined_experiment_14b.sh     # FP16: User 32B + Agent 14B
│   └── int8_experiment_14b.sh         # INT8: User 32B + Agent 14B (both INT8)
├── day4/
│   ├── combined_experiment_32b.sh     # FP16: User 32B + Agent 32B
│   └── int8_experiment_32b.sh         # INT8: User 32B + Agent 32B (both INT8)
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

```bash
# Submit a job
cd SOL_env/day2
sbatch combined_experiment_8b.sh

# Monitor job
squeue -u $USER
tail -f logs/combined_experiment_8b_<job_id>.out

# Check for errors
cat logs/combined_experiment_8b_<job_id>.err
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
