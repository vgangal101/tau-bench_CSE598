# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

**τ-bench** is a benchmark system for evaluating conversational agents in realistic tool-using scenarios. It simulates interactions between an AI agent and a user (or user simulator) within domain-specific environments (retail or airline) where the agent must use tools to complete tasks and engage in natural conversation.

## Development Setup

### Installation

```bash
pip install -e .
```

This installs the package in development mode along with dependencies: `openai`, `mistralai`, `anthropic`, `google-generativeai`, `litellm`, `tenacity`, `termcolor`, and `numpy`.

### Environment Variables

Set API keys before running:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...
export MISTRAL_API_KEY=...
```

## Common Commands

### Running the Benchmark

Basic run with a specific agent and environment:

```bash
# Tool-calling agent on retail environment
python run.py --agent-strategy tool-calling --env retail --model gpt-4o --model-provider openai --user-model gpt-4o --user-model-provider openai --user-strategy llm --max-concurrency 10

# Run specific tasks only
python run.py --agent-strategy tool-calling --env retail --model gpt-4o --model-provider openai --user-model gpt-4o --user-model-provider openai --user-strategy llm --max-concurrency 10 --task-ids 2 4 6

# Airline environment with different user strategy
python run.py --agent-strategy tool-calling --env airline --model gpt-4o --model-provider openai --user-model claude-3-5-sonnet-20240620 --user-model-provider anthropic --user-strategy react --max-concurrency 5
```

### Key Flags

- `--env`: Choose `retail` or `airline`
- `--agent-strategy`: `tool-calling`, `act`, `react`, or `few-shot`
- `--user-strategy`: `llm`, `react`, `verify`, `reflection`, or `human`
- `--model`, `--model-provider`: Agent model and provider
- `--user-model`, `--user-model-provider`: User simulator model and provider
- `--task-ids`: Run only specific task IDs
- `--task-split`: `train`, `test`, or `dev` (retail only)
- `--num-trials`: Run each task multiple times
- `--max-concurrency`: Number of parallel tasks
- `--seed`: Random seed for reproducibility
- `--temperature`: Sampling temperature (default 0.0)

### Auto Error Identification

Analyze failed trajectories to identify fault assignment and type:

```bash
python auto_error_identification.py --env retail --platform openai --results-path <results_file> --max-concurrency 16 --output-path <output_dir> --max-num-failed-results 10
```

This uses an LLM to classify errors as:
- **Fault Author**: user, agent, or environment
- **Fault Type**: called_wrong_tool, used_wrong_tool_argument, goal_partially_completed, took_unintended_action

## High-Level Architecture

### Core Components

The system has four main architectural layers:

#### 1. **Agents** (`tau_bench/agents/`)

Abstract `Agent` base class with implementations:
- **ToolCallingAgent**: Uses native LLM function-calling (fastest, most accurate)
- **ChatReActAgent**: Uses explicit "Thought:" -> "Action:" reasoning (ReAct/Act patterns)
- **FewShotAgent**: Uses in-context examples for prompting

All agents implement: `Agent.solve(env, task_index) -> SolveResult`

#### 2. **Environments** (`tau_bench/envs/`)

Abstract `Env` base class with domain implementations:
- **RetailEnv**: E-commerce customer service (300+ tasks, 16 tools)
  - Tools: check order status, apply discounts, modify shipping, etc.
  - Tasks: Retail customer inquiries and requests
- **AirlineEnv**: Flight booking/modification (tasks and 16 tools)
  - Tools: book flight, check reservation, modify booking, etc.
  - Tasks: Airline customer requests

Each environment has:
- **Tasks** dataset (train/test/dev splits with task indices)
- **Tools** (domain-specific functions with OpenAI function schemas)
- **Wiki** (domain knowledge, policies, and system instructions)
- **User Simulator** (simulates customer responses)

#### 3. **User Simulators** (`tau_bench/envs/user.py`)

Abstract `BaseUserSimulationEnv` with implementations:
- **LLMUserSimulationEnv**: Pure LLM-based user responses
- **ReactUserSimulationEnv**: LLM with explicit thinking step
- **VerifyUserSimulationEnv**: With verification loop for response quality
- **ReflectionUserSimulationEnv**: With reflection step before responding
- **HumanUserSimulationEnv**: For human evaluation

#### 4. **Model Utilities** (`tau_bench/model_utils/`)

Abstractions for LLM interaction:
- **Model Provider Integrations**: OpenAI, Anthropic, Mistral, Google (via LiteLLM)
- **API Abstraction**: Datapoint-based API interface for consistency
- **Function Tools**: Utilities for tool metadata and invocation

### Interaction Flow

Each benchmark run follows this pattern:

```
RunConfig (parsed from CLI args)
    ↓
Environment (retail or airline)
    ├─ Loads tasks (user instructions with expected outputs)
    ├─ Loads tools (domain-specific functions)
    ├─ Loads wiki (domain knowledge)
    └─ Initializes user simulator
    ↓
Agent (selected strategy)
    ├─ For each task:
    │   ├─ Call env.reset(task_id) → User gives initial instruction
    │   ├─ Loop until terminal or max_steps:
    │   │   ├─ Agent generates action (tool call or response)
    │   │   ├─ env.step(action) → Execute tool or evaluate response
    │   │   ├─ Get observation/tool result
    │   │   ├─ User simulator responds to any agent output
    │   │   └─ Check if conversation done (###STOP### marker)
    │   └─ env.calculate_reward() → Compare actual vs expected state
    └─ Return SolveResult (reward, messages, API cost)
    ↓
ThreadPoolExecutor (runs multiple tasks in parallel)
    ↓
Results saved to JSON, metrics calculated (Pass^k values)
```

### Reward System

The environment calculates reward by comparing:
- **Action-based reward**: Database state after tools match expected ground truth
- **Output-based reward**: Agent responses contain expected information

Result: 1.0 if both checks pass, 0.0 otherwise

### Key Design Patterns

1. **Factory Pattern**: Agent and environment creation based on strategy/domain selection
2. **Abstract Base Classes**: `Agent`, `Env`, `Tool`, `BaseUserSimulationEnv` enable extensibility
3. **Tool Schemas**: Tools provide OpenAI-compatible function schemas via `Tool.get_info()`
4. **Configuration-Driven**: `RunConfig` (Pydantic model) captures all parameters for reproducibility
5. **Parallel Execution**: ThreadPoolExecutor with isolated environment per task
6. **Type Safety**: Use of Pydantic models for `Action`, `Task`, `SolveResult`, etc.

## Extending the System

### Adding a New Agent Strategy

1. Create new class in `tau_bench/agents/` that inherits from `Agent`
2. Implement `solve(env, task_index, max_num_steps)` method
3. Register in agent factory function in `tau_bench/run.py`
4. Add choice to `--agent-strategy` argument in `run.py`

### Adding a New User Simulation Strategy

1. Create new class in `tau_bench/envs/user.py` that inherits from `BaseUserSimulationEnv`
2. Implement `reset()`, `step()`, and `get_total_cost()` methods
3. Add enum value to `UserStrategy` in `user.py`
4. Register in user factory in environment initialization

### Adding a New Domain

1. Create directory `tau_bench/envs/<domain>/`
2. Implement environment class inheriting from `Env`
3. Define `Task` and `Tool` subclasses for the domain
4. Create task dataset files (tasks_test.py, tasks_train.py, etc.)
5. Create wiki file with domain knowledge and policies
6. Register in environment factory in `tau_bench/envs/__init__.py`
7. Add choice to `--env` argument in `run.py`

## Important Data Structures

### Core Types (`tau_bench/types.py`)

- **Task**: User instruction, expected actions, expected outputs
- **Action**: Tool call (name + args) or text response
- **SolveResult**: Task outcome with reward, messages, cost
- **RunConfig**: All benchmark configuration parameters
- **EnvResponse**: Environment response to an action (observation, done flag)

### Task Dataset Format (`tasks_test.py`)

```python
TASKS = [
    Task(
        id=0,
        user_instruction="Customer wants to...",
        ground_truth_actions=[Action(...), ...],
        ground_truth_outputs=["Expected response 1", ...],
        expected_action_count=2,
    ),
    ...
]
```

## Testing

The `tasks_test.py` files in each domain contain the task definitions used by the benchmark. These are not traditional unit tests but rather task datasets that define ground truth for evaluation.

To verify tasks are correctly defined, check:
- All expected actions are semantically valid for the domain
- Ground truth outputs match the domain knowledge
- Task instructions are clear and unambiguous
