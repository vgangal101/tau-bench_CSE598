# Running τ-bench with Qwen Models - Complete Examples

This document shows comprehensive examples for running the benchmark with all available Qwen models from DashScope, OpenRouter, and local servers.

## Quick Setup

```bash
# 1. Activate conda environment
conda activate Tau1

# 2. Create .env file with your API keys
cp .env.template .env
nano .env

# Add these lines:
# DASHSCOPE_API_KEY=sk-...  (from https://dashscope.console.aliyun.com/)
# OPENROUTER_API_KEY=sk-or-v1-...  (from https://openrouter.ai/)
```

---

## 1. ALIBABA DASHSCOPE - Singapore Endpoint

### Basic Setup

Available models: `qwen3-4b`, `qwen3-8b`, `qwen3-14b`, `qwen3-32b`

Default region: `singapore`

### Example 1: Qwen3-4B Agent + OpenRouter User (Singapore)

```bash
python run.py \
  --env retail \
  --agent-strategy tool-calling \
  --model qwen3-4b \
  --model-provider dashscope \
  --dashscope-region singapore \
  --user-model openai/gpt-oss-20b \
  --user-model-provider openrouter \
  --user-strategy llm \
  --task-ids 0 1 2 \
  --max-concurrency 1 \
  --max-tokens 1000 \
  --user-max-tokens 500
```

### Example 2: Qwen3-8B Agent + OpenRouter User (Singapore)

```bash
python run.py \
  --env retail \
  --agent-strategy tool-calling \
  --model qwen3-8b \
  --model-provider dashscope \
  --dashscope-region singapore \
  --user-model openai/gpt-oss-20b \
  --user-model-provider openrouter \
  --user-strategy llm \
  --task-split test \
  --max-concurrency 5
```

### Example 3: Qwen3-14B with ReAct Strategy (Singapore)

```bash
python run.py \
  --env airline \
  --agent-strategy react \
  --model qwen3-14b \
  --model-provider dashscope \
  --dashscope-region singapore \
  --user-model openai/gpt-oss-20b \
  --user-model-provider openrouter \
  --user-strategy react \
  --max-concurrency 3
```

### Example 4: Qwen3-32B Large Model (Singapore)

```bash
python run.py \
  --env retail \
  --agent-strategy tool-calling \
  --model qwen3-32b \
  --model-provider dashscope \
  --dashscope-region singapore \
  --user-model openai/gpt-oss-20b \
  --user-model-provider openrouter \
  --max-tokens 2000 \
  --user-max-tokens 800 \
  --max-concurrency 2
```

---

## 2. ALIBABA DASHSCOPE - US Endpoint

### Explicit US Region Selection

Same models but using the US data center.

### Example 5: Qwen3-8B Agent (US Endpoint)

```bash
python run.py \
  --env retail \
  --agent-strategy tool-calling \
  --model qwen3-8b \
  --model-provider dashscope \
  --dashscope-region us \
  --user-model openai/gpt-oss-20b \
  --user-model-provider openrouter \
  --user-strategy llm \
  --task-split test \
  --max-concurrency 5
```

### Example 6: Qwen3-14B Agent (US Endpoint)

```bash
python run.py \
  --env airline \
  --agent-strategy react \
  --model qwen3-14b \
  --model-provider dashscope \
  --dashscope-region us \
  --user-model openai/gpt-oss-20b \
  --user-model-provider openrouter \
  --user-strategy react \
  --max-concurrency 3
```

### Example 7: Qwen3-32B (US Endpoint) - Heavy Workload

```bash
python run.py \
  --env retail \
  --agent-strategy tool-calling \
  --model qwen3-32b \
  --model-provider dashscope \
  --dashscope-region us \
  --user-model openai/gpt-oss-20b \
  --user-model-provider openrouter \
  --task-split train \
  --num-trials 3 \
  --max-concurrency 2 \
  --max-tokens 2000
```

---

## 3. OPENROUTER - All Qwen Models

### Available Models

- `qwen/qwen3-8b`
- `qwen/qwen3-14b`
- `qwen/qwen3-32b`
- `openai/gpt-oss-20b` (not just for user simulator)


### Example 8: Qwen3-8B via OpenRouter

```bash
python run.py \
  --env retail \
  --agent-strategy tool-calling \
  --model qwen/qwen3-8b \
  --model-provider openrouter \
  --user-model openai/gpt-oss-20b \
  --user-model-provider openrouter \
  --user-strategy llm \
  --max-concurrency 5
```

### Example 10: Qwen3-14B via OpenRouter with ReAct

```bash
python run.py \
  --env airline \
  --agent-strategy react \
  --model qwen/qwen3-14b \
  --model-provider openrouter \
  --user-model openai/gpt-oss-20b \
  --user-model-provider openrouter \
  --user-strategy react \
  --max-concurrency 3
```

### Example 10: Qwen3-32B via OpenRouter

```bash
python run.py \
  --env retail \
  --agent-strategy tool-calling \
  --model qwen/qwen3-32b \
  --model-provider openrouter \
  --user-model openai/gpt-oss-20b \
  --user-model-provider openrouter \
  --task-split test \
  --max-tokens 2000 \
  --user-max-tokens 800 \
  --max-concurrency 2
```

### Example 11: GPT-OSS-20B as Agent (OpenRouter)

```bash
python run.py \
  --env retail \
  --agent-strategy tool-calling \
  --model openai/gpt-oss-20b \
  --model-provider openrouter \
  --user-model openai/gpt-oss-20b \
  --user-model-provider openrouter \
  --user-strategy llm \
  --max-concurrency 5
```

---

## 4. LOCAL MODELS - Self-Hosted on Servers

### Setup Requirements

You need to have Qwen models running on local servers with OpenAI-compatible API endpoints.

Example: Using `vLLM` or `LM Studio`

```bash
# Server 1: Qwen3-8B on port 8000
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen-3-8B --port 8000

# Server 2: Qwen3-4B on port 8001
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen-3-4B --port 8001

# Server 3: Qwen3-32B on port 8002
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen-3-32B --port 8002
```

### Example 12: Local Qwen3-8B Agent + OpenRouter User

Agent on localhost:8000, user uses OpenRouter

```bash
python run.py \
  --env retail \
  --agent-strategy tool-calling \
  --model qwen3-8b \
  --model-provider local \
  --model-base-url http://localhost:8000/v1 \
  --model-api-key "dummy-key" \
  --user-model openai/gpt-oss-20b \
  --user-model-provider openrouter \
  --user-strategy llm \
  --task-ids 0 1 2 \
  --max-concurrency 1
```

### Example 13: Both Local - Different Ports

Agent on port 8000, user on port 8001

```bash
python run.py \
  --env retail \
  --agent-strategy tool-calling \
  --model qwen3-8b \
  --model-provider local \
  --model-base-url http://localhost:8000/v1 \
  --model-api-key "dummy-key" \
  --user-model qwen3-4b \
  --user-model-provider local \
  --user-model-base-url http://localhost:8001/v1 \
  --user-model-api-key "dummy-key" \
  --user-strategy llm \
  --task-split test \
  --max-concurrency 1
```

### Example 14: Local Agent on Remote Server + OpenRouter User

Agent on remote server (e.g., 192.168.1.100), user uses OpenRouter

```bash
python run.py \
  --env airline \
  --agent-strategy react \
  --model qwen3-14b \
  --model-provider local \
  --model-base-url http://192.168.1.100:8000/v1 \
  --model-api-key "dummy-key" \
  --user-model openai/gpt-oss-20b \
  --user-model-provider openrouter \
  --user-strategy react \
  --max-concurrency 3
```

### Example 15: Remote Server with Authentication

If your local server requires authentication:

```bash
python run.py \
  --env retail \
  --agent-strategy tool-calling \
  --model qwen3-8b \
  --model-provider local \
  --model-base-url http://192.168.1.50:8000/v1 \
  --model-api-key "your-auth-token-here" \
  --user-model openai/gpt-oss-20b \
  --user-model-provider openrouter \
  --task-ids 0 1 2 3 4 5 \
  --max-concurrency 2
```

### Example 16: Multiple Local Models Across Servers

Agent on server A (port 8000), user on server B (port 8000)

```bash
python run.py \
  --env retail \
  --agent-strategy tool-calling \
  --model qwen3-32b \
  --model-provider local \
  --model-base-url http://server-a.local:8000/v1 \
  --model-api-key "server-a-key" \
  --user-model qwen3-8b \
  --user-model-provider local \
  --user-model-base-url http://server-b.local:8000/v1 \
  --user-model-api-key "server-b-key" \
  --user-strategy llm \
  --max-concurrency 2 \
  --max-tokens 2000 \
  --user-max-tokens 500
```

---

## 5. MIXED CONFIGURATIONS - Combining Providers

### Example 17: DashScope Agent + Local User

```bash
python run.py \
  --env retail \
  --agent-strategy tool-calling \
  --model qwen3-14b \
  --model-provider dashscope \
  --dashscope-region singapore \
  --user-model qwen3-4b \
  --user-model-provider local \
  --user-model-base-url http://localhost:8001/v1 \
  --user-model-api-key "dummy-key" \
  --user-strategy llm \
  --max-concurrency 2
```

### Example 18: OpenRouter Agent + Local User

```bash
python run.py \
  --env airline \
  --agent-strategy react \
  --model qwen/qwen3-32b \
  --model-provider openrouter \
  --user-model qwen3-8b \
  --user-model-provider local \
  --user-model-base-url http://192.168.1.50:8000/v1 \
  --user-model-api-key "dummy-key" \
  --user-strategy react \
  --max-concurrency 1
```

### Example 19: Local Agent + DashScope User

```bash
python run.py \
  --env retail \
  --agent-strategy tool-calling \
  --model qwen3-8b \
  --model-provider local \
  --model-base-url http://localhost:8000/v1 \
  --model-api-key "dummy-key" \
  --user-model qwen3-4b \
  --user-model-provider dashscope \
  --dashscope-region us \
  --user-strategy llm \
  --task-split dev \
  --max-concurrency 1
```

---

## 6. ADVANCED CONFIGURATIONS

### Example 20: Full Test Run with All Trials

```bash
python run.py \
  --env retail \
  --agent-strategy tool-calling \
  --model qwen3-8b \
  --model-provider dashscope \
  --dashscope-region singapore \
  --user-model openai/gpt-oss-20b \
  --user-model-provider openrouter \
  --task-split test \
  --num-trials 5 \
  --max-concurrency 3 \
  --seed 42
```

### Example 21: Specific Task Benchmarking

```bash
python run.py \
  --env retail \
  --agent-strategy react \
  --model qwen/qwen3-14b \
  --model-provider openrouter \
  --user-model openai/gpt-oss-20b \
  --user-model-provider openrouter \
  --user-strategy react \
  --task-ids 10 20 30 40 50 \
  --max-concurrency 5 \
  --temperature 0.1
```

### Example 22: Different Temperature Settings

```bash
python run.py \
  --env airline \
  --agent-strategy tool-calling \
  --model qwen3-14b \
  --model-provider dashscope \
  --dashscope-region us \
  --user-model openai/gpt-oss-20b \
  --user-model-provider openrouter \
  --temperature 0.5 \
  --max-concurrency 2
```

### Example 23: High Max Tokens for Complex Tasks

```bash
python run.py \
  --env retail \
  --agent-strategy tool-calling \
  --model qwen3-32b \
  --model-provider dashscope \
  --dashscope-region singapore \
  --user-model openai/gpt-oss-20b \
  --user-model-provider openrouter \
  --max-tokens 4000 \
  --user-max-tokens 2000 \
  --task-split train \
  --max-concurrency 1
```

---

## 7. QUICK REFERENCE TABLE

| Example | Agent Model | Agent Provider | User Model | User Provider | Best For |
|---------|-------------|----------------|------------|---------------|----------|
| 1 | qwen3-4b | dashscope (SG) | gpt-oss-20b | openrouter | Quick testing |
| 2 | qwen3-8b | dashscope (SG) | gpt-oss-20b | openrouter | Balanced performance |
| 3 | qwen3-14b | dashscope (SG) | gpt-oss-20b | openrouter | Better reasoning |
| 4 | qwen3-32b | dashscope (SG) | gpt-oss-20b | openrouter | Best quality |
| 5 | qwen3-8b | dashscope (US) | gpt-oss-20b | openrouter | US region |
| 8 | qwen/qwen3-4b | openrouter | gpt-oss-20b | openrouter | Unified OpenRouter |
| 13 | qwen3-8b | local (8000) | gpt-oss-20b | openrouter | Local + cloud user |
| 14 | qwen3-8b | local (8000) | qwen3-4b | local (8001) | Full local setup |
| 15 | qwen3-14b | local (remote) | gpt-oss-20b | openrouter | Remote inference |

---

## 8. Monitoring Results

After running, check results:

```bash
# List results
ls -lh results/

# View latest result file
cat results/tool-calling-qwen3-8b-0.0_range_0--1_user-openai-gpt-oss-20b-llm_*.json | jq '.'

# Check specific metrics
cat results/*.json | jq '.[].reward' | awk '{sum+=$1; count++} END {print "Average Reward:", sum/count}'
```

---

## 9. Troubleshooting

### Missing API Keys

```
Error: DASHSCOPE_API_KEY environment variable not set
```

**Fix:** Add to .env file:
```
DASHSCOPE_API_KEY=sk-...
OPENROUTER_API_KEY=sk-or-v1-...
```

### Local Server Connection Error

```
Error: Connection refused for http://localhost:8000/v1
```

**Fix:** Start your local server:
```bash
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen-3-8B --port 8000
```

### Model Not Found on Provider

**Check available models:**
- DashScope: `qwen3-4b`, `qwen3-8b`, `qwen3-14b`, `qwen3-32b`
- OpenRouter: `qwen/qwen3-4b`, `qwen/qwen3-8b`, `qwen/qwen3-14b`, `qwen/qwen3-32b`, `openai/gpt-oss-20b`

---

## 10. Performance Tips

1. **Use fewer parallel tasks** (`--max-concurrency 1-2`) for local servers
2. **Use higher concurrency** (`--max-concurrency 10`) for cloud providers
3. **Adjust `--max-tokens`** based on your VRAM:
   - 4GB VRAM: `--max-tokens 512`
   - 8GB VRAM: `--max-tokens 1000`
   - 16GB VRAM: `--max-tokens 2000`
   - 24GB VRAM: `--max-tokens 4000`
4. **Batch tasks** with `--task-ids` instead of sequential execution

---

Good luck running your benchmarks! 🚀
