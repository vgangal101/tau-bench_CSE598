# Setting Up vLLM for τ-bench (CSE598 Project)

## Step-by-Step vLLM Installation and Setup

### Prerequisites
- Conda environment: `Tau1` (you have this)
- GPU with sufficient VRAM (check below for requirements)
- CUDA installed

---

## Step 1: Install vLLM

```bash
# Activate your conda environment
conda activate Tau1

# Install vLLM (this will take a few minutes)
pip install vllm

# Verify installation
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
```

---

## Step 2: Choose Your Qwen Models

For your CSE598 project, you need **4 different model sizes**: 4B, 8B, 14B, 32B

### Available Qwen 2.5 Models on HuggingFace:

| Model Size | HuggingFace Model ID | GPU Memory Needed* |
|------------|---------------------|-------------------|
| 3B | `Qwen/Qwen2.5-3B-Instruct` | ~8 GB |
| 7B | `Qwen/Qwen2.5-7B-Instruct` | ~16 GB |
| 14B | `Qwen/Qwen2.5-14B-Instruct` | ~30 GB |
| 32B | `Qwen/Qwen2.5-32B-Instruct` | ~70 GB |
| 72B | `Qwen/Qwen2.5-72B-Instruct` | ~150 GB |

*Approximate memory for FP16. Can be reduced with quantization.

### Recommended Models for Your Project:
```bash
# Use these model IDs (closest to your requirements):
- Qwen/Qwen2.5-3B-Instruct    # ~4B requirement
- Qwen/Qwen2.5-7B-Instruct     # ~8B requirement  
- Qwen/Qwen2.5-14B-Instruct    # 14B requirement (exact match!)
- Qwen/Qwen2.5-32B-Instruct    # 32B requirement (exact match!)
```

---

## Step 3: Check Your GPU Memory

```bash
# Check available GPU memory
nvidia-smi

# Or more detailed info
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')"
```

**Important**: Choose models that fit in your GPU memory!
- If you have 24GB GPU → Start with 7B or 14B
- If you have 40GB GPU → You can run 14B or 32B
- If you have 80GB GPU → You can run 32B or even 72B

---

## Step 4: Start vLLM Server

vLLM will **automatically download** the model from HuggingFace the first time you run it.

### Option A: Start with 7B model (recommended for testing)

```bash
conda activate Tau1

# Start vLLM server (will auto-download model on first run)
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --max-model-len 8192

# Keep this terminal open - this is your vLLM server!
```

### Option B: Start with quantized model (saves GPU memory)

If you're running out of memory, use AWQ quantization:

```bash
# Use 4-bit quantized version (much smaller)
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
  --host 0.0.0.0 \
  --port 8000 \
  --quantization awq \
  --dtype auto
```

---

## Step 5: Test vLLM Server is Running

**In a NEW terminal** (keep vLLM server running in the first one):

```bash
# Test 1: Check if server is responding
curl http://localhost:8000/v1/models

# Expected output: JSON with model info

# Test 2: Send a test completion request
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.0
  }'
```

---

## Step 6: Configure τ-bench to Use vLLM (Zero Code Changes!)

```bash
# In your new terminal (NOT the vLLM server terminal)
conda activate Tau1

# Set environment variable to point to vLLM server
export OPENAI_API_BASE="http://localhost:8000/v1"
export OPENAI_API_KEY="dummy"  # vLLM doesn't need a real key

# Navigate to your project
cd /Users/hectorhernandez/Desktop/repo/tau-bench_CSE598

# Test connection with our test script
python test_vllm_connection.py
```

---

## Step 7: Run τ-bench with vLLM

```bash
conda activate Tau1
export OPENAI_API_BASE="http://localhost:8000/v1"

python run.py \
  --agent-strategy tool-calling \
  --env retail \
  --model Qwen/Qwen2.5-7B-Instruct \
  --model-provider openai \
  --user-model gpt-4o \
  --user-model-provider openai \
  --num-trials 5 \
  --max-concurrency 3 \
  --task-ids 0 1 2
```

---

## Step 8: Running Multiple Model Sizes

For your project, you need to run experiments with 4 different model sizes. Here's how:

### Start Each Model Size Separately

You can only run **one vLLM server at a time** (per port). So you'll need to:

1. **Run experiments with 7B model** → Stop server
2. **Restart with 14B model** → Run experiments → Stop server
3. **Restart with 32B model** → Run experiments → Stop server

### Example Workflow:

```bash
# Terminal 1: Start 7B model
conda activate Tau1
vllm serve Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 8000 --dtype auto

# Terminal 2: Run experiments
conda activate Tau1
export OPENAI_API_BASE="http://localhost:8000/v1"

# Run all baselines with 7B
python run.py --agent-strategy tool-calling --model Qwen/Qwen2.5-7B-Instruct --model-provider openai --env retail --num-trials 5 ...
python run.py --agent-strategy react --model Qwen/Qwen2.5-7B-Instruct --model-provider openai --env retail --num-trials 5 ...
python run.py --agent-strategy act --model Qwen/Qwen2.5-7B-Instruct --model-provider openai --env retail --num-trials 5 ...

# When done, stop vLLM server (Ctrl+C in Terminal 1)
# Then restart with 14B model and repeat
```

---

## Troubleshooting

### Issue: "CUDA out of memory"
**Solution**: Use smaller model or quantized version
```bash
# Use quantized model (4-bit)
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ --quantization awq ...
```

### Issue: Model download is slow
**Solution**: vLLM downloads from HuggingFace. First download can take 10-30 minutes depending on model size and internet speed. It only downloads once!

### Issue: "Address already in use"
**Solution**: Another process is using port 8000
```bash
# Use different port
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8001 ...

# Then update your API base
export OPENAI_API_BASE="http://localhost:8001/v1"
```

### Issue: vLLM server crashes
**Solution**: Check GPU memory, try smaller model or add `--gpu-memory-utilization 0.9`

---

## Quick Reference Commands

```bash
# 1. Activate environment
conda activate Tau1

# 2. Start vLLM (choose model size)
vllm serve Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 8000 --dtype auto

# 3. In new terminal, set environment
conda activate Tau1
export OPENAI_API_BASE="http://localhost:8000/v1"

# 4. Run τ-bench
python run.py --model Qwen/Qwen2.5-7B-Instruct --model-provider openai ...
```

---

## What's Different from Ollama?

| Feature | Ollama | vLLM |
|---------|--------|------|
| Ease of use | ⭐⭐⭐⭐⭐ Very easy | ⭐⭐⭐ Moderate |
| Performance | ⭐⭐⭐ Good | ⭐⭐⭐⭐⭐ Excellent |
| Memory efficiency | ⭐⭐⭐ Good | ⭐⭐⭐⭐ Better |
| Model selection | ⭐⭐⭐⭐ Curated | ⭐⭐⭐⭐⭐ Any HF model |
| Setup complexity | Simple | Moderate |

**Bottom line**: vLLM is faster and more flexible, but requires more manual setup!
