# Scripts Directory

This directory contains setup scripts and documentation for the CSE598 τ-bench project.

## 📚 Documentation

- **`claude.md`** - Complete project documentation covering τ-bench, vLLM, baseline methods, and setup
- **`VLLM_SETUP_GUIDE.md`** - Detailed step-by-step guide for setting up vLLM with Qwen models

## 🚀 Quick Start Scripts

### vLLM Setup (qwen3-4b)

1. **`START_VLLM_qwen3-4b.sh`** - Start vLLM server with qwen3-4b model
   ```bash
   ./scripts/START_VLLM_qwen3-4b.sh
   ```

2. **`RUN_EXPERIMENT_qwen3-4b.sh`** - Run τ-bench experiment (requires server running)
   ```bash
   ./scripts/RUN_EXPERIMENT_qwen3-4b.sh
   ```

3. **`test_vllm_connection.py`** - Test LiteLLM connection to vLLM server
   ```bash
   python scripts/test_vllm_connection.py
   ```

### Other Scripts

- **`vllm_quickstart.sh`** - Check vLLM installation and show next steps
- **`setup_ollama_env.sh`** - Configure Ollama environment (alternative to vLLM)

## 📖 Usage Workflow

```bash
# 1. Check installation
./scripts/vllm_quickstart.sh

# 2. Start vLLM server (Terminal 1)
./scripts/START_VLLM_qwen3-4b.sh

# 3. Test connection (Terminal 2)
conda activate Tau1
python scripts/test_vllm_connection.py

# 4. Run experiments
./scripts/RUN_EXPERIMENT_qwen3-4b.sh
```

## 🔧 Manual Usage

If you prefer manual control:

```bash
# Start server
conda activate Tau1
vllm serve qwen3-4b --host 0.0.0.0 --port 8000 --dtype auto

# In new terminal
conda activate Tau1
export OPENAI_API_BASE="http://localhost:8000/v1"

python run.py \
  --model qwen3-4b \
  --model-provider openai \
  --agent-strategy tool-calling \
  --env retail \
  --num-trials 5
```

## 📝 Notes

- All scripts assume conda environment `Tau1` is available
- vLLM server must be running before running experiments
- Scripts are configured for `qwen3-4b` model
- Main project files remain in repository root
