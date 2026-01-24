#!/bin/bash
# Quick Start Script for vLLM Setup
# Usage: ./vllm_quickstart.sh

echo "🚀 vLLM Quick Start for τ-bench"
echo "================================"
echo ""

# Activate conda environment
echo "Step 1: Activating Tau1 environment..."
conda activate Tau1

# Check if vLLM is installed
echo ""
echo "Step 2: Checking vLLM installation..."
if python -c "import vllm" 2>/dev/null; then
    echo "✅ vLLM is already installed"
    python -c "import vllm; print(f'   Version: {vllm.__version__}')"
else
    echo "❌ vLLM not found. Installing..."
    pip install vllm
    echo "✅ vLLM installed successfully"
fi

# Check GPU
echo ""
echo "Step 3: Checking GPU..."
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB' if torch.cuda.is_available() else 'No GPU found')"

echo ""
echo "📋 Next Steps:"
echo "==============="
echo ""
echo "1. Start vLLM server with your qwen3-4b model:"
echo ""
echo "   vllm serve qwen3-4b --host 0.0.0.0 --port 8000 --dtype auto"
echo ""
echo "2. In a NEW terminal, test the connection:"
echo "   conda activate Tau1"
echo "   python test_vllm_connection.py"
echo ""
echo "3. Run τ-bench experiments:"
echo "   export OPENAI_API_BASE='http://localhost:8000/v1'"
echo "   python run.py --model qwen3-4b --model-provider openai ..."
echo ""
echo "📖 See VLLM_SETUP_GUIDE.md for detailed instructions"
