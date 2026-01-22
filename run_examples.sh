#!/bin/bash

# τ-bench Model Runner - Convenience Script
# Run common configurations easily

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠️  .env file not found!${NC}"
    echo "Creating .env from template..."
    cp .env.template .env
    echo "Please edit .env and add your API keys:"
    echo "  - DASHSCOPE_API_KEY (from https://dashscope.console.aliyun.com/)"
    echo "  - OPENROUTER_API_KEY (from https://openrouter.ai/)"
    exit 1
fi

# Default values
ENV="retail"
TASK_SPLIT="test"
MAX_CONCURRENCY="1"
TASK_IDS=""
NUM_TRIALS="1"

# Show menu if no arguments
if [ $# -eq 0 ]; then
    print_header "τ-bench Model Runner"
    echo "Usage: ./run_examples.sh <command> [options]"
    echo ""
    echo "DASHSCOPE EXAMPLES:"
    echo "  ./run_examples.sh dashscope-sg-4b    - Qwen3-4B on Singapore"
    echo "  ./run_examples.sh dashscope-sg-8b    - Qwen3-8B on Singapore"
    echo "  ./run_examples.sh dashscope-sg-14b   - Qwen3-14B on Singapore"
    echo "  ./run_examples.sh dashscope-sg-32b   - Qwen3-32B on Singapore"
    echo "  ./run_examples.sh dashscope-us-8b    - Qwen3-8B on US"
    echo "  ./run_examples.sh dashscope-us-32b   - Qwen3-32B on US"
    echo ""
    echo "OPENROUTER EXAMPLES:"
    echo "  ./run_examples.sh openrouter-4b      - Qwen3-4B via OpenRouter"
    echo "  ./run_examples.sh openrouter-8b      - Qwen3-8B via OpenRouter"
    echo "  ./run_examples.sh openrouter-14b     - Qwen3-14B via OpenRouter"
    echo "  ./run_examples.sh openrouter-32b     - Qwen3-32B via OpenRouter"
    echo "  ./run_examples.sh openrouter-gpt     - GPT-OSS-20B via OpenRouter"
    echo ""
    echo "LOCAL EXAMPLES:"
    echo "  ./run_examples.sh local-8000         - Local model on port 8000"
    echo "  ./run_examples.sh local-dual         - Local agent (8000) + user (8001)"
    echo "  ./run_examples.sh local-remote       - Remote server at 192.168.1.50:8000"
    echo ""
    echo "OPTIONS:"
    echo "  --env retail|airline              Environment (default: retail)"
    echo "  --task-split test|train|dev       Task split (default: test)"
    echo "  --task-ids N1 N2 N3              Run specific task IDs"
    echo "  --max-concurrency N              Parallel tasks (default: 1)"
    echo "  --num-trials N                   Run N times (default: 1)"
    echo ""
    echo "EXAMPLES:"
    echo "  ./run_examples.sh dashscope-sg-8b --max-concurrency 5"
    echo "  ./run_examples.sh openrouter-32b --task-ids 0 1 2 3 4"
    echo "  ./run_examples.sh local-8000 --env airline --num-trials 3"
    exit 0
fi

# Parse options
COMMAND=$1
shift

while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            ENV="$2"
            shift 2
            ;;
        --task-split)
            TASK_SPLIT="$2"
            shift 2
            ;;
        --task-ids)
            shift
            TASK_IDS=""
            while [[ $# -gt 0 && $1 != --* ]]; do
                TASK_IDS="$TASK_IDS $1"
                shift
            done
            ;;
        --max-concurrency)
            MAX_CONCURRENCY="$2"
            shift 2
            ;;
        --num-trials)
            NUM_TRIALS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Build command
build_cmd() {
    local cmd="python run.py --env $ENV --task-split $TASK_SPLIT --max-concurrency $MAX_CONCURRENCY --num-trials $NUM_TRIALS"
    if [ ! -z "$TASK_IDS" ]; then
        cmd="$cmd --task-ids $TASK_IDS"
    fi
    echo "$cmd"
}

# Execute based on command
case $COMMAND in
    # DASHSCOPE SINGAPORE
    dashscope-sg-4b)
        print_header "Qwen3-4B (DashScope - Singapore)"
        CMD="$(build_cmd) --model qwen3-4b --model-provider dashscope --dashscope-region singapore --user-model openai/gpt-oss-20b --user-model-provider openrouter --agent-strategy tool-calling"
        print_info "Command: $CMD"
        eval $CMD
        ;;
    dashscope-sg-8b)
        print_header "Qwen3-8B (DashScope - Singapore)"
        CMD="$(build_cmd) --model qwen3-8b --model-provider dashscope --dashscope-region singapore --user-model openai/gpt-oss-20b --user-model-provider openrouter --agent-strategy tool-calling"
        print_info "Command: $CMD"
        eval $CMD
        ;;
    dashscope-sg-14b)
        print_header "Qwen3-14B (DashScope - Singapore)"
        CMD="$(build_cmd) --model qwen3-14b --model-provider dashscope --dashscope-region singapore --user-model openai/gpt-oss-20b --user-model-provider openrouter --agent-strategy react"
        print_info "Command: $CMD"
        eval $CMD
        ;;
    dashscope-sg-32b)
        print_header "Qwen3-32B (DashScope - Singapore)"
        CMD="$(build_cmd) --model qwen3-32b --model-provider dashscope --dashscope-region singapore --user-model openai/gpt-oss-20b --user-model-provider openrouter --agent-strategy tool-calling --max-tokens 2000"
        print_info "Command: $CMD"
        eval $CMD
        ;;
    # DASHSCOPE US
    dashscope-us-8b)
        print_header "Qwen3-8B (DashScope - US)"
        CMD="$(build_cmd) --model qwen3-8b --model-provider dashscope --dashscope-region us --user-model openai/gpt-oss-20b --user-model-provider openrouter --agent-strategy tool-calling"
        print_info "Command: $CMD"
        eval $CMD
        ;;
    dashscope-us-32b)
        print_header "Qwen3-32B (DashScope - US)"
        CMD="$(build_cmd) --model qwen3-32b --model-provider dashscope --dashscope-region us --user-model openai/gpt-oss-20b --user-model-provider openrouter --agent-strategy tool-calling --max-tokens 2000"
        print_info "Command: $CMD"
        eval $CMD
        ;;
    # OPENROUTER
    openrouter-4b)
        print_header "Qwen3-4B (OpenRouter)"
        CMD="$(build_cmd) --model qwen/qwen3-4b --model-provider openrouter --user-model openai/gpt-oss-20b --user-model-provider openrouter --agent-strategy tool-calling"
        print_info "Command: $CMD"
        eval $CMD
        ;;
    openrouter-8b)
        print_header "Qwen3-8B (OpenRouter)"
        CMD="$(build_cmd) --model qwen/qwen3-8b --model-provider openrouter --user-model openai/gpt-oss-20b --user-model-provider openrouter --agent-strategy tool-calling"
        print_info "Command: $CMD"
        eval $CMD
        ;;
    openrouter-14b)
        print_header "Qwen3-14B (OpenRouter)"
        CMD="$(build_cmd) --model qwen/qwen3-14b --model-provider openrouter --user-model openai/gpt-oss-20b --user-model-provider openrouter --agent-strategy react"
        print_info "Command: $CMD"
        eval $CMD
        ;;
    openrouter-32b)
        print_header "Qwen3-32B (OpenRouter)"
        CMD="$(build_cmd) --model qwen/qwen3-32b --model-provider openrouter --user-model openai/gpt-oss-20b --user-model-provider openrouter --agent-strategy tool-calling --max-tokens 2000"
        print_info "Command: $CMD"
        eval $CMD
        ;;
    openrouter-gpt)
        print_header "GPT-OSS-20B (OpenRouter)"
        CMD="$(build_cmd) --model openai/gpt-oss-20b --model-provider openrouter --user-model openai/gpt-oss-20b --user-model-provider openrouter --agent-strategy tool-calling"
        print_info "Command: $CMD"
        eval $CMD
        ;;
    # LOCAL
    local-8000)
        print_header "Local Model on Port 8000"
        CMD="$(build_cmd) --model qwen3-8b --model-provider local --model-base-url http://localhost:8000/v1 --model-api-key dummy-key --user-model openai/gpt-oss-20b --user-model-provider openrouter --agent-strategy tool-calling"
        print_info "Command: $CMD"
        eval $CMD
        ;;
    local-dual)
        print_header "Local Dual (Agent 8000, User 8001)"
        CMD="$(build_cmd) --model qwen3-8b --model-provider local --model-base-url http://localhost:8000/v1 --model-api-key dummy-key --user-model qwen3-4b --user-model-provider local --user-model-base-url http://localhost:8001/v1 --user-model-api-key dummy-key --agent-strategy tool-calling"
        print_info "Command: $CMD"
        eval $CMD
        ;;
    local-remote)
        print_header "Remote Server at 192.168.1.50:8000"
        CMD="$(build_cmd) --model qwen3-14b --model-provider local --model-base-url http://192.168.1.50:8000/v1 --model-api-key dummy-key --user-model openai/gpt-oss-20b --user-model-provider openrouter --agent-strategy react"
        print_info "Command: $CMD"
        eval $CMD
        ;;
    *)
        echo "Unknown command: $COMMAND"
        echo "Run './run_examples.sh' without arguments to see all available commands"
        exit 1
        ;;
esac

print_success "Benchmark completed!"
echo "Check results/ directory for output files"
