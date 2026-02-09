#!/usr/bin/env python3
"""
run_with_vllm.py - Launches two vLLM servers (user simulator + agent),
waits for them to be ready, then runs the tau-bench benchmark exactly
as run.py does.

Usage:
    python run_with_vllm.py \
        --user-model Qwen/Qwen3-32B \
        --model Qwen/Qwen3-8B \
        --env retail \
        --agent-strategy tool-calling \
        --max-concurrency 20 \
        --num-trials 5

    # With quantization:
    python run_with_vllm.py \
        --user-model zankich/Qwen3-32B-INT8 \
        --model Qwen/Qwen3-8B \
        --quantization gptq \
        --user-quantization gptq \
        --max-model-len 50000

    # Custom ports (e.g. if defaults are in use):
    python run_with_vllm.py \
        --user-model Qwen/Qwen3-32B \
        --model Qwen/Qwen3-4B \
        --user-port 8000 \
        --agent-port 8001
"""

import os
import sys
import time
import signal
import atexit
import argparse
import subprocess
import urllib.request
import urllib.error

from tau_bench.types import RunConfig
from tau_bench.run import run
from litellm import provider_list
from tau_bench.envs.user import UserStrategy


# Global list of vLLM processes for cleanup
_vllm_processes = []


def cleanup():
    """Terminate all vLLM server processes."""
    for proc in _vllm_processes:
        if proc.poll() is None:
            print(f"Shutting down vLLM server (PID {proc.pid})...")
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print(f"Force killing vLLM server (PID {proc.pid})...")
                proc.kill()
                proc.wait()
    _vllm_processes.clear()


def signal_handler(signum, frame):
    """Handle signals by cleaning up and exiting."""
    print(f"\nReceived signal {signum}, shutting down...")
    cleanup()
    sys.exit(1)


def check_server_health(url, timeout=5):
    """Check if a vLLM server is healthy by hitting its /health endpoint."""
    try:
        req = urllib.request.Request(f"{url}/health", method="GET")
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except (urllib.error.URLError, urllib.error.HTTPError, OSError):
        return False


def wait_for_server(url, name, max_wait_seconds=600, poll_interval=10):
    """Wait for a vLLM server to become healthy.

    Args:
        url: Base URL of the server (e.g. http://localhost:8000)
        name: Human-readable name for logging
        max_wait_seconds: Maximum time to wait before giving up
        poll_interval: Seconds between health checks

    Returns:
        True if server became healthy, False if timed out
    """
    print(f"Waiting for {name} ({url}) to be ready...", end="", flush=True)
    elapsed = 0
    while elapsed < max_wait_seconds:
        if check_server_health(url):
            print(f" Ready! ({elapsed}s)")
            return True
        print(".", end="", flush=True)
        time.sleep(poll_interval)
        elapsed += poll_interval
    print(f" FAILED after {max_wait_seconds}s!")
    return False


def start_vllm_server(
    model,
    port,
    name,
    log_file=None,
    gpu_memory_utilization=0.90,
    max_model_len=32768,
    tensor_parallel_size=1,
    quantization=None,
    enable_tool_choice=False,
    tool_call_parser="hermes",
    enforce_eager=True,
    trust_remote_code=True,
    extra_args=None,
):
    """Start a vLLM server as a subprocess.

    Args:
        model: HuggingFace model name/path
        port: Port to serve on
        name: Human-readable name for logging
        log_file: Path to write server logs (None = /dev/null)
        gpu_memory_utilization: Fraction of GPU memory to use
        max_model_len: Maximum context length
        tensor_parallel_size: Number of GPUs for tensor parallelism
        quantization: Quantization method (e.g. "gptq", "awq", None)
        enable_tool_choice: Whether to enable auto tool choice (agent only)
        tool_call_parser: Tool call parser to use when enable_tool_choice=True
        enforce_eager: Use eager mode (no CUDA graphs)
        trust_remote_code: Trust remote code from HuggingFace
        extra_args: Additional CLI args to pass to vllm serve

    Returns:
        subprocess.Popen process handle
    """
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-model-len", str(max_model_len),
        "--disable-log-requests",
    ]

    if trust_remote_code:
        cmd.append("--trust-remote-code")
    if enforce_eager:
        cmd.append("--enforce-eager")
    if quantization:
        cmd.extend(["--quantization", quantization])
    if enable_tool_choice:
        cmd.extend(["--enable-auto-tool-choice", "--tool-call-parser", tool_call_parser])
    if extra_args:
        cmd.extend(extra_args)

    env = os.environ.copy()
    env.setdefault("VLLM_USE_V1", "0")

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        log_fh = open(log_file, "w")
        print(f"Starting {name} vLLM server: {model} on port {port}")
        print(f"  Log file: {log_file}")
    else:
        log_fh = subprocess.DEVNULL
        print(f"Starting {name} vLLM server: {model} on port {port}")

    print(f"  Command: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        stdout=log_fh if log_file else subprocess.DEVNULL,
        stderr=subprocess.STDOUT if log_file else subprocess.DEVNULL,
        env=env,
    )

    _vllm_processes.append(proc)
    print(f"  PID: {proc.pid}")
    return proc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch vLLM servers and run tau-bench experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── vLLM server configuration ──
    vllm_group = parser.add_argument_group("vLLM Server Configuration")
    vllm_group.add_argument(
        "--user-port", type=int, default=8000,
        help="Port for the user simulator vLLM server (default: 8000)",
    )
    vllm_group.add_argument(
        "--agent-port", type=int, default=8001,
        help="Port for the agent vLLM server (default: 8001)",
    )
    vllm_group.add_argument(
        "--gpu-memory-utilization", type=float, default=0.90,
        help="Fraction of GPU memory for vLLM to use (default: 0.90)",
    )
    vllm_group.add_argument(
        "--max-model-len", type=int, default=32768,
        help="Maximum context length for both vLLM servers (default: 32768)",
    )
    vllm_group.add_argument(
        "--user-max-model-len", type=int, default=None,
        help="Override max context length for user server (defaults to --max-model-len)",
    )
    vllm_group.add_argument(
        "--agent-max-model-len", type=int, default=None,
        help="Override max context length for agent server (defaults to --max-model-len)",
    )
    vllm_group.add_argument(
        "--tensor-parallel-size", type=int, default=1,
        help="Number of GPUs for tensor parallelism (default: 1)",
    )
    vllm_group.add_argument(
        "--quantization", type=str, default=None,
        help="Quantization method for agent model (e.g. gptq, awq)",
    )
    vllm_group.add_argument(
        "--user-quantization", type=str, default=None,
        help="Quantization method for user model (e.g. gptq, awq)",
    )
    vllm_group.add_argument(
        "--tool-call-parser", type=str, default="hermes",
        help="Tool call parser for agent vLLM server (default: hermes)",
    )
    vllm_group.add_argument(
        "--no-enforce-eager", action="store_true",
        help="Disable --enforce-eager flag for vLLM servers",
    )
    vllm_group.add_argument(
        "--vllm-log-dir", type=str, default="vllm_logs",
        help="Directory for vLLM server log files (default: vllm_logs)",
    )
    vllm_group.add_argument(
        "--server-ready-timeout", type=int, default=600,
        help="Max seconds to wait for each vLLM server to start (default: 600)",
    )
    vllm_group.add_argument(
        "--extra-user-vllm-args", type=str, nargs="*", default=None,
        help="Extra CLI arguments to pass to the user vLLM server",
    )
    vllm_group.add_argument(
        "--extra-agent-vllm-args", type=str, nargs="*", default=None,
        help="Extra CLI arguments to pass to the agent vLLM server",
    )

    # ── Benchmark configuration (mirrors run.py) ──
    bench_group = parser.add_argument_group("Benchmark Configuration")
    bench_group.add_argument("--num-trials", type=int, default=1)
    bench_group.add_argument(
        "--env", type=str, choices=["retail", "airline"], default="retail",
    )
    bench_group.add_argument(
        "--model", type=str, required=True,
        help="Agent model name (HuggingFace model ID, also used as vLLM model)",
    )
    bench_group.add_argument(
        "--user-model", type=str, default="Qwen/Qwen3-32B",
        help="User simulator model name (HuggingFace model ID)",
    )
    bench_group.add_argument(
        "--agent-strategy", type=str, default="tool-calling",
        choices=["tool-calling", "act", "react", "few-shot"],
    )
    bench_group.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature for the agent model",
    )
    bench_group.add_argument(
        "--task-split", type=str, default="test",
        choices=["train", "test", "dev"],
    )
    bench_group.add_argument("--start-index", type=int, default=0)
    bench_group.add_argument("--end-index", type=int, default=-1)
    bench_group.add_argument(
        "--task-ids", type=int, nargs="+",
        help="(Optional) run only the tasks with the given IDs",
    )
    bench_group.add_argument("--log-dir", type=str, default="results")
    bench_group.add_argument(
        "--max-concurrency", type=int, default=1,
        help="Number of tasks to run in parallel",
    )
    bench_group.add_argument("--seed", type=int, default=10)
    bench_group.add_argument("--shuffle", type=int, default=0)
    bench_group.add_argument(
        "--user-strategy", type=str, default="llm",
        choices=[item.value for item in UserStrategy],
    )
    bench_group.add_argument(
        "--few-shot-displays-path", type=str,
        help="Path to a jsonlines file containing few-shot displays",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Register cleanup handlers
    atexit.register(cleanup)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    user_max_model_len = args.user_max_model_len or args.max_model_len
    agent_max_model_len = args.agent_max_model_len or args.max_model_len
    enforce_eager = not args.no_enforce_eager

    print("=" * 60)
    print("  tau-bench with vLLM servers")
    print("=" * 60)
    print(f"  User model:  {args.user_model} (port {args.user_port})")
    print(f"  Agent model: {args.model} (port {args.agent_port})")
    print(f"  Environment: {args.env}")
    print(f"  Strategy:    {args.agent_strategy}")
    print(f"  Trials:      {args.num_trials}")
    print(f"  Concurrency: {args.max_concurrency}")
    print("=" * 60)
    print()

    # ── Step 1: Start User Simulator vLLM server ──
    user_log = os.path.join(args.vllm_log_dir, "user_server.log")
    user_proc = start_vllm_server(
        model=args.user_model,
        port=args.user_port,
        name="User Simulator",
        log_file=user_log,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=user_max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        quantization=args.user_quantization,
        enable_tool_choice=False,
        enforce_eager=enforce_eager,
        extra_args=args.extra_user_vllm_args,
    )

    # Brief delay before starting second server
    time.sleep(2)

    # ── Step 2: Start Agent vLLM server ──
    agent_log = os.path.join(args.vllm_log_dir, "agent_server.log")
    agent_proc = start_vllm_server(
        model=args.model,
        port=args.agent_port,
        name="Agent",
        log_file=agent_log,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=agent_max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        quantization=args.quantization,
        enable_tool_choice=True,
        tool_call_parser=args.tool_call_parser,
        enforce_eager=enforce_eager,
        extra_args=args.extra_agent_vllm_args,
    )

    print()

    # ── Step 3: Wait for both servers to be ready ──
    user_url = f"http://localhost:{args.user_port}"
    agent_url = f"http://localhost:{args.agent_port}"

    if not wait_for_server(user_url, "User Simulator", max_wait_seconds=args.server_ready_timeout):
        # Check if process died
        if user_proc.poll() is not None:
            print(f"User Simulator process exited with code {user_proc.returncode}")
        print(f"Check log file: {user_log}")
        cleanup()
        sys.exit(1)

    if not wait_for_server(agent_url, "Agent", max_wait_seconds=args.server_ready_timeout):
        if agent_proc.poll() is not None:
            print(f"Agent process exited with code {agent_proc.returncode}")
        print(f"Check log file: {agent_log}")
        cleanup()
        sys.exit(1)

    print()
    print("Both vLLM servers are ready!")
    print()

    # ── Step 4: Run the benchmark (same as run.py) ──
    # Set dummy API key for vLLM's OpenAI-compatible endpoint
    os.environ.setdefault("OPENAI_API_KEY", "dummy")

    config = RunConfig(
        model_provider="openai",
        user_model_provider="openai",
        model=args.model,
        user_model=args.user_model,
        num_trials=args.num_trials,
        env=args.env,
        agent_strategy=args.agent_strategy,
        temperature=args.temperature,
        task_split=args.task_split,
        start_index=args.start_index,
        end_index=args.end_index,
        task_ids=args.task_ids,
        log_dir=args.log_dir,
        max_concurrency=args.max_concurrency,
        seed=args.seed,
        shuffle=args.shuffle,
        user_strategy=args.user_strategy,
        few_shot_displays_path=args.few_shot_displays_path,
        model_base_url=f"{agent_url}/v1",
        user_model_base_url=f"{user_url}/v1",
    )

    print("=" * 60)
    print("  Starting benchmark run")
    print("=" * 60)
    print(config)
    print()

    try:
        run(config)
    finally:
        print()
        print("Benchmark complete. Shutting down vLLM servers...")
        cleanup()


if __name__ == "__main__":
    main()
