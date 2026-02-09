#!/usr/bin/env python3
"""
run_with_vllm.py - Launches two vLLM servers (user simulator + agent),
waits for them to be ready, then runs the tau-bench benchmark exactly
as run.py does.

Usage with YAML config:
    python run_with_vllm.py --config config.yaml

Usage with CLI overrides:
    python run_with_vllm.py --config config.yaml --benchmark.env airline

Generate a default config:
    python run_with_vllm.py --print_config > config.yaml
"""

import os
import sys
import time
import signal
import atexit
import subprocess
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import List, Optional

from jsonargparse import CLI

from tau_bench.types import RunConfig
from tau_bench.run import run


# ── Dataclass Configuration ──


@dataclass
class VllmServerConfig:
    """Configuration for a single vLLM server instance."""
    model: str = "Qwen/Qwen3-32B"
    port: int = 8000
    max_model_len: int = 32768
    gpu_memory_utilization: float = 0.90
    tensor_parallel_size: int = 1
    quantization: Optional[str] = None
    enforce_eager: bool = True
    trust_remote_code: bool = True
    extra_args: Optional[List[str]] = None


@dataclass
class VllmConfig:
    """Configuration for both vLLM servers."""
    user_server: VllmServerConfig = field(default_factory=lambda: VllmServerConfig(
        model="Qwen/Qwen3-32B",
        port=8000,
    ))
    agent_server: VllmServerConfig = field(default_factory=lambda: VllmServerConfig(
        model="Qwen/Qwen3-8B",
        port=8001,
    ))
    tool_call_parser: str = "hermes"
    log_dir: str = "vllm_logs"
    server_ready_timeout: int = 600


@dataclass
class BenchmarkConfig:
    """Configuration for the tau-bench benchmark run (mirrors run.py)."""
    env: str = "retail"
    agent_strategy: str = "tool-calling"
    num_trials: int = 1
    temperature: float = 0.0
    task_split: str = "test"
    start_index: int = 0
    end_index: int = -1
    task_ids: Optional[List[int]] = None
    log_dir: str = "results"
    max_concurrency: int = 1
    seed: int = 10
    shuffle: int = 0
    user_strategy: str = "llm"
    few_shot_displays_path: Optional[str] = None


# ── vLLM Server Management ──

_vllm_processes: list = []


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
    """Check if a vLLM server is healthy via /health endpoint."""
    try:
        req = urllib.request.Request(f"{url}/health", method="GET")
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except (urllib.error.URLError, urllib.error.HTTPError, OSError):
        return False


def wait_for_server(url, name, max_wait_seconds=600, poll_interval=10):
    """Wait for a vLLM server to become healthy."""
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
    server_cfg: VllmServerConfig,
    name: str,
    log_file: str,
    enable_tool_choice: bool = False,
    tool_call_parser: str = "hermes",
):
    """Start a vLLM server as a subprocess."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", server_cfg.model,
        "--host", "0.0.0.0",
        "--port", str(server_cfg.port),
        "--tensor-parallel-size", str(server_cfg.tensor_parallel_size),
        "--gpu-memory-utilization", str(server_cfg.gpu_memory_utilization),
        "--max-model-len", str(server_cfg.max_model_len),
        "--disable-log-requests",
    ]

    if server_cfg.trust_remote_code:
        cmd.append("--trust-remote-code")
    if server_cfg.enforce_eager:
        cmd.append("--enforce-eager")
    if server_cfg.quantization:
        cmd.extend(["--quantization", server_cfg.quantization])
    if enable_tool_choice:
        cmd.extend(["--enable-auto-tool-choice", "--tool-call-parser", tool_call_parser])
    if server_cfg.extra_args:
        cmd.extend(server_cfg.extra_args)

    env = os.environ.copy()
    env.setdefault("VLLM_USE_V1", "0")

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    log_fh = open(log_file, "w")

    print(f"Starting {name} vLLM server: {server_cfg.model} on port {server_cfg.port}")
    print(f"  Log file: {log_file}")
    print(f"  Command: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        env=env,
    )

    _vllm_processes.append(proc)
    print(f"  PID: {proc.pid}")
    return proc


# ── Main Entry Point ──


def main(vllm: VllmConfig = VllmConfig(), benchmark: BenchmarkConfig = BenchmarkConfig()):
    """Launch vLLM servers and run tau-bench experiments.

    Args:
        vllm: Configuration for the two vLLM servers.
        benchmark: Configuration for the tau-bench benchmark run.
    """
    # Register cleanup handlers
    atexit.register(cleanup)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("=" * 60)
    print("  tau-bench with vLLM servers")
    print("=" * 60)
    print(f"  User model:  {vllm.user_server.model} (port {vllm.user_server.port})")
    print(f"  Agent model: {vllm.agent_server.model} (port {vllm.agent_server.port})")
    print(f"  Environment: {benchmark.env}")
    print(f"  Strategy:    {benchmark.agent_strategy}")
    print(f"  Trials:      {benchmark.num_trials}")
    print(f"  Concurrency: {benchmark.max_concurrency}")
    print("=" * 60)
    print()

    # ── Step 1: Start User Simulator vLLM server ──
    user_log = os.path.join(vllm.log_dir, "user_server.log")
    user_proc = start_vllm_server(
        server_cfg=vllm.user_server,
        name="User Simulator",
        log_file=user_log,
        enable_tool_choice=False,
    )

    time.sleep(2)

    # ── Step 2: Start Agent vLLM server ──
    agent_log = os.path.join(vllm.log_dir, "agent_server.log")
    agent_proc = start_vllm_server(
        server_cfg=vllm.agent_server,
        name="Agent",
        log_file=agent_log,
        enable_tool_choice=True,
        tool_call_parser=vllm.tool_call_parser,
    )

    print()

    # ── Step 3: Wait for both servers to be ready ──
    user_url = f"http://localhost:{vllm.user_server.port}"
    agent_url = f"http://localhost:{vllm.agent_server.port}"

    if not wait_for_server(user_url, "User Simulator", max_wait_seconds=vllm.server_ready_timeout):
        if user_proc.poll() is not None:
            print(f"User Simulator process exited with code {user_proc.returncode}")
        print(f"Check log file: {user_log}")
        cleanup()
        sys.exit(1)

    if not wait_for_server(agent_url, "Agent", max_wait_seconds=vllm.server_ready_timeout):
        if agent_proc.poll() is not None:
            print(f"Agent process exited with code {agent_proc.returncode}")
        print(f"Check log file: {agent_log}")
        cleanup()
        sys.exit(1)

    print()
    print("Both vLLM servers are ready!")
    print()

    # ── Step 4: Run the benchmark ──
    os.environ.setdefault("OPENAI_API_KEY", "dummy")

    config = RunConfig(
        model_provider="openai",
        user_model_provider="openai",
        model=vllm.agent_server.model,
        user_model=vllm.user_server.model,
        num_trials=benchmark.num_trials,
        env=benchmark.env,
        agent_strategy=benchmark.agent_strategy,
        temperature=benchmark.temperature,
        task_split=benchmark.task_split,
        start_index=benchmark.start_index,
        end_index=benchmark.end_index,
        task_ids=benchmark.task_ids,
        log_dir=benchmark.log_dir,
        max_concurrency=benchmark.max_concurrency,
        seed=benchmark.seed,
        shuffle=benchmark.shuffle,
        user_strategy=benchmark.user_strategy,
        few_shot_displays_path=benchmark.few_shot_displays_path,
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
    CLI(main, as_positional=False)