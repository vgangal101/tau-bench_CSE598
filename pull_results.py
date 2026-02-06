#!/usr/bin/env python3
"""
Pull experiment result files from SOL cluster to local Downloads folder.
Uses scp to transfer SLURM output files, logs, and result JSONs.
"""

import subprocess
import os
import sys
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────
SOL_USER = "hehernan"
SOL_HOST = "sol.asu.edu"
REMOTE_BASE = f"/scratch/{SOL_USER}/tau-bench-project/tau-bench_CSE598"
LOCAL_BASE = Path.home() / "Downloads" / "sol_results"

# ── Experiment list from track.txt ─────────────────────────────────────────
# Format: (job_id, model_size, env, strategy)
# job_id=None means job ID is unknown yet (marked "new" in track.txt)
EXPERIMENTS = [
    ("46703271", "32b", "retail",  "act"),
    ("46703281", "32b", "airline", "act"),
    ("46703282", "32b", "airline", "react"),
    ("46703283", "32b", "airline", "tool-calling"),
    ("46703299", "14b", "retail",  "act"),
    ("46703278", "14b", "airline", "react"),
    ("46703279", "14b", "airline", "tool-calling"),
    ("46703280", "14b", "airline", "act"),
    ("46703298", "8b",  "retail",  "act"),
    (None,       "8b",  "airline", "act"),
    (None,       "8b",  "airline", "react"),
    (None,       "8b",  "airline", "tool-calling"),
    ("46703272", "4b",  "airline", "act"),
    ("46703273", "4b",  "airline", "react"),
    ("46703274", "4b",  "airline", "tool-calling"),
    ("46703297", "4b",  "retail",  "act"),
]


def scp(remote_path: str, local_path: Path, recursive: bool = False) -> bool:
    """Run scp and return True on success."""
    local_path.mkdir(parents=True, exist_ok=True)
    cmd = ["scp", "-o", "ConnectTimeout=10"]
    if recursive:
        cmd.append("-r")
    cmd.append(f"{SOL_USER}@{SOL_HOST}:{remote_path}")
    cmd.append(str(local_path) + "/")

    print(f"  scp {'(dir) ' if recursive else ''}{remote_path}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        if "No such file" in stderr or "not a regular file" in stderr:
            print(f"    ⚠ Not found (skipped)")
        else:
            print(f"    ✗ Error: {stderr}")
        return False
    print(f"    ✓ Downloaded")
    return True


def pull_slurm_outputs(job_id: str, model: str, env: str, strategy: str):
    """Pull the SLURM .out and .err files from the repo root."""
    dest = LOCAL_BASE / "slurm_outputs" / f"{model}_{env}_{strategy}"
    # SLURM output pattern: {model}-{env}-{strategy}-tau-gaudi_{job_id}.{out,err}
    slug = f"{model}-{env}-{strategy}-tau-gaudi"
    for ext in ("out", "err"):
        remote = f"{REMOTE_BASE}/{slug}_{job_id}.{ext}"
        scp(remote, dest)


def pull_logs(job_id: str, model: str, env: str, strategy: str):
    """Pull vLLM server logs and experiment logs from the logs/ directory."""
    dest = LOCAL_BASE / "logs" / f"{model}_{env}_{strategy}"
    log_dir = f"{REMOTE_BASE}/SOL_env/{model}_run/logs"

    # Main experiment log
    slug = f"tau-gaudi-{model}-{env}-{strategy}"
    for ext in ("out", "err"):
        remote = f"{log_dir}/{slug}_{job_id}.{ext}"
        scp(remote, dest)

    # vLLM server logs
    for server_type in ("user_32b", f"agent_{model}"):
        remote = f"{log_dir}/gaudi_vllm_{server_type}_{job_id}.log"
        scp(remote, dest)


def pull_results(model: str, env: str, strategy: str):
    """Pull the results_gaudi JSON files."""
    dest = LOCAL_BASE / "results" / f"{model}_{env}_{strategy}"
    remote_dir = f"{REMOTE_BASE}/SOL_env/{model}_run/results_gaudi/{env}/{strategy}"
    scp(remote_dir, dest, recursive=True)


def main():
    # Allow filtering by model size from command line
    filter_model = None
    filter_env = None
    skip_unknown = False

    args = sys.argv[1:]
    for arg in args:
        if arg in ("32b", "14b", "8b", "4b"):
            filter_model = arg
        elif arg in ("retail", "airline"):
            filter_env = arg
        elif arg == "--skip-unknown":
            skip_unknown = True
        elif arg in ("-h", "--help"):
            print("Usage: python pull_results.py [model] [env] [--skip-unknown]")
            print()
            print("Options:")
            print("  model           Filter by model size: 32b, 14b, 8b, 4b")
            print("  env             Filter by environment: retail, airline")
            print("  --skip-unknown  Skip experiments with unknown job IDs")
            print()
            print("Examples:")
            print("  python pull_results.py              # Pull everything")
            print("  python pull_results.py 4b           # Pull only 4B experiments")
            print("  python pull_results.py 8b airline   # Pull only 8B airline")
            print("  python pull_results.py --skip-unknown")
            return

    print(f"╔══════════════════════════════════════════════╗")
    print(f"║  SOL Results Puller                          ║")
    print(f"║  {SOL_USER}@{SOL_HOST}                      ║")
    print(f"║  → {LOCAL_BASE}  ║")
    print(f"╚══════════════════════════════════════════════╝")
    print()

    LOCAL_BASE.mkdir(parents=True, exist_ok=True)

    total = 0
    skipped = 0

    for job_id, model, env, strategy in EXPERIMENTS:
        # Apply filters
        if filter_model and model != filter_model:
            continue
        if filter_env and env != filter_env:
            continue

        tag = f"[{model.upper():>3s} | {env:<7s} | {strategy:<13s}]"

        if job_id is None:
            if skip_unknown:
                print(f"\n{tag} SKIPPED (no job ID)")
                skipped += 1
                continue
            else:
                print(f"\n{tag} ⚠ No job ID — pulling results directory only")
                pull_results(model, env, strategy)
                total += 1
                continue

        print(f"\n{tag} Job {job_id}")
        print(f"  --- SLURM outputs ---")
        pull_slurm_outputs(job_id, model, env, strategy)
        print(f"  --- Experiment logs ---")
        pull_logs(job_id, model, env, strategy)
        print(f"  --- Results ---")
        pull_results(model, env, strategy)
        total += 1

    print(f"\n{'='*50}")
    print(f"Done. Processed {total} experiments, skipped {skipped}.")
    print(f"Files saved to: {LOCAL_BASE}")


if __name__ == "__main__":
    main()
