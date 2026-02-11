#!/usr/bin/env python3
"""Analyze tau-bench experiment result files and produce a comprehensive summary.

Scans a directory for merged result JSON files, deduplicates when multiple
files exist for the same (strategy, model, env) combination, and reports:
  - Task coverage and trial completeness
  - Success rates
  - pass^k metrics
  - Overall progress toward Phase 1 completion

Usage:
    python analyze_results.py                              # scan ~/Downloads
    python analyze_results.py /path/to/results             # custom directory
    python analyze_results.py --verbose                    # per-file details
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path


# ── Constants ─────────────────────────────────────────────────────────────────
EXPECTED_TASKS = {"retail": 115, "airline": 50}
NUM_TRIALS = 5
STRATEGIES = ["act", "react", "tool-calling"]
MODELS = ["4B", "8B", "14B", "32B"]
ENVS = ["airline", "retail"]


def parse_filename(filename: str) -> dict | None:
    """Extract strategy, model size, and environment from a merged result filename."""
    normalized = filename.replace("-0_0_", "-0.0_")
    pattern = r"^(act|react|tool-calling)-Qwen3-(\d+B)-0\.0_range_0-(\d+)(?:_job\d+)?_merged\.json$"
    match = re.match(pattern, normalized)
    if not match:
        return None
    strategy = match.group(1)
    model_size = match.group(2)
    total_tasks = int(match.group(3))
    if total_tasks == 115:
        env = "retail"
    elif total_tasks == 50:
        env = "airline"
    else:
        return None
    has_job_id = "_job" in filename
    return {
        "strategy": strategy,
        "model": model_size,
        "env": env,
        "total_tasks": total_tasks,
        "has_job_id": has_job_id,
    }


def load_results(filepath: str) -> list[dict]:
    """Load a merged result JSON file."""
    with open(filepath) as fp:
        data = json.load(fp)
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        return [data]
    return []


def compute_pass_k(rewards_by_task: dict[int, list[float]], k: int) -> float | None:
    """Compute pass^k: probability that at least one of k trials succeeds."""
    eligible = []
    for task_id, rewards in rewards_by_task.items():
        n = len(rewards)
        if n < k:
            continue
        s = sum(1 for r in rewards if r >= 1.0)
        if n - s < k:
            eligible.append(1.0)
        else:
            fail_prob = 1.0
            for i in range(k):
                fail_prob *= (n - s - i) / (n - i)
            eligible.append(1.0 - fail_prob)
    if not eligible:
        return None
    return sum(eligible) / len(eligible)


def analyze_file(filepath: str, total_tasks: int) -> dict:
    """Analyze a single result file and return statistics."""
    results = load_results(filepath)

    rewards_by_task: dict[int, list[float]] = defaultdict(list)
    trial_counts: dict[int, set] = defaultdict(set)

    for r in results:
        if not isinstance(r, dict):
            continue
        task_id = r.get("task_id")
        trial = r.get("trial", 0)
        reward = r.get("reward", 0.0)
        if task_id is not None:
            rewards_by_task[int(task_id)].append(reward)
            trial_counts[int(task_id)].add(trial)

    covered_tasks = set(rewards_by_task.keys())
    missing_tasks = sorted(set(range(total_tasks)) - covered_tasks)

    trial_dist = defaultdict(int)
    for task_id in range(total_tasks):
        n_trials = len(trial_counts.get(task_id, set()))
        trial_dist[n_trials] += 1

    complete_tasks = sum(1 for t in range(total_tasks)
                         if len(trial_counts.get(t, set())) >= NUM_TRIALS)

    total_trajectories = len(results)
    successes = sum(1 for r in results if isinstance(r, dict) and r.get("reward", 0) >= 1.0)
    success_rate = successes / total_trajectories if total_trajectories > 0 else 0.0

    traj_lengths = []
    for r in results:
        if isinstance(r, dict) and "traj" in r:
            traj_lengths.append(len(r["traj"]))

    pass_k = {}
    for k in range(1, NUM_TRIALS + 1):
        pk = compute_pass_k(rewards_by_task, k)
        if pk is not None:
            pass_k[k] = pk

    return {
        "total_trajectories": total_trajectories,
        "unique_tasks": len(covered_tasks),
        "total_expected": total_tasks,
        "missing_tasks": missing_tasks,
        "complete_tasks": complete_tasks,
        "expected_trajectories": total_tasks * NUM_TRIALS,
        "missing_trajectories": total_tasks * NUM_TRIALS - total_trajectories,
        "trial_distribution": dict(trial_dist),
        "success_rate": success_rate,
        "successes": successes,
        "failures": total_trajectories - successes,
        "pass_k": pass_k,
        "avg_traj_length": sum(traj_lengths) / len(traj_lengths) if traj_lengths else 0,
        "min_traj_length": min(traj_lengths) if traj_lengths else 0,
        "max_traj_length": max(traj_lengths) if traj_lengths else 0,
        "file_size_mb": os.path.getsize(filepath) / (1024 * 1024),
    }


def find_best_files(directory: str) -> dict[tuple[str, str, str], str]:
    """Find the best file for each (strategy, model, env) combination."""
    candidates: dict[tuple, list] = defaultdict(list)
    for filename in os.listdir(directory):
        if not filename.endswith("_merged.json"):
            continue
        info = parse_filename(filename)
        if info is None:
            continue
        key = (info["strategy"], info["model"], info["env"])
        filepath = os.path.join(directory, filename)
        file_size = os.path.getsize(filepath)
        candidates[key].append({
            "path": filepath,
            "filename": filename,
            "has_job_id": info["has_job_id"],
            "file_size": file_size,
        })
    best = {}
    for key, files in candidates.items():
        files.sort(key=lambda f: (f["has_job_id"], -f["file_size"]))
        best[key] = files[0]["path"]
    return best


def main():
    parser = argparse.ArgumentParser(description="Analyze tau-bench experiment results")
    parser.add_argument(
        "directory",
        nargs="?",
        default=str(Path.home() / "Downloads"),
        help="Directory containing merged result JSON files (default: ~/Downloads)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Show per-file details")
    args = parser.parse_args()

    directory = args.directory
    if not os.path.isdir(directory):
        print(f"ERROR: Directory not found: {directory}")
        sys.exit(1)

    best_files = find_best_files(directory)
    if not best_files:
        print(f"No merged result files found in: {directory}")
        sys.exit(1)

    print("=" * 80)
    print("  tau-Bench Experiment Results Summary")
    print("=" * 80)
    print(f"  Directory: {directory}")
    print(f"  Files found: {len(best_files)} unique (strategy, model, env) combinations")
    print()

    all_stats = {}
    for key in sorted(best_files.keys()):
        strategy, model, env = key
        filepath = best_files[key]
        total_tasks = EXPECTED_TASKS[env]
        stats = analyze_file(filepath, total_tasks)
        all_stats[key] = stats

    # ── Per-Environment Summary Tables ────────────────────────────────────────
    for env in ENVS:
        total_tasks = EXPECTED_TASKS[env]
        env_keys = [k for k in all_stats if k[2] == env]
        if not env_keys:
            continue

        print(f"{'─' * 80}")
        print(f"  {env.upper()} Domain ({total_tasks} tasks)")
        print(f"{'─' * 80}")
        print()

        header = f"{'Strategy':<14s} {'Model':<6s} {'Tasks':<10s} {'Trials':<14s} {'Complete':<10s} {'Success%':<10s} {'pass^1':<8s} {'pass^5':<8s}"
        print(header)
        print("-" * len(header))

        for strategy in STRATEGIES:
            for model in MODELS:
                key = (strategy, model, env)
                if key not in all_stats:
                    print(f"{strategy:<14s} {model:<6s} {'--':^10s} {'--':^14s} {'--':^10s} {'--':^10s} {'--':^8s} {'--':^8s}")
                    continue
                s = all_stats[key]
                tasks_str = f"{s['unique_tasks']}/{s['total_expected']}"
                trials_str = f"{s['total_trajectories']}/{s['expected_trajectories']}"
                complete_str = f"{s['complete_tasks']}/{s['total_expected']}"
                success_str = f"{s['success_rate']:.1%}"
                pk1 = f"{s['pass_k'].get(1, 0):.1%}" if 1 in s["pass_k"] else "--"
                pk5 = f"{s['pass_k'].get(5, 0):.1%}" if 5 in s["pass_k"] else "--"
                print(f"{strategy:<14s} {model:<6s} {tasks_str:<10s} {trials_str:<14s} {complete_str:<10s} {success_str:<10s} {pk1:<8s} {pk5:<8s}")
            print()

    # ── Verbose Per-File Details ──────────────────────────────────────────────
    if args.verbose:
        print(f"\n{'=' * 80}")
        print("  Detailed Per-File Analysis")
        print(f"{'=' * 80}\n")
        for key in sorted(all_stats.keys()):
            strategy, model, env = key
            s = all_stats[key]
            filepath = best_files[key]
            print(f"--- {strategy} / Qwen3-{model} / {env} ---")
            print(f"  File: {os.path.basename(filepath)} ({s['file_size_mb']:.1f} MB)")
            print(f"  Tasks: {s['unique_tasks']}/{s['total_expected']}")
            if s["missing_tasks"]:
                print(f"  Missing tasks: {s['missing_tasks']}")
            print(f"  Trajectories: {s['total_trajectories']}/{s['expected_trajectories']} ({s['missing_trajectories']} missing)")
            print(f"  Complete (5 trials): {s['complete_tasks']}/{s['total_expected']}")
            print(f"  Success rate: {s['successes']}/{s['total_trajectories']} ({s['success_rate']:.1%})")
            print(f"  Trajectory length: min={s['min_traj_length']}, avg={s['avg_traj_length']:.1f}, max={s['max_traj_length']}")
            print(f"  Trial distribution:")
            for n_trials in range(NUM_TRIALS + 1):
                count = s["trial_distribution"].get(n_trials, 0)
                if count > 0:
                    pct = count / s["total_expected"] * 100
                    print(f"    {n_trials} trials: {count} tasks ({pct:.1f}%)")
            if s["pass_k"]:
                pk_str = ", ".join(f"pass^{k}={v:.1%}" for k, v in sorted(s["pass_k"].items()))
                print(f"  {pk_str}")
            print()

    # ── Overall Progress ──────────────────────────────────────────────────────
    print(f"{'=' * 80}")
    print("  Overall Phase 1 Progress")
    print(f"{'=' * 80}\n")

    total_combos = len(STRATEGIES) * len(MODELS) * len(ENVS)
    found_combos = len(all_stats)
    complete_combos = sum(
        1 for s in all_stats.values()
        if s["unique_tasks"] == s["total_expected"]
        and s["complete_tasks"] == s["total_expected"]
    )
    task_coverage_combos = sum(
        1 for s in all_stats.values()
        if s["unique_tasks"] == s["total_expected"]
    )

    print(f"  Required combinations: {total_combos} (3 strategies x 4 models x 2 envs)")
    print(f"  Data files found:      {found_combos}/{total_combos} ({found_combos/total_combos:.0%})")
    print(f"  Full task coverage:    {task_coverage_combos}/{total_combos} (all tasks have >=1 trial)")
    print(f"  Fully complete:        {complete_combos}/{total_combos} (all tasks x all 5 trials)")
    print()

    missing = []
    for strategy in STRATEGIES:
        for model in MODELS:
            for env in ENVS:
                if (strategy, model, env) not in all_stats:
                    missing.append(f"  {strategy:<14s} Qwen3-{model:<4s} {env}")
    if missing:
        print(f"  Missing combinations ({len(missing)}):")
        for m in missing:
            print(m)
        print()

    incomplete = []
    for key, s in sorted(all_stats.items()):
        strategy, model, env = key
        if s["unique_tasks"] < s["total_expected"]:
            incomplete.append(
                f"  {strategy:<14s} Qwen3-{model:<4s} {env:<8s} "
                f"tasks={s['unique_tasks']}/{s['total_expected']}, "
                f"trials={s['total_trajectories']}/{s['expected_trajectories']}"
            )
        elif s["complete_tasks"] < s["total_expected"]:
            incomplete.append(
                f"  {strategy:<14s} Qwen3-{model:<4s} {env:<8s} "
                f"tasks=OK, trials={s['total_trajectories']}/{s['expected_trajectories']} "
                f"(complete={s['complete_tasks']}/{s['total_expected']})"
            )
    if incomplete:
        print(f"  Incomplete combinations ({len(incomplete)}):")
        for ic in incomplete:
            print(ic)
        print()


if __name__ == "__main__":
    main()
