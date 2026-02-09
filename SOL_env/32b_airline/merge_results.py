#!/usr/bin/env python3
"""Merge split 32B airline experiment results from multiple parts into a single file.

Protections against failed jobs:
  - --job-ids: Only merge results from specific SLURM job IDs (whitelist)
  - --exclude-jobs: Exclude results from specific failed job IDs (blacklist)
  - Corrupted/truncated JSON files are skipped with warnings
  - Empty result files (0 results) are skipped with warnings
  - Duplicate task_id+trial combos from retried jobs: keeps latest job's results
  - --dry-run: Preview everything before writing

Usage:
    python merge_results.py --strategy react
    python merge_results.py --strategy react --job-ids 46800001 46800002
    python merge_results.py --strategy react --exclude-jobs 46703271
    python merge_results.py --strategy react --dry-run
"""

import argparse
import glob
import json
import os
import re
import sys


def find_part_files(results_dir: str) -> list[str]:
    """Find all part result files in the given directory."""
    pattern = os.path.join(results_dir, "*_part*_batch*_job*.json")
    files = sorted(glob.glob(pattern))
    return files


def extract_part_info(filepath: str) -> dict | None:
    """Extract part number, batch number, and job ID from filename."""
    basename = os.path.basename(filepath)
    match = re.search(r"_part(\d+)_batch(\d+)_job(\d+)\.json$", basename)
    if match:
        return {
            "part": int(match.group(1)),
            "batch": int(match.group(2)),
            "job_id": match.group(3),
        }
    return None


def load_results_safe(filepath: str) -> tuple[list, str | None]:
    """Load a result file with error handling.

    Returns (results_list, error_message).
    error_message is None on success.
    """
    try:
        with open(filepath) as fp:
            data = json.load(fp)
    except json.JSONDecodeError as e:
        return [], f"Corrupted JSON: {e}"
    except OSError as e:
        return [], f"Read error: {e}"

    if isinstance(data, list):
        return data, None
    elif isinstance(data, dict):
        return [data], None
    else:
        return [], f"Unexpected data type: {type(data).__name__}"


def deduplicate_results(results: list) -> tuple[list, int]:
    """Remove duplicate task_id+trial entries, keeping the last occurrence.

    When a job is retried, the newer job's results appear later in the file list
    (higher job ID = later submission). By keeping the last occurrence, we
    automatically prefer the retry over the original failed run.

    Returns (deduplicated_results, num_duplicates_removed).
    """
    seen = {}
    for r in results:
        if not isinstance(r, dict):
            continue
        task_id = r.get("task_id")
        trial = r.get("trial", 0)
        if task_id is not None:
            key = (task_id, trial)
            seen[key] = r  # Last write wins
        else:
            # No task_id, keep as-is with a unique key
            seen[id(r)] = r

    deduped = list(seen.values())
    num_removed = len(results) - len(deduped)
    return deduped, num_removed


def validate_coverage(results: list, total_tasks: int = 50) -> dict:
    """Check which task indices are covered in the results."""
    covered = set()
    for r in results:
        if isinstance(r, dict):
            for key in ("task_id", "task_index", "index"):
                if key in r:
                    try:
                        covered.add(int(r[key]))
                    except (ValueError, TypeError):
                        pass
                    break

    expected = set(range(total_tasks))
    missing = expected - covered
    extra = covered - expected

    return {
        "covered": sorted(covered),
        "missing": sorted(missing),
        "extra": sorted(extra),
        "total_covered": len(covered),
        "total_expected": total_tasks,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Merge split 32B airline experiment part results"
    )
    parser.add_argument(
        "--strategy",
        required=True,
        choices=["act", "react", "tool-calling"],
        help="Strategy to merge results for",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Custom results directory (default: auto-detect from script location)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Custom output filename (default: auto-generated)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be merged without writing",
    )
    parser.add_argument(
        "--total-tasks",
        type=int,
        default=50,
        help="Total expected tasks (default: 50)",
    )
    parser.add_argument(
        "--job-ids",
        nargs="+",
        default=None,
        help="Only include results from these SLURM job IDs (whitelist)",
    )
    parser.add_argument(
        "--exclude-jobs",
        nargs="+",
        default=None,
        help="Exclude results from these SLURM job IDs (blacklist)",
    )
    args = parser.parse_args()

    if args.job_ids and args.exclude_jobs:
        print("ERROR: Cannot use both --job-ids and --exclude-jobs")
        sys.exit(1)

    # Determine results directory
    if args.results_dir:
        results_dir = args.results_dir
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(
            script_dir, "results_gaudi", "airline", args.strategy
        )

    if not os.path.isdir(results_dir):
        print(f"ERROR: Results directory not found: {results_dir}")
        sys.exit(1)

    # Find part files
    all_files = find_part_files(results_dir)
    if not all_files:
        print(f"No part files found in: {results_dir}")
        print(f"  Pattern: *_part*_batch*_job*.json")
        sys.exit(1)

    # Apply job ID filters
    job_whitelist = set(args.job_ids) if args.job_ids else None
    job_blacklist = set(args.exclude_jobs) if args.exclude_jobs else None

    files = []
    skipped_by_filter = []
    for f in all_files:
        info = extract_part_info(f)
        if info is None:
            files.append(f)
            continue
        job_id = info["job_id"]
        if job_whitelist and job_id not in job_whitelist:
            skipped_by_filter.append((f, job_id, "not in --job-ids"))
            continue
        if job_blacklist and job_id in job_blacklist:
            skipped_by_filter.append((f, job_id, "excluded by --exclude-jobs"))
            continue
        files.append(f)

    # Display header
    print(f"Strategy: {args.strategy}")
    print(f"Results dir: {results_dir}")
    print(f"Found {len(all_files)} total part files, {len(files)} after filtering")
    if job_whitelist:
        print(f"Job ID whitelist: {sorted(job_whitelist)}")
    if job_blacklist:
        print(f"Job ID blacklist: {sorted(job_blacklist)}")
    print()

    # Show filtered-out files
    if skipped_by_filter:
        print(f"FILTERED OUT ({len(skipped_by_filter)} files):")
        for f, job_id, reason in skipped_by_filter:
            print(f"  SKIP job {job_id}: {os.path.basename(f)} ({reason})")
        print()

    if not files:
        print("ERROR: No files remaining after filtering")
        sys.exit(1)

    # Load and validate each file
    print(f"Loading {len(files)} files:")
    parts_seen = {}
    all_results = []
    load_errors = []

    for f in files:
        info = extract_part_info(f)
        basename = os.path.basename(f)
        size_kb = os.path.getsize(f) / 1024

        results, error = load_results_safe(f)

        if error:
            load_errors.append((f, error))
            print(f"  SKIP {basename} ({size_kb:.0f}KB) - {error}")
            continue

        if len(results) == 0:
            load_errors.append((f, "Empty file (0 results)"))
            print(f"  SKIP {basename} ({size_kb:.0f}KB) - Empty file (0 results)")
            continue

        if info:
            part = info["part"]
            if part not in parts_seen:
                parts_seen[part] = []
            parts_seen[part].append(f)
            print(
                f"  OK   Part {info['part']}, Batch {info['batch']}, "
                f"Job {info['job_id']}: {len(results)} results ({size_kb:.0f}KB)"
            )
        else:
            print(f"  OK   {basename}: {len(results)} results ({size_kb:.0f}KB)")

        all_results.extend(results)

    print()

    # Report load errors
    if load_errors:
        print(f"WARNING: {len(load_errors)} files skipped due to errors:")
        for f, err in load_errors:
            print(f"  {os.path.basename(f)}: {err}")
        print()

    # Check parts coverage
    print(f"Parts found: {sorted(parts_seen.keys())}")
    expected_parts = set(range(1, 5))
    missing_parts = expected_parts - set(parts_seen.keys())
    if missing_parts:
        print(f"WARNING: Missing parts: {sorted(missing_parts)}")
        print("  Some jobs may still be running or failed.")
    print()

    # Deduplicate (handles retried jobs)
    all_results, num_dupes = deduplicate_results(all_results)
    if num_dupes > 0:
        print(
            f"Deduplicated: removed {num_dupes} duplicate task_id+trial entries "
            f"(kept latest job's results)"
        )
        print()

    print(f"Total results after validation: {len(all_results)}")

    # Validate task coverage
    coverage = validate_coverage(all_results, args.total_tasks)
    if coverage["total_covered"] > 0:
        print(
            f"Task coverage: {coverage['total_covered']}/{coverage['total_expected']}"
        )
        if coverage["missing"]:
            print(f"  Missing tasks: {coverage['missing']}")
        if coverage["extra"]:
            print(f"  Extra tasks: {coverage['extra']}")
    else:
        print("  (Could not determine task coverage from result format)")

    if args.dry_run:
        print()
        print("[DRY RUN] Would merge the above results. No output written.")
        sys.exit(0)

    # Write output
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(
            results_dir,
            f"{args.strategy}-Qwen3-32B-0.0_range_0-{args.total_tasks}_merged.json",
        )

    with open(output_path, "w") as fp:
        json.dump(all_results, fp, indent=2)

    print()
    print(f"Merged output written to: {output_path}")
    print(f"  {len(files) - len(load_errors)} valid files -> {len(all_results)} results")


if __name__ == "__main__":
    main()
