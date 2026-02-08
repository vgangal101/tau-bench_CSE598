#!/usr/bin/env python3
"""Merge split 32B retail experiment results from multiple parts into a single file.

Usage:
    python merge_results.py --strategy react
    python merge_results.py --strategy react --dry-run
    python merge_results.py --strategy act --results-dir /custom/path
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


def extract_part_info(filepath: str) -> dict:
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


def merge_results(files: list[str]) -> list:
    """Load and merge all result files."""
    all_results = []
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
            if isinstance(data, list):
                all_results.extend(data)
            else:
                all_results.append(data)
    return all_results


def validate_coverage(results: list, total_tasks: int = 115) -> dict:
    """Check which task indices are covered in the results."""
    covered = set()
    for r in results:
        if isinstance(r, dict):
            # Try common field names for task index
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
        description="Merge split 32B retail experiment part results"
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
        default=115,
        help="Total expected tasks (default: 115)",
    )
    args = parser.parse_args()

    # Determine results directory
    if args.results_dir:
        results_dir = args.results_dir
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(
            script_dir, "results_gaudi", "retail", args.strategy
        )

    if not os.path.isdir(results_dir):
        print(f"ERROR: Results directory not found: {results_dir}")
        sys.exit(1)

    # Find part files
    files = find_part_files(results_dir)
    if not files:
        print(f"No part files found in: {results_dir}")
        print(f"  Pattern: *_part*_batch*_job*.json")
        sys.exit(1)

    # Display found files
    print(f"Strategy: {args.strategy}")
    print(f"Results dir: {results_dir}")
    print(f"Found {len(files)} part files:")
    print()

    parts_seen = {}
    for f in files:
        info = extract_part_info(f)
        basename = os.path.basename(f)
        size_mb = os.path.getsize(f) / (1024 * 1024)
        if info:
            part = info["part"]
            if part not in parts_seen:
                parts_seen[part] = []
            parts_seen[part].append(f)
            print(f"  Part {info['part']}, Batch {info['batch']}, "
                  f"Job {info['job_id']}: {basename} ({size_mb:.1f}MB)")
        else:
            print(f"  (unknown format): {basename} ({size_mb:.1f}MB)")

    print()
    print(f"Parts found: {sorted(parts_seen.keys())}")
    expected_parts = {1, 2, 3}
    missing_parts = expected_parts - set(parts_seen.keys())
    if missing_parts:
        print(f"WARNING: Missing parts: {sorted(missing_parts)}")
        print("  Some jobs may still be running or failed.")

    if args.dry_run:
        print()
        print("[DRY RUN] Would merge the above files. No output written.")
        sys.exit(0)

    # Merge
    print()
    print("Merging results...")
    all_results = merge_results(files)
    print(f"Total results: {len(all_results)}")

    # Validate coverage
    coverage = validate_coverage(all_results, args.total_tasks)
    if coverage["total_covered"] > 0:
        print(f"Task coverage: {coverage['total_covered']}/{coverage['total_expected']}")
        if coverage["missing"]:
            print(f"  Missing tasks: {coverage['missing']}")
        if coverage["extra"]:
            print(f"  Extra tasks: {coverage['extra']}")
    else:
        print("  (Could not determine task coverage from result format)")

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
    print(f"  {len(files)} files -> {len(all_results)} results")


if __name__ == "__main__":
    main()
