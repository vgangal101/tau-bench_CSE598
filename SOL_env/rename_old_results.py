#!/usr/bin/env python3
"""Rename old result files to match new 1-batch-per-part naming convention.

Old 4B/8B/14B experiments used 2 batches per part. The new split uses 1 batch
per part. This script renames old files so merge_results.py picks them up
correctly alongside new results.

Determines the correct new part number from the task RANGE in the filename
(not the old part/batch labels, which may be inconsistent).

Usage:
    python rename_old_results.py --size 4b --env airline --dry-run
    python rename_old_results.py --size 4b --env airline
    python rename_old_results.py --all --dry-run
    python rename_old_results.py --all
"""

import argparse
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# New part mapping: range_start → new_part_number
# Based on the 1-batch-per-part split in generate_split_scripts.py
AIRLINE_RANGE_TO_PART = {
    0: 1,
    13: 2,
    25: 3,
    38: 4,
}

RETAIL_RANGE_TO_PART = {
    0: 1,
    20: 2,
    40: 3,
    60: 4,
    80: 5,
    100: 6,
}

# Pattern to extract range and part/batch/job from filename
# e.g. react-Qwen3-4B-0.0_range_0-13_user-..._part1_batch2_job46911280.json
PART_BATCH_RE = re.compile(r"^(.+)_part(\d+)_batch(\d+)_job(\d+)(\.json)$")
RANGE_RE = re.compile(r"_range_(\d+)-(\d+)_")


def get_range_to_part(env):
    return AIRLINE_RANGE_TO_PART if env == "airline" else RETAIL_RANGE_TO_PART


def find_closest_start(range_start, range_to_part):
    """Find the closest matching start value in the mapping."""
    starts = sorted(range_to_part.keys())
    best = min(starts, key=lambda s: abs(s - range_start))
    # Allow up to 2 off (e.g., range_0-13 vs config start=0)
    if abs(best - range_start) <= 2:
        return best
    return None


def rename_files(size, env, dry_run=True):
    """Rename old result files for a given size/env combo."""
    results_dir = os.path.join(SCRIPT_DIR, f"{size}_{env}", "results_gaudi", env)

    if not os.path.isdir(results_dir):
        print(f"  No results dir: {results_dir}")
        return 0, 0

    range_to_part = get_range_to_part(env)
    renamed = 0
    skipped = 0

    for strategy in sorted(os.listdir(results_dir)):
        strategy_dir = os.path.join(results_dir, strategy)
        if not os.path.isdir(strategy_dir):
            continue

        for filename in sorted(os.listdir(strategy_dir)):
            if not filename.endswith(".json"):
                continue

            # Skip merged files
            if "_merged" in filename:
                continue

            # Only process files with _part*_batch*_job* suffix
            part_match = PART_BATCH_RE.match(filename)
            if not part_match:
                continue

            prefix = part_match.group(1)
            old_part = int(part_match.group(2))
            old_batch = int(part_match.group(3))
            job_id = part_match.group(4)
            ext = part_match.group(5)

            # Extract range from filename
            range_match = RANGE_RE.search(filename)
            if not range_match:
                print(f"  SKIP (no range): {filename}")
                skipped += 1
                continue

            range_start = int(range_match.group(1))

            # Map range start to new part number
            closest = find_closest_start(range_start, range_to_part)
            if closest is None:
                print(f"  SKIP (unknown range start {range_start}): {filename}")
                skipped += 1
                continue

            new_part = range_to_part[closest]

            # Check if rename is needed
            if old_part == new_part and old_batch == 1:
                continue  # Already correct

            new_filename = f"{prefix}_part{new_part}_batch1_job{job_id}{ext}"
            old_path = os.path.join(strategy_dir, filename)
            new_path = os.path.join(strategy_dir, new_filename)

            if os.path.exists(new_path) and old_path != new_path:
                print(f"  SKIP (target exists): {filename} -> {new_filename}")
                skipped += 1
                continue

            if dry_run:
                print(f"  WOULD RENAME: {filename}")
                print(f"            ->  {new_filename}")
            else:
                os.rename(old_path, new_path)
                print(f"  RENAMED: {filename}")
                print(f"       ->  {new_filename}")
            renamed += 1

    return renamed, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Rename old result files to match new 1-batch-per-part naming"
    )
    parser.add_argument("--size", type=str, choices=["4b", "8b", "14b"],
                        help="Model size to rename")
    parser.add_argument("--env", type=str, choices=["airline", "retail"],
                        help="Environment to rename")
    parser.add_argument("--all", action="store_true",
                        help="Rename all 4B/8B/14B results")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview renames without executing")
    args = parser.parse_args()

    if not args.all and (not args.size or not args.env):
        parser.error("Either --all or both --size and --env are required")

    if args.all:
        combos = [(s, e) for s in ["4b", "8b", "14b"] for e in ["airline", "retail"]]
    else:
        combos = [(args.size, args.env)]

    total_renamed = 0
    total_skipped = 0

    for size, env in combos:
        print(f"\n{'='*60}")
        print(f"=== {size}_{env} ===")
        print(f"{'='*60}")
        renamed, skipped = rename_files(size, env, dry_run=args.dry_run)
        total_renamed += renamed
        total_skipped += skipped
        print(f"  {'Would rename' if args.dry_run else 'Renamed'}: {renamed}, Skipped: {skipped}")

    print(f"\n{'='*60}")
    mode = "DRY RUN" if args.dry_run else "DONE"
    print(f"{mode}: {total_renamed} files {'would be ' if args.dry_run else ''}renamed, {total_skipped} skipped")

    if args.dry_run and total_renamed > 0:
        print("\nRe-run without --dry-run to execute renames.")


if __name__ == "__main__":
    main()
