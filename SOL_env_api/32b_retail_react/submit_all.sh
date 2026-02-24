#!/bin/bash
# Submit both parts of the 32B retail react (API user) experiment.
# Usage: ./submit_all.sh
#
# Prerequisites:
#   export SOL_API_KEY="your-api-key"  (get from https://voyager.rc.asu.edu/)
#
# Example:
#   ./submit_all.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check SOL_API_KEY before submitting
if [ -z "$SOL_API_KEY" ]; then
    echo "ERROR: SOL_API_KEY environment variable not set"
    echo ""
    echo "To get an API key:"
    echo "1. Go to https://voyager.rc.asu.edu/"
    echo "2. Navigate to 'LLM Access' tab"
    echo "3. Click 'Create Key'"
    echo "4. Run: export SOL_API_KEY='your-key'"
    echo ""
    exit 1
fi

echo "================================================"
echo "=== Submitting 32B Retail React (API User) ==="
echo "================================================"
echo ""
echo "Agent: Qwen/Qwen3-32B (local vLLM on Gaudi)"
echo "User:  qwen3-235b-a22b-instruct-2507 (SOL API)"
echo ""

JOBS=()
for PART in 1 2; do
    SCRIPT="${SCRIPT_DIR}/part${PART}_react.sh"
    if [ ! -f "$SCRIPT" ]; then
        echo "ERROR: Script not found: $SCRIPT"
        exit 1
    fi
    echo "Submitting Part ${PART}: ${SCRIPT}"
    JOB_OUTPUT=$(sbatch --export=ALL "$SCRIPT")
    JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP '\d+$')
    JOBS+=("$JOB_ID")
    echo "  -> Job ID: $JOB_ID"
done

echo ""
echo "================================================"
echo "Both parts submitted successfully!"
echo "================================================"
echo ""
echo "Job IDs:"
echo "  Part 1 (tasks 0-57):   ${JOBS[0]}"
echo "  Part 2 (tasks 58-114): ${JOBS[1]}"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "After all jobs complete, merge results with:"
echo "  python ${SCRIPT_DIR}/merge_results.py --dry-run"
echo "  python ${SCRIPT_DIR}/merge_results.py"
