#!/bin/bash
# Submit all 12 parts of a 32B retail experiment for a given strategy.
# Usage: ./submit_all.sh <strategy>
#   strategy: act, react, or tool-calling
#
# Example:
#   ./submit_all.sh react        # Submit parts 1-12 for react
#   ./submit_all.sh act           # Submit parts 1-12 for act
#   ./submit_all.sh tool-calling  # Submit parts 1-12 for tool-calling

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

STRATEGY="$1"

if [ -z "$STRATEGY" ]; then
    echo "Usage: $0 <strategy>"
    echo "  strategy: act, react, or tool-calling"
    exit 1
fi

if [ "$STRATEGY" != "act" ] && [ "$STRATEGY" != "react" ] && [ "$STRATEGY" != "tool-calling" ]; then
    echo "ERROR: Invalid strategy '$STRATEGY'"
    echo "  Valid strategies: act, react, tool-calling"
    exit 1
fi

echo "=========================================="
echo "=== Submitting 32B Retail ${STRATEGY} ==="
echo "=========================================="
echo ""

JOBS=()
for PART in $(seq 1 12); do
    SCRIPT="${SCRIPT_DIR}/part${PART}_${STRATEGY}.sh"
    if [ ! -f "$SCRIPT" ]; then
        echo "ERROR: Script not found: $SCRIPT"
        exit 1
    fi
    echo "Submitting Part ${PART}: ${SCRIPT}"
    JOB_OUTPUT=$(sbatch "$SCRIPT")
    JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP '\d+$')
    JOBS+=("$JOB_ID")
    echo "  -> Job ID: $JOB_ID"
done

echo ""
echo "=========================================="
echo "All 12 parts submitted successfully!"
echo "=========================================="
echo ""
echo "Strategy: ${STRATEGY}"
echo "Job IDs:"
echo "  Part 1 (tasks 0-9):   ${JOBS[0]}"
echo "  Part 2 (tasks 10-19):   ${JOBS[1]}"
echo "  Part 3 (tasks 20-29):   ${JOBS[2]}"
echo "  Part 4 (tasks 30-39):   ${JOBS[3]}"
echo "  Part 5 (tasks 40-49):   ${JOBS[4]}"
echo "  Part 6 (tasks 50-59):   ${JOBS[5]}"
echo "  Part 7 (tasks 60-69):   ${JOBS[6]}"
echo "  Part 8 (tasks 70-79):   ${JOBS[7]}"
echo "  Part 9 (tasks 80-89):   ${JOBS[8]}"
echo "  Part 10 (tasks 90-99):   ${JOBS[9]}"
echo "  Part 11 (tasks 100-109):   ${JOBS[10]}"
echo "  Part 12 (tasks 110-114):   ${JOBS[11]}"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "After all jobs complete, merge results with:"
echo "  python ${SCRIPT_DIR}/merge_results.py --strategy ${STRATEGY}"
