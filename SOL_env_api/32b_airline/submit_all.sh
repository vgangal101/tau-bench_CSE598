#!/bin/bash
# Submit all 8 parts of a 32B airline (API user) experiment for a given strategy.
# Usage: ./submit_all.sh <strategy>
#   strategy: act, react, or tool-calling
#
# Prerequisites:
#   export SOL_API_KEY="your-api-key"  (get from https://voyager.rc.asu.edu/)
#
# Example:
#   ./submit_all.sh react        # Submit parts 1-8 for react
#   ./submit_all.sh act           # Submit parts 1-8 for act
#   ./submit_all.sh tool-calling  # Submit parts 1-8 for tool-calling

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

echo "=========================================================="
echo "=== Submitting 32B Airline ${STRATEGY} (API User) ==="
echo "=========================================================="
echo ""
echo "Agent: Qwen/Qwen3-32B (local vLLM on Gaudi)"
echo "User:  qwen3-235b-a22b-instruct-2507 (SOL API)"
echo ""

JOBS=()
for PART in $(seq 1 8); do
    SCRIPT="${SCRIPT_DIR}/part${PART}_${STRATEGY}.sh"
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
echo "=========================================================="
echo "All 8 parts submitted successfully!"
echo "=========================================================="
echo ""
echo "Strategy: ${STRATEGY}"
echo "Job IDs:"
echo "  Part 1 (tasks 0-6):   ${JOBS[0]}"
echo "  Part 2 (tasks 7-12):   ${JOBS[1]}"
echo "  Part 3 (tasks 13-18):   ${JOBS[2]}"
echo "  Part 4 (tasks 19-24):   ${JOBS[3]}"
echo "  Part 5 (tasks 25-31):   ${JOBS[4]}"
echo "  Part 6 (tasks 32-37):   ${JOBS[5]}"
echo "  Part 7 (tasks 38-43):   ${JOBS[6]}"
echo "  Part 8 (tasks 44-49):   ${JOBS[7]}"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "After all jobs complete, merge results with:"
echo "  python ${SCRIPT_DIR}/merge_results.py --strategy ${STRATEGY}"
