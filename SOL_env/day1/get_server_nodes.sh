#!/bin/bash

# Get the job IDs for the most recent submissions (you might need to adjust logic if you have many jobs)
# This assumes tasks are named specific things
USER_JOB=$(squeue --me --name=vllm-gpt-oss-20b --noheader --format=%i | head -n 1)
AGENT_JOB=$(squeue --me --name=vllm-4b --noheader --format=%i | head -n 1)

if [ -z "$USER_JOB" ]; then
    echo "User server job (vllm-gpt-oss-20b) not found running."
else
    USER_NODE=$(squeue -j $USER_JOB --noheader --format=%N)
    echo "USER server is on node: $USER_NODE (Job $USER_JOB)"
fi

if [ -z "$AGENT_JOB" ]; then
    echo "Agent server job (vllm-4b) not found running."
else
    AGENT_NODE_4B=$(squeue -j $AGENT_JOB --noheader --format=%N)
    echo "AGENT_4B server is on node: $AGENT_NODE_4B (Job $AGENT_JOB)"
fi

echo ""
echo "To export these variables, run:"
if [ ! -z "$USER_NODE" ]; then
    echo "export USER_NODE=$USER_NODE"
fi
if [ ! -z "$AGENT_NODE_4B" ]; then
    echo "export AGENT_NODE_4B=$AGENT_NODE_4B"
fi
