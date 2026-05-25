#!/bin/bash
# GPU Guard: checks target GPU is free before running any GPU workload.
# Usage: ./gpu_guard.sh <gpu_id> <command...>
# Example: ./gpu_guard.sh 0 python3 eval_sta.py --gpu 0 ...
# Exits with code 77 if GPU is occupied.

GPU_ID="${1:?Usage: gpu_guard.sh <gpu_id> <command...>}"
shift

# Check if any process is using the target GPU
PROCS=$(nvidia-smi --id=$GPU_ID --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null)
if [ -n "$PROCS" ]; then
    echo "ERROR: GPU $GPU_ID is occupied by:"
    echo "$PROCS"
    echo "Aborting to prevent OOM. Kill the occupying process first."
    exit 77
fi

echo "GPU $GPU_ID is free. Proceeding..."
exec "$@"
