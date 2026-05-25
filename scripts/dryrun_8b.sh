#!/bin/bash
# Wait for eval process to release GPU 0, then run 8B dry-run
set -e
echo "Waiting for GPU 0 to be free..."
while nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | grep -q "GPU-.*"; do
    if nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader | grep -v "768871" | grep -q "[0-9]"; then
        sleep 5
    else
        break
    fi
done

export CUDA_VISIBLE_DEVICES=0
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
cd /root/solver-trace-ablation
echo "GPU 0 free, starting dry-run..."
python3 scripts/train_grpo.py \
    --config configs/grpo_exec_only_8b.yaml \
    --seed 42 --num_steps 2 --num_generations 4 \
    --output_dir /tmp/8b_dryrun_final --no_wandb 2>&1 | tee /tmp/8b_dryrun_final.log
echo "Dry-run complete. Check /tmp/8b_dryrun_final.log"
