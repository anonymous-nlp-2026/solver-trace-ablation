#!/bin/bash
set -e

echo "[$(date)] Waiting for PID 136339 to finish..."
while kill -0 136339 2>/dev/null; do
    sleep 30
done
echo "[$(date)] PID 136339 finished. Waiting 10s for GPU memory release..."
sleep 10

export CUDA_VISIBLE_DEVICES=1
cd /root/solver-trace-ablation
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base

echo "[$(date)] Starting eval: plan001-r2 (exec-only)"
python3 scripts/eval_sta.py \
    --model_path /root/autodl-tmp/outputs/plan001_exec_only_s42 \
    --data_path data/proofwriter_fol_100.jsonl \
    --output_path /root/autodl-tmp/outputs/eval_sta_plan001_r2.json
echo "[$(date)] plan001-r2 eval done."

echo "[$(date)] Starting eval: plan013-r4 (random reward)"
python3 scripts/eval_sta.py \
    --model_path /root/autodl-tmp/outputs/plan013_random_reward_s42_r4/checkpoint-500 \
    --data_path data/proofwriter_fol_100.jsonl \
    --output_path /root/autodl-tmp/outputs/eval_sta_plan013_r4.json
echo "[$(date)] plan013-r4 eval done."

echo "[$(date)] All evals complete."
