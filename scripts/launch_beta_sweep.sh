#!/bin/bash
# β sweep: ISR experiments for β ∈ {0.05, 0.2, 0.5} × seeds {42, 43, 44}
# 9 experiments total, ~500 steps each
#
# GPU allocation: assign CUDA_VISIBLE_DEVICES=0 or 1 before running.
# Run at most 2 experiments concurrently (one per GPU).
# DO NOT run this script directly — launch commands individually.

set -euo pipefail

CONDA_INIT="source /root/miniconda3/etc/profile.d/conda.sh && conda activate base"
CD_PROJECT="cd /root/solver-trace-ablation"
COMMON="python3 scripts/train_grpo.py --config configs/grpo_additive_bonus.yaml --data_path data/train_500_seed42.jsonl --num_steps 500 --reward_mode additive_bonus"

# ── β=0.05 ──────────────────────────────────────────────

# β=0.05, seed=42
# bash -c "export CUDA_VISIBLE_DEVICES=0 && ${CONDA_INIT} && ${CD_PROJECT} && ${COMMON} --seed 42 --beta 0.05 --output_dir /root/autodl-tmp/outputs/sweep_isr_b005_s42"

# β=0.05, seed=43
# bash -c "export CUDA_VISIBLE_DEVICES=1 && ${CONDA_INIT} && ${CD_PROJECT} && ${COMMON} --seed 43 --beta 0.05 --output_dir /root/autodl-tmp/outputs/sweep_isr_b005_s43"

# β=0.05, seed=44
# bash -c "export CUDA_VISIBLE_DEVICES=0 && ${CONDA_INIT} && ${CD_PROJECT} && ${COMMON} --seed 44 --beta 0.05 --output_dir /root/autodl-tmp/outputs/sweep_isr_b005_s44"

# ── β=0.2 ───────────────────────────────────────────────

# β=0.2, seed=42
# bash -c "export CUDA_VISIBLE_DEVICES=0 && ${CONDA_INIT} && ${CD_PROJECT} && ${COMMON} --seed 42 --beta 0.2 --output_dir /root/autodl-tmp/outputs/sweep_isr_b02_s42"

# β=0.2, seed=43
# bash -c "export CUDA_VISIBLE_DEVICES=1 && ${CONDA_INIT} && ${CD_PROJECT} && ${COMMON} --seed 43 --beta 0.2 --output_dir /root/autodl-tmp/outputs/sweep_isr_b02_s43"

# β=0.2, seed=44
# bash -c "export CUDA_VISIBLE_DEVICES=0 && ${CONDA_INIT} && ${CD_PROJECT} && ${COMMON} --seed 44 --beta 0.2 --output_dir /root/autodl-tmp/outputs/sweep_isr_b02_s44"

# ── β=0.5 ───────────────────────────────────────────────

# β=0.5, seed=42
# bash -c "export CUDA_VISIBLE_DEVICES=0 && ${CONDA_INIT} && ${CD_PROJECT} && ${COMMON} --seed 42 --beta 0.5 --output_dir /root/autodl-tmp/outputs/sweep_isr_b05_s42"

# β=0.5, seed=43
# bash -c "export CUDA_VISIBLE_DEVICES=1 && ${CONDA_INIT} && ${CD_PROJECT} && ${COMMON} --seed 43 --beta 0.5 --output_dir /root/autodl-tmp/outputs/sweep_isr_b05_s43"

# β=0.5, seed=44
# bash -c "export CUDA_VISIBLE_DEVICES=0 && ${CONDA_INIT} && ${CD_PROJECT} && ${COMMON} --seed 44 --beta 0.5 --output_dir /root/autodl-tmp/outputs/sweep_isr_b05_s44"

echo "All 9 sweep commands listed above. Uncomment and run individually."
