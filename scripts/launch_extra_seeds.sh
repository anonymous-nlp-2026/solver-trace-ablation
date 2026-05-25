#!/bin/bash
# Extra seeds (s45, s46) for 3 conditions: exec-only, ISR β=0.1, Random β=0.1
# All use data/train_500_seed42.jsonl (C001), 500 steps
# GPU assignment: set CUDA_VISIBLE_DEVICES before running each command (C002/C005)
# Do NOT run all 6 at once — schedule 2 at a time (one per GPU)

ENV_ACTIVATE="source /root/miniconda3/etc/profile.d/conda.sh && conda activate base"
PROJECT_DIR="/root/solver-trace-ablation"
DATA="data/train_500_seed42.jsonl"
STEPS=500

# --- Exec-only ---

# [1] Exec-only s45  (assign GPU via: export CUDA_VISIBLE_DEVICES=0 or 1)
# export CUDA_VISIBLE_DEVICES=X
# bash -c "$ENV_ACTIVATE && cd $PROJECT_DIR && python3 scripts/train_grpo.py \
#   --config configs/grpo_exec_only.yaml \
#   --data_path $DATA --num_steps $STEPS --seed 45 \
#   --output_dir /root/autodl-tmp/outputs/extra_exec_only_s45"

# [2] Exec-only s46
# export CUDA_VISIBLE_DEVICES=X
# bash -c "$ENV_ACTIVATE && cd $PROJECT_DIR && python3 scripts/train_grpo.py \
#   --config configs/grpo_exec_only.yaml \
#   --data_path $DATA --num_steps $STEPS --seed 46 \
#   --output_dir /root/autodl-tmp/outputs/extra_exec_only_s46"

# --- ISR β=0.1 (additive_bonus) ---

# [3] ISR β=0.1 s45
# export CUDA_VISIBLE_DEVICES=X
# bash -c "$ENV_ACTIVATE && cd $PROJECT_DIR && python3 scripts/train_grpo.py \
#   --config configs/grpo_additive_bonus.yaml \
#   --data_path $DATA --num_steps $STEPS --seed 45 \
#   --beta 0.1 --reward_mode additive_bonus \
#   --output_dir /root/autodl-tmp/outputs/extra_isr_b01_s45"

# [4] ISR β=0.1 s46
# export CUDA_VISIBLE_DEVICES=X
# bash -c "$ENV_ACTIVATE && cd $PROJECT_DIR && python3 scripts/train_grpo.py \
#   --config configs/grpo_additive_bonus.yaml \
#   --data_path $DATA --num_steps $STEPS --seed 46 \
#   --beta 0.1 --reward_mode additive_bonus \
#   --output_dir /root/autodl-tmp/outputs/extra_isr_b01_s46"

# --- Random β=0.1 (additive_random) ---

# [5] Random β=0.1 s45
# export CUDA_VISIBLE_DEVICES=X
# bash -c "$ENV_ACTIVATE && cd $PROJECT_DIR && python3 scripts/train_grpo.py \
#   --config configs/grpo_additive_random_b01.yaml \
#   --data_path $DATA --num_steps $STEPS --seed 45 \
#   --beta 0.1 --reward_mode additive_random \
#   --output_dir /root/autodl-tmp/outputs/extra_random_b01_s45"

# [6] Random β=0.1 s46
# export CUDA_VISIBLE_DEVICES=X
# bash -c "$ENV_ACTIVATE && cd $PROJECT_DIR && python3 scripts/train_grpo.py \
#   --config configs/grpo_additive_random_b01.yaml \
#   --data_path $DATA --num_steps $STEPS --seed 46 \
#   --beta 0.1 --reward_mode additive_random \
#   --output_dir /root/autodl-tmp/outputs/extra_random_b01_s46"
