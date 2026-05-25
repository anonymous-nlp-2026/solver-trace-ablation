#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0
PYTHON=/root/miniconda3/bin/python3
SCRIPT_PY=/root/solver-trace-ablation/scripts/train_grpo.py
EXEC_CFG=/root/solver-trace-ablation/configs/grpo_prooffol_exec_only.yaml
ISR_CFG=/root/solver-trace-ablation/configs/grpo_prooffol_isr_b01.yaml
OUT=/root/autodl-tmp/outputs
cd /root/solver-trace-ablation

echo "$(date) [D146] exec s42" && $PYTHON $SCRIPT_PY --config $EXEC_CFG --num_steps 500 --seed 42 --output_dir $OUT/d146_pf_exec_s42 --no_wandb
echo "$(date) [D146] exec s43" && $PYTHON $SCRIPT_PY --config $EXEC_CFG --num_steps 500 --seed 43 --output_dir $OUT/d146_pf_exec_s43 --no_wandb
echo "$(date) [D146] exec s44" && $PYTHON $SCRIPT_PY --config $EXEC_CFG --num_steps 500 --seed 44 --output_dir $OUT/d146_pf_exec_s44 --no_wandb
echo "$(date) [D146] isr s42" && $PYTHON $SCRIPT_PY --config $ISR_CFG --num_steps 500 --seed 42 --output_dir $OUT/d146_pf_isr_s42 --no_wandb
echo "$(date) [D146] isr s43" && $PYTHON $SCRIPT_PY --config $ISR_CFG --num_steps 500 --seed 43 --output_dir $OUT/d146_pf_isr_s43 --no_wandb
echo "$(date) [D146] isr s44" && $PYTHON $SCRIPT_PY --config $ISR_CFG --num_steps 500 --seed 44 --output_dir $OUT/d146_pf_isr_s44 --no_wandb
echo "$(date) [D146] ALL DONE"
