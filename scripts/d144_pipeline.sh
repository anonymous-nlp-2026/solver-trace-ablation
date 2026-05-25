#!/bin/bash
# D144 LLaMA-3.2-3B pipeline: 5 remaining experiments
# exec-only: s43, s44 → ISR β=0.1: s42, s43, s44
# All on cuda:0, serial execution
set -e

export CUDA_VISIBLE_DEVICES=0

PYTHON=/root/miniconda3/bin/python3
SCRIPT=/root/solver-trace-ablation/scripts/train_grpo.py
EXEC_CONFIG=/root/solver-trace-ablation/configs/grpo_llama_exec_only.yaml
ISR_CONFIG=/root/solver-trace-ablation/configs/grpo_llama_additive_bonus_b01.yaml
OUTPUT_BASE=/root/autodl-tmp/outputs

cd /root/solver-trace-ablation

echo "$(date) [D144] Starting exec-only s43"
$PYTHON $SCRIPT --config $EXEC_CONFIG --num_steps 500 --seed 43 --output_dir $OUTPUT_BASE/d144_llama_exec_s43
echo "$(date) [D144] exec-only s43 DONE"

echo "$(date) [D144] Starting exec-only s44"
$PYTHON $SCRIPT --config $EXEC_CONFIG --num_steps 500 --seed 44 --output_dir $OUTPUT_BASE/d144_llama_exec_s44
echo "$(date) [D144] exec-only s44 DONE"

echo "$(date) [D144] Starting ISR s42"
$PYTHON $SCRIPT --config $ISR_CONFIG --num_steps 500 --seed 42 --output_dir $OUTPUT_BASE/d144_llama_isr_s42
echo "$(date) [D144] ISR s42 DONE"

echo "$(date) [D144] Starting ISR s43"
$PYTHON $SCRIPT --config $ISR_CONFIG --num_steps 500 --seed 43 --output_dir $OUTPUT_BASE/d144_llama_isr_s43
echo "$(date) [D144] ISR s43 DONE"

echo "$(date) [D144] Starting ISR s44"
$PYTHON $SCRIPT --config $ISR_CONFIG --num_steps 500 --seed 44 --output_dir $OUTPUT_BASE/d144_llama_isr_s44
echo "$(date) [D144] ISR s44 DONE"

echo "$(date) [D144] ALL 5 EXPERIMENTS COMPLETE"
