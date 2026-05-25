#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0

MODEL_PATH="/root/autodl-tmp/models/Llama-3.2-3B"
DATA_PATH="data/train_500_seed42.jsonl"
EVAL_DATA="data/proofwriter_fol_100.jsonl"
PYTHON="/root/miniconda3/bin/python3"
SCRIPT="scripts/train_grpo.py"
EVAL_SCRIPT="scripts/eval_sta.py"
NUM_STEPS=500
BATCH=2
GRAD_ACC=4
LOG_FILE="/root/autodl-tmp/d144_llama_pipeline.log"

cd /root/solver-trace-ablation

echo "=== D144 LLaMA Pipeline Started: $(date) ===" | tee -a $LOG_FILE

# Exec-only × 3 seeds
for SEED in 42 43 44; do
  OUT_DIR="/root/autodl-tmp/outputs/d144_llama_exec_s${SEED}"
  echo "=== [$(date)] Starting exec-only seed=$SEED ===" | tee -a $LOG_FILE
  $PYTHON $SCRIPT \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --reward_mode execution_only \
    --output_dir $OUT_DIR \
    --num_steps $NUM_STEPS \
    --seed $SEED \
    --per_device_batch_size $BATCH \
    --gradient_accumulation_steps $GRAD_ACC \
    --no_wandb 2>&1 | tee -a $LOG_FILE

  echo "=== [$(date)] Eval exec-only seed=$SEED ===" | tee -a $LOG_FILE
  $PYTHON $EVAL_SCRIPT \
    --model_path $OUT_DIR \
    --data_path $EVAL_DATA \
    --output_path ${OUT_DIR}/eval_results.json \
    --solver prover9 2>&1 | tee -a $LOG_FILE || echo "EVAL_FAILED seed=$SEED" | tee -a $LOG_FILE
  echo "=== [$(date)] DONE exec-only seed=$SEED ===" | tee -a $LOG_FILE
done

# ISR β=0.1 × 3 seeds
for SEED in 42 43 44; do
  OUT_DIR="/root/autodl-tmp/outputs/d144_llama_isr_s${SEED}"
  echo "=== [$(date)] Starting ISR seed=$SEED ===" | tee -a $LOG_FILE
  $PYTHON $SCRIPT \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --reward_mode additive_bonus \
    --beta 0.1 \
    --output_dir $OUT_DIR \
    --num_steps $NUM_STEPS \
    --seed $SEED \
    --per_device_batch_size $BATCH \
    --gradient_accumulation_steps $GRAD_ACC \
    --no_wandb 2>&1 | tee -a $LOG_FILE

  echo "=== [$(date)] Eval ISR seed=$SEED ===" | tee -a $LOG_FILE
  $PYTHON $EVAL_SCRIPT \
    --model_path $OUT_DIR \
    --data_path $EVAL_DATA \
    --output_path ${OUT_DIR}/eval_results.json \
    --solver prover9 2>&1 | tee -a $LOG_FILE || echo "EVAL_FAILED seed=$SEED" | tee -a $LOG_FILE
  echo "=== [$(date)] DONE ISR seed=$SEED ===" | tee -a $LOG_FILE
done

echo "=== D144 LLaMA Pipeline FINISHED: $(date) ===" | tee -a $LOG_FILE
