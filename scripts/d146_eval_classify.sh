#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0
PYTHON=/root/miniconda3/bin/python3
cd /root/solver-trace-ablation

OUT=/root/autodl-tmp/outputs
DATA=data/prooffol_eval_100.jsonl

EXPERIMENTS=(
    d146_pf_exec_s42
    d146_pf_exec_s43
    d146_pf_exec_s44
    d146_pf_isr_s42
    d146_pf_isr_s43
    d146_pf_isr_s44
)

EVAL_FILES=()

# Step 0: Base model (Qwen3-4B) ProofFOL ISR baseline
BASE_DIR="$OUT/d146_pf_base"
BASE_EVAL="$BASE_DIR/eval_sta_results.json"
BASE_CLASSIFY="$BASE_DIR/classify.json"
echo "$(date) [D146-eval] === Step 0: Base model ProofFOL baseline ==="
mkdir -p "$BASE_DIR"

$PYTHON scripts/eval_sta.py     --model_path /root/autodl-tmp/models/Qwen3-4B     --data_path "$DATA"     --output_path "$BASE_EVAL"     --logic_type fol

$PYTHON scripts/classify_h7.py     --eval_path "$BASE_EVAL"     --json_out "$BASE_CLASSIFY"

EVAL_FILES+=("$BASE_EVAL")
echo "  Base model baseline done."
echo ""

for EXP in "${EXPERIMENTS[@]}"; do
    EXP_DIR="$OUT/$EXP"
    EVAL_OUT="$EXP_DIR/eval_sta_results.json"
    CLASSIFY_OUT="$EXP_DIR/classify.json"

    echo "$(date) [D146-eval] === $EXP ==="

    if [ ! -d "$EXP_DIR" ]; then
        echo "  SKIP: $EXP_DIR does not exist"
        continue
    fi

    if [ ! -f "$EXP_DIR/model.safetensors.index.json" ] && [ ! -f "$EXP_DIR/model.safetensors" ]; then
        echo "  SKIP: $EXP — no final model found (training incomplete?)"
        continue
    fi

    echo "  eval_sta.py → $EVAL_OUT"
    $PYTHON scripts/eval_sta.py \
        --model_path "$EXP_DIR" \
        --data_path "$DATA" \
        --output_path "$EVAL_OUT" \
        --logic_type fol

    echo "  classify_h7.py → $CLASSIFY_OUT"
    $PYTHON scripts/classify_h7.py \
        --eval_path "$EVAL_OUT" \
        --json_out "$CLASSIFY_OUT"

    EVAL_FILES+=("$EVAL_OUT")
    echo "  DONE: $EXP"
    echo ""
done

echo "$(date) [D146-eval] === BATCH SUMMARY ==="
if [ ${#EVAL_FILES[@]} -gt 0 ]; then
    $PYTHON scripts/classify_h7.py \
        --batch "${EVAL_FILES[@]}" \
        --json_out "$OUT/d146_classify_summary.json"
else
    echo "  No experiments completed eval. Nothing to summarize."
fi
echo "$(date) [D146-eval] ALL DONE"
