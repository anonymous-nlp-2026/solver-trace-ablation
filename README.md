# Solver Trace Ablation (STA)

Code and data for the paper *"Solver Trace Ablation Reveals Structural Degeneration Behind Execution-Verified Reasoning"*.

## Overview

STA applies leave-one-out ablation to GRPO-trained formalizations, testing whether each component is necessary for proof success. The resulting **Intensional Structural Reward (ISR)** reveals that prove-rate gains under execution-only reward conceal a compositional shift toward hacking modes.

## Setup

```bash
pip install torch transformers datasets accelerate trl wandb
pip install z3-solver
```

For Prolog experiments, install [SWI-Prolog](https://www.swi-prolog.org/) and the `pyswip` package:
```bash
pip install pyswip
```

## Repository Structure

```
src/
  sta/          # STA ablation and ISR computation
  solvers/      # Z3, Prover9, SWI-Prolog solver backends
scripts/
  train_grpo.py       # GRPO training with configurable reward modes
  train_sft.py        # SFT baseline training
  eval_sta.py         # STA evaluation and ISR computation
  classify_h7.py      # Hacking-mode classifier (CE/TS/PG)
  generate_figures.py # Reproduce paper figures
  prepare_data.py     # Data preprocessing
configs/              # YAML configs for all experimental conditions
data/                 # FOLIO and ProofFOL datasets (preprocessed)
analysis/             # Analysis utilities
```

## Quick Start

### Training (exec-only GRPO)

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_grpo.py \
  --model_path Qwen/Qwen3-4B \
  --data_path data/train_500.jsonl \
  --reward_mode execution_only \
  --output_dir outputs/grpo_exec_only \
  --num_steps 500 --seed 42
```

### Training (ISR reward, beta=0.1)

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_grpo.py \
  --model_path Qwen/Qwen3-4B \
  --data_path data/train_500.jsonl \
  --reward_mode execution_sta --isr_beta 0.1 \
  --output_dir outputs/grpo_isr_b01 \
  --num_steps 500 --seed 42
```

### STA Evaluation

```bash
python scripts/eval_sta.py \
  --model_path outputs/grpo_exec_only/checkpoint-500 \
  --data_path data/test_100.jsonl \
  --output_dir outputs/eval_results
```

### Hacking Classification

```bash
python scripts/classify_h7.py \
  --input outputs/eval_results/sta_results.jsonl \
  --output outputs/eval_results/classified.jsonl
```

## Data

- `data/train_500.jsonl` — 500-example FOLIO training subset (FOL)
- `data/test_100.jsonl` — 100-example FOLIO held-out evaluation set
- `data/prooffol_train_500.jsonl` — ProofFOL training set (silver-standard FOL from ProofWriter)
- `data/prooffol_eval_100.jsonl` — ProofFOL evaluation set

## Configs

The `configs/` directory contains YAML configurations for all experimental conditions reported in the paper, including exec-only, ISR reward (multiple beta values), random reward controls, Prolog variants, and ProofFOL cross-dataset experiments.

## License

MIT
