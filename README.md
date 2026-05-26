# Solver Trace Ablation (STA)

Code and data for the paper *"Observe but Don't Optimize: Structural Rewards Trigger Goodhart Effects in Neurosymbolic Reasoning"*.

## Overview

Neurosymbolic reasoning pipelines evaluate formalizations by execution alone, treating all successful proofs as equal. This single-criterion reward conflates genuine derivation with structural shortcuts: embedding the conclusion among the premises proves just as reliably, and RL optimizers exploit the indifference.

STA applies leave-one-out (LOO) ablation to GRPO-trained formalizations, testing whether each component is necessary for proof success. The resulting **Intensional Structural Reward (ISR)** reveals that prove-rate gains conceal a compositional shift toward hacking modes — but ISR occupies a paradoxical position: the same LOO structure that makes it informative also makes it gameable. When repurposed as a training reward, the model learns to game the diagnostic rather than the task (Goodhart effect).

Key findings:
- **Diagnostic value**: ISR reliably detects structural degeneration invisible to execution-based evaluation
- **Goodhart effect**: ISR-as-reward shifts hacking mode composition (CE → TS) without reducing aggregate hacking
- **Seed dependence**: 3 of 5 seeds converge to gaming; 2 retain genuine reasoning
- **Cross-domain**: The diagnostic–optimization asymmetry persists across logic types (FOL, Prolog) and datasets (FOLIO, ProofFOL)

## Setup

```bash
pip install torch transformers datasets accelerate trl wandb
```

For Prolog experiments, install [SWI-Prolog](https://www.swi-prolog.org/) and the `pyswip` package:
```bash
pip install pyswip
```

## Repository Structure

```
src/
  sta/          # STA ablation and ISR computation
  solvers/      # Prover9, SWI-Prolog solver backends
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

### Naming Convention

The codebase uses internal identifiers for hacking modes that differ from the paper terminology:

| Code | Paper | Description |
|------|-------|-------------|
| H1   | CE    | Conclusion embedding (conclusion among multiple premises) |
| H7   | TS    | Tautological shortcut (single premise = conclusion) |
| H7a  | TSa   | Standard tautology (text-match detectable) |
| H7b  | TSb   | Obfuscated tautology (evades text matching) |
| H8   | PG    | Premise-subset gaming |

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

The `configs/` directory contains YAML configurations for all experimental conditions reported in the paper, including exec-only, ISR reward (multiple beta values), random reward controls, shuffled-ISR deconfounding control, Prolog variants, and ProofFOL cross-dataset experiments.

## License

MIT
