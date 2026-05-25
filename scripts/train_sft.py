"""SFT baseline training on FOLIO NL→FOL (plan_009).

Supervised fine-tuning: input = NL premises + conclusion,
output = FOL formalization. Uses trl.SFTTrainer.

Inputs:
  --model_path   Base model (default: Qwen3-4B local path)
  --data_path    FOLIO JSONL (default: data/train_500_seed42.jsonl)
  --output_dir   Checkpoint output directory
  --num_steps    Training steps (default 500)
  --seed         Random seed (default 42)
  --gpu          GPU index to use (sets CUDA_VISIBLE_DEVICES)

Data format: each JSONL line must have:
  context, question, fol_premises (list[str]), fol_conclusion (str)

Dependencies: transformers, trl, datasets, torch
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are a first-order logic (FOL) expert. "
    "Given natural language premises and a conclusion, formalize them into FOL.\n\n"
    "Output format:\n"
    "Premises:\n"
    "1. <FOL formula>\n"
    "2. <FOL formula>\n"
    "...\n"
    "Conclusion: <FOL formula>\n\n"
    "Use standard FOL notation with predicate-argument syntax: "
    "all x (P(x) -> Q(x)), exists x P(x), P(a) & Q(a), -P(a), P(a) | Q(a), P(a) -> Q(a)."
)

USER_TEMPLATE = (
    "Formalize the following in first-order logic.\n\n"
    "Context:\n{context}\n\n"
    "Statement to formalize:\n{question}"
)


def format_fol_output(fol_premises: list[str], fol_conclusion: str) -> str:
    lines = ["Premises:"]
    for i, p in enumerate(fol_premises, 1):
        lines.append(f"{i}. {p}")
    lines.append(f"Conclusion: {fol_conclusion}")
    return "\n".join(lines)


def load_and_format_data(data_path: str) -> list[dict]:
    examples = []
    skipped = 0
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)

            fol_premises = item.get("fol_premises")
            fol_conclusion = item.get("fol_conclusion")
            if not fol_premises or not fol_conclusion:
                skipped += 1
                continue

            context = item.get("context", "")
            question = item.get("question", "")
            user_msg = USER_TEMPLATE.format(context=context, question=question)
            assistant_msg = format_fol_output(fol_premises, fol_conclusion)

            examples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": assistant_msg},
                ]
            })

    logger.info(f"Loaded {len(examples)} training examples, skipped {skipped}")
    return examples


def main():
    parser = argparse.ArgumentParser(description="SFT baseline on FOLIO NL→FOL (plan_009)")
    parser.add_argument("--model_path", type=str,
                        default="/root/autodl-tmp/models/Qwen3-4B",
                        help="Base model path (default: Qwen3-4B)")
    parser.add_argument("--data_path", type=str,
                        default="data/train_500_seed42.jsonl",
                        help="FOLIO JSONL training data")
    parser.add_argument("--output_dir", type=str,
                        default="/root/autodl-tmp/outputs/plan009_sft_s42",
                        help="Checkpoint output directory")
    parser.add_argument("--num_steps", type=int, default=500,
                        help="Max training steps")
    parser.add_argument("--num_train_epochs", type=int, default=None,
                        help="If set, overrides num_steps")
    parser.add_argument("--per_device_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU index (sets CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--bf16", action="store_true", default=True)
    args = parser.parse_args()

    # GPU selection (must happen before torch import) — C002/C005
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        logger.info(f"Set CUDA_VISIBLE_DEVICES={args.gpu}")
    elif "CUDA_VISIBLE_DEVICES" not in os.environ:
        logger.error("CUDA_VISIBLE_DEVICES not set and --gpu not specified. "
                     "Refusing to run without explicit GPU isolation (C002/C005).")
        sys.exit(1)

    # W&B setup
    if args.no_wandb:
        os.environ["WANDB_DISABLED"] = "true"
    else:
        if args.wandb_project:
            os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_run:
            os.environ["WANDB_NAME"] = args.wandb_run

    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    logger.info(f"Loading data from {args.data_path}...")
    examples = load_and_format_data(args.data_path)
    dataset = Dataset.from_list(examples)
    logger.info(f"Dataset: {len(dataset)} examples")

    logger.info(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16 if args.bf16 else "auto",
        device_map={"": 0},
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    training_args = SFTConfig(
        output_dir=args.output_dir,
        max_steps=args.num_steps if args.num_train_epochs is None else -1,
        num_train_epochs=args.num_train_epochs if args.num_train_epochs else 1,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        max_length=args.max_seq_length,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=1,
        seed=args.seed,
        lr_scheduler_type="cosine",
        report_to="wandb" if not args.no_wandb else "none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    logger.info("Starting SFT training...")
    trainer.train()

    logger.info(f"Saving final model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Fix LLaMA generation_config: include <|eot_id|> in eos_token_id
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot_id, int) and eot_id != getattr(tokenizer, 'unk_token_id', None):
        import json as _json
        gc_path = os.path.join(args.output_dir, "generation_config.json")
        if os.path.exists(gc_path):
            with open(gc_path) as _f:
                gc = _json.load(_f)
            existing = gc.get("eos_token_id", [])
            if isinstance(existing, int):
                existing = [existing]
            if eot_id not in existing:
                gc["eos_token_id"] = existing + [eot_id]
                with open(gc_path, "w") as _f:
                    _json.dump(gc, _f, indent=2)
                logger.info(f"Updated generation_config with <|eot_id|> ({eot_id})")

    logger.info("Done.")


if __name__ == "__main__":
    main()
