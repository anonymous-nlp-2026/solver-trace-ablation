"""Phase 1 GRPO training: execution-only vs execution+STA reward.

Usage:
  # execution-only (GPU 0)
  CUDA_VISIBLE_DEVICES=0 python scripts/train_grpo.py \
    --model_path /root/autodl-tmp/models/Qwen3-4B \
    --data_path data/proofwriter_fol_500.jsonl \
    --reward_mode execution_only --output_dir outputs/grpo_exec_only \
    --num_steps 100 --wandb_project solver-trace-ablation

  # execution+STA (GPU 1)
  CUDA_VISIBLE_DEVICES=1 python scripts/train_grpo.py \
    --model_path /root/autodl-tmp/models/Qwen3-4B \
    --data_path data/proofwriter_fol_500.jsonl \
    --reward_mode execution_sta --output_dir outputs/grpo_exec_sta \
    --num_steps 100 --wandb_project solver-trace-ablation
"""

from __future__ import annotations

import argparse
import json
import math
import logging
import os
import random as _random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.solvers.factory import get_solver
from src.sta.reward import _parse_formalization, fast_sta_reward, sta_reward
from src.sta.prolog_reward import (
    fast_prolog_sta_reward, parse_prolog_formalization, prolog_sta_reward,
)

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


PROLOG_SYSTEM_PROMPT = (
    "You are a Prolog logic programming expert. "
    "Given natural language premises and a conclusion, formalize them into Prolog.\n\n"
    "Output format:\n"
    "Facts/Rules:\n"
    "<fact_or_rule>.\n"
    "...\n\n"
    "Query: <goal>\n\n"
    "Use standard Prolog syntax: facts (predicate(args).), rules (head :- body.), "
    "variables start with uppercase, atoms/predicates start with lowercase."
)

PROLOG_USER_TEMPLATE = (
    "Formalize the following in Prolog.\n\n"
    "Context:\n{context}\n\n"
    "Statement to formalize:\n{question}"
)



def load_jsonl(path: str) -> list[dict]:
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def build_dataset(data: list[dict]) -> Dataset:
    records = []
    for item in data:
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(
                context=item["context"], question=item["question"]
            )},
        ]
        records.append({
            "prompt": prompt,
            "id": item.get("id", ""),
            "answer": item.get("answer", ""),
            "fol_premises": json.dumps(item.get("fol_premises", [])),
            "fol_conclusion": item.get("fol_conclusion", ""),
        })
    return Dataset.from_list(records)

def build_prolog_dataset(data: list[dict]) -> Dataset:
    records = []
    for item in data:
        prompt = [
            {"role": "system", "content": PROLOG_SYSTEM_PROMPT},
            {"role": "user", "content": PROLOG_USER_TEMPLATE.format(
                context=item["context"], question=item["question"]
            )},
        ]
        records.append({
            "prompt": prompt,
            "id": item.get("id", ""),
            "answer": item.get("answer", ""),
            "prolog_premises": json.dumps(item.get("prolog_premises", [])),
            "prolog_query": item.get("prolog_query", ""),
        })
    return Dataset.from_list(records)



def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def extract_completion_text(completion) -> str:
    """Extract text from a completion (handles both string and conversational formats)."""
    if isinstance(completion, list):
        return completion[-1]["content"] if completion else ""
    return completion



def _is_h7(premises: list[str], conclusion: str) -> bool:
    conclusion_norm = conclusion.strip().lower()
    for p in premises:
        if p.strip().lower() == conclusion_norm:
            return True
    return False


def make_reward_fn(reward_mode: str, solver, exec_weight: float = 0.5, sta_weight: float = 0.5, logic_type: str = "fol", beta: float = 0.3, beta_schedule: str = "fixed", beta_schedule_midpoint: int = 250, max_steps: int = 500, isr_threshold: float = 0.3, min_components: int = 2, warmup_steps: int = 0):
    stats = {"total": 0, "parsed": 0, "proved": 0, "isr_sum": 0.0}
    step_buf = {"total": 0, "parsed": 0, "proved": 0, "isr_sum": 0.0, "reward_sum": 0.0, "random_sum": 0.0, "random_n": 0, "beta_t": None, "gated_passed": 0, "gated_total": 0, "warmup_exec": 0, "h7_blocked": 0, "shuffle_n_proved": 0, "shuffle_identity": 0}
    schedule_state = {"current_step": 0, "phase_announced": False}

    def compute_beta_t():
        """Compute time-varying beta coefficient based on the chosen schedule."""
        step = schedule_state["current_step"]
        if beta_schedule == "linear":
            # beta_t = beta * (t / T): linear ramp from 0 to beta over max_steps
            return beta * min(step / max_steps, 1.0) if max_steps > 0 else beta
        elif beta_schedule == "sigmoid":
            # beta_t = beta * sigma((t - midpoint) / (T/10)): S-curve centered at midpoint
            x = (step - beta_schedule_midpoint) / (max_steps / 10) if max_steps > 0 else 0
            return beta / (1 + math.exp(-x))
        else:
            return beta


    def reward_fn(prompts, completions, **kwargs):
        rewards = []
        _shuffle_buf = []
        for completion in completions:
            text = extract_completion_text(completion)
            cleaned = strip_thinking(text)
            stats["total"] += 1
            step_buf["total"] += 1

            if logic_type == "prolog":
                _fast_fn = fast_prolog_sta_reward
                _full_fn = prolog_sta_reward
            else:
                _fast_fn = fast_sta_reward
                _full_fn = sta_reward

            if reward_mode == "execution_only":
                result = _full_fn(
                    {}, cleaned, solver,
                    execution_weight=1.0, sta_weight=0.0, timeout=5,
                )
                reward_val = result["execution_reward"]
            elif reward_mode == "random_reward":
                result = _full_fn(
                    {}, cleaned, solver,
                    execution_weight=1.0, sta_weight=0.0, timeout=5,
                )
                proved = result["execution_reward"] > 0
                if proved:
                    rng = _random.Random(stats["total"])
                    rand_val = rng.random()
                    reward_val = exec_weight * 1.0 + sta_weight * rand_val
                    result["random_reward"] = rand_val
                    step_buf["random_sum"] += rand_val
                    step_buf["random_n"] += 1
                else:
                    reward_val = 0.0
                    result["random_reward"] = 0.0
            elif reward_mode == "additive_bonus":
                # D010: additive ISR bonus — r = r_exec * (1 + beta_t * ISR)
                result = _full_fn(
                    {}, cleaned, solver,
                    execution_weight=1.0, sta_weight=1.0, timeout=5,
                )
                execution_reward = result["execution_reward"]
                isr_val = result.get("sta_reward", 0.0)
                beta_t = compute_beta_t()
                reward_val = execution_reward * (1 + beta_t * isr_val)
                step_buf["beta_t"] = beta_t
            elif reward_mode == "shuffled_isr":
                # Control: batch-level ISR permutation breaks sample-ISR association
                result = _full_fn(
                    {}, cleaned, solver,
                    execution_weight=1.0, sta_weight=1.0, timeout=5,
                )
                execution_reward = result["execution_reward"]
                isr_val = result.get("sta_reward", 0.0)
                beta_t = compute_beta_t()
                reward_val = 0.0
                step_buf["beta_t"] = beta_t
                _shuffle_buf.append((len(rewards), execution_reward, isr_val))
            elif reward_mode == "additive_random":
                # plan017: matched control for additive_bonus — random replaces ISR to isolate structural signal
                result = _full_fn(
                    {}, cleaned, solver,
                    execution_weight=1.0, sta_weight=1.0, timeout=5,
                )
                execution_reward = result["execution_reward"]
                rng = _random.Random(stats["total"])
                rand_val = rng.random()
                beta_t = compute_beta_t()
                reward_val = execution_reward * (1 + beta_t * rand_val)
                result["random_reward"] = rand_val
                step_buf["random_sum"] += rand_val
                step_buf["random_n"] += 1
                step_buf["beta_t"] = beta_t
            elif reward_mode == "isr_gated":
                cur_step = schedule_state["current_step"]
                in_warmup = warmup_steps > 0 and cur_step <= warmup_steps
                if in_warmup:
                    result = _full_fn(
                        {}, cleaned, solver,
                        execution_weight=1.0, sta_weight=1.0, timeout=30,
                    )
                    execution_reward = result["execution_reward"]
                    reward_val = 1.0 if execution_reward > 0 else 0.0
                    step_buf["warmup_exec"] += 1
                else:
                    if not schedule_state["phase_announced"]:
                        logger.info(f"=== Phase 2: ISR-gated reward activated (step {cur_step}, warmup_steps={warmup_steps}) ===")
                        schedule_state["phase_announced"] = True
                    result = _full_fn(
                        {}, cleaned, solver,
                        execution_weight=1.0, sta_weight=1.0, timeout=30,
                    )
                    execution_reward = result["execution_reward"]
                    isr_val = result.get("sta_reward", 0.0)
                    details = result.get("details", {})
                    n_comp = details.get("n_components", 0)
                    if execution_reward > 0:
                        step_buf["gated_total"] += 1
                        premises_parsed = details.get("premises", [])
                        conclusion_parsed = details.get("conclusion", "")
                        h7 = _is_h7(premises_parsed, conclusion_parsed)
                        if h7:
                            step_buf["h7_blocked"] += 1
                        if isr_val > isr_threshold and n_comp > min_components and not h7:
                            reward_val = 1.0
                            step_buf["gated_passed"] += 1
                        else:
                            reward_val = 0.0
                    else:
                        reward_val = 0.0
            elif reward_mode == "additive_bonus_gated":
                result = _full_fn(
                    {}, cleaned, solver,
                    execution_weight=1.0, sta_weight=1.0, timeout=5,
                )
                execution_reward = result["execution_reward"]
                isr_val = result.get("sta_reward", 0.0)
                details = result.get("details", {})
                n_premises = len(details.get("premises", []))
                beta_t = compute_beta_t()
                gate = 1 if n_premises >= 2 else 0
                reward_val = execution_reward + beta_t * isr_val * gate
                if execution_reward > 0:
                    step_buf["gated_total"] += 1
                    if gate == 0:
                        step_buf["h7_blocked"] += 1
                    else:
                        step_buf["gated_passed"] += 1
                step_buf["beta_t"] = beta_t
            elif reward_mode == "execution_fast_sta":
                result = _fast_fn(
                    {}, cleaned, solver,
                    execution_weight=exec_weight, sta_weight=sta_weight, timeout=5,
                )
                reward_val = result["combined_reward"]
            else:
                result = _full_fn(
                    {}, cleaned, solver,
                    execution_weight=exec_weight, sta_weight=sta_weight, timeout=5,
                )
                reward_val = result["combined_reward"]

            parsed = "error" not in result.get("details", {})
            proved = result["execution_reward"] > 0
            isr = result.get("sta_reward", 0.0)

            if parsed:
                stats["parsed"] += 1
                step_buf["parsed"] += 1
            if proved:
                stats["proved"] += 1
                step_buf["proved"] += 1
            stats["isr_sum"] += isr
            step_buf["isr_sum"] += isr
            step_buf["reward_sum"] += reward_val

            rewards.append(reward_val)
        if reward_mode == "shuffled_isr" and _shuffle_buf:
            beta_t = compute_beta_t()
            proved_indices = []
            isr_vals = []
            for _si, (_idx, _exec_r, _isr) in enumerate(_shuffle_buf):
                if _exec_r > 0:
                    proved_indices.append((_idx, _exec_r))
                    isr_vals.append(_isr)
            n_proved = len(proved_indices)
            step_buf["shuffle_n_proved"] += n_proved
            if n_proved > 1:
                batch_rng = _random.Random(stats["total"])
                batch_rng.shuffle(isr_vals)
            else:
                step_buf["shuffle_identity"] += 1
            for (_idx, _exec_r), _s_isr in zip(proved_indices, isr_vals):
                _r = _exec_r * (1 + beta_t * _s_isr)
                rewards[_idx] = _r
                step_buf["reward_sum"] += _r
        return rewards

    def get_stats():
        n = stats["total"]
        if n == 0:
            return {}
        return {
            "parse_rate": stats["parsed"] / n,
            "prove_rate": stats["proved"] / n,
            "mean_isr": stats["isr_sum"] / n,
        }

    def get_step_stats():
        n = step_buf["total"]
        if n == 0:
            return {}
        result = {
            "parse_rate": step_buf["parsed"] / n,
            "prove_rate": step_buf["proved"] / n,
            "mean_isr": step_buf["isr_sum"] / n,
            "reward_mean": step_buf["reward_sum"] / n,
            "reward_std": 0.0,
        }
        result["warmup_exec"] = step_buf["warmup_exec"]
        if step_buf["gated_total"] > 0:
            result["gate_pass_rate"] = step_buf["gated_passed"] / step_buf["gated_total"]
        result["h7_blocked"] = step_buf["h7_blocked"]
        if step_buf["random_n"] > 0:
            result["random_mean"] = step_buf["random_sum"] / step_buf["random_n"]
        if step_buf.get("beta_t") is not None:
            result["beta_t"] = step_buf["beta_t"]
        if step_buf["shuffle_n_proved"] > 0:
            result["shuffle_n_proved"] = step_buf["shuffle_n_proved"]
            result["shuffle_identity"] = step_buf["shuffle_identity"]
        step_buf["total"] = 0
        step_buf["parsed"] = 0
        step_buf["proved"] = 0
        step_buf["isr_sum"] = 0.0
        step_buf["reward_sum"] = 0.0
        step_buf["random_sum"] = 0.0
        step_buf["random_n"] = 0
        step_buf["beta_t"] = None
        step_buf["gated_passed"] = 0
        step_buf["gated_total"] = 0
        step_buf["warmup_exec"] = 0
        step_buf["h7_blocked"] = 0
        step_buf["shuffle_n_proved"] = 0
        step_buf["shuffle_identity"] = 0
        return result

    def reset_stats():
        stats["total"] = 0
        stats["parsed"] = 0
        stats["proved"] = 0
        stats["isr_sum"] = 0.0

    reward_fn.get_stats = get_stats
    reward_fn.get_step_stats = get_step_stats
    reward_fn.reset_stats = reset_stats
    reward_fn.schedule_state = schedule_state
    return reward_fn


class LocalLogCallback(TrainerCallback):
    def __init__(self, output_dir, reward_fn):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "reward_log.jsonl"
        self.reward_fn = reward_fn
        self.last_step = 0

    def on_step_begin(self, args, state, control, **kwargs):
        if hasattr(self.reward_fn, 'schedule_state'):
            self.reward_fn.schedule_state['current_step'] = state.global_step

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or state.global_step == 0:
            return
        if state.global_step == self.last_step:
            return
        self.last_step = state.global_step
        step_stats = self.reward_fn.get_step_stats()
        entry = {
            "step": state.global_step,
            "reward_mean": logs.get("reward", step_stats.get("reward_mean", 0.0)),
            "reward_std": logs.get("reward_std", step_stats.get("reward_std", 0.0)),
            "prove_rate": step_stats.get("prove_rate", 0.0),
            "isr_mean": step_stats.get("mean_isr", 0.0),
            "parse_rate": step_stats.get("parse_rate", 0.0),
            "random_mean": step_stats.get("random_mean", None),
            "loss": logs.get("loss", 0.0),
            "lr": logs.get("learning_rate", 0.0),
            "beta_t": step_stats.get("beta_t", None),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
            f.flush()
        if step_stats.get("gate_pass_rate") is not None:
            entry["gate_pass_rate"] = step_stats["gate_pass_rate"]
        entry["warmup_exec"] = step_stats.get("warmup_exec", 0)
        print(
            f"[REWARD_LOG] step={entry['step']} reward={entry['reward_mean']:.4f} "
            f"prove_rate={entry['prove_rate']:.3f} isr={entry['isr_mean']:.3f} "
            f"parse_rate={entry['parse_rate']:.3f}"
            + (f" random={entry['random_mean']:.3f}" if entry.get('random_mean') is not None else "")
            + (f" beta_t={entry['beta_t']:.4f}" if entry.get('beta_t') is not None else "")
            + (f" gate_pass={entry['gate_pass_rate']:.3f}" if entry.get('gate_pass_rate') is not None else "")
            + (f" shuffle_proved={step_stats['shuffle_n_proved']}" if step_stats.get('shuffle_n_proved') else "")
            + (f" shuffle_id={step_stats['shuffle_identity']}" if step_stats.get('shuffle_identity') else "")
            + (f" [WARMUP]" if entry.get('warmup_exec', 0) > 0 else ""),
            flush=True,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1 GRPO Training")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--reward_mode", type=str, default=None,
                        choices=["execution_only", "execution_sta", "random_reward", "additive_bonus", "additive_random", "shuffled_isr", "execution_fast_sta", "isr_gated", "additive_bonus_gated"])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--per_device_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--num_generations", type=int, default=None)
    parser.add_argument("--max_completion_length", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--exec_weight", type=float, default=None)
    parser.add_argument("--sta_weight", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None,
                        help="Additive bonus coefficient for ISR (used with additive_bonus mode)")
    parser.add_argument("--beta_schedule", type=str, default=None,
                        choices=["fixed", "linear", "sigmoid"],
                        help="Beta schedule: fixed (constant), linear (ramp 0->beta), sigmoid (S-curve 0->beta)")
    parser.add_argument("--beta_schedule_midpoint", type=int, default=None,
                        help="Midpoint step for sigmoid schedule (default 250)")
    parser.add_argument("--isr_threshold", type=float, default=None,
                        help="ISR threshold for isr_gated mode (default 0.3)")
    parser.add_argument("--min_components", type=int, default=None,
                        help="Minimum n_components for isr_gated mode (default 2)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--solver", type=str, default=None)
    parser.add_argument("--logic_type", type=str, default=None,
                        choices=["fol", "prolog"])
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--warmup_ratio", type=float, default=None)
    parser.add_argument("--optim", type=str, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None,
                        help="Number of warmup steps using exec-only reward before ISR-gated (default 0)")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config file (overrides CLI args)")
    args = parser.parse_args()

    if args.config:
        cfg = load_yaml_config(args.config)
        for k, v in cfg.items():
            if hasattr(args, k):
                current = getattr(args, k)
                if current is None:
                    setattr(args, k, v)
            else:
                setattr(args, k, v)

    # Apply defaults for optional args after config merge
    if args.num_steps is None:
        args.num_steps = 100
    if args.per_device_batch_size is None:
        args.per_device_batch_size = 4
    if args.gradient_accumulation_steps is None:
        args.gradient_accumulation_steps = 2
    if args.num_generations is None:
        args.num_generations = 4
    if args.max_completion_length is None:
        args.max_completion_length = 512
    if args.learning_rate is None:
        args.learning_rate = 1e-6
    if args.exec_weight is None:
        args.exec_weight = 0.5
    if args.sta_weight is None:
        args.sta_weight = 0.5
    if args.beta is None:
        args.beta = 0.3
    if args.beta_schedule is None:
        args.beta_schedule = "fixed"
    if args.beta_schedule_midpoint is None:
        args.beta_schedule_midpoint = 250
    if args.seed is None:
        args.seed = 42
    if args.warmup_ratio is None:
        args.warmup_ratio = 0.0
    if args.isr_threshold is None:
        args.isr_threshold = 0.3
    if args.min_components is None:
        args.min_components = 2
    if args.warmup_steps is None:
        args.warmup_steps = 0
    if args.solver is None:
        args.solver = "prover9"
    if args.logic_type is None:
        args.logic_type = "fol"

    missing = [f for f in ("model_path", "data_path", "reward_mode", "output_dir")
               if getattr(args, f, None) is None]
    if missing:
        parser.error(f"the following arguments are required: {', '.join('--' + f for f in missing)}")

    return args


def load_yaml_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()

    if args.no_wandb:
        os.environ["WANDB_MODE"] = "disabled"

    report_to = "none" if args.no_wandb else "wandb"
    if args.wandb_project and not args.no_wandb:
        import wandb
        run_name = args.wandb_run or f"grpo_{args.reward_mode}"
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    logger.info(f"Loading data from {args.data_path}")
    raw_data = load_jsonl(args.data_path)
    logic_type = getattr(args, "logic_type", "fol")
    if logic_type == "prolog":
        dataset = build_prolog_dataset(raw_data)
    else:
        dataset = build_dataset(raw_data)
    logger.info(f"Dataset: {len(dataset)} examples")

    logger.info(f"Initializing solver: {args.solver}")
    solver = get_solver(args.solver)
    logger.info(f"Solver ready: {type(solver).__name__}")

    reward_fn = make_reward_fn(
        args.reward_mode, solver,
        exec_weight=args.exec_weight, sta_weight=args.sta_weight,
        logic_type=logic_type,
        beta=getattr(args, "beta", 0.3),
        beta_schedule=getattr(args, "beta_schedule", "fixed"),
        beta_schedule_midpoint=getattr(args, "beta_schedule_midpoint", 250),
        max_steps=args.num_steps,
        isr_threshold=getattr(args, "isr_threshold", 0.3),
        min_components=getattr(args, "min_components", 2),
        warmup_steps=getattr(args, "warmup_steps", 0),
    )

    # Probe tokenizer chat_template to decide which kwargs are supported.
    # Qwen3 templates use `enable_thinking`; LLaMA-3 templates do not reference it.
    # Passing unused kwargs is harmless on most templates but we keep the dict minimal.
    from transformers import AutoTokenizer as _AutoTok
    _probe_tok = _AutoTok.from_pretrained(args.model_path, trust_remote_code=True)
    _ct = getattr(_probe_tok, 'chat_template', '') or ''
    _chat_template_kwargs = {}
    if 'enable_thinking' in _ct:
        _chat_template_kwargs['enable_thinking'] = False
    del _probe_tok

    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.num_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        logging_steps=1,
        save_steps=getattr(args, "save_steps", 50),
        save_total_limit=getattr(args, "save_total_limit", 1),
        save_only_model=True,
        report_to=report_to,
        bf16=True,
        seed=args.seed,
        warmup_ratio=args.warmup_ratio,
        log_completions=False,
        num_completions_to_print=2,
        chat_template_kwargs=_chat_template_kwargs,
        gradient_checkpointing=getattr(args, "gradient_checkpointing", False),
        generation_batch_size=args.num_generations,
        optim=getattr(args, "optim", None) or "adamw_torch",
        model_init_kwargs={"dtype": "bfloat16"},
    )

    logger.info(f"Loading model from {args.model_path}")
    trainer = GRPOTrainer(
        model=args.model_path,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=dataset,
        callbacks=[LocalLogCallback(args.output_dir, reward_fn)],
    )

    # Fix LLaMA eos: TRL GRPOTrainer builds its own self.generation_config from
    # tokenizer.eos_token_id, ignoring model.generation_config. Patch trainer's copy.
    _tok = trainer.processing_class
    _eot_id = _tok.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(_eot_id, int) and _eot_id != getattr(_tok, 'unk_token_id', None):
        _gen_cfg = trainer.generation_config
        _existing = _gen_cfg.eos_token_id
        if isinstance(_existing, int):
            _existing = [_existing]
        elif _existing is None:
            _existing = []
        if _eot_id not in _existing:
            _eos_list = list(_existing) + [_eot_id]
            _gen_cfg.eos_token_id = _eos_list
            if hasattr(trainer, 'generation_kwargs') and trainer.generation_kwargs is not None:
                trainer.generation_kwargs["eos_token_id"] = _eos_list
            logger.info(f"Patched trainer.generation_config.eos_token_id -> {_eos_list}")

    logger.info(f"Starting GRPO training: mode={args.reward_mode}, logic={logic_type}, steps={args.num_steps}")
    trainer.train()

    logger.info("Saving final model")
    trainer.save_model(args.output_dir)

    final_stats = reward_fn.get_stats()
    logger.info(f"Final reward stats: {final_stats}")

    if not args.no_wandb and args.wandb_project:
        import wandb
        wandb.log({"final/" + k: v for k, v in final_stats.items()})
        wandb.finish()

    logger.info("Done.")


if __name__ == "__main__":
    main()
