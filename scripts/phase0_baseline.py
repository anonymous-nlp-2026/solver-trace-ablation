"""Phase 0: Base Model ISR Baseline.

Generate FOL formalizations from base Qwen3-4B on FOLIO problems,
then compute ISR baselines using STA.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.solvers.factory import get_solver
from src.sta.reward import sta_reward


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


def load_data(path: str) -> list[dict]:
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_model(model_path: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def generate_formalization(
    model, tokenizer, context: str, question: str, max_new_tokens: int = 512
) -> str:
    import torch

    user_msg = USER_TEMPLATE.format(context=context, question=question)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(generated, skip_special_tokens=True).strip()
    # Strip residual thinking blocks
    result = re.sub(r"<think>.*?</think>\s*", "", result, flags=re.DOTALL).strip()
    return result


def main():
    parser = argparse.ArgumentParser(description="Phase 0: ISR Baseline")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--solver", type=str, default="prover9")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=5)
    args = parser.parse_args()

    print(f"Loading data from {args.data_path}...")
    data = load_data(args.data_path)
    print(f"  {len(data)} problems loaded.")

    print(f"Loading model from {args.model_path}...")
    model, tokenizer = load_model(args.model_path)
    print("  Model loaded.")

    solver = get_solver(args.solver)
    print(f"  Solver: {type(solver).__name__}")

    results = []
    isr_scores = []
    t_start = time.time()

    for i, problem in enumerate(data):
        context = problem.get("context", "")
        question = problem.get("question", "")
        answer = problem.get("answer", "")

        print(f"[{i+1}/{len(data)}] id={problem.get('id', i)} ...", end=" ", flush=True)

        t_sample = time.time()
        formalization = generate_formalization(
            model, tokenizer, context, question, args.max_new_tokens
        )
        gen_time = time.time() - t_sample

        t_eval = time.time()
        reward = sta_reward(problem, formalization, solver, timeout=args.timeout)
        eval_time = time.time() - t_eval

        record = {
            "index": i,
            "id": problem.get("id", ""),
            "context": context,
            "question": question,
            "answer": answer,
            "formalization": formalization,
            "gen_time": round(gen_time, 2),
            "eval_time": round(eval_time, 2),
            **reward,
        }
        results.append(record)

        isr_val = reward["sta_reward"]
        isr_scores.append(isr_val)
        status = "proved" if reward["execution_reward"] > 0 else "FAILED"
        print(f"{status}  ISR={isr_val:.3f}  gen={gen_time:.1f}s  eval={eval_time:.1f}s")

    elapsed = time.time() - t_start

    n = len(isr_scores)
    proved_count = sum(1 for r in results if r["execution_reward"] > 0)
    solver_errors = sum(
        1 for r in results
        if r.get("details", {}).get("error") or r.get("details", {}).get("proof_failed")
    )

    mean_isr = statistics.mean(isr_scores) if isr_scores else 0.0
    std_isr = statistics.stdev(isr_scores) if len(isr_scores) > 1 else 0.0
    median_isr = statistics.median(isr_scores) if isr_scores else 0.0

    # By answer type
    by_answer = {}
    for r in results:
        a = r["answer"]
        if a not in by_answer:
            by_answer[a] = {"isr_scores": [], "proved": 0, "total": 0}
        by_answer[a]["isr_scores"].append(r["sta_reward"])
        by_answer[a]["total"] += 1
        if r["execution_reward"] > 0:
            by_answer[a]["proved"] += 1

    answer_stats = {}
    for a, stats in by_answer.items():
        answer_stats[a] = {
            "count": stats["total"],
            "proved": stats["proved"],
            "prove_rate": stats["proved"] / stats["total"] if stats["total"] else 0.0,
            "mean_isr": statistics.mean(stats["isr_scores"]) if stats["isr_scores"] else 0.0,
        }

    summary = {
        "n_problems": n,
        "n_proved": proved_count,
        "prove_rate": proved_count / n if n else 0.0,
        "n_solver_errors": solver_errors,
        "solver_success_rate": (n - solver_errors) / n if n else 0.0,
        "mean_isr": round(mean_isr, 4),
        "std_isr": round(std_isr, 4),
        "median_isr": round(median_isr, 4),
        "min_isr": round(min(isr_scores), 4) if isr_scores else 0.0,
        "max_isr": round(max(isr_scores), 4) if isr_scores else 0.0,
        "by_answer_type": answer_stats,
        "elapsed_seconds": round(elapsed, 1),
        "avg_time_per_sample": round(elapsed / n, 1) if n else 0.0,
        "solver": type(solver).__name__,
        "model_path": args.model_path,
    }

    output = {"summary": summary, "results": results}

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Results saved to {args.output}")
    print(f"  Problems:  {n}")
    print(f"  Proved:    {proved_count}/{n} ({summary['prove_rate']:.1%})")
    print(f"  Solver errors: {solver_errors}")
    print(f"  Mean ISR:  {mean_isr:.4f} +/- {std_isr:.4f}")
    print(f"  Median:    {median_isr:.4f}")
    print(f"  Range:     [{summary['min_isr']:.4f}, {summary['max_isr']:.4f}]")
    print(f"  By answer type:")
    for a, s in answer_stats.items():
        print(f"    {a}: n={s['count']}, proved={s['proved']}, ISR={s['mean_isr']:.4f}")
    print(f"  Time: {elapsed:.1f}s ({summary['avg_time_per_sample']:.1f}s/sample)")


if __name__ == "__main__":
    main()
