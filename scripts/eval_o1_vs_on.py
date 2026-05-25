"""O(1) vs O(N) ISR comparison (plan_004).

Compares fast_isr_fol (O(1), binary) vs compute_isr (O(N), continuous)
on the same model outputs to quantify information loss from the binary
approximation. Documents why r4 (fast ISR as GRPO reward) underperforms.

Inputs:
  --model_path   HF checkpoint directory
  --data_path    JSONL test file
  --output_path  JSON report output

Output JSON:
  summary: correlation, agreement_rate, mean_abs_diff, n_disagreements, ...
  results[]: per-sample fast_isr, full_isr, diff, component details

Dependencies: transformers, torch, scipy
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
from src.sta.ablation import run_sta
from src.sta.isr import compute_isr, fast_isr_fol
from src.sta.reward import _parse_formalization


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
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_model(model_path: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True
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
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
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
    result = re.sub(r"<think>.*?</think>\s*", "", result, flags=re.DOTALL).strip()
    return result


def main():
    parser = argparse.ArgumentParser(
        description="O(1) vs O(N) ISR comparison (plan_004)"
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--solver", type=str, default="prover9")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=5)
    args = parser.parse_args()

    data = load_data(args.data_path)
    print(f"Loaded {len(data)} problems from {args.data_path}")

    print(f"Loading model from {args.model_path}...")
    model, tokenizer = load_model(args.model_path)
    print("Model loaded.")

    solver = get_solver(args.solver)
    print(f"Solver: {type(solver).__name__}")

    results = []
    fast_isrs = []
    full_isrs = []
    t_start = time.time()

    for i, problem in enumerate(data):
        context = problem.get("context", "")
        question = problem.get("question", "")

        print(f"[{i+1}/{len(data)}] id={problem.get('id', i)} ...", end=" ", flush=True)

        formalization = generate_formalization(
            model, tokenizer, context, question, args.max_new_tokens
        )

        try:
            premises, conclusion = _parse_formalization(formalization)
        except Exception as e:
            print(f"PARSE_ERR: {e}")
            results.append({
                "index": i, "id": problem.get("id", ""),
                "error": f"parse: {e}", "fast_isr": None, "full_isr": None,
            })
            continue

        if not conclusion:
            print("EMPTY_CONCL")
            results.append({
                "index": i, "id": problem.get("id", ""),
                "error": "empty conclusion", "fast_isr": None, "full_isr": None,
            })
            continue

        # O(1) fast ISR
        fast_result = fast_isr_fol(premises, conclusion, solver, timeout=args.timeout)

        # O(N) full ISR — only if original proved
        if fast_result["original_proved"]:
            original_proof = solver.prove(premises, conclusion, timeout=args.timeout)
            ablation_results = run_sta(
                premises, conclusion, solver, timeout=args.timeout,
                original_result=original_proof,
            )
            full_isr = compute_isr(ablation_results)
            n_components = len(ablation_results)
            n_necessary = sum(1 for r in ablation_results if r.is_necessary)
            component_details = [
                {
                    "type": r.component.type,
                    "content": r.component.content,
                    "is_necessary": r.is_necessary,
                }
                for r in ablation_results
            ]
        else:
            full_isr = 0.0
            n_components = 0
            n_necessary = 0
            component_details = []

        fast_v = fast_result["fast_isr"]
        diff = fast_v - full_isr
        fast_isrs.append(fast_v)
        full_isrs.append(full_isr)

        record = {
            "index": i,
            "id": problem.get("id", ""),
            "formalization": formalization,
            "premises": premises,
            "conclusion": conclusion,
            "original_proved": fast_result["original_proved"],
            "conclusion_tautology": fast_result["conclusion_tautology"],
            "fast_isr": fast_v,
            "full_isr": round(full_isr, 4),
            "diff": round(diff, 4),
            "agree": (fast_v > 0.5) == (full_isr > 0.5),
            "n_components": n_components,
            "n_necessary": n_necessary,
            "components": component_details,
        }
        results.append(record)

        tag = "AGREE" if record["agree"] else "DISAGREE"
        print(f"fast={fast_v:.1f} full={full_isr:.3f} diff={diff:+.3f} {tag}")

    elapsed = time.time() - t_start

    # Aggregate statistics
    valid = [(f, u) for f, u in zip(fast_isrs, full_isrs)]
    n_valid = len(valid)

    if n_valid > 1:
        try:
            from scipy.stats import pearsonr, spearmanr
            pr, p_pval = pearsonr(fast_isrs, full_isrs)
            sr, s_pval = spearmanr(fast_isrs, full_isrs)
        except ImportError:
            pr = s_pval = sr = p_pval = None
    else:
        pr = sr = p_pval = s_pval = None

    n_agree = sum(1 for r in results if r.get("agree"))
    n_disagree = sum(1 for r in results if r.get("agree") is False)
    mean_abs_diff = statistics.mean(abs(d) for d in (r["diff"] for r in results if r.get("diff") is not None)) if n_valid else 0.0

    # Breakdown: where does fast_isr lose information?
    # Case 1: fast=1, full<1 → binary hides partial redundancy
    # Case 2: fast=0, full>0 → false negative (shouldn't happen logically)
    # Case 3: fast=1, full=1 → perfect agreement
    # Case 4: fast=0, full=0 → perfect agreement
    case_counts = {"fast1_full1": 0, "fast1_full_lt1": 0, "fast0_full0": 0, "fast0_full_gt0": 0}
    for r in results:
        if r.get("fast_isr") is None:
            continue
        fi, ui = r["fast_isr"], r["full_isr"]
        if fi == 1.0 and ui == 1.0:
            case_counts["fast1_full1"] += 1
        elif fi == 1.0 and ui < 1.0:
            case_counts["fast1_full_lt1"] += 1
        elif fi == 0.0 and ui == 0.0:
            case_counts["fast0_full0"] += 1
        elif fi == 0.0 and ui > 0.0:
            case_counts["fast0_full_gt0"] += 1

    summary = {
        "n_problems": len(data),
        "n_valid": n_valid,
        "n_parse_errors": len(data) - n_valid,
        "pearson_r": round(pr, 4) if pr is not None else None,
        "pearson_pval": round(p_pval, 6) if p_pval is not None else None,
        "spearman_r": round(sr, 4) if sr is not None else None,
        "spearman_pval": round(s_pval, 6) if s_pval is not None else None,
        "agreement_rate": round(n_agree / n_valid, 4) if n_valid else 0.0,
        "n_agree": n_agree,
        "n_disagree": n_disagree,
        "mean_abs_diff": round(mean_abs_diff, 4),
        "mean_fast_isr": round(statistics.mean(fast_isrs), 4) if fast_isrs else 0.0,
        "mean_full_isr": round(statistics.mean(full_isrs), 4) if full_isrs else 0.0,
        "case_breakdown": case_counts,
        "elapsed_seconds": round(elapsed, 1),
        "model_path": args.model_path,
    }

    output = {"summary": summary, "results": results}
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Results saved to {args.output_path}")
    print(f"  Valid samples:    {n_valid}/{len(data)}")
    print(f"  Pearson r:        {pr}")
    print(f"  Spearman rho:     {sr}")
    print(f"  Agreement rate:   {n_agree}/{n_valid} ({summary['agreement_rate']:.1%})")
    print(f"  Mean |diff|:      {mean_abs_diff:.4f}")
    print(f"  Mean fast ISR:    {summary['mean_fast_isr']:.4f}")
    print(f"  Mean full ISR:    {summary['mean_full_isr']:.4f}")
    print(f"  Case breakdown:   {case_counts}")
    print(f"  Time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
