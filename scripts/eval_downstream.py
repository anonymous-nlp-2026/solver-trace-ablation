"""Downstream NLI validation: does high ISR translate to correct entailment?

Loads a checkpoint, generates K FOL formalizations per held-out sample,
uses Prover9 for 3-way entailment classification (true/false/unknown),
computes ISR via full STA, and outputs per-sample + aggregated results.

Usage:
  python scripts/eval_downstream.py \
    --checkpoint_path outputs/plan001-exec-only-s42/checkpoint-500 \
    --data_path data/test_held_out.jsonl \
    --output_path results/plan014_downstream.json \
    --num_generations 8 --workers 4
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics

from scipy.stats import spearmanr, mannwhitneyu
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.solvers.factory import get_solver
from src.sta.ablation import run_sta
from src.sta.isr import compute_isr
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


def load_jsonl(path: str) -> list[dict]:
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_model(model_path: str, base_model: str | None = None):
    import os
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer_path = model_path
    if not os.path.isfile(os.path.join(model_path, "tokenizer_config.json")):
        if base_model is None:
            raise ValueError(
                f"No tokenizer found in {model_path} and --base_model not set. "
                "Provide --base_model to load tokenizer from the base model."
            )
        tokenizer_path = base_model
        print(f"Tokenizer not found in checkpoint, loading from: {base_model}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def generate_k_formalizations(
    model, tokenizer, context: str, question: str,
    k: int = 8, temperature: float = 0.7, top_p: float = 0.9,
    max_new_tokens: int = 512,
) -> list[str]:
    import torch

    user_msg = USER_TEMPLATE.format(context=context, question=question)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=k,
            pad_token_id=tokenizer.pad_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    results = []
    for i in range(outputs.shape[0]):
        generated = outputs[i][input_len:]
        decoded = tokenizer.decode(generated, skip_special_tokens=True).strip()
        decoded = re.sub(r"<think>.*?</think>\s*", "", decoded, flags=re.DOTALL).strip()
        results.append(decoded)
    return results


def negate_conclusion(conclusion: str) -> str:
    c = conclusion.strip()
    if c.startswith("¬") or c.startswith("-"):
        return c[1:].strip()
    if c.startswith("(") and c.endswith(")"):
        return f"¬{c}"
    return f"¬({c})"


def eval_single_formalization(args_tuple):
    """Evaluate one formalization: parse, 3-way prove, full STA ISR.

    Designed for multiprocessing.Pool — takes a tuple of serializable args
    and creates its own solver instance.
    """
    formalization, gold_label, solver_name, timeout = args_tuple

    solver = get_solver(solver_name)

    result = {
        "formalization": formalization,
        "gold_label": gold_label,
        "predicted_label": "unknown",
        "correct": False,
        "proved_pos": False,
        "proved_neg": False,
        "isr": None,
        "n_components": 0,
        "n_necessary": 0,
        "parse_error": None,
    }

    try:
        premises, conclusion = _parse_formalization(formalization)
    except Exception as e:
        result["parse_error"] = str(e)
        result["correct"] = gold_label == "unknown"
        return result

    if not conclusion:
        result["parse_error"] = "empty conclusion"
        result["correct"] = gold_label == "unknown"
        return result

    result["premises"] = premises
    result["conclusion"] = conclusion

    # Prove premises → conclusion
    pos_result = solver.prove(premises, conclusion, timeout=timeout)
    result["proved_pos"] = pos_result.proved

    # Prove premises → ¬conclusion
    neg_c = negate_conclusion(conclusion)
    neg_result = solver.prove(premises, neg_c, timeout=timeout)
    result["proved_neg"] = neg_result.proved

    # 3-way classification
    if pos_result.proved and not neg_result.proved:
        result["predicted_label"] = "true"
    elif neg_result.proved and not pos_result.proved:
        result["predicted_label"] = "false"
    elif pos_result.proved and neg_result.proved:
        # Contradictory premises — default to true (entailment)
        result["predicted_label"] = "true"
    else:
        result["predicted_label"] = "unknown"

    result["correct"] = result["predicted_label"] == gold_label

    # Full STA ISR (only when original proof succeeded)
    if pos_result.proved:
        try:
            ablation_results = run_sta(
                premises, conclusion, solver, timeout=timeout,
                original_result=pos_result,
            )
            isr = compute_isr(ablation_results)
            result["isr"] = round(isr, 4)
            result["n_components"] = len(ablation_results)
            result["n_necessary"] = sum(
                1 for r in ablation_results if r.is_necessary
            )
        except Exception as e:
            result["isr"] = None
            result["parse_error"] = f"ISR error: {e}"

    return result


def compute_best_of_n(all_results: list[dict]) -> dict:
    """Compare Best-of-N selection strategies: Oracle, ISR-best, Random, Prove-first."""
    strategies = {
        "oracle": {"correct": 0, "total": 0},
        "isr_best": {"correct": 0, "total": 0},
        "random": {"correct_sum": 0.0, "total": 0},
        "prove_first": {"correct": 0, "total": 0},
    }

    rng = random.Random(42)

    for sample in all_results:
        gens = sample["generations"]
        if not gens:
            continue

        # Oracle: any generation correct?
        strategies["oracle"]["total"] += 1
        if any(g["correct"] for g in gens):
            strategies["oracle"]["correct"] += 1

        # ISR-best: generation with highest ISR
        gens_with_isr = [g for g in gens if g.get("isr") is not None]
        strategies["isr_best"]["total"] += 1
        if gens_with_isr:
            best = max(gens_with_isr, key=lambda g: g["isr"])
            if best["correct"]:
                strategies["isr_best"]["correct"] += 1
        else:
            if rng.choice(gens)["correct"]:
                strategies["isr_best"]["correct"] += 1

        # Prove-first: first generation that proved
        strategies["prove_first"]["total"] += 1
        proved_gens = [g for g in gens if g.get("proved_pos")]
        if proved_gens:
            if proved_gens[0]["correct"]:
                strategies["prove_first"]["correct"] += 1
        else:
            if rng.choice(gens)["correct"]:
                strategies["prove_first"]["correct"] += 1

        # Random: expected accuracy
        strategies["random"]["total"] += 1
        n_correct = sum(1 for g in gens if g["correct"])
        strategies["random"]["correct_sum"] += n_correct / len(gens)

    results = {}
    for name, stats in strategies.items():
        total = stats["total"]
        if name == "random":
            acc = stats["correct_sum"] / total if total else 0.0
            results[name] = {"accuracy": round(acc, 4), "total": total}
        else:
            acc = stats["correct"] / total if total else 0.0
            results[name] = {
                "accuracy": round(acc, 4),
                "correct": stats["correct"],
                "total": total,
            }
    return results


def statistical_tests(flat: list[dict]) -> dict:
    """Spearman, Mann-Whitney U, bootstrap CI for ISR-accuracy relationship."""
    with_isr = [(r["isr"], int(r["correct"])) for r in flat if r.get("isr") is not None]
    stats_out = {}

    if len(with_isr) >= 5:
        isrs, labels = zip(*with_isr)
        rho, p = spearmanr(isrs, labels)
        stats_out["spearman"] = {"rho": round(rho, 4), "p": round(p, 6), "n": len(with_isr)}

    sorted_by_isr = sorted(with_isr, key=lambda x: x[0])
    q_size = len(sorted_by_isr) // 4
    if q_size >= 2:
        q1 = [c for _, c in sorted_by_isr[:q_size]]
        q4 = [c for _, c in sorted_by_isr[-q_size:]]
        u_stat, u_p = mannwhitneyu(q4, q1, alternative="greater")
        stats_out["mannwhitney_q4_vs_q1"] = {
            "U": round(float(u_stat), 2), "p": round(float(u_p), 6),
            "q1_acc": round(sum(q1) / len(q1), 4), "q4_acc": round(sum(q4) / len(q4), 4),
        }

    return stats_out


def bootstrap_bon_ci(all_results: list[dict], n_boot: int = 1000, seed: int = 42) -> dict:
    """Bootstrap 95% CI for ISR-best minus random accuracy difference."""
    import random as _rng
    rng = _rng.Random(seed)
    diffs = []
    samples_with_isr = []
    for s in all_results:
        gens = s["generations"]
        gens_isr = [g for g in gens if g.get("isr") is not None]
        if gens_isr:
            samples_with_isr.append(s)
    if len(samples_with_isr) < 5:
        return {}
    for _ in range(n_boot):
        boot = rng.choices(samples_with_isr, k=len(samples_with_isr))
        isr_correct = 0
        rand_correct = 0.0
        for s in boot:
            gens = s["generations"]
            gens_isr = [g for g in gens if g.get("isr") is not None]
            best = max(gens_isr, key=lambda g: g["isr"])
            isr_correct += int(best["correct"])
            n_corr = sum(1 for g in gens if g["correct"])
            rand_correct += n_corr / len(gens)
        n = len(boot)
        diffs.append(isr_correct / n - rand_correct / n)
    diffs.sort()
    lo = diffs[int(0.025 * len(diffs))]
    hi = diffs[int(0.975 * len(diffs))]
    mean_diff = sum(diffs) / len(diffs)
    return {"mean_diff": round(mean_diff, 4), "ci_95": [round(lo, 4), round(hi, 4)], "n_boot": n_boot}


def aggregate_results(all_results: list[dict]) -> dict:
    flat = []
    for sample in all_results:
        for gen in sample["generations"]:
            flat.append({**gen, "sample_id": sample["id"]})

    total = len(flat)
    correct = sum(1 for r in flat if r["correct"])
    overall_acc = correct / total if total else 0.0

    # By gold label
    by_label: dict[str, dict] = {}
    for r in flat:
        gl = r["gold_label"]
        if gl not in by_label:
            by_label[gl] = {"correct": 0, "total": 0}
        by_label[gl]["total"] += 1
        if r["correct"]:
            by_label[gl]["correct"] += 1
    for stats in by_label.values():
        stats["accuracy"] = round(stats["correct"] / stats["total"], 4) if stats["total"] else 0.0

    # ISR buckets: high > 0.7, mid 0.3-0.7, low < 0.3
    with_isr = [r for r in flat if r.get("isr") is not None]
    buckets = {"high": [], "mid": [], "low": []}
    for r in with_isr:
        isr_val = r["isr"]
        if isr_val > 0.7:
            buckets["high"].append(r)
        elif isr_val >= 0.3:
            buckets["mid"].append(r)
        else:
            buckets["low"].append(r)

    bucket_stats = {}
    for bname, items in buckets.items():
        if items:
            bc = sum(1 for r in items if r["correct"])
            bucket_stats[bname] = {
                "n": len(items),
                "correct": bc,
                "accuracy": round(bc / len(items), 4),
                "mean_isr": round(statistics.mean(r["isr"] for r in items), 4),
            }
        else:
            bucket_stats[bname] = {"n": 0, "correct": 0, "accuracy": 0.0, "mean_isr": 0.0}

    # ISR quartile analysis (for samples with ISR)
    if len(with_isr) >= 4:
        sorted_by_isr = sorted(with_isr, key=lambda r: r["isr"])
        q_size = len(sorted_by_isr) // 4
        quartiles = {}
        for qi, qname in enumerate(["Q1_lowest", "Q2", "Q3", "Q4_highest"]):
            start = qi * q_size
            end = start + q_size if qi < 3 else len(sorted_by_isr)
            q_items = sorted_by_isr[start:end]
            qc = sum(1 for r in q_items if r["correct"])
            quartiles[qname] = {
                "n": len(q_items),
                "correct": qc,
                "accuracy": round(qc / len(q_items), 4) if q_items else 0.0,
                "isr_range": [round(q_items[0]["isr"], 4), round(q_items[-1]["isr"], 4)] if q_items else [],
            }
    else:
        quartiles = {}

    stat_tests = statistical_tests(flat)
    boot_ci = bootstrap_bon_ci(all_results)

    bon = compute_best_of_n(all_results)

    return {
        "n_samples": len(all_results),
        "n_generations": total,
        "overall_accuracy": round(overall_acc, 4),
        "by_gold_label": by_label,
        "by_isr_bucket": bucket_stats,
        "by_isr_quartile": quartiles,
        "n_with_isr": len(with_isr),
        "n_proved": sum(1 for r in flat if r.get("proved_pos")),
        "n_parse_errors": sum(1 for r in flat if r.get("parse_error")),
        "best_of_n": bon,
        "statistical_tests": stat_tests,
        "bootstrap_bon_ci": boot_ci,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Downstream NLI validation via Prover9 entailment"
    )
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to fine-tuned checkpoint directory")
    parser.add_argument("--base_model", type=str, default=None,
                        help="Base model path for tokenizer (when checkpoint lacks tokenizer files)")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to held-out test JSONL with NLI labels")
    parser.add_argument("--output_path", type=str,
                        default="results/downstream_eval.json")
    parser.add_argument("--solver", type=str, default="prover9")
    parser.add_argument("--num_generations", "-K", type=int, default=8,
                        help="Number of formalizations to generate per sample")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=5,
                        help="Prover9 per-query timeout in seconds")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers for CPU-bound solver evaluation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = load_jsonl(args.data_path)
    print(f"Loaded {len(data)} problems from {args.data_path}")

    print(f"Loading model from {args.checkpoint_path} ...")
    model, tokenizer = load_model(args.checkpoint_path, base_model=args.base_model)
    print("Model loaded.")

    # --- Phase 1: GPU inference ---
    print(f"\n=== Phase 1: Inference (K={args.num_generations}, T={args.temperature}) ===")
    all_generations: list[tuple[dict, list[str]]] = []
    t_infer = time.time()

    for i, problem in enumerate(data):
        pid = problem.get("id", i)
        print(f"[{i+1}/{len(data)}] {pid} ...", end=" ", flush=True)
        gens = generate_k_formalizations(
            model, tokenizer,
            context=problem["context"],
            question=problem["question"],
            k=args.num_generations,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )
        all_generations.append((problem, gens))
        print(f"{len(gens)} generated")

    t_infer = time.time() - t_infer
    print(f"Inference done: {t_infer:.1f}s")

    del model
    torch.cuda.empty_cache()

    # --- Phase 2+3: solver eval (CPU) ---
    print(f"\n=== Phase 2+3: ISR + Label Prediction (workers={args.workers}) ===")

    eval_tasks = []
    for problem, gens in all_generations:
        gold = problem["answer"].strip().lower()
        for gen_text in gens:
            eval_tasks.append((gen_text, gold, args.solver, args.timeout))

    print(f"Evaluating {len(eval_tasks)} formalizations ...")
    t_eval = time.time()

    if args.workers > 1:
        with Pool(args.workers) as pool:
            eval_results = pool.map(eval_single_formalization, eval_tasks)
    else:
        eval_results = []
        for j, task in enumerate(eval_tasks):
            if (j + 1) % 50 == 0 or j == 0:
                print(f"  [{j+1}/{len(eval_tasks)}]", flush=True)
            eval_results.append(eval_single_formalization(task))

    t_eval = time.time() - t_eval
    print(f"Evaluation done: {t_eval:.1f}s")

    # Reassemble per sample
    all_results = []
    idx = 0
    for problem, gens in all_generations:
        sample_gens = eval_results[idx : idx + len(gens)]
        idx += len(gens)
        all_results.append({
            "id": problem.get("id", ""),
            "gold_label": problem["answer"].strip().lower(),
            "context": problem["context"],
            "question": problem["question"],
            "generations": sample_gens,
        })

    summary = aggregate_results(all_results)
    summary["inference_time_s"] = round(t_infer, 1)
    summary["eval_time_s"] = round(t_eval, 1)
    summary["checkpoint_path"] = args.checkpoint_path
    summary["solver"] = args.solver
    summary["num_generations"] = args.num_generations
    summary["temperature"] = args.temperature

    output = {"summary": summary, "results": all_results}

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Results: {args.output_path}")
    print(f"  Samples:      {summary['n_samples']}")
    print(f"  Generations:  {summary['n_generations']}")
    print(f"  Overall acc:  {summary['overall_accuracy']:.1%}")
    print(f"  Proved:       {summary['n_proved']}/{summary['n_generations']}")
    print(f"  Parse errors: {summary['n_parse_errors']}")
    print(f"\n  By gold label:")
    for label, s in summary["by_gold_label"].items():
        print(f"    {label}: {s['correct']}/{s['total']} ({s['accuracy']:.1%})")
    print(f"\n  By ISR bucket (n={summary['n_with_isr']} with ISR):")
    for bname, s in summary["by_isr_bucket"].items():
        print(f"    {bname} (μ={s['mean_isr']:.2f}): {s['correct']}/{s['n']} ({s['accuracy']:.1%})")
    if summary["by_isr_quartile"]:
        print(f"\n  By ISR quartile:")
        for qname, s in summary["by_isr_quartile"].items():
            r = f"[{s['isr_range'][0]:.2f},{s['isr_range'][1]:.2f}]" if s["isr_range"] else "[]"
            print(f"    {qname} {r}: {s['correct']}/{s['n']} ({s['accuracy']:.1%})")
    print(f"\n  Best-of-N:")
    for sname, s in summary["best_of_n"].items():
        print(f"    {sname}: {s['accuracy']:.1%}")
    if summary.get("statistical_tests"):
        print(f"\n  Statistical tests:")
        st = summary["statistical_tests"]
        if "spearman" in st:
            print(f"    Spearman ISR-correct: rho={st['spearman']['rho']:.3f} p={st['spearman']['p']:.4f}")
        if "mannwhitney_q4_vs_q1" in st:
            mw = st["mannwhitney_q4_vs_q1"]
            print(f"    Mann-Whitney Q4>Q1: U={mw['U']:.0f} p={mw['p']:.4f} (Q1={mw['q1_acc']:.1%} Q4={mw['q4_acc']:.1%})")
    if summary.get("bootstrap_bon_ci"):
        bc = summary["bootstrap_bon_ci"]
        print(f"    Bootstrap BoN ISR-best vs random: diff={bc['mean_diff']:.3f} 95%CI={bc['ci_95']}")
    print(f"\n  Time: infer={t_infer:.1f}s eval={t_eval:.1f}s")


if __name__ == "__main__":
    main()
