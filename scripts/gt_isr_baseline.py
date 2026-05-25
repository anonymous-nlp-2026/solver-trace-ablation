"""
Ground-truth ISR baseline: use dataset's human-annotated FOL to compute ISR.
No model inference — pure solver calls.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, "/root/solver-trace-ablation")

from src.solvers.factory import get_solver
from src.sta.ablation import run_sta
from src.sta.isr import compute_isr


DATA_PATH = "/root/solver-trace-ablation/data/proofwriter_fol_100.jsonl"
OUTPUT_PATH = "/root/solver-trace-ablation/outputs/gt_isr_baseline.json"
TIMEOUT = 10


def load_data(path: str) -> list[dict]:
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def has_xor(premises: list[str], conclusion: str) -> bool:
    for p in premises:
        if "⊕" in p:
            return True
    return "⊕" in conclusion


def main():
    data = load_data(DATA_PATH)
    print(f"Loaded {len(data)} problems")

    solver = get_solver("prover9")
    print(f"Solver: {type(solver).__name__}")

    results = []
    isr_scores = []
    proved_ids = []
    failed_ids = []
    error_ids = []
    xor_samples = {"total": 0, "proved": 0, "isr_scores": []}

    t_start = time.time()

    for i, problem in enumerate(data):
        pid = problem.get("id", i)
        premises = problem["fol_premises"]
        conclusion = problem["fol_conclusion"]
        answer = problem.get("answer", "")
        is_xor = has_xor(premises, conclusion)

        if is_xor:
            xor_samples["total"] += 1

        print(f"[{i+1:3d}/100] {pid:15s} ({len(premises)} premises, answer={answer})", end=" ", flush=True)

        try:
            t0 = time.time()
            original = solver.prove(premises, conclusion, timeout=TIMEOUT)
            ablation_results = run_sta(premises, conclusion, solver, timeout=TIMEOUT, original_result=original)
            elapsed = time.time() - t0

            if original.proved:
                isr = compute_isr(ablation_results)
                isr_scores.append(isr)
                proved_ids.append(pid)
                if is_xor:
                    xor_samples["proved"] += 1
                    xor_samples["isr_scores"].append(isr)

                n_components = len(ablation_results)
                n_necessary = sum(1 for r in ablation_results if r.is_necessary)
                print(f"PROVED  ISR={isr:.3f} ({n_necessary}/{n_components}) {elapsed:.1f}s")
            else:
                failed_ids.append(pid)
                print(f"FAILED  {elapsed:.1f}s")

            record = {
                "index": i,
                "id": pid,
                "answer": answer,
                "n_premises": len(premises),
                "premises": premises,
                "conclusion": conclusion,
                "has_xor": is_xor,
                "proved": original.proved,
                "solver_success": original.success,
                "isr": round(isr, 4) if original.proved else None,
                "n_components": len(ablation_results),
                "n_necessary": sum(1 for r in ablation_results if r.is_necessary) if original.proved else None,
                "ablation_details": [
                    {
                        "type": ar.component.type,
                        "content": ar.component.content[:80],
                        "is_necessary": ar.is_necessary,
                        "original_proved": ar.original_proved,
                        "ablated_proved": ar.ablated_proved,
                    }
                    for ar in ablation_results
                ],
                "time_seconds": round(elapsed, 2),
                "error": None,
            }
        except Exception as e:
            error_ids.append(pid)
            tb = traceback.format_exc()
            print(f"ERROR: {e}")
            record = {
                "index": i,
                "id": pid,
                "answer": answer,
                "n_premises": len(premises),
                "premises": premises,
                "conclusion": conclusion,
                "has_xor": is_xor,
                "proved": False,
                "solver_success": False,
                "isr": None,
                "n_components": 0,
                "n_necessary": None,
                "ablation_details": [],
                "time_seconds": 0,
                "error": str(e),
            }

        results.append(record)

    elapsed_total = time.time() - t_start

    # --- Statistics ---
    n = len(data)
    n_proved = len(proved_ids)
    n_failed = len(failed_ids)
    n_errors = len(error_ids)
    n_solver_ran = n - n_errors

    mean_isr = statistics.mean(isr_scores) if isr_scores else 0.0
    std_isr = statistics.stdev(isr_scores) if len(isr_scores) > 1 else 0.0
    median_isr = statistics.median(isr_scores) if isr_scores else 0.0
    min_isr = min(isr_scores) if isr_scores else 0.0
    max_isr = max(isr_scores) if isr_scores else 0.0

    # By answer type
    by_answer = {}
    for r in results:
        a = r["answer"]
        if a not in by_answer:
            by_answer[a] = {"isr_scores": [], "proved": 0, "total": 0}
        by_answer[a]["total"] += 1
        if r["proved"]:
            by_answer[a]["proved"] += 1
            if r["isr"] is not None:
                by_answer[a]["isr_scores"].append(r["isr"])

    answer_stats = {}
    for a, s in by_answer.items():
        answer_stats[a] = {
            "count": s["total"],
            "proved": s["proved"],
            "prove_rate": round(s["proved"] / s["total"], 4) if s["total"] else 0.0,
            "mean_isr": round(statistics.mean(s["isr_scores"]), 4) if s["isr_scores"] else 0.0,
            "std_isr": round(statistics.stdev(s["isr_scores"]), 4) if len(s["isr_scores"]) > 1 else 0.0,
        }

    # ISR bucket distribution
    buckets = {"[0, 0.2)": 0, "[0.2, 0.4)": 0, "[0.4, 0.6)": 0, "[0.6, 0.8)": 0, "[0.8, 1.0]": 0}
    for isr in isr_scores:
        if isr < 0.2:
            buckets["[0, 0.2)"] += 1
        elif isr < 0.4:
            buckets["[0.2, 0.4)"] += 1
        elif isr < 0.6:
            buckets["[0.4, 0.6)"] += 1
        elif isr < 0.8:
            buckets["[0.6, 0.8)"] += 1
        else:
            buckets["[0.8, 1.0]"] += 1

    # XOR stats
    xor_stats = {
        "total_xor_samples": xor_samples["total"],
        "xor_proved": xor_samples["proved"],
        "xor_prove_rate": round(xor_samples["proved"] / xor_samples["total"], 4) if xor_samples["total"] else 0.0,
        "xor_mean_isr": round(statistics.mean(xor_samples["isr_scores"]), 4) if xor_samples["isr_scores"] else 0.0,
    }

    # Top/bottom cases
    proved_results = [r for r in results if r["proved"] and r["isr"] is not None]
    proved_results_sorted = sorted(proved_results, key=lambda r: r["isr"], reverse=True)
    top2 = proved_results_sorted[:2] if len(proved_results_sorted) >= 2 else proved_results_sorted
    bottom2 = proved_results_sorted[-2:] if len(proved_results_sorted) >= 2 else proved_results_sorted

    # Comparison with model baseline
    comparison = {
        "model_baseline": {
            "prove_rate": 0.12,
            "mean_isr": 0.064,
            "source": "phase0_baseline (Qwen3-4B)",
        },
        "gt_baseline": {
            "prove_rate": round(n_proved / n, 4) if n else 0.0,
            "mean_isr": round(mean_isr, 4),
        },
        "improvement": {
            "prove_rate_delta": round((n_proved / n) - 0.12, 4) if n else 0.0,
            "isr_delta": round(mean_isr - 0.064, 4),
        },
    }

    summary = {
        "n_problems": n,
        "gt_prove_rate": round(n_proved / n, 4) if n else 0.0,
        "n_proved": n_proved,
        "n_failed": n_failed,
        "n_errors": n_errors,
        "solver_execution_rate": round(n_solver_ran / n, 4) if n else 0.0,
        "gt_mean_isr": round(mean_isr, 4),
        "gt_isr_std": round(std_isr, 4),
        "gt_isr_median": round(median_isr, 4),
        "gt_isr_min": round(min_isr, 4),
        "gt_isr_max": round(max_isr, 4),
        "by_answer_type": answer_stats,
        "isr_bucket_distribution": buckets,
        "xor_stats": xor_stats,
        "comparison_with_model": comparison,
        "elapsed_seconds": round(elapsed_total, 1),
        "solver": "Prover9Solver",
        "timeout": TIMEOUT,
    }

    output = {
        "summary": summary,
        "top_isr_cases": top2,
        "bottom_isr_cases": bottom2,
        "results": results,
    }

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # --- Print ---
    print(f"\n{'='*60}")
    print(f"Ground-Truth ISR Baseline Results")
    print(f"{'='*60}")
    print(f"  Problems:           {n}")
    print(f"  Solver ran:         {n_solver_ran}/{n} ({summary['solver_execution_rate']:.1%})")
    print(f"  Proved:             {n_proved}/{n} ({summary['gt_prove_rate']:.1%})")
    print(f"  Errors:             {n_errors}")
    print(f"  GT Mean ISR:        {mean_isr:.4f} ± {std_isr:.4f}")
    print(f"  GT Median ISR:      {median_isr:.4f}")
    print(f"  GT ISR Range:       [{min_isr:.4f}, {max_isr:.4f}]")
    print(f"\n  ISR Bucket Distribution (proved samples only):")
    for bucket, count in buckets.items():
        pct = count / len(isr_scores) * 100 if isr_scores else 0
        bar = "█" * int(pct / 2)
        print(f"    {bucket:12s}: {count:3d} ({pct:5.1f}%) {bar}")
    print(f"\n  By Answer Type:")
    for a, s in answer_stats.items():
        print(f"    {a:8s}: n={s['count']}, proved={s['proved']} ({s['prove_rate']:.1%}), ISR={s['mean_isr']:.4f}")
    print(f"\n  XOR Stats:")
    print(f"    Samples with ⊕:   {xor_stats['total_xor_samples']}")
    print(f"    XOR proved:       {xor_stats['xor_proved']} ({xor_stats['xor_prove_rate']:.1%})")
    print(f"    XOR mean ISR:     {xor_stats['xor_mean_isr']:.4f}")
    print(f"\n  Comparison with Model Baseline:")
    print(f"    Model prove rate: {comparison['model_baseline']['prove_rate']:.1%} → GT: {comparison['gt_baseline']['prove_rate']:.1%} (Δ={comparison['improvement']['prove_rate_delta']:+.1%})")
    print(f"    Model mean ISR:   {comparison['model_baseline']['mean_isr']:.4f} → GT: {comparison['gt_baseline']['mean_isr']:.4f} (Δ={comparison['improvement']['isr_delta']:+.4f})")
    print(f"\n  Time: {elapsed_total:.1f}s")
    print(f"  Output: {OUTPUT_PATH}")

    # Print top/bottom cases
    print(f"\n  Top ISR Cases:")
    for r in top2:
        print(f"    {r['id']}: ISR={r['isr']:.4f}, {r['n_necessary']}/{r['n_components']} necessary, answer={r['answer']}")
    print(f"\n  Bottom ISR Cases:")
    for r in bottom2:
        print(f"    {r['id']}: ISR={r['isr']:.4f}, {r['n_necessary']}/{r['n_components']} necessary, answer={r['answer']}")


if __name__ == "__main__":
    main()
