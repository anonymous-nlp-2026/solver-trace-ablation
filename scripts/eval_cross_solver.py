"""Cross-Solver Agreement: compare Prover9 vs Z3 on FOL formalizations.

Reads eval_downstream.py output JSON, re-evaluates each formalization with
Z3, compares results against the original Prover9 evaluation, and reports
agreement statistics by ISR bucket.

Usage:
  python scripts/eval_cross_solver.py \
    --input_path results/plan014_downstream.json \
    --output_path results/cross_solver_agreement.json \
    --workers 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.solvers.factory import get_solver


def z3_eval_single(args_tuple):
    premises, conclusion, timeout = args_tuple
    solver = get_solver("z3")
    if not solver.is_available():
        return {"z3_success": False, "z3_proved": False, "error": "z3 not installed"}

    result = solver.prove(premises, conclusion, timeout=timeout)
    return {
        "z3_success": result.success,
        "z3_proved": result.proved,
        "z3_time": round(result.time_seconds, 4),
        "error": result.error,
    }


def classify_agreement(p9_proved: bool, z3_proved: bool) -> str:
    if p9_proved and z3_proved:
        return "both_proved"
    elif not p9_proved and not z3_proved:
        return "both_unproved"
    elif p9_proved and not z3_proved:
        return "p9_only"
    else:
        return "z3_only"


def bucket_for_isr(isr: float | None) -> str:
    if isr is None:
        return "no_isr"
    if isr > 0.7:
        return "high"
    elif isr >= 0.3:
        return "mid"
    else:
        return "low"


def main():
    parser = argparse.ArgumentParser(
        description="Cross-Solver Agreement: Prover9 vs Z3"
    )
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to eval_downstream.py output JSON")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output JSON path for cross-solver results")
    parser.add_argument("--timeout", type=int, default=5,
                        help="Z3 timeout per proof (seconds)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers")
    args = parser.parse_args()

    with open(args.input_path) as f:
        data = json.load(f)

    all_results = data["results"]

    tasks = []
    task_index = []
    for si, sample in enumerate(all_results):
        for gi, gen in enumerate(sample["generations"]):
            premises = gen.get("premises")
            conclusion = gen.get("conclusion")
            if premises and conclusion:
                tasks.append((premises, conclusion, args.timeout))
                task_index.append((si, gi))

    print(f"Evaluating {len(tasks)} formalizations with Z3 (workers={args.workers}) ...")
    t0 = time.time()

    if args.workers > 1:
        with Pool(args.workers) as pool:
            z3_results = pool.map(z3_eval_single, tasks)
    else:
        z3_results = [z3_eval_single(t) for t in tasks]

    elapsed = time.time() - t0
    print(f"Z3 evaluation done: {elapsed:.1f}s")

    for idx, (si, gi) in enumerate(task_index):
        gen = all_results[si]["generations"][gi]
        gen["z3_proved"] = z3_results[idx]["z3_proved"]
        gen["z3_success"] = z3_results[idx]["z3_success"]
        gen["z3_time"] = z3_results[idx].get("z3_time", 0.0)

    # Aggregate agreement statistics
    total = 0
    agreement_counts = {"both_proved": 0, "both_unproved": 0, "p9_only": 0, "z3_only": 0}
    by_bucket = {}

    for sample in all_results:
        for gen in sample["generations"]:
            if "z3_proved" not in gen:
                continue
            p9 = gen.get("proved_pos", False)
            z3 = gen["z3_proved"]
            isr = gen.get("isr")
            cat = classify_agreement(p9, z3)
            total += 1
            agreement_counts[cat] += 1

            bkt = bucket_for_isr(isr)
            if bkt not in by_bucket:
                by_bucket[bkt] = {"both_proved": 0, "both_unproved": 0, "p9_only": 0, "z3_only": 0, "n": 0}
            by_bucket[bkt][cat] += 1
            by_bucket[bkt]["n"] += 1

    agree = agreement_counts["both_proved"] + agreement_counts["both_unproved"]
    agreement_rate = agree / total if total else 0.0

    for bkt, stats in by_bucket.items():
        n = stats["n"]
        stats["agreement_rate"] = round((stats["both_proved"] + stats["both_unproved"]) / n, 4) if n else 0.0

    summary = {
        "total_evaluated": total,
        "agreement_rate": round(agreement_rate, 4),
        "agreement_counts": agreement_counts,
        "by_isr_bucket": by_bucket,
        "z3_eval_time_s": round(elapsed, 1),
        "input_path": args.input_path,
    }

    output = {"summary": summary, "results": all_results}
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Results: {args.output_path}")
    print(f"  Total evaluated: {total}")
    print(f"  Agreement rate:  {agreement_rate:.1%}")
    print(f"  Both proved:     {agreement_counts['both_proved']}")
    print(f"  Both unproved:   {agreement_counts['both_unproved']}")
    print(f"  Prover9 only:    {agreement_counts['p9_only']}")
    print(f"  Z3 only:         {agreement_counts['z3_only']}")
    print(f"\n  By ISR bucket:")
    for bkt in ["high", "mid", "low", "no_isr"]:
        if bkt in by_bucket:
            s = by_bucket[bkt]
            print(f"    {bkt}: {s['agreement_rate']:.1%} agreement ({s['n']} samples)")
    print(f"\n  Z3 eval time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
