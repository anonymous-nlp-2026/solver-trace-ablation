"""R16: O(1) vs O(n) ISR consistency analysis.

Compares the O(1) binary tautology check (used during training) against
the O(n) LOO ISR (used during eval) across multiple model checkpoints.
"""
import json
import sys
import time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.solvers.factory import get_solver

EVAL_RESULTS = {
    "exec_s42": "/root/autodl-tmp/outputs/d146_pf_exec_s42/eval_sta_results.json",
    "exec_s43": "/root/autodl-tmp/outputs/d146_pf_exec_s43/eval_sta_results.json",
    "exec_s44": "/root/autodl-tmp/outputs/d146_pf_exec_s44/eval_sta_results.json",
    "isr_s42": "/root/autodl-tmp/outputs/d146_pf_isr_s42/eval_sta_results.json",
    "isr_s43": "/root/autodl-tmp/outputs/d146_pf_isr_s43/eval_sta_results.json",
    "isr_s44": "/root/autodl-tmp/outputs/d146_pf_isr_s44/eval_sta_results.json",
}

OUTPUT_PATH = "/root/autodl-tmp/outputs/r16_o1_vs_on_consistency.json"


def load_proved_samples(path: str) -> list[dict]:
    """Load eval results, return only proved samples with their LOO ISR."""
    data = json.load(open(path))
    proved = []
    for r in data["results"]:
        if r.get("execution_reward", 0) != 1.0:
            continue
        details = r.get("details", {})
        premises = details.get("premises", [])
        conclusion = details.get("conclusion", "")
        if not premises or not conclusion:
            continue
        proved.append({
            "id": r.get("id", ""),
            "premises": premises,
            "conclusion": conclusion,
            "loo_isr": details.get("isr", 0.0),
            "n_components": details.get("n_components", 0),
            "n_necessary": details.get("n_necessary", 0),
        })
    return proved


def run_tautology_check(solver, conclusion: str, timeout: int = 5) -> bool:
    """Check if conclusion is provable from empty premises (tautology)."""
    result = solver.prove([], conclusion, timeout=timeout)
    return result.proved


def compute_binary_isr(is_tautology: bool) -> float:
    """O(1) binary ISR: 0 if tautology, 1 otherwise."""
    return 0.0 if is_tautology else 1.0


def main():
    solver = get_solver("prover9")
    print(f"Solver: {type(solver).__name__}")

    all_results = {}
    all_samples = []  # flat list for aggregate stats
    t_start = time.time()

    for name, path in EVAL_RESULTS.items():
        if not Path(path).exists():
            print(f"SKIP {name}: {path} not found")
            continue

        proved = load_proved_samples(path)
        print(f"\n{'='*60}")
        print(f"{name}: {len(proved)} proved samples")

        per_model_results = []
        for i, s in enumerate(proved):
            is_taut = run_tautology_check(solver, s["conclusion"])
            binary_isr = compute_binary_isr(is_taut)
            loo_isr = s["loo_isr"]

            # Agreement: binarize LOO (any ISR > 0 → 1, else → 0)
            loo_binary = 1.0 if loo_isr > 0 else 0.0
            agree = (binary_isr == loo_binary)

            record = {
                "id": s["id"],
                "premises": s["premises"],
                "conclusion": s["conclusion"],
                "loo_isr": loo_isr,
                "binary_isr": binary_isr,
                "is_tautology": is_taut,
                "loo_binary": loo_binary,
                "agree": agree,
                "diff": round(binary_isr - loo_isr, 4),
                "n_components": s["n_components"],
                "n_necessary": s["n_necessary"],
                "model": name,
            }
            per_model_results.append(record)
            all_samples.append(record)

            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(proved)}] done")

        # Per-model summary
        n = len(per_model_results)
        n_agree = sum(1 for r in per_model_results if r["agree"])
        n_taut = sum(1 for r in per_model_results if r["is_tautology"])

        loo_vals = [r["loo_isr"] for r in per_model_results]
        bin_vals = [r["binary_isr"] for r in per_model_results]

        mean_loo = sum(loo_vals) / n if n else 0
        mean_bin = sum(bin_vals) / n if n else 0
        mae = sum(abs(a - b) for a, b in zip(bin_vals, loo_vals)) / n if n else 0

        # Confusion matrix (binarized)
        tp = sum(1 for r in per_model_results if r["binary_isr"] == 1 and r["loo_binary"] == 1)
        tn = sum(1 for r in per_model_results if r["binary_isr"] == 0 and r["loo_binary"] == 0)
        fp = sum(1 for r in per_model_results if r["binary_isr"] == 1 and r["loo_binary"] == 0)
        fn = sum(1 for r in per_model_results if r["binary_isr"] == 0 and r["loo_binary"] == 1)

        # Case breakdown
        cases = {"b1_l1": 0, "b1_l_lt1": 0, "b0_l0": 0, "b0_l_gt0": 0}
        for r in per_model_results:
            bi, li = r["binary_isr"], r["loo_isr"]
            if bi == 1 and li == 1:
                cases["b1_l1"] += 1
            elif bi == 1 and li < 1:
                cases["b1_l_lt1"] += 1
            elif bi == 0 and li == 0:
                cases["b0_l0"] += 1
            elif bi == 0 and li > 0:
                cases["b0_l_gt0"] += 1

        model_summary = {
            "n_proved": n,
            "agreement_rate": round(n_agree / n, 4) if n else 0,
            "n_tautology": n_taut,
            "tautology_rate": round(n_taut / n, 4) if n else 0,
            "mean_loo_isr": round(mean_loo, 4),
            "mean_binary_isr": round(mean_bin, 4),
            "mae": round(mae, 4),
            "confusion": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
            "case_breakdown": cases,
        }
        all_results[name] = {"summary": model_summary, "samples": per_model_results}

        print(f"  Agreement: {n_agree}/{n} ({model_summary['agreement_rate']:.1%})")
        print(f"  Tautology: {n_taut}/{n} ({model_summary['tautology_rate']:.1%})")
        print(f"  MAE: {mae:.4f}")
        print(f"  Confusion: TP={tp} TN={tn} FP={fp} FN={fn}")
        print(f"  Cases: {cases}")

    elapsed = time.time() - t_start

    # Aggregate statistics
    n_total = len(all_samples)
    if n_total > 1:
        try:
            from scipy.stats import pearsonr, spearmanr
            loo_all = [r["loo_isr"] for r in all_samples]
            bin_all = [r["binary_isr"] for r in all_samples]
            pr, p_pval = pearsonr(bin_all, loo_all)
            sr, s_pval = spearmanr(bin_all, loo_all)
        except ImportError:
            pr = sr = p_pval = s_pval = None
    else:
        pr = sr = p_pval = s_pval = None

    n_agree_all = sum(1 for r in all_samples if r["agree"])
    mae_all = sum(abs(r["binary_isr"] - r["loo_isr"]) for r in all_samples) / n_total if n_total else 0

    # Aggregate confusion
    tp_a = sum(1 for r in all_samples if r["binary_isr"] == 1 and r["loo_binary"] == 1)
    tn_a = sum(1 for r in all_samples if r["binary_isr"] == 0 and r["loo_binary"] == 0)
    fp_a = sum(1 for r in all_samples if r["binary_isr"] == 1 and r["loo_binary"] == 0)
    fn_a = sum(1 for r in all_samples if r["binary_isr"] == 0 and r["loo_binary"] == 1)

    # By model type (exec vs ISR)
    by_type = {}
    for mtype in ["exec", "isr"]:
        samples = [r for r in all_samples if r["model"].startswith(mtype)]
        n = len(samples)
        if n == 0:
            continue
        n_ag = sum(1 for r in samples if r["agree"])
        mae_t = sum(abs(r["binary_isr"] - r["loo_isr"]) for r in samples) / n
        by_type[mtype] = {
            "n": n,
            "agreement_rate": round(n_ag / n, 4),
            "mae": round(mae_t, 4),
            "mean_loo_isr": round(sum(r["loo_isr"] for r in samples) / n, 4),
            "mean_binary_isr": round(sum(r["binary_isr"] for r in samples) / n, 4),
        }

    aggregate = {
        "n_total": n_total,
        "agreement_rate": round(n_agree_all / n_total, 4) if n_total else 0,
        "mae": round(mae_all, 4),
        "pearson_r": round(pr, 4) if pr is not None else None,
        "pearson_p": round(p_pval, 6) if p_pval is not None else None,
        "spearman_r": round(sr, 4) if sr is not None else None,
        "spearman_p": round(s_pval, 6) if s_pval is not None else None,
        "confusion": {"TP": tp_a, "TN": tn_a, "FP": fp_a, "FN": fn_a},
        "by_model_type": by_type,
        "elapsed_seconds": round(elapsed, 1),
    }

    output = {"aggregate": aggregate, "per_model": {k: v["summary"] for k, v in all_results.items()}}

    # Also save per-sample results for detailed inspection
    disagree_samples = [r for r in all_samples if not r["agree"]]
    output["disagreement_examples"] = disagree_samples[:20]

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"AGGREGATE ({n_total} samples)")
    print(f"  Agreement: {n_agree_all}/{n_total} ({aggregate['agreement_rate']:.1%})")
    print(f"  MAE: {mae_all:.4f}")
    print(f"  Pearson r: {pr}")
    print(f"  Spearman rho: {sr}")
    print(f"  Confusion: TP={tp_a} TN={tn_a} FP={fp_a} FN={fn_a}")
    print(f"  By type: {json.dumps(by_type, indent=2)}")
    print(f"  Disagreements: {len(disagree_samples)}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
