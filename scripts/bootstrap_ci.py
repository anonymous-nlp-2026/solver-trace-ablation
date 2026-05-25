"""Bootstrap confidence intervals for hacking rates and ISR metrics.

Reads eval_sta JSON outputs, classifies each proved sample into hacking
modes (H7/H1/genuine), performs bootstrap resampling (n=10000) on
per-sample results, outputs 95% CIs and pairwise comparison p-values.

Hacking mode classification:
  H7  — single premise identical to conclusion (tautological)
  H1  — conclusion appears as one of multiple premises (conclusion embedding)
  N-H — neither H7 nor H1 (genuine proof with reasoning)

Usage:
  python scripts/bootstrap_ci.py --output_path /root/autodl-tmp/outputs/bootstrap_ci_results.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

EVAL_FILES = {
    "exec_only_s42": "/root/autodl-tmp/outputs/eval_sta_plan001_r2.json",
    "isr_b01_s42": "/root/autodl-tmp/eval_results_backup/plan002_additive_b01_s42_eval_sta_results.json",
    "random_b01_s42": "/root/autodl-tmp/outputs/eval_sta_plan018_additive_random_b01_s42.json",
    "isr_b03_s43": "/root/autodl-tmp/outputs/eval_sta_exp1_isr_b03_s43_r3.json",
    "random_b03_s43": "/root/autodl-tmp/outputs/eval_sta_exp1_random_b03_s43_r4.json",
    "exec_only_s44": "/root/autodl-tmp/outputs/eval_sta_plan001_exec_only_s44.json",
}

N_BOOTSTRAP = 10000
SEED = 42


def classify_sample(r):
    """Classify a proved sample into H7, H1, or genuine."""
    det = r["details"]
    if "error" in det or "proof_failed" in det:
        return "not_proved"

    components = det.get("components", [])
    concl_parts = [c["content"].strip() for c in components if c["type"] == "conclusion_whole"]
    concl_text = concl_parts[0] if concl_parts else det.get("conclusion", "").strip()
    premises = [c["content"].strip() for c in components if c["type"] == "premise"]

    if not concl_text:
        return "genuine"

    if concl_text in premises:
        if len(premises) <= 1:
            return "H7"
        return "H1"

    return "genuine"


def extract_per_sample(results):
    """Extract per-sample arrays from eval results."""
    n = len(results)
    proved = np.zeros(n, dtype=bool)
    isr = np.zeros(n)
    mode = []  # 'H7', 'H1', 'genuine', or 'not_proved'

    for i, r in enumerate(results):
        if r["execution_reward"] > 0:
            proved[i] = True
            isr[i] = r["sta_reward"]
            mode.append(classify_sample(r))
        else:
            mode.append("not_proved")

    return proved, isr, np.array(mode)


def bootstrap_ci(data, stat_fn, n_boot=N_BOOTSTRAP, rng=None):
    """Compute bootstrap 95% CI for stat_fn(data)."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    n = len(data)
    if n == 0:
        return {"point": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "n": 0}

    point = float(stat_fn(data))
    boot_stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_stats[b] = stat_fn(data[idx])

    ci_lo, ci_hi = np.percentile(boot_stats, [2.5, 97.5])
    return {
        "point": round(point, 4),
        "ci_lower": round(float(ci_lo), 4),
        "ci_upper": round(float(ci_hi), 4),
        "n": int(n),
    }


def bootstrap_rate(labels, target, n_boot=N_BOOTSTRAP, rng=None):
    """Bootstrap CI for the rate of 'target' in labels."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    data = (labels == target).astype(float)
    return bootstrap_ci(data, np.mean, n_boot, rng)


def permutation_test(a, b, stat_fn=np.mean, n_perm=10000, rng=None):
    """Two-sided permutation test for difference in stat_fn."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    observed = stat_fn(a) - stat_fn(b)
    combined = np.concatenate([a, b])
    na = len(a)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(combined)
        perm_diff = stat_fn(combined[:na]) - stat_fn(combined[na:])
        if abs(perm_diff) >= abs(observed):
            count += 1
    p = (count + 1) / (n_perm + 1)  # +1 for continuity correction
    return {
        "diff": round(float(observed), 4),
        "p_value": round(float(p), 4),
        "significant": p < 0.05,
    }


def analyze_experiment(results, rng):
    """Compute bootstrap CIs for one experiment."""
    proved, isr, mode = extract_per_sample(results)
    n_proved = int(proved.sum())

    # Subset to proved samples for hacking mode analysis
    proved_mask = proved.astype(bool)
    proved_modes = mode[proved_mask]
    proved_isr = isr[proved_mask]

    result = {
        "n_total": len(results),
        "n_proved": n_proved,
        "prove_rate": bootstrap_ci(proved.astype(float), np.mean, rng=rng),
    }

    if n_proved == 0:
        for k in ["h7_rate", "h1_rate", "genuine_rate", "isr_star"]:
            result[k] = {"point": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "n": 0}
        return result

    result["h7_rate"] = bootstrap_rate(proved_modes, "H7", rng=rng)
    result["h1_rate"] = bootstrap_rate(proved_modes, "H1", rng=rng)
    result["genuine_rate"] = bootstrap_rate(proved_modes, "genuine", rng=rng)
    result["isr_star"] = bootstrap_ci(proved_isr, np.mean, rng=rng)

    # Raw counts for reference
    result["counts"] = {
        "H7": int((proved_modes == "H7").sum()),
        "H1": int((proved_modes == "H1").sum()),
        "genuine": int((proved_modes == "genuine").sum()),
    }

    return result


def compute_comparisons(experiments_data):
    """Compute pairwise comparisons between key experiments."""
    comparisons = {}
    rng = np.random.default_rng(SEED + 1)

    def get_proved_arrays(name):
        results = experiments_data[name]
        proved, isr, mode = extract_per_sample(results)
        m = proved.astype(bool)
        return mode[m], isr[m]

    def safe_compare(name_a, name_b, field, target=None):
        if name_a not in experiments_data or name_b not in experiments_data:
            return {"error": f"missing experiment", "available": list(experiments_data.keys())}
        modes_a, isr_a = get_proved_arrays(name_a)
        modes_b, isr_b = get_proved_arrays(name_b)

        if field == "genuine_rate":
            a = (modes_a == "genuine").astype(float)
            b = (modes_b == "genuine").astype(float)
        elif field == "h7_rate":
            a = (modes_a == "H7").astype(float)
            b = (modes_b == "H7").astype(float)
        elif field == "h1_rate":
            a = (modes_a == "H1").astype(float)
            b = (modes_b == "H1").astype(float)
        elif field == "isr":
            a, b = isr_a, isr_b
        elif field == "nh_rate":
            a = (modes_a != "not_proved").astype(float)
            genuine_a = (modes_a == "genuine").astype(float)
            a = genuine_a
            b = (modes_b == "genuine").astype(float)
        else:
            return {"error": f"unknown field {field}"}

        return permutation_test(a, b, rng=rng)

    # ISR β=0.1 genuine rate vs Random β=0.1 genuine rate
    comparisons["isr_b01_vs_random_b01_genuine"] = safe_compare(
        "isr_b01_s42", "random_b01_s42", "genuine_rate"
    )

    # ISR β=0.1 H7 rate vs Random β=0.1 H7 rate
    comparisons["isr_b01_vs_random_b01_h7"] = safe_compare(
        "isr_b01_s42", "random_b01_s42", "h7_rate"
    )

    # Exec-only H1 rate vs ISR β=0.1 H1 rate
    comparisons["exec_vs_isr_b01_h1"] = safe_compare(
        "exec_only_s42", "isr_b01_s42", "h1_rate"
    )

    # Exec-only genuine rate vs ISR β=0.1 genuine rate
    comparisons["exec_vs_isr_b01_genuine"] = safe_compare(
        "exec_only_s42", "isr_b01_s42", "genuine_rate"
    )

    # ISR β=0.1 ISR* vs Random β=0.1 ISR*
    comparisons["isr_b01_vs_random_b01_isr"] = safe_compare(
        "isr_b01_s42", "random_b01_s42", "isr"
    )

    # ISR β=0.3 genuine rate vs Random β=0.3 genuine rate
    comparisons["isr_b03_vs_random_b03_genuine"] = safe_compare(
        "isr_b03_s43", "random_b03_s43", "genuine_rate"
    )

    # Exec-only genuine vs Random β=0.1 genuine
    comparisons["exec_vs_random_b01_genuine"] = safe_compare(
        "exec_only_s42", "random_b01_s42", "genuine_rate"
    )

    return comparisons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_path",
        default="/root/autodl-tmp/outputs/bootstrap_ci_results.json",
    )
    parser.add_argument("--n_bootstrap", type=int, default=N_BOOTSTRAP)
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)
    experiments_data = {}
    experiment_results = {}

    for name, path in EVAL_FILES.items():
        p = Path(path)
        if not p.exists():
            print(f"SKIP {name}: {path} not found")
            continue
        data = json.loads(p.read_text())
        experiments_data[name] = data["results"]
        print(f"Loaded {name}: {data['summary']['n_proved']}/{data['summary']['n_problems']} proved")

        result = analyze_experiment(data["results"], rng)
        experiment_results[name] = result

        # Print summary
        pr = result["prove_rate"]
        print(f"  PR:      {pr['point']:.3f} [{pr['ci_lower']:.3f}, {pr['ci_upper']:.3f}]")
        if result["n_proved"] > 0:
            h7 = result["h7_rate"]
            h1 = result["h1_rate"]
            gen = result["genuine_rate"]
            isr = result["isr_star"]
            print(f"  H7:      {h7['point']:.3f} [{h7['ci_lower']:.3f}, {h7['ci_upper']:.3f}] (n={result['counts']['H7']})")
            print(f"  H1:      {h1['point']:.3f} [{h1['ci_lower']:.3f}, {h1['ci_upper']:.3f}] (n={result['counts']['H1']})")
            print(f"  Genuine: {gen['point']:.3f} [{gen['ci_lower']:.3f}, {gen['ci_upper']:.3f}] (n={result['counts']['genuine']})")
            print(f"  ISR*:    {isr['point']:.3f} [{isr['ci_lower']:.3f}, {isr['ci_upper']:.3f}]")
        print()

    print("=" * 60)
    print("Computing pairwise comparisons...")
    comparisons = compute_comparisons(experiments_data)

    for name, result in comparisons.items():
        if "error" in result:
            print(f"  {name}: {result['error']}")
        else:
            sig = "*" if result["significant"] else ""
            print(f"  {name}: diff={result['diff']:+.4f}, p={result['p_value']:.4f}{sig}")

    output = {
        "experiments": experiment_results,
        "comparisons": comparisons,
        "config": {
            "n_bootstrap": args.n_bootstrap,
            "seed": SEED,
            "eval_files": EVAL_FILES,
            "classification": {
                "H7": "single premise identical to conclusion",
                "H1": "conclusion appears as one of multiple premises",
                "genuine": "no conclusion duplication in premises",
            },
        },
    }

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output_path}")


if __name__ == "__main__":
    main()
