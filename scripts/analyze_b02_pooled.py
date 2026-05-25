#!/usr/bin/env python3
"""Pooled analysis for β=0.2 ISR GRPO across 3 seeds.

Reads eval_sta_results.json + hacking_taxonomy.json from each seed's
eval/ directory. Computes pooled statistics and compares to baselines.

Usage:
    python scripts/analyze_b02_pooled.py                    # all 3 seeds
    python scripts/analyze_b02_pooled.py --seeds 42         # s42 only
    python scripts/analyze_b02_pooled.py --seeds 42 43 44   # explicit
"""

import argparse
import json
import os
import sys
import statistics
from pathlib import Path
from collections import OrderedDict

OUTPUT_BASE = "/root/autodl-tmp/outputs"

SEED_DIR_MAP = {
    42: "sweep_isr_b02_s42_v2",
    43: "sweep_isr_b02_s43",
    44: "sweep_isr_b02_s44",
}

# Paper-authoritative baselines (from experiments.tex, Table 1)
BASELINES = OrderedDict([
    ("ISR β=0.1 (3-seed)", {
        "pr_mean": 0.77, "pr_std": 0.14,
        "genuine_pct": 71, "h7_pct": 18, "h1_pct": 6,
        "note": "2/3 seeds meta-Goodhart; ISR* misleading",
    }),
    ("ISR β=0.3 (3-seed)", {
        "pr_mean": 0.66, "pr_std": 0.42,
        "isr_mean": 0.39, "isr_std": 0.31,
        "genuine_pct": 50, "h7_pct": 37, "h1_pct": 12,
        "note": "extreme cross-seed instability",
    }),
    ("ISR β=0.05 (s42)", {
        "pr_mean": None,
        "note": "no eval data yet (sweep_isr_b005_s42)",
    }),
    ("Random β=0.1 (s42)", {
        "pr_mean": 0.79, "isr_mean": 0.131,
        "note": "control, no ISR-specific gaming",
    }),
    ("Random β=0.3 (s42)", {
        "pr_mean": 0.48, "isr_mean": 0.094,
        "note": "control",
    }),
    ("Exec-only 500 (3-seed)", {
        "pr_mean": 0.68, "pr_std": 0.47,
        "isr_mean": 0.24, "isr_std": 0.30,
        "note": "2/3 seeds degenerate",
    }),
])


def load_seed_data(seed):
    dirname = SEED_DIR_MAP.get(seed)
    if not dirname:
        return None, f"Unknown seed {seed}"
    
    base = os.path.join(OUTPUT_BASE, dirname)
    if not os.path.isdir(base):
        return None, f"Directory not found: {base}"
    
    eval_dir = os.path.join(base, "eval")
    if not os.path.isdir(eval_dir):
        return None, f"No eval/ directory (training may still be running): {base}"

    # Load eval_sta_results
    sta_path = os.path.join(eval_dir, "eval_sta_results.json")
    if not os.path.exists(sta_path):
        sta_path = os.path.join(eval_dir, "eval_results.json")
    if not os.path.exists(sta_path):
        return None, f"No eval results JSON in {eval_dir}"
    
    with open(sta_path) as f:
        sta_data = json.load(f)

    # Load hacking taxonomy
    tax_path = os.path.join(eval_dir, "hacking_taxonomy.json")
    tax_data = None
    if os.path.exists(tax_path):
        with open(tax_path) as f:
            raw = json.load(f)
            # taxonomy file may be keyed by dirname
            if dirname in raw:
                tax_data = raw[dirname]
            elif len(raw) == 1:
                tax_data = list(raw.values())[0]
            else:
                tax_data = raw

    # Load training log (last step)
    log_path = os.path.join(base, "reward_log.jsonl")
    last_step = None
    if os.path.exists(log_path):
        with open(log_path) as f:
            lines = f.readlines()
            if lines:
                last_step = json.loads(lines[-1].strip())

    return {
        "seed": seed,
        "dirname": dirname,
        "sta": sta_data,
        "taxonomy": tax_data,
        "last_step": last_step,
    }, None


def extract_metrics(data):
    s = data["sta"]["summary"]
    t = data["taxonomy"]
    
    m = {
        "seed": data["seed"],
        "n_problems": s["n_problems"],
        "n_proved": s["n_proved"],
        "prove_rate": s["prove_rate"],
        "mean_isr": s["mean_isr"],
        "isr_std": s["isr_std"],
        "median_isr": s["median_isr"],
        "proved_only_mean_isr": s.get("proved_only_mean_isr", s["mean_isr"]),
    }
    
    if t:
        m.update({
            "h7a": t.get("h7a_count", 0),
            "h7b": t.get("h7b_count", 0),
            "h7_total": t.get("h7_total_count", 0),
            "h1": t.get("h1_strict_count", 0),
            "h8": t.get("h8_conjunction_collapse_count", 0),
            "genuine": t.get("genuine_count", 0),
            "genuine_rate": t.get("genuine_rate", 0),
        })

    if data["last_step"]:
        m["train_final_step"] = data["last_step"].get("step")
        m["train_final_pr"] = data["last_step"].get("prove_rate")
        m["train_final_isr"] = data["last_step"].get("isr_mean")
    
    # ISR distribution from per-sample results
    results = data["sta"].get("results", [])
    isr_values = []
    for r in results:
        if r.get("execution_reward", 0) > 0:
            det = r.get("details", {})
            isr = det.get("isr", r.get("sta_reward"))
            if isr is not None:
                isr_values.append(isr)
    m["isr_values"] = isr_values
    
    return m


def pooled_stats(metrics_list):
    n = len(metrics_list)
    
    prs = [m["prove_rate"] for m in metrics_list]
    isrs = [m["mean_isr"] for m in metrics_list]
    proved_isrs = [m["proved_only_mean_isr"] for m in metrics_list]
    
    pool = {
        "n_seeds": n,
        "pr_mean": statistics.mean(prs),
        "pr_std": statistics.stdev(prs) if n > 1 else 0,
        "pr_values": prs,
        "isr_mean": statistics.mean(isrs),
        "isr_std": statistics.stdev(isrs) if n > 1 else 0,
        "isr_values": isrs,
        "isr_proved_mean": statistics.mean(proved_isrs),
        "isr_proved_std": statistics.stdev(proved_isrs) if n > 1 else 0,
    }
    
    if all("genuine" in m for m in metrics_list):
        genuines = [m["genuine"] for m in metrics_list]
        genuine_rates = [m["genuine_rate"] for m in metrics_list]
        h7s = [m["h7_total"] for m in metrics_list]
        h1s = [m["h1"] for m in metrics_list]
        h8s = [m["h8"] for m in metrics_list]
        n_proved_list = [m["n_proved"] for m in metrics_list]
        
        pool.update({
            "genuine_values": genuines,
            "genuine_rate_mean": statistics.mean(genuine_rates),
            "genuine_rate_std": statistics.stdev(genuine_rates) if n > 1 else 0,
            "h7_values": h7s,
            "h7_rate_mean": statistics.mean([h/p if p else 0 for h, p in zip(h7s, n_proved_list)]),
            "h1_values": h1s,
            "h1_rate_mean": statistics.mean([h/p if p else 0 for h, p in zip(h1s, n_proved_list)]),
            "h8_values": h8s,
            "h8_rate_mean": statistics.mean([h/p if p else 0 for h, p in zip(h8s, n_proved_list)]),
        })
    
    # Pool all per-sample ISR values
    all_isr = []
    for m in metrics_list:
        all_isr.extend(m.get("isr_values", []))
    if all_isr:
        pool["pooled_isr_median"] = statistics.median(all_isr)
        pool["pooled_isr_mean"] = statistics.mean(all_isr)
        pool["pooled_n_samples"] = len(all_isr)
    
    return pool


def fmt_pct(v, digits=1):
    return f"{v*100:.{digits}f}%"


def fmt_pm(mean, std, pct=True):
    if pct:
        return f"{mean*100:.0f}±{std*100:.0f}%"
    return f"{mean:.3f}±{std:.3f}"


def generate_report(metrics_list, pool, errors):
    lines = []
    lines.append("# β=0.2 ISR GRPO — Pooled Analysis")
    lines.append("")
    lines.append(f"**Seeds analyzed**: {len(metrics_list)}/3")
    if errors:
        lines.append(f"**Missing seeds**: {', '.join(errors)}")
    lines.append("")
    
    # Per-seed table
    lines.append("## Per-Seed Breakdown")
    lines.append("")
    lines.append("| Seed | PR | ISR* (proved) | Med ISR | H7a | H7b | H1 | H8 | Genuine | Gen% |")
    lines.append("|------|----|--------------|---------|-----|-----|----|----|---------|------|")
    
    for m in metrics_list:
        h7a = m.get("h7a", "—")
        h7b = m.get("h7b", "—")
        h1 = m.get("h1", "—")
        h8 = m.get("h8", "—")
        gen = m.get("genuine", "—")
        gen_r = fmt_pct(m["genuine_rate"]) if "genuine_rate" in m else "—"
        lines.append(
            f"| s{m['seed']} | {fmt_pct(m['prove_rate'])} | {m['proved_only_mean_isr']:.3f} "
            f"| {m['median_isr']:.3f} | {h7a} | {h7b} | {h1} | {h8} | {gen} | {gen_r} |"
        )
    
    # Pooled row
    if pool["n_seeds"] > 1:
        lines.append(
            f"| **Pooled** | **{fmt_pm(pool['pr_mean'], pool['pr_std'])}** "
            f"| **{fmt_pm(pool['isr_proved_mean'], pool['isr_proved_std'], pct=False)}** "
            f"| {pool.get('pooled_isr_median', 0):.3f} "
            f"| — | — | — | — "
            f"| — | **{fmt_pct(pool.get('genuine_rate_mean', 0))}** |"
        )
    lines.append("")
    
    # Training summary
    lines.append("## Training Summary")
    lines.append("")
    for m in metrics_list:
        step = m.get("train_final_step", "?")
        pr = m.get("train_final_pr")
        isr = m.get("train_final_isr")
        pr_s = f"{pr:.3f}" if pr is not None else "?"
        isr_s = f"{isr:.4f}" if isr is not None else "?"
        lines.append(f"- **s{m['seed']}**: final step {step}, train PR={pr_s}, train ISR={isr_s}")
    lines.append("")
    
    # Pooled stats
    lines.append("## Pooled Statistics")
    lines.append("")
    lines.append(f"- **Prove Rate**: {fmt_pm(pool['pr_mean'], pool['pr_std'])}")
    lines.append(f"- **ISR* (proved-only mean)**: {fmt_pm(pool['isr_proved_mean'], pool['isr_proved_std'], pct=False)}")
    if "pooled_isr_median" in pool:
        lines.append(f"- **Pooled ISR median** (across {pool['pooled_n_samples']} samples): {pool['pooled_isr_median']:.3f}")
    if "genuine_rate_mean" in pool:
        lines.append(f"- **Genuine rate**: {fmt_pm(pool['genuine_rate_mean'], pool['genuine_rate_std'])}")
        lines.append(f"- **H7 rate**: {fmt_pct(pool['h7_rate_mean'])}")
        lines.append(f"- **H1 rate**: {fmt_pct(pool['h1_rate_mean'])}")
        lines.append(f"- **H8 rate**: {fmt_pct(pool['h8_rate_mean'])}")
    lines.append("")
    
    # Baseline comparison
    lines.append("## Baseline Comparison")
    lines.append("")
    lines.append("| Condition | PR | ISR* | Genuine% | H7% | Note |")
    lines.append("|-----------|-----|------|----------|-----|------|")
    
    # Current β=0.2 row
    gen_s = fmt_pct(pool.get("genuine_rate_mean", 0)) if "genuine_rate_mean" in pool else "—"
    h7_s = fmt_pct(pool.get("h7_rate_mean", 0)) if "h7_rate_mean" in pool else "—"
    lines.append(
        f"| **ISR β=0.2 ({pool['n_seeds']}-seed)** "
        f"| **{fmt_pm(pool['pr_mean'], pool['pr_std'])}** "
        f"| **{fmt_pm(pool['isr_proved_mean'], pool['isr_proved_std'], pct=False)}** "
        f"| **{gen_s}** | **{h7_s}** | **this analysis** |"
    )
    
    for name, bl in BASELINES.items():
        pr_s = fmt_pm(bl["pr_mean"], bl.get("pr_std", 0)) if bl.get("pr_mean") is not None else "—"
        isr_s = fmt_pm(bl["isr_mean"], bl.get("isr_std", 0), pct=False) if bl.get("isr_mean") is not None else "—"
        gen_s = f"{bl['genuine_pct']}%" if bl.get("genuine_pct") is not None else "—"
        h7_s = f"{bl['h7_pct']}%" if bl.get("h7_pct") is not None else "—"
        lines.append(f"| {name} | {pr_s} | {isr_s} | {gen_s} | {h7_s} | {bl.get('note', '')} |")
    
    lines.append("")
    
    # ISR distribution analysis
    if any(m.get("isr_values") for m in metrics_list):
        lines.append("## ISR Distribution")
        lines.append("")
        for m in metrics_list:
            vals = m.get("isr_values", [])
            if not vals:
                continue
            lines.append(f"### s{m['seed']} (n={len(vals)} proved)")
            buckets = {"0.0": 0, "(0,0.25]": 0, "(0.25,0.5]": 0, "(0.5,0.75]": 0, "(0.75,1.0)": 0, "1.0": 0}
            for v in vals:
                if v == 0:
                    buckets["0.0"] += 1
                elif v <= 0.25:
                    buckets["(0,0.25]"] += 1
                elif v <= 0.5:
                    buckets["(0.25,0.5]"] += 1
                elif v <= 0.75:
                    buckets["(0.5,0.75]"] += 1
                elif v < 1.0:
                    buckets["(0.75,1.0)"] += 1
                else:
                    buckets["1.0"] += 1
            lines.append("| Range | Count | % |")
            lines.append("|-------|-------|---|")
            for rng, cnt in buckets.items():
                lines.append(f"| {rng} | {cnt} | {cnt/len(vals)*100:.1f}% |")
            lines.append("")
    
    # Interpretation
    lines.append("## Interpretation")
    lines.append("")
    
    if pool["n_seeds"] >= 3:
        pr = pool["pr_mean"]
        gen = pool.get("genuine_rate_mean", 0)
        h7 = pool.get("h7_rate_mean", 0)
        
        if gen < 0.05:
            lines.append("**Complete Goodhart collapse**: 0% genuine proofs. β=0.2 ISR reward is fully gamed.")
        elif gen < 0.20:
            lines.append(f"**Severe hacking**: only {fmt_pct(gen)} genuine. Dominant mode: {'H7 tautology' if h7 > 0.5 else 'mixed'}.")
        elif gen > 0.50:
            lines.append(f"**Partial success**: {fmt_pct(gen)} genuine proofs survive. Best β tested so far.")
        
        if pr > 0.90:
            lines.append(f"- High PR ({fmt_pct(pr)}) suggests training converges but may be surface-level.")
        
        lines.append("")
        lines.append("### β Dose-Response Position")
        lines.append("- β=0.1: high PR, 71% genuine (best), but 2/3 seeds meta-Goodhart")
        lines.append(f"- **β=0.2: PR={fmt_pm(pool['pr_mean'], pool['pr_std'])}, genuine={fmt_pct(gen)}** ← this")
        lines.append("- β=0.3: PR=66±42%, 50% genuine, extreme instability")
    else:
        lines.append(f"*Only {pool['n_seeds']}/3 seeds available. Pooled stats are preliminary.*")
    
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Pooled analysis for β=0.2 ISR")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--output", type=str, default=None,
                        help="Output markdown file path")
    args = parser.parse_args()

    metrics_list = []
    errors = []
    
    for seed in args.seeds:
        data, err = load_seed_data(seed)
        if err:
            errors.append(f"s{seed}: {err}")
            print(f"[WARN] s{seed}: {err}", file=sys.stderr)
            continue
        m = extract_metrics(data)
        metrics_list.append(m)
    
    if not metrics_list:
        print("ERROR: No seed data available. Exiting.", file=sys.stderr)
        sys.exit(1)
    
    pool = pooled_stats(metrics_list)
    report = generate_report(metrics_list, pool, errors)
    
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report saved to {args.output}", file=sys.stderr)
    
    print(report)


if __name__ == "__main__":
    main()
