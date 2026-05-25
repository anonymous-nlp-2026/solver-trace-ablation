"""
XOR 展开验证脚本。

输入: data/train_500.jsonl, data/test_100.jsonl
功能:
  - 统计含 ⊕ 的样本数
  - 调用 Prover9Solver._expand_xor() 展开 XOR
  - 验证展开后无残留 ⊕
  - 对展开后公式调用 _normalize_fol() 转为 Prover9 语法，检查基本合法性
依赖: src/solvers/prover9.py
"""

import json
import sys
import os

sys.path.insert(0, "/root/solver-trace-ablation")
from src.solvers.prover9 import Prover9Solver

DATA_DIR = "/root/solver-trace-ablation/data"
solver = Prover9Solver()


def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def check_balanced_parens(s):
    depth = 0
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if depth < 0:
            return False
    return depth == 0


def verify_dataset(name, data):
    total = len(data)
    xor_premise_count = 0
    xor_conclusion_count = 0
    xor_samples = 0
    expand_errors = 0
    residual_xor = 0
    normalize_errors = 0
    paren_errors = 0
    error_details = []

    for item in data:
        has_xor = False
        premises_xor = sum(1 for p in item["fol_premises"] if "⊕" in p)
        conclusion_xor = 1 if "⊕" in item["fol_conclusion"] else 0

        if premises_xor > 0 or conclusion_xor > 0:
            has_xor = True
            xor_samples += 1
            xor_premise_count += premises_xor
            xor_conclusion_count += conclusion_xor

        # Expand XOR on all formulas
        all_formulas = item["fol_premises"] + [item["fol_conclusion"]]
        for formula in all_formulas:
            try:
                expanded = Prover9Solver._expand_xor(formula)
            except Exception as e:
                expand_errors += 1
                error_details.append(f"  _expand_xor error on {item['id']}: {e}")
                continue

            if "⊕" in expanded:
                residual_xor += 1
                error_details.append(f"  Residual ⊕ in {item['id']}: {expanded[:80]}")

            # Normalize to Prover9 syntax
            try:
                normalized = solver._normalize_fol(formula)
            except Exception as e:
                normalize_errors += 1
                error_details.append(f"  _normalize_fol error on {item['id']}: {e}")
                continue

            if not check_balanced_parens(normalized):
                paren_errors += 1
                error_details.append(f"  Unbalanced parens in {item['id']}: {normalized[:80]}")

    print(f"\n{'='*50}")
    print(f"Dataset: {name} ({total} samples)")
    print(f"{'='*50}")
    print(f"  Samples with XOR (⊕):     {xor_samples}")
    print(f"  Premise formulas with ⊕:  {xor_premise_count}")
    print(f"  Conclusions with ⊕:       {xor_conclusion_count}")
    print(f"  Expand errors:            {expand_errors}")
    print(f"  Residual ⊕ after expand:  {residual_xor}")
    print(f"  Normalize errors:         {normalize_errors}")
    print(f"  Unbalanced parentheses:   {paren_errors}")

    if error_details:
        print(f"\n  Error details:")
        for d in error_details[:20]:
            print(d)
        if len(error_details) > 20:
            print(f"  ... and {len(error_details) - 20} more")

    return xor_samples, expand_errors, residual_xor, normalize_errors, paren_errors


def main():
    datasets = {
        "train_500": os.path.join(DATA_DIR, "train_500.jsonl"),
        "test_100": os.path.join(DATA_DIR, "test_100.jsonl"),
        "phase0_100": os.path.join(DATA_DIR, "phase0_100.jsonl"),
    }

    total_issues = 0
    for name, path in datasets.items():
        if not os.path.exists(path):
            print(f"SKIP {name}: {path} not found")
            continue
        data = load_jsonl(path)
        xor, exp_err, res, norm_err, paren_err = verify_dataset(name, data)
        total_issues += exp_err + res + norm_err + paren_err

    # Cross-set overlap check
    print(f"\n{'='*50}")
    print("Cross-set overlap check")
    print(f"{'='*50}")
    sets = {}
    for name, path in datasets.items():
        if os.path.exists(path):
            data = load_jsonl(path)
            sets[name] = {d["id"] for d in data}

    for a in sets:
        for b in sets:
            if a < b:
                overlap = sets[a] & sets[b]
                status = "OK (0)" if len(overlap) == 0 else f"FAIL ({len(overlap)} overlapping)"
                print(f"  {a} ∩ {b}: {status}")
                if overlap:
                    total_issues += len(overlap)

    print(f"\n{'='*50}")
    if total_issues == 0:
        print("ALL CHECKS PASSED")
    else:
        print(f"ISSUES FOUND: {total_issues}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
