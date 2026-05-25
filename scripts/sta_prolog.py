"""Prolog STA standalone test and CLI.

Usage:
    python scripts/sta_prolog.py --test          # run built-in test cases
    python scripts/sta_prolog.py --file FILE     # evaluate a JSONL file
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.solvers.factory import get_solver
from src.sta.prolog_reward import (
    fast_prolog_sta_reward,
    parse_prolog_formalization,
    prolog_sta_reward,
)


TEST_CASES = [
    {
        "name": "grandparent (all necessary)",
        "formalization": (
            "parent(tom, bob).\n"
            "parent(bob, ann).\n"
            "grandparent(X, Y) :- parent(X, Z), parent(Z, Y).\n"
            "?- grandparent(tom, ann).\n"
        ),
        "expect_proved": True,
        "expect_isr_range": (0.9, 1.0),
        "fast_isr_range": (0.9, 1.0),
    },
    {
        "name": "grandparent with redundant fact",
        "formalization": (
            "parent(tom, bob).\n"
            "parent(bob, ann).\n"
            "parent(alice, charlie).\n"
            "grandparent(X, Y) :- parent(X, Z), parent(Z, Y).\n"
            "?- grandparent(tom, ann).\n"
        ),
        "expect_proved": True,
        "expect_isr_range": (0.5, 0.8),
        "fast_isr_range": (0.9, 1.0),
    },
    {
        "name": "missing fact (should fail)",
        "formalization": (
            "parent(tom, bob).\n"
            "grandparent(X, Y) :- parent(X, Z), parent(Z, Y).\n"
            "?- grandparent(tom, ann).\n"
        ),
        "expect_proved": False,
        "expect_isr_range": (0.0, 0.0),
        "fast_isr_range": (0.0, 0.0),
    },
    {
        "name": "simple fact query",
        "formalization": (
            "likes(mary, food).\n"
            "likes(mary, wine).\n"
            "?- likes(mary, food).\n"
        ),
        "expect_proved": True,
        "expect_isr_range": (0.4, 0.6),
        "fast_isr_range": (0.9, 1.0),
    },
    {
        "name": "chain reasoning",
        "formalization": (
            "mortal(X) :- human(X).\n"
            "human(X) :- greek(X).\n"
            "greek(socrates).\n"
            "?- mortal(socrates).\n"
        ),
        "expect_proved": True,
        "expect_isr_range": (0.9, 1.0),
        "fast_isr_range": (0.9, 1.0),
    },
]


def run_tests(solver, use_fast: bool = False):
    print(f"Solver: {type(solver).__name__}")
    print(f"Mode: {'fast (O(1))' if use_fast else 'full (O(n))'}")
    print("=" * 60)

    passed = 0
    for tc in TEST_CASES:
        name = tc["name"]
        reward_fn = fast_prolog_sta_reward if use_fast else prolog_sta_reward
        result = reward_fn({}, tc["formalization"], solver, timeout=5)

        proved = result["execution_reward"] > 0
        isr = result["sta_reward"]
        lo, hi = tc.get("fast_isr_range", tc["expect_isr_range"]) if use_fast else tc["expect_isr_range"]

        proved_ok = proved == tc["expect_proved"]
        isr_ok = lo <= isr <= hi if tc["expect_proved"] else isr == 0.0

        status = "PASS" if (proved_ok and isr_ok) else "FAIL"
        if status == "PASS":
            passed += 1

        print(f"\n[{status}] {name}")
        print(f"  proved={proved} (expect={tc['expect_proved']})")
        print(f"  ISR={isr:.3f} (expect=[{lo:.1f}, {hi:.1f}])")

        if "components" in result.get("details", {}):
            for comp in result["details"]["components"]:
                mark = "*" if comp["is_necessary"] else " "
                print(f"    [{mark}] {comp['content']}")

    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{len(TEST_CASES)} passed")
    return passed == len(TEST_CASES)


def run_file(solver, path: str, use_fast: bool = False):
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    reward_fn = fast_prolog_sta_reward if use_fast else prolog_sta_reward
    proved_count = 0
    isr_sum = 0.0

    for i, item in enumerate(data):
        formalization = ""
        if "prolog_premises" in item:
            for p in item["prolog_premises"]:
                formalization += p.strip()
                if not formalization.endswith('.'):
                    formalization += '.'
                formalization += "\n"
            formalization += f"?- {item.get('prolog_query', '')}.\n"
        elif "formalization" in item:
            formalization = item["formalization"]
        else:
            print(f"[{i}] SKIP: no formalization")
            continue

        result = reward_fn(item, formalization, solver, timeout=5)
        proved = result["execution_reward"] > 0
        isr = result["sta_reward"]
        if proved:
            proved_count += 1
        isr_sum += isr
        status = "proved" if proved else "FAILED"
        print(f"[{i}] {item.get('id', '?'):>12s}  {status}  ISR={isr:.3f}")

    n = len(data)
    print(f"\nTotal: {n}, Proved: {proved_count}/{n} ({proved_count/n:.1%}), "
          f"Mean ISR: {isr_sum/n:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Prolog STA test/eval")
    parser.add_argument("--test", action="store_true", help="Run built-in test cases")
    parser.add_argument("--file", type=str, help="JSONL file to evaluate")
    parser.add_argument("--fast", action="store_true", help="Use fast O(1) ISR")
    parser.add_argument("--timeout", type=int, default=5)
    args = parser.parse_args()

    solver = get_solver("prolog")
    print(f"swipl available: {solver.is_available()}")

    if args.test:
        ok = run_tests(solver, use_fast=args.fast)
        sys.exit(0 if ok else 1)
    elif args.file:
        run_file(solver, args.file, use_fast=args.fast)
    else:
        # Default: run tests
        ok = run_tests(solver, use_fast=args.fast)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
