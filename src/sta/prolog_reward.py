"""Prolog STA reward: execution + ISR for Prolog formalizations.

Mirrors src/sta/reward.py but for Prolog logic type:
- Parses Prolog clauses + query from model output
- Uses PrologSolver (swipl) instead of Prover9
- Ablation: remove one clause at a time, check if query breaks
- ISR = necessary_clauses / total_clauses
"""

from __future__ import annotations

import re
import time
from typing import Optional

from ..solvers.base import FOLSolver, SolverResult
from .ablation import AblationResult
from .components import FOLComponent


def parse_prolog_formalization(text: str) -> tuple[list[str], str]:
    """Extract Prolog clauses and query from model output.

    Supported formats:

    1. Inline query:
        parent(tom, bob).
        grandparent(X,Y) :- parent(X,Z), parent(Z,Y).
        ?- grandparent(tom, ann).

    2. Labeled sections:
        Facts/Rules:
        parent(tom, bob).
        Query: grandparent(tom, ann)

    Returns (clauses, query). Query has no '?-' prefix or trailing '.'.
    """
    lines = text.strip().splitlines()
    query: Optional[str] = None
    clauses: list[str] = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith('%'):
            continue

        if line.startswith('?-'):
            query = line[2:].strip().rstrip('.')
            continue

        m = re.match(r'^Query\s*:\s*(.+)', line, re.IGNORECASE)
        if m:
            q = m.group(1).strip()
            if q.startswith('?-'):
                q = q[2:].strip()
            query = q.rstrip('.')
            continue

        if re.match(
            r'^(Facts|Rules|Clauses|Program|Premises)'
            r'(\s*/\s*(Rules|Facts|Clauses))?\s*:\s*$',
            line, re.IGNORECASE,
        ):
            continue

        line = re.sub(r'^\d+[.)]\s*', '', line)

        if line.startswith(':- ') or line.startswith(':-\t'):
            continue

        if line:
            clauses.append(line)

    if query is None and clauses:
        query = clauses.pop().rstrip('.')

    return clauses, query or ""


def prolog_sta_ablation(
    clauses: list[str],
    query: str,
    solver: FOLSolver,
    timeout: int = 5,
    original_result: Optional[SolverResult] = None,
) -> list[AblationResult]:
    """Remove each clause one at a time, check if query breaks."""
    if original_result is None:
        original_result = solver.prove(clauses, query, timeout=timeout)

    results: list[AblationResult] = []
    for i, clause in enumerate(clauses):
        component = FOLComponent(type="premise", content=clause, index=i)
        ablated = clauses[:i] + clauses[i + 1:]
        ablated_result = solver.prove(ablated, query, timeout=timeout)

        results.append(AblationResult(
            component=component,
            original_proved=original_result.proved,
            ablated_proved=ablated_result.proved,
            is_necessary=original_result.proved and not ablated_result.proved,
            solver_result=ablated_result,
        ))

    return results


def prolog_isr(ablation_results: list[AblationResult]) -> float:
    if not ablation_results:
        return 0.0
    return sum(1 for r in ablation_results if r.is_necessary) / len(ablation_results)


def prolog_sta_reward(
    problem: dict,
    formalization: str,
    solver: FOLSolver,
    execution_weight: float = 0.5,
    sta_weight: float = 0.5,
    timeout: int = 5,
) -> dict:
    """Full Prolog STA reward (O(n) solver calls)."""
    try:
        clauses, query = parse_prolog_formalization(formalization)
    except Exception as e:
        return _error_result(f"parse error: {e}", formalization)

    if not query:
        return _error_result("empty query", formalization)

    proof = solver.prove(clauses, query, timeout=timeout)
    exec_reward = 1.0 if proof.proved else 0.0

    if proof.proved:
        abl = prolog_sta_ablation(
            clauses, query, solver, timeout=timeout, original_result=proof,
        )
        isr = prolog_isr(abl)
        details = {
            "clauses": clauses, "query": query,
            "n_clauses": len(abl),
            "n_necessary": sum(1 for r in abl if r.is_necessary),
            "isr": isr,
            "components": [
                {"content": r.component.content, "is_necessary": r.is_necessary}
                for r in abl
            ],
        }
    else:
        isr = 0.0
        details = {
            "clauses": clauses, "query": query,
            "proof_failed": True, "solver_error": proof.error,
        }

    return {
        "execution_reward": exec_reward,
        "sta_reward": isr,
        "combined_reward": execution_weight * exec_reward + sta_weight * isr,
        "details": details,
    }


def fast_prolog_sta_reward(
    problem: dict,
    formalization: str,
    solver: FOLSolver,
    execution_weight: float = 0.5,
    sta_weight: float = 0.5,
    timeout: int = 5,
) -> dict:
    """Fast Prolog STA reward (O(1): 2 solver calls).

    Checks if query succeeds without any clauses (trivially true).
    If so, ISR=0. Otherwise ISR=1.
    """
    try:
        clauses, query = parse_prolog_formalization(formalization)
    except Exception as e:
        return _error_result(f"parse error: {e}", formalization)

    if not query:
        return _error_result("empty query", formalization)

    start = time.time()
    original = solver.prove(clauses, query, timeout=timeout)
    trivial = solver.prove([], query, timeout=timeout)
    elapsed = time.time() - start

    exec_reward = 1.0 if original.proved else 0.0
    if not original.proved:
        fast_isr = 0.0
    elif trivial.proved:
        fast_isr = 0.0
    else:
        fast_isr = 1.0

    return {
        "execution_reward": exec_reward,
        "sta_reward": fast_isr,
        "combined_reward": execution_weight * exec_reward + sta_weight * fast_isr,
        "details": {
            "clauses": clauses, "query": query,
            "fast_isr": fast_isr,
            "original_proved": original.proved,
            "query_trivial": trivial.proved,
            "solver_time": elapsed,
        },
    }


def _error_result(msg: str, formalization: str) -> dict:
    return {
        "execution_reward": 0.0, "sta_reward": 0.0, "combined_reward": 0.0,
        "details": {"error": msg, "formalization": formalization},
    }
