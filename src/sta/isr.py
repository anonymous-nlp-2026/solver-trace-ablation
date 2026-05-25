"""ISR (Intensional Score Rate) computation.

ISR = (# necessary components) / (# total components).
High ISR means every part of the formalization contributes to the proof
(good formalization). Low ISR indicates redundant or vacuous components
(potential reward hacking).

Input:  list of AblationResult from run_sta().
Output: ISR score (float in [0, 1]).
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Optional

from ..solvers.base import FOLSolver
from .ablation import AblationResult, run_sta


def compute_isr(ablation_results: list[AblationResult]) -> float:
    """Compute ISR for a single problem.

    Returns 0.0 if no components (degenerate case).
    """
    if not ablation_results:
        return 0.0
    necessary = sum(1 for r in ablation_results if r.is_necessary)
    return necessary / len(ablation_results)


@dataclass
class ISRBatchStats:
    """Aggregate ISR statistics over a batch of problems."""
    mean_isr: float
    std_isr: float
    median_isr: float
    solver_success_rate: float  # fraction of problems where solver ran successfully
    avg_time_seconds: float
    n_problems: int
    n_solver_failures: int


def compute_isr_batch(
    problems: list[dict],
    solver: FOLSolver,
    timeout: int = 5,
) -> ISRBatchStats:
    """Compute ISR over a batch of problems.

    Each problem dict must have keys:
        - "premises": list[str]
        - "conclusion": str

    Returns aggregate statistics.
    """
    isr_scores: list[float] = []
    total_time = 0.0
    n_failures = 0

    for prob in problems:
        premises = prob["premises"]
        conclusion = prob["conclusion"]

        results = run_sta(premises, conclusion, solver, timeout=timeout)

        # Check if solver succeeded on at least the original proof
        if results and results[0].original_proved:
            isr = compute_isr(results)
            isr_scores.append(isr)
            total_time += sum(r.solver_result.time_seconds for r in results)
        else:
            n_failures += 1

    n_total = len(problems)
    n_success = len(isr_scores)

    if not isr_scores:
        return ISRBatchStats(
            mean_isr=0.0, std_isr=0.0, median_isr=0.0,
            solver_success_rate=0.0,
            avg_time_seconds=0.0,
            n_problems=n_total,
            n_solver_failures=n_failures,
        )

    return ISRBatchStats(
        mean_isr=statistics.mean(isr_scores),
        std_isr=statistics.stdev(isr_scores) if len(isr_scores) > 1 else 0.0,
        median_isr=statistics.median(isr_scores),
        solver_success_rate=n_success / n_total if n_total > 0 else 0.0,
        avg_time_seconds=total_time / n_success,
        n_problems=n_total,
        n_solver_failures=n_failures,
    )


def fast_isr_fol(
    premises: list[str],
    conclusion: str,
    solver: FOLSolver,
    timeout: int = 5,
) -> dict:
    """O(1) approximate ISR: check if conclusion is a tautology (provable without premises).

    Only 2 solver calls regardless of component count. Used for GRPO reward.

    Returns:
        fast_isr: 0.0 if original unprovable or conclusion is tautology, else 1.0
        original_proved: whether the full problem is provable
        conclusion_tautology: whether conclusion is provable without premises
        solver_time: total wall-clock time for both calls
    """
    start = time.time()

    original = solver.prove(premises, conclusion, timeout=timeout)
    tautology_check = solver.prove([], conclusion, timeout=timeout)

    elapsed = time.time() - start

    if not original.proved:
        fast_isr = 0.0
    elif tautology_check.proved:
        fast_isr = 0.0
    else:
        fast_isr = 1.0

    return {
        "fast_isr": fast_isr,
        "original_proved": original.proved,
        "conclusion_tautology": tautology_check.proved,
        "solver_time": elapsed,
    }
