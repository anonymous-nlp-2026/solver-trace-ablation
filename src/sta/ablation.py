"""STA core ablation logic.

For each FOLComponent in a proof problem, ablation removes or neutralizes
that component and re-runs the solver. A component is *necessary* iff the
proof breaks (proved → not proved) after its removal.

Input:  premises, conclusion, solver.
Output: list of AblationResult (one per component).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..solvers.base import FOLSolver, SolverResult
from .components import FOLComponent, decompose_fol


@dataclass
class AblationResult:
    """Outcome of ablating a single component."""
    component: FOLComponent
    original_proved: bool
    ablated_proved: bool
    is_necessary: bool          # original_proved and not ablated_proved
    solver_result: SolverResult  # solver result for the ablated problem


def _ablate_premise(
    premises: list[str],
    conclusion: str,
    component: FOLComponent,
    solver: FOLSolver,
    timeout: int,
) -> SolverResult:
    """Remove one premise and re-prove."""
    ablated_premises = [p for i, p in enumerate(premises) if i != component.index]
    return solver.prove(ablated_premises, conclusion, timeout=timeout)


def _ablate_conclusion_part(
    premises: list[str],
    conclusion: str,
    component: FOLComponent,
    solver: FOLSolver,
    timeout: int,
) -> SolverResult:
    """Neutralize a conclusion sub-part.

    Strategy per component type:
    - conclusion_quantifier: drop the quantifier (weaken the claim)
    - conclusion_antecedent: replace with T (True) — makes implication trivially
      equivalent to consequent, so if proof still works, antecedent was unnecessary
    - conclusion_consequent: replace with ⊤ — implication becomes trivially true,
      so proof should always succeed; if it does, consequent was NOT necessary
      (but we invert: the test is whether removing it *breaks* the proof)
      Actually: replace with ⊥ (False) to test necessity.
    - conclusion_whole: replace conclusion with a tautology (provable without premises)
    """
    if component.type == "conclusion_quantifier":
        # Remove quantifier prefix from conclusion
        new_conclusion = conclusion.replace(f"{component.content}", "", 1).strip()
        # Strip leading '(' and trailing ')' if they became unbalanced
        if new_conclusion.startswith("(") and new_conclusion.endswith(")"):
            new_conclusion = new_conclusion[1:-1].strip()
        return solver.prove(premises, new_conclusion, timeout=timeout)

    elif component.type == "conclusion_antecedent":
        new_conclusion = conclusion.replace(component.content, "$T", 1)
        return solver.prove(premises, new_conclusion, timeout=timeout)

    elif component.type == "conclusion_consequent":
        new_conclusion = conclusion.replace(component.content, "$F", 1)
        return solver.prove(premises, new_conclusion, timeout=timeout)

    elif component.type == "conclusion_whole":
        return solver.prove(premises, "$T", timeout=timeout)

    # Fallback: try proving with unmodified conclusion (should not happen)
    return solver.prove(premises, conclusion, timeout=timeout)


def ablate_component(
    premises: list[str],
    conclusion: str,
    component: FOLComponent,
    solver: FOLSolver,
    timeout: int = 5,
    original_result: Optional[SolverResult] = None,
) -> AblationResult:
    """Ablate a single component and compare with the original proof.

    Returns an AblationResult with necessity judgment.
    """
    if original_result is None:
        original_result = solver.prove(premises, conclusion, timeout=timeout)

    if component.type == "premise":
        ablated = _ablate_premise(premises, conclusion, component, solver, timeout)
    else:
        ablated = _ablate_conclusion_part(premises, conclusion, component, solver, timeout)

    is_necessary = original_result.proved and not ablated.proved

    return AblationResult(
        component=component,
        original_proved=original_result.proved,
        ablated_proved=ablated.proved,
        is_necessary=is_necessary,
        solver_result=ablated,
    )


def run_sta(
    premises: list[str],
    conclusion: str,
    solver: FOLSolver,
    timeout: int = 5,
    original_result: Optional[SolverResult] = None,
) -> list[AblationResult]:
    """Run full Solver Trace Ablation on all components.

    Decomposes the problem, ablates each component, returns results.
    """
    if original_result is None:
        original_result = solver.prove(premises, conclusion, timeout=timeout)

    components = decompose_fol(premises, conclusion)
    results: list[AblationResult] = []
    for comp in components:
        result = ablate_component(
            premises, conclusion, comp, solver, timeout,
            original_result=original_result,
        )
        results.append(result)
    return results
