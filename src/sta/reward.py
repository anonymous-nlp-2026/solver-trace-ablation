"""STA-based reward function for GRPO training.

Combines two signals:
  1. Execution reward: did the solver prove the formalization? (0 or 1)
  2. STA reward: ISR — what fraction of components is necessary? (0 to 1)

A good formalization scores high on both; a hacked one may score 1 on
execution but low on STA (redundant/vacuous components).

Input:  problem dict, formalization string, solver.
Output: reward dict with execution_reward, sta_reward, combined_reward, details.
"""

from __future__ import annotations

import re
from typing import Optional

from ..solvers.base import FOLSolver
from .ablation import run_sta
from .isr import compute_isr, fast_isr_fol


def _parse_formalization(formalization: str) -> tuple[list[str], str]:
    """Extract premises and conclusion from a model-generated formalization.

    Expected format (flexible):
        Premises:
        1. P(a)
        2. ∀x(P(x) → Q(x))
        Conclusion: Q(a)

    Or a simpler format:
        P(a); ∀x(P(x) → Q(x)) |- Q(a)

    Returns (premises, conclusion).
    """
    # Try "Premises: ... Conclusion: ..." format
    m = re.search(r"(?:Conclusion|Goal)\s*:\s*(.+)", formalization, re.IGNORECASE)
    if m:
        conclusion = m.group(1).strip().rstrip(".")
        # Everything before "Conclusion:" is premises
        prem_text = formalization[:m.start()]
        # Remove "Premises:" header
        prem_text = re.sub(r"(?:Premises|Assumptions)\s*:", "", prem_text, flags=re.IGNORECASE)
        # Split by numbered lines or semicolons
        raw_premises = re.split(r"\n\s*\d+[.)]\s*|\n\s*[-•]\s*|;\s*", prem_text)
        premises = [p.strip().rstrip(".") for p in raw_premises if p.strip()]
        return premises, conclusion

    # Try "|- " (turnstile) format
    if "|-" in formalization:
        parts = formalization.split("|-", 1)
        prem_text = parts[0].strip()
        conclusion = parts[1].strip().rstrip(".")
        premises = [p.strip().rstrip(".") for p in re.split(r";\s*|,\s*(?=[A-Z∀∃¬])", prem_text) if p.strip()]
        return premises, conclusion

    # Fallback: last line is conclusion, rest are premises
    lines = [l.strip() for l in formalization.strip().splitlines() if l.strip()]
    if len(lines) >= 2:
        conclusion = lines[-1].rstrip(".")
        premises = [l.rstrip(".") for l in lines[:-1]]
        # Strip numbering
        premises = [re.sub(r"^\d+[.)]\s*", "", p) for p in premises]
        return premises, conclusion

    # Single line — treat as conclusion with no premises
    return [], formalization.strip().rstrip(".")


def sta_reward(
    problem: dict,
    formalization: str,
    solver: FOLSolver,
    execution_weight: float = 0.5,
    sta_weight: float = 0.5,
    timeout: int = 5,
) -> dict:
    """Compute STA reward for a single formalization.

    Args:
        problem: Original problem (unused in reward computation, kept for logging).
        formalization: Model-generated FOL formalization string.
        solver: FOL solver instance.
        execution_weight: Weight for the execution (proof success) signal.
        sta_weight: Weight for the ISR signal.
        timeout: Per-query solver timeout in seconds.

    Returns:
        Dict with keys: execution_reward, sta_reward, combined_reward, details.
    """
    # Parse formalization into premises + conclusion
    try:
        premises, conclusion = _parse_formalization(formalization)
    except Exception as e:
        return {
            "execution_reward": 0.0,
            "sta_reward": 0.0,
            "combined_reward": 0.0,
            "details": {"error": f"parse error: {e}", "formalization": formalization},
        }

    if not conclusion:
        return {
            "execution_reward": 0.0,
            "sta_reward": 0.0,
            "combined_reward": 0.0,
            "details": {"error": "empty conclusion", "formalization": formalization},
        }

    # Execution reward: can the solver prove it?
    proof_result = solver.prove(premises, conclusion, timeout=timeout)
    exec_reward = 1.0 if proof_result.proved else 0.0

    # STA reward: ISR (pass proof_result to avoid re-proving the original)
    if proof_result.proved:
        ablation_results = run_sta(
            premises, conclusion, solver, timeout=timeout,
            original_result=proof_result,
        )
        isr = compute_isr(ablation_results)
        details = {
            "premises": premises,
            "conclusion": conclusion,
            "n_components": len(ablation_results),
            "n_necessary": sum(1 for r in ablation_results if r.is_necessary),
            "isr": isr,
            "components": [
                {
                    "type": r.component.type,
                    "content": r.component.content,
                    "is_necessary": r.is_necessary,
                }
                for r in ablation_results
            ],
        }
    else:
        isr = 0.0
        details = {
            "premises": premises,
            "conclusion": conclusion,
            "proof_failed": True,
            "solver_error": proof_result.error,
        }

    combined = execution_weight * exec_reward + sta_weight * isr

    return {
        "execution_reward": exec_reward,
        "sta_reward": isr,
        "combined_reward": combined,
        "details": details,
    }


def fast_sta_reward(
    problem: dict,
    formalization: str,
    solver: FOLSolver,
    execution_weight: float = 0.5,
    sta_weight: float = 0.5,
    timeout: int = 5,
) -> dict:
    """Fast STA reward using O(1) approximate ISR (2 solver calls total).

    Suitable for GRPO training where per-sample latency matters.
    """
    try:
        premises, conclusion = _parse_formalization(formalization)
    except Exception as e:
        return {
            "execution_reward": 0.0,
            "sta_reward": 0.0,
            "combined_reward": 0.0,
            "details": {"error": f"parse error: {e}", "formalization": formalization},
        }

    if not conclusion:
        return {
            "execution_reward": 0.0,
            "sta_reward": 0.0,
            "combined_reward": 0.0,
            "details": {"error": "empty conclusion", "formalization": formalization},
        }

    result = fast_isr_fol(premises, conclusion, solver, timeout=timeout)

    exec_reward = 1.0 if result["original_proved"] else 0.0
    isr = result["fast_isr"]
    combined = execution_weight * exec_reward + sta_weight * isr

    return {
        "execution_reward": exec_reward,
        "sta_reward": isr,
        "combined_reward": combined,
        "details": {
            "premises": premises,
            "conclusion": conclusion,
            "fast_isr": isr,
            "original_proved": result["original_proved"],
            "conclusion_tautology": result["conclusion_tautology"],
            "solver_time": result["solver_time"],
        },
    }
