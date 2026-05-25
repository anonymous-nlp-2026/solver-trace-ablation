"""Prover9 CLI backend.

Calls the `prover9` binary via subprocess. Input is written as a temp file
in Prover9 native syntax (formulas(assumptions)/formulas(goals)).

Dependencies: prover9 binary in PATH.
"""

import os
import shutil
import subprocess
import tempfile
import time
from typing import Optional

from .base import FOLSolver, SolverResult


class Prover9Solver(FOLSolver):
    """FOL solver backed by the Prover9 theorem prover."""

    def __init__(self, binary: str = "prover9"):
        self._binary = binary

    def is_available(self) -> bool:
        return shutil.which(self._binary) is not None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def _expand_xor(formula: str) -> str:
        while '⊕' in formula:
            idx = formula.index('⊕')

            # --- Left operand ---
            i = idx - 1
            while i >= 0 and formula[i] == ' ':
                i -= 1
            if i < 0:
                return formula
            left_end = i + 1

            if formula[i] == ')':
                depth = 1
                i -= 1
                while i >= 0 and depth > 0:
                    if formula[i] == ')':
                        depth += 1
                    elif formula[i] == '(':
                        depth -= 1
                    i -= 1
                i += 1
                k = i - 1
                while k >= 0 and (formula[k].isalnum() or formula[k] == '_'):
                    k -= 1
                left_start = k + 1
            else:
                while i > 0 and (formula[i - 1].isalnum() or formula[i - 1] == '_'):
                    i -= 1
                left_start = i

            k = left_start - 1
            while k >= 0 and formula[k] == ' ':
                k -= 1
            if k >= 0 and formula[k] == '¬':
                left_start = k

            A = formula[left_start:left_end]

            # --- Right operand ---
            j = idx + 1
            while j < len(formula) and formula[j] == ' ':
                j += 1
            if j >= len(formula):
                return formula
            right_start = j

            if formula[j] == '¬':
                j += 1
                while j < len(formula) and formula[j] == ' ':
                    j += 1

            if j < len(formula) and formula[j] in ('∀', '∃'):
                j += 1
                while j < len(formula) and (formula[j].isalnum() or formula[j] == '_'):
                    j += 1

            if j < len(formula) and formula[j] == '(':
                depth = 1
                j += 1
                while j < len(formula) and depth > 0:
                    if formula[j] == '(':
                        depth += 1
                    elif formula[j] == ')':
                        depth -= 1
                    j += 1
            elif j < len(formula) and (formula[j].isalnum() or formula[j] == '_'):
                while j < len(formula) and (formula[j].isalnum() or formula[j] == '_'):
                    j += 1
                if j < len(formula) and formula[j] == '(':
                    depth = 1
                    j += 1
                    while j < len(formula) and depth > 0:
                        if formula[j] == '(':
                            depth += 1
                        elif formula[j] == ')':
                            depth -= 1
                        j += 1

            B = formula[right_start:j]
            replacement = f"(({A} ∨ {B}) ∧ ¬({A} ∧ {B}))"
            formula = formula[:left_start] + replacement + formula[j:]

        return formula

    def _normalize_fol(self, formula: str) -> str:
        formula = self._expand_xor(formula)
        replacements = {
            '∀': 'all ',
            '∃': 'exists ',
            '→': ' -> ',
            '¬': '-',
            '∧': ' & ',
            '∨': ' | ',
            '↔': ' <-> ',
            '⊤': '$T',
            '⊥': '$F',
        }
        for unicode_sym, ascii_sym in replacements.items():
            formula = formula.replace(unicode_sym, ascii_sym)
        return formula

    def prove(self, premises: list[str], conclusion: str, timeout: int = 5) -> SolverResult:
        premises = [self._normalize_fol(p) for p in premises]
        conclusion = self._normalize_fol(conclusion)
        input_text = self._build_input(premises, [conclusion])
        return self._run(input_text, timeout)

    def check_consistency(self, formulas: list[str], timeout: int = 5) -> SolverResult:
        formulas = [self._normalize_fol(f) for f in formulas]
        # Consistency = the set is satisfiable.
        # Prover9 is a refutation prover: we ask it to derive a contradiction.
        # If it *fails* to find a proof, the set is consistent.
        input_text = self._build_input(formulas, ["$F"])  # $F = false
        result = self._run(input_text, timeout)
        # Invert: proved contradiction ⇒ inconsistent; failed ⇒ consistent
        return SolverResult(
            success=result.success,
            proved=not result.proved,  # consistent = could NOT prove contradiction
            time_seconds=result.time_seconds,
            raw_output=result.raw_output,
            error=result.error,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _build_input(assumptions: list[str], goals: list[str]) -> str:
        lines: list[str] = []
        lines.append("formulas(assumptions).")
        for a in assumptions:
            stmt = a.strip().rstrip(".")
            lines.append(f"  {stmt}.")
        lines.append("end_of_list.")
        lines.append("")
        lines.append("formulas(goals).")
        for g in goals:
            stmt = g.strip().rstrip(".")
            lines.append(f"  {stmt}.")
        lines.append("end_of_list.")
        return "\n".join(lines)

    def _run(self, input_text: str, timeout: int) -> SolverResult:
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".in", prefix="prover9_")
        try:
            with os.fdopen(tmp_fd, "w") as f:
                f.write(input_text)

            t0 = time.monotonic()
            try:
                proc = subprocess.run(
                    [self._binary, "-t", str(timeout), "-f", tmp_path],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=timeout + 10,  # generous grace period
                )
                elapsed = time.monotonic() - t0
                raw = proc.stdout + proc.stderr
                proved = self._parse_proved(raw)
                return SolverResult(
                    success=True,
                    proved=proved,
                    time_seconds=elapsed,
                    raw_output=raw,
                )
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - t0
                return SolverResult(
                    success=False,
                    proved=False,
                    time_seconds=elapsed,
                    raw_output="",
                    error="subprocess timeout",
                )
        finally:
            os.unlink(tmp_path)

    @staticmethod
    def _parse_proved(output: str) -> bool:
        # Prover9 prints "THEOREM PROVED" on success
        for line in output.splitlines():
            if "THEOREM PROVED" in line.upper():
                return True
        return False
