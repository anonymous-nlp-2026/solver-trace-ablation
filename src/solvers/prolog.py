"""SWI-Prolog backend.

Calls the `swipl` binary via subprocess. Input is written as a temp .pl file.
Queries are embedded as directives: :- (Query -> halt(0) ; halt(1)).

Exit codes:
  0 — query succeeded (proved)
  1 — query failed (not proved)
  2 — fallback halt (syntax error / directive didn't execute)

Dependencies: swipl (SWI-Prolog) in PATH.
"""

import os
import shutil
import subprocess
import tempfile
import time

from .base import FOLSolver, SolverResult


class PrologSolver(FOLSolver):

    def __init__(self, binary: str = "swipl"):
        self._binary = binary

    def is_available(self) -> bool:
        return shutil.which(self._binary) is not None

    def prove(self, premises: list[str], conclusion: str, timeout: int = 5) -> SolverResult:
        input_text = self._build_input(premises, conclusion)
        return self._run(input_text, timeout)

    def check_consistency(self, formulas: list[str], timeout: int = 5) -> SolverResult:
        body = "\n".join(self._ensure_dot(f) for f in formulas)
        body += "\n:- halt(0).\n"
        result = self._run(body, timeout)
        return SolverResult(
            success=result.success,
            proved=result.success and result.error is None,
            time_seconds=result.time_seconds,
            raw_output=result.raw_output,
            error=result.error,
        )

    @staticmethod
    def _ensure_dot(clause: str) -> str:
        clause = clause.strip()
        if clause and not clause.endswith('.'):
            clause += '.'
        return clause

    def _build_input(self, premises: list[str], query: str) -> str:
        lines = [self._ensure_dot(p) for p in premises if p.strip()]
        q = query.strip()
        if q.startswith("?-"):
            q = q[2:].strip()
        q = q.rstrip('.').strip()
        lines.append(f":- ({q} -> halt(0) ; halt(1)).")
        return "\n".join(lines) + "\n"

    def _run(self, input_text: str, timeout: int) -> SolverResult:
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pl", prefix="prolog_")
        try:
            with os.fdopen(tmp_fd, "w") as f:
                f.write(input_text)

            t0 = time.monotonic()
            try:
                proc = subprocess.run(
                    [self._binary, "-q", "-f", tmp_path, "-g", "halt(2)"],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=timeout + 5,
                )
                elapsed = time.monotonic() - t0
                raw = proc.stdout + proc.stderr

                if proc.returncode == 0:
                    return SolverResult(
                        success=True, proved=True,
                        time_seconds=elapsed, raw_output=raw,
                    )
                elif proc.returncode == 1:
                    return SolverResult(
                        success=True, proved=False,
                        time_seconds=elapsed, raw_output=raw,
                    )
                else:
                    return SolverResult(
                        success=False, proved=False,
                        time_seconds=elapsed, raw_output=raw,
                        error=f"swipl exit code {proc.returncode}",
                    )

            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - t0
                return SolverResult(
                    success=False, proved=False,
                    time_seconds=elapsed, raw_output="",
                    error="subprocess timeout",
                )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
