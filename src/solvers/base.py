"""Abstract solver interface for FOL theorem proving.

Defines SolverResult (proof outcome) and FOLSolver (abstract base class).
All concrete solvers (Prover9, Z3) implement FOLSolver.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SolverResult:
    """Result of a solver invocation."""
    success: bool           # solver executed without error/timeout
    proved: bool            # whether the goal was proved
    time_seconds: float     # wall-clock time
    raw_output: str         # solver's stdout/stderr
    error: Optional[str] = None


class FOLSolver(ABC):
    """Abstract first-order logic solver."""

    @abstractmethod
    def prove(self, premises: list[str], conclusion: str, timeout: int = 5) -> SolverResult:
        """Try to prove conclusion from premises."""
        pass

    @abstractmethod
    def check_consistency(self, formulas: list[str], timeout: int = 5) -> SolverResult:
        """Check whether a set of formulas is consistent (satisfiable)."""
        pass

    def is_available(self) -> bool:
        """Check whether this solver backend is usable."""
        return True
