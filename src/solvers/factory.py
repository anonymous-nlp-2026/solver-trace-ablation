"""Solver factory — returns the first available backend by priority.

Usage:
    solver = get_solver()            # auto-detect (prover9)
    solver = get_solver("z3")        # prefer Z3
    solver = get_solver("prolog")    # SWI-Prolog
"""

from .base import FOLSolver
from .prover9 import Prover9Solver
from .z3_solver import Z3Solver
from .prolog import PrologSolver

_REGISTRY: dict[str, type[FOLSolver]] = {
    "prover9": Prover9Solver,
    "z3": Z3Solver,
    "prolog": PrologSolver,
    "swipl": PrologSolver,
}

_FALLBACK_ORDER = ["prover9", "z3"]


def get_solver(preferred: str = "prover9") -> FOLSolver:
    """Return an available solver, falling back through alternatives.

    Args:
        preferred: Name of the preferred backend.

    Returns:
        An instantiated FOLSolver.

    Raises:
        RuntimeError: No solver backend is available.
    """
    if preferred in ("prolog", "swipl"):
        cls = _REGISTRY[preferred]
        instance = cls()
        if instance.is_available():
            return instance
        raise RuntimeError(f"SWI-Prolog (swipl) not found in PATH.")

    order = [preferred] + [s for s in _FALLBACK_ORDER if s != preferred]

    for name in order:
        cls = _REGISTRY.get(name)
        if cls is None:
            continue
        instance = cls()
        if instance.is_available():
            return instance

    available = list(_REGISTRY.keys())
    raise RuntimeError(
        f"No solver available. Tried: {order}. "
        f"Install prover9 (CLI), z3-solver (pip), or swipl."
    )
