"""Z3 SMT backend (fallback solver).

Encodes FOL strings into Z3 Python API objects via a lightweight parser
that handles ProofWriter-style syntax:
  ∀x(P(x) → Q(x)),  ∃x(P(x) ∧ Q(x)),  P(a),  ¬P(a)

Proof strategy: premises ∧ ¬conclusion is UNSAT ⟹ conclusion follows.

Dependencies: z3-solver (pip install z3-solver).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional

try:
    import z3
    _Z3_AVAILABLE = True
except ImportError:
    _Z3_AVAILABLE = False

from .base import FOLSolver, SolverResult


# ======================================================================
# Lightweight FOL parser  (ProofWriter subset)
# ======================================================================

# Supported token types
_TOKEN_PATTERNS = [
    ("FORALL",  r"(?:∀|forall)\s*"),
    ("EXISTS",  r"(?:∃|exists)\s*"),
    ("IMPLIES", r"(?:→|->|=>)"),
    ("AND",     r"(?:∧|&)"),
    ("OR",      r"(?:∨|\|)"),
    ("XOR",     r"⊕"),
    ("NOT",     r"(?:¬|~|!)"),
    ("LPAREN",  r"\("),
    ("RPAREN",  r"\)"),
    ("COMMA",   r","),
    ("IDENT",   r"[A-Za-z_][A-Za-z0-9_]*"),
    ("WS",      r"\s+"),
]
_TOKEN_RE = re.compile("|".join(f"(?P<{n}>{p})" for n, p in _TOKEN_PATTERNS))


@dataclass
class _Token:
    type: str
    value: str
    pos: int


def _tokenize(text: str) -> list[_Token]:
    tokens: list[_Token] = []
    for m in _TOKEN_RE.finditer(text):
        kind = m.lastgroup
        if kind == "WS":
            continue
        tokens.append(_Token(kind, m.group(), m.start()))
    return tokens


class FOLParser:
    """Recursive-descent parser: FOL string → Z3 expression.

    Grammar (simplified):
        expr     = quant | implExpr
        quant    = (FORALL | EXISTS) IDENT ( '(' expr ')' )
        implExpr = xorExpr ( IMPLIES xorExpr )*
        xorExpr  = orExpr ( XOR orExpr )*
        orExpr   = andExpr ( OR andExpr )*
        andExpr  = unary ( AND unary )*
        unary    = NOT unary | atom
        atom     = IDENT '(' arglist ')' | IDENT | '(' expr ')'
        arglist  = IDENT ( ',' IDENT )*
    """

    def __init__(self):
        # Accumulated predicate/function/constant declarations
        self.sorts: dict[str, "z3.SortRef"] = {}
        self.predicates: dict[str, "z3.FuncDeclRef"] = {}
        self.constants: dict[str, "z3.ExprRef"] = {}
        self._domain: Optional["z3.SortRef"] = None

    @property
    def domain(self) -> "z3.SortRef":
        if self._domain is None:
            self._domain = z3.DeclareSort("D")
        return self._domain

    def _get_const(self, name: str) -> "z3.ExprRef":
        if name not in self.constants:
            self.constants[name] = z3.Const(name, self.domain)
        return self.constants[name]

    def _get_predicate(self, name: str, arity: int) -> "z3.FuncDeclRef":
        key = f"{name}/{arity}"
        if key not in self.predicates:
            self.predicates[key] = z3.Function(name, *([self.domain] * arity), z3.BoolSort())
        return self.predicates[key]

    # ---- Recursive descent ----

    def parse(self, text: str) -> "z3.ExprRef":
        self._tokens = _tokenize(text)
        self._pos = 0
        result = self._expr()
        return result

    def _peek(self) -> Optional[_Token]:
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return None

    def _consume(self, expected: Optional[str] = None) -> _Token:
        tok = self._tokens[self._pos]
        if expected and tok.type != expected:
            raise ValueError(f"Expected {expected}, got {tok.type} ({tok.value!r}) at pos {tok.pos}")
        self._pos += 1
        return tok

    def _expr(self) -> "z3.ExprRef":
        tok = self._peek()
        if tok and tok.type in ("FORALL", "EXISTS"):
            return self._quant()
        return self._impl_expr()

    def _quant(self) -> "z3.ExprRef":
        tok = self._consume()  # FORALL or EXISTS
        var_tok = self._consume("IDENT")
        var = z3.Const(var_tok.value, self.domain)
        # Allow optional parenthesized body
        if self._peek() and self._peek().type == "LPAREN":
            self._consume("LPAREN")
            body = self._expr()
            self._consume("RPAREN")
        else:
            body = self._expr()
        if tok.type == "FORALL":
            return z3.ForAll([var], body)
        else:
            return z3.Exists([var], body)

    def _impl_expr(self) -> "z3.ExprRef":
        left = self._xor_expr()
        while self._peek() and self._peek().type == "IMPLIES":
            self._consume()
            right = self._xor_expr()
            left = z3.Implies(left, right)
        return left

    def _xor_expr(self) -> "z3.ExprRef":
        left = self._or_expr()
        while self._peek() and self._peek().type == "XOR":
            self._consume()
            right = self._or_expr()
            left = z3.Xor(left, right)
        return left

    def _or_expr(self) -> "z3.ExprRef":
        left = self._and_expr()
        while self._peek() and self._peek().type == "OR":
            self._consume()
            right = self._and_expr()
            left = z3.Or(left, right)
        return left

    def _and_expr(self) -> "z3.ExprRef":
        left = self._unary()
        while self._peek() and self._peek().type == "AND":
            self._consume()
            right = self._unary()
            left = z3.And(left, right)
        return left

    def _unary(self) -> "z3.ExprRef":
        if self._peek() and self._peek().type == "NOT":
            self._consume()
            inner = self._unary()
            return z3.Not(inner)
        return self._atom()

    def _atom(self) -> "z3.ExprRef":
        tok = self._peek()
        if tok is None:
            raise ValueError("Unexpected end of expression")

        if tok.type == "LPAREN":
            self._consume("LPAREN")
            inner = self._expr()
            self._consume("RPAREN")
            return inner

        if tok.type == "IDENT":
            name_tok = self._consume("IDENT")
            # Check if followed by '(' → predicate/function application
            if self._peek() and self._peek().type == "LPAREN":
                self._consume("LPAREN")
                args: list["z3.ExprRef"] = []
                if self._peek() and self._peek().type != "RPAREN":
                    args.append(self._parse_arg())
                    while self._peek() and self._peek().type == "COMMA":
                        self._consume("COMMA")
                        args.append(self._parse_arg())
                self._consume("RPAREN")
                pred = self._get_predicate(name_tok.value, len(args))
                return pred(*args)
            else:
                # Bare identifier — could be a boolean constant or 0-ary predicate
                # Treat uppercase-initial as 0-ary predicate, lowercase as constant
                if name_tok.value[0].isupper():
                    pred = self._get_predicate(name_tok.value, 0)
                    return pred()
                else:
                    return self._get_const(name_tok.value)

        # Handle quantifiers that appear in atom position
        if tok.type in ("FORALL", "EXISTS"):
            return self._quant()

        raise ValueError(f"Unexpected token {tok.type} ({tok.value!r}) at pos {tok.pos}")

    def _parse_arg(self) -> "z3.ExprRef":
        """Parse a predicate argument (variable/constant name)."""
        tok = self._consume("IDENT")
        return self._get_const(tok.value)


# ======================================================================
# Z3 Solver
# ======================================================================

class Z3Solver(FOLSolver):
    """FOL solver using Z3's Python API."""

    def is_available(self) -> bool:
        return _Z3_AVAILABLE

    def prove(self, premises: list[str], conclusion: str, timeout: int = 5) -> SolverResult:
        if not _Z3_AVAILABLE:
            return SolverResult(False, False, 0.0, "", error="z3 not installed")

        parser = FOLParser()
        t0 = time.monotonic()
        try:
            premise_exprs = [parser.parse(p) for p in premises]
            conclusion_expr = parser.parse(conclusion)
        except Exception as e:
            return SolverResult(
                success=False, proved=False,
                time_seconds=time.monotonic() - t0,
                raw_output="", error=f"parse error: {e}",
            )

        # Refutation: premises ∧ ¬conclusion should be UNSAT
        solver = z3.Solver()
        solver.set("timeout", timeout * 1000)
        for p in premise_exprs:
            solver.add(p)
        solver.add(z3.Not(conclusion_expr))

        result = solver.check()
        elapsed = time.monotonic() - t0
        raw = str(result)

        if result == z3.unsat:
            return SolverResult(True, True, elapsed, raw)
        elif result == z3.sat:
            return SolverResult(True, False, elapsed, raw)
        else:
            # unknown — typically timeout
            return SolverResult(False, False, elapsed, raw, error="solver returned unknown")

    def check_consistency(self, formulas: list[str], timeout: int = 5) -> SolverResult:
        if not _Z3_AVAILABLE:
            return SolverResult(False, False, 0.0, "", error="z3 not installed")

        parser = FOLParser()
        t0 = time.monotonic()
        try:
            exprs = [parser.parse(f) for f in formulas]
        except Exception as e:
            return SolverResult(
                success=False, proved=False,
                time_seconds=time.monotonic() - t0,
                raw_output="", error=f"parse error: {e}",
            )

        solver = z3.Solver()
        solver.set("timeout", timeout * 1000)
        for expr in exprs:
            solver.add(expr)

        result = solver.check()
        elapsed = time.monotonic() - t0
        raw = str(result)

        if result == z3.sat:
            return SolverResult(True, True, elapsed, raw)  # consistent
        elif result == z3.unsat:
            return SolverResult(True, False, elapsed, raw)  # inconsistent
        else:
            return SolverResult(False, False, elapsed, raw, error="solver returned unknown")
