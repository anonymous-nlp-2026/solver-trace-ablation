"""FOL component decomposition for STA.

Breaks a FOL proof problem (premises + conclusion) into individually
ablatable semantic components: each premise, and sub-parts of the
conclusion (quantifier bindings, antecedent/consequent of implications).

Input:  list of premise strings + conclusion string (ProofWriter-style FOL).
Output: list of FOLComponent, each representing one ablatable unit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class FOLComponent:
    """A single ablatable unit in a FOL proof problem."""
    type: str       # "premise" | "conclusion_quantifier" | "conclusion_antecedent" | "conclusion_consequent" | "conclusion_whole"
    content: str    # the original text of this component
    index: int      # position index (premises: 0..n-1; conclusion parts: n..)


def _split_implication(formula: str) -> tuple[str, str] | None:
    """Try to split 'A → B' at the top-level implication.

    Returns (antecedent, consequent) or None if no top-level implication.
    Handles nested parentheses so we don't split inside sub-expressions.
    """
    # Normalize arrow variants
    arrow_patterns = ["→", "->", "=>"]
    depth = 0
    for i, ch in enumerate(formula):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0:
            for arrow in arrow_patterns:
                if formula[i:i+len(arrow)] == arrow:
                    ante = formula[:i].strip()
                    cons = formula[i+len(arrow):].strip()
                    if ante and cons:
                        return ante, cons
    return None


def _strip_outer_quantifier(formula: str) -> tuple[str, str] | None:
    """Strip one leading quantifier: '∀x(body)' → ('∀x', 'body').

    Returns (quantifier_text, body) or None.
    """
    m = re.match(r'^((?:∀|forall|∃|exists)\s*[A-Za-z_]\w*)\s*\(\s*', formula)
    if not m:
        return None
    # Find the matching closing paren
    start = m.end() - 1  # position of '('
    depth = 0
    for i in range(start, len(formula)):
        if formula[i] == "(":
            depth += 1
        elif formula[i] == ")":
            depth -= 1
            if depth == 0:
                body = formula[start+1:i].strip()
                quant_text = m.group(1).strip()
                return quant_text, body
    return None


def decompose_fol(premises: list[str], conclusion: str) -> list[FOLComponent]:
    """Decompose a FOL proof problem into ablatable components.

    Each premise becomes one component. The conclusion is further
    decomposed into quantifier(s), antecedent, and consequent when
    it has implication structure.
    """
    components: list[FOLComponent] = []
    idx = 0

    # Each premise is one component
    for p in premises:
        components.append(FOLComponent(type="premise", content=p.strip(), index=idx))
        idx += 1

    # Decompose conclusion
    body = conclusion.strip()

    # Strip quantifiers
    while True:
        result = _strip_outer_quantifier(body)
        if result is None:
            break
        quant_text, inner_body = result
        components.append(FOLComponent(type="conclusion_quantifier", content=quant_text, index=idx))
        idx += 1
        body = inner_body

    # Try to split implication
    split = _split_implication(body)
    if split:
        ante, cons = split
        components.append(FOLComponent(type="conclusion_antecedent", content=ante, index=idx))
        idx += 1
        components.append(FOLComponent(type="conclusion_consequent", content=cons, index=idx))
        idx += 1
    else:
        # Conclusion is atomic or has no top-level implication
        components.append(FOLComponent(type="conclusion_whole", content=body, index=idx))
        idx += 1

    return components
