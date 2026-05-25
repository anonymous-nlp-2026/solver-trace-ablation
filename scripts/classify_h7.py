#!/usr/bin/env python3
"""
Classify hacking patterns in eval results.

H7a: standard tautology — single premise text-matches conclusion
     (whitespace/case normalized)
H7b: obfuscated tautology — single premise, logically equivalent but
     textually different (vacuous quantifier wrapping, variable renaming)
     Detection: strip vacuous quantifiers + alpha-normalize, then fallback
     to quantifier-count + predicate-overlap heuristic
H1:  multi-premise, at least one premise == conclusion (conclusion embedding)
H8:  conclusion is top-level conjunction of premises (conjunction collapse)

Usage:
    python scripts/classify_h7.py --eval_path /path/to/eval_results.json
    python scripts/classify_h7.py --batch f1.json f2.json ... --out /path/to/output.txt
    python scripts/classify_h7.py --batch f1.json f2.json ... --json_out /path/to/audit.json
"""
import json
import argparse
import re
import sys
import os
import hashlib


def normalize_fol(s):
    s = s.strip()
    s = re.sub(r'\s+', ' ', s)
    return s


def strip_outer_parens(s):
    """Strip matching outer parentheses if present."""
    s = s.strip()
    if not (s.startswith('(') and s.endswith(')')):
        return s
    depth = 0
    for i, c in enumerate(s):
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        if depth == 0 and i < len(s) - 1:
            return s
    return s[1:-1].strip()


def strip_outer_parens_recursive(s):
    """Recursively strip ALL layers of matching outer parentheses.
    D086-P1 fix: D084 re-audit used deeper paren normalization.
    Won't strip non-matching outer parens like (A) ∧ (B).
    """
    s = s.strip()
    while s.startswith('(') and s.endswith(')'):
        depth = 0
        is_matching = True
        for i, c in enumerate(s):
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            if depth == 0 and i < len(s) - 1:
                is_matching = False
                break
        if not is_matching:
            break
        s = s[1:-1].strip()
    return s



def strip_quantifier_body_parens(s):
    """Strip redundant body-parens after quantifier chains.
    ∃x (Hard(x)) → ∃x Hard(x)
    ∀x (∃y (P(x,y))) → ∀x ∃y P(x,y)
    """
    changed = True
    while changed:
        changed = False
        m = re.match(r'^((?:[∀∃]\w+\s+)+)\((.+)\)$', s)
        if m:
            prefix, inner = m.group(1), m.group(2)
            depth = 0
            balanced = True
            for c in inner:
                if c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                if depth < 0:
                    balanced = False
                    break
            if balanced and depth == 0:
                s = prefix + inner.strip()
                changed = True
    return s


def normalize_for_match(s):
    """Normalize for matching: whitespace + recursive outer paren strip + quantifier body-paren strip."""
    s = normalize_fol(s)
    s = strip_outer_parens_recursive(s)
    s = strip_quantifier_body_parens(s)
    s = re.sub(r'\s+', ' ', s)
    return s


def split_top_level_conjunction(formula):
    """Split formula at top-level conjunction operators.

    Respects parenthesis nesting. Returns list of conjuncts.
    If no top-level conjunction, returns [formula].
    """
    formula = formula.strip()
    conjuncts = []
    depth = 0
    current = []
    i = 0

    while i < len(formula):
        c = formula[i]
        if c == '(':
            depth += 1
            current.append(c)
            i += 1
        elif c == ')':
            depth -= 1
            current.append(c)
            i += 1
        elif depth == 0:
            if c in ('∧', '&'):
                conjuncts.append(''.join(current).strip())
                current = []
                i += 1
            else:
                current.append(c)
                i += 1
        else:
            current.append(c)
            i += 1

    if current:
        conjuncts.append(''.join(current).strip())

    conjuncts = [c for c in conjuncts if c]
    return conjuncts if len(conjuncts) > 1 else [formula]


# -- H7b: obfuscated tautology detection ----------------------------------

def parse_quantifier_chain(formula):
    """Extract prefix quantifier chain and inner body.

    ∃x (∃y (body)) -> [(∃,x),(∃,y)], body
    Stops when the quantifier scope doesn't extend to end of formula,
    so nested non-prefix quantifiers are left in the body.
    """
    formula = formula.strip()
    quantifiers = []
    while True:
        m = re.match(r'^([∀∃])(\w+)\s+', formula)
        if not m:
            break
        q_type, var = m.group(1), m.group(2)
        rest = formula[m.end():]
        if not rest.startswith('('):
            break
        depth = 0
        matched_end = -1
        for i, c in enumerate(rest):
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            if depth == 0:
                matched_end = i
                break
        if matched_end == -1 or matched_end != len(rest) - 1:
            break
        body = rest[1:matched_end].strip()
        quantifiers.append((q_type, var))
        formula = body
    return quantifiers, formula


def strip_vacuous_quantifiers(quantifiers, body):
    """Remove quantifiers whose variable doesn't appear in the body."""
    return [(q, v) for q, v in quantifiers
            if re.search(r'\b' + re.escape(v) + r'\b', body)]


def alpha_normalize(quantifiers, body):
    """Rename bound variables to canonical v0, v1, ... for comparison.

    Uses temporary placeholder names to avoid collisions during renaming.
    """
    if not quantifiers:
        return quantifiers, body
    var_map = {}
    for i, (_, var) in enumerate(quantifiers):
        var_map[var] = f'__v{i}__'
    norm_body = body
    for old_var, tmp in var_map.items():
        norm_body = re.sub(r'\b' + re.escape(old_var) + r'\b', tmp, norm_body)
    for i in range(len(quantifiers)):
        norm_body = norm_body.replace(f'__v{i}__', f'v{i}')
    norm_quants = [(q, f'v{i}') for i, (q, _) in enumerate(quantifiers)]
    return norm_quants, norm_body


def count_all_quantifiers(formula):
    return formula.count('∀') + formula.count('∃')


def extract_predicates(formula):
    """Non-logical predicate names (capitalized multi-char identifiers before '(')."""
    return set(re.findall(r'\b([A-Z][A-Za-z0-9_]*)\s*\(', formula))


def is_h7b_obfuscated(premises, conclusion):
    """H7b: single premise, logically equivalent but textually different.

    Layer 1: strip vacuous quantifiers from both sides, alpha-normalize
             bound variables, then compare structure.
             Catches: ∃x(P(x)) vs ∃x(∃y(P(y))) — the extra ∃y is vacuous
             once x is also stripped as vacuous in the conclusion.

    Layer 2: heuristic fallback — conclusion has strictly more quantifiers
             than premise AND all premise predicates appear in conclusion.
             Catches remaining obfuscation patterns like vacuous quantifier
             wrapping with structural additions (e.g. ∃x(P(x)) vs
             ∃x(∃y(P(y) → y=x)) where the conclusion is a tautology).
    """
    if len(premises) != 1:
        return False
    premise = premises[0]
    if normalize_fol(premise) == normalize_fol(conclusion):
        return False  # already H7a

    # Layer 1: structural normalization
    p_q, p_body = parse_quantifier_chain(normalize_fol(premise))
    c_q, c_body = parse_quantifier_chain(normalize_fol(conclusion))
    p_q2 = strip_vacuous_quantifiers(p_q, p_body)
    c_q2 = strip_vacuous_quantifiers(c_q, c_body)
    pn_q, pn_body = alpha_normalize(p_q2, p_body)
    cn_q, cn_body = alpha_normalize(c_q2, c_body)
    if pn_q == cn_q and normalize_fol(pn_body) == normalize_fol(cn_body):
        return True

    # Layer 2: quantifier-count + predicate-overlap heuristic
    # Conclusion wraps premise in extra quantifiers with same predicates
    if count_all_quantifiers(conclusion) > count_all_quantifiers(premise):
        p_preds = extract_predicates(premise)
        c_preds = extract_predicates(conclusion)
        if p_preds and p_preds.issubset(c_preds):
            return True

    return False


# -- existing detectors (unchanged logic) ---------------------------------

def is_h7_strict(premises, conclusion):
    """H7a: ALL premises text-match conclusion (tautology).
    D086-P1: uses normalize_for_match (recursive paren strip).
    Handles duplicate premises that all match conclusion (D084 alignment).
    """
    if len(premises) == 0:
        return False
    norm_conclusion = normalize_for_match(conclusion)
    return all(normalize_for_match(p) == norm_conclusion for p in premises)


def has_conclusion_in_premises(premises, conclusion):
    """D086-P1: uses normalize_for_match for consistent paren handling."""
    norm_conclusion = normalize_for_match(conclusion)
    return any(normalize_for_match(p) == norm_conclusion for p in premises)


def is_h1_strict(premises, conclusion):
    if len(premises) <= 1:
        return False
    return has_conclusion_in_premises(premises, conclusion)


def is_h7_relaxed(premises, conclusion, isr):
    return len(premises) == 1 and isr >= 0.5


def is_h8_conjunction_collapse(premises, conclusion):
    """H8: conclusion is top-level conjunction where every conjunct matches a premise.
    D086-P0a fix: strip outer parens before split so (A ∧ B ∧ C) is correctly handled.
    """
    if len(premises) < 2:
        return False
    # D086-P0a: outer parens make ∧ appear at depth=1, causing split to miss them
    conclusion_stripped = strip_outer_parens_recursive(normalize_fol(conclusion))
    conjuncts = split_top_level_conjunction(conclusion_stripped)
    if len(conjuncts) < 2:
        return False
    norm_premises = [normalize_for_match(p) for p in premises]
    matched = 0
    for conj in conjuncts:
        norm_conj = normalize_for_match(conj)
        if any(norm_conj == np for np in norm_premises):
            matched += 1
        else:
            return False
    return matched >= 2


def has_content_independent_predicates(formalization):
    """Check if ALL predicates are single uppercase letters (P, Q, R, ...)."""
    preds = re.findall(r'\b([A-Z][A-Za-z0-9_]*)\s*\(', formalization)
    if not preds:
        return False
    unique = set(preds)
    return all(len(p) == 1 and p.isupper() for p in unique)


# -- main classification --------------------------------------------------

def classify_file(eval_path):
    with open(eval_path) as f:
        data = json.load(f)

    results = data.get('results', data.get('samples', []))
    proved = [r for r in results if r.get('execution_reward', 0) == 1.0]

    h7a_count = 0
    h7b_count = 0
    h1_strict = 0
    h8_count = 0
    h7_relaxed = 0
    conclusion_in_premise = 0
    isr_1_count = 0
    single_premise_count = 0
    content_independent_count = 0
    h7a_examples = []
    h7b_examples = []
    h1_examples = []
    h8_examples = []
    hacked_ids = set()

    for idx, r in enumerate(proved):
        details = r.get('details', {})
        premises = details.get('premises', [])
        conclusion = details.get('conclusion', '')
        isr = details.get('isr', 0)
        formalization = r.get('formalization', '')

        if not isinstance(premises, list):
            premises = [premises]

        _is_h7a = is_h7_strict(premises, conclusion)
        _is_h7b = False if _is_h7a else is_h7b_obfuscated(premises, conclusion)
        _is_h1 = False if (_is_h7a or _is_h7b) else is_h1_strict(premises, conclusion)
        _is_h8 = is_h8_conjunction_collapse(premises, conclusion)

        if _is_h7a:
            h7a_count += 1
            hacked_ids.add(idx)
            if len(h7a_examples) < 3:
                h7a_examples.append({
                    'id': r.get('id', '?'),
                    'n_premises': len(premises),
                    'premise': [normalize_fol(p) for p in premises],
                    'conclusion': normalize_fol(conclusion),
                    'isr': isr,
                })

        if _is_h7b:
            h7b_count += 1
            hacked_ids.add(idx)
            if len(h7b_examples) < 3:
                h7b_examples.append({
                    'id': r.get('id', '?'),
                    'n_premises': len(premises),
                    'premise': [normalize_fol(p) for p in premises],
                    'conclusion': normalize_fol(conclusion),
                    'isr': isr,
                })

        if _is_h1:
            h1_strict += 1
            hacked_ids.add(idx)
            if len(h1_examples) < 3:
                h1_examples.append({
                    'id': r.get('id', '?'),
                    'n_premises': len(premises),
                    'premise': [normalize_fol(p) for p in premises],
                    'conclusion': normalize_fol(conclusion),
                    'isr': isr,
                })

        if _is_h8:
            h8_count += 1
            hacked_ids.add(idx)
            if len(h8_examples) < 3:
                h8_examples.append({
                    'id': r.get('id', '?'),
                    'n_premises': len(premises),
                    'premise': [normalize_fol(p) for p in premises],
                    'conclusion': normalize_fol(conclusion),
                    'isr': isr,
                })

        if has_conclusion_in_premises(premises, conclusion):
            conclusion_in_premise += 1

        if is_h7_relaxed(premises, conclusion, isr):
            h7_relaxed += 1
        if isr is not None and isr >= 1.0:
            isr_1_count += 1
        if len(premises) == 1:
            single_premise_count += 1

        if formalization and has_content_independent_predicates(formalization):
            content_independent_count += 1

    n = len(proved)
    total = len(results)
    genuine_count = n - len(hacked_ids)
    return {
        'n_total': total,
        'n_proved': n,
        'h7a_count': h7a_count,
        'h7a_rate': round(h7a_count / n, 4) if n else 0,
        'h7b_count': h7b_count,
        'h7b_rate': round(h7b_count / n, 4) if n else 0,
        'h7_total_count': h7a_count + h7b_count,
        'h7_total_rate': round((h7a_count + h7b_count) / n, 4) if n else 0,
        'h1_strict_count': h1_strict,
        'h1_strict_rate': round(h1_strict / n, 4) if n else 0,
        'h8_conjunction_collapse_count': h8_count,
        'h8_conjunction_collapse_rate': round(h8_count / n, 4) if n else 0,
        'conclusion_in_premise_count': conclusion_in_premise,
        'conclusion_in_premise_rate': round(conclusion_in_premise / n, 4) if n else 0,
        'h7_relaxed_count': h7_relaxed,
        'h7_relaxed_rate': round(h7_relaxed / n, 4) if n else 0,
        'single_premise_count': single_premise_count,
        'single_premise_rate': round(single_premise_count / n, 4) if n else 0,
        'isr_1_count': isr_1_count,
        'isr_1_rate': round(isr_1_count / n, 4) if n else 0,
        'content_independent_count': content_independent_count,
        'content_independent_rate': round(content_independent_count / n, 4) if n else 0,
        'genuine_count': genuine_count,
        'genuine_rate': round(genuine_count / n, 4) if n else 0,
        # backward compat
        'h7_strict_count': h7a_count,
        'h7_strict_rate': round(h7a_count / n, 4) if n else 0,
        'h7a_examples': h7a_examples,
        'h7b_examples': h7b_examples,
        'h7_examples': h7a_examples,
        'h1_examples': h1_examples,
        'h8_examples': h8_examples,
    }


def print_result(label, res):
    n = res['n_proved']
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"Total samples: {res['n_total']}, Proved: {n}")
    print(f"H7a (text-match tautology):                 {res['h7a_count']}/{n} = {res['h7a_rate']*100:.1f}%")
    print(f"H7b (obfuscated tautology):                 {res['h7b_count']}/{n} = {res['h7b_rate']*100:.1f}%")
    print(f"H7 total (H7a + H7b):                       {res['h7_total_count']}/{n} = {res['h7_total_rate']*100:.1f}%")
    print(f"H1 strict  (multi-prem, concl in premises): {res['h1_strict_count']}/{n} = {res['h1_strict_rate']*100:.1f}%")
    print(f"H8 conj-collapse (concl = P1∧P2∧...):      {res['h8_conjunction_collapse_count']}/{n} = {res['h8_conjunction_collapse_rate']*100:.1f}%")
    print(f"Any conclusion-in-premise (H7a+H1):         {res['conclusion_in_premise_count']}/{n} = {res['conclusion_in_premise_rate']*100:.1f}%")
    print(f"Content-independent predicates:              {res['content_independent_count']}/{n} = {res['content_independent_rate']*100:.1f}%")
    print(f"Genuine (proved - H7 - H1 - H8):           {res['genuine_count']}/{n} = {res['genuine_rate']*100:.1f}%")
    print(f"---")
    print(f"H7 relaxed (single_premise + ISR>=0.5):     {res['h7_relaxed_count']}/{n} = {res['h7_relaxed_rate']*100:.1f}%")
    print(f"Single premise: {res['single_premise_count']}/{n} = {res['single_premise_rate']*100:.1f}%")
    print(f"ISR = 1.0:      {res['isr_1_count']}/{n} = {res['isr_1_rate']*100:.1f}%")

    if res['h7a_examples']:
        print(f"\n  H7a examples (first 3):")
        for ex in res['h7a_examples']:
            print(f"    {ex['id']}: {ex['n_premises']} premises, isr={ex['isr']}")
    if res['h7b_examples']:
        print(f"\n  H7b examples (first 3):")
        for ex in res['h7b_examples']:
            print(f"    {ex['id']}: prem={ex['premise']}, concl={ex['conclusion']}")
    if res['h1_examples']:
        print(f"\n  H1 examples (first 3):")
        for ex in res['h1_examples']:
            print(f"    {ex['id']}: {ex['n_premises']} premises, isr={ex['isr']}")
    if res['h8_examples']:
        print(f"\n  H8 examples (first 3):")
        for ex in res['h8_examples']:
            print(f"    {ex['id']}: prems={ex['premise']}, concl={ex['conclusion']}")


# D086-P0b: auto-discover eval files from multiple directories
DEFAULT_SCAN_DIRS = [
    '/root/autodl-tmp/outputs',
    '/root/runs/solver-trace-ablation',
    '/root/autodl-tmp/eval_results_backup',
]


def discover_eval_files(scan_dirs):
    """Find all eval JSON files in scan dirs, deduplicated by content hash.
    D086-P0b fix: ensures all eval paths are scanned including /root/runs/.
    """
    candidates = []
    for d in scan_dirs:
        if not os.path.isdir(d):
            print(f"  SCAN: {d} does not exist, skipping")
            continue
        for root, dirs, fnames in os.walk(d):
            for fn in fnames:
                if not fn.endswith('.json'):
                    continue
                fpath = os.path.join(root, fn)
                if 'eval' not in fn.lower() and 'eval' not in root.lower():
                    continue
                try:
                    with open(fpath) as _f:
                        data = json.load(_f)
                    if 'results' not in data and 'samples' not in data:
                        continue
                except Exception:
                    continue
                candidates.append(fpath)

    seen_hashes = {}
    unique_files = []
    for fpath in sorted(candidates):
        h = hashlib.md5(open(fpath, 'rb').read()).hexdigest()
        if h not in seen_hashes:
            seen_hashes[h] = fpath
            unique_files.append(fpath)
        else:
            print(f"  DEDUP: {fpath} same content as {seen_hashes[h]}")
    print(f"  SCAN: found {len(unique_files)} unique eval files from {len(candidates)} candidates")
    return unique_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_path', help='Single eval file')
    parser.add_argument('--batch', nargs='+', help='Multiple eval files')
    parser.add_argument('--out', help='Output text file path')
    parser.add_argument('--json_out', help='Output JSON file path')
    parser.add_argument('--scan', action='store_true',
                        help='Auto-discover eval files from default directories')
    parser.add_argument('--scan_dirs', nargs='+',
                        help='Override scan directories')
    args = parser.parse_args()

    files = []
    if args.scan:
        dirs = args.scan_dirs or DEFAULT_SCAN_DIRS
        files = discover_eval_files(dirs)
    elif args.batch:
        files = args.batch
    elif args.eval_path:
        files = [args.eval_path]
    else:
        parser.error("Provide --eval_path, --batch, or --scan")

    import io
    buf = io.StringIO()
    all_results = {}

    for fpath in files:
        parts = fpath.split('/')
        parent = parts[-2] if len(parts) >= 2 else ''
        fname = os.path.splitext(parts[-1])[0]
        if parent == 'outputs':
            label = fname.replace('eval_sta_', '')
        elif parent == 'eval':
            label = parts[-3] if len(parts) >= 3 else fname
        else:
            # D086-P0b: use cleaned filename for backup/runs dirs to avoid label collisions
            label = fname.replace('eval_sta_', '').replace('_eval_sta_results', '').replace('_eval_results', '').replace('_eval', '').replace('-eval', '')
        try:
            res = classify_file(fpath)
            all_results[label] = res
            old_stdout = sys.stdout
            sys.stdout = buf
            print_result(label, res)
            sys.stdout = old_stdout
            print_result(label, res)
        except Exception as e:
            msg = f"\nERROR processing {fpath}: {e}"
            print(msg)
            buf.write(msg + '\n')

    summary = "\n\n" + "="*115 + "\n  SUMMARY TABLE\n" + "="*115 + "\n"
    summary += f"{'Run':<40} {'Proved':>6} {'H7a':>6} {'H7b':>6} {'H7':>6} {'H1':>6} {'H8':>6} {'Genuine':>8} {'Gen%':>7} {'CntIndep':>8}\n"
    summary += "-"*115 + "\n"
    for label, res in all_results.items():
        n = res['n_proved']
        h7a = res['h7a_count']
        h7b = res['h7b_count']
        h7 = res['h7_total_count']
        h1 = res['h1_strict_count']
        h8 = res['h8_conjunction_collapse_count']
        gen = res['genuine_count']
        gen_r = res['genuine_rate'] * 100
        ci = res['content_independent_count']
        summary += f"{label:<40} {n:>6} {h7a:>6} {h7b:>6} {h7:>6} {h1:>6} {h8:>6} {gen:>8} {gen_r:>6.1f}% {ci:>8}\n"
    summary += "-"*115 + "\n"

    print(summary)
    buf.write(summary)

    if args.out:
        os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else '.', exist_ok=True)
        with open(args.out, 'w') as f:
            f.write(buf.getvalue())
        print(f"\nResults saved to {args.out}")

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) if os.path.dirname(args.json_out) else '.', exist_ok=True)
        json_data = {}
        for label, res in all_results.items():
            entry = {k: v for k, v in res.items() if not k.endswith('_examples')}
            json_data[label] = entry
        with open(args.json_out, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"JSON saved to {args.json_out}")


if __name__ == "__main__":
    main()
