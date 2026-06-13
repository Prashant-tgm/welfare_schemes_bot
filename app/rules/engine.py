"""
Hybrid Eligibility Rules Engine
================================
Two rule types, stored per-scheme in the `rules` table:

1. rule_type="json" — a condition tree:
   {
     "op": "and" | "or" | "not",
     "conditions": [ ... nested trees or leaf conditions ... ]
   }
   Leaf condition:
   {
     "field": "annual_income",
     "op": "<=" | ">=" | "<" | ">" | "==" | "!=" | "in" | "not_in" | "contains",
     "value": 250000
   }

   Supports arbitrary nesting -> handles "AND of (income<=2.5L) AND (OR of
   BPL-card OR SC/ST OR disability) AND (state == 'Bihar' OR state == 'UP')"
   style regulatory logic without writing code.

2. rule_type="python" — escape hatch for genuinely complex/state-specific
   regulations that don't fit a condition tree (slab-based formulas,
   date-dependent windows, cross-field dependencies). `rule_code` is a
   single Python expression evaluated in a RESTRICTED namespace (no
   builtins/imports) against a `user` dict.

Both rule types return (eligible: bool, reasons: list[str]).
ALL rules for a scheme must pass for the scheme to be eligible. This lets
you mix one JSON rule for the general policy + one Python rule for a
state-specific override.
"""
import ast
import operator
import logging

logger = logging.getLogger("rules.engine")

_COMPARATORS = {
    "<=": operator.le,
    ">=": operator.ge,
    "<": operator.lt,
    ">": operator.gt,
    "==": operator.eq,
    "!=": operator.ne,
}


# ---------------------------------------------------------------------------
# JSON condition tree evaluator
# ---------------------------------------------------------------------------
def eval_json_rule(node: dict, user: dict) -> tuple:
    """Recursively evaluate a JSON condition tree against user dict."""
    if "op" in node and node["op"] in ("and", "or", "not"):
        op = node["op"]
        conditions = node.get("conditions", [])
        results = [eval_json_rule(c, user) for c in conditions]

        if op == "and":
            ok = all(r[0] for r in results)
        elif op == "or":
            ok = any(r[0] for r in results)
        else:  # not
            ok = not results[0][0] if results else True

        reasons = []
        for r in results:
            reasons.extend(r[1])
        return ok, reasons

    field = node.get("field")
    cmp_op = node.get("op")
    expected = node.get("value")
    actual = user.get(field)

    ok, reason = _eval_leaf(field, cmp_op, actual, expected)
    return ok, ([] if ok else [reason])


def _eval_leaf(field, cmp_op, actual, expected):
    try:
        if cmp_op in _COMPARATORS:
            if actual is None:
                return False, f"'{field}' not provided"
            return _COMPARATORS[cmp_op](actual, expected), (
                f"'{field}'={actual} fails {cmp_op} {expected}"
            )
        elif cmp_op == "in":
            ok = actual in expected
            return ok, f"'{field}'={actual} not in {expected}"
        elif cmp_op == "not_in":
            ok = actual not in expected
            return ok, f"'{field}'={actual} is in excluded set {expected}"
        elif cmp_op == "contains":
            ok = expected in (actual or [])
            return ok, f"'{field}' does not contain '{expected}'"
        elif cmp_op == "exists":
            ok = actual is not None
            return ok, f"'{field}' not provided"
        else:
            logger.warning(f"Unknown rule operator: {cmp_op}")
            return False, f"unknown operator '{cmp_op}' for field '{field}'"
    except TypeError as e:
        return False, f"type error evaluating '{field}': {e}"


# ---------------------------------------------------------------------------
# Python expression escape hatch — RESTRICTED evaluation
# ---------------------------------------------------------------------------
_ALLOWED_AST_NODES = (
    ast.Expression, ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Compare,
    ast.Name, ast.Load, ast.Constant, ast.And, ast.Or, ast.Not,
    ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq, ast.In, ast.NotIn,
    ast.List, ast.Tuple, ast.Dict, ast.Call, ast.Attribute, ast.Subscript,
    ast.GeneratorExp, ast.comprehension, ast.Store,
)

_ALLOWED_CALL_NAMES = {"len", "min", "max", "abs", "any", "all"}
_ALLOWED_NAMES = {"user", "True", "False", "None"} | _ALLOWED_CALL_NAMES


def _validate_python_rule(code: str):
    """Parse and whitelist-check the expression AST before allowing eval."""
    tree = ast.parse(code, mode="eval")

    # Collect names bound by comprehensions (e.g. "c" in "any(c in X for c in Y)")
    bound_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.comprehension) and isinstance(node.target, ast.Name):
            bound_names.add(node.target.id)

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_AST_NODES):
            raise ValueError(f"Disallowed expression element: {type(node).__name__}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) and not isinstance(node.func, ast.Attribute):
                raise ValueError("Disallowed call target")
            if isinstance(node.func, ast.Name) and node.func.id not in _ALLOWED_CALL_NAMES:
                raise ValueError(f"Disallowed function call: {node.func.id}")
        if isinstance(node, ast.Attribute) and node.attr not in ("get",):
            raise ValueError(f"Disallowed attribute access: .{node.attr}")
        if isinstance(node, ast.Name) and node.id not in _ALLOWED_NAMES and node.id not in bound_names:
            raise ValueError(f"Disallowed name: {node.id}")
    return tree


def eval_python_rule(code: str, user: dict) -> tuple:
    """
    Evaluate a restricted Python expression against `user`.
    Only `user.get()`, comparisons, boolean ops, and a small whitelist of
    builtins (len/min/max/abs/any/all) are permitted. Raises ValueError if
    the expression contains disallowed constructs.
    """
    tree = _validate_python_rule(code)
    compiled = compile(tree, "<rule>", "eval")

    builtins_src = __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__)
    safe_globals = {"__builtins__": {}, "user": user}
    for name in _ALLOWED_CALL_NAMES:
        if name in builtins_src:
            safe_globals[name] = builtins_src[name]

    try:
        result = eval(compiled, safe_globals)
    except Exception as e:
        return False, [f"rule evaluation error: {e}"]

    if result:
        return True, []
    return False, [f"custom rule not satisfied: {code[:120]}"]


# ---------------------------------------------------------------------------
# Top-level: evaluate all rules for a scheme
# ---------------------------------------------------------------------------
def check_scheme_eligibility(rules: list, user: dict) -> tuple:
    """
    rules: list of db.Rule ORM objects (or dicts with rule_type/rule_body/
    rule_code/id/priority). Returns (eligible, reasons, matched_rule_ids) —
    matched_rule_ids feeds the provenance/claim graph so we know exactly
    which rule rows produced the verdict.
    """
    if not rules:
        # No rules scraped/defined yet -> cannot assert eligibility (avoid
        # silently saying "yes" with zero basis).
        return False, ["no eligibility rules available for this scheme yet"], []

    def _get(r, attr, default=None):
        return getattr(r, attr, None) if hasattr(r, attr) else r.get(attr, default)

    all_reasons = []
    used_rule_ids = []
    overall_ok = True

    for rule in sorted(rules, key=lambda r: _get(r, "priority", 0) or 0):
        rule_type = _get(rule, "rule_type")
        rule_id = _get(rule, "id")
        used_rule_ids.append(rule_id)

        if rule_type == "json":
            body = _get(rule, "rule_body")
            ok, reasons = eval_json_rule(body, user)
        elif rule_type == "python":
            code = _get(rule, "rule_code")
            try:
                ok, reasons = eval_python_rule(code, user)
            except ValueError as e:
                logger.error(f"Rule {rule_id} rejected by validator: {e}")
                ok, reasons = False, [f"rule {rule_id} invalid: {e}"]
        else:
            ok, reasons = False, [f"unknown rule_type '{rule_type}'"]

        if not ok:
            overall_ok = False
            all_reasons.extend(reasons)

    return overall_ok, all_reasons, used_rule_ids


# ---------------------------------------------------------------------------
# Default rule generation from scraped eligibility text (heuristic seed)
# ---------------------------------------------------------------------------
def generate_default_rules(db, scheme, record: dict):
    """
    Dispatches to a precise generator if the record carries a structured
    `_raw_eligibility` dict (from local_seed), otherwise falls back to the
    heuristic text-based extractor for live-scraped records.
    """
    if record.get("_raw_eligibility") is not None:
        _generate_rules_from_structured(db, scheme, record["_raw_eligibility"])
    else:
        _generate_rules_from_text(db, scheme, record)


def _generate_rules_from_structured(db, scheme, elig: dict):
    """
    Build precise JSON + python rules from a structured eligibility dict
    (the shape used by v1's data/schemes.json). This demonstrates the
    hybrid approach: most fields map to simple JSON leaf conditions; a
    couple of cross-field/complex cases use python rules.
    """
    from app.db import Rule

    db.query(Rule).filter(Rule.scheme_id == scheme.id).delete()

    conditions = []

    if "occupation" in elig:
        conditions.append({"field": "occupation", "op": "in", "value": elig["occupation"]})

    if "gender" in elig:
        conditions.append({"field": "gender", "op": "in", "value": elig["gender"]})

    if elig.get("min_age") is not None:
        conditions.append({"field": "age", "op": ">=", "value": elig["min_age"]})

    if elig.get("max_age") is not None:
        conditions.append({"field": "age", "op": "<=", "value": elig["max_age"]})

    if elig.get("income_limit") is not None:
        conditions.append({"field": "annual_income", "op": "<=", "value": elig["income_limit"]})

    if elig.get("land_owner"):
        conditions.append({"field": "land_owner", "op": "==", "value": True})

    if elig.get("house_status") == "kutcha_or_no_house":
        conditions.append({"field": "has_pucca_house", "op": "==", "value": False})

    if elig.get("residence"):
        conditions.append({"field": "residence", "op": "==", "value": elig["residence"]})

    if elig.get("marital_status") == "widow":
        conditions.append({"field": "is_widow", "op": "==", "value": True})

    if elig.get("vendor_type") == "street_vendor_or_hawker":
        conditions.append({"field": "is_street_vendor", "op": "==", "value": True})

    if conditions:
        rule = Rule(
            scheme_id=scheme.id,
            rule_type="json",
            rule_body={"op": "and", "conditions": conditions},
            description=f"Core eligibility criteria for '{scheme.name}' (structured seed data).",
            source_url=scheme.source_url,
            priority=0,
        )
        db.add(rule)

    # --- special_categories: OR-of-membership in a list field -> python rule
    # (JSON 'contains' only checks a single value; an OR-across-categories
    # check against the user's special_categories list needs a small
    # expression, demonstrating the escape hatch for non-trivial logic.)
    se_categories = elig.get("se_categories")
    if se_categories:
        cats_repr = repr(se_categories)
        code = (
            f"any(c in user.get('special_categories', []) for c in {cats_repr}) "
            f"or user.get('is_widow', False) "
            f"or user.get('has_girl_child_under10', False) "
            f"or user.get('is_street_vendor', False)"
        )
        rule = Rule(
            scheme_id=scheme.id,
            rule_type="python",
            rule_code=code,
            description=(
                f"'{scheme.name}' requires membership in at least one special "
                f"category: {se_categories} (or an equivalent flag). "
                f"Escape-hatch rule: OR-across-list membership check."
            ),
            source_url=scheme.source_url,
            priority=1,
        )
        db.add(rule)

    if elig.get("land_owner_or_tenant"):
        code = "user.get('land_owner', False) or user.get('is_tenant_farmer', False)"
        rule = Rule(
            scheme_id=scheme.id,
            rule_type="python",
            rule_code=code,
            description=f"'{scheme.name}' requires land ownership OR tenancy.",
            source_url=scheme.source_url,
            priority=1,
        )
        db.add(rule)

    if elig.get("max_age_dependent") is not None:
        code = "user.get('has_girl_child_under10', False)"
        rule = Rule(
            scheme_id=scheme.id,
            rule_type="python",
            rule_code=code,
            description=f"'{scheme.name}' requires a dependent girl child under {elig['max_age_dependent']}.",
            source_url=scheme.source_url,
            priority=1,
        )
        db.add(rule)

    if not conditions and not se_categories and not elig.get("land_owner_or_tenant") and elig.get("max_age_dependent") is None:
        rule = Rule(
            scheme_id=scheme.id,
            rule_type="json",
            rule_body={"op": "and", "conditions": [{"field": "_manual_review_required", "op": "==", "value": True}]},
            description=f"No structured criteria found for '{scheme.name}'. Needs manual review.",
            source_url=scheme.source_url,
            priority=10,
        )
        db.add(rule)


def _generate_rules_from_text(db, scheme, record: dict):
    """
    Generate a *baseline* JSON rule from scraped eligibility text using
    simple keyword heuristics. Conservative by design — extracts obvious
    structured signals (income caps, age, gender, occupation keywords) and
    leaves everything else for manual/python rule refinement.

    Heuristic rules are tagged priority=10. Hand-authored rules (lower
    priority values) run first, but in check_scheme_eligibility ALL rules
    must pass — heuristics act as conservative additional gates. Maintainers
    should review generated rules and refine with python rules as needed.
    """
    from app.db import Rule
    import re

    text = (record.get("eligibility_text") or "") + " " + (record.get("description") or "")
    text_lower = text.lower()

    conditions = []

    income_match = re.search(r"(?:income|earn).{0,40}?(?:rs\.?|₹)\s*([\d,]+)", text_lower)
    if income_match:
        try:
            val = int(income_match.group(1).replace(",", ""))
            conditions.append({"field": "annual_income", "op": "<=", "value": val})
        except ValueError:
            pass

    if ("women" in text_lower or "female" in text_lower or "girl" in text_lower) and "men" not in text_lower.replace("women", ""):
        conditions.append({"field": "gender", "op": "==", "value": "female"})

    age_match = re.search(r"(\d{1,2})\s*(?:years|yrs)?\s*(?:of age|and above|or above)", text_lower)
    if age_match:
        try:
            conditions.append({"field": "age", "op": ">=", "value": int(age_match.group(1))})
        except ValueError:
            pass

    if "farmer" in text_lower:
        conditions.append({"field": "occupation", "op": "==", "value": "farmer"})
    elif "street vendor" in text_lower:
        conditions.append({"field": "is_street_vendor", "op": "==", "value": True})

    db.query(Rule).filter(Rule.scheme_id == scheme.id, Rule.priority == 10).delete()

    if conditions:
        rule = Rule(
            scheme_id=scheme.id,
            rule_type="json",
            rule_body={"op": "and", "conditions": conditions},
            description=(
                f"Auto-extracted from scraped eligibility text for '{scheme.name}'. "
                f"Heuristic — review and refine with a python rule if needed."
            ),
            source_url=scheme.source_url,
            priority=10,
        )
        db.add(rule)
    else:
        rule = Rule(
            scheme_id=scheme.id,
            rule_type="json",
            rule_body={"op": "and", "conditions": [{"field": "_manual_review_required", "op": "==", "value": True}]},
            description=(
                f"No structured eligibility criteria could be extracted for "
                f"'{scheme.name}'. Needs manual rule authoring."
            ),
            source_url=scheme.source_url,
            priority=10,
        )
        db.add(rule)
