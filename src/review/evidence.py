"""Field-level discrepancy engine (deterministic, no ML).

For one canonical project, compare the same logical field across sources
(CRM / ERP / Partner / signed Document) and emit per-field verdicts:

    MATCH   — values agree (within tolerance for numerics)
    FORMAT  — same meaning, different formatting/casing
    REVIEW  — differs inside the grey zone, needs a human look
    CONFLICT— differs beyond the review threshold
    MISSING — fewer than two sources report the field

Plus severity (HIGH/MEDIUM/LOW) and the authoritative source/value
according to ``review.source_priority`` in config/models.yaml.
"""

import re

VERDICTS = ("MATCH", "FORMAT", "REVIEW", "CONFLICT", "MISSING")
SEVERITY_OF = {"CONFLICT": "HIGH", "REVIEW": "MEDIUM", "FORMAT": "MEDIUM",
               "MISSING": "MEDIUM", "MATCH": "LOW"}
CATEGORY_OF = {"system_size_kw": "NUMERIC_CONFLICT", "status": "STATUS_CONFLICT",
               "customer_name": "ENTITY_MISMATCH", "installer": "POTENTIAL_DUPLICATE",
               "address": "FORMATTING_DIFFERENCE"}

_WS_RE = re.compile(r"\s+")


def normalize_text(value):
    if value is None:
        return None
    text = _WS_RE.sub(" ", str(value)).strip().lower()
    return text or None


def _norm_number(value):
    try:
        if value is None or (isinstance(value, float) and value != value):
            return None
        return float(str(value).replace(",", "").strip())
    except (ValueError, TypeError):
        return None


def compare_numeric(values, tolerance, review_above):
    """values: {source: raw}. Returns (verdict, detail)."""
    nums = {s: _norm_number(v) for s, v in values.items()}
    nums = {s: n for s, n in nums.items() if n is not None}
    if len(nums) < 2:
        return "MISSING", "fewer than two sources report this field"
    spread = max(nums.values()) - min(nums.values())
    if spread <= tolerance:
        return "MATCH", f"max spread {spread:.2f} within tolerance {tolerance}"
    if spread > review_above:
        return "CONFLICT", f"max spread {spread:.2f} exceeds review limit {review_above}"
    return "REVIEW", f"max spread {spread:.2f} inside grey zone ({tolerance}, {review_above}]"


def compare_text(values, match_threshold):
    """values: {source: raw}. Returns (verdict, detail)."""
    texts = {s: normalize_text(v) for s, v in values.items()}
    texts = {s: t for s, t in texts.items() if t is not None}
    if len(texts) < 2:
        return "MISSING", "fewer than two sources report this field"
    uniq = set(texts.values())
    if len(uniq) == 1:
        return "MATCH", "identical after normalization"
    # Same alphanumeric core, different punctuation/casing/spacing?
    cores = {re.sub(r"[^a-z0-9]", "", t) for t in uniq}
    if len(cores) == 1:
        return "FORMAT", "same content, different formatting"
    return "CONFLICT", f"{len(uniq)} distinct normalized values"


def compare_category(values, equivalents):
    """values: {source: raw}. equivalents: list of same-meaning groups."""
    texts = {s: normalize_text(v) for s, v in values.items()}
    texts = {s: t for s, t in texts.items() if t is not None}
    if len(texts) < 2:
        return "MISSING", "fewer than two sources report this field"

    def group_of(text):
        for i, group in enumerate(equivalents):
            if text in {normalize_text(g) for g in group}:
                return i
        return f"other:{text}"

    groups = {group_of(t) for t in texts.values()}
    if len(groups) == 1:
        return "MATCH", "same status group across sources"
    return "CONFLICT", f"{len(groups)} distinct status groups"


def compare_field(field, values_by_source, field_rules):
    """Full per-field record: verdict, severity, category, detail."""
    rule = (field_rules or {}).get(field, {"kind": "text", "match_threshold": 0.85})
    kind = rule.get("kind", "text")
    if kind == "numeric":
        verdict, detail = compare_numeric(values_by_source, rule.get("tolerance", 1.0),
                                          rule.get("review_above", 2.0))
    elif kind == "category":
        verdict, detail = compare_category(values_by_source, rule.get("equivalents", []))
    else:
        verdict, detail = compare_text(values_by_source, rule.get("match_threshold", 0.85))
    return {
        "field": field,
        "verdict": verdict,
        "severity": SEVERITY_OF[verdict],
        "category": CATEGORY_OF.get(field, "OTHER"),
        "values": {s: (None if v is None else str(v)) for s, v in values_by_source.items()},
        "detail": detail,
    }


def authoritative_value(field, values_by_source, source_priority):
    """Pick the winning source: lowest priority number among reporting sources."""
    present = {s: v for s, v in values_by_source.items()
               if v is not None and str(v).strip() not in ("", "—", "nan", "None")}
    if not present:
        return {"source": None, "value": None, "reason": "no source reports this field"}
    ranked = sorted(source_priority.items(), key=lambda kv: kv[1])
    for source, _ in ranked:
        if source in present:
            return {"source": source, "value": str(present[source]),
                    "reason": f"{source} outranks others by source priority"}
    first = next(iter(present.items()))
    return {"source": first[0], "value": str(first[1]), "reason": "only reporting source"}


def recommended_action(comparison, authority):
    """Reviewer-facing next step for one field comparison."""
    if comparison["verdict"] == "MATCH":
        return "No action — sources agree."
    if comparison["verdict"] == "MISSING":
        return "Request the missing value from the non-reporting source."
    auth = authority["source"] or "n/a"
    return (f"{comparison['verdict']}: authoritative source is {auth} "
            f"({authority['value']}); update derived values or request clarification.")
