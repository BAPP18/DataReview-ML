"""Reviewer KPIs, data-quality scores, and queue priority.

All functions are pure (inputs in, numbers out) so they are unit-testable
and reusable in notebooks, the app, and reports.
"""

from datetime import datetime

SEVERITY_SCORE = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, None: 0}


def priority_score(severity, uncertainty, age_days, weights):
    """Priority = w_sev*severity + w_unc*uncertainty + w_age*age_days."""
    sev = SEVERITY_SCORE.get(severity, 0)
    unc = max(0.0, min(1.0, float(uncertainty or 0.0)))
    age = max(0.0, float(age_days or 0.0))
    return round(weights.get("severity", 10.0) * sev
                 + weights.get("uncertainty", 5.0) * unc
                 + weights.get("age_days", 0.5) * age, 2)


def reviewer_kpis(cases, decisions, qa_reviews, sla_hours, today=None):
    """Operational KPIs over cases (dicts), decisions and QA reviews (lists)."""
    today = today or datetime.now().date().isoformat()
    open_statuses = ("NEW", "IN_REVIEW", "WAITING_CLARIFICATION", "REOPENED")
    open_cases = [c for c in cases if c.get("status") in open_statuses]
    resolved = [c for c in cases if c.get("status") in ("RESOLVED", "SYNCED", "REJECTED")]
    resolved_today = [c for c in resolved if (c.get("resolved_at") or "")[:10] == today]
    handling = []
    for c in resolved:
        try:
            start = datetime.fromisoformat(c["created_at"])
            end = datetime.fromisoformat(c["resolved_at"])
            handling.append((end - start).total_seconds() / 3600.0)
        except (KeyError, TypeError, ValueError):
            continue
    clar = sum(1 for d in decisions if d.get("decision") == "CLARIFICATION")
    rework = sum(1 for c in cases if c.get("status") == "REOPENED")
    breached = sum(1 for c in open_cases
                   if _age_hours(c, today) > sla_hours)
    qa = qa_metrics_simple(qa_reviews)
    reasons = {}
    for d in decisions:
        code = d.get("reason_code", "OTHER")
        reasons[code] = reasons.get(code, 0) + 1
    return {
        "open_cases": len(open_cases),
        "resolved_total": len(resolved),
        "resolved_today": len(resolved_today),
        "avg_resolution_hours": round(sum(handling) / len(handling), 2) if handling else None,
        "clarification_rate": round(clar / len(decisions), 3) if decisions else None,
        "rework_reopened": rework,
        "sla_breaches": breached,
        "qa_agreement_rate": qa["agreement_rate"],
        "qa_sampled": qa["qa_sampled"],
        "reason_distribution": reasons,
    }


def _age_hours(case, today_iso):
    try:
        start = datetime.fromisoformat(case["created_at"])
        end = datetime.fromisoformat(case["resolved_at"]) if case.get("resolved_at") \
            else datetime.fromisoformat(today_iso + "T23:59:59")
        return max(0.0, (end - start).total_seconds() / 3600.0)
    except (KeyError, TypeError, ValueError):
        return 0.0


def qa_metrics_simple(qa_reviews):
    total = len(qa_reviews)
    agree = sum(1 for r in qa_reviews if r.get("verdict") == "AGREE")
    return {"qa_sampled": total,
            "agreement_rate": round(agree / total, 3) if total else None}


def completeness(records, fields):
    """1 - missing/total over the given fields (records = list of dicts)."""
    total = len(records) * max(len(fields), 1)
    if not total or not records:
        return None
    missing = sum(1 for r in records for f in fields
                  if r.get(f) is None or str(r.get(f)).strip() in ("", "nan", "None", "—"))
    return round(1 - missing / total, 4)


def conflict_rate(comparisons):
    """Share of field comparisons with CONFLICT verdict."""
    if not comparisons:
        return None
    bad = sum(1 for c in comparisons if c.get("verdict") == "CONFLICT")
    return round(bad / len(comparisons), 4)


def duplicate_rate(n_rows, n_unique_keys):
    """1 - unique/total over a key column."""
    if not n_rows:
        return None
    return round(1 - n_unique_keys / n_rows, 4)
