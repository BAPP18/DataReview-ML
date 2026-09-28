"""Review case lifecycle + correction handoff.

Statuses: NEW -> IN_REVIEW -> {WAITING_CLARIFICATION | RESOLVED | REJECTED}
          WAITING_CLARIFICATION -> IN_REVIEW | RESOLVED
          RESOLVED -> REOPENED | SYNCED   (SYNCED = correction handed off as file)
          REJECTED -> REOPENED
          REOPENED -> IN_REVIEW

Cases persist as JSONL (one object per case_id == pair_id). SYNCED appends
approved corrections to a handoff CSV — the file-based equivalent of
"sync to CRM" (no fake API integration).
"""

import csv
import json
import os
from datetime import datetime

TERMINAL_RESOLUTIONS = ("RESOLVED", "REJECTED", "SYNCED")
ALLOWED = {
    "NEW": ("IN_REVIEW",),
    "IN_REVIEW": ("WAITING_CLARIFICATION", "RESOLVED", "REJECTED"),
    "WAITING_CLARIFICATION": ("IN_REVIEW", "RESOLVED"),
    "RESOLVED": ("REOPENED", "SYNCED"),
    "REJECTED": ("REOPENED",),
    "REOPENED": ("IN_REVIEW",),
    "SYNCED": (),
}


def _now_iso(now=None):
    if now is None:
        return datetime.now().isoformat(timespec="seconds")
    if isinstance(now, datetime):
        return now.isoformat(timespec="seconds")
    return str(now)


def blank_case(case_id, now=None):
    ts = _now_iso(now)
    return {"case_id": case_id, "status": "NEW", "assigned_to": None,
            "created_at": ts, "started_at": None, "resolved_at": None,
            "corrections": [], "history": [
                {"from": None, "to": "NEW", "actor": "system",
                 "note": "case opened", "at": ts}]}


def load_cases(path):
    if not os.path.exists(path):
        return {}
    cases = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                case = json.loads(line)
                cases[case["case_id"]] = case
    return cases


def save_cases(path, cases):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for case in cases.values():
            f.write(json.dumps(case) + "\n")


def get_or_create(cases, case_id, now=None):
    if case_id not in cases:
        cases[case_id] = blank_case(case_id, now)
    return cases[case_id]


def transition(case, to, actor="reviewer", note="", now=None):
    """Move a case; raises ValueError on illegal transitions."""
    frm = case["status"]
    if to not in ALLOWED.get(frm, ()):
        raise ValueError(f"illegal transition {frm} -> {to}")
    ts = _now_iso(now)
    case["status"] = to
    if to == "IN_REVIEW" and not case.get("started_at"):
        case["started_at"] = ts
    if to in TERMINAL_RESOLUTIONS and not case.get("resolved_at"):
        case["resolved_at"] = ts
    if to == "REOPENED":
        case["resolved_at"] = None
    case["history"].append({"from": frm, "to": to, "actor": actor, "note": note, "at": ts})
    return case


def propose_correction(case, field, old_value, new_value, authority_source, actor="reviewer", now=None):
    """Attach an approved correction proposal (values from authoritative source)."""
    case["corrections"].append({
        "field": field,
        "old_value": None if old_value is None else str(old_value),
        "new_value": None if new_value is None else str(new_value),
        "authority_source": authority_source,
        "proposed_by": actor,
        "at": _now_iso(now),
    })
    return case


def mark_synced(case, handoff_path, actor="reviewer", now=None):
    """Export approved corrections to the handoff CSV and close as SYNCED."""
    if case["status"] != "RESOLVED":
        raise ValueError("only RESOLVED cases can be synced")
    if not case.get("corrections"):
        raise ValueError("no approved corrections to sync")
    os.makedirs(os.path.dirname(handoff_path) or ".", exist_ok=True)
    exists = os.path.exists(handoff_path)
    with open(handoff_path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["case_id", "field", "old_value", "new_value",
                        "authority_source", "proposed_by", "synced_at"])
        ts = _now_iso(now)
        for c in case["corrections"]:
            w.writerow([case["case_id"], c["field"], c["old_value"], c["new_value"],
                        c["authority_source"], c["proposed_by"], ts])
    return transition(case, "SYNCED", actor, f"{len(case['corrections'])} corrections handed off", now)


def age_hours(case, now=None):
    """Hours since creation for open cases; total handling time if resolved."""
    end = case.get("resolved_at") if case.get("status") in TERMINAL_RESOLUTIONS else None
    start = datetime.fromisoformat(case["created_at"])
    stop = datetime.fromisoformat(end) if end else (now or datetime.now())
    if isinstance(stop, str):
        stop = datetime.fromisoformat(stop)
    return max(0.0, (stop - start).total_seconds() / 3600.0)


def is_sla_breach(case, sla_hours, now=None):
    return case["status"] not in TERMINAL_RESOLUTIONS and age_hours(case, now) > sla_hours
