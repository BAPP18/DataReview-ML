"""QA second-review: deterministic sampling + agreement metrics.

A fixed fraction of RESOLVED cases is sampled for second review using a
stable hash of the case id (reproducible, no random draws drifting between
runs). A QA reviewer records AGREE/DISAGREE per sampled case.
"""

import hashlib
import json
import os
from datetime import datetime


def _now_iso(now=None):
    if now is None:
        return datetime.now().isoformat(timespec="seconds")
    if isinstance(now, datetime):
        return now.isoformat(timespec="seconds")
    return str(now)


def is_qa_sampled(case_id, fraction):
    """Deterministic sampler: sha256(case_id) % 100 < fraction*100."""
    if fraction <= 0:
        return False
    if fraction >= 1:
        return True
    digest = hashlib.sha256(str(case_id).encode()).hexdigest()
    return int(digest, 16) % 100 < round(fraction * 100)


def qa_queue(resolved_case_ids, done_case_ids, fraction):
    """Cases eligible for QA: sampled and not yet reviewed."""
    done = set(done_case_ids)
    return [c for c in resolved_case_ids
            if c not in done and is_qa_sampled(c, fraction)]


def load_qa(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def record_qa(path, case_id, qa_reviewer, verdict, note="", model_version="v1.0", now=None):
    """verdict: AGREE or DISAGREE."""
    if verdict not in ("AGREE", "DISAGREE"):
        raise ValueError("verdict must be AGREE or DISAGREE")
    entry = {"case_id": case_id, "qa_reviewer": qa_reviewer, "verdict": verdict,
             "note": note, "model_version": model_version, "at": _now_iso(now)}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def qa_metrics(qa_reviews):
    """Agreement / disagreement / rework-signal rates."""
    total = len(qa_reviews)
    agree = sum(1 for r in qa_reviews if r.get("verdict") == "AGREE")
    disagree = total - agree
    return {
        "qa_sampled": total,
        "agreements": agree,
        "disagreements": disagree,
        "agreement_rate": round(agree / total, 3) if total else None,
        "rework_signal_rate": round(disagree / total, 3) if total else None,
    }
