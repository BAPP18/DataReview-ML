"""Tests for the reviewer workstation: evidence, truth, lifecycle, QA, KPIs."""

import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.matching.train_model import load_artifact, meta_path_for
from src.review import cases as case_store
from src.review import evidence as ev
from src.review import metrics as rmetrics
from src.review import qa as qa_mod

RULES = {
    "system_size_kw": {"kind": "numeric", "tolerance": 1.0, "review_above": 2.0},
    "customer_name": {"kind": "text", "match_threshold": 0.92},
    "status": {"kind": "category", "equivalents": [["Completed", "ACTIVE", "SIGNED"]]},
}
PRIORITY = {"signed_document": 1, "crm": 2, "erp": 3, "partner": 4}


class TestFieldComparison(unittest.TestCase):
    def test_numeric_match_within_tolerance(self):
        v, _ = ev.compare_numeric({"crm": "250", "partner": "250.5"}, 1.0, 2.0)
        self.assertEqual(v, "MATCH")

    def test_numeric_conflict_above_limit(self):
        v, detail = ev.compare_numeric({"crm": "250", "partner": "255"}, 1.0, 2.0)
        self.assertEqual(v, "CONFLICT")
        self.assertIn("5", detail)

    def test_numeric_grey_zone_review(self):
        v, _ = ev.compare_numeric({"crm": "250", "partner": "251.5"}, 1.0, 2.0)
        self.assertEqual(v, "REVIEW")

    def test_numeric_missing_single_source(self):
        v, _ = ev.compare_numeric({"crm": "250", "partner": None}, 1.0, 2.0)
        self.assertEqual(v, "MISSING")

    def test_text_case_only_is_match(self):
        v, _ = ev.compare_text({"crm": "ABC Solar LLC", "partner": "abc solar llc"}, 0.85)
        self.assertEqual(v, "MATCH")

    def test_text_format_variant(self):
        v, _ = ev.compare_text({"crm": "ABC Solar LLC", "partner": "ABC Solar L.L.C."}, 0.85)
        self.assertEqual(v, "FORMAT")

    def test_text_conflict(self):
        v, _ = ev.compare_text({"crm": "ABC Solar", "partner": "XYZ Energy"}, 0.85)
        self.assertEqual(v, "CONFLICT")

    def test_status_equivalents_match(self):
        v, _ = ev.compare_category({"crm": "Completed", "erp": "ACTIVE", "doc": "SIGNED"},
                                   [["Completed", "ACTIVE", "SIGNED"]])
        self.assertEqual(v, "MATCH")

    def test_compare_field_record_shape(self):
        rec = ev.compare_field("system_size_kw", {"crm": "250", "partner": "255"}, RULES)
        self.assertEqual(rec["verdict"], "CONFLICT")
        self.assertEqual(rec["severity"], "HIGH")
        self.assertEqual(rec["category"], "NUMERIC_CONFLICT")

    def test_authoritative_picks_signed_document(self):
        auth = ev.authoritative_value("system_size_kw",
                                      {"crm": "255", "partner": "255", "signed_document": "250"},
                                      PRIORITY)
        self.assertEqual(auth["source"], "signed_document")
        self.assertEqual(auth["value"], "250")


class TestLifecycle(unittest.TestCase):
    def test_happy_path_new_to_resolved(self):
        c = case_store.blank_case("P1")
        case_store.transition(c, "IN_REVIEW", "r1", "")
        case_store.transition(c, "RESOLVED", "r1", "")
        self.assertEqual(c["status"], "RESOLVED")
        self.assertIsNotNone(c["started_at"])
        self.assertIsNotNone(c["resolved_at"])
        self.assertEqual(len(c["history"]), 3)

    def test_illegal_transition_rejected(self):
        c = case_store.blank_case("P1")
        with self.assertRaises(ValueError):
            case_store.transition(c, "RESOLVED", "r1", "")

    def test_clarification_loop_and_reopen(self):
        c = case_store.blank_case("P1")
        case_store.transition(c, "IN_REVIEW", "r1", "")
        case_store.transition(c, "WAITING_CLARIFICATION", "r1", "")
        case_store.transition(c, "IN_REVIEW", "r1", "")
        case_store.transition(c, "REJECTED", "r1", "")
        case_store.transition(c, "REOPENED", "r1", "")
        self.assertEqual(c["status"], "REOPENED")
        self.assertIsNone(c["resolved_at"])

    def test_sync_requires_corrections(self):
        c = case_store.blank_case("P1")
        case_store.transition(c, "IN_REVIEW", "r1", "")
        case_store.transition(c, "RESOLVED", "r1", "")
        with self.assertRaises(ValueError):
            case_store.mark_synced(c, "nowhere.csv", "r1")
        case_store.propose_correction(c, "system_size_kw", "255", "250", "signed_document", "r1")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "handoff.csv")
            case_store.mark_synced(c, path, "r1")
            self.assertEqual(c["status"], "SYNCED")
            self.assertTrue(os.path.exists(path))

    def test_sla_breach(self):
        old = datetime.now() - timedelta(hours=100)
        c = case_store.blank_case("P1", old)
        self.assertTrue(case_store.is_sla_breach(c, 72))
        self.assertFalse(case_store.is_sla_breach(c, 720))

    def test_persistence_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cases.jsonl")
            cases = {"P1": case_store.blank_case("P1")}
            case_store.save_cases(path, cases)
            loaded = case_store.load_cases(path)
            self.assertEqual(loaded["P1"]["status"], "NEW")


class TestQA(unittest.TestCase):
    def test_sampling_deterministic(self):
        a = [c for c in [f"PAIR-{i:06d}" for i in range(200)] if qa_mod.is_qa_sampled(c, 0.10)]
        b = [c for c in [f"PAIR-{i:06d}" for i in range(200)] if qa_mod.is_qa_sampled(c, 0.10)]
        self.assertEqual(a, b)
        self.assertTrue(5 <= len(a) <= 40)

    def test_queue_excludes_done(self):
        pending = qa_mod.qa_queue(["A", "B", "C"], ["A"], 1.0)
        self.assertEqual(pending, ["B", "C"])

    def test_record_validates_verdict(self):
        with self.assertRaises(ValueError):
            qa_mod.record_qa("/tmp/x.jsonl", "A", "qa", "MAYBE")

    def test_metrics(self):
        m = qa_mod.qa_metrics([{"verdict": "AGREE"}] * 9 + [{"verdict": "DISAGREE"}])
        self.assertEqual(m["agreement_rate"], 0.9)
        self.assertEqual(qa_mod.qa_metrics([])["agreement_rate"], None)


class TestMetrics(unittest.TestCase):
    def test_priority_score(self):
        w = {"severity": 10.0, "uncertainty": 5.0, "age_days": 0.5}
        high = rmetrics.priority_score("HIGH", 0.4, 2.0, w)
        low = rmetrics.priority_score("LOW", 0.1, 0.0, w)
        self.assertGreater(high, low)

    def test_reviewer_kpis(self):
        now = datetime.now()
        cases = [
            {"case_id": "A", "status": "RESOLVED", "created_at": (now - timedelta(hours=5)).isoformat(),
             "resolved_at": now.isoformat()},
            {"case_id": "B", "status": "IN_REVIEW", "created_at": (now - timedelta(hours=100)).isoformat(),
             "resolved_at": None},
        ]
        decisions = [{"decision": "MATCH", "reason_code": "SOURCE_CONFLICT"},
                     {"decision": "CLARIFICATION", "reason_code": "MISSING_INFORMATION"}]
        k = rmetrics.reviewer_kpis(cases, decisions, [{"verdict": "AGREE"}], 72)
        self.assertEqual(k["open_cases"], 1)
        self.assertEqual(k["resolved_total"], 1)
        self.assertEqual(k["sla_breaches"], 1)
        self.assertEqual(k["clarification_rate"], 0.5)
        self.assertEqual(k["reason_distribution"]["SOURCE_CONFLICT"], 1)

    def test_completeness_and_conflict(self):
        recs = [{"a": "x", "b": None}, {"a": "y", "b": "z"}]
        self.assertEqual(rmetrics.completeness(recs, ["a", "b"]), 0.75)
        comps = [{"verdict": "MATCH"}, {"verdict": "CONFLICT"}]
        self.assertEqual(rmetrics.conflict_rate(comps), 0.5)
        self.assertEqual(rmetrics.duplicate_rate(100, 97), 0.03)


class TestModelArtifact(unittest.TestCase):
    def test_artifact_exists_and_loads(self):
        path = "D:/data viewer/models/entity_matcher_v1.joblib"
        if not os.path.exists(path):
            self.skipTest("artifact not built")
        model, meta = load_artifact(path)
        self.assertEqual(meta.get("model_version"), "v1.0")
        self.assertEqual(len(meta.get("features", [])), 7)
        import numpy as np
        proba = model.predict_proba(np.zeros((1, 7)))
        self.assertEqual(proba.shape, (1, 2))

    def test_meta_path_rule(self):
        self.assertTrue(meta_path_for("m/entity_matcher_v1.joblib").endswith("entity_matcher_v1.meta.json"))


if __name__ == "__main__":
    unittest.main()
