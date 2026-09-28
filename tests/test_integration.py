"""Integration tests for end-to-end pipeline."""

import unittest
import pandas as pd
import os
import sys
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestEndToEndPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output_dir = "D:/data viewer/data/processed"
        cls.report_dir = "D:/data viewer/reports"

    def test_master_exists(self):
        self.assertTrue(os.path.exists(f"{self.output_dir}/master_projects.csv"))

    def test_pairs_exist(self):
        self.assertTrue(os.path.exists(f"{self.output_dir}/entity_pairs.csv"))

    def test_features_exist(self):
        self.assertTrue(os.path.exists(f"{self.output_dir}/pairwise_features.csv"))

    def test_ai_output_exists(self):
        self.assertTrue(os.path.exists(f"{self.output_dir}/ai_reviewer_output.csv"))

    def test_canonical_records_exist(self):
        self.assertTrue(os.path.exists(f"{self.output_dir}/canonical_source_records.csv"))

    def test_reports_exist(self):
        expected_reports = [
            "dataset_suitability.md",
            "data_dictionary.md",
            "data_lineage.md",
            "phase5_6_entity_matching.txt",
            "phase7_anomaly_detection.txt",
            "phase8_9_ai_reviewer_evaluation.txt",
            "phase10_11_postgresql_ingestion.txt",
        ]
        for report in expected_reports:
            self.assertTrue(
                os.path.exists(f"{self.report_dir}/{report}"),
                f"Missing report: {report}"
            )

    def test_ingestion_pipeline_output(self):
        canonical = pd.read_csv(f"{self.output_dir}/canonical_source_records.csv")
        self.assertIn("source_system", canonical.columns)
        self.assertIn("source_record_id", canonical.columns)
        self.assertIn("master_project_id", canonical.columns)

    def test_corruption_log_exists(self):
        self.assertTrue(os.path.exists(f"{self.output_dir}/corruption_log.json"))

    def test_ground_truth_consistency(self):
        master = pd.read_csv(f"{self.output_dir}/master_projects.csv")
        pairs = pd.read_csv(f"{self.output_dir}/entity_pairs.csv")

        master_ids = set(master["project_id"])
        pair_master_ids = set(pairs["master_id_a"]).union(set(pairs["master_id_b"]))

        self.assertTrue(pair_master_ids.issubset(master_ids))


if __name__ == "__main__":
    unittest.main()
