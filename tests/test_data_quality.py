"""Data quality tests."""

import unittest
import pandas as pd
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestDataQuality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.master = pd.read_csv("D:/data viewer/data/processed/master_projects.csv")
        cls.pairs = pd.read_csv("D:/data viewer/data/processed/entity_pairs.csv")
        cls.features = pd.read_csv("D:/data viewer/data/processed/pairwise_features.csv")

    def test_master_unique_project_ids(self):
        self.assertEqual(len(self.master), len(self.master["project_id"].unique()))

    def test_master_no_null_project_ids(self):
        self.assertFalse(self.master["project_id"].isnull().any())

    def test_master_positive_capacity(self):
        self.assertTrue((self.master["system_size_kw"] > 0).all())

    def test_pairs_valid_labels(self):
        valid_labels = [0, 1]
        self.assertTrue(self.pairs["label"].isin(valid_labels).all())

    def test_pairs_master_ids_exist(self):
        master_ids = set(self.master["project_id"])
        pair_ids = set(self.pairs["master_id_a"]).union(set(self.pairs["master_id_b"]))
        self.assertTrue(pair_ids.issubset(master_ids))

    def test_features_no_nan(self):
        self.assertFalse(self.features.isnull().any().any())

    def test_features_similarity_range(self):
        for col in ["customer_similarity", "installer_similarity", "address_similarity"]:
            self.assertTrue((self.features[col] >= 0).all())
            self.assertTrue((self.features[col] <= 1).all())

    def test_no_leakage_positives(self):
        pos = self.pairs[self.pairs["label"] == 1]
        self.assertTrue((pos["master_id_a"] == pos["master_id_b"]).all())

    def test_no_leakage_negatives(self):
        neg = self.pairs[self.pairs["label"] == 0]
        self.assertTrue((neg["master_id_a"] != neg["master_id_b"]).all())

    def test_class_balance_reported(self):
        counts = self.pairs["label"].value_counts()
        self.assertIn(1, counts)
        self.assertIn(0, counts)


if __name__ == "__main__":
    unittest.main()
