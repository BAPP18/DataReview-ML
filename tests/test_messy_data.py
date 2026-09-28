"""Unit + data tests for real messy public datasets (NAB, DeepMatcher, HoloClean)."""

import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.matching.real_data import (
    dm_pair_features,
    fit_title_tfidf,
    flight_conflicts,
    hospital_constraint_violations,
    load_deepmatcher_pairs,
    load_nab_series,
    load_nab_windows,
    repair_by_group_mode,
)

RAW = "D:/data viewer/data/raw"


@unittest.skipUnless(os.path.exists(RAW + "/nab"), "NAB data not downloaded")
class TestNAB(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nab_dir = RAW + "/nab"
        cls.windows = load_nab_windows(cls.nab_dir)

    def test_windows_cover_expected_series(self):
        self.assertIn("realKnownCause/machine_temperature_system_failure.csv", self.windows)
        self.assertIn("realKnownCause/nyc_taxi.csv", self.windows)
        self.assertEqual(len(self.windows), 58)

    def test_point_labels_match_windows(self):
        df = load_nab_series(self.nab_dir, "realKnownCause/nyc_taxi.csv", self.windows)
        self.assertIn("label", df.columns)
        self.assertEqual(int(df["label"].sum()), 1035)
        self.assertTrue(set(df["label"].unique()).issubset({0, 1}))

    def test_series_sorted_by_time(self):
        df = load_nab_series(self.nab_dir, "realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv", self.windows)
        self.assertTrue(df["timestamp"].is_monotonic_increasing)


@unittest.skipUnless(os.path.exists(RAW + "/deepmatcher"), "DeepMatcher data not downloaded")
class TestDirtyER(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = load_deepmatcher_pairs(RAW + "/deepmatcher", "dirty_walmart_amazon", "train")
        cls.test = load_deepmatcher_pairs(RAW + "/deepmatcher", "dirty_walmart_amazon", "test")
        titles = pd.concat([cls.train["left_title"], cls.train["right_title"]])
        cls.vec = fit_title_tfidf(titles)

    def test_labels_valid(self):
        self.assertTrue(self.test["label"].isin([0, 1]).all())

    def test_dirty_has_more_missing_than_clean(self):
        clean = load_deepmatcher_pairs(RAW + "/deepmatcher", "walmart_amazon", "test")
        self.assertGreater(int(self.test.isnull().sum().sum()), int(clean.isnull().sum().sum()))

    def test_tfidf_features_shape_and_range(self):
        feats = dm_pair_features(self.test.head(50), self.vec)
        self.assertEqual(len(feats), 50)
        self.assertTrue(((feats["title_tfidf_cosine"] >= 0) & (feats["title_tfidf_cosine"] <= 1)).all())
        self.assertFalse(feats.isnull().any().any())


@unittest.skipUnless(os.path.exists(RAW + "/holoclean"), "HoloClean data not downloaded")
class TestHospitalFlights(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hosp = pd.read_csv(RAW + "/holoclean/hospital.csv", low_memory=False)
        cls.flight = pd.read_csv(RAW + "/holoclean/flight.csv", low_memory=False)

    def test_hospital_has_missing_and_violations(self):
        self.assertGreater(int(self.hosp.isnull().sum().sum()), 0)
        res = hospital_constraint_violations(self.hosp)
        self.assertGreater(res["violating_entities"], 0)

    def test_mode_repair_reduces_missing(self):
        before = int(self.hosp["ZipCode"].isna().sum())
        fixed = repair_by_group_mode(self.hosp, "HospitalName", "ZipCode")
        self.assertLessEqual(int(fixed["ZipCode"].isna().sum()), before)

    def test_flight_conflicts_exist(self):
        res = flight_conflicts(self.flight)
        self.assertGreater(res["flights_tracked"], 0)
        self.assertGreater(res["conflicting_scheduled"] + res["conflicting_actual"], 0)


if __name__ == "__main__":
    unittest.main()
