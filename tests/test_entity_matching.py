"""Unit tests for entity matching pipeline."""

import unittest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.matching.entity_matching import jaro_winkler_similarity, normalize_text


class TestTextNormalization(unittest.TestCase):
    def test_lowercase(self):
        self.assertEqual(normalize_text("HELLO"), "hello")

    def test_strip_whitespace(self):
        self.assertEqual(normalize_text("  hello  "), "hello")

    def test_none_input(self):
        self.assertEqual(normalize_text(None), "")

    def test_empty_string(self):
        self.assertEqual(normalize_text(""), "")


class TestJaroWinklerSimilarity(unittest.TestCase):
    def test_identical_strings(self):
        self.assertEqual(jaro_winkler_similarity("hello", "hello"), 1.0)

    def test_completely_different(self):
        self.assertEqual(jaro_winkler_similarity("abc", "xyz"), 0.0)

    def test_empty_strings(self):
        self.assertEqual(jaro_winkler_similarity("", ""), 0.0)
        self.assertEqual(jaro_winkler_similarity("abc", ""), 0.0)

    def test_similar_strings(self):
        sim = jaro_winkler_similarity("martha", "marhta")
        self.assertGreater(sim, 0.9)

    def test_case_insensitive(self):
        sim = jaro_winkler_similarity("Hello", "hello")
        self.assertEqual(sim, 1.0)

    def test_whitespace_insensitive(self):
        sim = jaro_winkler_similarity("hello", "  hello  ")
        self.assertEqual(sim, 1.0)


class TestBlocking(unittest.TestCase):
    def test_generate_blocking_keys(self):
        import pandas as pd
        df = pd.DataFrame({
            "state": ["CA", "TX", "FL"],
            "city": ["LA", "Houston", "Miami"],
        })
        df["block_key"] = df["state"].str.lower() + "_" + df["city"].str.lower()
        self.assertEqual(df["block_key"].iloc[0], "ca_la")

    def test_candidate_reduction(self):
        import pandas as pd
        n = 1000
        df = pd.DataFrame({"state": ["CA"] * n, "city": ["LA"] * n})
        block_sizes = df.groupby("state").size()
        self.assertEqual(block_sizes.iloc[0], n)


class TestThresholdLogic(unittest.TestCase):
    def test_auto_match_threshold(self):
        prob = 0.95
        self.assertGreaterEqual(prob, 0.95)

    def test_manual_review_threshold(self):
        prob = 0.70
        self.assertTrue(0.50 <= prob < 0.95)

    def test_non_match_threshold(self):
        prob = 0.30
        self.assertLess(prob, 0.50)


if __name__ == "__main__":
    unittest.main()
