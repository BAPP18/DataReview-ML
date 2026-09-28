"""Smoke tests for the Streamlit reviewer workbench."""

import ast
import os
import py_compile
import unittest

import yaml

APP = "D:/data viewer/app/dashboard.py"
CONFIG = "D:/data viewer/config/models.yaml"


def _luminance(hex_color):
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (0, 2, 4))

    def lin(c):
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def _contrast(fg, bg):
    l1, l2 = _luminance(fg), _luminance(bg)
    lighter, darker = max(l1, l2), min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def _palettes_from_source():
    with open(APP, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "PALETTES":
                    return ast.literal_eval(node.value)
    raise AssertionError("PALETTES not found in dashboard.py")


class TestApp(unittest.TestCase):
    def test_dashboard_compiles(self):
        py_compile.compile(APP, doraise=True)

    def test_config_thresholds_valid(self):
        with open(CONFIG) as f:
            cfg = yaml.safe_load(f)
        hi = cfg["thresholds"]["auto_match"]
        lo = cfg["thresholds"]["manual_review_low"]
        self.assertGreater(hi, lo)
        self.assertLessEqual(hi, 1.0)
        self.assertGreaterEqual(lo, 0.0)
        self.assertTrue(len(cfg["features"]) > 0)

    def test_routing_logic_matches_config(self):
        with open(CONFIG) as f:
            cfg = yaml.safe_load(f)
        hi, lo = cfg["thresholds"]["auto_match"], cfg["thresholds"]["manual_review_low"]

        def route(p):
            if p >= hi:
                return "AUTO_MATCH"
            if p < lo:
                return "NON_MATCH"
            return "MANUAL_REVIEW"

        self.assertEqual(route(0.95), "AUTO_MATCH")
        self.assertEqual(route(0.50), "MANUAL_REVIEW")
        self.assertEqual(route(0.10), "NON_MATCH")

    def test_required_data_files_exist(self):
        base = "D:/data viewer/data/processed"
        for name in ["master_projects.csv", "entity_pairs.csv", "pairwise_features.csv",
                     "ai_reviewer_output.csv", "canonical_source_records.csv",
                     "nab_labeled_eval.csv", "dirty_er_ml.csv"]:
            self.assertTrue(os.path.exists(os.path.join(base, name)), f"missing {name}")

    def test_theme_text_contrast_both_modes(self):
        # Body text must be dark-on-light in Light mode and light-on-dark in Dark mode.
        palettes = _palettes_from_source()
        self.assertIn("Light", palettes)
        self.assertIn("Dark", palettes)
        self.assertGreater(_contrast(palettes["Light"]["ink"], palettes["Light"]["card_bg"]), 7.0)
        self.assertGreater(_contrast(palettes["Dark"]["ink"], palettes["Dark"]["card_bg"]), 7.0)

    def test_pill_contrast_both_modes(self):
        # Status pills must stay readable in both modes (min 3:1 for large/bold UI text).
        palettes = _palettes_from_source()
        for mode, pal in palettes.items():
            for name, (bg, fg) in pal["pill"].items():
                self.assertGreater(
                    _contrast(fg, bg), 3.0, f"pill {name} unreadable in {mode} mode")

    def test_no_hardcoded_slop_gradients(self):
        with open(APP, encoding="utf-8") as f:
            src = f.read().lower()
        self.assertNotIn("linear-gradient", src)
        self.assertNotIn("radial-gradient", src)

    def test_native_widgets_pinned_to_palette(self):
        # Sidebar buttons, toggle, inputs and dropdowns must not rely on
        # Streamlit's theme.base alone (it can lag behind the in-app toggle).
        with open(APP, encoding="utf-8") as f:
            src = f.read()
        for selector in ["stBaseButton-secondary", "stBaseButton-primary",
                         "stSegmentedControl", "stTextInput",
                         'data-baseweb="select"', 'data-baseweb="menu"']:
            self.assertIn(selector, src, f"missing widget override: {selector}")


if __name__ == "__main__":
    unittest.main()
