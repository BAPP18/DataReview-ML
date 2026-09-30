"""Validate the local environment before running the app or notebooks.

Usage: python validate_env.py
Exits non-zero when something required is missing.
"""

import importlib.metadata as md
import os
import sys

REQUIRED = ["pandas", "numpy", "scikit-learn", "streamlit", "pyyaml",
            "openpyxl", "recordlinkage"]
OPTIONAL = ["matplotlib", "seaborn", "plotly", "kaggle", "psycopg2-binary"]

DATA_FILES = [
    "data/processed/master_projects.csv",
    "data/processed/entity_pairs.csv",
    "data/processed/pairwise_features.csv",
    "data/processed/ai_reviewer_output.csv",
    "data/processed/canonical_source_records.csv",
    "models/entity_matcher_v1.joblib",
]


def main():
    ok = True
    print(f"Python: {sys.version.split()[0]} (need >= 3.10)")
    if sys.version_info < (3, 10):
        ok = False
    for pkg in REQUIRED:
        try:
            print(f"  [ok] {pkg} {md.version(pkg)}")
        except md.PackageNotFoundError:
            print(f"  [MISSING] {pkg}  -> pip install {pkg}")
            ok = False
    for pkg in OPTIONAL:
        try:
            print(f"  [ok, optional] {pkg} {md.version(pkg)}")
        except md.PackageNotFoundError:
            print(f"  [--, optional] {pkg} not installed")
    for path in DATA_FILES:
        here = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
        print(f"  [{'ok' if os.path.exists(here) else 'MISSING'}] {path}")
        ok = ok and os.path.exists(here)
    print("Environment ready." if ok else "Environment INCOMPLETE — fix items above.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
