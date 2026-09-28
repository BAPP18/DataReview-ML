"""Reusable helpers for real messy public datasets.

- NAB (labeled time-series anomalies)
- DeepMatcher dirty ER pairs (TF-IDF + similarity features)
- HoloClean Hospital (constraint violations) & Flights (cross-source conflicts)

Notebook cells are intentionally self-contained for Google Colab;
this module mirrors the same logic for reuse and unit testing.
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_RAW = "D:/data viewer/data/raw"


# ---------------- NAB ----------------

def load_nab_windows(nab_dir=None):
    """Load NAB anomaly windows mapping: {series_key: [[start, end], ...]}."""
    nab_dir = nab_dir or os.path.join(BASE_RAW, "nab")
    with open(os.path.join(nab_dir, "labels", "combined_windows.json")) as f:
        return json.load(f)


def load_nab_series(nab_dir, key, windows=None):
    """Load one NAB series and map anomaly windows to point labels."""
    nab_dir = nab_dir or os.path.join(BASE_RAW, "nab")
    windows = windows if windows is not None else load_nab_windows(nab_dir)
    df = pd.read_csv(os.path.join(nab_dir, key), parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["label"] = 0
    for start, end in windows.get(key, []):
        s, e = pd.to_datetime(start), pd.to_datetime(end)
        df.loc[(df["timestamp"] >= s) & (df["timestamp"] <= e), "label"] = 1
    return df


# ---------------- DeepMatcher dirty ER ----------------

DM_TEXT_COLS = ["title", "category", "brand", "modelno"]


def load_deepmatcher_pairs(base_dir=None, dataset="dirty_walmart_amazon", split="test"):
    """Load labeled DeepMatcher-format pairs (id, label, left_*, right_*)."""
    base_dir = base_dir or os.path.join(BASE_RAW, "deepmatcher")
    path = os.path.join(base_dir, dataset, f"{split}.csv")
    return pd.read_csv(path)


def fit_title_tfidf(titles, max_features=5000):
    """Fit a TF-IDF vectorizer on lowercased product titles."""
    titles = pd.Series(titles).fillna("").astype(str).str.lower()
    return TfidfVectorizer(max_features=max_features, ngram_range=(1, 2)).fit(titles)


def dm_pair_features(df, vectorizer):
    """Build dirty-ER pairwise features incl. TF-IDF cosine similarity."""
    lt = df["left_title"].fillna("").astype(str).str.lower()
    rt = df["right_title"].fillna("").astype(str).str.lower()
    cos = [
        float(cosine_similarity(vectorizer.transform([a]), vectorizer.transform([b]))[0, 0])
        for a, b in zip(lt, rt)
    ]
    lb = df["left_brand"].fillna("").astype(str).str.lower()
    rb = df["right_brand"].fillna("").astype(str).str.lower()
    lc = df["left_category"].fillna("").astype(str).str.lower()
    rc = df["right_category"].fillna("").astype(str).str.lower()
    pa = pd.to_numeric(df["left_price"], errors="coerce")
    pb = pd.to_numeric(df["right_price"], errors="coerce")
    return pd.DataFrame({
        "title_tfidf_cosine": cos,
        "title_exact": (lt == rt).astype(int),
        "brand_match": (lb == rb).astype(int),
        "category_match": (lc == rc).astype(int),
        "price_diff": (pa - pb).abs().fillna(0),
    })


# ---------------- Hospital / Flights ----------------

def hospital_constraint_violations(hosp_df, name_col="HospitalName", target_col="ZipCode"):
    """Count entities violating name->attribute functional dependency."""
    grp = hosp_df.groupby(name_col)[target_col].nunique(dropna=True)
    violators = grp[grp > 1].index.tolist()
    return {"violating_entities": len(violators), "total_entities": int(hosp_df[name_col].nunique()), "violators": violators[:10]}


def repair_by_group_mode(df, group_col, target_col):
    """Repair missing target values with the mode inside each group."""
    out = df.copy()
    mode_map = out.groupby(group_col)[target_col].agg(
        lambda s: s.mode().iloc[0] if s.notna().any() else np.nan
    )
    mask = out[target_col].isna()
    out.loc[mask, target_col] = out.loc[mask, group_col].map(mode_map)
    return out


def flight_conflicts(flight_df, key_col="flight"):
    """Count flights with conflicting reports across sources."""
    g = flight_df.groupby(key_col)
    sched = g["scheduled_dept"].nunique(dropna=True)
    actual = g["actual_dept"].nunique(dropna=True)
    return {
        "flights_tracked": int(len(g)),
        "conflicting_scheduled": int((sched > 1).sum()),
        "conflicting_actual": int((actual > 1).sum()),
        "example_flight": sched[sched > 1].index[0] if (sched > 1).any() else None,
    }
