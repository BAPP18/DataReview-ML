"""Train the entity-matching model ONCE and persist the artifact.

Pipeline:  training script  ->  models/entity_matcher_v1.joblib (+ .meta.json)
           -> model registry columns (model_versions table, sql/schema.sql)
           -> app/dashboard.py loads the artifact (inference only).

The dashboard must never re-fit on every page load.
"""

import json
import os
from datetime import date

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
ARTIFACT = os.path.join(MODEL_DIR, "entity_matcher_v1.joblib")
META_PATH = os.path.join(MODEL_DIR, "entity_matcher_v1.meta.json")

FEATURES = ["customer_similarity", "installer_similarity", "address_similarity",
            "city_match", "state_match", "capacity_diff", "status_match"]


def train(output_artifact=ARTIFACT, random_state=42):
    """Fit Logistic Regression on pairwise features; persist artifact + meta."""
    feats = pd.read_csv(os.path.join(OUTPUT_DIR, "pairwise_features.csv")).fillna(0)
    X = feats[FEATURES]
    y = feats["label"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=random_state, stratify=y)
    model = LogisticRegression(random_state=random_state, max_iter=1000)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)
    meta = {
        "model_name": "entity_matcher",
        "model_version": "v1.0",
        "model_type": "logistic_regression",
        "feature_schema_version": "v1.0",
        "features": FEATURES,
        "training_rows": int(len(X_train)),
        "training_date": date.today().isoformat(),
        "metrics": {
            "precision": round(float(precision_score(y_test, pred)), 4),
            "recall": round(float(recall_score(y_test, pred)), 4),
            "f1": round(float(f1_score(y_test, pred)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        },
    }
    os.makedirs(os.path.dirname(output_artifact) or ".", exist_ok=True)
    joblib.dump(model, output_artifact)
    with open(meta_path_for(output_artifact), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {output_artifact}  f1={meta['metrics']['f1']} auc={meta['metrics']['roc_auc']}")
    return model, meta


def meta_path_for(artifact_path):
    base = artifact_path[:-len(".joblib")] if artifact_path.endswith(".joblib") else artifact_path
    return base + ".meta.json"


def load_artifact(path=ARTIFACT):
    """Load a persisted artifact; raises FileNotFoundError if absent."""
    model = joblib.load(path)
    meta_path = meta_path_for(path)
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    return model, meta


if __name__ == "__main__":
    train()
