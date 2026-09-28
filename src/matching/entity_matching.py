"""Phase 5-6: Entity matching baseline + ML.

Stage 1: Deterministic baseline (exact match, normalized match)
Stage 2: Candidate generation / blocking
Stage 3: Pairwise features
Stage 4: Supervised ML (Logistic Regression, Random Forest)
Stage 5: Threshold calibration
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import os
import json

OUTPUT_DIR = "D:/data viewer/data/processed"
REPORT_DIR = "D:/data viewer/reports"
MODEL_DIR = "D:/data viewer/models"

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def normalize_text(s):
    if pd.isna(s):
        return ""
    return str(s).strip().lower()


def jaro_winkler_similarity(s1, s2):
    s1, s2 = normalize_text(s1), normalize_text(s2)
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0

    len1, len2 = len(s1), len(s2)
    match_distance = max(len1, len2) // 2 - 1

    s1_matches = [False] * len1
    s2_matches = [False] * len2

    matches = 0
    transpositions = 0

    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)

        for j in range(start, end):
            if s2_matches[j]:
                continue
            if s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    transpositions //= 2

    jaro = (matches / len1 + matches / len2 + (matches - transpositions) / matches) / 3

    prefix = 0
    for i in range(min(4, len1, len2)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break

    return jaro + prefix * 0.1 * (1 - jaro)


def build_pairwise_features(pairs, sources):
    """Build pairwise features for entity pairs."""
    features = []

    for _, pair in pairs.iterrows():
        src_a = sources[pair["source_a"]]
        src_b = sources[pair["source_b"]]

        id_col_a = {
            "crm": "project_id",
            "erp": "project_reference",
            "partner": "external_reference",
            "document": "project_reference",
        }.get(pair["source_a"], "project_id")

        id_col_b = {
            "crm": "project_id",
            "erp": "project_reference",
            "partner": "external_reference",
            "document": "project_reference",
        }.get(pair["source_b"], "project_id")

        rec_a = src_a[src_a[id_col_a].astype(str) == str(pair["record_a_id"])]
        rec_b = src_b[src_b[id_col_b].astype(str) == str(pair["record_b_id"])]

        if len(rec_a) == 0 or len(rec_b) == 0:
            continue

        rec_a = rec_a.iloc[0]
        rec_b = rec_b.iloc[0]

        # Get canonical field values
        def get_field(rec, *names):
            for name in names:
                if name in rec.index:
                    return rec[name]
            return None

        customer_a = get_field(rec_a, "customer_name", "customer")
        customer_b = get_field(rec_b, "customer_name", "customer")
        installer_a = get_field(rec_a, "installer", "installer_name")
        installer_b = get_field(rec_b, "installer", "installer_name")
        address_a = get_field(rec_a, "address", "site_address")
        address_b = get_field(rec_b, "address", "site_address")
        city_a = get_field(rec_a, "city")
        city_b = get_field(rec_b, "city")
        state_a = get_field(rec_a, "state", "region")
        state_b = get_field(rec_b, "state", "region")
        capacity_a = get_field(rec_a, "system_size_kw", "installed_capacity", "capacity")
        capacity_b = get_field(rec_b, "system_size_kw", "installed_capacity", "capacity")
        status_a = get_field(rec_a, "status", "record_status", "completion_status")
        status_b = get_field(rec_b, "status", "record_status", "completion_status")

        customer_sim = jaro_winkler_similarity(customer_a, customer_b) if customer_a and customer_b else 0.0
        installer_sim = jaro_winkler_similarity(installer_a, installer_b) if installer_a and installer_b else 0.0
        address_sim = jaro_winkler_similarity(address_a, address_b) if address_a and address_b else 0.0
        city_match = 1.0 if normalize_text(city_a) == normalize_text(city_b) else 0.0
        state_match = 1.0 if normalize_text(state_a) == normalize_text(state_b) else 0.0

        try:
            cap_a = float(capacity_a) if capacity_a is not None else None
            cap_b = float(capacity_b) if capacity_b is not None else None
            capacity_diff = abs(cap_a - cap_b) if cap_a is not None and cap_b is not None else None
        except (ValueError, TypeError):
            capacity_diff = None

        status_match = 1.0 if normalize_text(status_a) == normalize_text(status_b) else 0.0

        features.append({
            "pair_id": pair["pair_id"],
            "customer_similarity": customer_sim,
            "installer_similarity": installer_sim,
            "address_similarity": address_sim,
            "city_match": city_match,
            "state_match": state_match,
            "capacity_diff": capacity_diff if capacity_diff is not None else 0.0,
            "status_match": status_match,
            "label": pair["label"],
        })

    return pd.DataFrame(features)


def deterministic_baseline(features):
    """Stage 1: Deterministic baseline."""
    predictions = []
    for _, row in features.iterrows():
        score = (row["customer_similarity"] + row["installer_similarity"] +
                 row["address_similarity"] + row["city_match"] + row["state_match"]) / 5
        pred = 1 if score >= 0.8 else 0
        predictions.append(pred)

    y_true = features["label"].values
    y_pred = np.array(predictions)

    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def train_ml_models(features):
    """Stage 4: Train supervised ML models."""
    X = features[["customer_similarity", "installer_similarity", "address_similarity",
                   "city_match", "state_match", "capacity_diff", "status_match"]]
    X = X.fillna(0)
    y = features["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    results = {}

    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_prob_lr = lr.predict_proba(X_test)[:, 1]

    results["logistic_regression"] = {
        "precision": precision_score(y_test, y_pred_lr),
        "recall": recall_score(y_test, y_pred_lr),
        "f1": f1_score(y_test, y_pred_lr),
        "roc_auc": roc_auc_score(y_test, y_prob_lr),
        "confusion_matrix": confusion_matrix(y_test, y_pred_lr).tolist(),
    }

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]

    results["random_forest"] = {
        "precision": precision_score(y_test, y_pred_rf),
        "recall": recall_score(y_test, y_pred_rf),
        "f1": f1_score(y_test, y_pred_rf),
        "roc_auc": roc_auc_score(y_test, y_prob_rf),
        "confusion_matrix": confusion_matrix(y_test, y_pred_rf).tolist(),
    }

    return results, lr, rf


def main():
    print("Loading data...")
    pairs = pd.read_csv(f"{OUTPUT_DIR}/entity_pairs.csv")

    sources = {
        "crm": pd.read_csv(f"{OUTPUT_DIR}/crm_export_corrupted.csv"),
        "erp": pd.read_csv(f"{OUTPUT_DIR}/erp_export_corrupted.csv"),
        "partner": pd.read_excel(f"{OUTPUT_DIR}/partner_export_corrupted.xlsx"),
        "document": pd.read_csv(f"{OUTPUT_DIR}/document_metadata_corrupted.csv"),
    }

    print("Building pairwise features...")
    features = build_pairwise_features(pairs, sources)
    features = features.fillna(0)
    features.to_csv(f"{OUTPUT_DIR}/pairwise_features.csv", index=False)
    print(f"  pairwise_features.csv: {len(features)} rows")

    print("\nStage 1: Deterministic baseline...")
    baseline_results = deterministic_baseline(features)
    print(f"  {baseline_results}")

    print("\nStage 4: Training ML models...")
    ml_results, lr, rf = train_ml_models(features)
    for model_name, metrics in ml_results.items():
        print(f"  {model_name}: {metrics}")

    with open(f"{REPORT_DIR}/phase5_6_entity_matching.txt", "w") as f:
        f.write("Phase 5-6: Entity Matching Baseline + ML\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total pairs: {len(features)}\n")
        f.write(f"Class balance:\n{features['label'].value_counts()}\n\n")
        f.write("Stage 1: Deterministic Baseline\n")
        f.write(f"  {baseline_results}\n\n")
        f.write("Stage 4: Supervised ML\n")
        for model_name, metrics in ml_results.items():
            f.write(f"  {model_name}:\n")
            for k, v in metrics.items():
                f.write(f"    {k}: {v}\n")
            f.write("\n")

    print(f"\nReport saved to {REPORT_DIR}/phase5_6_entity_matching.txt")


if __name__ == "__main__":
    main()
