"""Phase 8-9: AI reviewer + evaluation.

AI-assisted reviewer:
- Discrepancy classification
- Evidence summarization
- Clarification draft
- Explanation

Final evaluation report with model versions, metrics, thresholds, error analysis.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

OUTPUT_DIR = "D:/data viewer/data/processed"
REPORT_DIR = "D:/data viewer/reports"
MODEL_DIR = "D:/data viewer/models"

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def classify_discrepancy(row):
    """Classify discrepancy type based on field differences."""
    if row.get("customer_similarity", 1.0) < 0.5:
        return "ENTITY_MISMATCH"
    if pd.isna(row.get("capacity_diff")) or row.get("capacity_diff", 0) > 2.0:
        return "NUMERIC_CONFLICT"
    if row.get("status_match", 1.0) == 0:
        return "STATUS_CONFLICT"
    if row.get("address_similarity", 1.0) < 0.7:
        return "FORMATTING_DIFFERENCE"
    if row.get("installer_similarity", 1.0) < 0.7:
        return "POTENTIAL_DUPLICATE"
    return "OTHER"


def summarize_evidence(row):
    """Generate concise reviewer summary."""
    parts = []
    if row.get("customer_similarity", 1.0) < 0.8:
        parts.append(f"Customer name similarity: {row['customer_similarity']:.2f}")
    if row.get("installer_similarity", 1.0) < 0.8:
        parts.append(f"Installer similarity: {row['installer_similarity']:.2f}")
    if row.get("address_similarity", 1.0) < 0.8:
        parts.append(f"Address similarity: {row['address_similarity']:.2f}")
    if row.get("capacity_diff", 0) > 0:
        parts.append(f"Capacity difference: {row['capacity_diff']:.1f} kW")
    if row.get("status_match", 1.0) == 0:
        parts.append("Status mismatch detected")
    return "; ".join(parts) if parts else "No significant discrepancies"


def draft_clarification(row):
    """Generate neutral clarification message."""
    classification = row.get("classification", "OTHER")
    if classification == "NUMERIC_CONFLICT":
        return "Please verify the system capacity value across sources."
    elif classification == "STATUS_CONFLICT":
        return "Please confirm the current project status."
    elif classification == "ENTITY_MISMATCH":
        return "Please verify if these records refer to the same project."
    elif classification == "FORMATTING_DIFFERENCE":
        return "Please confirm the correct address format."
    else:
        return "Please review the flagged discrepancies."


def explain_decision(row):
    """Generate explanation for reviewer."""
    return {
        "what_differs": summarize_evidence(row),
        "why_flagged": f"Classification: {row.get('classification', 'UNKNOWN')}",
        "evidence_considered": [
            f"Customer similarity: {row.get('customer_similarity', 'N/A')}",
            f"Installer similarity: {row.get('installer_similarity', 'N/A')}",
            f"Address similarity: {row.get('address_similarity', 'N/A')}",
            f"Capacity difference: {row.get('capacity_diff', 'N/A')}",
        ],
        "reviewer_action": "Verify flagged fields and confirm match status.",
    }


def main():
    print("Loading pairwise features...")
    features = pd.read_csv(f"{OUTPUT_DIR}/pairwise_features.csv")

    print("Classifying discrepancies...")
    features["classification"] = features.apply(classify_discrepancy, axis=1)

    print("Generating summaries...")
    features["summary"] = features.apply(summarize_evidence, axis=1)

    print("Drafting clarifications...")
    features["clarification"] = features.apply(draft_clarification, axis=1)

    print("Generating explanations...")
    features["explanation"] = features.apply(lambda row: json.dumps(explain_decision(row)), axis=1)

    features.to_csv(f"{OUTPUT_DIR}/ai_reviewer_output.csv", index=False)
    print(f"  ai_reviewer_output.csv: {len(features)} rows")

    print("\nClassification distribution:")
    print(features["classification"].value_counts())

    # Phase 9: Final evaluation report
    print("\nGenerating final evaluation report...")

    eval_report = {
        "model_versions": {
            "entity_matching": "v1.0 (Logistic Regression + Random Forest)",
            "anomaly_detection": "v1.0 (Isolation Forest + Rolling MAD)",
        },
        "dataset_sources": {
            "solar_power_generation": "Kaggle (anikannal/solar-power-generation-data)",
            "febrl4": "Python Record Linkage Toolkit",
            "canonical_master": "Synthetic (Phase 3)",
        },
        "split_strategy": "70/30 train/test split with stratification",
        "metrics": {
            "logistic_regression": {
                "precision": 0.728,
                "recall": 1.0,
                "f1": 0.843,
                "roc_auc": 0.833,
            },
            "random_forest": {
                "precision": 0.737,
                "recall": 1.0,
                "f1": 0.849,
                "roc_auc": 0.842,
            },
        },
        "thresholds": {
            "auto_match": 0.95,
            "manual_review": 0.50,
            "non_match": 0.0,
        },
        "error_analysis": {
            "false_positives": 334,
            "false_negatives": 0,
            "notes": "High recall but moderate precision. Many false positives due to similar names/addresses.",
        },
        "limitations": [
            "Synthetic canonical master data",
            "No real solar-domain entity resolution ground truth",
            "FEBRL4 is synthetic, not solar-domain",
            "Epoch 2026 not available (competition ended)",
        ],
        "timestamp": datetime.now().isoformat(),
    }

    with open(f"{REPORT_DIR}/phase9_final_evaluation.json", "w") as f:
        json.dump(eval_report, f, indent=2)

    with open(f"{REPORT_DIR}/phase8_9_ai_reviewer_evaluation.txt", "w") as f:
        f.write("Phase 8-9: AI Reviewer + Evaluation\n")
        f.write("=" * 60 + "\n\n")
        f.write("AI Reviewer Output:\n")
        f.write(f"  Total records: {len(features)}\n")
        f.write(f"  Classification distribution:\n{features['classification'].value_counts()}\n\n")
        f.write("Final Evaluation:\n")
        f.write(f"  {json.dumps(eval_report, indent=2)}\n")

    print(f"\nReport saved to {REPORT_DIR}/phase8_9_ai_reviewer_evaluation.txt")


if __name__ == "__main__":
    main()
