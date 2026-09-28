"""Phase 4: Ground truth + pairs.

Builds:
- master_projects (one canonical project/entity per row)
- record_map: source_name, source_record_id, master_project_id
- entity_pairs: pair_id, source_a, source_b, record_a_id, record_b_id, master_id_a, master_id_b, label

Labels: 1 = same entity, 0 = different entity
"""

import pandas as pd
import numpy as np
import random
import os

random.seed(42)
np.random.seed(42)

OUTPUT_DIR = "D:/data viewer/data/processed"
REPORT_DIR = "D:/data viewer/reports"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


def build_record_map(master, sources):
    """Build record_map: source_name, source_record_id, master_project_id."""
    records = []

    for source_name, df in sources.items():
        id_col = {
            "crm": "project_id",
            "erp": "project_reference",
            "partner": "external_reference",
            "document": "project_reference",
        }.get(source_name, "project_id")

        for _, row in df.iterrows():
            records.append({
                "source_name": source_name,
                "source_record_id": str(row[id_col]),
                "master_project_id": str(row[id_col]),
            })

    return pd.DataFrame(records)


def build_positive_pairs(master, sources):
    """Build positive pairs: records from different sources referring to same master project."""
    pairs = []
    pair_id = 0

    source_names = list(sources.keys())

    for _, proj in master.iterrows():
        pid = proj["project_id"]

        records_for_project = []
        for source_name in sources:
            df = sources[source_name]
            id_col = {
                "crm": "project_id",
                "erp": "project_reference",
                "partner": "external_reference",
                "document": "project_reference",
            }.get(source_name, "project_id")

            matching = df[df[id_col] == pid]
            if len(matching) > 0:
                records_for_project.append((source_name, str(matching.iloc[0][id_col])))

        for i in range(len(records_for_project)):
            for j in range(i + 1, len(records_for_project)):
                pair_id += 1
                pairs.append({
                    "pair_id": f"PAIR-{pair_id:06d}",
                    "source_a": records_for_project[i][0],
                    "source_b": records_for_project[j][0],
                    "record_a_id": records_for_project[i][1],
                    "record_b_id": records_for_project[j][1],
                    "master_id_a": pid,
                    "master_id_b": pid,
                    "label": 1,
                })

    return pd.DataFrame(pairs)


def build_negative_pairs(master, sources, n_pairs=2000):
    """Build negative pairs: records from different sources referring to different master projects."""
    pairs = []
    pair_id = 100000

    source_names = list(sources.keys())

    master_ids = master["project_id"].tolist()

    for _ in range(n_pairs):
        pid_a, pid_b = random.sample(master_ids, 2)

        src_a = random.choice(source_names)
        src_b = random.choice([s for s in source_names if s != src_a])

        df_a = sources[src_a]
        df_b = sources[src_b]

        id_col_a = {
            "crm": "project_id",
            "erp": "project_reference",
            "partner": "external_reference",
            "document": "project_reference",
        }.get(src_a, "project_id")

        id_col_b = {
            "crm": "project_id",
            "erp": "project_reference",
            "partner": "external_reference",
            "document": "project_reference",
        }.get(src_b, "project_id")

        rec_a = df_a[df_a[id_col_a] == pid_a]
        rec_b = df_b[df_b[id_col_b] == pid_b]

        if len(rec_a) == 0 or len(rec_b) == 0:
            continue

        pair_id += 1
        pairs.append({
            "pair_id": f"PAIR-{pair_id:06d}",
            "source_a": src_a,
            "source_b": src_b,
            "record_a_id": str(rec_a.iloc[0][id_col_a]),
            "record_b_id": str(rec_b.iloc[0][id_col_b]),
            "master_id_a": pid_a,
            "master_id_b": pid_b,
            "label": 0,
        })

    return pd.DataFrame(pairs)


def build_hard_negatives(master, sources, n_pairs=500):
    """Build hard negative pairs: similar but different entities."""
    pairs = []
    pair_id = 200000

    master_ids = master["project_id"].tolist()

    for _ in range(n_pairs):
        pid_a, pid_b = random.sample(master_ids, 2)

        proj_a = master[master["project_id"] == pid_a].iloc[0]
        proj_b = master[master["project_id"] == pid_b].iloc[0]

        if (proj_a["customer_name"] == proj_b["customer_name"] or
            proj_a["installer"] == proj_b["installer"] or
            proj_a["address"] == proj_b["address"]):

            src_a = random.choice(["crm", "erp", "partner"])
            src_b = random.choice([s for s in ["crm", "erp", "partner"] if s != src_a])

            df_a = sources[src_a]
            df_b = sources[src_b]

            id_col_a = {
                "crm": "project_id",
                "erp": "project_reference",
                "partner": "external_reference",
            }.get(src_a, "project_id")

            id_col_b = {
                "crm": "project_id",
                "erp": "project_reference",
                "partner": "external_reference",
            }.get(src_b, "project_id")

            rec_a = df_a[df_a[id_col_a] == pid_a]
            rec_b = df_b[df_b[id_col_b] == pid_b]

            if len(rec_a) > 0 and len(rec_b) > 0:
                pair_id += 1
                pairs.append({
                    "pair_id": f"PAIR-{pair_id:06d}",
                    "source_a": src_a,
                    "source_b": src_b,
                    "record_a_id": str(rec_a.iloc[0][id_col_a]),
                    "record_b_id": str(rec_b.iloc[0][id_col_b]),
                    "master_id_a": pid_a,
                    "master_id_b": pid_b,
                    "label": 0,
                })

    return pd.DataFrame(pairs)


def main():
    print("Loading data...")
    master = pd.read_csv(f"{OUTPUT_DIR}/master_projects.csv")

    sources = {
        "crm": pd.read_csv(f"{OUTPUT_DIR}/crm_export_corrupted.csv"),
        "erp": pd.read_csv(f"{OUTPUT_DIR}/erp_export_corrupted.csv"),
        "partner": pd.read_excel(f"{OUTPUT_DIR}/partner_export_corrupted.xlsx"),
        "document": pd.read_csv(f"{OUTPUT_DIR}/document_metadata_corrupted.csv"),
    }

    print("Building record_map...")
    record_map = build_record_map(master, sources)
    record_map.to_csv(f"{OUTPUT_DIR}/record_map.csv", index=False)
    print(f"  record_map.csv: {len(record_map)} rows")

    print("Building positive pairs...")
    pos_pairs = build_positive_pairs(master, sources)
    print(f"  positive pairs: {len(pos_pairs)}")

    print("Building negative pairs...")
    neg_pairs = build_negative_pairs(master, sources, n_pairs=2000)
    print(f"  negative pairs: {len(neg_pairs)}")

    print("Building hard negative pairs...")
    hard_neg = build_hard_negatives(master, sources, n_pairs=500)
    print(f"  hard negative pairs: {len(hard_neg)}")

    all_pairs = pd.concat([pos_pairs, neg_pairs, hard_neg], ignore_index=True)
    all_pairs.to_csv(f"{OUTPUT_DIR}/entity_pairs.csv", index=False)
    print(f"  entity_pairs.csv: {len(all_pairs)} rows")

    print("\nClass balance:")
    print(all_pairs["label"].value_counts())

    print("\nLeakage check: verifying master_id_a != master_id_b for negatives...")
    neg_check = all_pairs[all_pairs["label"] == 0]
    same_entity = (neg_check["master_id_a"] == neg_check["master_id_b"]).sum()
    print(f"  Negatives with same master_id: {same_entity} (should be 0)")

    print("\nLeakage check: verifying master_id_a == master_id_b for positives...")
    pos_check = all_pairs[all_pairs["label"] == 1]
    diff_entity = (pos_check["master_id_a"] != pos_check["master_id_b"]).sum()
    print(f"  Positives with different master_id: {diff_entity} (should be 0)")

    with open(f"{REPORT_DIR}/phase4_ground_truth.txt", "w") as f:
        f.write("Phase 4: Ground Truth + Pairs\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Master projects: {len(master)} rows\n")
        f.write(f"Record map: {len(record_map)} rows\n")
        f.write(f"Positive pairs: {len(pos_pairs)}\n")
        f.write(f"Negative pairs: {len(neg_pairs)}\n")
        f.write(f"Hard negative pairs: {len(hard_neg)}\n")
        f.write(f"Total pairs: {len(all_pairs)}\n\n")
        f.write(f"Class balance:\n{all_pairs['label'].value_counts()}\n\n")
        f.write(f"Leakage checks:\n")
        f.write(f"  Negatives with same master_id: {same_entity} (should be 0)\n")
        f.write(f"  Positives with different master_id: {diff_entity} (should be 0)\n")

    print(f"\nReport saved to {REPORT_DIR}/phase4_ground_truth.txt")


if __name__ == "__main__":
    main()
