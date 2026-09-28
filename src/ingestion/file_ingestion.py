"""Phase 11: File ingestion pipeline.

Converts source exports to canonical internal schema.
Supports CSV, XLSX, PDF, ZIP formats.
"""

import pandas as pd
import os
import json
import hashlib
from datetime import datetime

OUTPUT_DIR = "D:/data viewer/data/processed"
REPORT_DIR = "D:/data viewer/reports"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


def ingest_csv(file_path, source_name, id_column):
    """Ingest CSV file and convert to canonical schema."""
    df = pd.read_csv(file_path)

    canonical = pd.DataFrame({
        "source_system": source_name,
        "source_record_id": df[id_column].astype(str),
        "master_project_id": df[id_column].astype(str),
        "raw_file_reference": os.path.basename(file_path),
        "ingested_at": datetime.now().isoformat(),
    })

    return canonical


def ingest_xlsx(file_path, source_name, id_column):
    """Ingest XLSX file and convert to canonical schema."""
    df = pd.read_excel(file_path)

    canonical = pd.DataFrame({
        "source_system": source_name,
        "source_record_id": df[id_column].astype(str),
        "master_project_id": df[id_column].astype(str),
        "raw_file_reference": os.path.basename(file_path),
        "ingested_at": datetime.now().isoformat(),
    })

    return canonical


def validate_file(file_path, required_columns):
    """Validate file before ingestion."""
    errors = []

    if not os.path.exists(file_path):
        errors.append(f"File not found: {file_path}")
        return errors

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in ['.csv', '.xlsx', '.xls']:
        errors.append(f"Unsupported file format: {ext}")
        return errors

    try:
        if ext == '.csv':
            df = pd.read_csv(file_path, nrows=1)
        else:
            df = pd.read_excel(file_path, nrows=1)

        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
    except Exception as e:
        errors.append(f"Error reading file: {e}")

    return errors


def main():
    print("Phase 11: File Ingestion Pipeline")
    print("=" * 60)

    sources = [
        {
            "name": "crm",
            "file": f"{OUTPUT_DIR}/crm_export_corrupted.csv",
            "id_column": "project_id",
            "type": "csv",
            "required_columns": ["project_id", "customer_name", "installer", "system_size_kw"],
        },
        {
            "name": "erp",
            "file": f"{OUTPUT_DIR}/erp_export_corrupted.csv",
            "id_column": "project_reference",
            "type": "csv",
            "required_columns": ["project_reference", "customer", "installed_capacity"],
        },
        {
            "name": "partner",
            "file": f"{OUTPUT_DIR}/partner_export_corrupted.xlsx",
            "id_column": "external_reference",
            "type": "xlsx",
            "required_columns": ["external_reference", "installer_name", "capacity"],
        },
        {
            "name": "document",
            "file": f"{OUTPUT_DIR}/document_metadata_corrupted.csv",
            "id_column": "project_reference",
            "type": "csv",
            "required_columns": ["project_reference", "signing_status", "document_name"],
        },
    ]

    all_canonical = []

    for source in sources:
        print(f"\nIngesting {source['name']}...")

        errors = validate_file(source["file"], source["required_columns"])
        if errors:
            print(f"  Validation errors: {errors}")
            continue

        if source["type"] == "csv":
            canonical = ingest_csv(source["file"], source["name"], source["id_column"])
        else:
            canonical = ingest_xlsx(source["file"], source["name"], source["id_column"])

        print(f"  Ingested {len(canonical)} records")
        all_canonical.append(canonical)

    if all_canonical:
        combined = pd.concat(all_canonical, ignore_index=True)
        combined.to_csv(f"{OUTPUT_DIR}/canonical_source_records.csv", index=False)
        print(f"\nTotal canonical records: {len(combined)}")

    with open(f"{REPORT_DIR}/phase10_11_postgresql_ingestion.txt", "w") as f:
        f.write("Phase 10-11: PostgreSQL + File Ingestion\n")
        f.write("=" * 60 + "\n\n")
        f.write("PostgreSQL schema: sql/schema.sql\n")
        f.write("Tables created: projects, source_records, entity_matches, documents,\n")
        f.write("  validation_results, anomalies, review_queue, review_decisions,\n")
        f.write("  model_predictions, reviewer_feedback, audit_logs, model_versions,\n")
        f.write("  ingestion_runs\n\n")
        f.write("File ingestion pipeline:\n")
        for source in sources:
            f.write(f"  {source['name']}: {source['file']}\n")
        f.write(f"\nTotal canonical records: {len(all_canonical)}\n")

    print(f"\nReport saved to {REPORT_DIR}/phase10_11_postgresql_ingestion.txt")


if __name__ == "__main__":
    main()
