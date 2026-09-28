"""Phase 10-11: PostgreSQL Migration Script.

Runs schema.sql and migrates data from CSV/XLSX to PostgreSQL.
"""

import pandas as pd
import os
import sys

OUTPUT_DIR = "D:/data viewer/data/processed"
SQL_DIR = "D:/data viewer/sql"
REPORT_DIR = "D:/data viewer/reports"

os.makedirs(REPORT_DIR, exist_ok=True)


def migrate_to_postgresql():
    """Migrate data to PostgreSQL."""
    try:
        import psycopg2
        from psycopg2.extras import execute_values
    except ImportError:
        print("psycopg2 not available. Install with: pip install psycopg2-binary")
        return False

    try:
        conn = psycopg2.connect(
            host=os.environ.get("PGHOST", "localhost"),
            port=int(os.environ.get("PGPORT", "5432")),
            database=os.environ.get("PGDATABASE", "solar_review"),
            user=os.environ.get("PGUSER", "postgres"),
            password=os.environ.get("PGPASSWORD", ""),
        )
        conn.autocommit = True
        cur = conn.cursor()

        # Run schema
        with open(f"{SQL_DIR}/schema.sql", "r") as f:
            cur.execute(f.read())
        print("Schema created")

        # Migrate projects
        master = pd.read_csv(f"{OUTPUT_DIR}/master_projects.csv")
        cur.execute("DELETE FROM projects")
        execute_values(
            cur,
            "INSERT INTO projects (project_id, customer_name, installer, system_size_kw, address, city, state, postcode, status, install_date, updated_at) VALUES %s",
            [tuple(row) for row in master.values]
        )
        print(f"Projects: {len(master)} rows")

        # Migrate source records
        canonical = pd.read_csv(f"{OUTPUT_DIR}/canonical_source_records.csv")
        cur.execute("DELETE FROM source_records")
        execute_values(
            cur,
            "INSERT INTO source_records (source_system, source_record_id, master_project_id, raw_file_reference, ingested_at) VALUES %s",
            [tuple(row) for row in canonical.values]
        )
        print(f"Source records: {len(canonical)} rows")

        # Migrate entity matches
        pairs = pd.read_csv(f"{OUTPUT_DIR}/entity_pairs.csv")
        cur.execute("DELETE FROM entity_matches")
        execute_values(
            cur,
            "INSERT INTO entity_matches (pair_id, source_a, source_b, record_a_id, record_b_id, master_id_a, master_id_b, label) VALUES %s",
            [tuple(row) for row in pairs.values]
        )
        print(f"Entity matches: {len(pairs)} rows")

        # Migrate AI reviewer output
        ai_output = pd.read_csv(f"{OUTPUT_DIR}/ai_reviewer_output.csv")
        cur.execute("DELETE FROM model_predictions")
        execute_values(
            cur,
            "INSERT INTO model_predictions (model_name, model_version, prediction, probability, feature_schema_version, created_at) VALUES %s",
            [("ai_reviewer", "v1.0", row["classification"], 0.5, "v1.0", row["timestamp"]) for _, row in ai_output.iterrows()]
        )
        print(f"Model predictions: {len(ai_output)} rows")

        # Migrate reviewer feedback
        feedback_path = f"{OUTPUT_DIR}/reviewer_feedback.csv"
        if os.path.exists(feedback_path):
            feedback = pd.read_csv(feedback_path)
            cur.execute("DELETE FROM reviewer_feedback")
            execute_values(
                cur,
                "INSERT INTO reviewer_feedback (pair_id, reviewer_decision, reviewer_reason, reviewer_comment, timestamp, model_version) VALUES %s",
                [tuple(row) for row in feedback.values]
            )
            print(f"Reviewer feedback: {len(feedback)} rows")

        cur.close()
        conn.close()
        return True

    except Exception as e:
        print(f"Migration failed: {e}")
        return False


def main():
    print("Phase 10-11: PostgreSQL Migration")
    print("=" * 60)

    success = migrate_to_postgresql()

    with open(f"{REPORT_DIR}/phase10_11_postgresql_ingestion.txt", "w") as f:
        f.write("Phase 10-11: PostgreSQL + File Ingestion\n")
        f.write("=" * 60 + "\n\n")
        f.write("PostgreSQL schema: sql/schema.sql\n")
        f.write("Migration script: sql/migrate.py\n")
        f.write(f"Migration status: {'SUCCESS' if success else 'FAILED'}\n\n")
        f.write("Tables: projects, source_records, entity_matches, documents,\n")
        f.write("  validation_results, anomalies, review_queue, review_decisions,\n")
        f.write("  model_predictions, reviewer_feedback, audit_logs, model_versions,\n")
        f.write("  ingestion_runs\n")

    print(f"\nReport saved to {REPORT_DIR}/phase10_11_postgresql_ingestion.txt")


if __name__ == "__main__":
    main()
