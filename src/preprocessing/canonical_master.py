"""Phase 3: Canonical master + source simulation.

Builds canonical master projects and simulated enterprise sources:
- CRM-like export (CSV)
- ERP-like export (CSV)
- Partner export (XLSX)
- Document/e-signature metadata (CSV)

Also builds a synthetic corruption engine that injects controlled
data-quality problems and logs every corruption operation.
"""

import pandas as pd
import numpy as np
import json
import os
import hashlib
import random
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

DATA_DIR = "D:/data viewer/data/raw/solar_power_generation"
OUTPUT_DIR = "D:/data viewer/data/processed"
REPORT_DIR = "D:/data viewer/reports"
CONFIG_DIR = "D:/data viewer/config"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)

STATES = ["CA", "TX", "FL", "NY", "AZ", "NV", "CO", "MA", "WA", "OR"]
CITIES = {
    "CA": ["Los Angeles", "San Diego", "San Jose", "Sacramento"],
    "TX": ["Houston", "Dallas", "Austin", "San Antonio"],
    "FL": ["Miami", "Orlando", "Tampa", "Jacksonville"],
    "NY": ["New York", "Buffalo", "Rochester", "Albany"],
    "AZ": ["Phoenix", "Tucson", "Mesa", "Scottsdale"],
    "NV": ["Las Vegas", "Reno", "Henderson"],
    "CO": ["Denver", "Colorado Springs", "Aurora"],
    "MA": ["Boston", "Worcester", "Cambridge"],
    "WA": ["Seattle", "Spokane", "Tacoma"],
    "OR": ["Portland", "Eugene", "Salem"],
}
INSTALLERS = [
    "SunPeak Solar LLC", "BrightSky Energy", "HelioPower Inc",
    "Solaris Technologies", "SunVolt Energy", "RayBright Solar",
    "PhotonWorks", "SunStream Energy", "VoltEdge Solar", "Lumina Solar Co",
]


def generate_canonical_master(n_projects=500):
    """Generate canonical master projects from solar plant data."""
    master = []

    for i in range(n_projects):
        state = random.choice(STATES)
        city = random.choice(CITIES[state])
        installer = random.choice(INSTALLERS)

        project_id = f"PRJ-{2024}-{i+1:05d}"
        capacity_kw = round(random.uniform(3.0, 15.0), 1)

        master.append({
            "project_id": project_id,
            "customer_name": f"Customer {i+1}",
            "installer": installer,
            "system_size_kw": capacity_kw,
            "address": f"{random.randint(100, 9999)} {random.choice(['Oak', 'Maple', 'Cedar', 'Pine', 'Elm'])} {random.choice(['Ave', 'St', 'Blvd', 'Dr', 'Ln'])}",
            "city": city,
            "state": state,
            "postcode": f"{random.randint(10000, 99999)}",
            "status": random.choice(["Completed", "In Progress", "Pending", "On Hold"]),
            "install_date": (datetime(2020, 5, 15) + timedelta(days=random.randint(0, 365))).strftime("%Y-%m-%d"),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

    return pd.DataFrame(master)


def generate_crm_export(master):
    """Generate CRM-like export with different field names."""
    crm = pd.DataFrame({
        "project_id": master["project_id"],
        "customer_name": master["customer_name"],
        "installer": master["installer"],
        "system_size_kw": master["system_size_kw"],
        "address": master["address"],
        "city": master["city"],
        "state": master["state"],
        "status": master["status"],
        "updated_at": master["updated_at"],
    })
    return crm


def generate_erp_export(master):
    """Generate ERP-like export with different field names."""
    erp = pd.DataFrame({
        "customer_ref": master["customer_name"],
        "project_reference": master["project_id"],
        "customer": master["customer_name"],
        "installed_capacity": master["system_size_kw"],
        "region": master["state"],
        "record_status": master["status"].map({
            "Completed": "ACTIVE",
            "In Progress": "PENDING",
            "Pending": "PENDING",
            "On Hold": "SUSPENDED",
        }),
    })
    return erp


def generate_partner_export(master):
    """Generate Partner export with different field names."""
    partner = pd.DataFrame({
        "external_reference": master["project_id"],
        "installer_name": master["installer"],
        "capacity": master["system_size_kw"],
        "site_address": master["address"],
        "completion_status": master["status"],
    })
    return partner


def generate_document_metadata(master):
    """Generate Document/e-signature metadata."""
    doc = pd.DataFrame({
        "envelope_reference": [f"ENV-{hashlib.md5(pid.encode()).hexdigest()[:12].upper()}" for pid in master["project_id"]],
        "project_reference": master["project_id"],
        "signing_status": master["status"].map({
            "Completed": "SIGNED",
            "In Progress": "PENDING_SIGNATURE",
            "Pending": "NOT_SENT",
            "On Hold": "DECLINED",
        }),
        "signed_date": master["install_date"],
        "document_name": [f"solar_Agreement_{pid}.pdf" for pid in master["project_id"]],
    })
    return doc


def inject_corruption(df, corruption_log, source_name):
    """Inject controlled corruption into a dataframe."""
    df = df.copy()
    n = len(df)

    for _ in range(int(n * 0.05)):
        idx = random.randint(0, n - 1)
        field = random.choice(df.columns)
        old_val = df.at[idx, field]

        corruption_type = random.choice([
            "text_formatting", "numeric_drift", "missing", "status_variant"
        ])

        if corruption_type == "text_formatting":
            new_val = str(old_val).upper() if random.random() > 0.5 else str(old_val).lower()
        elif corruption_type == "numeric_drift" and isinstance(old_val, (int, float)):
            new_val = float(old_val) * random.uniform(0.8, 1.2)
        elif corruption_type == "missing":
            new_val = None
        elif corruption_type == "status_variant":
            variants = {
                "Completed": ["Complete", "COMPLETED", "completed"],
                "In Progress": ["In Progress", "IN PROGRESS", "in progress"],
                "Pending": ["Pending", "PENDING", "pending"],
                "On Hold": ["On Hold", "ON HOLD", "on hold"],
            }
            new_val = random.choice(variants.get(str(old_val), [str(old_val)]))
        else:
            new_val = old_val

        df.at[idx, field] = new_val
        corruption_log.append({
            "source": source_name,
            "row_index": idx,
            "field": field,
            "old_value": str(old_val),
            "new_value": str(new_val),
            "corruption_type": corruption_type,
            "timestamp": datetime.now().isoformat(),
        })

    return df


def main():
    print("Generating canonical master...")
    master = generate_canonical_master(500)
    master.to_csv(f"{OUTPUT_DIR}/master_projects.csv", index=False)
    print(f"  master_projects.csv: {len(master)} rows")

    print("Generating CRM export...")
    crm = generate_crm_export(master)
    crm.to_csv(f"{OUTPUT_DIR}/crm_export.csv", index=False)
    print(f"  crm_export.csv: {len(crm)} rows")

    print("Generating ERP export...")
    erp = generate_erp_export(master)
    erp.to_csv(f"{OUTPUT_DIR}/erp_export.csv", index=False)
    print(f"  erp_export.csv: {len(erp)} rows")

    print("Generating Partner export...")
    partner = generate_partner_export(master)
    partner.to_excel(f"{OUTPUT_DIR}/partner_export.xlsx", index=False)
    print(f"  partner_export.xlsx: {len(partner)} rows")

    print("Generating Document metadata...")
    doc = generate_document_metadata(master)
    doc.to_csv(f"{OUTPUT_DIR}/document_metadata.csv", index=False)
    print(f"  document_metadata.csv: {len(doc)} rows")

    print("Injecting corruption...")
    corruption_log = []
    crm_corrupted = inject_corruption(crm, corruption_log, "crm")
    erp_corrupted = inject_corruption(erp, corruption_log, "erp")
    partner_corrupted = inject_corruption(partner, corruption_log, "partner")
    doc_corrupted = inject_corruption(doc, corruption_log, "document")

    crm_corrupted.to_csv(f"{OUTPUT_DIR}/crm_export_corrupted.csv", index=False)
    erp_corrupted.to_csv(f"{OUTPUT_DIR}/erp_export_corrupted.csv", index=False)
    partner_corrupted.to_excel(f"{OUTPUT_DIR}/partner_export_corrupted.xlsx", index=False)
    doc_corrupted.to_csv(f"{OUTPUT_DIR}/document_metadata_corrupted.csv", index=False)

    with open(f"{OUTPUT_DIR}/corruption_log.json", "w") as f:
        json.dump(corruption_log, f, indent=2)

    print(f"  Corruption operations: {len(corruption_log)}")

    with open(f"{REPORT_DIR}/phase3_source_simulation.txt", "w") as f:
        f.write("Phase 3: Canonical Master + Source Simulation\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Master projects: {len(master)} rows\n")
        f.write(f"CRM export: {len(crm)} rows\n")
        f.write(f"ERP export: {len(erp)} rows\n")
        f.write(f"Partner export: {len(partner)} rows\n")
        f.write(f"Document metadata: {len(doc)} rows\n")
        f.write(f"Corruption operations: {len(corruption_log)}\n")

    print(f"\nReport saved to {REPORT_DIR}/phase3_source_simulation.txt")


if __name__ == "__main__":
    main()
