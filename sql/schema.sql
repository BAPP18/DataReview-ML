-- Phase 10: PostgreSQL Schema
-- Solar Data Review & AI/ML Reconciliation Platform

CREATE TABLE IF NOT EXISTS projects (
    project_id VARCHAR(50) PRIMARY KEY,
    customer_name VARCHAR(255),
    installer VARCHAR(255),
    system_size_kw DECIMAL(10,2),
    address VARCHAR(500),
    city VARCHAR(100),
    state VARCHAR(10),
    postcode VARCHAR(10),
    status VARCHAR(50),
    install_date DATE,
    updated_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS source_records (
    id SERIAL PRIMARY KEY,
    source_system VARCHAR(50) NOT NULL,
    source_record_id VARCHAR(100) NOT NULL,
    master_project_id VARCHAR(50) REFERENCES projects(project_id),
    ingestion_run_id VARCHAR(50),
    raw_file_reference VARCHAR(500),
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_system, source_record_id)
);

CREATE TABLE IF NOT EXISTS entity_matches (
    id SERIAL PRIMARY KEY,
    pair_id VARCHAR(50) UNIQUE,
    source_a VARCHAR(50),
    source_b VARCHAR(50),
    record_a_id VARCHAR(100),
    record_b_id VARCHAR(100),
    master_id_a VARCHAR(50),
    master_id_b VARCHAR(50),
    label INTEGER,
    match_probability DECIMAL(5,4),
    model_version VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    envelope_reference VARCHAR(100) UNIQUE,
    project_reference VARCHAR(50) REFERENCES projects(project_id),
    signing_status VARCHAR(50),
    signed_date DATE,
    document_name VARCHAR(255)
);

CREATE TABLE IF NOT EXISTS validation_results (
    id SERIAL PRIMARY KEY,
    source_system VARCHAR(50),
    source_record_id VARCHAR(100),
    rule_name VARCHAR(100),
    rule_type VARCHAR(50),
    passed BOOLEAN,
    details TEXT,
    validated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS anomalies (
    id SERIAL PRIMARY KEY,
    source_system VARCHAR(50),
    source_record_id VARCHAR(100),
    anomaly_type VARCHAR(50),
    anomaly_score DECIMAL(10,4),
    evidence TEXT,
    reason TEXT,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS review_queue (
    id SERIAL PRIMARY KEY,
    pair_id VARCHAR(50) REFERENCES entity_matches(pair_id),
    priority INTEGER,
    project_id VARCHAR(50) REFERENCES projects(project_id),
    source_systems TEXT,
    issue_type VARCHAR(50),
    match_probability DECIMAL(5,4),
    anomaly_score DECIMAL(10,4),
    status VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS review_decisions (
    id SERIAL PRIMARY KEY,
    pair_id VARCHAR(50) REFERENCES entity_matches(pair_id),
    reviewer_id VARCHAR(50),
    decision VARCHAR(50),
    reason TEXT,
    comment TEXT,
    model_version VARCHAR(50),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_predictions (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100),
    model_version VARCHAR(50),
    prediction VARCHAR(50),
    probability DECIMAL(5,4),
    feature_schema_version VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS reviewer_feedback (
    id SERIAL PRIMARY KEY,
    pair_id VARCHAR(50) REFERENCES entity_matches(pair_id),
    model_probability DECIMAL(5,4),
    anomaly_score DECIMAL(10,4),
    ai_classification VARCHAR(50),
    reviewer_decision VARCHAR(50),
    reviewer_reason TEXT,
    reviewer_comment TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_version VARCHAR(50)
);

CREATE TABLE IF NOT EXISTS audit_logs (
    id SERIAL PRIMARY KEY,
    who VARCHAR(100),
    what TEXT,
    when TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    why TEXT,
    model_version VARCHAR(50),
    source_evidence TEXT
);

CREATE TABLE IF NOT EXISTS model_versions (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100),
    model_version VARCHAR(50),
    training_date DATE,
    metrics JSONB,
    status VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ingestion_runs (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(50) UNIQUE,
    source_system VARCHAR(50),
    file_name VARCHAR(255),
    rows_processed INTEGER,
    rows_successful INTEGER,
    rows_failed INTEGER,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    status VARCHAR(50)
);
