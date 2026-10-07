# Architecture

DataReview-ML is structured as a human-in-the-loop data reconciliation system rather than a standalone prediction model.

## System context

```mermaid
flowchart LR
    A[CRM export] --> I[Ingestion adapters]
    B[ERP export] --> I
    C[Partner export] --> I
    D[Signed document] --> I

    I --> V[Validation & canonical schema]
    V --> M[Entity matching]
    V --> N[Anomaly detection]
    M --> R[Routing engine]
    N --> R

    R --> Q[Reviewer queue]
    Q --> E[Evidence view]
    E --> H[Human decision]

    H --> C1[Correction handoff]
    H --> A1[Audit trail]
    H --> QA[QA sample / second review]
```

## Component boundaries

| Layer | Responsibility | Main location |
|---|---|---|
| Ingestion | Normalize source exports and retain provenance | `src/ingestion/` |
| Preprocessing | Build canonical/master representations | `src/preprocessing/` |
| Matching | Feature generation, entity-resolution model, artifact loading | `src/matching/` |
| Anomaly | Statistical/ML anomaly scoring | `src/anomaly/` |
| Review | Evidence, lifecycle, QA, reviewer metrics | `src/review/` |
| AI/explanation | Reviewer-facing explanations and clarification drafts | `src/ai/` |
| UI | Streamlit reviewer workstation | `app/dashboard.py` |
| Persistence | Demo files + optional PostgreSQL schema | `data/processed/`, `sql/` |

## Design principles

1. **Human decision > model suggestion.** Model scores route work; they do not overwrite source truth.
2. **Evidence before action.** Reviewer decisions are backed by field-level source comparison.
3. **Traceability.** Model version, timestamps, reason codes, and source provenance are preserved.
4. **Reproducibility.** Model artifacts and metadata are versioned; model training is separated from app inference.
5. **Honest evaluation.** Weak benchmark results are reported rather than hidden.

## Runtime modes

### File-based demo
The default portfolio path. Generated processed files are used directly by the Streamlit application.

### Docker Compose
Runs the application plus an optional PostgreSQL service for schema/migration demonstration.

## ML lifecycle

```mermaid
flowchart TD
    D[Public + documented synthetic data] --> F[Feature engineering]
    F --> S[Entity-level split]
    S --> T[Model training]
    T --> E[Evaluation]
    E --> A[Versioned artifact + metadata]
    A --> P[App inference]
    P --> H[Human review]
    H --> Q[QA & feedback]
```
