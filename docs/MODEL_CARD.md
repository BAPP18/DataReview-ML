# Model Card — Entity Matcher v1

## Model details

- **Name:** Entity Matcher
- **Version:** v1.0
- **Type:** Logistic Regression
- **Artifact:** `models/entity_matcher_v1.joblib`
- **Metadata:** `models/entity_matcher_v1.meta.json`
- **Primary use:** assist routing of candidate cross-source record pairs into automatic match, manual review, or non-match paths.

## Intended use

The model is a decision-support component for a human reviewer workflow. It should help prioritize records and surface likely matches.

It is **not** intended to:
- establish legal identity;
- autonomously overwrite source systems;
- make financial or eligibility decisions;
- replace evidence review.

## Features

- customer similarity
- installer similarity
- address similarity
- city match
- state match
- capacity difference
- status match

## Reported metrics

The committed metadata currently records:

| Metric | Value |
|---|---:|
| Precision | 0.7282 |
| Recall | 1.0000 |
| F1 | 0.8427 |
| ROC-AUC | 0.8331 |

These metrics belong to the documented training/evaluation setup and should not be assumed to generalize to unrelated domains.

## Data

The project uses public benchmark datasets plus documented synthetic enterprise-style records. See the root README and `reports/data_lineage.md`.

## Limitations and risks

- Distribution shift can materially change precision/recall.
- Dirty real-world entity-resolution datasets show much lower recall than the synthetic workflow.
- Similarity features may over-weight common names/addresses if blocking and thresholds are poorly calibrated.
- A high score is not source truth.
- The reviewer UI must continue to show underlying evidence and uncertainty.

## Human oversight

Final decisions are made by a reviewer. Decisions require structured reason codes and are written to an audit trail.

## Reproducibility

Rebuild the artifact with:

```bash
python src/matching/train_model.py
```

After retraining, update metadata and re-run the full test/validation suite before replacing the versioned artifact.
