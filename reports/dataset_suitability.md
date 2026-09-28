# Dataset Suitability Report

## Summary

| Dataset | Status | Rows | Columns | Missing Values | Duplicates |
|---------|--------|------|---------|----------------|------------|
| Solar Power Generation (Plant 1 Gen) | LOADED | 68,778 | 7 | None | 0 |
| Solar Power Generation (Plant 1 Weather) | LOADED | 3,182 | 6 | None | 0 |
| Solar Power Generation (Plant 2 Gen) | LOADED | 67,698 | 7 | None | 0 |
| Solar Power Generation (Plant 2 Weather) | LOADED | 3,259 | 6 | None | 0 |
| FEBRL4 (df_a) | LOADED | 5,000 | 10 | Yes (8 fields) | 0 |
| FEBRL4 (df_b) | LOADED | 5,000 | 10 | Yes (8 fields) | 0 |
| NAB (4 series + labels) | LOADED | 4,032–22,695/file | 2 | None | 0 |
| Dirty Walmart-Amazon (train/test) | LOADED | 6,144 / 2,049 | 12 | 26,007 / 8,713 | 0 |
| Clean Walmart-Amazon (test) | LOADED | 2,049 | 12 | 1,278 | 0 |
| Dirty DBLP-ACM (test) | LOADED | 2,473 | 10 | 7,436 | 0 |
| Hospital (dirty) | LOADED | 1,000 | 19 | 2,227 | 0 |
| Flights (dirty) | LOADED | 57,246 | 8 | 116,388 | 0 |
| UCI Donation block_1 | LOADED | 100,000 sampled | 12 | None | 0 |
| Epoch 2026 | NOT LOADED | - | - | - | - |

## Epoch 2026 — BLOCKED

- **Reason**: Kaggle competition data requires authentication (API key).
- **Action needed**: User must provide `~/.kaggle/kaggle.json` or run `kaggle competitions download -c epoch-2026`.
- **Expected files**: T1_train.csv, T1_test.csv, T2_train.csv, T2_test.csv.
- **License**: MIT (per Kaggle page).

## Solar Power Generation Data — SUITABLE

- **Source**: https://www.kaggle.com/anikannannal/solar-power-generation-data
- **License**: Verify current Kaggle terms before redistribution.
- **Use case**: Time-series anomaly detection, sensor validation, missing/invalid measurement checks.
- **Key fields**: DC_POWER, AC_POWER, DAILY_YIELD, TOTAL_YIELD, AMBIENT_TEMPERATURE, MODULE_TEMPERATURE, IRRADIATION.
- **Time range**: 2020-05-15 to 2020-06-17 (~34 days, 15-min intervals).
- **Limitations**: No business entity fields (customer, installer, project). Cannot be used for entity resolution.

## FEBRL4 — SUITABLE

- **Source**: Python Record Linkage Toolkit (`recordlinkage.datasets.load_febrl4()`).
- **License**: Public domain (synthetic benchmark).
- **Use case**: Entity resolution benchmark, blocking, pairwise similarity, supervised match classification.
- **Key fields**: given_name, surname, street_number, address_1, address_2, suburb, postcode, state, date_of_birth, soc_sec_id.
- **Ground truth**: Known true links between df_a and df_b.
- **Limitations**: Synthetic data, not solar-domain. Use for methodology validation before domain adaptation.

## NAB (Numenta Anomaly Benchmark) — SUITABLE

- **Source**: https://github.com/numenta/NAB (MIT license).
- **Files**: `realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv` (4,032 rows), `realAWSCloudwatch/grok_asg_anomaly.csv` (4,621 rows), `realKnownCause/machine_temperature_system_failure.csv` (22,695 rows), `realKnownCause/nyc_taxi.csv` (10,320 rows) + `labels/combined_windows.json` (58 series).
- **Use case**: Labeled real-world anomaly detection — replaces synthetic-only injection with ground-truth anomaly windows.
- **Label key format**: `realKnownCause/machine_temperature_system_failure.csv` (no `nab/` prefix).

## DeepMatcher Dirty ER — SUITABLE

- **Source**: https://github.com/icip-cas/EntityMatcher (DeepMatcher-format benchmarks).
- **Files**: dirty_walmart_amazon train (6,144 pairs, 576 matches, 26,007 missing cells) / test (2,049 pairs), clean walmart_amazon test (2,049 pairs, only 1,278 missing), dirty_dblp_acm test (2,473 pairs, 444 matches).
- **Use case**: Real messy product/citation records with match/non-match labels. Dirty vs clean comparison + TF-IDF features.
- **Key fields**: left_title/category/brand/modelno/price, right_* (Walmart-Amazon); left_title/authors/venue/year, right_* (DBLP-ACM).

## Hospital + Flights (HoloClean) — SUITABLE

- **Source**: https://github.com/HoloClean/holoclean (`testdata/`, Apache-2.0).
- **Files**: hospital.csv (1,000 rows × 19 cols, 2,227 missing, ~5% errors) + hospital_clean.csv + hospital_constraints.txt (denial constraints); flight.csv (57,246 rows × 8 cols, 116,388 missing, multi-source conflicts) + flight_clean.csv.
- **Use case**: "Messy → clean" portfolio story: constraint-based error detection + repair (Hospital), cross-source conflict triage (Flights → review queue).

## UCI Donation — SUITABLE

- **Source**: UCI ML Repository (Record Linkage Comparison Patterns), public research dataset.
- **Files**: block_1 (sampled 100,000 of ~5.7M pairs, 2,093 matches in sample).
- **Use case**: Large-scale real ER training data with comparison-pattern features + match labels.

## Suitability Matrix

| Requirement | Solar Power Gen | FEBRL4 | NAB | Dirty ER | Hospital/Flights | Epoch 2026 |
|-------------|-----------------|--------|-----|----------|------------------|------------|
| Entity resolution | No | Yes | No | Yes (real dirty) | Partial (dupes) | Partial |
| Anomaly detection | Yes (unlabeled) | No | Yes (labeled) | No | No | Yes |
| Time-series behavior | Yes | No | Yes | No | No | No |
| Cross-source reconciliation | No | Yes | No | Yes | Yes (Flights) | Yes |
| Data cleaning story | No | No | No | Partial | Yes | No |
| Ground truth for matching | No | Yes | No | Yes | Partial | No |

## Unsupported Assumptions

1. **Epoch 2026 not available**: Solar-domain ER uses synthetic canonical master instead (documented synthetic layer, Section 30).
2. **No business entity fields in Solar Power Gen**: Customer/installer/project fields synthetically generated (Section 5).
3. **FEBRL4 is synthetic**: Methodology validation only; real dirty ER now covered by DeepMatcher sets + UCI Donation.

## Next Steps

1. NAB labeled evaluation in notebook 07.
2. Dirty vs clean ER comparison + TF-IDF in notebooks 05/06.
3. Hospital/Flights cleaning demo in notebook 02.
