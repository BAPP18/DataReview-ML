# Data Lineage

## Solar Power Generation Data

- **Source URL**: https://www.kaggle.com/anikannal/solar-power-generation-data
- **Download method**: Kaggle API v1 (direct download, no auth required)
- **Download date**: 2026-09-27
- **Local path**: `data/raw/solar_power_generation/`
- **Files**:
  - Plant_1_Generation_Data.csv (4,839,076 bytes)
  - Plant_1_Weather_Sensor_Data.csv (287,847 bytes)
  - Plant_2_Generation_Data.csv (5,805,157 bytes)
  - Plant_2_Weather_Sensor_Data.csv (301,443 bytes)
- **License**: Verify current Kaggle terms before redistribution.
- **Provenance**: Kaggle dataset by anikannal. Two solar plants, 15-min interval measurements, ~34 days (2020-05-15 to 2020-06-17).

## FEBRL4

- **Source**: Python Record Linkage Toolkit v0.16
- **Load method**: `recordlinkage.datasets.load_febrl4()`
- **Download date**: 2026-09-27
- **License**: Public domain (synthetic benchmark)
- **Provenance**: FEBRL (Freely Extensible Biomedical Record Linkage) project. Synthetic data with known true links.
- **Structure**: df_a (5,000 records) + df_b (5,000 records), 5,000 originals + 5,000 duplicates.

## NAB (Numenta Anomaly Benchmark)

- **Source URL**: https://github.com/numenta/NAB
- **Download method**: Direct raw download (no auth required)
- **Download date**: 2026-09-28
- **Local path**: `data/raw/nab/`
- **Files**: realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv, realAWSCloudwatch/grok_asg_anomaly.csv, realKnownCause/machine_temperature_system_failure.csv, realKnownCause/nyc_taxi.csv, labels/combined_windows.json, labels/combined_labels.json
- **License**: MIT.
- **Provenance**: NAB v1.1 corpus — real-world labeled time-series (AWS metrics, machine sensors, NYC taxi) with hand-labeled anomaly windows.

## DeepMatcher Dirty ER

- **Source URL**: https://github.com/icip-cas/EntityMatcher (`data/`, DeepMatcher-format)
- **Download method**: Direct raw download (no auth required)
- **Download date**: 2026-09-28
- **Local path**: `data/raw/deepmatcher/`
- **Files**: dirty_walmart_amazon/{train,valid,test}.csv, walmart_amazon/{train,valid,test}.csv, dirty_dblp_acm/test.csv
- **License**: Research benchmarks (SIGMOD 2018 DeepMatcher paper). Verify terms before redistribution.
- **Provenance**: Product (Abt/Buy-style, Walmart-Amazon) and citation (DBLP-ACM) record pairs with match labels; dirty variants have attributes randomly emptied/moved.

## Hospital + Flights (HoloClean)

- **Source URL**: https://github.com/HoloClean/holoclean (`testdata/`)
- **Download method**: Direct raw download (no auth required)
- **Download date**: 2026-09-28
- **Local path**: `data/raw/holoclean/`
- **Files**: hospital.csv, hospital_clean.csv, hospital_constraints.txt, flight.csv, flight_clean.csv
- **License**: Apache-2.0.
- **Provenance**: Real hospital records (~5% errors, 19 attrs) and multi-source flight schedules with cross-source conflicts + denial constraints.

## UCI Donation (Record Linkage Comparison Patterns)

- **Source URL**: https://archive.ics.uci.edu/ml/machine-learning-databases/00210/donation.zip
- **Download method**: Direct download (no auth required)
- **Download date**: 2026-09-27
- **Local path**: `data/raw/donation/`
- **License**: Public research dataset (IMBEI Mainz / UCI ML Repository).
- **Provenance**: 5,749,132 comparison-pattern pairs from cancer-registry record linkage evaluation; 20,931 matches; split into 10 blocks.

## Epoch 2026

- **Source URL**: https://www.kaggle.com/competitions/epoch-2026/data
- **Status**: NOT DOWNLOADED (competition ended 2026-04-11; API returns 403 Forbidden)
- **License**: MIT (per Kaggle page)
- **Expected files**: T1_train.csv, T1_test.csv, T2_train.csv, T2_test.csv
