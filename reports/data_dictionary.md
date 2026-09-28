# Data Dictionary

## Solar Power Generation Data

### Plant_1_Generation_Data.csv

| Column | Type | Description | Missing | Example |
|--------|------|-------------|---------|---------|
| DATE_TIME | object | Timestamp (DD-MM-YYYY HH:MM) | 0 | 01-06-2020 00:00 |
| PLANT_ID | int64 | Plant identifier | 0 | 4135001 |
| SOURCE_KEY | object | Inverter/source identifier | 0 | 1BY6WEcLGh8j5v7 |
| DC_POWER | float64 | DC power output (kW) | 0 | 0.0 |
| AC_POWER | float64 | AC power output (kW) | 0 | 0.0 |
| DAILY_YIELD | float64 | Daily energy yield (kWh) | 0 | 0.0 |
| TOTAL_YIELD | float64 | Total energy yield (kWh) | 0 | 6259.0 |

### Plant_1_Weather_Sensor_Data.csv

| Column | Type | Description | Missing | Example |
|--------|------|-------------|---------|---------|
| DATE_TIME | object | Timestamp (YYYY-MM-DD HH:MM:SS) | 0 | 2020-05-15 00:00:00 |
| PLANT_ID | int64 | Plant identifier | 0 | 4135001 |
| SOURCE_KEY | object | Sensor identifier | 0 | 1BY6WEcLGh8j5v7 |
| AMBIENT_TEMPERATURE | float64 | Ambient temperature (°C) | 0 | 23.5 |
| MODULE_TEMPERATURE | float64 | Module temperature (°C) | 0 | 22.1 |
| IRRADIATION | float64 | Solar irradiation (W/m²) | 0 | 0.0 |

### Plant_2_Generation_Data.csv

Same schema as Plant_1_Generation_Data.csv.

### Plant_2_Weather_Sensor_Data.csv

Same schema as Plant_1_Weather_Sensor_Data.csv.

## FEBRL4

### df_a / df_b (identical schema)

| Column | Type | Description | Missing (a) | Missing (b) |
|--------|------|-------------|-------------|-------------|
| given_name | object | First name | 112 | 234 |
| surname | object | Last name | 48 | 102 |
| street_number | object | Street number | 158 | 287 |
| address_1 | object | Address line 1 | 98 | 220 |
| address_2 | object | Address line 2 | 420 | 851 |
| suburb | object | Suburb/city | 55 | 106 |
| postcode | object | Postal code | 0 | 0 |
| state | object | State code | 50 | 107 |
| date_of_birth | object | Date of birth | 94 | 199 |
| soc_sec_id | object | Social security ID | 0 | 0 |

## NAB Time Series

### All NAB series (identical schema)

| Column | Type | Description | Missing | Example |
|--------|------|-------------|---------|---------|
| timestamp | object | Timestamp (YYYY-MM-DD HH:MM:SS) | 0 | 2014-02-27 00:00:00 |
| value | float64 | Metric value | 0 | 20.12 |

Labels in `labels/combined_windows.json`: `{series_path: [[start, end], ...]}` anomaly windows (key format `realKnownCause/machine_temperature_system_failure.csv`).

## DeepMatcher Dirty ER

### dirty_walmart_amazon / walmart_amazon (identical schema)

| Column | Type | Description | Missing (dirty train) | Missing (clean test) |
|--------|------|-------------|----------------------|---------------------|
| id | int64 | Pair identifier | 0 | 0 |
| label | int64 | 1 = match, 0 = non-match | 0 | 0 |
| left_title / right_title | object | Product title | high (dirty) | low |
| left_category / right_category | object | Product category | high (dirty) | low |
| left_brand / right_brand | object | Product brand | high (dirty) | low |
| left_modelno / right_modelno | object | Model number | high (dirty) | low |
| left_price / right_price | object | Price | high (dirty) | low |

Class balance: dirty train 576/6,144 matches (9.4%); test 193/2,049 (9.4%).

### dirty_dblp_acm test (2,473 pairs, 444 matches)

| Column | Type | Description |
|--------|------|-------------|
| id | int64 | Pair identifier |
| label | int64 | 1 = match, 0 = non-match |
| left_title / right_title | object | Paper title |
| left_authors / right_authors | object | Authors |
| left_venue / right_venue | object | Venue |
| left_year / right_year | object | Year |

## Hospital (HoloClean)

| Column | Type | Description | Missing |
|--------|------|-------------|---------|
| ProviderNumber | object | Provider ID | varies |
| HospitalName | object | Hospital name | varies |
| Address1/2/3 | object | Address lines | varies |
| City / State / ZipCode / CountyName | object | Location | varies |
| PhoneNumber | object | Phone | varies |
| HospitalType / HospitalOwner | object | Type/owner | varies |
| EmergencyService | object | Emergency flag | varies |
| Condition / MeasureCode / MeasureName | object | Measure | varies |
| Score / Sample / Stateavg | object | Scores | varies |

19 columns, 1,000 rows, 2,227 missing cells, ~5% erroneous. Constraints in `hospital_constraints.txt` (denial constraints). Clean version: `hospital_clean.csv`.

## Flights (HoloClean)

| Column | Type | Description | Missing |
|--------|------|-------------|---------|
| src | object | Source website | 0 |
| flight | object | Flight number | varies |
| scheduled_dept / actual_dept | object | Departure times | varies |
| dept_gate | object | Departure gate | varies |
| scheduled_arrival / actual_arrival | object | Arrival times | varies |
| arrival_gate | object | Arrival gate | varies |

57,246 rows, 116,388 missing cells, multi-source conflicts. Clean version: `flight_clean.csv`.

## UCI Donation

| Column | Type | Description |
|--------|------|-------------|
| id_1 / id_2 | int64 | Record identifiers |
| cmp_fname_c1/c2, cmp_lname_c1/c2 | object/float | Name agreement scores |
| cmp_sex, cmp_bd/bm/by, cmp_plz | mixed | Sex/DOB/postcode agreement |
| is_match | bool | Ground-truth match label |

## Epoch 2026 (NOT LOADED)

Expected files: T1_train.csv, T1_test.csv, T2_train.csv, T2_test.csv.
Schema unknown until downloaded.
