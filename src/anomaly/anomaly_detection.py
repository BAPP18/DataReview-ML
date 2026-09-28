"""Phase 7: Anomaly detection.

Statistical baselines: percentile, IQR, z-score, rolling median/MAD
ML models: Isolation Forest, Local Outlier Factor
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import precision_score, recall_score, f1_score
import os

DATA_DIR = "D:/data viewer/data/raw/solar_power_generation"
OUTPUT_DIR = "D:/data viewer/data/processed"
REPORT_DIR = "D:/data viewer/reports"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


def parse_datetime(series):
    for fmt in ('%d-%m-%Y %H:%M', '%Y-%m-%d %H:%M:%S'):
        try:
            return pd.to_datetime(series, format=fmt)
        except (ValueError, TypeError):
            continue
    return pd.to_datetime(series, format='mixed')


def statistical_baseline(df, column, method='iqr'):
    """Statistical baseline for anomaly detection."""
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return (df[column] < lower) | (df[column] > upper)
    elif method == 'zscore':
        mean = df[column].mean()
        std = df[column].std()
        z_scores = np.abs((df[column] - mean) / std)
        return z_scores > 3
    elif method == 'percentile':
        lower = df[column].quantile(0.01)
        upper = df[column].quantile(0.99)
        return (df[column] < lower) | (df[column] > upper)
    else:
        raise ValueError(f"Unknown method: {method}")


def rolling_mad_baseline(df, column, window=96, threshold=3.5):
    """Rolling median/MAD baseline for time-series anomaly detection."""
    rolling_median = df[column].rolling(window=window, center=True, min_periods=1).median()
    rolling_mad = (df[column] - rolling_median).abs().rolling(window=window, center=True, min_periods=1).median()
    modified_z_scores = 0.6745 * (df[column] - rolling_median) / (rolling_mad + 1e-10)
    return np.abs(modified_z_scores) > threshold


def isolation_forest_baseline(df, columns, contamination=0.05):
    """Isolation Forest for anomaly detection."""
    X = df[columns].fillna(0)
    clf = IsolationForest(contamination=contamination, random_state=42)
    predictions = clf.fit_predict(X)
    return predictions == -1


def lof_baseline(df, columns, n_neighbors=20, contamination=0.05):
    """Local Outlier Factor for anomaly detection."""
    X = df[columns].fillna(0)
    clf = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    predictions = clf.fit_predict(X)
    return predictions == -1


def inject_anomalies(df, column, n_anomalies=50):
    """Inject synthetic anomalies for testing."""
    df = df.copy()
    anomaly_indices = np.random.choice(df.index, size=min(n_anomalies, len(df)), replace=False)
    df.loc[anomaly_indices, column] = df[column].max() * np.random.uniform(2, 5, size=len(anomaly_indices))
    return df, anomaly_indices


def main():
    print("Loading data...")
    gen1 = pd.read_csv(f"{DATA_DIR}/Plant_1_Generation_Data.csv")
    gen2 = pd.read_csv(f"{DATA_DIR}/Plant_2_Generation_Data.csv")
    wx1 = pd.read_csv(f"{DATA_DIR}/Plant_1_Weather_Sensor_Data.csv")
    wx2 = pd.read_csv(f"{DATA_DIR}/Plant_2_Weather_Sensor_Data.csv")

    for df in [gen1, gen2, wx1, wx2]:
        df['DATE_TIME'] = parse_datetime(df['DATE_TIME'])

    results = {}

    print("\n--- Plant 1 Generation: DC_POWER ---")
    col = 'DC_POWER'

    iqr_anomalies = statistical_baseline(gen1, col, 'iqr')
    zscore_anomalies = statistical_baseline(gen1, col, 'zscore')
    percentile_anomalies = statistical_baseline(gen1, col, 'percentile')
    rolling_anomalies = rolling_mad_baseline(gen1, col)

    print(f"  IQR anomalies: {iqr_anomalies.sum()}")
    print(f"  Z-score anomalies: {zscore_anomalies.sum()}")
    print(f"  Percentile anomalies: {percentile_anomalies.sum()}")
    print(f"  Rolling MAD anomalies: {rolling_anomalies.sum()}")

    iso_anomalies = isolation_forest_baseline(gen1, [col])
    print(f"  Isolation Forest anomalies: {iso_anomalies.sum()}")

    lof_anomalies = lof_baseline(gen1, [col])
    print(f"  LOF anomalies: {lof_anomalies.sum()}")

    results['plant1_dc_power'] = {
        'iqr': int(iqr_anomalies.sum()),
        'zscore': int(zscore_anomalies.sum()),
        'percentile': int(percentile_anomalies.sum()),
        'rolling_mad': int(rolling_anomalies.sum()),
        'isolation_forest': int(iso_anomalies.sum()),
        'lof': int(lof_anomalies.sum()),
    }

    print("\n--- Plant 1 Weather: IRRADIATION ---")
    col = 'IRRADIATION'

    iqr_anomalies = statistical_baseline(wx1, col, 'iqr')
    zscore_anomalies = statistical_baseline(wx1, col, 'zscore')
    rolling_anomalies = rolling_mad_baseline(wx1, col)

    print(f"  IQR anomalies: {iqr_anomalies.sum()}")
    print(f"  Z-score anomalies: {zscore_anomalies.sum()}")
    print(f"  Rolling MAD anomalies: {rolling_anomalies.sum()}")

    results['plant1_irradiation'] = {
        'iqr': int(iqr_anomalies.sum()),
        'zscore': int(zscore_anomalies.sum()),
        'rolling_mad': int(rolling_anomalies.sum()),
    }

    print("\n--- Injected Anomaly Recovery Test ---")
    gen1_injected, injected_indices = inject_anomalies(gen1.copy(), 'DC_POWER', n_anomalies=50)
    rolling_anomalies_injected = rolling_mad_baseline(gen1_injected, 'DC_POWER')

    true_positives = rolling_anomalies_injected[injected_indices].sum()
    print(f"  Injected anomalies: {len(injected_indices)}")
    print(f"  Detected by rolling MAD: {true_positives}")
    print(f"  Recovery rate: {true_positives / len(injected_indices):.2%}")

    results['injected_anomaly_recovery'] = {
        'injected': len(injected_indices),
        'detected': int(true_positives),
        'recovery_rate': float(true_positives / len(injected_indices)),
    }

    with open(f"{REPORT_DIR}/phase7_anomaly_detection.txt", "w") as f:
        f.write("Phase 7: Anomaly Detection\n")
        f.write("=" * 60 + "\n\n")
        for key, value in results.items():
            f.write(f"{key}:\n")
            for k, v in value.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")

    print(f"\nReport saved to {REPORT_DIR}/phase7_anomaly_detection.txt")


if __name__ == "__main__":
    main()
