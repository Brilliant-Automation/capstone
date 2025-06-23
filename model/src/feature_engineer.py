#!/usr/bin/env python3
# feature_eng.py
#
# This script will:
#  1. Find all .json files under each “<date>#Belt Conveyer” folder inside data/voltage/
#  2. Read axisX/axisY from each JSON
#  3. Compute fs = 1 / (axisX[1] - axisX[0])
#  4. Treat axisY as the raw waveform
#  5. Compute DSP metrics (velocity_rms, crest_factor, etc.)
#  6. Save metrics to data/voltage/metrics_json.csv
#  7. Read merged ratings/features from data/processed/8#Belt Conveyer_merged.csv
#  8. Rename all 12 rating columns (append "_rating")
#  9. Bucket and summarize measurement features per rating/time window
# 10. Join JSON metrics + bucket summary on matching location & time
# 11. Save full features to data/processed/8#Belt Conveyer_full_features.csv
# ---------------------------------------------------------------------------- #

import os
import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats

# —————————— CONFIGURATION ——————————

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VOLTAGE_DIR = PROJECT_ROOT / "data" / "voltage"
PROCESS_DIR = PROJECT_ROOT / "data" / "preprocessed"

METRIC_COLS = [
    "velocity_rms",
    "crest_factor",
    "kurtosis_opt",
    "peak_value_opt",
    "rms_0_10hz",
    "rms_10_100hz",
    "rms_1_10khz",
    "rms_10_25khz",
    "peak_10_1000hz",
]

RATING_COLS = [
    "alignment_status",
    "bearing_lubrication",
    "crest_factor",
    "electromagnetic_status",
    "fit_condition",
    "kurtosis_opt",
    "rms_10_25khz",
    "rms_1_10khz",
    "rotor_balance_status",
    "rubbing_condition",
    "velocity_rms",
    "peak_value_opt",
]

SENSOR_MAP = {
    "67c29baa30e6dd385f031b30": "Motor Non-Drive End",
    "67c29baa30e6dd385f031b39": "Motor Drive End",
    "67c29baa30e6dd385f031b42": "Fan Free End",
    "67c29baa30e6dd385f031b4b": "Fan Inlet End",
    "67c29bab30e6dd385f031c98": "Motor Drive End",
    "67c29bab30e6dd385f031ca1": "Gearbox First Shaft Input End",
    "67c29bab30e6dd385f031caa": "Gear Reducer",
    "67c29bab30e6dd385f031c2c": "Motor Non-Drive End",
    "67c29bab30e6dd385f031c35": "Motor Drive End",
    "67c29bab30e6dd385f031c3e": "Gearbox First Shaft Input End",
    "67c29bab30e6dd385f031c47": "Left-side Bearing Housing of Gear Set",
    "67c29bab30e6dd385f031c50": "Right-side Bearing Housing of Gear Set",
}

FILENAME_PATTERN = re.compile(
    r"^\d{8} \d{6}_[0-9a-f]{24}_([0-9a-f]{24})_([A-Z]{2})\.json$"
)


# —————————— HELPER FUNCTIONS ——————————

def read_json_file(path: Path) -> (float, np.ndarray):
    """Read JSON file for axis data."""
    with open(path, 'r') as f:
        data = json.load(f)
    axis_x = np.array(data["axisX"], dtype=np.float64)
    axis_y = np.array(data["axisY"], dtype=np.float32)

    if len(axis_x) < 2:
        raise ValueError(f"Not enough points in axisX of {path.name}")

    dt = float(axis_x[1] - axis_x[0])
    if dt <= 0:
        raise ValueError(f"Invalid dt ({dt}) in {path.name}")

    return 1.0 / dt, axis_y


def band_rms(signal: np.ndarray, sampling_rate: float, f_low: float, f_high: float) -> float:
    """Compute RMS value of a signal within a frequency band."""
    n = signal.size
    freqs = np.fft.rfftfreq(n, d=1 / sampling_rate)
    fft_values = np.fft.rfft(signal - signal.mean())
    power = (np.abs(fft_values) ** 2) / n
    mask = (freqs >= f_low) & (freqs <= f_high)
    return np.sqrt(power[mask].sum()) if np.any(mask) else np.nan


def band_peak(signal: np.ndarray, sampling_rate: float, f_low: float, f_high: float) -> float:
    """Compute peak value of a signal within a frequency band."""
    n = signal.size
    fft_values = np.fft.rfft(signal - signal.mean())
    freqs = np.fft.rfftfreq(n, d=1 / sampling_rate)

    mask = (freqs >= f_low) & (freqs <= f_high)
    fft_filtered = np.zeros_like(fft_values)
    fft_filtered[mask] = fft_values[mask]
    return np.abs(np.fft.irfft(fft_filtered, n)).max()


def bucket_summary(df: pd.DataFrame, measurement_cols: list, rating_cols: list,
                   time_col='datetime', location_col='location', bucket_minutes=20) -> pd.DataFrame:
    """Summarize data into buckets."""
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values([location_col, time_col]).reset_index(drop=True)

    bucket_ids = [None] * len(df)
    counter = 0

    for loc, grp in df.groupby(location_col, sort=False):
        current_id = None
        start_time = None
        current_rt = None

        for idx in grp.index:
            rtup = tuple(df.at[idx, c] for c in rating_cols)
            t = df.at[idx, time_col]

            if current_id is None:
                counter += 1
                current_id = counter
                start_time = t
                current_rt = rtup
            elif rtup != current_rt or (t - start_time) >= pd.Timedelta(minutes=bucket_minutes):
                counter += 1
                current_id = counter
                start_time = t
                current_rt = rtup

            bucket_ids[idx] = current_id

    df['bucket_id'] = bucket_ids

    aggregations = {}
    for col in measurement_cols:
        aggregations[f"{col}_count"] = (col, 'count')
        aggregations[f"{col}_mean"] = (col, 'mean')
        aggregations[f"{col}_std"] = (col, 'std')
        aggregations[f"{col}_min"] = (col, 'min')
        aggregations[f"{col}_max"] = (col, 'max')
    for col in rating_cols:
        aggregations[col] = (col, 'first')

    aggregations['bucket_start'] = (time_col, 'min')
    aggregations['bucket_end'] = (time_col, 'max')

    return df.groupby([location_col, 'bucket_id']).agg(**aggregations).reset_index()


# —————————— FUNCTIONS FOR STEPS ——————————

def collect_metrics():
    """Step 1: Collect and compute metrics from JSON files."""
    records = []
    # Updated regex pattern for better timestamp detection
    regex_pattern = re.compile(r"(\d{8})[_ ]?(\d{6}).*\.json$")

    for subdir in VOLTAGE_DIR.iterdir():
        if not subdir.is_dir() or "Belt Conveyer" not in subdir.name:
            continue

        for root, _, files in os.walk(subdir):
            for fn in files:
                # Only proceed with JSON files
                if not fn.lower().endswith('.json'):
                    continue

                # Attempt to match the filename with the regex
                match = regex_pattern.search(fn)
                if not match:
                    print(f"Warning: Skipping file with unexpected name format: {fn}")
                    continue

                try:
                    # Extract timestamp from the filename
                    ts = datetime.strptime(match.group(1) + match.group(2), '%Y%m%d%H%M%S')
                except ValueError as e:
                    print(f"Error parsing timestamp for file {fn}: {e}")
                    continue

                # Build the record for the current file
                filepath_relative = str(Path(root) / fn).split(str(PROJECT_ROOT) + os.sep)[1]
                record = {"timestamp": ts, "filepath": filepath_relative}

                for col in METRIC_COLS:
                    record[col] = pd.NA

                # Append record
                records.append(record)

    # Create DataFrame for metrics
    records.sort(key=lambda r: r['timestamp'])  # Ensure records sorted by timestamp
    metrics_df = pd.DataFrame(records)

    # Extract sensor and wave information
    metrics_df['file_name'] = metrics_df['filepath'].apply(os.path.basename)
    extracted = metrics_df['file_name'].str.extract(FILENAME_PATTERN)
    metrics_df['sensor_id'] = extracted[0]
    metrics_df['wave_code'] = extracted[1]
    metrics_df['location'] = metrics_df['sensor_id'].map(SENSOR_MAP)
    metrics_df.drop(columns=['file_name'], inplace=True)

    # Compute DSP metrics for each file
    for i, row in metrics_df.iterrows():
        p = PROJECT_ROOT / row['filepath']
        try:
            fs, w = read_json_file(p)
        except Exception as e:
            print(f"Error processing file {p}: {e}")
            continue

        vrs = np.sqrt(np.mean(w ** 2))
        pk = np.max(np.abs(w))
        vals = [
            vrs,
            pk / vrs if vrs > 0 else np.nan,
            stats.kurtosis(w, fisher=False),
            pk,
            band_rms(w, fs, 0.1, 10),
            band_rms(w, fs, 10, 100),
            band_rms(w, fs, 1000, 10000),
            band_rms(w, fs, 10000, 25000),
            band_peak(w, fs, 10, 1000),
        ]
        for col, val in zip(METRIC_COLS, vals):
            metrics_df.at[i, col] = val

    # Save metrics to CSV
    out_metrics_file = VOLTAGE_DIR / "metrics_json.csv"
    metrics_df.to_csv(out_metrics_file, index=False)
    print(f"Metrics saved to {out_metrics_file}")

    return metrics_df


def load_merged_data():
    """Step 2: Load merged data and rename rating columns."""
    merged_path = PROCESS_DIR / "8#Belt Conveyer_merged.csv"
    merged_df = pd.read_csv(merged_path, parse_dates=["datetime"])
    merged_df.rename(columns={col: f"{col}_rating" for col in RATING_COLS}, inplace=True)
    return merged_df


def summarize_buckets(merged_df):
    """Step 3: Summarize data into buckets."""
    measurement_cols = ['High-Frequency Acceleration', 'Low-Frequency Acceleration Z', 'Temperature',
                        'Vibration Velocity Z']
    rating_cols_renamed = [f"{col}_rating" for col in RATING_COLS]

    summary_df = bucket_summary(
        merged_df, measurement_cols, rating_cols_renamed,
        time_col='datetime', location_col='location', bucket_minutes=20
    )
    summary_df.rename(columns={"bucket_start": "datetime"}, inplace=True)

    out_summary_file = PROCESS_DIR / "8#Belt Conveyer_bucket_summary.csv"
    summary_df.to_csv(out_summary_file, index=False)
    print(f"Bucket Summary saved to {out_summary_file}")

    return summary_df


def merge_features(metrics_df, summary_df):
    """Step 4: Merge metrics and summarized buckets."""
    # Ensure datetime is converted to proper format
    metrics_df['datetime'] = pd.to_datetime(metrics_df['timestamp'])
    summary_df['bucket_start'] = pd.to_datetime(summary_df['datetime'])
    summary_df['bucket_end'] = pd.to_datetime(summary_df['bucket_end'])

    # Create an IntervalIndex for bucket merging
    intervals = pd.IntervalIndex.from_arrays(
        summary_df['bucket_start'], summary_df['bucket_end'], closed='both'
    )
    summary_df['interval'] = intervals

    # Index summary dataframe by (location, interval)
    summary_indexed = summary_df.set_index(['location', 'interval'])

    combined_records = []

    for _, metric_row in metrics_df.iterrows():
        try:
            # Match a bucket row for the metric row based on location and datetime
            bucket_row = summary_indexed.loc[(metric_row['location'], metric_row['datetime'])]
        except KeyError:
            # Skip if no matching bucket is found
            continue

        # Combine the rows into a single record
        combined_record = {**metric_row.to_dict(), **bucket_row.to_dict()}
        combined_records.append(combined_record)

    # Create the final DataFrame
    full_features_df = pd.DataFrame(combined_records)

    # Drop unnecessary non-numeric columns or metadata
    full_features_df.drop(columns=['timestamp', 'bucket_start', 'bucket_end', 'interval', 'filepath'], inplace=True, errors='ignore')

    # Save the final dataset to CSV
    out_features_file = PROCESS_DIR / "8#Belt Conveyer_full_features.csv"
    full_features_df.to_csv(out_features_file, index=False)
    print(f"Full Features saved to {out_features_file}")

# —————————— MAIN EXECUTION ——————————

def main():
    metrics_df = collect_metrics()
    merged_df = load_merged_data()
    summary_df = summarize_buckets(merged_df)
    merge_features(metrics_df, summary_df)


if __name__ == "__main__":
    main()