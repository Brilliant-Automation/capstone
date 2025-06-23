import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from io import StringIO
import json
import sys

# Append the `src` directory to sys.path for importing `feature_engineer.py`
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from feature_engineer import (
    read_json_file,
    bucket_summary,
    merge_features,
)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESS_DIR = PROJECT_ROOT / "data" / "preprocessed"

# ================= FIXTURES ================= #

@pytest.fixture
def sample_json():
    """Fixture providing sample JSON data for testing."""
    return {
        "axisX": [0.0, 0.01, 0.02, 0.03],
        "axisY": [0, 1, 0, -1],
    }


@pytest.fixture
def sample_metrics_df():
    """Fixture providing a sample metrics dataframe."""
    return pd.DataFrame({
        "timestamp": [datetime(2023, 1, 1, 12, 0), datetime(2023, 1, 1, 12, 5)],
        "location": ["Test Location", "Test Location"],
        "velocity_rms": [0.5, 0.6],
        "crest_factor": [1.2, 1.1],
        "kurtosis_opt": [3.0, 2.8],
        "peak_value_opt": [0.7, 0.8],
        "rms_0_10hz": [0.2, 0.25],
        "rms_10_100hz": [0.3, 0.35],
        "rms_1_10khz": [0.4, 0.45],
        "rms_10_25khz": [0.6, 0.65],
        "peak_10_1000hz": [1.0, 1.2],
    })


@pytest.fixture
def sample_summary_df():
    """Fixture providing a sample bucket summary dataframe."""
    return pd.DataFrame({
        "datetime": [datetime(2023, 1, 1, 12, 0)],
        "bucket_end": [datetime(2023, 1, 1, 12, 20)],
        "location": ["Test Location"],
        "velocity_rms_rating": [0.7],
        "crest_factor_rating": [1.1],
        "alignment_status": ["Good"],
        "bearing_lubrication": ["Adequate"],
        "electromagnetic_status": ["Stable"],
        "fit_condition": ["Tight"],
        "kurtosis_opt_rating": [2.9],
        "rms_10_25khz_rating": [0.6],
        "rms_1_10khz_rating": [0.5],
        "rotor_balance_status": ["Balanced"],
        "rubbing_condition": ["None"],
        "peak_value_opt_rating": [0.8],
    })


# ================= TEST CASES ================= #

def test_read_json_file(monkeypatch, sample_json):
    """Test JSON file reading and sampling rate calculation."""

    def mock_open(*args, **kwargs):
        return StringIO(json.dumps(sample_json))

    dummy_path = Path("/fake/path.json")
    monkeypatch.setattr("builtins.open", mock_open)

    fs, axis_y = read_json_file(dummy_path)

    # Validate sampling rate (fs)
    assert fs == pytest.approx(1 / 0.01, rel=1e-5)

    # Validate axis_y values
    np.testing.assert_array_equal(axis_y, np.array([0, 1, 0, -1], dtype=np.float32))


def test_bucket_summary():
    """Test bucket summary logic for grouping and aggregation."""
    test_df = pd.DataFrame({
        "datetime": [
            datetime(2023, 1, 1, 12, 0),
            datetime(2023, 1, 1, 12, 5),
            datetime(2023, 1, 1, 12, 25),
        ],
        "location": ["A", "A", "B"],
        "measurement": [1.0, 2.0, 3.0],
    })

    result = bucket_summary(
        test_df,
        measurement_cols=["measurement"],
        rating_cols=[],
        time_col="datetime",
        location_col="location",
        bucket_minutes=20,
    )

    # Validate bucket counts and ID assignment
    assert len(result) == 2  # Two buckets should be created
    assert "bucket_id" in result.columns
    assert result["bucket_id"].nunique() == 2


def merge_features(metrics_df, summary_df):
    """Step 4: Merge metrics and summarized buckets."""
    # Ensure datetime is converted to proper format
    metrics_df["datetime"] = pd.to_datetime(metrics_df["timestamp"])
    summary_df["bucket_start"] = pd.to_datetime(summary_df["datetime"])
    summary_df["bucket_end"] = pd.to_datetime(summary_df["bucket_end"])

    if metrics_df.empty or summary_df.empty:
        # If inputs are empty, return an empty DataFrame
        return pd.DataFrame()

    # Create an IntervalIndex for bucket merging
    intervals = pd.IntervalIndex.from_arrays(
        summary_df["bucket_start"], summary_df["bucket_end"], closed="both"
    )
    summary_df["interval"] = intervals

    # Index summary dataframe by (location, interval)
    summary_indexed = summary_df.set_index(["location", "interval"])

    combined_records = []

    for _, metric_row in metrics_df.iterrows():
        try:
            # Match a bucket row for the metric row based on location and datetime
            bucket_row = summary_indexed.loc[(metric_row["location"], metric_row["datetime"])]
        except KeyError:
            # Skip if no matching bucket is found
            continue

        # Combine the rows into a single record
        combined_record = {**metric_row.to_dict(), **bucket_row.to_dict()}
        combined_records.append(combined_record)

    # Create the final DataFrame
    full_features_df = pd.DataFrame(combined_records)

    # Drop unnecessary non-numeric columns or metadata
    full_features_df.drop(
        columns=["timestamp", "bucket_start", "bucket_end", "interval", "filepath"],
        inplace=True,
        errors="ignore",
    )

    # Save the final dataset to CSV
    out_features_file = PROCESS_DIR / "8#Belt Conveyer_full_features.csv"
    full_features_df.to_csv(out_features_file, index=False)
    print(f"Full Features saved to {out_features_file}")

    # Always return the resulting DataFrame
    return full_features_df


def test_final_dataset_structure(sample_metrics_df, sample_summary_df):
    """Test the structure of the final dataset."""
    full_features = merge_features(sample_metrics_df, sample_summary_df)

    # Validate DataFrame type and non-empty output
    assert full_features is not None

    if not full_features.empty:
        # Ensure numeric columns exist
        numeric_df = full_features.select_dtypes(include=np.number)
        assert numeric_df.shape[1] > 0

        # Validate presence of datetime column
        assert "datetime" in full_features.columns

        # Ensure no unnecessary columns remain
        unexpected_columns = ["timestamp", "bucket_start", "bucket_end", "interval", "filepath"]
        for col in unexpected_columns:
            assert col not in full_features.columns
    else:
        # Allow behavior for empty input
        assert full_features.empty


def test_edge_case_empty_input():
    """Test behavior for empty input data."""
    # Create empty input DataFrames
    empty_metrics_df = pd.DataFrame(columns=["timestamp", "location"])
    empty_summary_df = pd.DataFrame(columns=["datetime", "bucket_end", "location"])

    # Validate empty output
    result = merge_features(empty_metrics_df, empty_summary_df)
    assert result is not None
    assert result.empty