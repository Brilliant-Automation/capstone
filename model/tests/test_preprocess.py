import pytest
import pandas as pd
import os
import sys

# Add the src directory to the module path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from preprocess import filter_datetime_range


def test_filter_datetime_range():
    """Test that filtering datetime range works correctly."""
    # Create a sample DataFrame
    data = {
        "datetime": pd.date_range("2023-01-01", periods=5, freq="D"),
        "value": [1, 2, 3, 4, 5],
    }
    df = pd.DataFrame(data)

    # Define the range to filter
    start_date = "2023-01-02"
    end_date = "2023-01-04"

    # Filter the DataFrame
    filtered_df = filter_datetime_range(df, start_date, end_date)
    print("Filtered DataFrame:\n", filtered_df)

    # Assertions
    assert len(filtered_df) == 3  # Only 3 rows should remain
    assert (filtered_df["datetime"] >= start_date).all()
    assert (filtered_df["datetime"] <= end_date).all()


def test_datetime_conversion():
    """Test the conversion of Date and Time columns into a single combined datetime."""
    data = {
        "Date": ["2023-01-01", "2023-01-02", None],
        "Time": ["12:00:00", "15:30:00", "23:59:59"],
    }
    df = pd.DataFrame(data)

    # Add a datetime column
    df["datetime"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str), errors="coerce"
    )

    print("Converted DataFrame with datetime:\n", df)

    # Assertions
    assert df["datetime"].iloc[0] == pd.Timestamp("2023-01-01 12:00:00")
    assert df["datetime"].iloc[1] == pd.Timestamp("2023-01-02 15:30:00")
    assert pd.isnull(df["datetime"].iloc[2])