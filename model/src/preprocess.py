import logging
import pandas as pd
import os
import glob
import re
import argparse
import boto3
import io

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(SCRIPT_DIR, "preprocessing.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w"),
    ]
)

print(f"Logs are being stored in: {LOG_FILE}")
logging.info(f"Log file location: {LOG_FILE}")

def log_dataframe_metadata(df, df_name):
    """Logs basic metadata about a DataFrame"""
    separator = "=" * 80
    inner_separator = "-" * 80

    logging.info(f"\n{separator}")
    logging.info(f"DATAFRAME SUMMARY: {df_name}")
    logging.info(f"{separator}")
    logging.info(f"Shape: {df.shape}")
    logging.info(f"Columns: {', '.join(df.columns)}")
    logging.info(f"\n{inner_separator}")
    logging.info("Null values:")

    null_vals = df.isnull().sum()
    for col, val in null_vals.items():
        logging.info(f"  {col:25}: {val}")

    memory_usage = df.memory_usage(deep=True).sum() / (1024 ** 2)
    logging.info(f"\n{inner_separator}")
    logging.info(f"Memory Usage: {memory_usage:.2f} MB")

    logging.info(f"\n{inner_separator}")
    logging.info(f"Preview Dataframe Head:\n")
    preview = df.head(5).to_string(index=False).split('\n')
    for line in preview:
        logging.info(f"  {line}")

    logging.info(f"{separator}\n")

def read_device_files(device_name, data_dir="../Data/raw", aws_mode=False, s3_bucket='brilliant-automation-capstone'):
    """
    Reads all Excel files for a given device name, ignoring any prefix like date ranges.
    If aws_mode is True, reads files from S3 bucket brilliant-automation-capstone/raw instead of local directory.
    """
    logging.info(f"Reading files for device: {device_name} (AWS mode: {aws_mode})")

    if aws_mode:
        logging.info(f"Connecting to AWS S3 bucket: {s3_bucket}")
        s3 = boto3.client('s3')
        # Always use the fixed S3 prefix
        prefix = 'raw/'
        all_files = []
        paginator = s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=s3_bucket, Prefix=prefix):
            for obj in page.get('Contents', []):
                if obj['Key'].endswith('.xlsx'):
                    all_files.append(obj['Key'])
        logging.info(f"Files found in bucket: {all_files}")
        pattern = f"\\({re.escape(device_name)}\\)"
        matched_files = [f for f in all_files if re.search(pattern, os.path.basename(f))]
        if not matched_files:
            logging.error(f"No files found for device: {device_name} in S3 bucket {s3_bucket}")
            raise FileNotFoundError(f"No files found for device: {device_name} in S3 bucket {s3_bucket}")
        rating_file = None
        feature_files = []
        for f in matched_files:
            filename = os.path.basename(f)
            if re.search(r"\bRating\b", filename, re.IGNORECASE):
                rating_file = f
            else:
                feature_files.append(f)
        if not rating_file:
            logging.error(f"No Rating file found for device: {device_name} in S3 bucket {s3_bucket}")
            raise FileNotFoundError(f"No Rating file found for device: {device_name} in S3 bucket {s3_bucket}")

        logging.info(f"Rating file: {rating_file}")
        logging.info(f"Feature files: {feature_files}")

        feature_dfs = []
        for file in feature_files:
            filename = os.path.basename(file)
            location = filename.split(")")[-1].replace(".xlsx", "").strip()
            obj = s3.get_object(Bucket=s3_bucket, Key=file)
            df = pd.read_excel(io.BytesIO(obj['Body'].read()))
            df["location"] = location
            feature_dfs.append(df)
        features_df = pd.concat(feature_dfs, ignore_index=True)
        obj = s3.get_object(Bucket=s3_bucket, Key=rating_file)
        rating_df = pd.read_excel(io.BytesIO(obj['Body'].read()))
        return features_df, rating_df
    else:
        logging.info(f"Reading files from local directory: {data_dir}")
        # List all .xlsx files in the folder
        all_files = glob.glob(os.path.join(data_dir, "*.xlsx"))
        # Match files that contain the device name in parentheses
        logging.info(f"Files found in directory: {all_files}")
        pattern = f"\\({re.escape(device_name)}\\)"
        matched_files = [f for f in all_files if re.search(pattern, os.path.basename(f))]
        if not matched_files:
            logging.error(f"No files found for device: {device_name}")
            raise FileNotFoundError(f"No files found for device: {device_name}")
        # Separate rating file from location files
        rating_file = None
        feature_files = []
        for f in matched_files:
            filename = os.path.basename(f)
            if re.search(r"\bRating\b", filename, re.IGNORECASE):
                rating_file = f
            else:
                feature_files.append(f)
        if not rating_file:
            logging.error(f"No Rating file found for device: {device_name}")
            raise FileNotFoundError(f"No Rating file found for device: {device_name}")
        # Read and combine all feature files

        logging.info(f"Rating file: {rating_file}")
        logging.info(f"Feature files: {feature_files}")

        feature_dfs = []
        for file in feature_files:
            filename = os.path.basename(file)
            location = filename.split(")")[-1].replace(".xlsx", "").strip()
            df = pd.read_excel(file)
            df["location"] = location
            feature_dfs.append(df)
        features_df = pd.concat(feature_dfs, ignore_index=True)
        rating_df = pd.read_excel(rating_file)
        return features_df, rating_df

def filter_datetime_range(df, col, start, end):
    """Filters a DataFrame to include rows where col is between start and end."""
    logging.info(f"Filtering DataFrame by date range: {start} to {end} on column: {col}")
    result_df = df[(df[col] >= start) & (df[col] <= end)]
    logging.info(f"Shape after filtering: {result_df.shape}")
    return result_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess sensor and ratings data for a device."
    )
    parser.add_argument("--device", required=True, help="Device name, e.g. '8#Belt Conveyer'")
    parser.add_argument("--data_dir", default="Data/raw", help="Directory containing raw .xlsx data files (default: Data/raw)")
    parser.add_argument("--output_dir", default="Data/process", help="Directory to save processed data (default: Data/process)")
    parser.add_argument("--aws", action="store_true", help="Read/write data from/to S3 bucket instead of local directory")

    args = parser.parse_args()
    device = args.device
    data_dir = args.data_dir
    output_dir = args.output_dir
    aws_mode = args.aws
    s3_bucket = 'brilliant-automation-capstone'

    logging.info("Starting preprocessing...")
    if not aws_mode:
        os.makedirs(output_dir, exist_ok=True)

    features_df, rating_df = read_device_files(device, data_dir=data_dir, aws_mode=aws_mode)
    log_dataframe_metadata(features_df, "Sensor DataFrame")
    log_dataframe_metadata(rating_df, "Ratings DataFrame")

    features_df["datetime"] = pd.to_datetime(
        features_df["Date"].astype(str).str.strip() + " " +
        features_df["Time"].astype(str).str.strip(),
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce"
    )

    rating_df["datetime"] = pd.to_datetime(
        rating_df["Date"].astype(str).str.strip() + " " +
        rating_df["Time"].astype(str).str.strip(),
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce"
    )

    features_df.drop(columns=["Date", "Time", "id"], inplace=True)
    rating_df.drop(columns=["Date", "Time"], inplace=True)
    
    pivot_features_df = features_df.pivot_table(
        index=["datetime", "location"],
        columns="Measurement",
        values="data"
    ).reset_index()

    pivot_rating_df = rating_df.pivot_table(
        index=["datetime", "Device"],
        columns="Metric",
        values="Rating"
    ).reset_index()

    logging.info("Pivoting completed.")
    log_dataframe_metadata(pivot_features_df, "Pivoted Sensor DataFrame")
    log_dataframe_metadata(pivot_rating_df, "Pivoted Ratings DataFrame")

    pivot_features_df['Temperature'] = pivot_features_df.groupby('location')['Temperature'].ffill()

    pivot_features_df = pivot_features_df.sort_values("datetime")
    pivot_rating_df = pivot_rating_df.sort_values("datetime")

    # Ensure overlapping datetime range between feature and ratings data
    min_feature_time = pivot_features_df["datetime"].min()
    max_feature_time = pivot_features_df["datetime"].max()

    min_rating_time = pivot_rating_df["datetime"].min()
    max_rating_time = pivot_rating_df["datetime"].max()

    overlap_start = max(min_feature_time, min_rating_time)
    overlap_end = min(max_feature_time, max_rating_time)

    pivot_features_df = filter_datetime_range(pivot_features_df, "datetime", overlap_start, overlap_end)
    pivot_rating_df = filter_datetime_range(pivot_rating_df, "datetime", overlap_start, overlap_end)

    merged_df = pd.merge_asof(
        pivot_features_df,
        pivot_rating_df,
        on="datetime",
        direction="forward"
    )

    logging.info("Merging completed.")
    log_dataframe_metadata(merged_df, "Merged DataFrame")

    if aws_mode:
        # Write merged_df to S3 as CSV
        s3 = boto3.client('s3')
        output_key = f"process/{device}_merged.csv"
        csv_buffer = io.StringIO()
        merged_df.to_csv(csv_buffer, index=False)
        s3.put_object(Bucket=s3_bucket, Key=output_key, Body=csv_buffer.getvalue())
        logging.info(f"Merged CSV written to s3://{s3_bucket}/{output_key}")
    else:
        output_path = os.path.join(output_dir, f"{device}_merged.csv")
        merged_df.to_csv(output_path, index=False)
        logging.info(f"Merged CSV written to local path: {output_path}")



