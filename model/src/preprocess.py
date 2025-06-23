import logging
import pandas as pd
import os
import glob
import re
import argparse
import boto3
import io

# ============================= Logging =============================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(SCRIPT_DIR, "preprocessing.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w"),
    ]
)

print(f"Logs are being stored in {LOG_FILE}")
logging.info(f"Log file location: {LOG_FILE}")


# ============================= Utilities =============================
def log_dataframe_metadata(df, df_name):
    separator = "=" * 80
    inner_separator = "-" * 80
    logging.info(f"\n{separator}")
    logging.info(f"DATAFRAME SUMMARY: {df_name}")
    logging.info(f"{separator}")
    logging.info(f"Shape: {df.shape}")
    logging.info(f"Columns: {', '.join(df.columns)}")
    logging.info(f"\n{inner_separator}")
    logging.info("Null Values:")
    for col, val in df.isnull().sum().items():
        logging.info(f"  {col:25}: {val}")
    logging.info(f"\n{inner_separator}")
    logging.info(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")
    logging.info(f"\n{inner_separator}")
    logging.info(f"Preview DataFrame Head:")
    logging.info(df.head(5).to_string(index=False))
    logging.info(f"\n{separator}")


def read_device_files(device_name, data_dir, aws_mode, s3_bucket):
    logging.info(f"Reading files for device: {device_name} (AWS Mode: {aws_mode})")

    if aws_mode:
        s3_client = boto3.client('s3')
        files = []
        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=s3_bucket, Prefix="raw/"):
            for obj in page.get('Contents', []):
                if obj['Key'].endswith(".xlsx"):
                    files.append(obj['Key'])

        logging.info(f"All files in S3 bucket: {files}")
        pattern = f"\\({re.escape(device_name)}\\)"
        matched_files = [f for f in files if re.search(pattern, os.path.basename(f))]
    else:
        all_files = glob.glob(os.path.join(data_dir, "*.xlsx"))
        pattern = f"\\({re.escape(device_name)}\\)"
        matched_files = [f for f in all_files if re.search(pattern, os.path.basename(f))]

    if not matched_files:
        raise FileNotFoundError(f"No files found for device: {device_name}")

    logging.info(f"Files matched for {device_name}: {matched_files}")

    feature_files = [f for f in matched_files if not re.search(r"\bRating\b", f, re.IGNORECASE)]
    rating_files = [f for f in matched_files if re.search(r"\bRating\b", f, re.IGNORECASE)]

    if not rating_files:
        raise FileNotFoundError(f"No Rating file found for device: {device_name}")

    # ================================= Added Location Extraction =================================
    feature_dfs = []
    if aws_mode:
        for file in feature_files:
            filename = os.path.basename(file)
            location = filename.split(")")[-1].replace(".xlsx", "").strip()  # Extract location from file name
            obj = s3_client.get_object(Bucket=s3_bucket, Key=file)
            df = pd.read_excel(io.BytesIO(obj['Body'].read()))
            df["location"] = location  # Add location column
            feature_dfs.append(df)
    else:
        for file in feature_files:
            filename = os.path.basename(file)
            location = filename.split(")")[-1].replace(".xlsx", "").strip()  # Extract location from file name
            df = pd.read_excel(file)
            df["location"] = location  # Add location column
            feature_dfs.append(df)

    features_df = pd.concat(feature_dfs, ignore_index=True)

    rating_dfs = []
    if aws_mode:
        for file in rating_files:
            obj = s3_client.get_object(Bucket=s3_bucket, Key=file)
            df = pd.read_excel(io.BytesIO(obj['Body'].read()))
            rating_dfs.append(df)
    else:
        for file in rating_files:
            df = pd.read_excel(file)
            rating_dfs.append(df)

    rating_df = pd.concat(rating_dfs, ignore_index=True)

    return features_df, rating_df


def filter_datetime_range(df, start, end):
    logging.info(f"Filtering DataFrame from {start} to {end}.")
    mask = (df['datetime'] >= start) & (df['datetime'] <= end)
    return df.loc[mask]


# ============================= Main Processing =============================
def main(device, data_dir, output_dir, aws_mode, s3_bucket):
    logging.info("Starting preprocessing...")

    features_df, rating_df = read_device_files(
        device_name=device,
        data_dir=data_dir,
        aws_mode=aws_mode,
        s3_bucket=s3_bucket
    )

    # Log DataFrame summaries
    log_dataframe_metadata(features_df, "Features DataFrame")
    log_dataframe_metadata(rating_df, "Rating DataFrame")

    # Combine Date and Time columns into datetime
    features_df['datetime'] = pd.to_datetime(
        features_df['Date'].astype(str) + ' ' + features_df['Time'].astype(str), errors='coerce'
    )
    rating_df['datetime'] = pd.to_datetime(
        rating_df['Date'].astype(str) + ' ' + rating_df['Time'].astype(str), errors='coerce'
    )

    # Drop unnecessary columns
    features_df.drop(columns=['Date', 'Time'], inplace=True)
    rating_df.drop(columns=['Date', 'Time'], inplace=True)

    # ================================= Modified Pivot Operations =================================
    pivot_features = features_df.pivot_table(
        index=["datetime", "location"],  # Include "location" in index
        columns="Measurement",
        values="data"
    ).reset_index()

    pivot_ratings = rating_df.pivot_table(
        index=["datetime"],  # Ratings do not use "location"
        columns="Metric",
        values="Rating"
    ).reset_index()

    # Ensure overlapping datetime range between feature and ratings data
    overlap_start = max(pivot_features["datetime"].min(), pivot_ratings["datetime"].min())
    overlap_end = min(pivot_features["datetime"].max(), pivot_ratings["datetime"].max())

    pivot_features = filter_datetime_range(pivot_features, overlap_start, overlap_end)
    pivot_ratings = filter_datetime_range(pivot_ratings, overlap_start, overlap_end)

    # ================================= Modified Merge Operation =================================
    merged_df = pd.merge_asof(
        pivot_features.sort_values(["datetime", "location"]),
        pivot_ratings.sort_values(["datetime"]),
        on="datetime",
        direction="nearest"
    )

    log_dataframe_metadata(merged_df, "Merged DataFrame")

    if aws_mode:
        s3_client = boto3.client('s3')
        output_key = f"processed/{device}_merged.csv"
        csv_buffer = io.StringIO()
        merged_df.to_csv(csv_buffer, index=False)
        s3_client.put_object(Bucket=s3_bucket, Key=output_key, Body=csv_buffer.getvalue())
        logging.info(f"Merged CSV written to s3://{s3_bucket}/{output_key}")
    else:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{device}_merged.csv")
        merged_df.to_csv(output_path, index=False)
        logging.info(f"Merged CSV written to local path: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess sensor and ratings data for a device.")
    parser.add_argument("--device", required=True, help="Device name, e.g., '8#Belt Conveyer'")
    parser.add_argument("--data_dir", default="data/raw",
                        help="Directory containing raw .xlsx data files (default: data/raw)")
    parser.add_argument("--output_dir", default="data/preprocessed",
                        help="Directory to save processed data (default: data/preprocessed)")
    parser.add_argument("--aws", action="store_true", help="Use AWS S3 for input/output operations")
    parser.add_argument("--s3_bucket", default="brilliant-automation-capstone",
                        help="S3 bucket name (default: brilliant-automation-capstone)")

    args = parser.parse_args()
    main(device=args.device, data_dir=args.data_dir, output_dir=args.output_dir, aws_mode=args.aws,
         s3_bucket=args.s3_bucket)