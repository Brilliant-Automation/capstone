import pandas as pd

def load_data(path="../../data/process/sample_belt_conveyer.csv"):
    df = pd.read_csv(path)

    df = df.rename(columns={"datetime": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    return df

def get_unique_locations(df):
    return sorted(df["location"].dropna().unique())
