# data_prep.py, Part 1: load the UCI CSV, parse timestamps, check missing
# values, resample 10-minute data to hourly means.

from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "energydata_complete.csv"
PROCESSED = ROOT / "data" / "appliance_hourly.csv"
URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
       "00374/energydata_complete.csv")

TARGET = "Appliances"


def download():
    "Fetch the raw CSV from UCI once; reuse the local copy afterwards."
    if not RAW_CSV.exists():
        print("downloading", URL)
        r = requests.get(URL, timeout=300,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        RAW_CSV.write_bytes(r.content)
    return RAW_CSV


def load_raw():
    "Load the raw 10-minute CSV with parsed timestamps."
    df = pd.read_csv(download())
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def missing_summary(df):
    "Missing values per column and any gaps in the 10-minute grid."
    na = df.isna().sum()
    full_grid = pd.date_range(df.index.min(), df.index.max(), freq="10min")
    gaps = len(full_grid) - len(df)
    return {"na_per_column": na[na > 0].to_dict(),
            "expected_rows": len(full_grid), "actual_rows": len(df),
            "missing_timestamps": gaps}


def to_hourly(df):
    "Resample to hourly means; interpolate any interior gaps in time."
    hourly = df.resample("h").mean()
    hourly = hourly.interpolate("time").dropna()
    return hourly


def prepare(save=True):
    "Full Part-1 preparation. Returns the hourly dataframe."
    df = load_raw()
    summary = missing_summary(df)
    hourly = to_hourly(df)
    if save:
        PROCESSED.parent.mkdir(parents=True, exist_ok=True)
        hourly.to_csv(PROCESSED)
    print("raw:", df.shape, "| hourly:", hourly.shape)
    print("missing:", summary)
    return hourly


if __name__ == "__main__":
    prepare()
