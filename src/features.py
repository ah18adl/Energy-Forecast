# features.py, Part 5: sensor, weather and time-based covariates plus
# lagged and rolling target features. All lag/rolling features use only
# past values (shifted), so nothing leaks from the future.

import numpy as np
import pandas as pd

TARGET = "Appliances"

# covariates taken directly from the dataset
SENSOR_COLS = ["T1", "RH_1", "T2", "RH_2", "T3", "RH_3", "lights"]
WEATHER_COLS = ["T_out", "RH_out", "Press_mm_hg", "Windspeed",
                "Visibility", "Tdewpoint"]

LAGS = [1, 2, 3, 6, 12, 24, 48, 168]
WINDOWS = [3, 6, 12, 24, 168]


def add_time_features(df):
    """Hour-of-day and day-of-week features, cyclically encoded.

    The sine and cosine pairs are needed so that hour 23 and hour 0 are close
    together in feature space; the raw integer hour would make them maximally
    far apart.
    """
    out = df.copy()
    out["hour"] = out.index.hour
    out["dayofweek"] = out.index.dayofweek
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7)
    return out


def add_lag_features(df, target=TARGET):
    """Lagged target values and shifted rolling statistics.

    Every rolling window is shifted by one hour before it is computed, so a
    feature at time t can never contain the value being predicted at time t.
    The lag set spans short persistence (1 to 6 h), the daily cycle (24, 48 h)
    and the weekly cycle (168 h).
    """
    out = df.copy()
    for lag in LAGS:
        out[f"lag_{lag}"] = out[target].shift(lag)
    for w in WINDOWS:
        out[f"roll_mean_{w}"] = out[target].shift(1).rolling(w).mean()
        out[f"roll_std_{w}"] = out[target].shift(1).rolling(w).std()
    return out


def make_table(df, strict=False):
    "Supervised table. strict=True lags covariates 24h (known at origin); strict=False uses contemporaneous values (conditional forecast)."
    cols = [TARGET] + [c for c in SENSOR_COLS + WEATHER_COLS
                       if c in df.columns]
    out = add_time_features(df[cols])
    out = add_lag_features(out)
    if strict:
        for c in SENSOR_COLS + WEATHER_COLS:
            if c in out.columns:
                out[f"{c}_lag24"] = out[c].shift(24)
                out = out.drop(columns=c)
    return out.dropna()
