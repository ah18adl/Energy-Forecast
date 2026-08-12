# test_pipeline.py, sanity checks for the forecasting pipeline.
# Run with:  python -m pytest tests -q
# These guard the two things most likely to break silently: the shape of the
# prepared data, and leakage of test information into the features.

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import benchmarks
import evaluate
import features


@pytest.fixture(scope="module")
def hourly():
    "The prepared hourly table, skipped if the pipeline has not been run."
    path = ROOT / "data" / "appliance_hourly.csv"
    if not path.exists():
        pytest.skip("run run_pipeline.py first")
    return pd.read_csv(path, index_col=0, parse_dates=True)


def test_hourly_index_is_complete(hourly):
    "No gaps or duplicates in the hourly index."
    assert hourly.index.is_monotonic_increasing
    assert not hourly.index.duplicated().any()
    expected = pd.date_range(hourly.index.min(), hourly.index.max(), freq="h")
    assert len(hourly) == len(expected)


def test_no_missing_values(hourly):
    assert not hourly.isna().any().any()


def test_lag_features_do_not_leak(hourly):
    "lag_k at time t must equal the target at t minus k, never the target at t."
    table = features.make_table(hourly)
    y = hourly["Appliances"]
    for lag in (1, 24, 168):
        col = table[f"lag_{lag}"]
        expected = y.shift(lag).loc[col.index]
        assert np.allclose(col.values, expected.values)


def test_rolling_features_exclude_current_hour(hourly):
    "Rolling statistics are shifted, so they cannot contain the current value."
    table = features.make_table(hourly)
    y = hourly["Appliances"]
    expected = y.shift(1).rolling(3).mean().loc[table.index]
    assert np.allclose(table["roll_mean_3"].values, expected.values)


def test_strict_table_has_no_contemporaneous_covariates(hourly):
    "In strict mode every sensor and weather column is lagged."
    table = features.make_table(hourly, strict=True)
    for col in features.SENSOR_COLS + features.WEATHER_COLS:
        assert col not in table.columns
        assert f"{col}_lag24" in table.columns


def test_seasonal_naive_repeats_the_previous_cycle():
    "A daily seasonal naive forecast repeats the last 24 observed values."
    idx = pd.date_range("2016-01-01", periods=48, freq="h")
    y = pd.Series(np.arange(48, dtype=float), index=idx)
    fc = benchmarks.seasonal_naive_fc(y, horizon=24, seasonality=24)
    assert np.allclose(fc, np.arange(24, 48))


def test_mase_scale_matches_manual_calculation():
    "MASE denominator is the in-sample seasonal naive MAE."
    y = pd.Series(np.arange(100, dtype=float))
    scale = evaluate.mase_scale(y, seasonality=24)
    assert scale == pytest.approx(24.0)


def test_metrics_are_zero_for_a_perfect_forecast():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    m = evaluate.metrics(y, y, scale=1.0)
    assert m["MAE"] == 0 and m["RMSE"] == 0 and m["Bias"] == 0
