# benchmarks.py, Part 3: mean, naive, daily/weekly seasonal naive and drift
# forecasts, all with a 24-hour horizon.

import numpy as np
import pandas as pd

DAILY = 24
WEEKLY = 168


def mean_fc(history, horizon):
    "Flat forecast at the historical mean."
    return np.full(horizon, history.mean())


def naive_fc(history, horizon):
    "Flat forecast at the last observed value."
    return np.full(horizon, history.iloc[-1])


def seasonal_naive_fc(history, horizon, seasonality):
    """Repeat the last full seasonal cycle.

    seasonality=24 gives "same hour yesterday", seasonality=168 gives "same
    hour last week". Values are appended to the history as they are produced,
    so a horizon longer than one cycle keeps repeating correctly.
    """
    vals = list(history.values)
    out = []
    for _ in range(horizon):
        out.append(vals[-seasonality])
        vals.append(out[-1])
    return np.array(out)


def drift_fc(history, horizon):
    """Extrapolate the straight line joining the first and last observation.

    This is the standard drift benchmark: the naive forecast plus the average
    change per period seen over the whole history.
    """
    slope = (history.iloc[-1] - history.iloc[0]) / (len(history) - 1)
    return history.iloc[-1] + slope * np.arange(1, horizon + 1)


def all_benchmarks(history, horizon):
    "Dict of name -> forecast array for one origin."
    return {
        "mean": mean_fc(history, horizon),
        "naive": naive_fc(history, horizon),
        "seasonal_naive_daily": seasonal_naive_fc(history, horizon, DAILY),
        "seasonal_naive_weekly": seasonal_naive_fc(history, horizon, WEEKLY),
        "drift": drift_fc(history, horizon),
    }
