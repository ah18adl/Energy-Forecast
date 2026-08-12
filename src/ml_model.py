# ml_model.py, Part 6: XGBoost on the engineered features, with proper
# 24-hour-ahead forecasting. Two variants:
#   conditional: contemporaneous sensor/weather covariates, i.e.
#                perfect covariate knowledge at the forecast origin
#   strict:      only information available at the forecast origin
# Forecasts are recursive within each 24 h window: lag features beyond the
# origin are filled with the model's own predictions, never with actuals.

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

import features
from features import TARGET, LAGS, WINDOWS

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "outputs" / "figures"

PARAMS = dict(n_estimators=600, learning_rate=0.03, max_depth=6,
              subsample=0.9, colsample_bytree=0.9, random_state=0,
              n_jobs=4)


def fit(table, test_steps):
    """Train XGBoost on everything before the test period.

    The split is positional rather than random because this is a time series:
    a random split would let the model learn from the future.
    """
    train = table.iloc[:-test_steps]
    X = train.drop(columns=TARGET)
    model = XGBRegressor(**PARAMS)
    model.fit(X, train[TARGET])
    return model, list(X.columns)


def recursive_forecast(model, cols, table, y, test_index, horizon=24):
    "24h-ahead recursive forecasts, one day per fold; in-window lags use predicted values."
    y_work = y.copy()
    preds = []
    for start in range(0, len(test_index) - horizon + 1, horizon):
        window = test_index[start:start + horizon]
        # reset history to actuals up to the fold origin
        y_hist = y.loc[:window[0]].iloc[:-1].copy()
        for ts in window:
            row = table.loc[ts].copy()
            hist = y_hist
            for lag in LAGS:
                row[f"lag_{lag}"] = hist.iloc[-lag]
            shifted = hist
            for w in WINDOWS:
                row[f"roll_mean_{w}"] = shifted.iloc[-w:].mean()
                row[f"roll_std_{w}"] = shifted.iloc[-w:].std()
            pred = float(model.predict(
                row[cols].to_frame().T.astype(float))[0])
            preds.append((ts, pred))
            y_hist = pd.concat([y_hist, pd.Series([pred], index=[ts])])
    s = pd.Series(dict(preds))
    s.index = pd.DatetimeIndex(s.index)
    return s


def importance_plot(model, cols, fname="xgb_importance.png", top=20):
    "Top feature importances by gain."
    imp = pd.Series(model.feature_importances_, index=cols)
    imp = imp.sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(7, 5))
    imp.plot.barh(ax=ax)
    ax.set_title("XGBoost feature importance (top 20)")
    ax.set_xlabel("importance (share of total gain)")
    ax.set_ylabel("feature")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / fname, dpi=200)
    plt.close(fig)
    return imp
