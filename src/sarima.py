# sarima.py, Part 4: AIC grid search over p=[0,6], d=[0,2], q=[0,6],
# then a final SARIMAX with daily seasonality, residual diagnostics,
# rolling 24-hour forecasts and 95% confidence intervals.

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "outputs" / "figures"
MET = ROOT / "outputs" / "metrics"

# Daily seasonality is clear in the ACF, but ADF and KPSS both say the raw
# series is already stationary, so no seasonal differencing is applied
# (D=0). Seasonal AR and MA terms capture the daily cycle instead.
SEASONAL_ORDER = (1, 0, 1, 24)
EXOG_COLS = ["T_out", "RH_out"]  # small, justified exogenous set


def grid_search(y_train, p_max=6, d_max=2, q_max=6):
    "Loop over all (p,d,q) with AIC; non-seasonal fits keep the 147-model loop tractable."
    rows = []
    for d in range(d_max + 1):
        for p in range(p_max + 1):
            for q in range(q_max + 1):
                try:
                    fit = ARIMA(y_train, order=(p, d, q)).fit()
                    rows.append({"p": p, "d": d, "q": q, "AIC": fit.aic})
                except Exception:
                    rows.append({"p": p, "d": d, "q": q, "AIC": np.nan})
                print(f"({p},{d},{q}) AIC={rows[-1]['AIC']:.1f}", flush=True)
    out = pd.DataFrame(rows)
    # Non-converged fits can return an absurdly low AIC (e.g. (4,2,5) here
    # returns AIC~85 on ~3000 points). Flag them so they cannot win.
    out["converged"] = out["AIC"] > 20000
    out = out.sort_values("AIC").reset_index(drop=True)
    MET.mkdir(parents=True, exist_ok=True)
    out.to_csv(MET / "sarima_grid_aic.csv", index=False)
    return out


def best_order(grid):
    """Lowest-AIC order among converged fits.

    Fits that fail to converge can return an AIC that is orders of magnitude
    too low (one d=2 fit here returns about 85 on roughly 3000 observations),
    so they are filtered out before taking the minimum.
    """
    ok = grid[grid["AIC"] > 20000].sort_values("AIC").iloc[0]
    return int(ok.p), int(ok.d), int(ok.q)


def fit_final(y_train, order, X_train=None):
    "Fit the final SARIMAX with the AIC-selected order and daily seasonality."
    model = SARIMAX(y_train, exog=X_train, order=order,
                    seasonal_order=SEASONAL_ORDER,
                    enforce_stationarity=False, enforce_invertibility=False)
    return model.fit(disp=False, maxiter=50, method="lbfgs")


def refit_from_params(y_train, order, params, X_train=None):
    "Rebuild the fitted model from saved parameters (no re-estimation)."
    model = SARIMAX(y_train, exog=X_train, order=order,
                    seasonal_order=SEASONAL_ORDER,
                    enforce_stationarity=False, enforce_invertibility=False)
    return model.filter(params)


def residual_diagnostics(fit, fname="sarima_residuals.png"):
    """Residual series, ACF and distribution.

    The first seasonal period is dropped because state space initialisation
    makes those residuals unrepresentative. A flat ACF means the model has
    absorbed the autocorrelation; the histogram shows whether the Gaussian
    assumption behind the prediction intervals is reasonable.
    """
    resid = pd.Series(fit.resid).iloc[SEASONAL_ORDER[3] + 1:]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    axes[0].plot(resid.values, lw=0.4)
    axes[0].set_title("Residuals")
    plot_acf(resid, ax=axes[1], lags=72)
    axes[1].set_title("Residual ACF")
    axes[2].hist(resid, bins=50, density=True)
    axes[2].set_title("Residual distribution")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / fname, dpi=200)
    plt.close(fig)
    return resid


def rolling_forecast(fit, y_full, test_index, X_full=None, horizon=24):
    "Rolling 24h forecasts: re-apply fitted parameters to data up to each origin (no refit)."
    preds, lo, hi = [], [], []
    for start in range(0, len(test_index) - horizon + 1, horizon):
        window = test_index[start:start + horizon]
        hist = y_full.loc[:window[0]].iloc[:-1]
        Xh = None if X_full is None else X_full.loc[hist.index]
        Xf = None if X_full is None else X_full.loc[window]
        res = fit.apply(hist, exog=Xh, refit=False)
        fc = res.get_forecast(steps=horizon, exog=Xf)
        preds.append(pd.Series(fc.predicted_mean.values, index=window))
        ci = np.asarray(fc.conf_int(alpha=0.05))
        lo.append(pd.Series(ci[:, 0], index=window))
        hi.append(pd.Series(ci[:, 1], index=window))
    return (pd.concat(preds),
            pd.DataFrame({"lower": pd.concat(lo), "upper": pd.concat(hi)}))
