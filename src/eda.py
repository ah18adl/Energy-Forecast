# eda.py, Part 1: initial plots, seasonal decomposition, ACF/PACF and
# stationarity tests (ADF + KPSS).

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "outputs" / "figures"
MET = ROOT / "outputs" / "metrics"


def overview_plot(y):
    "Full series plus a two-week zoom to show the daily pattern."
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.6))
    y.plot(ax=axes[0], lw=0.5)
    axes[0].set_title("Hourly appliance energy use, full series")
    axes[0].set_ylabel("appliance energy (Wh)")
    axes[0].set_xlabel("date")
    y.iloc[:14 * 24].plot(ax=axes[1], lw=1)
    axes[1].set_title("First two weeks (daily cycle visible)")
    axes[1].set_ylabel("appliance energy (Wh)")
    axes[1].set_xlabel("date")
    fig.tight_layout()
    fig.savefig(FIG / "series_overview.png", dpi=200)
    plt.close(fig)


def profile_plots(y):
    "Mean use by hour of day and by day of week."
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
    y.groupby(y.index.hour).mean().plot(ax=axes[0], marker="o")
    axes[0].set_title("Mean use by hour of day")
    axes[0].set_xlabel("hour of day")
    axes[0].set_ylabel("mean appliance energy (Wh)")
    y.groupby(y.index.dayofweek).mean().plot(ax=axes[1], marker="o")
    axes[1].set_title("Mean use by day of week (0=Mon)")
    axes[1].set_xlabel("day of week")
    axes[1].set_ylabel("mean appliance energy (Wh)")
    fig.tight_layout()
    fig.savefig(FIG / "profiles.png", dpi=200)
    plt.close(fig)


def decomposition_plot(y):
    "Additive decomposition with daily period."
    dec = seasonal_decompose(y, period=24, model="additive")
    fig = dec.plot()
    fig.set_size_inches(11, 6)
    for ax in fig.axes:
        for line in ax.lines:
            line.set_linewidth(0.5)
    fig.tight_layout()
    fig.savefig(FIG / "decomposition.png", dpi=200)
    plt.close(fig)
    seasonal_strength = max(0, 1 - dec.resid.var() /
                            (dec.resid + dec.seasonal).var())
    return seasonal_strength


def acf_pacf_plot(y, name, lags=72):
    "ACF and PACF side by side."
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
    plot_acf(y.dropna(), ax=axes[0], lags=lags)
    plot_pacf(y.dropna(), ax=axes[1], lags=min(lags, 40), method="ywm")
    fig.tight_layout()
    fig.savefig(FIG / f"acf_pacf_{name}.png", dpi=200)
    plt.close(fig)


def stationarity_tests(series_dict):
    "ADF and KPSS for each named series; returns a dataframe."
    rows = []
    for name, s in series_dict.items():
        s = s.dropna()
        adf_stat, adf_p = adfuller(s, autolag="AIC")[:2]
        kpss_stat, kpss_p = kpss(s, regression="c", nlags="auto")[:2]
        rows.append({"series": name,
                     "ADF_stat": round(adf_stat, 3), "ADF_p": round(adf_p, 4),
                     "KPSS_stat": round(kpss_stat, 3), "KPSS_p": round(kpss_p, 4),
                     "ADF_says": "stationary" if adf_p < 0.05 else "non-stationary",
                     "KPSS_says": "non-stationary" if kpss_p < 0.05 else "stationary"})
    out = pd.DataFrame(rows)
    MET.mkdir(parents=True, exist_ok=True)
    out.to_csv(MET / "stationarity_tests.csv", index=False)
    return out


def run(y):
    "All Part-1 EDA. Returns the stationarity table."
    FIG.mkdir(parents=True, exist_ok=True)
    overview_plot(y)
    profile_plots(y)
    strength = decomposition_plot(y)
    print(f"seasonal strength (daily): {strength:.3f}")
    # ACF and PACF on the raw series and on both differenced versions, so the
    # effect of differencing on the autocorrelation structure can be compared
    acf_pacf_plot(y, "raw")
    acf_pacf_plot(y.diff(), "first_diff")
    acf_pacf_plot(y.diff(24), "seasonal_diff")
    tests = stationarity_tests({
        "raw": y,
        "first_diff": y.diff(),
        "seasonal_diff_24": y.diff(24),
        "seasonal_plus_first_diff": y.diff(24).diff(),
    })
    print(tests.to_string(index=False))
    return tests


if __name__ == "__main__":
    import data_prep
    run(data_prep.prepare()[data_prep.TARGET])
