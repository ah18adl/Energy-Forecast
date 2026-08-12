# evaluate.py, Part 8: metrics (MAE, RMSE, MAPE, MASE), forecast plots and
# error diagnostics. Forecasts are produced per 24-hour fold; metrics are
# computed on the concatenated test-period predictions.

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "outputs" / "figures"
MET = ROOT / "outputs" / "metrics"
FC = ROOT / "outputs" / "forecasts"


def mase_scale(y_train, seasonality=24):
    """In-sample seasonal naive MAE, used as the MASE denominator.

    Computing the scale on the training data (not the test data) is what makes
    MASE comparable across models and across series (Hyndman and Koehler,
    2006). A MASE below 1 means the model beats a seasonal naive forecast.
    """
    v = y_train.values
    return np.mean(np.abs(v[seasonality:] - v[:-seasonality]))


def metrics(y_true, y_pred, scale):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ae = np.abs(y_true - y_pred)
    return {"MAE": ae.mean(),
            "RMSE": np.sqrt(((y_true - y_pred) ** 2).mean()),
            "MAPE_pct": float(np.mean(ae / np.abs(y_true)) * 100),
            "MASE": ae.mean() / scale,
            "Bias": float((y_pred - y_true).mean())}


def metrics_table(forecast_df, y_train, out_name="model_comparison.csv"):
    "One row of metrics per model column (everything except 'actual')."
    scale = mase_scale(y_train)
    rows = []
    for col in forecast_df.columns:
        if col == "actual":
            continue
        valid = forecast_df[col].notna() & forecast_df["actual"].notna()
        m = metrics(forecast_df.loc[valid, "actual"],
                    forecast_df.loc[valid, col], scale)
        rows.append({"model": col, **{k: round(v, 3) for k, v in m.items()}})
    out = pd.DataFrame(rows).sort_values("MASE").reset_index(drop=True)
    MET.mkdir(parents=True, exist_ok=True)
    out.to_csv(MET / out_name, index=False)
    return out


def forecast_plot(train, forecast_df, cols, fname, title, ci=None):
    "Last week of training plus test actuals and chosen forecasts."
    fig, ax = plt.subplots(figsize=(11, 4))
    train.tail(7 * 24).plot(ax=ax, label="train (last week)", lw=0.8,
                            color="#888888")
    forecast_df["actual"].plot(ax=ax, label="actual", lw=1.6, color="black")
    for col in cols:
        forecast_df[col].plot(ax=ax, label=col, lw=1.1, alpha=0.9)
    if ci is not None:
        ax.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1],
                        alpha=0.2, label="95% interval")
    ax.set_title(title)
    ax.set_xlabel("date"); ax.set_ylabel("appliance energy (Wh)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=200)
    plt.close(fig)


def final_forecast_plot(train, forecast_df, fname="forecasts_all_models.png"):
    """Summary figure with every model on one axis.

    Top panel: the full 14 day test period with all nine forecasts plus the
    actuals, so the overall spread between models is visible.
    Bottom panel: the first three days only, where individual 24 hour folds
    can be told apart. Benchmarks are drawn thin and dashed so the four
    fitted models stand out against them.
    """
    models = [c for c in forecast_df.columns if c != "actual"]
    bench = ["mean", "naive", "seasonal_naive_daily",
             "seasonal_naive_weekly", "drift"]
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    for ax, data, title in [
            (axes[0], forecast_df, "All models, full 14 day test period"),
            (axes[1], forecast_df.iloc[:72],
             "All models, first three forecast days")]:
        if ax is axes[0]:                     # context before the test window
            train.tail(3 * 24).plot(ax=ax, color="#999999", lw=0.8,
                                    label="train (last 3 days)")
        data["actual"].plot(ax=ax, color="black", lw=2.0, label="actual",
                            zorder=5)
        for col in models:
            is_bench = col in bench
            data[col].plot(ax=ax, lw=0.9 if is_bench else 1.4,
                           ls=":" if is_bench else "-",
                           alpha=0.65 if is_bench else 0.95, label=col)
        ax.set_title(title)
        ax.set_xlabel("date")
        ax.set_ylabel("appliance energy (Wh)")
    axes[0].legend(fontsize=7, ncol=5, loc="upper left")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / fname, dpi=200)
    plt.close(fig)


def error_diagnostics(forecast_df, fname="error_diagnostics.png"):
    "Error by hour of day and error distribution for each model."
    models = [c for c in forecast_df.columns if c != "actual"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for col in models:
        err = forecast_df[col] - forecast_df["actual"]
        err.groupby(err.index.hour).apply(
            lambda e: e.abs().mean()).plot(ax=axes[0], label=col, alpha=0.8)
        axes[1].hist(err.dropna(), bins=40, histtype="step", label=col)
    axes[0].set_title("MAE by hour of day")
    axes[0].set_xlabel("hour of day"); axes[0].set_ylabel("MAE (Wh)")
    axes[1].set_title("Error distribution")
    axes[1].set_xlabel("forecast error (Wh)"); axes[1].set_ylabel("count")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=200)
    plt.close(fig)
