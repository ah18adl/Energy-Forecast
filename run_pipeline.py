# run_pipeline.py, end-to-end runnable pipeline for the whole assignment.
#
#   python run_pipeline.py            full run (AIC grid takes ~10 min)
#   python run_pipeline.py --fast     skip the AIC grid, reuse saved results
#
# Parts covered: 1 data prep and EDA, 2 problem definition, 3 benchmarks,
# 4 SARIMAX with grid search and CIs, 5 covariates, 6 XGBoost,
# 7 Chronos (reads outputs/forecasts/chronos_forecasts.csv if present),
# 8 evaluation table, plots and diagnostics.

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

import benchmarks
import data_prep
import eda
import evaluate
import features
import ml_model
import sarima

TARGET = data_prep.TARGET
TEST_DAYS = 14
TEST_STEPS = TEST_DAYS * 24
HORIZON = 24

FC = ROOT / "outputs" / "forecasts"
MET = ROOT / "outputs" / "metrics"


def rolling_benchmarks(y, test_index):
    "Re-forecast every 24h using only data up to each fold origin."
    out = {name: [] for name in
           ["mean", "naive", "seasonal_naive_daily",
            "seasonal_naive_weekly", "drift"]}
    for start in range(0, len(test_index) - HORIZON + 1, HORIZON):
        window = test_index[start:start + HORIZON]
        history = y.loc[:window[0]].iloc[:-1]
        fc = benchmarks.all_benchmarks(history, HORIZON)
        for name, values in fc.items():
            out[name].append(pd.Series(values, index=window))
    return {name: pd.concat(parts) for name, parts in out.items()}


def main(fast=False):
    # Part 1: data preparation and EDA
    hourly = data_prep.prepare()
    y = hourly[TARGET]
    if not fast:
        eda.run(y)

    # Part 2: forecasting problem
    train, test = y.iloc[:-TEST_STEPS], y.iloc[-TEST_STEPS:]
    print(f"\ntarget: {TARGET} (Wh, hourly)")
    print(f"train: {train.index.min()} to {train.index.max()} ({len(train)} h)")
    print(f"test:  {test.index.min()} to {test.index.max()} ({len(test)} h)")
    print(f"horizon: {HORIZON} h, re-forecast every {HORIZON} h "
          f"({len(test) // HORIZON} folds)")

    forecasts = {}

    # Part 3: benchmarks
    forecasts.update(rolling_benchmarks(y, test.index))

    # Part 4: SARIMAX
    grid_path = MET / "sarima_grid_aic.csv"
    if fast and grid_path.exists():
        grid = pd.read_csv(grid_path)
    else:
        grid = sarima.grid_search(train)
    order = sarima.best_order(grid)
    print(f"\nbest ARIMA order by AIC (converged fits): {order}")
    print(f"final model: SARIMAX{order}{sarima.SEASONAL_ORDER}")

    exog = [c for c in sarima.EXOG_COLS if c in hourly.columns]
    hourly.index.freq = "h"
    y.index.freq = "h"
    X = hourly[exog]
    params_path = MET / "sarimax_params.csv"
    if fast and params_path.exists():
        params = pd.read_csv(params_path, index_col=0).iloc[:, 0].values
        fit = sarima.refit_from_params(train, order, params,
                                       X.iloc[:-TEST_STEPS])
    else:
        fit = sarima.fit_final(train, order, X.iloc[:-TEST_STEPS])
        pd.Series(fit.params).to_csv(params_path)
    sarima.residual_diagnostics(fit)
    sx, sx_ci = sarima.rolling_forecast(fit, y, test.index, X, HORIZON)
    forecasts["sarimax"] = sx
    FC.mkdir(parents=True, exist_ok=True)
    sx_ci.to_csv(FC / "sarimax_intervals.csv")

    # Parts 5 and 6: covariates and XGBoost, conditional and strict variants
    for label, strict in [("xgboost_conditional", False),
                          ("xgboost_strict", True)]:
        table = features.make_table(hourly, strict=strict)
        model, cols = ml_model.fit(table, TEST_STEPS)
        idx = table.index[-TEST_STEPS:]
        pred = ml_model.recursive_forecast(model, cols, table, y, idx, HORIZON)
        forecasts[label] = pred.reindex(test.index)
        pred.to_frame(label).to_csv(FC / f"{label}.csv")
        if not strict:
            ml_model.importance_plot(model, cols)

    # Part 7: Chronos, if the local run has been done
    chronos_path = FC / "chronos_forecasts.csv"
    if chronos_path.exists():
        ch = pd.read_csv(chronos_path, index_col=0, parse_dates=True)
        forecasts["chronos"] = ch["chronos"].reindex(test.index)
        print("\nChronos forecasts loaded")
    else:
        print("\nChronos forecasts not found; run: python src/foundation.py")

    # Part 8: evaluation
    fdf = pd.DataFrame({"actual": test})
    for name, series in forecasts.items():
        fdf[name] = series.reindex(test.index)
    FC.mkdir(parents=True, exist_ok=True)
    fdf.to_csv(FC / "all_forecasts.csv")

    results = evaluate.metrics_table(fdf, train)
    print("\nmodel comparison:")
    print(results.to_string(index=False))

    bench_cols = [c for c in fdf.columns
                  if c.startswith(("mean", "naive", "seasonal", "drift"))]
    model_cols = [c for c in fdf.columns
                  if c in ("sarimax", "xgboost_conditional",
                           "xgboost_strict", "chronos")]
    # headline figure: every model and the actuals on one axis
    evaluate.final_forecast_plot(train, fdf)
    evaluate.forecast_plot(train, fdf, bench_cols, "forecasts_benchmarks.png",
                           "Benchmark forecasts, 14-day test period")
    evaluate.forecast_plot(train, fdf, model_cols, "forecasts_models.png",
                           "Model forecasts, 14-day test period")
    evaluate.forecast_plot(train, fdf.iloc[:HORIZON * 3],
                           model_cols, "forecasts_zoom.png",
                           "First three forecast days",
                           ci=sx_ci.iloc[:HORIZON * 3])
    evaluate.error_diagnostics(fdf[["actual"] + model_cols])
    return results, fdf


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true",
                    help="skip the AIC grid search and EDA re-run")
    main(**vars(ap.parse_args()))
