# stages.py, run one pipeline stage at a time and cache its forecasts.
# Used for development and to keep each step independently reproducible:
#   python src/stages.py benchmarks
#   python src/stages.py sarimax
#   python src/stages.py xgb_conditional
#   python src/stages.py xgb_strict
# run_pipeline.py performs the same steps in one go.

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

import benchmarks
import features
import ml_model
import sarima

TARGET = "Appliances"
TEST_STEPS = 14 * 24
HORIZON = 24
FC = ROOT / "outputs" / "forecasts"
MET = ROOT / "outputs" / "metrics"


def load():
    hourly = pd.read_csv(ROOT / "data" / "appliance_hourly.csv",
                         index_col=0, parse_dates=True)
    y = hourly[TARGET]
    return hourly, y, y.iloc[:-TEST_STEPS], y.iloc[-TEST_STEPS:]


def save(df, name):
    FC.mkdir(parents=True, exist_ok=True)
    df.to_csv(FC / name)
    print("saved", name)


def run_benchmarks():
    hourly, y, train, test = load()
    cols = {}
    for start in range(0, len(test) - HORIZON + 1, HORIZON):
        window = test.index[start:start + HORIZON]
        history = y.loc[:window[0]].iloc[:-1]
        for name, values in benchmarks.all_benchmarks(history, HORIZON).items():
            cols.setdefault(name, []).append(pd.Series(values, index=window))
    save(pd.DataFrame({k: pd.concat(v) for k, v in cols.items()}),
         "benchmarks.csv")


def run_sarimax(stage="all"):
    hourly, y, train, test = load()
    hourly.index.freq = "h"
    y.index.freq = "h"
    grid = pd.read_csv(MET / "sarima_grid_aic.csv")
    order = sarima.best_order(grid)
    print("order:", order, flush=True)
    exog = [c for c in sarima.EXOG_COLS if c in hourly.columns]
    X = hourly[exog]
    # cache only the fitted parameters; the state-space object is large
    cache = MET / "sarimax_params.csv"
    if cache.exists():
        params = pd.read_csv(cache, index_col=0).iloc[:, 0].values
        fit = sarima.refit_from_params(train, order, params,
                                       X.iloc[:-TEST_STEPS])
        print("loaded cached parameters", flush=True)
    else:
        fit = sarima.fit_final(train, order, X.iloc[:-TEST_STEPS])
        pd.Series(fit.params).to_csv(cache)
        print("fitted and cached parameters", flush=True)
    if stage in ("all", "diag"):
        sarima.residual_diagnostics(fit)
        print("diagnostics done", flush=True)
    if stage in ("all", "fc"):
        pred, ci = sarima.rolling_forecast(fit, y, test.index, X, HORIZON)
        save(pred.to_frame("sarimax"), "sarimax.csv")
        save(ci, "sarimax_intervals.csv")


def run_xgb(strict):
    hourly, y, train, test = load()
    table = features.make_table(hourly, strict=strict)
    model, cols = ml_model.fit(table, TEST_STEPS)
    idx = table.index[-TEST_STEPS:]
    pred = ml_model.recursive_forecast(model, cols, table, y, idx, HORIZON)
    name = "xgboost_strict" if strict else "xgboost_conditional"
    if not strict:
        ml_model.importance_plot(model, cols)
    save(pred.to_frame(name), f"{name}.csv")


if __name__ == "__main__":
    stage = sys.argv[1]
    {"benchmarks": run_benchmarks,
     "sarimax": run_sarimax,
     "sarimax_fit": lambda: run_sarimax("fit"),
     "sarimax_diag": lambda: run_sarimax("diag"),
     "sarimax_fc": lambda: run_sarimax("fc"),
     "xgb_conditional": lambda: run_xgb(False),
     "xgb_strict": lambda: run_xgb(True)}[stage]()
