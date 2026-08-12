# foundation.py, Part 7: Chronos (amazon/chronos-bolt-small) zero-shot
# forecasts, rolling 24 h windows over the test period, with quantile
# uncertainty bands.
#
# Run locally (downloads ~190 MB of weights on first use):
#   pip install chronos-forecasting torch
#   python src/foundation.py
# Writes outputs/forecasts/chronos_forecasts.csv

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FC = ROOT / "outputs" / "forecasts"

TARGET = "Appliances"
TEST_STEPS = 14 * 24
HORIZON = 24
CONTEXT = 512   # hours of history given to the model per window


def run():
    "Rolling Chronos forecasts over the last 14 days."
    import torch
    from chronos import BaseChronosPipeline

    pipe = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-bolt-small", device_map="cpu",
        torch_dtype=torch.float32)

    hourly = pd.read_csv(ROOT / "data" / "appliance_hourly.csv",
                         index_col=0, parse_dates=True)
    y = hourly[TARGET]
    test = y.iloc[-TEST_STEPS:]

    def quantiles(context):
        "Call predict_quantiles across chronos 1.x and 2.x signatures."
        levels = [0.1, 0.5, 0.9]
        try:                       # chronos 2.x: first argument is positional
            out = pipe.predict_quantiles(context, prediction_length=HORIZON,
                                         quantile_levels=levels)
        except TypeError:          # chronos 1.x: keyword 'context'
            out = pipe.predict_quantiles(context=context,
                                         prediction_length=HORIZON,
                                         quantile_levels=levels)
        q = out[0] if isinstance(out, tuple) else out
        return np.asarray(q.detach().cpu() if hasattr(q, "detach") else q)

    rows = []
    for start in range(0, len(test) - HORIZON + 1, HORIZON):
        origin = len(y) - TEST_STEPS + start
        context = torch.tensor(y.iloc[max(0, origin - CONTEXT):origin].values,
                               dtype=torch.float32)
        q = quantiles(context)
        if q.ndim == 2:            # some versions drop the batch dimension
            q = q[None, ...]
        window = test.index[start:start + HORIZON]
        for i, ts in enumerate(window):
            rows.append({"date": ts,
                         "chronos": float(q[0, i, 1]),
                         "chronos_lo": float(q[0, i, 0]),
                         "chronos_hi": float(q[0, i, 2])})
        print(f"window {start // HORIZON + 1}/14 done", flush=True)

    out = pd.DataFrame(rows).set_index("date")
    FC.mkdir(parents=True, exist_ok=True)
    out.to_csv(FC / "chronos_forecasts.csv")
    print("saved", FC / "chronos_forecasts.csv")


if __name__ == "__main__":
    run()
