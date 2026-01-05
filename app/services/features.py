import pandas as pd
import numpy as np
from datetime import timedelta

REGRESSORS = ["T", "RH", "NO2(GT)"]

def build_future_df(fallback_csv_path: str, new_feats: dict) -> pd.DataFrame:
    """
    Construit un dataframe avec historique + 1 pas futur.
    new_feats doit contenir T, RH, NO2(GT).
    """
    hist = pd.read_csv(fallback_csv_path)
    hist["ds"] = pd.to_datetime(hist["ds"])
    hist = hist.sort_values("ds")

    # On garde un contexte suffisant (>= n_lags, je prends 48 pour être safe)
    context = hist[["ds", "y"] + REGRESSORS].tail(48).copy()

    # Next timestamp = dernière ds + 1 heure
    last_ds = context["ds"].iloc[-1]
    next_ds = last_ds + pd.Timedelta(hours=1)

    future = {
        "ds": next_ds,
        "y": np.nan,
        "T": float(new_feats["T"]),
        "RH": float(new_feats["RH"]),
        "NO2(GT)": float(new_feats["NO2(GT)"]) if new_feats["NO2(GT)"] is not None else context["NO2(GT)"].iloc[-1],
    }
    future_df = pd.concat([context, pd.DataFrame([future])], ignore_index=True)
    return future_df
