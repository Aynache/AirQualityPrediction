import pandas as pd
import numpy as np

REGRESSORS = ["T", "RH", "NO2(GT)"]

def build_future_df(fallback_csv_path: str, new_feats: dict, n_context: int = 48) -> pd.DataFrame:
    hist = pd.read_csv(fallback_csv_path)
    hist["ds"] = pd.to_datetime(hist["ds"])
    hist = hist.sort_values("ds")

    context = hist[["ds", "y"] + REGRESSORS].tail(n_context).copy()

    # Remapper les dates du contexte pour finir "maintenant" (heure courante arrondie)
    now = pd.Timestamp.now(tz="America/Toronto").floor("h").tz_localize(None)
    start = now - pd.Timedelta(hours=n_context - 1)
    context["ds"] = pd.date_range(start=start, periods=n_context, freq="h")

    # Ligne future (t+1h)
    future_ds = now + pd.Timedelta(hours=1)

    future = {
        "ds": future_ds,
        "y": np.nan,
        "T": float(new_feats["T"]),
        "RH": float(new_feats["RH"]),
        "NO2(GT)": float(new_feats["NO2(GT)"]) if new_feats["NO2(GT)"] is not None else float(context["NO2(GT)"].iloc[-1]),
    }

    df_future = pd.concat([context, pd.DataFrame([future])], ignore_index=True)
    return df_future
