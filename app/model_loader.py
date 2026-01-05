import joblib
import pandas as pd
from functools import lru_cache

REGRESSORS = ["T", "RH", "NO2(GT)"]

@lru_cache(maxsize=1)
def load_and_warm_model(model_path: str, train_csv_path: str):
    m = joblib.load(model_path)

    train_df = pd.read_csv(train_csv_path)
    train_df["ds"] = pd.to_datetime(train_df["ds"])

    # mini-fit pour remettre fitted=True (évite LR finder)
    m.fit(
        train_df,
        freq="h",
        epochs=1,
        learning_rate=1e-2,
        progress="off",
        minimal=True
    )
    return m
