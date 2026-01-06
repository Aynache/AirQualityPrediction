import random
import numpy as np
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

from app.schemas import (
    PredictRequest,
    PredictResponse,
    RealtimeResponse,
    RealtimeCityResponse,
)
from app.model_loader import load_and_warm_model
from app.services.weatherapi import fetch_weather, extract_features, extract_realtime
from app.services.features import build_future_df
from app.schemas import RealtimeResponse, RealtimeCityResponse


load_dotenv()

MODEL_PATH = "models/neuralprophet_co_deployable.pkl"
TRAIN_CSV = "models/train_df_deploy.csv"
FALLBACK = "models/airquality_fallback_final.csv"

app = FastAPI(title="Air Quality CO Predictor")

# Villes attendues par l'énoncé (Montréal et Trois-Rivières)
# On utilise des coordonnées pour éviter les ambiguïtés de geocoding.
CITY_QUERIES = {
    "Montreal": "45.5017,-73.5673",
    "Trois-Rivieres": "46.3438,-72.5430",
}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/realtime", response_model=RealtimeResponse)
def realtime(city: Optional[str] = None):
    """Retourne les mesures *temps réel* de qualité de l'air.

    - Si `city` est fourni, on renvoie la ville demandée (si elle fait partie des villes attendues).
    - Sinon, on renvoie Montréal + Trois-Rivières.
    """
    try:
        cities: List[str]
        if city:
            city_key = city.strip()
            if city_key not in CITY_QUERIES:
                raise HTTPException(
                    status_code=400,
                    detail=f"city invalide. Valeurs acceptées: {list(CITY_QUERIES.keys())}",
                )
            cities = [city_key]
        else:
            cities = list(CITY_QUERIES.keys())

        out: List[RealtimeCityResponse] = []
        for c in cities:
            payload = fetch_weather(CITY_QUERIES[c])  # coordonnées => plus fiable
            normalized = extract_realtime(payload, city_fallback=c)
            out.append(RealtimeCityResponse(**normalized))

        return RealtimeResponse(cities=out)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        # 1) Features: soit fournies, soit récupérées via WeatherAPI
        feats = {}
        if req.temp_c is not None and req.rh is not None:
            feats["T"] = float(req.temp_c)
            feats["RH"] = float(req.rh)
            feats["NO2(GT)"] = float(req.no2_ugm3) if req.no2_ugm3 is not None else None
        else:
            payload = fetch_weather(req.city)
            feats = extract_features(payload)

            # Clipping simple pour éviter les valeurs hors-distribution
            feats["RH"] = float(np.clip(feats["RH"], 0.0, 100.0))
            if feats.get("NO2(GT)") is not None:
                feats["NO2(GT)"] = float(np.clip(feats["NO2(GT)"], 0.0, 500.0))
            feats["T"] = float(np.clip(feats["T"], -50.0, 50.0))

        # 2) Build df_future (historique fallback + 1 pas futur)
        df_future = build_future_df(FALLBACK, feats)

        # 3) Load model
        m = load_and_warm_model(MODEL_PATH, TRAIN_CSV)

        # 4) Predict
        fc = m.predict(df_future)
        raw_yhat = float(fc["yhat1"].iloc[-1])

        # ✅ Clip physique + clip "dataset-realistic"
        raw_yhat *= -1 if raw_yhat < 0 else 1
        yhat = float(np.clip(raw_yhat, 0.0, 15.0))
        ds = str(fc["ds"].iloc[-1])

        return PredictResponse(city=req.city, ds=ds, yhat1=yhat, inputs=feats)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
