from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

from app.schemas import PredictRequest, PredictResponse
from app.model_loader import load_and_warm_model
from app.services.weatherapi import fetch_weather, extract_features
from app.services.features import build_future_df

load_dotenv()

MODEL_PATH = "models/neuralprophet_co_deployable.pkl"
TRAIN_CSV  = "models/train_df_deploy.csv"
FALLBACK   = "models/airquality_fallback_final.csv"

app = FastAPI(title="Air Quality CO Predictor")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # 1) Features: soit fournis, soit récupérés via WeatherAPI
    try:
        feats = {}
        if req.temp_c is not None and req.rh is not None:
            feats["T"] = float(req.temp_c)
            feats["RH"] = float(req.rh)
            feats["NO2(GT)"] = float(req.no2_ugm3) if req.no2_ugm3 is not None else None
        else:
            payload = fetch_weather(req.city)
            feats = extract_features(payload)

        # 2) Build df_future (historique fallback + 1 pas futur)
        df_future = build_future_df(FALLBACK, feats)

        # 3) Load model + warm (mini-fit)
        m = load_and_warm_model(MODEL_PATH, TRAIN_CSV)

        # 4) Predict
        fc = m.predict(df_future)
        yhat = float(fc["yhat1"].iloc[-1])
        ds = str(fc["ds"].iloc[-1])

        return PredictResponse(city=req.city, ds=ds, yhat1=yhat, inputs=feats)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
