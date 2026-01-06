from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any


class PredictRequest(BaseModel):
    city: str = Field(..., examples=["Montreal"])
    no2_ugm3: Optional[float] = None
    temp_c: Optional[float] = None
    rh: Optional[float] = None


class PredictResponse(BaseModel):
    city: str
    ds: str
    yhat1: float
    inputs: Dict[str, Any]


# ----------------------------
# REALTIME (nouvelle structure)
# ----------------------------

class RealtimeAirQuality(BaseModel):
    pollutants_ugm3: Dict[str, Optional[float]]
    aqi: Dict[str, Optional[float]] = Field(default_factory=dict)
    availability: Dict[str, bool] = Field(default_factory=dict)


class RealtimeWeather(BaseModel):
    temp_c: Optional[float] = None
    humidity: Optional[float] = None
    wind_kph: Optional[float] = None
    wind_dir: Optional[str] = None
    pressure_mb: Optional[float] = None
    precip_mm: Optional[float] = None
    cloud: Optional[float] = None
    feelslike_c: Optional[float] = None
    vis_km: Optional[float] = None


class FeaturesUsedForPrediction(BaseModel):
    # Ce que ton modèle utilise réellement
    T: Optional[float] = None
    RH: Optional[float] = None
    NO2_GT: Optional[float] = Field(None, alias="NO2(GT)")


class RealtimeCityResponse(BaseModel):
    city: str
    lat: float
    lon: float
    ts: str
    source: str

    current_air_quality: RealtimeAirQuality
    current_weather: RealtimeWeather
    features_used_for_prediction: FeaturesUsedForPrediction

    raw: Optional[Dict[str, Any]] = None


class RealtimeResponse(BaseModel):
    cities: List[RealtimeCityResponse]
