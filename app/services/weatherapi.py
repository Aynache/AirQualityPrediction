import os
import requests

WEATHERAPI_SOURCE_NAME = "WeatherAPI"


def fetch_weather(q: str) -> dict:
    """
    Appelle WeatherAPI current.json.

    `q` peut être:
    - un nom de ville ("Montreal")
    - ou une coordonnée "lat,lon" (recommandé pour éviter ambiguïtés)
    """
    key = os.getenv("WEATHER_API_KEY")
    if not key:
        raise RuntimeError("WEATHER_API_KEY manquant dans .env")

    url = "https://api.weatherapi.com/v1/current.json"
    params = {"key": key, "q": q, "aqi": "yes"}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def extract_features(payload: dict) -> dict:
    """Features minimales utilisées par ton modèle (T, RH, NO2)."""
    cur = payload["current"]
    aq = cur.get("air_quality", {})

    # WeatherAPI: humidity (%) = RH, temp_c (°C) = T
    T = cur.get("temp_c")
    RH = cur.get("humidity")
    NO2 = aq.get("no2")  # WeatherAPI no2 est en µg/m³ (souvent)

    if T is None or RH is None:
        raise ValueError("temp_c ou humidity manquant dans la réponse WeatherAPI.")

    return {
        "T": float(T),
        "RH": float(RH),
        "NO2(GT)": float(NO2) if NO2 is not None else None
    }

from app.services.weatherapi import WEATHERAPI_SOURCE_NAME  # si besoin (sinon garde ta constante)


def extract_realtime(payload: dict, city_fallback: str) -> dict:
    loc = payload.get("location", {})
    cur = payload.get("current", {})
    aq = cur.get("air_quality", {}) or {}

    city = (loc.get("name") or city_fallback).strip()
    lat = float(loc.get("lat")) if loc.get("lat") is not None else float("nan")
    lon = float(loc.get("lon")) if loc.get("lon") is not None else float("nan")

    # Timestamp : last_updated est souvent plus "mesure", localtime = heure locale
    ts = str(cur.get("last_updated") or loc.get("localtime") or "")

    # --------- (A) current_air_quality ---------
    pollutants_keys = {
        "co": "CO",
        "no2": "NO2",
        "o3": "O3",
        "so2": "SO2",
        "pm2_5": "PM2.5",
        "pm10": "PM10",
    }

    pollutants = {}
    availability = {}
    for k, label in pollutants_keys.items():
        v = aq.get(k)
        pollutants[label] = float(v) if v is not None else None
        availability[label] = v is not None

    aqi = {
        "us_epa_index": float(aq.get("us-epa-index")) if aq.get("us-epa-index") is not None else None,
        "gb_defra_index": float(aq.get("gb-defra-index")) if aq.get("gb-defra-index") is not None else None,
    }
    availability["us_epa_index"] = aqi["us_epa_index"] is not None
    availability["gb_defra_index"] = aqi["gb_defra_index"] is not None

    current_air_quality = {
        "pollutants_ugm3": pollutants,
        "aqi": aqi,
        "availability": availability,
    }

    # --------- (B) current_weather ---------
    current_weather = {
        "temp_c": float(cur["temp_c"]) if cur.get("temp_c") is not None else None,
        "humidity": float(cur["humidity"]) if cur.get("humidity") is not None else None,
        "wind_kph": float(cur["wind_kph"]) if cur.get("wind_kph") is not None else None,
        "wind_dir": cur.get("wind_dir"),
        "pressure_mb": float(cur["pressure_mb"]) if cur.get("pressure_mb") is not None else None,
        "precip_mm": float(cur["precip_mm"]) if cur.get("precip_mm") is not None else None,
        "cloud": float(cur["cloud"]) if cur.get("cloud") is not None else None,
        "feelslike_c": float(cur["feelslike_c"]) if cur.get("feelslike_c") is not None else None,
        "vis_km": float(cur["vis_km"]) if cur.get("vis_km") is not None else None,
    }

    # --------- (C) features_used_for_prediction ---------
    # Ce sont EXACTEMENT les features que tu utilises pour predict (T, RH, NO2(GT))
    features_used = {
        "T": current_weather["temp_c"],
        "RH": current_weather["humidity"],
        "NO2(GT)": float(aq["no2"]) if aq.get("no2") is not None else None,
    }

    return {
        "city": city,
        "lat": lat,
        "lon": lon,
        "ts": ts,
        "source": WEATHERAPI_SOURCE_NAME,
        "current_air_quality": current_air_quality,
        "current_weather": current_weather,
        "features_used_for_prediction": features_used,
        "raw": None,  # mets payload si tu veux debug
    }
