import os
import requests

def fetch_weather(city: str) -> dict:
    key = os.getenv("WEATHER_API_KEY")
    if not key:
        raise RuntimeError("WEATHER_API_KEY manquant dans .env")

    url = "https://api.weatherapi.com/v1/current.json"
    params = {"key": key, "q": city, "aqi": "yes"}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()

def extract_features(payload: dict) -> dict:
    cur = payload["current"]
    aq = cur.get("air_quality", {})

    # WeatherAPI: humidity (%) = RH, temp_c (°C) = T
    T = cur.get("temp_c")
    RH = cur.get("humidity")
    NO2 = aq.get("no2")  # WeatherAPI no2 est en µg/m³

    if T is None or RH is None:
        raise ValueError("temp_c ou humidity manquant dans la réponse WeatherAPI.")
    # NO2 peut être None selon l'API/plan

    return {"T": float(T), "RH": float(RH), "NO2(GT)": float(NO2) if NO2 is not None else None}
