import os
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

# Plotly optionnel (si pas installé, on retombe sur charts Streamlit)
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False


# ===========================
# CONFIG STREAMLIT
# ===========================

st.set_page_config(
    page_title="Qualité de l'Air (Temps réel) + Prédiction CO",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.25rem;
    }
    .subtle {
        color: #6b7280;
        font-size: 0.95rem;
        text-align: center;
        margin-bottom: 1.2rem;
    }
    .card {
        background: #f7f7fb;
        padding: 1rem 1.2rem;
        border-radius: 16px;
        border: 1px solid #e6e6ef;
    }
</style>
""",
    unsafe_allow_html=True,
)

DEFAULT_API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

# IMPORTANT: doit matcher exactement les valeurs acceptées par ton backend
VALID_CITIES = ["Montreal", "Trois-Rivieres"]


# ===========================
# HELPERS API
# ===========================

def api_get(api_base: str, path: str, params: Optional[dict] = None) -> dict:
    url = f"{api_base.rstrip('/')}{path}"
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def api_post(api_base: str, path: str, payload: dict) -> dict:
    url = f"{api_base.rstrip('/')}{path}"
    r = requests.post(url, json=payload, timeout=40)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=60)
def cached_realtime(api_base: str) -> dict:
    # /realtime renvoie déjà les 2 villes
    return api_get(api_base, "/realtime")

def fetch_realtime_for_city(realtime_payload: dict, city: str) -> Optional[dict]:
    for c in realtime_payload.get("cities", []):
        if c.get("city") == city:
            return c
    return None


# ===========================
# HELPERS UI
# ===========================

def aqi_label(us_epa_index: Optional[float]) -> Tuple[str, str]:
    """Mapping simple pour affichage (WeatherAPI us-epa-index: 1..6)."""
    if us_epa_index is None:
        return "Inconnu", "❓"
    m = {
        1: ("Bon", "😊"),
        2: ("Modéré", "😐"),
        3: ("Moyen", "😷"),
        4: ("Mauvais", "😨"),
        5: ("Très mauvais", "☠️"),
        6: ("Dangereux", "☠️"),
    }
    label, emoji = m.get(int(us_epa_index), ("Inconnu", "❓"))
    return label, emoji

def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None

def mk_pollutants_df(pollutants_ugm3: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for k, v in (pollutants_ugm3 or {}).items():
        rows.append({"Polluant": k, "Valeur (µg/m³)": safe_float(v)})
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Polluant")
    return df

def mk_weather_df(weather: Dict[str, Any]) -> pd.DataFrame:
    if not weather:
        return pd.DataFrame(columns=["Variable", "Valeur"])

    df = pd.DataFrame([weather]).T.reset_index()
    df.columns = ["Variable", "Valeur"]

    # ✅ Force Arrow compatibility (évite mix str/float)
    df["Valeur"] = df["Valeur"].apply(lambda x: "" if x is None else str(x))
    return df


def mk_features_df(feats: Dict[str, Any]) -> pd.DataFrame:
    if not feats:
        return pd.DataFrame(columns=["Feature", "Valeur"])

    df = pd.DataFrame([feats]).T.reset_index()
    df.columns = ["Feature", "Valeur"]

    # ✅ Force Arrow compatibility
    df["Valeur"] = df["Valeur"].apply(lambda x: "" if x is None else str(x))
    return df



# ===========================
# SESSION HISTORY (optionnel mais utile pour "tendance")
# ===========================

def init_history():
    if "history" not in st.session_state:
        st.session_state.history = []  # list of dict rows

def append_history(city: str, ts: str, pollutants: Dict[str, Any], aqi: Dict[str, Any]):
    init_history()
    row = {
        "city": city,
        "ts": ts,
        "CO": safe_float(pollutants.get("CO")),
        "NO2": safe_float(pollutants.get("NO2")),
        "PM2.5": safe_float(pollutants.get("PM2.5")),
        "PM10": safe_float(pollutants.get("PM10")),
        "O3": safe_float(pollutants.get("O3")),
        "SO2": safe_float(pollutants.get("SO2")),
        "us_epa_index": safe_float(aqi.get("us_epa_index")),
    }
    st.session_state.history.append(row)

def history_df() -> pd.DataFrame:
    init_history()
    if not st.session_state.history:
        return pd.DataFrame()
    df = pd.DataFrame(st.session_state.history)
    # convert ts for plotting if possible
    try:
        df["ts_dt"] = pd.to_datetime(df["ts"])
    except Exception:
        df["ts_dt"] = df["ts"]
    return df


# ===========================
# APP
# ===========================

def main():
    st.markdown('<div class="main-header">🌍 Qualité de l’air (temps réel) + Prédiction de CO</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtle">Frontend Streamlit connecté à ton backend FastAPI : <code>/realtime</code> et <code>/predict</code></div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        api_base = st.text_input(
            "URL de l’API FastAPI",
            value=DEFAULT_API_BASE,
            help="Ex: http://127.0.0.1:8000",
        ).rstrip("/")

        city = st.selectbox("Ville", VALID_CITIES, index=0)

        st.markdown("---")
        colA, colB = st.columns(2)
        with colA:
            if st.button("🔄 Rafraîchir", type="primary"):
                cached_realtime.clear()
                st.rerun()
        with colB:
            if st.button("🧹 Vider historique"):
                st.session_state.history = []
                st.rerun()

        st.markdown("---")
        st.caption("Cache /realtime : 60 secondes (pour éviter de spam l’API).")

        st.markdown("---")
        st.caption("Astuce terminal :")
        st.code("export API_BASE_URL=http://127.0.0.1:8000", language="bash")

    # Fetch realtime (both cities)
    try:
        with st.spinner("📡 Récupération des données temps réel..."):
            rt = cached_realtime(api_base)
    except Exception as e:
        st.error("❌ Impossible de joindre l’API (/realtime). Vérifie que FastAPI tourne.")
        st.exception(e)
        return

    selected = fetch_realtime_for_city(rt, city)
    if not selected:
        st.error("❌ Ville introuvable dans la réponse /realtime.")
        st.code(json.dumps(rt, indent=2, ensure_ascii=False), language="json")
        return

    # Extract blocks
    air = selected.get("current_air_quality", {}) or {}
    pollutants = (air.get("pollutants_ugm3", {}) or {})
    aqi = (air.get("aqi", {}) or {})
    availability = (air.get("availability", {}) or {})
    weather = selected.get("current_weather", {}) or {}
    feats = selected.get("features_used_for_prediction", {}) or {}

    us_epa = safe_float(aqi.get("us_epa_index"))
    aqi_text, aqi_emoji = aqi_label(us_epa)

    # Append to history (for trend chart)
    append_history(
        city=selected.get("city", city),
        ts=str(selected.get("ts", "")),
        pollutants=pollutants,
        aqi=aqi
    )

    # Header + metadata
    st.subheader(f"📍 Temps réel — {selected.get('city')}")
    st.caption(
        f"Source: {selected.get('source')} • Timestamp: {selected.get('ts')} • "
        f"Coord: ({selected.get('lat')}, {selected.get('lon')})"
    )

    # Top metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("AQI (US EPA)", f"{aqi_emoji} {us_epa if us_epa is not None else '—'}")
        st.caption(aqi_text)
    with c2:
        st.metric("CO (µg/m³)", f"{pollutants.get('CO') if pollutants.get('CO') is not None else '—'}")
    with c3:
        st.metric("NO₂ (µg/m³)", f"{pollutants.get('NO2') if pollutants.get('NO2') is not None else '—'}")
    with c4:
        st.metric("PM2.5 (µg/m³)", f"{pollutants.get('PM2.5') if pollutants.get('PM2.5') is not None else '—'}")

    # Availability note
    missing = [k for k, ok in availability.items() if ok is False]
    if missing:
        st.info(f"Certaines données ne sont pas disponibles via la source actuelle: {', '.join(missing)}")

    st.markdown("---")

    # Layout: Pollutants / Weather+Features
    left, right = st.columns([1.25, 1.0])

    with left:
        st.markdown("### 🌫️ Polluants (µg/m³)")
        pol_df = mk_pollutants_df(pollutants)
        st.dataframe(pol_df, width="stretch")

        if not pol_df.empty:
            if PLOTLY_OK:
                fig = px.bar(pol_df, x="Polluant", y="Valeur (µg/m³)", title="Concentrations (µg/m³)")
                st.plotly_chart(fig, width="stretch")
            else:
                st.bar_chart(pol_df.set_index("Polluant"), height=280)

        st.markdown("### 📈 Tendance (historique local UI)")
        hdf = history_df()
        if hdf.empty:
            st.caption("Aucune donnée historique (rafraîchis pour accumuler).")
        else:
            # chart only for selected city
            h_city = hdf[hdf["city"] == selected.get("city")]
            if not h_city.empty:
                if PLOTLY_OK:
                    fig2 = go.Figure()
                    if "ts_dt" in h_city.columns:
                        x = h_city["ts_dt"]
                    else:
                        x = h_city["ts"]
                    fig2.add_trace(go.Scatter(x=x, y=h_city["CO"], mode="lines+markers", name="CO (µg/m³)"))
                    fig2.add_trace(go.Scatter(x=x, y=h_city["NO2"], mode="lines+markers", name="NO2 (µg/m³)"))
                    fig2.update_layout(
                        title="CO & NO2 (depuis les rafraîchissements)",
                        xaxis_title="Temps",
                        yaxis_title="µg/m³",
                        hovermode="x unified",
                        height=330,
                    )
                    st.plotly_chart(fig2, width="stretch")
                else:
                    st.line_chart(h_city.set_index("ts_dt")[["CO", "NO2"]], height=280)

    with right:
        st.markdown("### 🌤️ Météo (actuelle)")
        w_df = mk_weather_df(weather)
        st.dataframe(w_df, width="stretch")

        st.markdown("### 🧠 Features utilisées pour la prédiction")
        f_df = mk_features_df(feats)
        st.dataframe(f_df, width="stretch")

        st.markdown("### 🧩 Comparaison rapide (Montréal vs Trois-Rivières)")
        # Use rt payload to build a tiny comparison table
        comp_rows = []
        for city_block in rt.get("cities", []):
            a = (city_block.get("current_air_quality", {}) or {})
            p = (a.get("pollutants_ugm3", {}) or {})
            aq = (a.get("aqi", {}) or {})
            comp_rows.append({
                "city": city_block.get("city"),
                "ts": city_block.get("ts"),
                "CO (µg/m³)": safe_float(p.get("CO")),
                "NO2 (µg/m³)": safe_float(p.get("NO2")),
                "PM2.5 (µg/m³)": safe_float(p.get("PM2.5")),
                "AQI (US EPA)": safe_float(aq.get("us_epa_index")),
            })
        comp_df = pd.DataFrame(comp_rows)
        st.dataframe(comp_df, width="stretch")

    st.markdown("---")

    # Prediction section
    st.subheader("🔮 Prédiction CO (via le backend FastAPI)")

    st.caption("Le frontend n’exécute aucun modèle : il appelle uniquement l’endpoint `/predict`.")

    btn_col, info_col = st.columns([0.25, 0.75])
    with btn_col:
        do_pred = st.button("📈 Lancer la prédiction", type="primary")
    with info_col:
        st.info("Démo facile : montre `/realtime` (temps réel) puis clique sur 'Lancer la prédiction'.")

    if do_pred:
        try:
            with st.spinner("🧠 Appel à /predict..."):
                pred = api_post(api_base, "/predict", {"city": city})
        except Exception as e:
            st.error("❌ Erreur lors de l’appel à `/predict`.")
            st.exception(e)
            return

        # Ton API renvoie: city, ds, yhat1, inputs
        st.success("✅ Prédiction reçue.")
        p1, p2, p3 = st.columns(3)
        with p1:
            st.metric("Ville", pred.get("city", city))
        with p2:
            st.metric("Timestamp prédit (ds)", pred.get("ds", "—"))
        with p3:
            st.metric("CO prédit (valeur API)", pred.get("yhat1", "—"))

        with st.expander("Voir le JSON de prédiction"):
            st.code(json.dumps(pred, indent=2, ensure_ascii=False), language="json")

    st.markdown("---")

    # Downloads
    st.subheader("💾 Télécharger les données")
    colD1, colD2, colD3 = st.columns(3)

    with colD1:
        st.download_button(
            "📥 Télécharger /realtime (JSON)",
            data=json.dumps(rt, indent=2, ensure_ascii=False),
            file_name=f"realtime_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )

    with colD2:
        merged = {
            "city": selected.get("city"),
            "ts": selected.get("ts"),
            "source": selected.get("source"),
            "lat": selected.get("lat"),
            "lon": selected.get("lon"),
            "current_air_quality": selected.get("current_air_quality"),
            "current_weather": selected.get("current_weather"),
            "features_used_for_prediction": selected.get("features_used_for_prediction"),
        }
        st.download_button(
            "📥 Ville sélectionnée (JSON)",
            data=json.dumps(merged, indent=2, ensure_ascii=False),
            file_name=f"{city}_details_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )

    with colD3:
        hdf = history_df()
        if hdf.empty:
            st.download_button(
                "📥 Historique UI (CSV)",
                data="",
                file_name="history_empty.csv",
                mime="text/csv",
                disabled=True
            )
        else:
            st.download_button(
                "📥 Historique UI (CSV)",
                data=hdf.to_csv(index=False),
                file_name=f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

    st.caption(f"Dernière mise à jour UI : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
