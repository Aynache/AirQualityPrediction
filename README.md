# 🌍 Air Quality – Temps réel + Prédiction de CO (NeuralProphet)

Application complète (API + Frontend) permettant :
1) d’afficher des mesures **temps réel** météo/qualité de l’air via **WeatherAPI**  
2) de produire une **prédiction de CO** à court terme avec un modèle **NeuralProphet**

> Projet conçu pour fonctionner en **local** (déploiement cloud non inclus dans cette archive).

---

## ✅ Fonctionnalités

### API FastAPI (`app/`)
- `GET /health` : vérifie que l’API tourne
- `GET /realtime` : récupère les données temps réel pour :
  - Montréal
  - Trois-Rivières  
  (ou une seule ville via `?city=Montreal` / `?city=Trois-Rivieres`)
- `POST /predict` : prédit le CO à partir de features météo et NO₂

### Frontend Streamlit (`streamlit_app.py`)
- Interface interactive connectée au backend FastAPI
- Affiche :
  - la qualité de l’air (polluants + index AQI)
  - la météo
  - les features utilisées pour la prédiction
  - la prédiction CO via l’endpoint `/predict`

---

## 🧠 Modèle utilisé

- Modèle : **NeuralProphet** (chargé depuis `models/neuralprophet_co_deployable.pkl`)
- Régressseurs utilisés par le modèle :
  - `T` (température °C)
  - `RH` (humidité %)
  - `NO2(GT)` (NO₂ en µg/m³ si disponible)
- Stratégie de prédiction :
  - on reconstruit un contexte historique (48 heures par défaut) depuis
    `models/airquality_fallback_final.csv`
  - on remappe ce contexte pour qu’il se termine “maintenant” (heure courante)
  - on ajoute une ligne future à `t+1h` avec les features temps réel
  - NeuralProphet produit `yhat1` (prédiction)
  - la sortie est **clippée** dans `[0, 15]` (bornes de sécurité)

---

## 📦 Structure du projet

```text
projet/
├── app/
│   ├── main.py                 # API FastAPI
│   ├── model_loader.py         # chargement + warm (mini-fit) NeuralProphet
│   ├── schemas.py              # modèles Pydantic (request/response)
│   └── services/
│       ├── weatherapi.py       # appel WeatherAPI + parsing (météo + air quality)
│       └── features.py         # construction du df_future pour predict
├── models/
│   ├── neuralprophet_co_deployable.pkl
│   ├── train_df_deploy.csv
│   ├── airquality_fallback_final.csv
│   └── train_df_deploy.csv
├── requirements.txt
└── streamlit_app.py


🔐 Configuration du fichier .env

L’application nécessite une clé API WeatherAPI pour fonctionner.

1️⃣ Créer le fichier .env

À la racine du projet :

touch .env

2️⃣ Ajouter la clé API

Ouvre le fichier .env et ajoute :

WEATHER_API_KEY=VOTRE_CLE_API_ICI


👉 La clé peut être obtenue sur : https://www.weatherapi.com/

⚠️ Important

Le fichier .env ne doit pas être partagé publiquement.

Il est chargé automatiquement par l’application via python-dotenv.

🐍 Environnement virtuel et exécution

Il est recommandé d’utiliser un environnement virtuel Python.

1️⃣ Créer un environnement virtuel
python3 -m venv venv

2️⃣ Activer l’environnement virtuel
Linux / macOS
source venv/bin/activate

Windows (PowerShell)
venv\Scripts\activate


Une fois activé, le terminal affiche (venv).

3️⃣ Installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt

4️⃣ Lancer le backend (FastAPI)
uvicorn app.main:app --reload


Par défaut :

API : http://127.0.0.1:8000

Documentation interactive : http://127.0.0.1:8000/docs

5️⃣ Lancer le frontend (Streamlit)

Dans un second terminal (avec le même environnement virtuel activé) :

streamlit run streamlit_app.py


Interface accessible sur :

http://localhost:8501

6️⃣ Architecture d’exécution recommandée
Terminal 1 → FastAPI (backend)
Terminal 2 → Streamlit (frontend)


Le frontend communique automatiquement avec l’API FastAPI.