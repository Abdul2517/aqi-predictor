"""
app.py
-------
STEP 4 of the AQI Predictor project (Phase F: Web App / Dashboard).

WHY this design:
    Loads the LATEST registered model for each horizon (day1/day2/day3)
    from the Hopsworks Model Registry -- not a hardcoded specific version.
    This matches the project's automation spirit: the daily training
    pipeline keeps re-registering fresh models as new data comes in, so the
    dashboard should always reflect whatever is currently "best" per the
    automated pipeline, not a version frozen at build time.

WHY it handles BOTH tabular and LSTM models:
    Different horizons may currently be served by different model families
    (e.g. an LSTM for day1, Ridge for day2/day3) -- and which one is "latest"
    can change day to day as the automated pipeline retrains. Rather than
    assuming one model type, this app reads each model's own inference_notes.txt
    (written at training time) to know exactly how to build its input and
    reverses the log1p transform consistently, whichever type it turns out to be.

WHY it re-derives lag/rolling/cyclical features here instead of reading them
from the feature store directly:
    The Hopsworks feature group only stores the RAW hourly features (from
    feature_pipeline.py) -- lag/rolling/cyclical features are engineered on
    top of that raw history at training time, not stored as columns. So the
    app pulls the recent raw history and recomputes those same derived
    features the same way training did, to avoid train/serve mismatch.

Run locally:
    streamlit run app.py
"""

import ast
import os

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")
HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai")
CITY_NAME = os.getenv("CITY_NAME", "Rawalpindi")
CITY_LAT = float(os.getenv("CITY_LAT", "33.5651"))
CITY_LON = float(os.getenv("CITY_LON", "73.0169"))

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 3

HORIZONS = {"day1": 24, "day2": 48, "day3": 72}
MODEL_REGISTRY_NAME_TEMPLATE = "aqi_forecast_model_{horizon_key}"

# Standard US EPA AQI health guidance per category -- real, publicly documented
# guidance, not fabricated. Used in the Health Recommendations section.
HEALTH_GUIDANCE = {
    "Good": [
        ("\U0001F60A", "Air quality is satisfactory. Enjoy outdoor activities as normal."),
    ],
    "Moderate": [
        ("\U0001F642", "Acceptable air quality."),
        ("\u26A0\uFE0F", "Unusually sensitive individuals should consider limiting prolonged outdoor exertion."),
    ],
    "Unhealthy for Sensitive Groups": [
        ("\U0001F637", "Sensitive groups (children, elderly, asthma/heart conditions) should reduce prolonged outdoor exertion."),
        ("\U0001F3C3", "Everyone else can continue normal outdoor activities."),
    ],
    "Unhealthy": [
        ("\U0001F6AB", "Everyone should reduce prolonged or heavy outdoor exertion."),
        ("\U0001F637", "Sensitive groups should avoid prolonged outdoor exertion entirely."),
    ],
    "Very Unhealthy": [
        ("\U0001F3E0", "Health alert: everyone should avoid prolonged outdoor exertion."),
        ("\U0001F476", "Sensitive groups should remain indoors."),
    ],
    "Hazardous": [
        ("\U0001F6A8", "Health emergency: everyone should avoid all outdoor exertion."),
        ("\U0001F3E0", "Remain indoors with windows closed if possible."),
    ],
}

TABULAR_FEATURE_COLUMNS = [
    "hour", "day", "month", "day_of_week",
    "aqi", "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3",
    "temperature", "humidity", "pressure", "wind_speed", "wind_deg",
    "aqi_change_rate",
    "pm2_5_lag_24h", "pm2_5_lag_48h", "pm2_5_rolling_mean_24h",
    "aqi_rolling_mean_24h",
]
SEQ_FEATURE_COLUMNS = [
    "aqi", "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3",
    "temperature", "humidity", "pressure", "wind_speed", "wind_deg",
    "hour_sin", "hour_cos",
]

MAX_PLAUSIBLE_PM2_5 = 1000.0

# US EPA PM2.5 24-hour breakpoints (ug/m3) -- used for the hazard alerts and
# color-coded categories on the dashboard.
PM25_CATEGORIES = [
    (0, 12.0, "Good", "#22C55E"),
    (12.1, 35.4, "Moderate", "#FACC15"),
    (35.5, 55.4, "Unhealthy for Sensitive Groups", "#F97316"),
    (55.5, 150.4, "Unhealthy", "#EF4444"),
    (150.5, 250.4, "Very Unhealthy", "#8B5CF6"),
    (250.5, 1000.0, "Hazardous", "#7C2D2D"),
]


def categorize_pm25(value: float):
    for lo, hi, label, color in PM25_CATEGORIES:
        if lo <= value <= hi:
            return label, color
    return "Unknown", "#95a5a6"


# ---------------------------------------------------------------------------
# HOPSWORKS CONNECTION + DATA
# ---------------------------------------------------------------------------
@st.cache_resource
def connect_hopsworks():
    import hopsworks

    project = hopsworks.login(
        project=HOPSWORKS_PROJECT_NAME, host=HOPSWORKS_HOST, port=443, api_key_value=HOPSWORKS_API_KEY,
    )
    return project


@st.cache_data(ttl=3600)  # refresh hourly -- matches how often new raw data actually arrives
def load_recent_features(_project) -> pd.DataFrame:
    fs = _project.get_feature_store()
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    df = fg.read()
    df = df[df["city"] == CITY_NAME].sort_values("event_time").reset_index(drop=True)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pm2_5_lag_24h"] = df["pm2_5"].shift(24)
    df["pm2_5_lag_48h"] = df["pm2_5"].shift(48)
    df["pm2_5_rolling_mean_24h"] = df["pm2_5"].rolling(window=24, min_periods=1).mean()
    df["aqi_rolling_mean_24h"] = df["aqi"].rolling(window=24, min_periods=1).mean()
    return df


def add_cyclical_hour(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df


# ---------------------------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------------------------
def parse_inference_notes(path: str) -> dict:
    notes = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if value.startswith("["):
                try:
                    value = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    pass
            notes[key] = value
    return notes


@st.cache_resource
def load_horizon_model(_project, horizon_key: str):
    mr = _project.get_model_registry()
    registry_name = MODEL_REGISTRY_NAME_TEMPLATE.format(horizon_key=horizon_key)
    # IMPORTANT: mr.get_model(name) WITHOUT a version does not return the
    # latest version -- it silently defaults to version 1 (the very first
    # one ever registered), confirmed by Hopsworks' own "defaulting to 1"
    # warning. That bug meant this dashboard could show a stale, outdated
    # model even after better ones were registered. Fetch all versions and
    # explicitly pick the highest one instead.
    all_versions = mr.get_models(registry_name)
    if not all_versions:
        raise RuntimeError(f"No registered model found for '{registry_name}'.")
    model_meta = max(all_versions, key=lambda m: m.version)
    download_dir = model_meta.download()

    with open(os.path.join(download_dir, "model_type.txt")) as f:
        model_type = f.read().strip()

    scaler = joblib.load(os.path.join(download_dir, "scaler.pkl"))

    notes_path = os.path.join(download_dir, "inference_notes.txt")
    notes = parse_inference_notes(notes_path) if os.path.exists(notes_path) else {}

    is_keras = model_type == "neural_network" or model_type.startswith("lstm")
    if is_keras:
        import tensorflow as tf
        model = tf.keras.models.load_model(os.path.join(download_dir, "model.keras"))
    else:
        model = joblib.load(os.path.join(download_dir, "model.pkl"))

    # Pull the REAL cross-validated metrics that were recorded when this model
    # was registered (see training_pipeline.py) -- used for an honest
    # "reliability" indicator instead of a fabricated confidence score, since
    # these are point-prediction models without calibrated uncertainty.
    stored_metrics = getattr(model_meta, "training_metrics", None) or {}

    return {
        "model": model,
        "scaler": scaler,
        "model_type": model_type,
        "is_sequence": model_type.startswith("lstm"),
        "window_hours": int(notes.get("window_hours", 48)),
        "version": model_meta.version,
        "r2": stored_metrics.get("r2"),
        "feature_columns": notes.get("feature_columns", TABULAR_FEATURE_COLUMNS),
    }


# ---------------------------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------------------------
def predict_tabular(bundle: dict, latest_row: pd.Series) -> float:
    X = latest_row[TABULAR_FEATURE_COLUMNS].values.reshape(1, -1).astype(float)
    X_scaled = bundle["scaler"].transform(X)
    pred_log = bundle["model"].predict(X_scaled)
    pred = np.expm1(pred_log)[0]
    return float(np.clip(pred, 0, MAX_PLAUSIBLE_PM2_5))


def predict_sequence(bundle: dict, df_seq: pd.DataFrame) -> float:
    window = bundle["window_hours"]
    seq = df_seq.iloc[-window:][SEQ_FEATURE_COLUMNS].values.astype("float32")
    n_features = seq.shape[1]
    flat = seq.reshape(-1, n_features)
    flat_scaled = bundle["scaler"].transform(flat)
    seq_scaled = flat_scaled.reshape(1, window, n_features).astype("float32")
    pred_log = bundle["model"].predict(seq_scaled, verbose=0).flatten()[0]
    return float(np.clip(np.expm1(pred_log), 0, MAX_PLAUSIBLE_PM2_5))


def get_prediction(bundle: dict, latest_row: pd.Series, df_seq: pd.DataFrame) -> float:
    if bundle["is_sequence"]:
        return predict_sequence(bundle, df_seq)
    return predict_tabular(bundle, latest_row)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title=f"{CITY_NAME} AQI Forecast", page_icon="\U0001F32B\uFE0F", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;700&display=swap');

.stApp {
    background-color: #0A1B33;
    background-image:
        radial-gradient(circle at top left, rgba(56,189,248,0.08), transparent 45%),
        radial-gradient(circle at bottom right, rgba(16,185,129,0.05), transparent 45%),
        linear-gradient(180deg, #071426 0%, #0A1B33 50%, #10264A 100%);
    background-attachment: fixed;
}

html, body, [class*="css"], p, span, div, label {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3 {
    font-family: 'Fraunces', serif !important;
    color: #f5f2ea !important;
    letter-spacing: -0.01em;
    font-weight: 600 !important;
}

[data-testid="stCaptionContainer"], .stCaption, [data-testid="stMarkdownContainer"] p {
    color: #e3e0d6 !important;
}

[data-testid="stAlert"] { border-radius: 12px; font-family: 'Inter', sans-serif; }

.aqi-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.76em;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #7dd3c0;
    margin-bottom: 0.9em;
    margin-top: 0.9em;
    font-weight: 600;
}

h1 {
    font-size: 2.5em !important;
}
h2, h3 {
    font-size: 1.55em !important;
    margin-top: 0.2em !important;
    margin-bottom: 0.4em !important;
}

.aqi-card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 18px;
    padding: 1.9em 1.6em 1.7em;
    text-align: center;
    height: 100%;
    box-shadow: 0 4px 24px rgba(0,0,0,0.18);
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
    animation: aqiFadeInUp 0.6s ease both;
}
.aqi-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 16px 40px rgba(0,0,0,0.32);
    border-color: rgba(56, 189, 248, 0.35);
}

@keyframes aqiFadeInUp {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}

.aqi-hero-number {
    font-family: 'JetBrains Mono', monospace;
    font-size: 5.2em;
    font-weight: 700;
    color: #f9f7f1;
    line-height: 1;
    animation: aqiFadeInUp 0.7s ease both;
}
.aqi-hero-unit {
    font-size: 0.22em;
    color: #d8d4c8;
    font-weight: 500;
    margin-left: 0.15em;
}

.aqi-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.74em;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: #d8d4c8;
    margin-bottom: 0.35em;
}
.aqi-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.0em;
    font-weight: 700;
    color: #f9f7f1;
    line-height: 1.15;
}
.aqi-unit {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.5em;
    color: #d8d4c8;
    font-weight: 500;
}
.aqi-badge {
    display: inline-block;
    padding: 0.3em 0.8em;
    border-radius: 999px;
    font-size: 0.85em;
    font-weight: 600;
    margin-top: 0.5em;
}

.aqi-ai-card {
    background: rgba(159, 214, 184, 0.08);
    border: 1px solid rgba(159, 214, 184, 0.3);
    border-radius: 16px;
    padding: 1.4em 1.6em;
    font-family: 'Inter', sans-serif;
    font-size: 1.08em;
    line-height: 1.6;
    color: #f5f2ea;
    animation: aqiFadeInUp 0.6s ease both;
}

.aqi-health-card {
    background: rgba(245, 242, 234, 0.08);
    border-radius: 14px;
    padding: 1em 1.1em;
    margin-bottom: 0.55em;
    font-family: 'Inter', sans-serif;
    color: #f5f2ea;
    font-size: 0.95em;
    display: flex;
    align-items: center;
    gap: 0.6em;
    animation: aqiFadeInUp 0.6s ease both;
}
.aqi-health-icon { font-size: 1.5em; }

.aqi-model-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78em;
    color: #cfcabd;
}

/* -----------------------------------------------------------------------
   RESPONSIVE LAYOUT
   ----------------------------------------------------------------------- */

/* Max-width centered container -- keeps content from stretching edge to
   edge on very wide (1920px+) screens. */
.block-container {
    max-width: 1440px;
    margin: 0 auto;
    padding-top: 2.2rem;
    padding-bottom: 3.5rem;
}

/* More breathing room around each section divider */
hr {
    margin: 2.4em 0 !important;
    opacity: 0.14;
}

/* Charts: guarantee a minimum height so they never get squeezed illegibly
   thin, and let their container wrap naturally instead of overflowing. */
[data-testid="stPlotlyChart"] {
    min-height: 180px;
}

/* Below ~1200px: tighten outer padding a little so content isn't cramped
   against the edges, and give cards a touch more room to breathe. */
@media (max-width: 1200px) {
    .block-container { padding-left: 1.75rem; padding-right: 1.75rem; }
}

/* Below ~992px (tablet): force any multi-column row (hero gauge, forecast
   cards, confidence cards, about/map) to stack to full width instead of
   squeezing side by side. This is a deliberate override on top of
   Streamlit's own (narrower) default stacking threshold, using the stable
   data-testid Streamlit assigns to every column. */
@media (max-width: 992px) {
    [data-testid="column"] {
        flex: 1 1 100% !important;
        width: 100% !important;
        min-width: 100% !important;
        margin-bottom: 1.1em;
    }
    .aqi-hero-number { font-size: 3.4em; }
    .block-container { padding-left: 1.25rem; padding-right: 1.25rem; }
}

/* Below ~640px (mobile): reduce padding/type scale further so nothing
   overflows horizontally. */
@media (max-width: 640px) {
    .aqi-card { padding: 1.3em 1.1em 1.2em; }
    .aqi-hero-number { font-size: 2.6em; }
    .aqi-ai-card { padding: 1.1em 1.2em; font-size: 0.98em; }
    .block-container { padding-left: 0.9rem; padding-right: 0.9rem; }
}
</style>
""", unsafe_allow_html=True)

if not HOPSWORKS_API_KEY or not HOPSWORKS_PROJECT_NAME:
    st.error("Missing HOPSWORKS_API_KEY / HOPSWORKS_PROJECT_NAME. Set them in your .env file.")
    st.stop()

with st.spinner("Connecting to Hopsworks..."):
    project = connect_hopsworks()

with st.spinner("Loading recent air quality data..."):
    df_raw = load_recent_features(project)

if df_raw.empty:
    st.error(f"No data found for {CITY_NAME} in the feature store yet.")
    st.stop()

df_lag = add_lag_features(df_raw)
df_seq = add_cyclical_hour(df_raw)
latest_row = df_lag.iloc[-1]
latest_time = latest_row["event_time"]
current_pm25 = float(latest_row["pm2_5"])
current_label, current_color = categorize_pm25(current_pm25)
recent = df_raw[df_raw["event_time"] >= latest_time - pd.Timedelta(days=7)]

predictions = {}
model_info = {}
with st.spinner("Loading forecast models..."):
    for horizon_key in HORIZONS:
        bundle = load_horizon_model(project, horizon_key)
        predictions[horizon_key] = get_prediction(bundle, latest_row, df_seq)
        model_info[horizon_key] = bundle

average_pred = float(np.mean(list(predictions.values())))
labels = {"day1": "Day +1 (24h)", "day2": "Day +2 (48h)", "day3": "Day +3 (72h)"}

# --- Sidebar: live clock (self-contained JS, ticks locally without
# triggering Streamlit reruns) + real system status (no fabricated values) ---
with st.sidebar:
    st.markdown("<div class='aqi-eyebrow'>LIVE TIME</div>", unsafe_allow_html=True)
    st.components.v1.html(
        """
        <div id="aqi-clock" style="font-family:'JetBrains Mono',monospace;color:#f9f7f1;
                    font-size:1.6em;font-weight:700;line-height:1.2;"></div>
        <div id="aqi-clock-date" style="font-family:'Inter',sans-serif;color:#9fb0c8;
                    font-size:0.85em;margin-top:0.2em;"></div>
        <script>
        function updateAqiClock() {
            const utcNow = new Date();
            const pkt = new Date(utcNow.getTime() + (5 * 60 * 60 * 1000));
            const hh = String(pkt.getUTCHours()).padStart(2, '0');
            const mm = String(pkt.getUTCMinutes()).padStart(2, '0');
            const ss = String(pkt.getUTCSeconds()).padStart(2, '0');
            document.getElementById('aqi-clock').innerText = hh + ':' + mm + ':' + ss + ' PKT';
            const days = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
            const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
            document.getElementById('aqi-clock-date').innerText =
                days[pkt.getUTCDay()] + ', ' + months[pkt.getUTCMonth()] + ' ' + pkt.getUTCDate() + ' \u00b7 UTC+5';
        }
        updateAqiClock();
        setInterval(updateAqiClock, 1000);
        </script>
        """,
        height=70,
    )

    st.markdown("<div class='aqi-eyebrow' style='margin-top:1.2em;'>SYSTEM STATUS</div>", unsafe_allow_html=True)
    st.markdown(
        f"""
<div style='font-family:Inter,sans-serif;font-size:0.88em;color:#f0ede4;line-height:2.1;'>
\u2705 Data source: OpenWeather + Open-Meteo<br>
\u2705 Hopsworks: Connected<br>
\u2705 Models loaded: {len(model_info)}/3<br>
\u2705 Last data update: {latest_time.strftime('%H:%M UTC')}<br>
\u2705 Feature pipeline: hourly<br>
\u2705 Training pipeline: daily
</div>
""",
        unsafe_allow_html=True,
    )

day1_val, day3_val = predictions["day1"], predictions["day3"]
pct_change_3day = ((day3_val - day1_val) / day1_val * 100) if day1_val else 0.0
outlook_direction = "worsen" if day3_val > day1_val else "improve"
best_horizon = min(predictions, key=predictions.get)

worst_label, worst_color = max(
    (categorize_pm25(v) for v in list(predictions.values()) + [current_pm25]),
    key=lambda lc: [c[2] for c in PM25_CATEGORIES].index(lc[0]),
)

# Primary pollutant: purely descriptive (highest current raw concentration),
# NOT a feature-importance/causal claim -- we haven't run real SHAP analysis.
_pollutant_cols = ["pm2_5", "pm10", "co", "no", "no2", "o3", "so2", "nh3"]
_pollutant_display = {"pm2_5": "PM2.5", "pm10": "PM10", "co": "CO", "no": "NO",
                       "no2": "NO\u2082", "o3": "O\u2083", "so2": "SO\u2082", "nh3": "NH\u2083"}
primary_pollutant_col = max(_pollutant_cols, key=lambda c: float(latest_row[c]))
primary_pollutant_label = _pollutant_display[primary_pollutant_col]

pm25_24h_ago_val = float(df_raw["pm2_5"].iloc[-25]) if len(df_raw) > 24 else None
trend_word, trend_pct = "steady", 0.0
if pm25_24h_ago_val and pm25_24h_ago_val > 0:
    trend_pct = ((current_pm25 - pm25_24h_ago_val) / pm25_24h_ago_val) * 100
    trend_word = "rising" if trend_pct > 3 else ("falling" if trend_pct < -3 else "steady")

_r2_values_all = [m["r2"] for m in model_info.values() if m.get("r2") is not None]
avg_confidence = (sum(_r2_values_all) / len(_r2_values_all)) if _r2_values_all else None

# One real, checkable sentence -- not a fabricated causal claim.
ai_summary = (
    f"Air quality is currently <b>{current_label.upper()}</b> ({current_pm25:.0f} \u03bcg/m\u00b3 PM2.5). "
    f"PM2.5 is expected to {outlook_direction} over the next 3 days, shifting from {day1_val:.0f} "
    f"to {day3_val:.0f} \u03bcg/m\u00b3 ({'+' if pct_change_3day >= 0 else ''}{pct_change_3day:.0f}%). "
    f"{labels[best_horizon]} looks most favorable for outdoor activity."
)

# =============================================================================
# SECTION 1 -- HERO: current AQI, gauge, one-sentence AI summary
# =============================================================================
st.markdown(f"<div class='aqi-eyebrow'>LIVE \u00b7 {CITY_NAME.upper()}</div>", unsafe_allow_html=True)

hero_left, hero_right = st.columns([1, 1.1])
with hero_left:
    st.markdown(
        f"<div class='aqi-hero-number'>{current_pm25:.0f}"
        f"<span class='aqi-hero-unit'>\u03bcg/m\u00b3 PM2.5</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='aqi-badge' style='background-color:{current_color}33;color:{current_color};"
        f"border:1px solid {current_color};font-size:1.05em;'>{current_label}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(f"<div style='height:0.9em;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='aqi-ai-card'>"
        f"<span style='background:#22c55e33;color:#4ade80;padding:0.15em 0.6em;border-radius:999px;"
        f"font-size:0.7em;font-weight:700;letter-spacing:0.05em;'>\u25CF LIVE</span> "
        f"\U0001F916 {ai_summary}"
        f"<div style='margin-top:0.9em;display:flex;gap:1.8em;flex-wrap:wrap;font-size:0.82em;color:#c9d6e8;'>"
        f"<span>Trend: <b style='color:#f0ede4;'>{trend_word}</b></span>"
        + (f"<span>Confidence: <b style='color:#f0ede4;'>{avg_confidence:.2f} R\u00b2</b></span>"
           if avg_confidence is not None else "")
        + f"<span>Primary pollutant: <b style='color:#f0ede4;'>{primary_pollutant_label}</b></span>"
        f"</div></div>",
        unsafe_allow_html=True,
    )

with hero_right:
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=current_pm25,
        number={"suffix": " \u03bcg/m\u00b3", "font": {"size": 34, "family": "JetBrains Mono", "color": "#f9f7f1"}},
        gauge={
            "axis": {"range": [0, 300], "tickcolor": "#f0ede4", "tickfont": {"color": "#d8d4c8"}},
            "bar": {"color": current_color, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 12.0], "color": "rgba(46,204,113,0.30)"},
                {"range": [12.0, 35.4], "color": "rgba(241,196,15,0.30)"},
                {"range": [35.4, 55.4], "color": "rgba(230,126,34,0.30)"},
                {"range": [55.4, 150.4], "color": "rgba(231,76,60,0.30)"},
                {"range": [150.4, 250.4], "color": "rgba(142,68,173,0.30)"},
                {"range": [250.4, 300], "color": "rgba(125,60,60,0.30)"},
            ],
        },
    ))
    gauge_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", height=280,
        margin=dict(l=25, r=25, t=20, b=10),
        font=dict(color="#f0ede4", family="Inter"),
    )
    st.plotly_chart(gauge_fig, use_container_width=True)

st.caption(f"Last updated {latest_time.strftime('%b %d, %H:%M UTC')} \u00b7 updates hourly")
st.divider()

# =============================================================================
# SECTION 2 -- 3-DAY AI FORECAST (clean cards, no technical labels here --
# those live in the Model Confidence / About sections further down)
# =============================================================================
st.markdown("<div class='aqi-eyebrow'>3-DAY AI FORECAST</div>", unsafe_allow_html=True)

if worst_label in ("Unhealthy", "Very Unhealthy", "Hazardous"):
    st.warning(f"\u26A0\uFE0F Forecast reaches **{worst_label}** levels within 3 days.")
elif worst_label == "Unhealthy for Sensitive Groups":
    st.info(f"\u2139\uFE0F Forecast reaches **{worst_label}** levels within 3 days.")

cols = st.columns(4)
for col, horizon_key in zip(cols[:3], HORIZONS):
    pred = predictions[horizon_key]
    label, color = categorize_pm25(pred)
    pct_vs_current = ((pred - current_pm25) / current_pm25 * 100) if current_pm25 else 0.0
    arrow = "\u25B2" if pct_vs_current > 0 else ("\u25BC" if pct_vs_current < 0 else "\u25CF")
    with col:
        st.markdown(
            f"<div class='aqi-card'>"
            f"<div class='aqi-label'>{labels[horizon_key]}</div>"
            f"<div class='aqi-value'>{pred:.1f}<span class='aqi-unit'> \u03bcg/m\u00b3</span></div>"
            f"<div style='font-family:JetBrains Mono,monospace;font-size:0.78em;color:#c9d6e8;margin-top:0.2em;'>"
            f"{arrow} {abs(pct_vs_current):.0f}% vs now</div>"
            f"<div class='aqi-badge' style='background-color:{color}33;color:{color};border:1px solid {color};'>"
            f"{label}</div></div>",
            unsafe_allow_html=True,
        )

        spark_x = list(recent["event_time"].iloc[-24:]) + [latest_time + pd.Timedelta(hours=HORIZONS[horizon_key])]
        spark_y = list(recent["pm2_5"].iloc[-24:]) + [pred]
        spark_fig = go.Figure(go.Scatter(x=spark_x, y=spark_y, mode="lines", line=dict(color=color, width=2)))
        spark_fig.update_layout(
            height=48, margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(visible=False), yaxis=dict(visible=False), showlegend=False,
        )
        st.plotly_chart(spark_fig, use_container_width=True, config={"displayModeBar": False})
with cols[3]:
    label, color = categorize_pm25(average_pred)
    st.markdown(
        f"<div class='aqi-card'>"
        f"<div class='aqi-label'>3-Day Average</div>"
        f"<div class='aqi-value'>{average_pred:.1f}<span class='aqi-unit'> \u03bcg/m\u00b3</span></div>"
        f"<div class='aqi-badge' style='background-color:{color}33;color:{color};border:1px solid {color};'>"
        f"{label}</div></div>",
        unsafe_allow_html=True,
    )
st.divider()

# =============================================================================
# SECTION 3 -- INTERACTIVE TIMELINE
# =============================================================================
st.markdown("<div class='aqi-eyebrow'>INTERACTIVE TIMELINE</div>", unsafe_allow_html=True)

forecast_times = [latest_time + pd.Timedelta(hours=h) for h in HORIZONS.values()]
forecast_values = [predictions[k] for k in HORIZONS]

fig = go.Figure()
fig.add_trace(go.Scatter(x=recent["event_time"], y=recent["pm2_5"], mode="lines",
                          name="Actual (last 7 days)", line=dict(color="#3498db")))
fig.add_trace(go.Scatter(x=forecast_times, y=forecast_values, mode="lines+markers",
                          name="Forecast", line=dict(color="#e74c3c", dash="dash"), marker=dict(size=10)))
fig.add_trace(go.Scatter(x=[latest_time, forecast_times[0]], y=[current_pm25, forecast_values[0]],
                          mode="lines", line=dict(color="#e74c3c", dash="dash"), showlegend=False))
fig.update_layout(
    xaxis_title="Time", yaxis_title="PM2.5 (\u03bcg/m\u00b3)", hovermode="x unified",
    height=420, margin=dict(l=10, r=10, t=20, b=10),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(245,242,234,0.06)",
    font=dict(family="Inter, sans-serif", color="#f0ede4"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#f0ede4")),
    xaxis=dict(gridcolor="rgba(245,242,234,0.12)"), yaxis=dict(gridcolor="rgba(245,242,234,0.12)"),
)
st.plotly_chart(fig, use_container_width=True)
st.divider()

# =============================================================================
# SECTION 4 -- POLLUTANT BREAKDOWN (neutral -- no "top contributor" claim)
# =============================================================================
st.markdown("<div class='aqi-eyebrow'>POLLUTANT BREAKDOWN</div>", unsafe_allow_html=True)

pollutant_cols = ["pm2_5", "pm10", "co", "no", "no2", "o3", "so2", "nh3"]
pollutant_labels = ["PM2.5", "PM10", "CO", "NO", "NO\u2082", "O\u2083", "SO\u2082", "NH\u2083"]
pollutant_values = [float(latest_row[c]) for c in pollutant_cols]

poll_fig = go.Figure(go.Bar(
    x=pollutant_labels, y=pollutant_values,
    marker=dict(color="#6fb98f", line=dict(color="rgba(245,242,234,0.3)", width=1)),
    text=[f"{v:.1f}" for v in pollutant_values], textposition="outside",
    textfont=dict(color="#f0ede4", family="JetBrains Mono"),
))
poll_fig.update_layout(
    height=340, margin=dict(l=10, r=10, t=20, b=10),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(245,242,234,0.06)",
    font=dict(family="Inter, sans-serif", color="#f0ede4"), yaxis_title="\u03bcg/m\u00b3",
    xaxis=dict(gridcolor="rgba(245,242,234,0.12)"), yaxis=dict(gridcolor="rgba(245,242,234,0.12)"),
)
st.plotly_chart(poll_fig, use_container_width=True)
st.caption("Raw pollutant concentrations from the latest hourly reading.")
st.divider()

# =============================================================================
# SECTION 5 -- AI EXPLANATION (narrative, but every number is real)
# =============================================================================
st.markdown("<div class='aqi-eyebrow'>AI EXPLANATION</div>", unsafe_allow_html=True)

pm25_24h_ago = float(df_raw["pm2_5"].iloc[-25]) if len(df_raw) > 24 else None
trend_sentence = ""
if pm25_24h_ago and pm25_24h_ago > 0:
    pct_24h = ((current_pm25 - pm25_24h_ago) / pm25_24h_ago) * 100
    trend_sentence = (
        f"Over the last 24 hours, PM2.5 has {'risen' if pct_24h > 0 else 'fallen'} "
        f"{abs(pct_24h):.0f}%, from {pm25_24h_ago:.1f} to {current_pm25:.1f} \u03bcg/m\u00b3. "
    )

narrative = (
    f"{trend_sentence}"
    f"Looking ahead, the model forecasts PM2.5 to reach {predictions['day1']:.1f} \u03bcg/m\u00b3 tomorrow, "
    f"{predictions['day2']:.1f} \u03bcg/m\u00b3 the day after, and {predictions['day3']:.1f} \u03bcg/m\u00b3 in three days. "
    f"{labels[best_horizon]} currently shows the most favorable forecasted air quality of the period."
)
st.markdown(f"<div class='aqi-ai-card'>{narrative}</div>", unsafe_allow_html=True)
st.divider()

# =============================================================================
# SECTION 6 -- HEALTH RECOMMENDATIONS (real EPA guidance for the worst
# category forecast in the 3-day window)
# =============================================================================
st.markdown("<div class='aqi-eyebrow'>HEALTH RECOMMENDATIONS</div>", unsafe_allow_html=True)
st.caption(f"Based on the {worst_label} level reached in the current + 3-day forecast window")

for icon, text in HEALTH_GUIDANCE.get(worst_label, HEALTH_GUIDANCE["Moderate"]):
    st.markdown(
        f"<div class='aqi-health-card'><span class='aqi-health-icon'>{icon}</span><span>{text}</span></div>",
        unsafe_allow_html=True,
    )

# "Best Day" widget -- honestly day-level, not hour-level, since our model
# forecasts Day+1/2/3 (24h/48h/72h), not an hour-by-hour intraday curve.
st.markdown("<div style='height:0.6em;'></div>", unsafe_allow_html=True)
best_pred = predictions[best_horizon]
best_pct_of_scale = min(best_pred / 150.4, 1.0)  # relative position on the "Unhealthy" threshold, capped at 100%
best_col1, best_col2 = st.columns([1, 2])
with best_col1:
    ring_fig = go.Figure(go.Pie(
        values=[best_pct_of_scale, 1 - best_pct_of_scale], hole=0.75,
        marker=dict(colors=["#4ade80", "rgba(255,255,255,0.06)"]),
        textinfo="none", sort=False, direction="clockwise",
    ))
    ring_fig.update_layout(
        showlegend=False, height=160, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        annotations=[dict(text=f"{labels[best_horizon].split(' (')[0]}<br>{labels[best_horizon].split('(')[1][:-1]}",
                           font=dict(size=14, color="#f9f7f1", family="Inter"), showarrow=False)],
    )
    st.plotly_chart(ring_fig, use_container_width=True, config={"displayModeBar": False})
with best_col2:
    st.markdown(
        f"<div style='padding-top:1.2em;'>"
        f"<div class='aqi-label'>\U0001F333 Best Day for Outdoor Activity</div>"
        f"<div style='font-family:Inter,sans-serif;color:#f0ede4;font-size:1.0em;line-height:1.5;'>"
        f"<b>{labels[best_horizon]}</b> shows the most favorable forecasted air quality of the "
        f"3-day period, at {best_pred:.1f} \u03bcg/m\u00b3.</div></div>",
        unsafe_allow_html=True,
    )
st.divider()

# =============================================================================
# SECTION 7 -- MODEL CONFIDENCE (real cross-validated R\u00b2, explained honestly)
# =============================================================================
st.markdown("<div class='aqi-eyebrow'>MODEL CONFIDENCE</div>", unsafe_allow_html=True)
st.caption(
    "R\u00b2 (cross-validated) \u2014 how much of the real variation in PM2.5 the model explains. "
    "1.0 is perfect; 0 means no better than guessing the average. Near-term forecasts are "
    "typically more reliable than longer-range ones."
)
conf_cols = st.columns(3)
for col, horizon_key in zip(conf_cols, HORIZONS):
    r2 = model_info[horizon_key].get("r2")
    with col:
        if r2 is not None:
            bar_pct = max(0, min(r2, 1.0)) * 100
            st.markdown(
                f"<div class='aqi-card'><div class='aqi-label'>{labels[horizon_key]}</div>"
                f"<div class='aqi-value'>{r2:.2f}</div>"
                f"<div class='aqi-model-tag' style='margin-bottom:0.5em;'>R\u00b2</div>"
                f"<div style='background:rgba(255,255,255,0.08);border-radius:999px;height:8px;overflow:hidden;'>"
                f"<div style='width:{bar_pct:.0f}%;height:100%;background:linear-gradient(90deg,#38bdf8,#4ade80);'></div>"
                f"</div></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='aqi-card'><div class='aqi-label'>{labels[horizon_key]}</div>"
                f"<div class='aqi-model-tag'>not available</div></div>",
                unsafe_allow_html=True,
            )
st.divider()

# =============================================================================
# SECTION 8 -- ABOUT THIS MODEL
# =============================================================================
st.markdown("<div class='aqi-eyebrow'>ABOUT THIS MODEL</div>", unsafe_allow_html=True)

about_col, map_col = st.columns([1.3, 1])
with about_col:
    st.markdown(
        f"""
This forecast is produced by a fully automated MLOps pipeline built for **{CITY_NAME}**:

- **Data sources:** OpenWeather (live + historical pollution) and Open-Meteo (historical weather), both free tiers
- **History used for training:** 5+ years of hourly data (back to Nov 2020)
- **Feature store & model registry:** Hopsworks
- **Automation:** feature data collected hourly, models retrained daily via GitHub Actions
- **Current models in production:**
    - Day+1: `{model_info['day1']['model_type']}` (v{model_info['day1']['version']})
    - Day+2: `{model_info['day2']['model_type']}` (v{model_info['day2']['version']})
    - Day+3: `{model_info['day3']['model_type']}` (v{model_info['day3']['version']})

Each horizon is served by whichever model type performed best in cross-validated testing for
that specific horizon \u2014 not a single one-size-fits-all model.
""",
    )
with map_col:
    # Layered glow sized by current severity -- an honest single-point
    # indicator (not a fake multi-point heatmap, since we only monitor
    # this one location).
    glow_sizes = [50, 34, 18]
    glow_opacities = [0.08, 0.16, 1.0]
    map_fig = go.Figure()
    for size, opacity in zip(glow_sizes, glow_opacities):
        map_fig.add_trace(go.Scattermapbox(
            lat=[CITY_LAT], lon=[CITY_LON], mode="markers",
            marker=dict(size=size, color=current_color, opacity=opacity),
            text=[CITY_NAME] if opacity == 1.0 else None,
            hoverinfo="text" if opacity == 1.0 else "skip",
            showlegend=False,
        ))
    map_fig.update_layout(
        mapbox=dict(style="open-street-map", center=dict(lat=CITY_LAT, lon=CITY_LON), zoom=8),
        height=260, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(map_fig, use_container_width=True)
    st.caption(f"Monitored location: {CITY_NAME} \u00b7 glow color reflects current category")
