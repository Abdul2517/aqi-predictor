import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from cities_config import CITIES

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")
HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai")

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 3
HORIZONS = {"day1": 24, "day2": 48, "day3": 72}
MODEL_REGISTRY_NAME_TEMPLATE = "aqi_forecast_model_{city_key}_{horizon_key}"

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

PM25_CATEGORIES = [
    (0, 12.0, "Good"),
    (12.1, 35.4, "Moderate"),
    (35.5, 55.4, "Unhealthy for Sensitive Groups"),
    (55.5, 150.4, "Unhealthy"),
    (150.5, 250.4, "Very Unhealthy"),
    (250.5, 1000.0, "Hazardous"),
]

OUTPUT_PATH = os.getenv("PREDICTIONS_OUTPUT_PATH", "frontend/public/predictions.json")


def categorize_pm25(value):
    for lo, hi, label in PM25_CATEGORIES:
        if lo <= value <= hi:
            return label
    return "Unknown"


def add_lag_features(df):
    df = df.copy()
    df["pm2_5_lag_24h"] = df["pm2_5"].shift(24)
    df["pm2_5_lag_48h"] = df["pm2_5"].shift(48)
    df["pm2_5_rolling_mean_24h"] = df["pm2_5"].rolling(window=24, min_periods=1).mean()
    df["aqi_rolling_mean_24h"] = df["aqi"].rolling(window=24, min_periods=1).mean()
    return df


def add_cyclical_hour(df):
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df


def load_horizon_model(project, city_key, horizon_key):
    mr = project.get_model_registry()
    registry_name = MODEL_REGISTRY_NAME_TEMPLATE.format(city_key=city_key, horizon_key=horizon_key)
    all_versions = mr.get_models(registry_name)
    if not all_versions:
        raise RuntimeError(f"No registered model found for '{registry_name}'.")
    model_meta = max(all_versions, key=lambda m: m.version)
    download_dir = model_meta.download()

    with open(os.path.join(download_dir, "model_type.txt")) as f:
        model_type = f.read().strip()
    scaler = joblib.load(os.path.join(download_dir, "scaler.pkl"))

    window_hours = 48
    notes_path = os.path.join(download_dir, "inference_notes.txt")
    if os.path.exists(notes_path):
        with open(notes_path) as f:
            for line in f:
                if line.startswith("window_hours="):
                    window_hours = int(line.strip().split("=")[1])

    is_sequence = model_type.startswith("lstm")
    if is_sequence or model_type == "neural_network":
        import tensorflow as tf
        model = tf.keras.models.load_model(os.path.join(download_dir, "model.keras"))
    else:
        model = joblib.load(os.path.join(download_dir, "model.pkl"))

    r2 = None
    metrics = getattr(model_meta, "training_metrics", None) or getattr(model_meta, "metrics", None)
    if metrics and "r2" in metrics:
        r2 = float(metrics["r2"])

    return {
        "model": model, "scaler": scaler, "model_type": model_type,
        "is_sequence": is_sequence, "window_hours": window_hours,
        "version": model_meta.version, "r2": r2,
    }


def predict_tabular(bundle, latest_row):
    X = latest_row[TABULAR_FEATURE_COLUMNS].values.reshape(1, -1).astype(float)
    X_scaled = bundle["scaler"].transform(X)
    pred_log = bundle["model"].predict(X_scaled)
    pred = np.expm1(pred_log)[0]
    return float(np.clip(pred, 0, MAX_PLAUSIBLE_PM2_5))


def predict_sequence(bundle, df_seq):
    window = bundle["window_hours"]
    seq = df_seq.iloc[-window:][SEQ_FEATURE_COLUMNS].values.astype("float32")
    n_features = seq.shape[1]
    flat_scaled = bundle["scaler"].transform(seq.reshape(-1, n_features))
    seq_scaled = flat_scaled.reshape(1, window, n_features).astype("float32")
    pred_log = bundle["model"].predict(seq_scaled, verbose=0).flatten()[0]
    return float(np.clip(np.expm1(pred_log), 0, MAX_PLAUSIBLE_PM2_5))


def get_prediction(bundle, latest_row, df_seq):
    if bundle["is_sequence"]:
        return predict_sequence(bundle, df_seq)
    return predict_tabular(bundle, latest_row)


def process_city(project, df_all, city_key, city_info):
    city_name = city_info["name"]
    city_lat = city_info["lat"]
    city_lon = city_info["lon"]

    print(f"\n=== {city_name} ===")
    df = df_all[df_all["city"] == city_name].sort_values("event_time").reset_index(drop=True)
    print(f"  -> {len(df)} rows loaded")
    if df.empty:
        raise RuntimeError(f"No feature rows found for city='{city_name}' in the feature store.")

    df_lag = add_lag_features(df)
    df_seq = add_cyclical_hour(df)
    latest_row = df_lag.iloc[-1]
    latest_time = latest_row["event_time"]
    current_pm25 = float(latest_row["pm2_5"])

    predictions = {}
    model_info = {}
    for horizon_key in HORIZONS:
        print(f"  Loading model + predicting for {horizon_key}...")
        bundle = load_horizon_model(project, city_key, horizon_key)
        predictions[horizon_key] = get_prediction(bundle, latest_row, df_seq)
        model_info[horizon_key] = {
            "model_type": bundle["model_type"], "version": bundle["version"], "r2": bundle["r2"],
        }

    average_pred = float(np.mean(list(predictions.values())))

    recent = df[df["event_time"] >= latest_time - pd.Timedelta(days=7)]
    trend_points = [
        {"time": row["event_time"].isoformat(), "pm2_5": float(row["pm2_5"])}
        for _, row in recent.iloc[::3].iterrows()
    ]

    pollutant_cols = ["pm2_5", "pm10", "co", "no", "no2", "o3", "so2", "nh3"]
    pollutants = {c: float(latest_row[c]) for c in pollutant_cols}

    pm25_24h_ago = float(df["pm2_5"].iloc[-25]) if len(df) > 24 else None
    trend_pct_24h = None
    if pm25_24h_ago and pm25_24h_ago > 0:
        trend_pct_24h = ((current_pm25 - pm25_24h_ago) / pm25_24h_ago) * 100

    best_horizon = min(predictions, key=predictions.get)
    day1_val, day3_val = predictions["day1"], predictions["day3"]
    outlook_pct = ((day3_val - day1_val) / day1_val * 100) if day1_val else 0.0

    return {
        "city": city_name,
        "status": "ok",
        "lat": city_lat,
        "lon": city_lon,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "last_data_update": latest_time.isoformat(),
        "current": {
            "pm2_5": current_pm25,
            "category": categorize_pm25(current_pm25),
            "trend_pct_24h": trend_pct_24h,
        },
        "predictions": {
            k: {"pm2_5": v, "category": categorize_pm25(v), **model_info[k]}
            for k, v in predictions.items()
        },
        "average": {"pm2_5": average_pred, "category": categorize_pm25(average_pred)},
        "best_day": best_horizon,
        "outlook_pct_3day": outlook_pct,
        "pollutants": pollutants,
        "trend": trend_points,
    }


def main():
    import hopsworks

    print(f"Connecting to Hopsworks project '{HOPSWORKS_PROJECT_NAME}'...")
    project = hopsworks.login(
        project=HOPSWORKS_PROJECT_NAME, host=HOPSWORKS_HOST, port=443, api_key_value=HOPSWORKS_API_KEY,
    )
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)

    print("Loading feature data for all cities (single read, filtered per city)...")
    df_all = fg.read()
    print(f"  -> {len(df_all)} total rows across all cities")

    output = {}
    succeeded, failed = [], []

    for city_key, city_info in CITIES.items():
        try:
            output[city_key] = process_city(project, df_all, city_key, city_info)
            succeeded.append(city_key)
            print(f"  \u2713 {city_info['name']} published successfully")
        except Exception as e:
            print(f"  \u2717 {city_info['name']} FAILED: {e}")
            output[city_key] = {
                "city": city_info["name"],
                "status": "unavailable",
                "error": str(e),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            failed.append(city_key)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nWrote predictions to {OUTPUT_PATH}")
    print(f"Succeeded: {succeeded}")
    print(f"Failed: {failed}")

    if not succeeded:
        raise SystemExit("All cities failed -- refusing to treat this as a successful run.")


if __name__ == "__main__":
    main()
