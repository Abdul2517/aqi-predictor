import os

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from dotenv import load_dotenv

matplotlib.use("Agg")
load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")
HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai")
CITY_NAME = os.getenv("CITY_NAME", "Rawalpindi")

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 3
HORIZONS = {"day1": 24, "day2": 48, "day3": 72}
MODEL_REGISTRY_NAME_TEMPLATE = "aqi_forecast_model_{horizon_key}"

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

OUTPUT_DIR = "shap_output"


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


def load_horizon_model(project, horizon_key):
    mr = project.get_model_registry()
    registry_name = MODEL_REGISTRY_NAME_TEMPLATE.format(horizon_key=horizon_key)
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

    return {
        "model": model, "scaler": scaler, "model_type": model_type,
        "is_sequence": is_sequence, "window_hours": window_hours,
        "version": model_meta.version,
    }


def make_predict_fn(bundle):
    model = bundle["model"]
    if bundle["model_type"] == "neural_network":
        return lambda X: np.array(model.predict(X, verbose=0)).reshape(-1)
    if bundle["is_sequence"]:
        return lambda X: np.array(model.predict(X, verbose=0)).reshape(-1)
    return lambda X: np.array(model.predict(X)).reshape(-1)


def explain_tabular_horizon(bundle, df_tabular, horizon_key, n_background=100, n_explain=300):
    print(f"  Building SHAP explainer for {horizon_key} ({bundle['model_type']}, tabular)...")
    sample = df_tabular[TABULAR_FEATURE_COLUMNS].sample(
        min(n_background + n_explain, len(df_tabular)), random_state=42
    )
    background_raw = sample.iloc[:n_background]
    explain_raw = sample.iloc[n_background:n_background + n_explain]

    background_scaled = bundle["scaler"].transform(background_raw)
    explain_scaled = bundle["scaler"].transform(explain_raw)

    predict_fn = make_predict_fn(bundle)
    explainer = shap.Explainer(predict_fn, background_scaled, feature_names=TABULAR_FEATURE_COLUMNS)
    shap_values = explainer(explain_scaled)

    mean_abs = np.abs(shap_values.values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh([TABULAR_FEATURE_COLUMNS[i] for i in order[:12]][::-1], mean_abs[order[:12]][::-1], color="#2e7d5b")
    ax.set_xlabel("Mean |SHAP value| (impact on prediction, \u03bcg/m\u00b3)")
    ax.set_title(f"Feature Importance \u2014 {horizon_key} ({bundle['model_type']})")
    fig.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"shap_{horizon_key}_importance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    print(f"  Top 5 features for {horizon_key}:")
    for i in order[:5]:
        print(f"    {TABULAR_FEATURE_COLUMNS[i]:<25} mean|SHAP|={mean_abs[i]:.3f}")


def explain_sequence_horizon(bundle, df_seq, horizon_key, n_background=30, n_explain=60):
    print(f"  Building SHAP explainer for {horizon_key} ({bundle['model_type']}, sequence model)...")
    print(f"  Note: explaining WINDOW-AVERAGED feature levels (not per-hour timestep detail), "
          f"since the full per-timestep input space is too high-dimensional for tractable SHAP computation.")

    window = bundle["window_hours"]
    n_features = len(SEQ_FEATURE_COLUMNS)
    values = df_seq[SEQ_FEATURE_COLUMNS].values.astype("float32")
    n = len(df_seq)

    valid_starts = list(range(window - 1, n))
    rng = np.random.default_rng(42)
    chosen = rng.choice(valid_starts, size=min(n_background + n_explain, len(valid_starts)), replace=False)

    window_avg_rows = []
    for i in chosen:
        seq = values[i - window + 1: i + 1]
        window_avg_rows.append(seq.mean(axis=0))
    window_avg = np.array(window_avg_rows)

    background_avg = window_avg[:n_background]
    explain_avg = window_avg[n_background:n_background + n_explain]

    model_predict = make_predict_fn(bundle)
    scaler = bundle["scaler"]

    def predict_from_avg(X_avg):
        n_rows = X_avg.shape[0]
        tiled = np.repeat(X_avg[:, np.newaxis, :], window, axis=1)
        flat = tiled.reshape(-1, n_features)
        flat_scaled = scaler.transform(flat)
        seq_scaled = flat_scaled.reshape(n_rows, window, n_features).astype("float32")
        return model_predict(seq_scaled)

    explainer = shap.Explainer(predict_from_avg, background_avg, feature_names=SEQ_FEATURE_COLUMNS)
    shap_values = explainer(explain_avg)

    mean_abs = np.abs(shap_values.values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh([SEQ_FEATURE_COLUMNS[i] for i in order[:12]][::-1], mean_abs[order[:12]][::-1], color="#2e7d5b")
    ax.set_xlabel("Mean |SHAP value| (impact on prediction, \u03bcg/m\u00b3)")
    ax.set_title(f"Feature Importance \u2014 {horizon_key} ({bundle['model_type']}, window-averaged)")
    fig.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"shap_{horizon_key}_importance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    print(f"  Top 5 features for {horizon_key}:")
    for i in order[:5]:
        print(f"    {SEQ_FEATURE_COLUMNS[i]:<25} mean|SHAP|={mean_abs[i]:.3f}")


def main():
    import hopsworks

    print(f"Connecting to Hopsworks project '{HOPSWORKS_PROJECT_NAME}'...")
    project = hopsworks.login(
        project=HOPSWORKS_PROJECT_NAME, host=HOPSWORKS_HOST, port=443, api_key_value=HOPSWORKS_API_KEY,
    )
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)

    print(f"Loading stored feature data for {CITY_NAME}...")
    df = fg.read()
    df = df[df["city"] == CITY_NAME].sort_values("event_time").reset_index(drop=True)
    print(f"  -> {len(df)} rows loaded")

    df_tabular = add_lag_features(df).dropna(subset=TABULAR_FEATURE_COLUMNS).reset_index(drop=True)
    df_seq = add_cyclical_hour(df)

    for horizon_key in HORIZONS:
        print(f"\n=== {horizon_key} ===")
        bundle = load_horizon_model(project, horizon_key)
        print(f"  Deployed model: {bundle['model_type']} (v{bundle['version']})")
        if bundle["is_sequence"]:
            explain_sequence_horizon(bundle, df_seq, horizon_key)
        else:
            explain_tabular_horizon(bundle, df_tabular, horizon_key)

    print(f"\nAll SHAP charts saved to ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
