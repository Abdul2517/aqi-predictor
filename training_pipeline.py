import os
import sys

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")
HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai")
CITY_NAME = os.getenv("CITY_NAME", "Rawalpindi")

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 3

HORIZONS = {
    "day1": 24,
    "day2": 48,
    "day3": 72,
}
N_CV_FOLDS = 5

MODEL_REGISTRY_NAME_TEMPLATE = "aqi_forecast_model_{horizon_key}"

FEATURE_COLUMNS = [
    "hour", "day", "month", "day_of_week",
    "aqi", "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3",
    "temperature", "humidity", "pressure", "wind_speed", "wind_deg",
    "aqi_change_rate",
    "pm2_5_lag_24h", "pm2_5_lag_48h", "pm2_5_rolling_mean_24h",
    "aqi_rolling_mean_24h",
]
TARGET_COLUMN = "pm2_5_target"
SOURCE_COLUMN_FOR_TARGET = "pm2_5"

MAX_PLAUSIBLE_PM2_5 = 1000.0


def load_feature_data(fs) -> pd.DataFrame:
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


def build_training_target(df: pd.DataFrame, horizon_hours: int) -> pd.DataFrame:
    df = df.copy()
    df[TARGET_COLUMN] = df[SOURCE_COLUMN_FOR_TARGET].shift(-horizon_hours)
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).reset_index(drop=True)
    return df


def evaluate(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_ridge(X_train, y_train):
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(X_train, y_train):
    model = HistGradientBoostingRegressor(
        max_iter=300, max_depth=6, learning_rate=0.05, random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def train_neural_network(X_train, y_train, X_val=None, y_val=None, epochs=50):
    import tensorflow as tf

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    optimizer = tf.keras.optimizers.Adam(clipnorm=1.0)
    model.compile(optimizer=optimizer, loss="mse")

    fit_kwargs = {"epochs": epochs, "batch_size": 32, "verbose": 0}
    if X_val is not None:
        fit_kwargs["validation_data"] = (X_val, y_val)
        fit_kwargs["callbacks"] = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
        ]
    model.fit(X_train, y_train, **fit_kwargs)
    return model


MODEL_BUILDERS = {
    "ridge": lambda Xtr, ytr, Xte, yte: train_ridge(Xtr, ytr),
    "random_forest": lambda Xtr, ytr, Xte, yte: train_random_forest(Xtr, ytr),
    "gradient_boosting": lambda Xtr, ytr, Xte, yte: train_gradient_boosting(Xtr, ytr),
    "neural_network": lambda Xtr, ytr, Xte, yte: train_neural_network(Xtr, ytr, Xte, yte),
}


def predict(model_name: str, model, X):
    if model_name == "neural_network":
        raw = np.expm1(model.predict(X, verbose=0).flatten())
    else:
        raw = np.expm1(model.predict(X))
    return np.clip(raw, 0, MAX_PLAUSIBLE_PM2_5)


def cross_validate_models(df: pd.DataFrame, horizon_hours: int) -> dict:
    X_all = df[FEATURE_COLUMNS]
    y_all_raw = df[TARGET_COLUMN].values
    y_all_log = np.log1p(y_all_raw)

    tscv = TimeSeriesSplit(n_splits=N_CV_FOLDS, gap=horizon_hours)
    fold_metrics = {name: [] for name in MODEL_BUILDERS}

    for fold_i, (train_idx, test_idx) in enumerate(tscv.split(X_all), start=1):
        X_train_raw = X_all.iloc[train_idx]
        X_test_raw = X_all.iloc[test_idx]
        y_train = y_all_log[train_idx]
        y_test_log = y_all_log[test_idx]
        y_test_actual = y_all_raw[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        print(f"    Fold {fold_i}/{N_CV_FOLDS} (train={len(train_idx)}, test={len(test_idx)})")
        for name, builder in MODEL_BUILDERS.items():
            model = builder(X_train, y_train, X_test, y_test_log)
            preds = predict(name, model, X_test)
            m = evaluate(y_test_actual, preds)
            fold_metrics[name].append(m)
            print(f"      {name:<18} RMSE={m['rmse']:6.2f}  MAE={m['mae']:6.2f}  R2={m['r2']:6.3f}")

    avg_metrics = {}
    for name, metrics_list in fold_metrics.items():
        avg_metrics[name] = {
            "rmse": float(np.mean([m["rmse"] for m in metrics_list])),
            "mae": float(np.mean([m["mae"] for m in metrics_list])),
            "r2": float(np.mean([m["r2"] for m in metrics_list])),
        }
    return avg_metrics


def get_current_production_rmse(project, horizon_key: str):
    try:
        mr = project.get_model_registry()
        registry_name = MODEL_REGISTRY_NAME_TEMPLATE.format(horizon_key=horizon_key)
        all_versions = mr.get_models(registry_name)
        if not all_versions:
            return None
        model_meta = max(all_versions, key=lambda m: m.version)
        metrics = getattr(model_meta, "training_metrics", None) or getattr(model_meta, "metrics", None)
        if metrics and "rmse" in metrics:
            return float(metrics["rmse"])
    except Exception as e:
        print(f"  (Could not look up current production metrics for {horizon_key}: {e}. "
              f"Proceeding as if no prior model exists.)")
    return None


def train_and_register_horizon(project, fs, df_with_lags: pd.DataFrame, horizon_key: str, horizon_hours: int):
    print(f"\n{'=' * 70}")
    print(f"HORIZON: {horizon_key} ({horizon_hours}h ahead)")
    print(f"{'=' * 70}")

    df = build_training_target(df_with_lags, horizon_hours)
    print(f"  -> {len(df)} rows usable for this horizon")

    min_rows_needed = (N_CV_FOLDS + 1) * 200
    if len(df) < min_rows_needed:
        print(f"  SKIPPING {horizon_key}: only {len(df)} usable rows, need at least {min_rows_needed}.")
        return

    print(f"  Running {N_CV_FOLDS}-fold time-series cross-validation (gap={horizon_hours}h)...")
    avg_metrics = cross_validate_models(df, horizon_hours)

    print(f"\n  === {horizon_key} cross-validated average metrics ===")
    for name, m in avg_metrics.items():
        print(f"    {name:<18} RMSE={m['rmse']:6.2f}  MAE={m['mae']:6.2f}  R2={m['r2']:6.3f}")

    best_name = min(avg_metrics, key=lambda name: avg_metrics[name]["rmse"])
    best_metrics = avg_metrics[best_name]
    print(f"\n  Best model for {horizon_key}: {best_name} "
          f"(avg RMSE={best_metrics['rmse']:.3f}, MAE={best_metrics['mae']:.3f}, R2={best_metrics['r2']:.3f})")

    current_rmse = get_current_production_rmse(project, horizon_key)
    if current_rmse is not None and best_metrics["rmse"] >= current_rmse:
        print(f"  No improvement for {horizon_key}: today's best ({best_metrics['rmse']:.3f} RMSE) "
              f"does not beat the current production model ({current_rmse:.3f} RMSE). "
              f"Keeping the existing deployed model -- not registering.")
        return
    if current_rmse is not None:
        print(f"  IMPROVEMENT: {best_metrics['rmse']:.3f} < current production {current_rmse:.3f}. Proceeding.")
    else:
        print(f"  No existing production model found for {horizon_key} -- registering this as the first one.")

    print(f"  Retraining {best_name} on the FULL dataset for deployment...")
    X_all_raw = df[FEATURE_COLUMNS]
    y_all_log = np.log1p(df[TARGET_COLUMN].values)

    final_scaler = StandardScaler()
    X_all = final_scaler.fit_transform(X_all_raw)
    final_model = MODEL_BUILDERS[best_name](X_all, y_all_log, None, None)

    save_dir = f"saved_model_{horizon_key}"
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(final_scaler, f"{save_dir}/scaler.pkl")
    if best_name == "neural_network":
        final_model.save(f"{save_dir}/model.keras")
    else:
        joblib.dump(final_model, f"{save_dir}/model.pkl")

    with open(f"{save_dir}/model_type.txt", "w") as f:
        f.write(best_name)
    with open(f"{save_dir}/inference_notes.txt", "w") as f:
        f.write("target_transform=log1p (apply np.expm1 to model output to get real PM2.5)\n")
        f.write(f"feature_columns={FEATURE_COLUMNS}\n")
        f.write(f"forecast_horizon_hours={horizon_hours}\n")
        f.write(f"horizon_key={horizon_key}\n")

    print(f"  Registering {horizon_key} model in Hopsworks Model Registry...")
    mr = project.get_model_registry()
    registry_name = MODEL_REGISTRY_NAME_TEMPLATE.format(horizon_key=horizon_key)
    model = mr.python.create_model(
        name=registry_name,
        metrics=best_metrics,
        description=(
            f"AQI/PM2.5 {horizon_hours}h-ahead ({horizon_key}) forecast model for {CITY_NAME}. "
            f"Predicts pm2_5 concentration (continuous). Model type '{best_name}' selected via "
            f"{N_CV_FOLDS}-fold time-series cross-validation across multiple seasons; "
            f"final model retrained on full dataset. Part of a 3-day forecast set "
            f"(day1/day2/day3); the dashboard average is computed from all three at display time."
        ),
    )
    model.save(save_dir)
    print(f"  Registered: {registry_name}")


def main():
    if not HOPSWORKS_API_KEY or not HOPSWORKS_PROJECT_NAME:
        sys.exit("Missing HOPSWORKS_API_KEY / HOPSWORKS_PROJECT_NAME. Set them in .env or as secrets.")

    import hopsworks

    print(f"Connecting to Hopsworks project '{HOPSWORKS_PROJECT_NAME}' on {HOPSWORKS_HOST}...")
    project = hopsworks.login(
        project=HOPSWORKS_PROJECT_NAME, host=HOPSWORKS_HOST, port=443, api_key_value=HOPSWORKS_API_KEY,
    )
    fs = project.get_feature_store()

    print(f"Loading stored feature data for {CITY_NAME}...")
    df = load_feature_data(fs)
    print(f"  -> {len(df)} raw rows loaded")

    print("Adding lag/rolling trend features...")
    df = add_lag_features(df)

    for horizon_key, horizon_hours in HORIZONS.items():
        train_and_register_horizon(project, fs, df, horizon_key, horizon_hours)

    print(f"\n{'=' * 70}")
    print("All horizons complete. Three models registered:")
    for horizon_key, horizon_hours in HORIZONS.items():
        print(f"  - {MODEL_REGISTRY_NAME_TEMPLATE.format(horizon_key=horizon_key)} ({horizon_hours}h ahead)")
    print("The dashboard average = mean of day1/day2/day3 predictions, computed at display time.")


if __name__ == "__main__":
    main()
