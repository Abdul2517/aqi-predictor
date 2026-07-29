import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from training_pipeline import (
    CITY_NAME,
    FEATURE_COLUMNS,
    HOPSWORKS_API_KEY,
    HOPSWORKS_HOST,
    HOPSWORKS_PROJECT_NAME,
    HORIZONS,
    MAX_PLAUSIBLE_PM2_5,
    MODEL_REGISTRY_NAME_TEMPLATE,
    SOURCE_COLUMN_FOR_TARGET,
    TARGET_COLUMN,
    add_lag_features,
    build_training_target,
    evaluate,
    load_feature_data,
    train_gradient_boosting,
    train_ridge,
)

N_CV_FOLDS = 3
WINDOW_HOURS = 48

SEQ_FEATURE_COLUMNS = [
    "aqi", "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3",
    "temperature", "humidity", "pressure", "wind_speed", "wind_deg",
    "hour_sin", "hour_cos",
]


def add_cyclical_hour(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df


def build_sequences(df: pd.DataFrame, horizon_hours: int, window: int = WINDOW_HOURS):
    df = df.sort_values("event_time").reset_index(drop=True)
    feature_values = df[SEQ_FEATURE_COLUMNS].values.astype("float32")
    target_values = df[SOURCE_COLUMN_FOR_TARGET].values.astype("float32")
    n = len(df)

    X, y = [], []
    for i in range(window - 1, n - horizon_hours):
        X.append(feature_values[i - window + 1: i + 1])
        y.append(target_values[i + horizon_hours])
    return np.array(X, dtype="float32"), np.array(y, dtype="float32")


def train_lstm(X_train, y_train, X_val=None, y_val=None, epochs=30):
    import tensorflow as tf

    n_features = X_train.shape[2]
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(WINDOW_HOURS, n_features)),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    optimizer = tf.keras.optimizers.Adam(clipnorm=1.0)
    model.compile(optimizer=optimizer, loss="mse")

    fit_kwargs = {"epochs": epochs, "batch_size": 64, "verbose": 0}
    if X_val is not None:
        fit_kwargs["validation_data"] = (X_val, y_val)
        fit_kwargs["callbacks"] = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        ]
    model.fit(X_train, y_train, **fit_kwargs)
    return model


def scale_sequences(X_train, X_test, scaler=None):
    n_features = X_train.shape[2]
    if scaler is None:
        scaler = StandardScaler()
        X_train_2d = X_train.reshape(-1, n_features)
        scaler.fit(X_train_2d)
    X_train_scaled = scaler.transform(X_train.reshape(-1, n_features)).reshape(X_train.shape).astype("float32")
    X_test_scaled = None
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape).astype("float32")
    return X_train_scaled, X_test_scaled, scaler


def evaluate_tabular_baseline(df_tabular: pd.DataFrame, horizon_hours: int) -> dict:
    X_all = df_tabular[FEATURE_COLUMNS]
    y_all_raw = df_tabular[TARGET_COLUMN].values
    y_all_log = np.log1p(y_all_raw)

    tscv = TimeSeriesSplit(n_splits=N_CV_FOLDS, gap=horizon_hours)
    fold_metrics = {"ridge": [], "gradient_boosting": []}

    for train_idx, test_idx in tscv.split(X_all):
        X_train_raw = X_all.iloc[train_idx]
        X_test_raw = X_all.iloc[test_idx]
        y_train = y_all_log[train_idx]
        y_test_actual = y_all_raw[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        ridge_model = train_ridge(X_train, y_train)
        ridge_preds = np.clip(np.expm1(ridge_model.predict(X_test)), 0, MAX_PLAUSIBLE_PM2_5)
        fold_metrics["ridge"].append(evaluate(y_test_actual, ridge_preds))

        gb_model = train_gradient_boosting(X_train, y_train)
        gb_preds = np.clip(np.expm1(gb_model.predict(X_test)), 0, MAX_PLAUSIBLE_PM2_5)
        fold_metrics["gradient_boosting"].append(evaluate(y_test_actual, gb_preds))

    avg = {}
    for name, metrics_list in fold_metrics.items():
        avg[name] = {
            "rmse": float(np.mean([m["rmse"] for m in metrics_list])),
            "mae": float(np.mean([m["mae"] for m in metrics_list])),
            "r2": float(np.mean([m["r2"] for m in metrics_list])),
        }
    return avg


def evaluate_lstm(df_seq: pd.DataFrame, horizon_hours: int) -> dict:
    X_all, y_all_raw = build_sequences(df_seq, horizon_hours)
    y_all_log = np.log1p(y_all_raw)

    tscv = TimeSeriesSplit(n_splits=N_CV_FOLDS, gap=horizon_hours)
    fold_metrics = []

    for fold_i, (train_idx, test_idx) in enumerate(tscv.split(X_all), start=1):
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test_log = y_all_log[train_idx], y_all_log[test_idx]
        y_test_actual = y_all_raw[test_idx]

        X_train_scaled, X_test_scaled, _ = scale_sequences(X_train, X_test)

        model = train_lstm(X_train_scaled, y_train, X_test_scaled, y_test_log)
        preds = np.clip(np.expm1(model.predict(X_test_scaled, verbose=0).flatten()), 0, MAX_PLAUSIBLE_PM2_5)
        m = evaluate(y_test_actual, preds)
        fold_metrics.append(m)
        print(f"    Fold {fold_i}/{N_CV_FOLDS} (train={len(train_idx)}, test={len(test_idx)})  "
              f"RMSE={m['rmse']:.2f}  MAE={m['mae']:.2f}  R2={m['r2']:.3f}")

    return {
        "rmse": float(np.mean([m["rmse"] for m in fold_metrics])),
        "mae": float(np.mean([m["mae"] for m in fold_metrics])),
        "r2": float(np.mean([m["r2"] for m in fold_metrics])),
    }


def main():
    import hopsworks

    print(f"Connecting to Hopsworks project '{HOPSWORKS_PROJECT_NAME}'...")
    project = hopsworks.login(
        project=HOPSWORKS_PROJECT_NAME, host=HOPSWORKS_HOST, port=443, api_key_value=HOPSWORKS_API_KEY,
    )
    fs = project.get_feature_store()

    print(f"Loading stored feature data for {CITY_NAME}...")
    df = load_feature_data(fs)
    print(f"  -> {len(df)} raw rows loaded")

    df = add_lag_features(df)
    df = add_cyclical_hour(df)

    for horizon_key, horizon_hours in HORIZONS.items():
        print(f"\n{'=' * 70}")
        print(f"HORIZON: {horizon_key} ({horizon_hours}h ahead)")
        print(f"{'=' * 70}")

        df_tabular = build_training_target(df, horizon_hours)
        print(f"  Tabular baseline re-check ({N_CV_FOLDS}-fold, {len(df_tabular)} rows)...")
        tabular_avg = evaluate_tabular_baseline(df_tabular, horizon_hours)
        best_tabular_name = min(tabular_avg, key=lambda n: tabular_avg[n]["rmse"])
        best_tabular = tabular_avg[best_tabular_name]
        print(f"    Best tabular ({best_tabular_name}): RMSE={best_tabular['rmse']:.3f}  "
              f"MAE={best_tabular['mae']:.3f}  R2={best_tabular['r2']:.3f}")

        print(f"  LSTM ({N_CV_FOLDS}-fold, window={WINDOW_HOURS}h)...")
        lstm_avg = evaluate_lstm(df, horizon_hours)
        print(f"    LSTM: RMSE={lstm_avg['rmse']:.3f}  MAE={lstm_avg['mae']:.3f}  R2={lstm_avg['r2']:.3f}")

        if lstm_avg["rmse"] < best_tabular["rmse"]:
            print(f"  LSTM WINS for {horizon_key} (RMSE {lstm_avg['rmse']:.3f} < {best_tabular['rmse']:.3f}). "
                  f"Retraining on full data and registering...")

            X_full, y_full_raw = build_sequences(df, horizon_hours)
            y_full_log = np.log1p(y_full_raw)
            X_full_scaled, _, final_scaler = scale_sequences(X_full, None)
            final_model = train_lstm(X_full_scaled, y_full_log, epochs=30)

            save_dir = f"saved_model_{horizon_key}_lstm"
            os.makedirs(save_dir, exist_ok=True)
            joblib.dump(final_scaler, f"{save_dir}/scaler.pkl")
            final_model.save(f"{save_dir}/model.keras")
            with open(f"{save_dir}/model_type.txt", "w") as f:
                f.write("lstm")
            with open(f"{save_dir}/inference_notes.txt", "w") as f:
                f.write("target_transform=log1p (apply np.expm1 to model output to get real PM2.5)\n")
                f.write(f"sequence_feature_columns={SEQ_FEATURE_COLUMNS}\n")
                f.write(f"window_hours={WINDOW_HOURS}\n")
                f.write(f"forecast_horizon_hours={horizon_hours}\n")
                f.write(f"horizon_key={horizon_key}\n")
                f.write("input_shape=(window_hours, n_sequence_features)\n")

            mr = project.get_model_registry()
            registry_name = MODEL_REGISTRY_NAME_TEMPLATE.format(horizon_key=horizon_key)
            model = mr.python.create_model(
                name=registry_name,
                metrics=lstm_avg,
                description=(
                    f"LSTM sequence model, {horizon_hours}h-ahead ({horizon_key}) PM2.5 forecast for "
                    f"{CITY_NAME}. Uses raw {WINDOW_HOURS}h lookback window instead of hand-picked lag "
                    f"features. Selected via {N_CV_FOLDS}-fold CV, beat the best tabular model "
                    f"({best_tabular_name}, RMSE={best_tabular['rmse']:.3f}) with RMSE={lstm_avg['rmse']:.3f}."
                ),
            )
            model.save(save_dir)
            print(f"  Registered new version: {registry_name}")
        else:
            print(f"  Tabular model keeps its place for {horizon_key} "
                  f"(LSTM RMSE {lstm_avg['rmse']:.3f} did not beat {best_tabular['rmse']:.3f}). Not registering.")

    print(f"\n{'=' * 70}")
    print("Done. Only horizons where the LSTM genuinely won were updated in the registry.")


if __name__ == "__main__":
    main()
