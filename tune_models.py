import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
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
    TARGET_COLUMN,
    add_lag_features,
    build_training_target,
    evaluate,
    load_feature_data,
)
from train_lstm import SEQ_FEATURE_COLUMNS, add_cyclical_hour, build_sequences, scale_sequences

N_CV_FOLDS = 3

CURRENT_PRODUCTION_EQUIVALENT = {
    "day1": "lstm_w48_1layer",
    "day2": "ridge_a1",
    "day3": "ridge_a1",
}

TABULAR_CANDIDATE_BUILDERS = {
    "ridge_a0.1": lambda: Ridge(alpha=0.1, random_state=42),
    "ridge_a1": lambda: Ridge(alpha=1.0, random_state=42),
    "ridge_a10": lambda: Ridge(alpha=10.0, random_state=42),
    "ridge_a50": lambda: Ridge(alpha=50.0, random_state=42),
    "rf_tuned": lambda: RandomForestRegressor(
        n_estimators=400, max_depth=16, min_samples_leaf=3, random_state=42, n_jobs=-1
    ),
    "gb_tuned": lambda: HistGradientBoostingRegressor(
        max_iter=500, max_depth=8, learning_rate=0.03, random_state=42
    ),
}

LSTM_CANDIDATE_CONFIGS = {
    "lstm_w48_1layer": {"window": 48, "layers": [32], "epochs": 30},
    "lstm_w72_2layer": {"window": 72, "layers": [64, 32], "epochs": 40},
}


def build_lstm(input_window, n_features, layer_units, X_train, y_train, X_val=None, y_val=None, epochs=30):
    import tensorflow as tf

    layers = [tf.keras.layers.Input(shape=(input_window, n_features))]
    for i, units in enumerate(layer_units):
        return_sequences = i < len(layer_units) - 1
        layers.append(tf.keras.layers.LSTM(units, return_sequences=return_sequences))
    layers.append(tf.keras.layers.Dense(16, activation="relu"))
    layers.append(tf.keras.layers.Dense(1))

    model = tf.keras.Sequential(layers)
    optimizer = tf.keras.optimizers.Adam(clipnorm=1.0)
    model.compile(optimizer=optimizer, loss="mse")

    fit_kwargs = {"epochs": epochs, "batch_size": 64, "verbose": 0}
    if X_val is not None:
        fit_kwargs["validation_data"] = (X_val, y_val)
        fit_kwargs["callbacks"] = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
        ]
    model.fit(X_train, y_train, **fit_kwargs)
    return model


def evaluate_tabular_candidates(df_tabular: pd.DataFrame, horizon_hours: int) -> dict:
    X_all = df_tabular[FEATURE_COLUMNS]
    y_all_raw = df_tabular[TARGET_COLUMN].values
    y_all_log = np.log1p(y_all_raw)

    tscv = TimeSeriesSplit(n_splits=N_CV_FOLDS, gap=horizon_hours)
    fold_metrics = {name: [] for name in TABULAR_CANDIDATE_BUILDERS}

    for train_idx, test_idx in tscv.split(X_all):
        X_train_raw, X_test_raw = X_all.iloc[train_idx], X_all.iloc[test_idx]
        y_train = y_all_log[train_idx]
        y_test_actual = y_all_raw[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        for name, builder in TABULAR_CANDIDATE_BUILDERS.items():
            model = builder()
            model.fit(X_train, y_train)
            preds = np.clip(np.expm1(model.predict(X_test)), 0, MAX_PLAUSIBLE_PM2_5)
            fold_metrics[name].append(evaluate(y_test_actual, preds))

    return {
        name: {
            "rmse": float(np.mean([m["rmse"] for m in ms])),
            "mae": float(np.mean([m["mae"] for m in ms])),
            "r2": float(np.mean([m["r2"] for m in ms])),
        }
        for name, ms in fold_metrics.items()
    }


def evaluate_lstm_candidates(df_seq: pd.DataFrame, horizon_hours: int) -> dict:
    results = {}
    for name, cfg in LSTM_CANDIDATE_CONFIGS.items():
        X_all, y_all_raw = build_sequences(df_seq, horizon_hours, window=cfg["window"])
        y_all_log = np.log1p(y_all_raw)
        n_features = X_all.shape[2]

        tscv = TimeSeriesSplit(n_splits=N_CV_FOLDS, gap=horizon_hours)
        fold_metrics = []
        for train_idx, test_idx in tscv.split(X_all):
            X_train, X_test = X_all[train_idx], X_all[test_idx]
            y_train, y_test_log = y_all_log[train_idx], y_all_log[test_idx]
            y_test_actual = y_all_raw[test_idx]

            X_train_scaled, X_test_scaled, _ = scale_sequences(X_train, X_test)
            model = build_lstm(cfg["window"], n_features, cfg["layers"],
                                X_train_scaled, y_train, X_test_scaled, y_test_log, epochs=cfg["epochs"])
            preds = np.clip(np.expm1(model.predict(X_test_scaled, verbose=0).flatten()), 0, MAX_PLAUSIBLE_PM2_5)
            fold_metrics.append(evaluate(y_test_actual, preds))

        results[name] = {
            "rmse": float(np.mean([m["rmse"] for m in fold_metrics])),
            "mae": float(np.mean([m["mae"] for m in fold_metrics])),
            "r2": float(np.mean([m["r2"] for m in fold_metrics])),
        }
        print(f"    {name:<20} RMSE={results[name]['rmse']:6.2f}  "
              f"MAE={results[name]['mae']:6.2f}  R2={results[name]['r2']:6.3f}")
    return results


def retrain_and_register(project, df, horizon_key, horizon_hours, winner_name, winner_metrics):
    print(f"  Retraining winner '{winner_name}' on full data and registering...")

    save_dir = f"saved_model_{horizon_key}_tuned"
    os.makedirs(save_dir, exist_ok=True)

    if winner_name in TABULAR_CANDIDATE_BUILDERS:
        df_tabular = build_training_target(df, horizon_hours)
        X_all_raw = df_tabular[FEATURE_COLUMNS]
        y_all_log = np.log1p(df_tabular[TARGET_COLUMN].values)
        scaler = StandardScaler()
        X_all = scaler.fit_transform(X_all_raw)
        model = TABULAR_CANDIDATE_BUILDERS[winner_name]()
        model.fit(X_all, y_all_log)
        joblib.dump(scaler, f"{save_dir}/scaler.pkl")
        joblib.dump(model, f"{save_dir}/model.pkl")
        model_type_label = winner_name
    else:
        cfg = LSTM_CANDIDATE_CONFIGS[winner_name]
        X_all, y_all_raw = build_sequences(df, horizon_hours, window=cfg["window"])
        y_all_log = np.log1p(y_all_raw)
        X_all_scaled, _, scaler = scale_sequences(X_all, None)
        model = build_lstm(cfg["window"], X_all.shape[2], cfg["layers"], X_all_scaled, y_all_log, epochs=cfg["epochs"])
        joblib.dump(scaler, f"{save_dir}/scaler.pkl")
        model.save(f"{save_dir}/model.keras")
        model_type_label = winner_name

    with open(f"{save_dir}/model_type.txt", "w") as f:
        f.write(model_type_label)

    mr = project.get_model_registry()
    registry_name = MODEL_REGISTRY_NAME_TEMPLATE.format(horizon_key=horizon_key)
    reg_model = mr.python.create_model(
        name=registry_name,
        metrics=winner_metrics,
        description=f"Tuned candidate '{winner_name}' for {horizon_key} ({horizon_hours}h), "
                     f"RMSE={winner_metrics['rmse']:.3f} MAE={winner_metrics['mae']:.3f} R2={winner_metrics['r2']:.3f}. "
                     f"Selected via {N_CV_FOLDS}-fold CV tuning pass over Ridge/RF/GB configs and LSTM variants.",
    )
    reg_model.save(save_dir)
    print(f"  Registered new version: {registry_name}")


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
        print(f"  Testing tabular candidates ({N_CV_FOLDS}-fold, {len(df_tabular)} rows)...")
        tabular_results = evaluate_tabular_candidates(df_tabular, horizon_hours)
        for name, m in tabular_results.items():
            print(f"    {name:<20} RMSE={m['rmse']:6.2f}  MAE={m['mae']:6.2f}  R2={m['r2']:6.3f}")

        print(f"  Testing LSTM candidates ({N_CV_FOLDS}-fold)...")
        lstm_results = evaluate_lstm_candidates(df, horizon_hours)

        all_results = {**tabular_results, **lstm_results}
        winner_name = min(all_results, key=lambda n: all_results[n]["rmse"])
        winner_metrics = all_results[winner_name]
        current_equiv = CURRENT_PRODUCTION_EQUIVALENT[horizon_key]
        current_metrics = all_results[current_equiv]

        print(f"\n  Current production equivalent ({current_equiv}): "
              f"RMSE={current_metrics['rmse']:.3f}  R2={current_metrics['r2']:.3f}")
        print(f"  Best candidate overall ({winner_name}): "
              f"RMSE={winner_metrics['rmse']:.3f}  R2={winner_metrics['r2']:.3f}")

        if winner_name != current_equiv and winner_metrics["rmse"] < current_metrics["rmse"]:
            print(f"  IMPROVEMENT FOUND for {horizon_key}: '{winner_name}' beats '{current_equiv}'.")
            retrain_and_register(project, df, horizon_key, horizon_hours, winner_name, winner_metrics)
        else:
            print(f"  No improvement for {horizon_key} -- keeping current production model.")

    print(f"\n{'=' * 70}")
    print("Tuning pass complete.")


if __name__ == "__main__":
    main()
