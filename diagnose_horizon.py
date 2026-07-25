"""
diagnose_horizon.py
---------------------
Quick standalone diagnostic -- NOT part of the deployed pipeline, doesn't
touch the Model Registry. Answers one question: is 72-hour-ahead PM2.5
forecasting just inherently much harder than a shorter horizon (like 24h),
or is something else going on?

WHY this matters: if 24h forecasting scores meaningfully better than 72h
using the exact same data/features/models, that's strong evidence the
72h task is just genuinely hard for this city (a legitimate, reportable
finding) rather than a bug in the pipeline. If 24h is ALSO poor, that
points to something still wrong elsewhere.

Uses only Ridge + Gradient Boosting (the two fastest, and GB was the best
performer in the main pipeline) to keep runtime short -- this is a
diagnostic, not a full model comparison.

Run manually:
    python diagnose_horizon.py
"""

import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from training_pipeline import (
    CITY_NAME,
    FEATURE_COLUMNS,
    HOPSWORKS_API_KEY,
    HOPSWORKS_HOST,
    HOPSWORKS_PROJECT_NAME,
    SOURCE_COLUMN_FOR_TARGET,
    add_lag_features,
    evaluate,
    load_feature_data,
    train_gradient_boosting,
    train_ridge,
)

N_CV_FOLDS = 5


def build_target_for_horizon(df, horizon_hours):
    df = df.copy()
    df["target"] = df[SOURCE_COLUMN_FOR_TARGET].shift(-horizon_hours)
    df = df.dropna(subset=FEATURE_COLUMNS + ["target"]).reset_index(drop=True)
    return df


def run_cv_for_horizon(df, horizon_hours):
    df = build_target_for_horizon(df, horizon_hours)
    X_all = df[FEATURE_COLUMNS]
    y_all_raw = df["target"].values
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
        ridge_preds = np.clip(np.expm1(ridge_model.predict(X_test)), 0, 1000)
        fold_metrics["ridge"].append(evaluate(y_test_actual, ridge_preds))

        gb_model = train_gradient_boosting(X_train, y_train)
        gb_preds = np.clip(np.expm1(gb_model.predict(X_test)), 0, 1000)
        fold_metrics["gradient_boosting"].append(evaluate(y_test_actual, gb_preds))

    avg = {}
    for name, metrics_list in fold_metrics.items():
        avg[name] = {
            "rmse": float(np.mean([m["rmse"] for m in metrics_list])),
            "mae": float(np.mean([m["mae"] for m in metrics_list])),
            "r2": float(np.mean([m["r2"] for m in metrics_list])),
        }
    return avg, len(df)


def main():
    import hopsworks

    print(f"Connecting to Hopsworks project '{HOPSWORKS_PROJECT_NAME}'...")
    project = hopsworks.login(
        project=HOPSWORKS_PROJECT_NAME, host=HOPSWORKS_HOST, port=443, api_key_value=HOPSWORKS_API_KEY,
    )
    fs = project.get_feature_store()

    print(f"Loading stored feature data for {CITY_NAME}...")
    df = load_feature_data(fs)
    df = add_lag_features(df)
    print(f"  -> {len(df)} raw rows loaded\n")

    for horizon in [24, 72]:
        print(f"=== Horizon: {horizon}h ahead ===")
        avg_metrics, n_rows = run_cv_for_horizon(df, horizon)
        print(f"  ({n_rows} usable rows, {N_CV_FOLDS}-fold CV, gap={horizon}h)")
        for name, m in avg_metrics.items():
            print(f"  {name:<18} RMSE={m['rmse']:6.2f}  MAE={m['mae']:6.2f}  R2={m['r2']:6.3f}")
        print()


if __name__ == "__main__":
    main()
