"""
diagnose_data.py
-----------------
Quick one-off diagnostic -- NOT part of the pipeline. Checks whether the
stored AQI data actually has enough variation to be predictable at all,
before we spend more time tuning models.

Run manually:
    python diagnose_data.py
"""

import os

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")
HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai")
CITY_NAME = os.getenv("CITY_NAME", "Rawalpindi")


def main():
    import hopsworks

    project = hopsworks.login(
        project=HOPSWORKS_PROJECT_NAME,
        host=HOPSWORKS_HOST,
        port=443,
        api_key_value=HOPSWORKS_API_KEY,
    )
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name="aqi_features", version=3)
    df = fg.read()
    df = df[df["city"] == CITY_NAME].sort_values("event_time").reset_index(drop=True)

    print(f"\nTotal rows: {len(df)}")
    print(f"Date range: {df['event_time'].min()} to {df['event_time'].max()}")

    print("\n--- AQI (OpenWeather 1-5 index) value counts ---")
    print(df["aqi"].value_counts().sort_index())

    print("\n--- AQI descriptive stats ---")
    print(df["aqi"].describe())

    print("\n--- PM2.5 descriptive stats (continuous pollutant) ---")
    print(df["pm2_5"].describe())

    print("\n--- Correlation of current aqi with aqi 72h later ---")
    df["aqi_72h_later"] = df["aqi"].shift(-72)
    corr = df[["aqi", "aqi_72h_later"]].corr().iloc[0, 1]
    print(f"Correlation: {corr:.3f}")

    print("\n--- Correlation of current pm2_5 with pm2_5 72h later ---")
    df["pm2_5_72h_later"] = df["pm2_5"].shift(-72)
    corr2 = df[["pm2_5", "pm2_5_72h_later"]].corr().iloc[0, 1]
    print(f"Correlation: {corr2:.3f}")

    # Check the most recent 15% (our test set) vs the rest, since a big
    # difference here would explain poor test performance on its own.
    split_index = int(len(df) * 0.85)
    train_part = df.iloc[:split_index]
    test_part = df.iloc[split_index:]
    print("\n--- Train period AQI stats ---")
    print(train_part["aqi"].describe())
    print("\n--- Test period AQI stats (most recent ~15%) ---")
    print(test_part["aqi"].describe())


if __name__ == "__main__":
    main()
