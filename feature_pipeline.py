"""
feature_pipeline.py
--------------------
STEP 1 of the AQI Predictor project.

What this script does (in order):
    1. Loads config/secrets from environment variables (.env locally, GitHub
       Actions secrets in production).
    2. Calls two OpenWeather endpoints:
         - Air Pollution API  -> raw pollutant concentrations + OpenWeather's own AQI
         - Current Weather API -> temperature, humidity, wind, pressure
       (weather drives pollution dispersion, so it's a critical model input)
    3. Merges both into a single row of "raw data".
    4. Computes ENGINEERED FEATURES from that raw row:
         - time-based features (hour, day, month, day_of_week)
         - derived features (aqi_change_rate vs the last stored reading)
    5. Writes the final feature row into the Hopsworks Feature Store.

Run manually:
    python feature_pipeline.py

Run on a schedule (hourly) via GitHub Actions -> see .github/workflows/feature_pipeline.yml
"""

import os
import sys
from datetime import datetime, timezone

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()  # loads .env file if present; in CI, real env vars are used instead

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")
HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai")

CITY_NAME = os.getenv("CITY_NAME", "Rawalpindi")
CITY_LAT = float(os.getenv("CITY_LAT", "33.5651"))
CITY_LON = float(os.getenv("CITY_LON", "73.0169"))

AIR_POLLUTION_URL = "http://api.openweathermap.org/data/2.5/air_pollution"
WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 3  # bumped from 2: v2's schema got locked in with 'no' as an
                            # integer on its first insert, then broke on the next run
                            # when OpenWeather returned a decimal for the same field.
                            # v3 forces consistent float typing in fetch_air_pollution()
                            # and fetch_weather() so this can't happen again.


def fetch_air_pollution(lat: float, lon: float) -> dict:
    """Step 2a: raw pollutant data + OpenWeather's own AQI (1-5 scale)."""
    params = {"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY}
    resp = requests.get(AIR_POLLUTION_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()["list"][0]
    return {
        # aqi is OpenWeather's own 1-5 index, always a whole number -> keep as int
        "aqi": int(data["main"]["aqi"]),
        # Force every pollutant reading to float. OpenWeather sometimes returns a
        # whole number (e.g. 0) and sometimes a decimal (e.g. 0.02) for the same
        # field across different calls. Hopsworks locks a feature group's column
        # type on first insert, so if the type isn't forced consistently here,
        # a later run with a different type crashes with a schema mismatch error.
        "co": float(data["components"]["co"]),
        "no": float(data["components"]["no"]),
        "no2": float(data["components"]["no2"]),
        "o3": float(data["components"]["o3"]),
        "so2": float(data["components"]["so2"]),
        "pm2_5": float(data["components"]["pm2_5"]),
        "pm10": float(data["components"]["pm10"]),
        "nh3": float(data["components"]["nh3"]),
    }


def fetch_weather(lat: float, lon: float) -> dict:
    """Step 2b: weather conditions that influence how pollution disperses."""
    params = {"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY, "units": "metric"}
    resp = requests.get(WEATHER_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    return {
        # Same fix as above: force consistent types so the schema never breaks.
        "temperature": float(data["main"]["temp"]),
        "humidity": float(data["main"]["humidity"]),
        "pressure": float(data["main"]["pressure"]),
        "wind_speed": float(data["wind"]["speed"]),
        "wind_deg": float(data["wind"].get("deg", 0)),
    }


def get_previous_aqi(fs) -> float | None:
    """
    Step 4b helper: read the most recent stored AQI for this city so we can
    compute a rate of change. Returns None if this is the very first run
    (nothing stored yet).
    """
    try:
        fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
        df = fg.read()
        city_df = df[df["city"] == CITY_NAME].sort_values("event_time")
        if city_df.empty:
            return None
        return float(city_df.iloc[-1]["aqi"])
    except Exception:
        # Feature group doesn't exist yet, or store is empty -> treat as first run
        return None


def build_feature_row(raw_pollution: dict, raw_weather: dict, previous_aqi: float | None) -> dict:
    """
    Step 4: turn raw API data into the actual columns the model will train on.
    This is the ONLY function the training pipeline and the backfill script
    also depend on conceptually — keeping feature logic in one place avoids
    train/serve skew (a very common real-world ML bug).
    """
    now = datetime.now(timezone.utc)

    row = {
        "city": CITY_NAME,
        "event_time": now,
        "hour": now.hour,
        "day": now.day,
        "month": now.month,
        "day_of_week": now.weekday(),  # 0=Monday, helps model learn weekday/weekend pollution patterns
        **raw_pollution,
        **raw_weather,
    }

    # Derived feature: AQI change rate vs previous reading.
    # This tells the model whether air quality is trending up or down,
    # not just its current absolute level.
    if previous_aqi is not None:
        row["aqi_change_rate"] = row["aqi"] - previous_aqi
    else:
        row["aqi_change_rate"] = 0.0

    return row


def write_to_feature_store(fs, row: dict):
    """Step 5: insert the computed feature row into Hopsworks."""
    df = pd.DataFrame([row])

    fg = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        description="Hourly AQI features per city for AQI forecasting",
        primary_key=["city"],
        event_time="event_time",
        online_enabled=True,  # lets the web app read the LATEST row fast, without a full table scan
        time_travel_format="HUDI",  # avoids DELTA's direct-RPC-to-HopsFS requirement, which
                                     # external clients (GitHub Actions, local dev machines)
                                     # cannot reach; HUDI writes go through Hopsworks' REST
                                     # ingestion service instead
    )
    fg.insert(df)
    print(f"Inserted feature row for {row['city']} at {row['event_time']}")


def main():
    if not OPENWEATHER_API_KEY:
        sys.exit("Missing OPENWEATHER_API_KEY. Set it in .env or as a GitHub Actions secret.")
    if not HOPSWORKS_API_KEY or not HOPSWORKS_PROJECT_NAME:
        sys.exit("Missing HOPSWORKS_API_KEY / HOPSWORKS_PROJECT_NAME. Set them in .env or as secrets.")

    import hopsworks  # imported here so the script still runs --help/-checks without the dep installed

    print(f"Connecting to Hopsworks project '{HOPSWORKS_PROJECT_NAME}' on {HOPSWORKS_HOST}...")
    project = hopsworks.login(
        project=HOPSWORKS_PROJECT_NAME,
        host=HOPSWORKS_HOST,
        port=443,
        api_key_value=HOPSWORKS_API_KEY,
    )
    fs = project.get_feature_store()

    print(f"Fetching raw pollution + weather data for {CITY_NAME}...")
    raw_pollution = fetch_air_pollution(CITY_LAT, CITY_LON)
    raw_weather = fetch_weather(CITY_LAT, CITY_LON)

    print("Looking up previous AQI reading for change-rate calculation...")
    previous_aqi = get_previous_aqi(fs)

    print("Building feature row...")
    row = build_feature_row(raw_pollution, raw_weather, previous_aqi)

    print("Writing feature row to Hopsworks Feature Store...")
    write_to_feature_store(fs, row)


if __name__ == "__main__":
    main()