import os
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from dotenv import load_dotenv

from cities_config import CITIES

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")
HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai")

BACKFILL_DAYS = int(os.getenv("BACKFILL_DAYS", "30"))
CHUNK_DAYS = 90

EARLIEST_AVAILABLE_DATE = datetime(2020, 11, 27, tzinfo=timezone.utc)

AIR_POLLUTION_HISTORY_URL = "http://api.openweathermap.org/data/2.5/air_pollution/history"
OPEN_METEO_HISTORY_URL = "https://archive-api.open-meteo.com/v1/archive"

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 3


def date_chunks(start: datetime, end: datetime, chunk_days: int):
    current = start
    while current < end:
        chunk_end = min(current + timedelta(days=chunk_days), end)
        yield current, chunk_end
        current = chunk_end


def fetch_pollution_history(lat: float, lon: float, start: datetime, end: datetime) -> pd.DataFrame:
    all_rows = []
    chunks = list(date_chunks(start, end, CHUNK_DAYS))
    for i, (chunk_start, chunk_end) in enumerate(chunks, start=1):
        print(f"    Pollution chunk {i}/{len(chunks)}: {chunk_start.date()} to {chunk_end.date()}")
        params = {
            "lat": lat,
            "lon": lon,
            "start": int(chunk_start.timestamp()),
            "end": int(chunk_end.timestamp()),
            "appid": OPENWEATHER_API_KEY,
        }
        resp = requests.get(AIR_POLLUTION_HISTORY_URL, params=params, timeout=60)
        resp.raise_for_status()
        records = resp.json()["list"]
        for r in records:
            all_rows.append({
                "event_time": datetime.fromtimestamp(r["dt"], tz=timezone.utc),
                "aqi": int(r["main"]["aqi"]),
                "co": float(r["components"]["co"]),
                "no": float(r["components"]["no"]),
                "no2": float(r["components"]["no2"]),
                "o3": float(r["components"]["o3"]),
                "so2": float(r["components"]["so2"]),
                "pm2_5": float(r["components"]["pm2_5"]),
                "pm10": float(r["components"]["pm10"]),
                "nh3": float(r["components"]["nh3"]),
            })
        time.sleep(0.5)
    return pd.DataFrame(all_rows)


def fetch_weather_history(lat: float, lon: float, start: datetime, end: datetime) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m",
        "timezone": "UTC",
    }
    resp = requests.get(OPEN_METEO_HISTORY_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()["hourly"]

    df = pd.DataFrame({
        "event_time": pd.to_datetime(data["time"], utc=True),
        "temperature": [float(v) for v in data["temperature_2m"]],
        "humidity": [float(v) for v in data["relative_humidity_2m"]],
        "pressure": [float(v) for v in data["surface_pressure"]],
        "wind_speed": [float(v) for v in data["wind_speed_10m"]],
        "wind_deg": [float(v) for v in data["wind_direction_10m"]],
    })
    return df


def build_backfill_features(city_name: str, pollution_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    pollution_df["event_time"] = pollution_df["event_time"].dt.floor("h")
    weather_df["event_time"] = weather_df["event_time"].dt.floor("h")

    merged = pd.merge(pollution_df, weather_df, on="event_time", how="inner")
    merged = merged.sort_values("event_time").reset_index(drop=True)

    merged["city"] = city_name
    merged["hour"] = merged["event_time"].dt.hour.astype("int64")
    merged["day"] = merged["event_time"].dt.day.astype("int64")
    merged["month"] = merged["event_time"].dt.month.astype("int64")
    merged["day_of_week"] = merged["event_time"].dt.dayofweek.astype("int64")

    merged["aqi_change_rate"] = merged["aqi"].diff().fillna(0.0)

    return merged


def write_backfill_to_feature_store(fs, city_name: str, df: pd.DataFrame):
    fg = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        description="Hourly AQI features per city for AQI forecasting",
        primary_key=["city"],
        event_time="event_time",
        online_enabled=True,
        time_travel_format="HUDI",
    )
    df = df.copy()
    df["_year"] = df["event_time"].dt.year
    years = sorted(df["_year"].unique())
    for year in years:
        year_df = df[df["_year"] == year].drop(columns=["_year"])
        print(f"    Writing {len(year_df)} rows for {year}...")
        fg.insert(year_df)
    print(f"Inserted {len(df)} backfilled rows total for {city_name} across {len(years)} year(s)")


def run_for_city(fs, city_key: str, city_info: dict, start: datetime, end: datetime):
    city_name = city_info["name"]
    lat = city_info["lat"]
    lon = city_info["lon"]

    print(f"\n=== {city_name} ===")
    print("Fetching historical pollution data (OpenWeather, chunked)...")
    pollution_df = fetch_pollution_history(lat, lon, start, end)
    print(f"  -> {len(pollution_df)} hourly pollution records")

    print("Fetching historical weather data (Open-Meteo)...")
    weather_df = fetch_weather_history(lat, lon, start, end)
    print(f"  -> {len(weather_df)} hourly weather records")

    print("Merging and building features...")
    features_df = build_backfill_features(city_name, pollution_df, weather_df)
    print(f"  -> {len(features_df)} merged feature rows")

    if features_df.empty:
        print(f"  No overlapping data for {city_name} -- skipping.")
        return

    print(f"Writing backfilled rows to Hopsworks Feature Store for {city_name} (year by year)...")
    write_backfill_to_feature_store(fs, city_name, features_df)


def main():
    if not OPENWEATHER_API_KEY:
        sys.exit("Missing OPENWEATHER_API_KEY. Set it in .env or as a GitHub Actions secret.")
    if not HOPSWORKS_API_KEY or not HOPSWORKS_PROJECT_NAME:
        sys.exit("Missing HOPSWORKS_API_KEY / HOPSWORKS_PROJECT_NAME. Set them in .env or as secrets.")

    import hopsworks

    end = datetime.now(timezone.utc)
    requested_start = end - timedelta(days=BACKFILL_DAYS)
    start = max(requested_start, EARLIEST_AVAILABLE_DATE)

    print(f"Backfilling {(end - start).days} days ({start.date()} to {end.date()}) "
          f"for {len(CITIES)} cities: {', '.join(c['name'] for c in CITIES.values())}")
    if requested_start < EARLIEST_AVAILABLE_DATE:
        print(f"  Note: BACKFILL_DAYS={BACKFILL_DAYS} requested further back than OpenWeather's "
              f"free data covers -- capped at {EARLIEST_AVAILABLE_DATE.date()}.")

    print(f"Connecting to Hopsworks project '{HOPSWORKS_PROJECT_NAME}' on {HOPSWORKS_HOST}...")
    project = hopsworks.login(
        project=HOPSWORKS_PROJECT_NAME,
        host=HOPSWORKS_HOST,
        port=443,
        api_key_value=HOPSWORKS_API_KEY,
    )
    fs = project.get_feature_store()

    # WHY loop over CITIES: same reasoning as feature_pipeline.py -- this
    # is the scalability point of Phase 1. One connection to Hopsworks,
    # then each city backfills independently using identical logic.
    errors = []
    for city_key, city_info in CITIES.items():
        try:
            run_for_city(fs, city_key, city_info, start, end)
        except Exception as e:
            print(f"  FAILED for {city_info['name']}: {e}")
            errors.append(city_info["name"])

    if errors:
        sys.exit(f"\nCompleted with failures for: {', '.join(errors)}")
    print(f"\nBackfill completed successfully for all {len(CITIES)} cities.")


if __name__ == "__main__":
    main()
