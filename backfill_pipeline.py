"""
backfill_pipeline.py
---------------------
STEP 2 of the AQI Predictor project (Phase C: Historical Backfill).

WHY this script exists:
    The training pipeline (Phase D) needs many past (features, target) rows
    to learn from. The hourly feature_pipeline.py only adds ONE new row per
    hour going forward -- that's too slow to build a useful training set from
    scratch. This script instead pulls REAL historical data in one go, so we
    have enough rows to train on immediately.

WHY two different data sources:
    - OpenWeather's free tier gives real historical POLLUTION data (their
      /air_pollution/history endpoint), but real historical WEATHER data
      requires a paid OpenWeather subscription.
    - Open-Meteo (open-meteo.com) provides real historical WEATHER data for
      free, with no API key required at all.
    So: pollution history from OpenWeather, weather history from Open-Meteo,
    merged together into the exact same feature schema feature_pipeline.py
    uses -- so training data and live data always match (no train/serve skew).

WHY chunked requests instead of one giant request:
    OpenWeather's free pollution history only goes back to 27 Nov 2020 --
    but requesting nearly 5.5 years of hourly data (~48,000 records) in a
    SINGLE API call risks timeouts or an oversized response. Instead this
    fetches in ~90-day chunks and concatenates the results -- slower, but
    far more reliable for a request this large.

WHY chunked writes to Hopsworks too:
    Inserting ~48,000 rows in one call is a lot to hold in memory and
    upload at once from a personal machine. Writing year-by-year gives
    incremental progress feedback and is more resilient -- if one chunk's
    write fails partway through, you don't lose everything before it.

Run manually:
    python backfill_pipeline.py
"""

import os
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# CONFIG (same pattern as feature_pipeline.py, for consistency)
# ---------------------------------------------------------------------------
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")
HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai")

CITY_NAME = os.getenv("CITY_NAME", "Rawalpindi")
CITY_LAT = float(os.getenv("CITY_LAT", "33.5651"))
CITY_LON = float(os.getenv("CITY_LON", "73.0169"))

BACKFILL_DAYS = int(os.getenv("BACKFILL_DAYS", "30"))  # how many past days to pull
CHUNK_DAYS = 90  # size of each individual API request

# OpenWeather's free Air Pollution History endpoint has no data before this
# date -- requesting further back just returns nothing, so we cap here
# regardless of how large BACKFILL_DAYS is set.
EARLIEST_AVAILABLE_DATE = datetime(2020, 11, 27, tzinfo=timezone.utc)

AIR_POLLUTION_HISTORY_URL = "http://api.openweathermap.org/data/2.5/air_pollution/history"
OPEN_METEO_HISTORY_URL = "https://archive-api.open-meteo.com/v1/archive"

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 3  # MUST match feature_pipeline.py -- backfill and live data
                            # have to land in the exact same feature group/schema


def date_chunks(start: datetime, end: datetime, chunk_days: int):
    """Yield (chunk_start, chunk_end) pairs covering [start, end] in steps of chunk_days."""
    current = start
    while current < end:
        chunk_end = min(current + timedelta(days=chunk_days), end)
        yield current, chunk_end
        current = chunk_end


def fetch_pollution_history(lat: float, lon: float, start: datetime, end: datetime) -> pd.DataFrame:
    """Real historical pollutant readings from OpenWeather, fetched in chunks."""
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
        time.sleep(0.5)  # be polite to the free-tier API between chunks
    return pd.DataFrame(all_rows)


def fetch_weather_history(lat: float, lon: float, start: datetime, end: datetime) -> pd.DataFrame:
    """Real historical weather from Open-Meteo (free, no API key needed, no chunking needed --
    Open-Meteo handles multi-year single requests fine, unlike the pollution endpoint)."""
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


def build_backfill_features(pollution_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge pollution + weather on the nearest matching hour, then compute the
    SAME engineered features feature_pipeline.py computes -- time-based
    features and aqi_change_rate -- so training data matches live data exactly.
    """
    # Round both to the nearest hour so timestamps line up for merging
    pollution_df["event_time"] = pollution_df["event_time"].dt.floor("h")
    weather_df["event_time"] = weather_df["event_time"].dt.floor("h")

    merged = pd.merge(pollution_df, weather_df, on="event_time", how="inner")
    merged = merged.sort_values("event_time").reset_index(drop=True)

    merged["city"] = CITY_NAME
    # Cast explicitly to int64 ("bigint" in Hopsworks). pandas' .dt accessor
    # returns 32-bit ints by default, but the feature group schema (created by
    # feature_pipeline.py, using native Python ints) expects 64-bit -- without
    # this cast, insert fails with a schema type mismatch.
    merged["hour"] = merged["event_time"].dt.hour.astype("int64")
    merged["day"] = merged["event_time"].dt.day.astype("int64")
    merged["month"] = merged["event_time"].dt.month.astype("int64")
    merged["day_of_week"] = merged["event_time"].dt.dayofweek.astype("int64")

    # aqi_change_rate: difference from the PREVIOUS row in this same sequence.
    # First row has no prior reading, so it defaults to 0 -- same rule
    # feature_pipeline.py uses for its very first-ever run.
    merged["aqi_change_rate"] = merged["aqi"].diff().fillna(0.0)

    return merged


def write_backfill_to_feature_store(fs, df: pd.DataFrame):
    fg = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        description="Hourly AQI features per city for AQI forecasting",
        primary_key=["city"],
        event_time="event_time",
        online_enabled=True,
        time_travel_format="HUDI",
    )
    # Write year-by-year rather than all at once: gives incremental progress
    # feedback on a large backfill, and if one chunk fails you don't lose
    # everything that already succeeded before it.
    df = df.copy()
    df["_year"] = df["event_time"].dt.year
    years = sorted(df["_year"].unique())
    for year in years:
        year_df = df[df["_year"] == year].drop(columns=["_year"])
        print(f"    Writing {len(year_df)} rows for {year}...")
        fg.insert(year_df)
    print(f"Inserted {len(df)} backfilled rows total for {CITY_NAME} across {len(years)} year(s)")


def main():
    if not OPENWEATHER_API_KEY:
        sys.exit("Missing OPENWEATHER_API_KEY. Set it in .env or as a GitHub Actions secret.")
    if not HOPSWORKS_API_KEY or not HOPSWORKS_PROJECT_NAME:
        sys.exit("Missing HOPSWORKS_API_KEY / HOPSWORKS_PROJECT_NAME. Set them in .env or as secrets.")

    import hopsworks

    end = datetime.now(timezone.utc)
    requested_start = end - timedelta(days=BACKFILL_DAYS)
    # Never request further back than OpenWeather actually has data for --
    # doing so wouldn't error, it would just silently return nothing for
    # the out-of-range portion.
    start = max(requested_start, EARLIEST_AVAILABLE_DATE)

    print(f"Backfilling history for {CITY_NAME} ({start.date()} to {end.date()}, "
          f"{(end - start).days} days)...")
    if requested_start < EARLIEST_AVAILABLE_DATE:
        print(f"  Note: BACKFILL_DAYS={BACKFILL_DAYS} requested further back than OpenWeather's "
              f"free data covers -- capped at {EARLIEST_AVAILABLE_DATE.date()}.")

    print("Fetching historical pollution data (OpenWeather, chunked)...")
    pollution_df = fetch_pollution_history(CITY_LAT, CITY_LON, start, end)
    print(f"  -> {len(pollution_df)} hourly pollution records")

    print("Fetching historical weather data (Open-Meteo)...")
    weather_df = fetch_weather_history(CITY_LAT, CITY_LON, start, end)
    print(f"  -> {len(weather_df)} hourly weather records")

    print("Merging and building features...")
    features_df = build_backfill_features(pollution_df, weather_df)
    print(f"  -> {len(features_df)} merged feature rows")

    if features_df.empty:
        sys.exit("No overlapping data between pollution and weather history -- nothing to backfill.")

    print(f"Connecting to Hopsworks project '{HOPSWORKS_PROJECT_NAME}' on {HOPSWORKS_HOST}...")
    project = hopsworks.login(
        project=HOPSWORKS_PROJECT_NAME,
        host=HOPSWORKS_HOST,
        port=443,
        api_key_value=HOPSWORKS_API_KEY,
    )
    fs = project.get_feature_store()

    print("Writing backfilled rows to Hopsworks Feature Store (year by year)...")
    write_backfill_to_feature_store(fs, features_df)


if __name__ == "__main__":
    main()
