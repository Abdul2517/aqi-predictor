import os
import sys
from datetime import datetime, timezone

import pandas as pd
import requests
from dotenv import load_dotenv

from cities_config import CITIES

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")
HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai")

AIR_POLLUTION_URL = "http://api.openweathermap.org/data/2.5/air_pollution"
WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 3


def fetch_air_pollution(lat: float, lon: float) -> dict:
    params = {"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY}
    resp = requests.get(AIR_POLLUTION_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()["list"][0]
    return {
        "aqi": int(data["main"]["aqi"]),
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
    params = {"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY, "units": "metric"}
    resp = requests.get(WEATHER_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    return {
        "temperature": float(data["main"]["temp"]),
        "humidity": float(data["main"]["humidity"]),
        "pressure": float(data["main"]["pressure"]),
        "wind_speed": float(data["wind"]["speed"]),
        "wind_deg": float(data["wind"].get("deg", 0)),
    }


def get_previous_aqi(fs, city_name: str) -> float | None:
    # WHY filter instead of fg.read(): fg.read() pulls the ENTIRE feature
    # group (every city, full history -- 175k+ rows) just to find one
    # city's last AQI value. That happens up to 4x per hourly run (once
    # per city), which is exactly the kind of unnecessary Query Service
    # load Hopsworks flagged. Filtering by city on the query itself keeps
    # this to only that city's rows instead of the whole table.
    try:
        fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
        query = fg.filter(fg.city == city_name)
        city_df = query.read()
        if city_df.empty:
            return None
        city_df = city_df.sort_values("event_time")
        return float(city_df.iloc[-1]["aqi"])
    except Exception:
        return None


def build_feature_row(city_name: str, raw_pollution: dict, raw_weather: dict, previous_aqi: float | None) -> dict:
    now = datetime.now(timezone.utc)

    row = {
        "city": city_name,
        "event_time": now,
        "hour": now.hour,
        "day": now.day,
        "month": now.month,
        "day_of_week": now.weekday(),
        **raw_pollution,
        **raw_weather,
    }

    if previous_aqi is not None:
        row["aqi_change_rate"] = row["aqi"] - previous_aqi
    else:
        row["aqi_change_rate"] = 0.0

    return row


def write_to_feature_store(fs, row: dict):
    df = pd.DataFrame([row])

    fg = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        description="Hourly AQI features per city for AQI forecasting",
        primary_key=["city"],
        event_time="event_time",
        online_enabled=True,
        time_travel_format="HUDI",
    )
    fg.insert(df)
    print(f"Inserted feature row for {row['city']} at {row['event_time']}")


def run_for_city(fs, city_key: str, city_info: dict):
    city_name = city_info["name"]
    lat = city_info["lat"]
    lon = city_info["lon"]

    print(f"\n=== {city_name} ===")
    print(f"Fetching raw pollution + weather data for {city_name}...")
    raw_pollution = fetch_air_pollution(lat, lon)
    raw_weather = fetch_weather(lat, lon)

    print("Looking up previous AQI reading for change-rate calculation...")
    previous_aqi = get_previous_aqi(fs, city_name)

    print("Building feature row...")
    row = build_feature_row(city_name, raw_pollution, raw_weather, previous_aqi)

    print("Writing feature row to Hopsworks Feature Store...")
    write_to_feature_store(fs, row)


def main():
    if not OPENWEATHER_API_KEY:
        sys.exit("Missing OPENWEATHER_API_KEY. Set it in .env or as a GitHub Actions secret.")
    if not HOPSWORKS_API_KEY or not HOPSWORKS_PROJECT_NAME:
        sys.exit("Missing HOPSWORKS_API_KEY / HOPSWORKS_PROJECT_NAME. Set them in .env or as secrets.")

    import hopsworks

    print(f"Connecting to Hopsworks project '{HOPSWORKS_PROJECT_NAME}' on {HOPSWORKS_HOST}...")
    project = hopsworks.login(
        project=HOPSWORKS_PROJECT_NAME,
        host=HOPSWORKS_HOST,
        port=443,
        api_key_value=HOPSWORKS_API_KEY,
    )
    fs = project.get_feature_store()

    errors = []
    for city_key, city_info in CITIES.items():
        try:
            run_for_city(fs, city_key, city_info)
        except Exception as e:
            print(f"  FAILED for {city_info['name']}: {e}")
            errors.append(city_info["name"])

    if errors:
        sys.exit(f"\nCompleted with failures for: {', '.join(errors)}")
    print(f"\nCompleted successfully for all {len(CITIES)} cities.")


if __name__ == "__main__":
    main()
