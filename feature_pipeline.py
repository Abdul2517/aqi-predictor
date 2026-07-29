import os
import sys
from datetime import datetime, timezone

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

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


def get_previous_aqi(fs) -> float | None:
    try:
        fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
        df = fg.read()
        city_df = df[df["city"] == CITY_NAME].sort_values("event_time")
        if city_df.empty:
            return None
        return float(city_df.iloc[-1]["aqi"])
    except Exception:
        return None


def build_feature_row(raw_pollution: dict, raw_weather: dict, previous_aqi: float | None) -> dict:
    now = datetime.now(timezone.utc)

    row = {
        "city": CITY_NAME,
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
