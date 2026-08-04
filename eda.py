import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

matplotlib.use("Agg")
load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")
HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai")
CITY_NAME = os.getenv("CITY_NAME", "Rawalpindi")

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 3

OUTPUT_DIR = "eda_output"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#222222",
    "text.color": "#222222",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

ACCENT = "#2e7d5b"
ACCENT2 = "#c0392b"
GRID_COLOR = "#dddddd"


def save(fig, name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def plot_full_timeseries(df):
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(df["event_time"], df["pm2_5"], color=ACCENT, linewidth=0.4, alpha=0.6, label="Hourly PM2.5")
    rolling = df.set_index("event_time")["pm2_5"].rolling("30D").mean()
    ax.plot(rolling.index, rolling.values, color=ACCENT2, linewidth=2, label="30-day rolling average")
    ax.set_title(f"PM2.5 Over Time \u2014 {CITY_NAME} (Full History)")
    ax.set_xlabel("Date")
    ax.set_ylabel("PM2.5 (\u03bcg/m\u00b3)")
    ax.grid(color=GRID_COLOR, linewidth=0.5)
    ax.legend()
    save(fig, "01_full_timeseries.png")


def plot_monthly_seasonality(df):
    df = df.copy()
    df["month"] = df["event_time"].dt.month
    monthly = df.groupby("month")["pm2_5"].agg(["mean", "std"])
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(month_names, monthly["mean"].reindex(range(1, 13)), color=ACCENT, alpha=0.85,
           yerr=monthly["std"].reindex(range(1, 13)), capsize=3, ecolor="#888888")
    ax.set_title(f"Average PM2.5 by Month \u2014 {CITY_NAME} (All Years Combined)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average PM2.5 (\u03bcg/m\u00b3)")
    ax.grid(color=GRID_COLOR, linewidth=0.5, axis="y")
    save(fig, "02_monthly_seasonality.png")


def plot_year_over_year(df):
    df = df.copy()
    df["year"] = df["event_time"].dt.year
    df["month"] = df["event_time"].dt.month
    pivot = df.groupby(["year", "month"])["pm2_5"].mean().unstack(level=0)

    fig, ax = plt.subplots(figsize=(11, 6))
    for year in pivot.columns:
        ax.plot(pivot.index, pivot[year], marker="o", linewidth=1.8, label=str(year))
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_title(f"Monthly PM2.5 Pattern, Year by Year \u2014 {CITY_NAME}")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average PM2.5 (\u03bcg/m\u00b3)")
    ax.grid(color=GRID_COLOR, linewidth=0.5)
    ax.legend(title="Year", ncol=2)
    save(fig, "03_year_over_year.png")


def plot_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(df["pm2_5"], bins=60, color=ACCENT, alpha=0.85)
    axes[0].set_title("PM2.5 Distribution (Raw)")
    axes[0].set_xlabel("PM2.5 (\u03bcg/m\u00b3)")
    axes[0].set_ylabel("Frequency")
    axes[0].grid(color=GRID_COLOR, linewidth=0.5, axis="y")

    log_vals = np.log1p(df["pm2_5"])
    axes[1].hist(log_vals, bins=60, color=ACCENT2, alpha=0.85)
    axes[1].set_title("PM2.5 Distribution (log1p-transformed)")
    axes[1].set_xlabel("log(1 + PM2.5)")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(color=GRID_COLOR, linewidth=0.5, axis="y")

    fig.suptitle(f"PM2.5 Distribution Before/After Log Transform \u2014 {CITY_NAME}", fontweight="bold")
    save(fig, "04_distribution.png")


def plot_correlation_heatmap(df):
    cols = ["pm2_5", "pm10", "aqi", "co", "no", "no2", "o3", "so2", "nh3",
            "temperature", "humidity", "pressure", "wind_speed", "wind_deg"]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticklabels(cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="black" if abs(corr.iloc[i, j]) < 0.7 else "white")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Correlation")
    ax.set_title(f"Feature Correlation Heatmap \u2014 {CITY_NAME}")
    save(fig, "05_correlation_heatmap.png")


def plot_hourly_pattern(df):
    df = df.copy()
    hourly = df.groupby("hour")["pm2_5"].mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hourly.index, hourly.values, color=ACCENT, marker="o", linewidth=2)
    ax.fill_between(hourly.index, hourly.values, color=ACCENT, alpha=0.15)
    ax.set_title(f"Average PM2.5 by Hour of Day \u2014 {CITY_NAME}")
    ax.set_xlabel("Hour (UTC)")
    ax.set_ylabel("Average PM2.5 (\u03bcg/m\u00b3)")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(color=GRID_COLOR, linewidth=0.5)
    save(fig, "06_hourly_pattern.png")


def plot_weekday_pattern(df):
    df = df.copy()
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekday_avg = df.groupby("day_of_week")["pm2_5"].mean().reindex(range(7))

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [ACCENT if i < 5 else ACCENT2 for i in range(7)]
    ax.bar(day_names, weekday_avg.values, color=colors, alpha=0.85)
    ax.set_title(f"Average PM2.5 by Day of Week \u2014 {CITY_NAME}")
    ax.set_ylabel("Average PM2.5 (\u03bcg/m\u00b3)")
    ax.grid(color=GRID_COLOR, linewidth=0.5, axis="y")
    save(fig, "07_weekday_pattern.png")


def plot_weather_relationships(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    weather_vars = [("temperature", "Temperature (\u00b0C)"),
                     ("humidity", "Humidity (%)"),
                     ("wind_speed", "Wind Speed (m/s)")]
    for ax, (col, label) in zip(axes, weather_vars):
        sample = df.sample(min(5000, len(df)), random_state=42)
        ax.scatter(sample[col], sample["pm2_5"], s=4, alpha=0.25, color=ACCENT)
        ax.set_xlabel(label)
        ax.set_ylabel("PM2.5 (\u03bcg/m\u00b3)")
        ax.set_title(f"PM2.5 vs {label}")
        ax.grid(color=GRID_COLOR, linewidth=0.5)
    fig.suptitle(f"PM2.5 vs Weather Variables \u2014 {CITY_NAME}", fontweight="bold")
    save(fig, "08_weather_relationships.png")


def print_summary_stats(df):
    print("\n=== Summary Statistics ===")
    print(f"Date range: {df['event_time'].min()} to {df['event_time'].max()}")
    print(f"Total rows: {len(df)}")
    print(f"\nPM2.5 stats:\n{df['pm2_5'].describe()}")
    print(f"\nPM2.5 skewness: {df['pm2_5'].skew():.2f}")
    monthly_means = df.copy()
    monthly_means["month"] = monthly_means["event_time"].dt.month
    by_month = monthly_means.groupby("month")["pm2_5"].mean().sort_values(ascending=False)
    print(f"\nWorst month (highest avg PM2.5): month {by_month.index[0]} ({by_month.iloc[0]:.1f} \u03bcg/m\u00b3)")
    print(f"Best month (lowest avg PM2.5): month {by_month.index[-1]} ({by_month.iloc[-1]:.1f} \u03bcg/m\u00b3)")


def main():
    import hopsworks

    print(f"Connecting to Hopsworks project '{HOPSWORKS_PROJECT_NAME}'...")
    project = hopsworks.login(
        project=HOPSWORKS_PROJECT_NAME, host=HOPSWORKS_HOST, port=443, api_key_value=HOPSWORKS_API_KEY,
    )
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)

    print("Loading full historical dataset...")
    df = fg.read()
    df = df[df["city"] == CITY_NAME].sort_values("event_time").reset_index(drop=True)
    print(f"  -> {len(df)} rows loaded")

    print("\nGenerating EDA charts...")
    plot_full_timeseries(df)
    plot_monthly_seasonality(df)
    plot_year_over_year(df)
    plot_distribution(df)
    plot_correlation_heatmap(df)
    plot_hourly_pattern(df)
    plot_weekday_pattern(df)
    plot_weather_relationships(df)

    print_summary_stats(df)

    print(f"\nAll charts saved to ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
