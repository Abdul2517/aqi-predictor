# AQI Predictor

Started this as a single-city (Rawalpindi) AQI forecasting project and later expanded it to cover Islamabad, Lahore and Karachi as well. Predicts PM2.5 for the next 1/2/3 days using historical pollution + weather data.

Live: https://aqi-predictor-kohl.vercel.app/

## Why

Air quality data for Pakistani cities is scattered and forecasts basically don't exist. Wanted to see if I could build something that actually predicts a few days out instead of just showing current readings, and make it explainable (not just a black box number).

## How it works

Pulls hourly pollution + weather data (OpenWeather + Open-Meteo) for each city, stores it in a Hopsworks feature store, and trains a separate model per city per horizon (day1/day2/day3) — so 12 models total. Tried a single pooled model first but per-city models did noticeably better since pollution patterns differ a lot between, say, Karachi (coastal, industrial) and Islamabad (greener, less dense).

Each horizon gets 4 candidate models (Ridge, Random Forest, Gradient Boosting, a small NN) cross-validated with a time-series split, and whichever wins gets registered. A new model only replaces the one in production if it actually beats it on RMSE — otherwise the old one stays.

SHAP runs separately to generate feature importance charts per city/horizon so you can see *why* a prediction looks the way it does, not just the number.

Frontend is Next.js. City switching happens client-side, no reload — picks up whichever city's data from the generated `predictions.json`.

## Stack

- Python / scikit-learn / TensorFlow for the models
- Hopsworks for feature store + model registry
- SHAP for explainability
- Next.js + Tailwind for the frontend
- Leaflet for the map
- GitHub Actions for scheduled runs, deployed on Vercel

## Cities

Rawalpindi, Islamabad, Lahore, Karachi — configured in `cities_config.py`, easy enough to add more if there's data for them.

## Running it yourself

Backend needs a `.env` with Hopsworks + OpenWeather keys:

```
pip install -r requirements.txt
python backfill_pipeline.py     # historical data, first time per city
python training_pipeline.py     # trains/updates all 12 models
python publish_predictions.py   # writes predictions.json
python shap_explain.py          # generates SHAP charts
```

Frontend:

```
cd frontend
npm install
npm run dev
```

## Known limitations

- Islamabad/Lahore/Karachi have way less historical data than Rawalpindi right now (Rawalpindi's been running since the start), so their model accuracy is noticeably lower. Should improve as more data comes in.
- One monitoring point per city — not trying to claim nationwide coverage, it's whatever OpenWeather gives for that lat/lon.
- SHAP charts are pre-generated, not computed live (too slow for that).
