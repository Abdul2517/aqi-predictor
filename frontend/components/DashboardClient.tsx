"use client";

import { useState } from "react";
import dynamic from "next/dynamic";

import Hero from "./Hero";
import ForecastCards from "./ForecastCards";
import TrendChart from "./TrendChart";
import PollutantChart from "./PollutantChart";
import HealthRecommendations from "./HealthRecommendations";
import Insights from "./Insights";
import ModelConfidence from "./ModelConfidence";
import AboutSection from "./AboutSection";
import FadeIn from "./FadeIn";
import CitySelector from "./CitySelector";
import LiveClock from "./LiveClock";
import ShapSection from "./ShapSection";

import {
  CITY_COORDS,
  CITY_LABELS,
  CITY_ORDER,
  isCityOk,
  type CityKey,
  type PredictionsByCity,
} from "@/lib/types";

// Leaflet touches `window`/`document` at import time, so it must never be
// part of the server-rendered bundle. ssr:false is safe here because this
// whole file is already a Client Component.
const MonitoringMap = dynamic(() => import("./MonitoringMap"), {
  ssr: false,
  loading: () => <div className="glass-card animate-pulse" style={{ height: 320 }} />,
});

export default function DashboardClient({ allData }: { allData: PredictionsByCity }) {
  const [selected, setSelected] = useState<CityKey>("rawalpindi");
  const data = allData[selected];
  const coords = CITY_COORDS[selected];

  const availability = Object.fromEntries(
    CITY_ORDER.map((key) => [key, isCityOk(allData[key])])
  ) as Record<CityKey, boolean>;

  return (
    <main className="max-w-[1440px] mx-auto px-6 md:px-10 py-10 md:py-14 space-y-16">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex flex-col gap-1.5">
          <span className="eyebrow">Select city</span>
          <CitySelector
            selected={selected}
            onChange={setSelected}
            cityLabels={CITY_LABELS}
            order={CITY_ORDER}
            availability={availability}
          />
        </div>
        <LiveClock />
      </div>

      {isCityOk(data) ? (
        <>
          <FadeIn>
            <Hero data={data} />
          </FadeIn>
          <FadeIn>
            <ForecastCards data={data} />
          </FadeIn>
          <FadeIn>
            <TrendChart data={data} />
          </FadeIn>
          <FadeIn>
            <PollutantChart data={data} />
          </FadeIn>
          <FadeIn>
            <div>
              <h2 className="text-2xl font-semibold mb-1">Monitoring location</h2>
              <p className="text-sm text-[#9fb0c8] mb-5">
                Live station used for {CITY_LABELS[selected]}&apos;s AQI readings. One monitoring point per
                city.
              </p>
              <MonitoringMap cityName={CITY_LABELS[selected]} lat={coords.lat} lon={coords.lon} />
            </div>
          </FadeIn>
          <FadeIn>
            <Insights data={data} />
          </FadeIn>
          <FadeIn>
            <HealthRecommendations data={data} />
          </FadeIn>
          <FadeIn>
            <ShapSection cityKey={selected} cityName={CITY_LABELS[selected]} available={true} />
          </FadeIn>
          <FadeIn>
            <ModelConfidence data={data} />
          </FadeIn>
          <FadeIn>
            <AboutSection data={data} />
          </FadeIn>
        </>
      ) : (
        <div className="glass-card p-8 text-center">
          <div className="eyebrow mb-2">{CITY_LABELS[selected].toUpperCase()}</div>
          <p className="text-lg text-[#f5f2ea] mb-2">Data unavailable for {CITY_LABELS[selected]}</p>
          <p className="text-sm text-[#9fb0c8]">
            {data.error || "This city's forecast is still being prepared."}
          </p>
        </div>
      )}

      <footer className="text-center text-xs text-[#7a8ba3] pt-6 pb-4">
        {isCityOk(data) && (
          <>
            Generated {new Date(data.generated_at).toLocaleString("en-US", { timeZone: "Asia/Karachi" })}{" "}
            PKT
          </>
        )}
      </footer>
    </main>
  );
}
