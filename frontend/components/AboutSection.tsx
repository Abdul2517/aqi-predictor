import { CATEGORY_COLORS, type PredictionsData } from "@/lib/types";

export default function AboutSection({ data }: { data: PredictionsData }) {
  const color = CATEGORY_COLORS[data.current.category];
  const { day1, day2, day3 } = data.predictions;

  return (
    <div>
      <h2 className="text-2xl md:text-3xl font-semibold mb-5">About This Model</h2>
      <div className="grid md:grid-cols-[1.3fr_1fr] gap-8">
        <div className="glass-card p-6 leading-relaxed">
          <p className="mb-3">
            This forecast is produced by a fully automated MLOps pipeline built for{" "}
            <b>{data.city}</b>:
          </p>
          <ul className="space-y-2 text-[#e3e0d6]">
            <li>
              <b>Data sources:</b> OpenWeather (live + historical pollution) and Open-Meteo
              (historical weather), both free tiers
            </li>
            <li>
              <b>History used for training:</b> 5+ years of hourly data (back to Nov 2020)
            </li>
            <li>
              <b>Feature store &amp; model registry:</b> Hopsworks
            </li>
            <li>
              <b>Automation:</b> feature data collected hourly, models retrained daily via GitHub
              Actions
            </li>
          </ul>
          <p className="mt-4 mb-2 font-semibold">Current models in production:</p>
          <ul className="mono text-sm space-y-1 text-[#cfcabd]">
            <li>Day+1: {day1.model_type} (v{day1.version})</li>
            <li>Day+2: {day2.model_type} (v{day2.version})</li>
            <li>Day+3: {day3.model_type} (v{day3.version})</li>
          </ul>
          <p className="mt-4 text-sm text-[#9fb0c8]">
            Each horizon is served by whichever model type performed best in cross-validated
            testing for that specific horizon — not a single one-size-fits-all model.
          </p>
        </div>
        <div className="glass-card p-6 flex flex-col items-center justify-center text-center">
          <div className="relative w-32 h-32 flex items-center justify-center mb-4">
            <div
              className="absolute inset-0 rounded-full animate-pulse"
              style={{ backgroundColor: `${color}22` }}
            />
            <div
              className="absolute inset-4 rounded-full"
              style={{ backgroundColor: `${color}44` }}
            />
            <div
              className="relative w-6 h-6 rounded-full"
              style={{ backgroundColor: color, boxShadow: `0 0 20px ${color}` }}
            />
          </div>
          <p className="font-semibold">{data.city}</p>
          <p className="mono text-xs text-[#9fb0c8] mt-1">
            {data.lat.toFixed(4)}, {data.lon.toFixed(4)}
          </p>
          <p className="text-xs text-[#9fb0c8] mt-2">Glow color reflects current category</p>
        </div>
      </div>
    </div>
  );
}
