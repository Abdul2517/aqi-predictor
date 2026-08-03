import { CATEGORY_COLORS, HORIZON_LABELS, type PredictionsData } from "@/lib/types";
import FadeIn from "./FadeIn";

export default function ForecastCards({ data }: { data: PredictionsData }) {
  const horizons: ("day1" | "day2" | "day3")[] = ["day1", "day2", "day3"];

  return (
    <div>
      <h2 className="text-2xl md:text-3xl font-semibold mb-5">3-Day Forecast</h2>
      <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-5">
        {horizons.map((h, i) => {
          const pred = data.predictions[h];
          const color = CATEGORY_COLORS[pred.category];
          const pctVsCurrent =
            data.current.pm2_5 > 0
              ? ((pred.pm2_5 - data.current.pm2_5) / data.current.pm2_5) * 100
              : 0;
          const arrow = pctVsCurrent > 0 ? "▲" : pctVsCurrent < 0 ? "▼" : "●";

          return (
            <FadeIn key={h} delay={i * 80}>
              <div className="glass-card p-6 text-center h-full">
                <div className="eyebrow text-[0.7rem] mb-2">{HORIZON_LABELS[h]}</div>
                <div className="mono font-bold text-3xl text-[#f9f7f1]">
                  {pred.pm2_5.toFixed(1)}
                  <span className="text-sm font-medium text-[#d8d4c8] ml-1">μg/m³</span>
                </div>
                <div className="mono text-xs text-[#c9d6e8] mt-1">
                  {arrow} {Math.abs(pctVsCurrent).toFixed(0)}% vs now
                </div>
                <div
                  className="inline-block mt-3 px-3 py-1 rounded-full text-sm font-semibold"
                  style={{ backgroundColor: `${color}33`, color, border: `1px solid ${color}` }}
                >
                  {pred.category}
                </div>
              </div>
            </FadeIn>
          );
        })}
        <FadeIn delay={horizons.length * 80}>
          <div className="glass-card p-6 text-center h-full">
            <div className="eyebrow text-[0.7rem] mb-2">3-Day Average</div>
            <div className="mono font-bold text-3xl text-[#f9f7f1]">
              {data.average.pm2_5.toFixed(1)}
              <span className="text-sm font-medium text-[#d8d4c8] ml-1">μg/m³</span>
            </div>
            <div
              className="inline-block mt-3 px-3 py-1 rounded-full text-sm font-semibold"
              style={{
                backgroundColor: `${CATEGORY_COLORS[data.average.category]}33`,
                color: CATEGORY_COLORS[data.average.category],
                border: `1px solid ${CATEGORY_COLORS[data.average.category]}`,
              }}
            >
              {data.average.category}
            </div>
            <div className="mono text-xs text-[#cfcabd] mt-3">mean of the three</div>
          </div>
        </FadeIn>
      </div>
    </div>
  );
}
