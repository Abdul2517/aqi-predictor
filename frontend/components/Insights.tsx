import { HORIZON_LABELS, type PredictionsData } from "@/lib/types";

export default function Insights({ data }: { data: PredictionsData }) {
  const { day1, day2, day3 } = data.predictions;
  const bestLabel = HORIZON_LABELS[data.best_day];

  let trendSentence = "";
  if (data.current.trend_pct_24h !== null) {
    const pct = data.current.trend_pct_24h;
    trendSentence = `Over the last 24 hours, PM2.5 has ${pct > 0 ? "risen" : "fallen"} ${Math.abs(
      pct
    ).toFixed(0)}%, reaching ${data.current.pm2_5.toFixed(1)} μg/m³. `;
  }

  return (
    <div>
      <h2 className="text-2xl md:text-3xl font-semibold mb-5">AI Explanation</h2>
      <div className="ai-card p-6 text-lg leading-relaxed">
        {trendSentence}
        Looking ahead, the model forecasts PM2.5 to reach {day1.pm2_5.toFixed(1)} μg/m³ tomorrow,{" "}
        {day2.pm2_5.toFixed(1)} μg/m³ the day after, and {day3.pm2_5.toFixed(1)} μg/m³ in three
        days. <b>{bestLabel}</b> currently shows the most favorable forecasted air quality of the
        period.
      </div>
    </div>
  );
}
