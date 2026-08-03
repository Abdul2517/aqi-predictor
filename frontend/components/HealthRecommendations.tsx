import { CATEGORY_COLORS, HEALTH_GUIDANCE, HORIZON_LABELS, type PredictionsData } from "@/lib/types";

export default function HealthRecommendations({ data }: { data: PredictionsData }) {
  const categories = [
    data.current.category,
    data.predictions.day1.category,
    data.predictions.day2.category,
    data.predictions.day3.category,
  ];
  const order = ["Good", "Moderate", "Unhealthy for Sensitive Groups", "Unhealthy", "Very Unhealthy", "Hazardous"];
  const worst = categories.reduce((a, b) => (order.indexOf(b) > order.indexOf(a) ? b : a));
  const guidance = HEALTH_GUIDANCE[worst] ?? HEALTH_GUIDANCE["Moderate"];

  return (
    <div>
      <h2 className="text-2xl md:text-3xl font-semibold mb-2">Health Recommendations</h2>
      <p className="text-sm text-[#9fb0c8] mb-5">
        Based on the {worst} level reached in the current + 3-day forecast window
      </p>
      <div className="space-y-3">
        {guidance.map((g, i) => (
          <div key={i} className="glass-card p-4 flex items-center gap-4">
            <span className="text-2xl">{g.icon}</span>
            <span>{g.text}</span>
          </div>
        ))}
        <div className="glass-card p-4 flex items-center gap-4">
          <span className="text-2xl">🌳</span>
          <span>
            <b>{HORIZON_LABELS[data.best_day]}</b> is the most favorable window for outdoor
            activity in the current 3-day forecast.
          </span>
        </div>
      </div>
    </div>
  );
}
