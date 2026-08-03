import { HORIZON_LABELS, type PredictionsData } from "@/lib/types";

export default function ModelConfidence({ data }: { data: PredictionsData }) {
  const horizons: ("day1" | "day2" | "day3")[] = ["day1", "day2", "day3"];

  return (
    <div>
      <h2 className="text-2xl md:text-3xl font-semibold mb-2">Model Confidence</h2>
      <p className="text-sm text-[#9fb0c8] mb-5">
        R² (cross-validated) — how much of the real variation in PM2.5 the model explains. 1.0 is
        perfect; 0 means no better than guessing the average. Near-term forecasts are typically
        more reliable than longer-range ones.
      </p>
      <div className="grid sm:grid-cols-3 gap-5">
        {horizons.map((h) => {
          const r2 = data.predictions[h].r2;
          const pct = r2 !== null ? Math.max(0, Math.min(r2, 1)) * 100 : 0;
          return (
            <div key={h} className="glass-card p-6 text-center">
              <div className="eyebrow text-[0.7rem] mb-2">{HORIZON_LABELS[h]}</div>
              {r2 !== null ? (
                <>
                  <div className="mono font-bold text-3xl text-[#f9f7f1]">{r2.toFixed(2)}</div>
                  <div className="mono text-xs text-[#cfcabd] mb-2">R²</div>
                  <div className="h-2 rounded-full bg-white/10 overflow-hidden">
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: `${pct}%`,
                        background: "linear-gradient(90deg, #38bdf8, #4ade80)",
                      }}
                    />
                  </div>
                </>
              ) : (
                <div className="mono text-xs text-[#cfcabd]">not available</div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
