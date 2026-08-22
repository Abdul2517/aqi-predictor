import type { CityKey } from "@/lib/types";

const HORIZON_LABELS: Record<"day1" | "day2" | "day3", string> = {
  day1: "Day +1 (24h)",
  day2: "Day +2 (48h)",
  day3: "Day +3 (72h)",
};

const HORIZONS: ("day1" | "day2" | "day3")[] = ["day1", "day2", "day3"];

export default function ShapSection({
  cityKey,
  cityName,
  available,
}: {
  cityKey: CityKey;
  cityName: string;
  available: boolean;
}) {
  if (!available) {
    return (
      <div>
        <h2 className="text-2xl font-semibold mb-1">Why the model predicts this</h2>
        <div className="glass-card p-6 text-sm text-[#9fb0c8] mt-5">
          SHAP explainability charts for {cityName} are not available yet.
        </div>
      </div>
    );
  }

  return (
    <div>
      <h2 className="text-2xl font-semibold mb-1">Why the model predicts this</h2>
      <p className="text-sm text-[#9fb0c8] mb-5">
        Feature importance (SHAP) for {cityName}&apos;s forecast models, generated from real historical
        data.
      </p>
      <div className="grid gap-5 md:grid-cols-3">
        {HORIZONS.map((h) => (
          <div key={h} className="glass-card p-4">
            <div className="eyebrow mb-2">{HORIZON_LABELS[h]}</div>
            {/* Static pre-generated image from shap_explain.py, copied into
                public/shap/<city>/. Not regenerated or altered here. */}
            <img
              src={`/shap/${cityKey}/shap_${h}_importance.png`}
              alt={`SHAP feature importance for ${cityName} ${HORIZON_LABELS[h]}`}
              className="w-full h-auto rounded-lg"
            />
          </div>
        ))}
      </div>
    </div>
  );
}
