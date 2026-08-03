export type Category =
  | "Good"
  | "Moderate"
  | "Unhealthy for Sensitive Groups"
  | "Unhealthy"
  | "Very Unhealthy"
  | "Hazardous"
  | "Unknown";

export interface HorizonPrediction {
  pm2_5: number;
  category: Category;
  model_type: string;
  version: number;
  r2: number | null;
}

export interface TrendPoint {
  time: string;
  pm2_5: number;
}

export interface PredictionsData {
  city: string;
  lat: number;
  lon: number;
  generated_at: string;
  last_data_update: string;
  current: {
    pm2_5: number;
    category: Category;
    trend_pct_24h: number | null;
  };
  predictions: {
    day1: HorizonPrediction;
    day2: HorizonPrediction;
    day3: HorizonPrediction;
  };
  average: {
    pm2_5: number;
    category: Category;
  };
  best_day: "day1" | "day2" | "day3";
  outlook_pct_3day: number;
  pollutants: Record<string, number>;
  trend: TrendPoint[];
}

export const CATEGORY_COLORS: Record<Category, string> = {
  Good: "#2ecc71",
  Moderate: "#f1c40f",
  "Unhealthy for Sensitive Groups": "#e67e22",
  Unhealthy: "#e74c3c",
  "Very Unhealthy": "#8e44ad",
  Hazardous: "#7d3c3c",
  Unknown: "#95a5a6",
};

export const HORIZON_LABELS: Record<"day1" | "day2" | "day3", string> = {
  day1: "Day +1 (24h)",
  day2: "Day +2 (48h)",
  day3: "Day +3 (72h)",
};

export const HEALTH_GUIDANCE: Record<Category, { icon: string; text: string }[]> = {
  Good: [{ icon: "\u{1F60A}", text: "Air quality is satisfactory. Enjoy outdoor activities as normal." }],
  Moderate: [
    { icon: "\u{1F642}", text: "Acceptable air quality." },
    { icon: "\u26A0\uFE0F", text: "Unusually sensitive individuals should consider limiting prolonged outdoor exertion." },
  ],
  "Unhealthy for Sensitive Groups": [
    { icon: "\u{1F637}", text: "Sensitive groups (children, elderly, asthma/heart conditions) should reduce prolonged outdoor exertion." },
    { icon: "\u{1F3C3}", text: "Everyone else can continue normal outdoor activities." },
  ],
  Unhealthy: [
    { icon: "\u{1F6AB}", text: "Everyone should reduce prolonged or heavy outdoor exertion." },
    { icon: "\u{1F637}", text: "Sensitive groups should avoid prolonged outdoor exertion entirely." },
  ],
  "Very Unhealthy": [
    { icon: "\u{1F3E0}", text: "Health alert: everyone should avoid prolonged outdoor exertion." },
    { icon: "\u{1F476}", text: "Sensitive groups should remain indoors." },
  ],
  Hazardous: [
    { icon: "\u{1F6A8}", text: "Health emergency: everyone should avoid all outdoor exertion." },
    { icon: "\u{1F3E0}", text: "Remain indoors with windows closed if possible." },
  ],
  Unknown: [{ icon: "\u2139\uFE0F", text: "Air quality category unavailable." }],
};
