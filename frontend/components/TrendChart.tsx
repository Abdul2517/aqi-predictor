"use client";

import {
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  CartesianGrid,
  Legend,
} from "recharts";
import { HORIZON_LABELS, type PredictionsData } from "@/lib/types";

export default function TrendChart({ data }: { data: PredictionsData }) {
  const actualPoints = data.trend.map((p) => ({
    time: new Date(p.time).getTime(),
    label: new Date(p.time).toLocaleDateString("en-US", { month: "short", day: "numeric" }),
    actual: p.pm2_5,
  }));

  const lastActualTime = actualPoints.length
    ? actualPoints[actualPoints.length - 1].time
    : Date.now();

  const forecastOffsets: Record<"day1" | "day2" | "day3", number> = {
    day1: 24,
    day2: 48,
    day3: 72,
  };

  const forecastPoints = (["day1", "day2", "day3"] as const).map((h) => ({
    time: lastActualTime + forecastOffsets[h] * 3600 * 1000,
    label: HORIZON_LABELS[h],
    forecast: data.predictions[h].pm2_5,
  }));

  const merged = [...actualPoints, ...forecastPoints].sort((a, b) => a.time - b.time);

  return (
    <div>
      <h2 className="text-2xl md:text-3xl font-semibold mb-5">Interactive Timeline</h2>
      <div className="glass-card p-6">
        <ResponsiveContainer width="100%" height={380}>
          <LineChart data={merged} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
            <CartesianGrid stroke="rgba(245,242,234,0.1)" />
            <XAxis
              dataKey="label"
              stroke="#d8d4c8"
              fontSize={12}
              tick={{ fill: "#d8d4c8" }}
              minTickGap={30}
            />
            <YAxis stroke="#d8d4c8" fontSize={12} tick={{ fill: "#d8d4c8" }} />
            <Tooltip
              contentStyle={{
                backgroundColor: "#10264a",
                border: "1px solid rgba(255,255,255,0.15)",
                borderRadius: 10,
                color: "#f5f2ea",
              }}
            />
            <Legend wrapperStyle={{ color: "#f0ede4", fontSize: 13 }} />
            <Line
              type="monotone"
              dataKey="actual"
              name="Actual (last 7 days)"
              stroke="#3498db"
              strokeWidth={1.5}
              dot={false}
              connectNulls
            />
            <Line
              type="monotone"
              dataKey="forecast"
              name="Forecast"
              stroke="#e74c3c"
              strokeWidth={2.5}
              strokeDasharray="6 4"
              dot={{ r: 5 }}
              connectNulls
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
