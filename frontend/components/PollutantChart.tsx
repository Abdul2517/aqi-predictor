"use client";

import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis, CartesianGrid } from "recharts";
import type { PredictionsData } from "@/lib/types";

const LABELS: Record<string, string> = {
  pm2_5: "PM2.5",
  pm10: "PM10",
  co: "CO",
  no: "NO",
  no2: "NO₂",
  o3: "O₃",
  so2: "SO₂",
  nh3: "NH₃",
};

export default function PollutantChart({ data }: { data: PredictionsData }) {
  const chartData = Object.entries(data.pollutants).map(([key, value]) => ({
    name: LABELS[key] ?? key,
    value: Number(value.toFixed(1)),
  }));

  return (
    <div>
      <h2 className="text-2xl md:text-3xl font-semibold mb-5">Pollutant Breakdown</h2>
      <div className="glass-card p-6">
        <ResponsiveContainer width="100%" height={340}>
          <BarChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
            <CartesianGrid stroke="rgba(245,242,234,0.1)" />
            <XAxis dataKey="name" stroke="#d8d4c8" fontSize={12} tick={{ fill: "#d8d4c8" }} />
            <YAxis stroke="#d8d4c8" fontSize={12} tick={{ fill: "#d8d4c8" }} />
            <Tooltip
              contentStyle={{
                backgroundColor: "#10264a",
                border: "1px solid rgba(255,255,255,0.15)",
                borderRadius: 10,
                color: "#f5f2ea",
              }}
            />
            <Bar dataKey="value" fill="#6fb98f" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
        <p className="text-xs text-[#9fb0c8] mt-2">
          Raw pollutant concentrations from the latest hourly reading.
        </p>
      </div>
    </div>
  );
}
