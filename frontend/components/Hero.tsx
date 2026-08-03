import { CATEGORY_COLORS, type PredictionsData } from "@/lib/types";

function Gauge({ value, color }: { value: number; color: string }) {
  const max = 300;
  const pct = Math.min(value / max, 1);
  const radius = 80;
  const circumference = Math.PI * radius;
  const dash = pct * circumference;

  return (
    <svg viewBox="0 0 200 120" className="w-full max-w-xs">
      <path
        d="M 20 100 A 80 80 0 0 1 180 100"
        fill="none"
        stroke="rgba(255,255,255,0.08)"
        strokeWidth="14"
        strokeLinecap="round"
      />
      <path
        d="M 20 100 A 80 80 0 0 1 180 100"
        fill="none"
        stroke={color}
        strokeWidth="14"
        strokeLinecap="round"
        strokeDasharray={`${dash} ${circumference}`}
        style={{ transition: "stroke-dasharray 1s ease" }}
      />
      <text x="100" y="90" textAnchor="middle" className="mono" fill="#f9f7f1" fontSize="30" fontWeight="700">
        {value.toFixed(0)}
      </text>
      <text x="100" y="108" textAnchor="middle" fill="#d8d4c8" fontSize="10">
        μg/m³
      </text>
    </svg>
  );
}

export default function Hero({ data }: { data: PredictionsData }) {
  const color = CATEGORY_COLORS[data.current.category];
  const day1 = data.predictions.day1.pm2_5;
  const day3 = data.predictions.day3.pm2_5;
  const outlookWord = day3 > day1 ? "worsen" : "improve";

  return (
    <div>
      <div className="eyebrow mb-2">LIVE · {data.city.toUpperCase()}</div>
      <div className="grid md:grid-cols-2 gap-8 items-center">
        <div>
          <div className="mono font-bold text-6xl md:text-7xl text-[#f9f7f1] leading-none">
            {data.current.pm2_5.toFixed(0)}
            <span className="text-xl font-medium text-[#d8d4c8] ml-2">μg/m³ PM2.5</span>
          </div>
          <div
            className="inline-block mt-4 px-4 py-1.5 rounded-full font-semibold text-lg"
            style={{ backgroundColor: `${color}33`, color, border: `1px solid ${color}` }}
          >
            {data.current.category}
          </div>
          <div className="ai-card mt-5 p-5">
            <span className="inline-block bg-emerald-500/20 text-emerald-400 text-xs font-bold tracking-wide px-2.5 py-1 rounded-full mr-2">
              ● LIVE
            </span>
            🤖 Air quality is currently <b>{data.current.category.toUpperCase()}</b> (
            {data.current.pm2_5.toFixed(0)} μg/m³ PM2.5). PM2.5 is expected to {outlookWord} over the
            next 3 days, shifting from {day1.toFixed(0)} to {day3.toFixed(0)} μg/m³ (
            {data.outlook_pct_3day >= 0 ? "+" : ""}
            {data.outlook_pct_3day.toFixed(0)}%).
          </div>
        </div>
        <div className="flex justify-center">
          <Gauge value={data.current.pm2_5} color={color} />
        </div>
      </div>
      <p className="text-sm text-[#9fb0c8] mt-4">
        Last updated {new Date(data.last_data_update).toLocaleString("en-US", { timeZone: "Asia/Karachi" })} PKT ·
        updates hourly
      </p>
    </div>
  );
}
