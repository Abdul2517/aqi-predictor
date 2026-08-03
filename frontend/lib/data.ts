import fs from "fs";
import path from "path";
import type { PredictionsData } from "./types";

export function getPredictions(): PredictionsData {
  const filePath = path.join(process.cwd(), "public", "predictions.json");
  const raw = fs.readFileSync(filePath, "utf-8");
  return JSON.parse(raw) as PredictionsData;
}
