import fs from "fs";
import path from "path";
import type { PredictionsByCity } from "./types";

export function getAllPredictions(): PredictionsByCity {
  const filePath = path.join(process.cwd(), "public", "predictions.json");
  const raw = fs.readFileSync(filePath, "utf-8");
  return JSON.parse(raw) as PredictionsByCity;
}
