import { getAllPredictions } from "@/lib/data";
import DashboardClient from "@/components/DashboardClient";

export default function Home() {
  const allData = getAllPredictions();
  return <DashboardClient allData={allData} />;
}
