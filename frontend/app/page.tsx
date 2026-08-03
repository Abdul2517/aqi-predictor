import { getPredictions } from "@/lib/data";
import Hero from "@/components/Hero";
import ForecastCards from "@/components/ForecastCards";
import TrendChart from "@/components/TrendChart";
import PollutantChart from "@/components/PollutantChart";
import Insights from "@/components/Insights";
import HealthRecommendations from "@/components/HealthRecommendations";
import ModelConfidence from "@/components/ModelConfidence";
import AboutSection from "@/components/AboutSection";
import FadeIn from "@/components/FadeIn";

export default function Home() {
  const data = getPredictions();

  return (
    <main className="max-w-[1440px] mx-auto px-6 md:px-10 py-10 md:py-14 space-y-16">
      <FadeIn>
        <Hero data={data} />
      </FadeIn>
      <FadeIn>
        <ForecastCards data={data} />
      </FadeIn>
      <FadeIn>
        <TrendChart data={data} />
      </FadeIn>
      <FadeIn>
        <PollutantChart data={data} />
      </FadeIn>
      <FadeIn>
        <Insights data={data} />
      </FadeIn>
      <FadeIn>
        <HealthRecommendations data={data} />
      </FadeIn>
      <FadeIn>
        <ModelConfidence data={data} />
      </FadeIn>
      <FadeIn>
        <AboutSection data={data} />
      </FadeIn>
      <footer className="text-center text-xs text-[#7a8ba3] pt-6 pb-4">
        Generated {new Date(data.generated_at).toLocaleString("en-US", { timeZone: "Asia/Karachi" })}{" "}
        PKT
      </footer>
    </main>
  );
}
