import Navbar from "./sections/Navbar";
import Hero from "./sections/Hero";
import Authors from "./sections/Authors";
import Abstract from "./sections/Abstract";
import FigureGallery from "./sections/FigureGallery";
import PDETypes from "./sections/PDETypes";
import Leaderboard from "./sections/Leaderboard";
import ResultsExplorer from "./sections/ResultsExplorer";
import EvaluationPipeline from "./sections/EvaluationPipeline";
import KeyFindings from "./sections/KeyFindings";
import GettingStarted from "./sections/GettingStarted";
import Citation from "./sections/Citation";
import Footer from "./sections/Footer";

export default function App() {
  return (
    <div className="min-h-screen bg-white text-ink-900">
      <Navbar />
      <main>
        <Hero />
        <Authors />
        <Abstract />
        <FigureGallery />
        <PDETypes />
        <Leaderboard />
        <ResultsExplorer />
        <EvaluationPipeline />
        <KeyFindings />
        <GettingStarted />
        <Citation />
      </main>
      <Footer />
    </div>
  );
}
