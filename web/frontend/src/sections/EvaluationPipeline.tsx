import { motion } from "framer-motion";
import { Cog, Target, Timer, ArrowRight } from "lucide-react";
import SectionHeader from "@/components/SectionHeader";

const gates = [
  {
    icon: Cog,
    name: "Execution Gate",
    desc: "Code runs without syntax errors, import failures, or runtime exceptions within timeout.",
    color: "from-slate-500 to-slate-700",
  },
  {
    icon: Target,
    name: "Accuracy Gate",
    desc: "Relative L₂ error vs. the oracle solution falls below threshold (default 10×).",
    color: "from-brand-500 to-brand-700",
  },
  {
    icon: Timer,
    name: "Time Gate",
    desc: "Wall-clock runtime is within tolerance of the oracle solver (default 3×).",
    color: "from-accent-cyan to-accent-violet",
  },
];

const metrics = [
  { glyph: "L₂", color: "bg-blue-100 text-blue-700", title: "Relative L₂ Error", desc: "‖u_h − u_ref‖₂ / ‖u_ref‖₂ on a shared evaluation grid." },
  { glyph: "⏱", color: "bg-emerald-100 text-emerald-700", title: "Runtime", desc: "Wall-clock execution time in an isolated sandbox with controlled resources." },
  { glyph: "%", color: "bg-amber-100 text-amber-700", title: "Pass Rate", desc: "Fraction of cases passing all three gates." },
  { glyph: "$", color: "bg-pink-100 text-pink-700", title: "Cost / 1K", desc: "Total API cost in USD per 1,000 benchmark cases." },
];

export default function EvaluationPipeline() {
  return (
    <section id="metrics" className="py-20">
      <div className="container-page">
        <SectionHeader
          tag="Evaluation"
          title="Three-Stage Gate System"
          desc="Each generated solver is evaluated through a staged protocol that mirrors real scientific workflows: first executability, then numerical accuracy, then computational efficiency."
        />

        <div className="flex flex-col md:flex-row items-stretch gap-3 max-w-5xl mx-auto">
          {gates.map((g, i) => (
            <div key={g.name} className="flex items-stretch flex-1 gap-3">
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1 }}
                className="flex-1 card p-5 text-center"
              >
                <div className={`w-14 h-14 rounded-xl bg-gradient-to-br ${g.color} grid place-items-center text-white mx-auto`}>
                  <g.icon className="w-7 h-7" />
                </div>
                <div className="mt-3 font-bold text-ink-900">{g.name}</div>
                <p className="mt-2 text-xs text-slate-600 leading-relaxed">{g.desc}</p>
              </motion.div>
              {i < gates.length - 1 && (
                <div className="hidden md:grid place-items-center text-slate-300">
                  <ArrowRight className="w-6 h-6" />
                </div>
              )}
            </div>
          ))}
        </div>

        <div className="mt-12 grid grid-cols-2 md:grid-cols-4 gap-4">
          {metrics.map((m) => (
            <div key={m.title} className="card p-5">
              <div className={`w-11 h-11 rounded-lg ${m.color} grid place-items-center font-bold`}>{m.glyph}</div>
              <div className="mt-3 font-semibold text-ink-900 text-sm">{m.title}</div>
              <p className="mt-1 text-xs text-slate-600 leading-relaxed">{m.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
