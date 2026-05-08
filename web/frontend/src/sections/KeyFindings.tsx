import { motion } from "framer-motion";
import {
  AlertTriangle,
  Bot,
  Clock,
  GitFork,
  DollarSign,
  Layers,
  Activity,
  Gauge,
  type LucideIcon,
} from "lucide-react";
import SectionHeader from "@/components/SectionHeader";
import { useFindings } from "@/lib/api";

const ICON_MAP: Record<string, LucideIcon> = {
  AlertTriangle,
  Bot,
  Clock,
  GitFork,
  DollarSign,
  Layers,
  Activity,
  Gauge,
};

export default function KeyFindings() {
  const { data } = useFindings();
  const findings = data ?? [];

  return (
    <section id="findings" className="py-20 bg-ink-900 text-white relative overflow-hidden">
      <div className="absolute -top-20 right-0 w-[500px] h-[500px] bg-brand-500/15 rounded-full blur-3xl" />
      <div className="absolute -bottom-20 -left-20 w-[400px] h-[400px] bg-accent-cyan/10 rounded-full blur-3xl" />
      <div className="relative container-page">
        <SectionHeader
          tag="Key Findings"
          title="What we learned"
          desc="Eight headline takeaways from running the benchmark across frontier LLMs and code agents."
          light
        />

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {findings.map((f, i) => {
            const Icon = ICON_MAP[f.icon] ?? AlertTriangle;
            return (
              <motion.div
                key={f.n}
                initial={{ opacity: 0, y: 12 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: "-60px" }}
                transition={{ delay: i * 0.05 }}
                className="rounded-2xl border border-white/10 bg-white/5 backdrop-blur p-6 hover:border-white/20 hover:bg-white/[0.07] transition-colors"
              >
                <div className="flex items-start gap-4">
                  <div className="shrink-0 w-12 h-12 rounded-xl bg-gradient-to-br from-brand-500 to-accent-cyan grid place-items-center">
                    <Icon className="w-5 h-5 text-white" />
                  </div>
                  <div>
                    <div className="flex items-center gap-2 text-xs uppercase tracking-wider font-semibold text-white/60">
                      <span className="text-brand-300 font-mono">#{String(f.n).padStart(2, "0")}</span>
                      <span>finding</span>
                    </div>
                    <h3 className="mt-1 font-bold text-lg leading-tight">{f.title}</h3>
                    <p className="mt-2 text-sm text-white/70 leading-relaxed">{f.body}</p>
                  </div>
                </div>
              </motion.div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
