import { motion } from "framer-motion";
import SectionHeader from "@/components/SectionHeader";
import { usePdeTypes } from "@/lib/api";
import { cn } from "@/lib/utils";

const CATEGORY_BADGE: Record<string, string> = {
  elliptic: "bg-blue-100 text-blue-700",
  parabolic: "bg-amber-100 text-amber-700",
  hyperbolic: "bg-orange-100 text-orange-700",
  reaction_diffusion: "bg-pink-100 text-pink-700",
  incompressible_flow: "bg-violet-100 text-violet-700",
  mixed_type: "bg-slate-100 text-slate-700",
};

function categoryBadge(cat: string): string {
  // pick the first token if the paper assigns multiple categories ("elliptic / mixed_type / parabolic")
  const first = cat.split("/")[0].trim().replace(/\s+/g, "_");
  return CATEGORY_BADGE[first] ?? "bg-slate-100 text-slate-700";
}

export default function PDETypes() {
  const { data } = usePdeTypes();
  const types = data ?? [];

  return (
    <section id="benchmark" className="py-20">
      <div className="container-page">
        <SectionHeader
          tag="Coverage"
          title="11 PDE Families · 645 Cases"
          desc="Six mathematical categories spanning elliptic, parabolic, hyperbolic, mixed-type, incompressible-flow, and reaction-diffusion regimes — every case includes a problem spec, a reference solution on a prescribed grid, and per-case accuracy and runtime targets."
        />

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {types.map((t, i) => (
            <motion.div
              key={t.id}
              initial={{ opacity: 0, y: 10 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-60px" }}
              transition={{ delay: Math.min(i * 0.04, 0.4) }}
              className="card card-hover p-5 relative overflow-hidden"
            >
              <div
                className="absolute top-0 right-0 w-24 h-24 rounded-bl-full opacity-15"
                style={{ background: t.color }}
              />
              <div className="flex items-start justify-between gap-3">
                <div className="flex items-center gap-2">
                  <span
                    className="w-9 h-9 rounded-lg grid place-items-center font-bold text-xs"
                    style={{ background: t.bgColor, color: t.color }}
                  >
                    {t.abbr}
                  </span>
                  <div>
                    <div className="font-bold text-ink-900 leading-tight">{t.name}</div>
                    <div className="text-[10px] text-slate-500 font-medium uppercase tracking-wider">
                      {t.category}
                    </div>
                  </div>
                </div>
                <span className="text-xs font-bold tabular-nums px-2 py-0.5 rounded-md bg-slate-100 text-slate-700">
                  {t.cases}
                </span>
              </div>

              <div className="mt-4 font-mono text-[11px] bg-slate-50 rounded-md px-3 py-2 text-ink-800 border border-slate-100 overflow-x-auto leading-relaxed">
                {t.equation}
              </div>

              <p className="mt-3 text-xs text-slate-600 leading-relaxed line-clamp-3">
                {t.description}
              </p>

              <div className="mt-4 flex items-center justify-between gap-2 flex-wrap">
                <span
                  className={cn(
                    "text-[10px] font-bold uppercase tracking-wider px-2 py-1 rounded",
                    categoryBadge(t.category),
                  )}
                >
                  {t.category.split("/")[0].trim().replace(/_/g, " ")}
                </span>
                <div className="flex gap-1">
                  {(["dolfinx", "firedrake", "dealii"] as const).map((b) => {
                    const n = t.backendCases[b];
                    const label = b === "dealii" ? "deal.II" : b === "dolfinx" ? "DOLFINx" : "Firedrake";
                    return (
                      <span
                        key={b}
                        title={`${label}: ${n ?? "not included"}`}
                        className={cn(
                          "text-[10px] font-medium px-1.5 py-0.5 rounded-md tabular-nums",
                          n == null
                            ? "bg-slate-100 text-slate-300 line-through"
                            : "bg-slate-100 text-slate-600",
                        )}
                      >
                        {label.slice(0, 3)} {n ?? "—"}
                      </span>
                    );
                  })}
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
