import { useState, useMemo } from "react";
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Bar,
  BarChart,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
} from "recharts";
import SectionHeader from "@/components/SectionHeader";
import { useLeaderboard, usePdeTypes } from "@/lib/api";
import { cn, BACKEND_LABELS } from "@/lib/utils";

const PALETTE = ["#3f60e6", "#22d3ee", "#a78bfa", "#f59e0b", "#fb7185", "#34d399", "#0ea5e9", "#ef4444"];

export default function ResultsExplorer() {
  const [backend, setBackend] = useState<string>("dolfinx");
  const { data: rows = [] } = useLeaderboard(backend);
  const { data: pdes = [] } = usePdeTypes();

  // Only show PDE families covered by the selected backend (Firedrake skips N-S/Burgers/Wave; deal.II skips Burgers/Wave).
  const supportedPdes = useMemo(
    () => pdes.filter((p) => p.backendCases[backend] != null),
    [pdes, backend],
  );

  const sortedRows = useMemo(
    () => rows.slice().sort((a, b) => b.passRate - a.passRate),
    [rows],
  );

  const radarData = useMemo(() => {
    return supportedPdes.map((p) => {
      const point: Record<string, string | number> = { pde: p.abbr };
      sortedRows.forEach((m) => {
        const v = m.pdeScores[p.id];
        if (typeof v === "number") point[m.model] = v;
      });
      return point;
    });
  }, [supportedPdes, sortedRows]);

  const barData = useMemo(
    () =>
      sortedRows.map((m) => ({
        model: m.model,
        "Pass Rate": m.passRate,
      })),
    [sortedRows],
  );

  return (
    <section id="explorer" className="py-20 bg-slate-50/60 border-y border-slate-200/70">
      <div className="container-page">
        <SectionHeader
          tag="Interactive"
          title="Results Explorer"
          desc="Compare per-PDE-family pass rates across all 8 evaluated systems on each FEM-library track. Numbers are taken verbatim from Tables 2 and 9 of the paper."
        />

        <div className="mb-6 flex items-center gap-3 flex-wrap">
          <div className="text-xs font-semibold uppercase tracking-wider text-slate-500">
            FEM library
          </div>
          <div className="inline-flex p-1 rounded-xl bg-white border border-slate-200">
            {(["dolfinx", "firedrake", "dealii"] as const).map((b) => (
              <button
                key={b}
                onClick={() => setBackend(b)}
                className={cn(
                  "px-3 py-1.5 text-xs font-medium rounded-lg transition-colors",
                  backend === b
                    ? "bg-brand-600 text-white shadow-sm"
                    : "text-slate-600 hover:text-ink-900 hover:bg-slate-100",
                )}
              >
                {BACKEND_LABELS[b]}
              </button>
            ))}
          </div>
          <div className="text-xs text-slate-500">
            · {supportedPdes.length} PDE families on this track · {sortedRows.length} systems
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-5 gap-5">
          <div className="card p-5 lg:col-span-3">
            <div className="flex items-center justify-between mb-2">
              <h3 className="font-semibold text-ink-900">Per-PDE Pass Rate (single-shot)</h3>
              <span className="text-xs text-slate-500">radar · 0–100%</span>
            </div>
            <div className="h-[460px]">
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart data={radarData} outerRadius="78%">
                  <PolarGrid stroke="#e2e8f0" />
                  <PolarAngleAxis dataKey="pde" tick={{ fill: "#475569", fontSize: 11 }} />
                  <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fill: "#94a3b8", fontSize: 10 }} />
                  {sortedRows.map((m, i) => (
                    <Radar
                      key={m.model}
                      name={m.model}
                      dataKey={m.model}
                      stroke={PALETTE[i % PALETTE.length]}
                      fill={PALETTE[i % PALETTE.length]}
                      fillOpacity={0.1}
                      strokeWidth={1.8}
                    />
                  ))}
                  <Legend wrapperStyle={{ fontSize: 11, paddingTop: 12 }} />
                  <Tooltip />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="card p-5 lg:col-span-2">
            <div className="flex items-center justify-between mb-2">
              <h3 className="font-semibold text-ink-900">Overall Pass Rate</h3>
              <span className="text-xs text-slate-500">bars · %</span>
            </div>
            <div className="h-[460px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={barData}
                  layout="vertical"
                  margin={{ top: 8, right: 16, bottom: 8, left: 8 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 11, fill: "#475569" }} />
                  <YAxis
                    type="category"
                    dataKey="model"
                    width={140}
                    tick={{ fontSize: 11, fill: "#475569" }}
                  />
                  <Tooltip cursor={{ fill: "rgba(63,96,230,0.06)" }} />
                  <Bar dataKey="Pass Rate" fill="#3f60e6" radius={[0, 6, 6, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
