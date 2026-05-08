import { useState } from "react";
import { Copy, Check, Terminal } from "lucide-react";
import SectionHeader from "@/components/SectionHeader";

const steps = [
  {
    title: "Clone & Install",
    code: `git clone https://github.com/YusanX/pde-agent-bench
cd pde-agent-bench
pip install -e ".[fenicsx]"      # DOLFINx backend
# or: pip install -e ".[agents]"  # for CodePDE/OpenHands agent support`,
  },
  {
    title: "Configure API Keys",
    code: `export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."        # Gemini
export DASHSCOPE_API_KEY="..."     # Qwen`,
  },
  {
    title: "Run Evaluation",
    code: `# Evaluate GPT-4o on the full DOLFINx benchmark v2
python scripts/run_benchmark.py \\
  --agent gpt-4o \\
  --backend dolfinx \\
  --dataset data/benchmark_v2.jsonl

# Filter by PDE family
python scripts/run_benchmark.py \\
  --agent claude-sonnet-3-5 \\
  --pde_types poisson heat navier_stokes \\
  --mode fix_accuracy`,
  },
];

export default function GettingStarted() {
  return (
    <section id="getting-started" className="py-20">
      <div className="container-page">
        <SectionHeader
          tag="Quick Start"
          title="Getting Started"
          desc="Run PDEAgent-Bench in minutes. Pick your backend — DOLFINx, Firedrake, or deal.II."
        />

        <div className="space-y-4 max-w-4xl mx-auto">
          {steps.map((s, i) => (
            <CodeStep key={s.title} num={i + 1} {...s} />
          ))}
        </div>
      </div>
    </section>
  );
}

function CodeStep({ num, title, code }: { num: number; title: string; code: string }) {
  const [copied, setCopied] = useState(false);
  const handleCopy = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <div className="card overflow-hidden">
      <div className="flex items-center justify-between px-5 py-3 border-b border-slate-200 bg-slate-50/70">
        <div className="flex items-center gap-3">
          <span className="w-7 h-7 rounded-lg bg-brand-600 text-white grid place-items-center text-xs font-bold">
            {num}
          </span>
          <span className="font-semibold text-ink-900">{title}</span>
        </div>
        <button
          onClick={handleCopy}
          className="text-xs text-slate-600 hover:text-brand-600 inline-flex items-center gap-1.5 font-medium"
        >
          {copied ? (
            <>
              <Check className="w-3.5 h-3.5 text-emerald-500" /> Copied
            </>
          ) : (
            <>
              <Copy className="w-3.5 h-3.5" /> Copy
            </>
          )}
        </button>
      </div>
      <div className="relative">
        <Terminal className="absolute top-3 right-3 w-4 h-4 text-slate-500/30" />
        <pre className="code-block !rounded-none whitespace-pre overflow-x-auto leading-relaxed">
          {code}
        </pre>
      </div>
    </div>
  );
}
