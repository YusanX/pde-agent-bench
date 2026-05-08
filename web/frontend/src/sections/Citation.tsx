import { useState } from "react";
import { Copy, Check, FileText } from "lucide-react";
import SectionHeader from "@/components/SectionHeader";
import { useCitation } from "@/lib/api";

export default function Citation() {
  const { data } = useCitation();
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    if (!data?.bibtex) return;
    await navigator.clipboard.writeText(data.bibtex);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <section id="citation" className="py-20 bg-slate-50/60 border-t border-slate-200/70">
      <div className="container-page">
        <SectionHeader tag="Citation" title="Cite Our Work" desc={data?.venue} />

        <div className="card max-w-4xl mx-auto overflow-hidden">
          <div className="flex items-center justify-between px-5 py-3 border-b border-slate-200">
            <div className="flex items-center gap-2 font-semibold text-ink-900">
              <FileText className="w-4 h-4 text-brand-600" /> BibTeX
            </div>
            <button
              onClick={handleCopy}
              className="btn btn-secondary !py-1.5 !px-3 text-xs"
            >
              {copied ? (
                <>
                  <Check className="w-3.5 h-3.5 text-emerald-500" /> Copied
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5" /> Copy BibTeX
                </>
              )}
            </button>
          </div>
          <pre className="code-block !rounded-none whitespace-pre overflow-x-auto text-xs leading-relaxed">
            {data?.bibtex ?? "loading…"}
          </pre>
        </div>
      </div>
    </section>
  );
}
