import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, Image as ImageIcon } from "lucide-react";
import SectionHeader from "@/components/SectionHeader";
import { useFigures, withBase, type Figure } from "@/lib/api";

export default function FigureGallery() {
  const { data } = useFigures();
  const [active, setActive] = useState<Figure | null>(null);
  const figures = data ?? [];

  return (
    <section className="py-20 bg-slate-50/60 border-y border-slate-200/70">
      <div className="container-page">
        <SectionHeader
          tag="Visuals"
          title="Figures from the Paper"
          desc="A selection of key figures from the manuscript. Click to enlarge."
        />

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
          {figures.map((f, i) => (
            <motion.button
              key={f.id}
              initial={{ opacity: 0, y: 10 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.05 }}
              onClick={() => setActive(f)}
              className="card card-hover overflow-hidden text-left group"
            >
              <div className="aspect-[16/10] bg-gradient-to-br from-slate-100 to-slate-200 relative overflow-hidden">
                <img
                  src={withBase(f.thumb)}
                  alt={f.caption}
                  className="w-full h-full object-contain p-3 transition-transform group-hover:scale-[1.03]"
                  loading="lazy"
                  onError={(e) => {
                    (e.currentTarget.style as CSSStyleDeclaration).display = "none";
                  }}
                />
                <div className="absolute inset-0 grid place-items-center text-slate-300 -z-0 pointer-events-none">
                  <ImageIcon className="w-12 h-12" />
                </div>
                <div className="absolute top-2 left-2">
                  <span className="pill bg-white/90 text-ink-700 backdrop-blur">
                    {f.section}
                  </span>
                </div>
              </div>
              <div className="p-4">
                <p className="text-sm text-slate-700 leading-snug line-clamp-3">{f.caption}</p>
              </div>
            </motion.button>
          ))}
        </div>
      </div>

      <AnimatePresence>
        {active && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm grid place-items-center p-4"
            onClick={() => setActive(null)}
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
              className="bg-white rounded-2xl max-w-5xl w-full max-h-[88vh] overflow-auto"
            >
              <div className="flex items-center justify-between p-4 border-b border-slate-200 sticky top-0 bg-white">
                <span className="pill bg-brand-50 text-brand-700">{active.section}</span>
                <button
                  onClick={() => setActive(null)}
                  aria-label="Close"
                  className="p-2 rounded-lg hover:bg-slate-100"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
              <div className="p-6 bg-slate-50 grid place-items-center">
                <img src={withBase(active.src)} alt={active.caption} className="max-w-full max-h-[60vh]" />
              </div>
              <p className="p-6 text-sm text-slate-700 leading-relaxed">{active.caption}</p>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </section>
  );
}
