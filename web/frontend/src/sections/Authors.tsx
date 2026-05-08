import { motion } from "framer-motion";
import { useAuthors } from "@/lib/api";

export default function Authors() {
  const { data } = useAuthors();
  if (!data) return null;

  const hasEqual = data.authors.some((a) => a.isEqual);

  return (
    <section className="py-14 border-b border-slate-200/70 bg-slate-50/50">
      <div className="container-page">
        <div className="text-center mb-8">
          <span className="section-tag">Team</span>
          <h2 className="mt-3 text-2xl font-bold tracking-tight">Authors</h2>
        </div>
        <div className="flex flex-wrap items-stretch justify-center gap-2.5 max-w-5xl mx-auto">
          {data.authors.map((a, i) => (
            <motion.div
              key={a.name}
              initial={{ opacity: 0, y: 8 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: Math.min(i * 0.025, 0.4) }}
              className="card card-hover px-4 py-2.5 flex flex-col items-center min-w-[150px]"
            >
              <div className="font-semibold text-sm flex items-center gap-1 leading-tight">
                {a.homepage ? (
                  <a
                    href={a.homepage}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="hover:text-brand-600"
                  >
                    {a.name}
                  </a>
                ) : (
                  <span>{a.name}</span>
                )}
                {a.isEqual && (
                  <span title="Equal contribution" className="text-amber-500 text-xs font-bold">
                    *
                  </span>
                )}
                {a.isCorresponding && (
                  <span title="Corresponding author" className="text-brand-600 text-xs font-bold">
                    ✦
                  </span>
                )}
              </div>
              <div className="text-[11px] text-slate-500 mt-0.5">{a.affiliation}</div>
            </motion.div>
          ))}
        </div>
        {hasEqual && (
          <div className="mt-4 text-center text-xs text-slate-500">
            <span className="text-amber-500 font-bold">*</span> equal contribution &nbsp;·&nbsp;
            <span className="text-brand-600 font-bold">✦</span> corresponding author
          </div>
        )}
      </div>
    </section>
  );
}
