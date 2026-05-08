import { motion } from "framer-motion";
import SectionHeader from "@/components/SectionHeader";
import { useCitation } from "@/lib/api";

export default function Abstract() {
  const { data } = useCitation();

  return (
    <section className="py-20">
      <div className="container-page">
        <SectionHeader
          tag="Paper"
          title={data?.title ?? "PDEAgent-Bench"}
          desc={data?.venue}
        />
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
          className="card max-w-4xl mx-auto p-8 sm:p-10"
        >
          <h3 className="text-xs font-semibold uppercase tracking-wider text-brand-700 mb-4">
            Abstract
          </h3>
          <div className="prose-like text-slate-700 leading-relaxed space-y-4">
            {(data?.abstract ?? "").split("\n\n").map((p, i) => (
              <p key={i}>{p}</p>
            ))}
          </div>
        </motion.div>
      </div>
    </section>
  );
}
