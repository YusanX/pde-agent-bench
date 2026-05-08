import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

type Props = {
  tag: string;
  title: string;
  desc?: string;
  align?: "center" | "left";
  light?: boolean;
};

export default function SectionHeader({ tag, title, desc, align = "center", light }: Props) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-80px" }}
      transition={{ duration: 0.5 }}
      className={cn("max-w-3xl mb-12", align === "center" && "mx-auto text-center")}
    >
      <span
        className={cn(
          "section-tag",
          light && "bg-white/10 text-white border border-white/15",
        )}
      >
        {tag}
      </span>
      <h2
        className={cn(
          "mt-4 text-3xl sm:text-4xl font-extrabold tracking-tight",
          light ? "text-white" : "text-ink-900",
        )}
      >
        {title}
      </h2>
      {desc && (
        <p
          className={cn(
            "mt-3 text-base sm:text-lg leading-relaxed",
            light ? "text-white/75" : "text-slate-600",
          )}
        >
          {desc}
        </p>
      )}
    </motion.div>
  );
}
