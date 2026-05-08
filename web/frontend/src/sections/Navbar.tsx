import { useEffect, useState } from "react";
import { Github, BookOpen, Database, Menu, X } from "lucide-react";
import { cn } from "@/lib/utils";

const links = [
  { href: "#leaderboard", label: "Leaderboard" },
  { href: "#explorer", label: "Explorer" },
  { href: "#findings", label: "Findings" },
  { href: "#getting-started", label: "Get Started" },
];

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 12);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <header
      className={cn(
        "fixed top-0 inset-x-0 z-50 transition-all",
        scrolled
          ? "bg-white/85 backdrop-blur border-b border-slate-200/70 shadow-sm"
          : "bg-transparent",
      )}
    >
      <div className="container-page flex items-center justify-between h-16">
        <a href="#home" className="flex items-center gap-2 group">
          <span className="w-9 h-9 rounded-xl bg-gradient-to-br from-brand-600 to-accent-cyan grid place-items-center text-white font-serif italic text-xl">
            ∂
          </span>
          <span className={cn("font-bold text-lg", scrolled ? "text-ink-900" : "text-white")}>
            PDEAgent-Bench
          </span>
        </a>

        <nav className="hidden md:flex items-center gap-1">
          {links.map((l) => (
            <a
              key={l.href}
              href={l.href}
              className={cn(
                "px-3 py-2 rounded-lg text-sm font-medium transition-colors",
                scrolled
                  ? "text-ink-700 hover:bg-slate-100"
                  : "text-white/85 hover:text-white hover:bg-white/10",
              )}
            >
              {l.label}
            </a>
          ))}
          <span className="mx-2 h-6 w-px bg-slate-300/40" />
          <a
            href="#citation"
            className={cn(
              "btn !py-1.5 !px-3 text-xs",
              scrolled ? "btn-secondary" : "btn-ghost text-white",
            )}
          >
            <BookOpen className="w-3.5 h-3.5" /> Paper
          </a>
          <a
            href="https://huggingface.co/datasets/eclipse00/PDEAgent-Bench"
            target="_blank"
            rel="noopener noreferrer"
            className={cn(
              "btn !py-1.5 !px-3 text-xs",
              scrolled ? "btn-secondary" : "btn-ghost text-white",
            )}
          >
            <Database className="w-3.5 h-3.5" /> Dataset
          </a>
          <a
            href="https://github.com/YusanX/pde-agent-bench"
            target="_blank"
            rel="noopener noreferrer"
            className="btn btn-primary !py-1.5 !px-3 text-xs"
          >
            <Github className="w-3.5 h-3.5" /> GitHub
          </a>
        </nav>

        <button
          className={cn("md:hidden p-2 rounded-lg", scrolled ? "text-ink-900" : "text-white")}
          onClick={() => setOpen((o) => !o)}
          aria-label="Toggle menu"
        >
          {open ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
        </button>
      </div>

      {open && (
        <div className="md:hidden bg-white border-t border-slate-200 shadow-lg">
          <div className="container-page py-3 flex flex-col gap-1">
            {links.map((l) => (
              <a
                key={l.href}
                href={l.href}
                className="px-3 py-2 rounded-lg text-sm font-medium text-ink-700 hover:bg-slate-100"
                onClick={() => setOpen(false)}
              >
                {l.label}
              </a>
            ))}
            <div className="flex gap-2 pt-2">
              <a
                href="https://github.com/YusanX/pde-agent-bench"
                target="_blank"
                rel="noopener noreferrer"
                className="btn btn-primary flex-1 text-xs"
              >
                <Github className="w-3.5 h-3.5" /> GitHub
              </a>
              <a
                href="https://huggingface.co/datasets/eclipse00/PDEAgent-Bench"
                target="_blank"
                rel="noopener noreferrer"
                className="btn btn-secondary flex-1 text-xs"
              >
                <Database className="w-3.5 h-3.5" /> Dataset
              </a>
            </div>
          </div>
        </div>
      )}
    </header>
  );
}
